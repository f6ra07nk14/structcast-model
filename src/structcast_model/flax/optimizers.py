"""Read and mask optimizer state for Flax (nnx) learners, and scale one segment's update."""

from collections.abc import Callable
from re import compile as re_compile
from typing import TYPE_CHECKING, Any

from flax.training.dynamic_scale import DynamicScale
import jax
import jax.numpy as jnp
import optax

from flax import nnx


def unwrap_variables(tree: Any) -> Any:
    """Return the tree with every `flax.nnx.Variable` leaf replaced by the value it holds.

    This is `nnx.as_pure` written out by hand, because the supported flax floor, 0.12.6, still calls
    that function `nnx.pure` -- the rename landed in 0.12.7. Walking the tree here is what keeps one
    code path across the whole supported range instead of a version branch at every call site.

    Args:
        tree (Any): The pytree to strip, typically an optimizer state.

    Returns:
        Any: A new pytree of the same structure whose Variable leaves are their inner values.

    Example:
        >>> from flax import nnx
        >>> from structcast_model.flax.optimizers import unwrap_variables
        >>> unwrap_variables({"kernel": nnx.Param(1.0)})
        {'kernel': 1.0}
    """
    return jax.tree.map(
        lambda leaf: unwrap_variables(leaf.get_raw_value()) if isinstance(leaf, nnx.Variable) else leaf,
        tree,
        is_leaf=lambda leaf: isinstance(leaf, nnx.Variable),
    )


def get_learning_rate(optimizer: Any) -> jax.Array:
    """Return the learning rate the optimizer state currently reports.

    Optax stores no learning rate of its own: a constant lives in the update closure and a schedule
    leaves only its step count behind, so the rate is readable only when the transformation was built
    through `optax.inject_hyperparams`, which materializes it in a `hyperparams` dict. The builder
    wraps optimizer patterns for exactly that reason.

    The walk is a pure pytree traversal of the state, so calling this inside a traced training step
    compiles to a reference to the state array rather than to a host read.

    Args:
        optimizer (Any): The `flax.nnx.Optimizer` whose state to read.

    Returns:
        jax.Array: The reported rate as a float32 scalar, or NaN when the chain injects no rate at
            all or injects several of them, since neither case names a single rate to report.
    """
    try:
        rate = optax.tree_utils.tree_get(
            # The nnx `OptArray` wrappers defeat the filter unless the state is unwrapped first,
            # and the filter itself is required because a scheduled inject state carries a second
            # `learning_rate` entry (its schedule state) under `hyperparams_states`.
            unwrap_variables(optimizer.opt_state),
            "learning_rate",
            default=None,
            filtering=lambda _, value: isinstance(value, (jax.Array, float)),
        )
    except KeyError:
        rate = None
    return jnp.asarray(jnp.nan if rate is None else rate, dtype=jnp.float32)


def gradient_steps(optimizer: Any) -> jax.Array | None:
    """Return the updates the optimizer's outermost `optax.MultiSteps` has applied, None without one.

    Accumulation gates on the device, so the generated training step detects an update by comparing
    this count across its own `update` call rather than predicting one from a window read at
    construction. The read is a plain indexed read of the state, so calling it inside a traced step
    compiles to a reference to the counter array rather than to a host read; without a `MultiSteps`
    there is no counter, and the None reported here is what tells the step that every update applies.

    Only the outermost state is examined: the nnx wrappers replace the array leaves of `opt_state`
    but leave its NamedTuple shell intact, so a `MultiSteps` wrapping the whole transformation is
    exactly the case this identifies. One buried inside a `chain` is invisible here and reads as no
    accumulation at all, which the step reports as an update on every call.

    Args:
        optimizer (Any): The `flax.nnx.Optimizer` whose state to read.

    Returns:
        jax.Array | None: The `gradient_step` of the outermost `MultiSteps`, or None when the
            transformation is not wrapped in one.
    """
    state = optimizer.opt_state
    if isinstance(state, optax.MultiStepsState):
        return state.gradient_step[...]
    return None


def loss_scale(**options: Any) -> DynamicScale:
    """Build the `flax.training.dynamic_scale.DynamicScale` a float16 learner carries.

    The carry is pinned to its device element types instead of the Python `int` and `float` the
    dataclass defaults to: the step returns a new scale built from `jax.numpy.where`, so a carry
    handed in as a Python scalar has a different element type on the second call than on the first,
    which recompiles the whole step and leaves its first-call donation with nothing to donate.

    Args:
        **options: Keyword arguments of `DynamicScale`, e.g. `growth_interval` or `scale`.

    Returns:
        DynamicScale: The scale to hand to the first training step.

    Example:
        >>> from structcast_model.flax.optimizers import loss_scale
        >>> scale = loss_scale(growth_interval=100)
        >>> scale.scale.dtype, scale.fin_steps.dtype
        (dtype('float32'), dtype('int32'))
    """
    scale = DynamicScale(**options)
    return scale.replace(fin_steps=jnp.asarray(scale.fin_steps, jnp.int32), scale=jnp.asarray(scale.scale, jnp.float32))


def update_with_loss_scale(
    models: Any, optimizer: Any, grads: Any, dynamic_scale: DynamicScale, /, **extra: Any
) -> tuple[Any, DynamicScale]:
    """Apply one optimizer update against gradients a `DynamicScale` scaled, and advance the scale.

    The gradients arrive multiplied by `dynamic_scale.scale`, because the differentiated flow scaled
    its loss by it: they are divided back out in float32 here, exactly as `DynamicScale.value_and_grad`
    would have. That method is not what produced them -- it wraps `jax.value_and_grad`, which
    differentiates a plain pytree rather than the module graph `flax.nnx.value_and_grad` follows, so
    it would leave a model's batch statistics and its RNG counters behind -- so the scale is advanced
    here by handing its own dynamics a scalar carrying nothing but whether the gradients were finite.

    A non-finite gradient rolls the whole update back through `jax.numpy.where` over the state the
    update wrote: the parameters and the optimizer state, which under an `optax.MultiSteps` window
    includes the accumulator and the window counter, so the poisoned micro-step is dropped rather
    than the window it landed in. The RNG state is left out of the rollback, and the batch statistics
    the forward wrote sit in both snapshots unchanged: the forward pass ran, only the update is undone.

    The reported update is intent, not detection: the window counter is read before the rollback, so
    a step whose apply was skipped still reports the update it attempted, as the torch gradient
    scaler's does.

    Args:
        models (Any): The modules the optimizer owns, one module or a tuple of them.
        optimizer (Any): The `flax.nnx.Optimizer` to apply.
        grads (Any): The scaled gradients of the segment's flow.
        dynamic_scale (DynamicScale): The scale the loss was multiplied by.
        **extra: Further keyword arguments for `flax.nnx.Optimizer.update`.

    Returns:
        tuple[Any, DynamicScale]: Whether an update was attempted, and the advanced scale.
    """
    grads = jax.tree.map(lambda gradient: jnp.asarray(gradient, jnp.float32) / dynamic_scale.scale, grads)
    finite = jax.tree.reduce(lambda seen, g: seen & jnp.all(jnp.isfinite(g)), grads, jnp.asarray(True))
    # Differentiating `x * carrier` reproduces `carrier` as the gradient DynamicScale inspects, which
    # is finite exactly when the real gradients are: the growth interval, the backoff and the floor
    # then stay flax's own rather than a second copy of them here.
    carrier = jnp.where(finite, 0.0, jnp.nan)
    dynamic_scale, finite, _, _ = dynamic_scale.value_and_grad(lambda x: x * carrier)(jnp.float32(1.0))
    node = (models, optimizer)
    prior = nnx.to_pure_dict(nnx.state(node, nnx.Not(nnx.RngState)))
    before = gradient_steps(optimizer)
    optimizer.update(models, grads, **extra)
    # Read before the rollback below, which would revert the count with the rest of the state:
    # either read is None exactly when the transformation carries no window at all.
    after = gradient_steps(optimizer)
    has_updated = True if before is None or after is None else after > before
    applied = nnx.state(node, nnx.Not(nnx.RngState))
    kept = jax.tree.map(lambda new, old: jnp.where(finite, new, old), nnx.to_pure_dict(applied), prior)
    nnx.replace_by_pure_dict(applied, kept)
    nnx.update(node, applied)
    return has_updated, dynamic_scale


def no_weight_decay_mask(*regexes: str) -> Callable[[Any], Any]:
    r"""Build an optax mask excluding the parameters whose path matches any of the regexes.

    The returned callable maps a parameter tree to a same-structure tree of booleans, which is what
    `optax.adamw(mask=...)` and `optax.masked` consume: True keeps the transformation, False exempts
    the parameter from it. Each leaf is matched by its dotted path (`"encoder.bias"`), searched --
    not anchored -- so a plain `"bias"` matches every bias in the tree.

    Args:
        *regexes: The regular expressions matching the paths to exempt.

    Returns:
        Callable[[Any], Any]: The mask function to hand to an optax transformation.

    Example:
        >>> from structcast_model.flax.optimizers import no_weight_decay_mask
        >>> no_weight_decay_mask(r"\.bias$")({"layer": {"kernel": 1.0, "bias": 2.0}})
        {'layer': {'bias': False, 'kernel': True}}
    """
    patterns = [re_compile(regex) for regex in regexes]

    def mask(params: Any) -> Any:
        """Map the parameter tree to a same-structure tree of booleans."""
        return jax.tree_util.tree_map_with_path(
            lambda path, _: not any(p.search(jax.tree_util.keystr(path, simple=True, separator=".")) for p in patterns),
            params,
        )

    return mask


__all__ = [
    "get_learning_rate",
    "gradient_steps",
    "loss_scale",
    "no_weight_decay_mask",
    "unwrap_variables",
    "update_with_loss_scale",
]


if not TYPE_CHECKING:
    import sys

    from structcast.utils.lazy_import import LazySelectedImporter

    sys.modules[__name__] = LazySelectedImporter(__name__, globals())
