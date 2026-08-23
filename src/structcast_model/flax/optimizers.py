"""Read and mask optimizer state for Flax (nnx) learners."""

from collections.abc import Callable
from re import compile as re_compile
from typing import TYPE_CHECKING, Any

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


__all__ = ["get_learning_rate", "gradient_steps", "no_weight_decay_mask", "unwrap_variables"]


if not TYPE_CHECKING:
    import sys

    from structcast.utils.lazy_import import LazySelectedImporter

    sys.modules[__name__] = LazySelectedImporter(__name__, globals())
