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
    wraps optimizer patterns for exactly that reason (see `docs/adr/0013`).

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


def _count_multi_steps_states(node: Any) -> int:
    """Count the `MultiStepsState` shells hiding anywhere inside an optimizer state tree.

    A plain container walk suffices: the nnx wrappers replace the array leaves of `opt_state`, but
    the NamedTuple shells -- `MultiStepsState` included -- survive untouched, and a found state is
    itself a tuple the walk descends into, so a second state nested inside the first is counted too.
    """
    values = node.values() if isinstance(node, dict) else node if isinstance(node, tuple) else ()
    return int(isinstance(node, optax.MultiStepsState)) + sum(_count_multi_steps_states(value) for value in values)


def accumulation_window(optimizer: Any) -> int:
    """Return the accumulation window of the optimizer's outermost `optax.MultiSteps`, 1 without one.

    The generated flax learners call this after building their optimizers, so the host-side
    `update` gate is read back from the transformation the device actually applies instead of being
    parsed out of the optimizer pattern (see `docs/adr/0017`). Optax offers no public accessor for
    either `MultiSteps` argument, so two normalized private attributes are read: `__init__` wraps an
    int `every_k_schedule` in a local lambda -- `_every_k_schedule` then carries that lambda's
    qualname and calling it returns the int -- and replaces a `None` `should_skip_update_fn` with a
    local default, likewise identified by `_should_skip_update_fn`'s qualname. A user-passed
    callable keeps its own qualname in either slot, which is how the two rejections below tell the
    readable window apart; tests pin both dependencies across the supported optax range.

    Args:
        optimizer (Any): The `flax.nnx.Optimizer` whose transformation to inspect.

    Returns:
        int: The `every_k_schedule` of the outermost `MultiSteps`, or 1 when the transformation
            carries none.

    Raises:
        ValueError: When a `MultiSteps` hides inside the transformation instead of being its
            outermost wrapper, when the outermost `MultiSteps` nests a second one, when it carries
            a `should_skip_update_fn`, or when its `every_k_schedule` is a callable rather than an
            int literal.
    """
    tx = optimizer.tx
    states = _count_multi_steps_states(optimizer.opt_state)
    if not isinstance(tx, optax.MultiSteps):
        if states:
            raise ValueError(
                "MultiSteps must be the outermost transformation for the learner to read its "
                "accumulation window: the optimizer state carries a MultiStepsState, but the "
                "instance hides inside a wrapper such as optax.chain."
            )
        return 1
    if states > 1:
        raise ValueError(
            "MultiSteps nests a second MultiSteps: the optimizer state carries another "
            "MultiStepsState inside the outermost one, so the device would apply at the product of "
            "the windows while the learner gates on the outermost window alone."
        )
    if getattr(tx._should_skip_update_fn, "__qualname__", "") != "MultiSteps.__init__.<locals>.should_skip_update_fn":
        raise ValueError(
            "MultiSteps carries a should_skip_update_fn: a skipped update would desynchronize the "
            "device counter from the update gate the learner derives from the window."
        )
    if getattr(tx._every_k_schedule, "__qualname__", "") != "MultiSteps.__init__.<locals>.<lambda>":
        raise ValueError(
            "MultiSteps has no int literal every_k_schedule: only a literal window can be read back "
            "into the learner's update gate, not a schedule or callable."
        )
    return int(tx._every_k_schedule(0))


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


__all__ = ["accumulation_window", "get_learning_rate", "no_weight_decay_mask", "unwrap_variables"]


if not TYPE_CHECKING:
    import sys

    from structcast.utils.lazy_import import LazySelectedImporter

    sys.modules[__name__] = LazySelectedImporter(__name__, globals())
