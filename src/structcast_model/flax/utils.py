"""Shared JAX device helpers and the compilation contract of a generated step."""

from collections import OrderedDict
from collections.abc import Callable
from functools import lru_cache
from inspect import Parameter, signature
from typing import TYPE_CHECKING, Any

import jax


# `jax.Device` is Any to mypy: jaxlib re-exports it from its `_jax` C extension, which ships no stubs.
@lru_cache(maxsize=1)
def get_jax_devices() -> OrderedDict[str, jax.Device]:  # type: ignore[no-any-unimported]
    """Get a mapping of available JAX devices.

    Returns:
        OrderedDict[str, jax.Device]: An ordered dictionary mapping device strings (e.g., "cpu:0", "gpu:0") to
            JAX Device objects.
    """
    return OrderedDict((f"{d.platform}:{d.id}", d) for d in jax.devices())


# `jax.Device` is Any to mypy, as above.
def get_jax_device(device: str | None = None) -> jax.Device:  # type: ignore[no-any-unimported]
    """Get a JAX device based on the provided device string.

    Args:
        device (str | None): The device string to look for (e.g., "cpu:0", "gpu:0").
            If None, the first available device will be returned.

    Returns:
        jax.Device: The JAX device corresponding to the provided device string.
    """
    devices = get_jax_devices()
    if not devices:
        raise ValueError("No JAX devices are available.")
    if device is None:
        device = next(iter(devices))
    if device in devices:
        return devices[device]
    devices_str = ", ".join(f"{d!r}" for d in devices)
    raise ValueError(f"Specified device {device!r} is not available. Available devices: {devices_str}")


def donate_argnames(function: Callable[..., Any]) -> tuple[str, ...]:
    """Return the arguments of a training step to donate when compiling it, its batch excluded.

    The signature of a step is its donation contract (see `docs/adr/0019`): a generated training
    step takes every model and optimizer as its own positional-or-keyword parameter and the batch as
    keyword-only parameters, so the state it rewrites in place is exactly what it declares
    positionally. Donating that state is what keeps a compiled run from copying every parameter
    buffer once per step, and leaving the batch out of the donation is what keeps the caller's own
    arrays usable afterwards. A hand-written step following the same convention is donated the same
    way; one taking a non-state argument positionally sees it donated too, which is harmless for a
    per-step batch. `inspect.signature` follows `__wrapped__`, so a step another layer already
    wrapped still reports the parameters underneath. A positional-only parameter is not donated
    either: it has no name to donate by, and a step declaring one is outside the contract.

    Args:
        function (Callable[..., Any]): The step about to be compiled.

    Returns:
        tuple[str, ...]: The names of its positional-or-keyword parameters, empty when the callable
            has no readable signature -- donating nothing only costs the copy donation would save.

    Example:
        >>> from structcast_model.flax.utils import donate_argnames
        >>> def _training_step(model, optimizer, *, x, **kwargs): ...
        >>> donate_argnames(_training_step)
        ('model', 'optimizer')
    """
    try:
        parameters = signature(function).parameters.values()
    except (TypeError, ValueError):
        return ()
    return tuple(p.name for p in parameters if p.kind is Parameter.POSITIONAL_OR_KEYWORD)


def dot_general_out(*spec: Any) -> Callable[..., Any]:
    """Return a `jax.lax.dot_general` that places its result on `PartitionSpec(*spec)`.

    The hook a tensor-parallel layer needs when the model axis is Explicit and the compiler is
    therefore not allowed to pick the output's sharding itself. A model template reaches it through
    an `eval:` expression -- `dot_general: "eval: dot_general_out(None, 'model')"` -- so the
    annotation stays in the template while the strategy decides whether the axis is Explicit at all.

    A closure and not `functools.partial`: Flax passes `out_sharding=` explicitly on every call to
    the hook (`None` when its own caller gave none), and that pass-through silently overrides a
    keyword `partial` bound. The keyword is therefore overridden here instead.

    Args:
        *spec (Any): The `PartitionSpec` entries of the output, e.g. `None, "model"`.

    Returns:
        Callable[..., Any]: A drop-in `dot_general` naming that sharding on every call.

    Example:
        >>> import jax, jax.numpy as jnp
        >>> from structcast_model.flax.utils import dot_general_out
        >>> with jax.set_mesh(jax.make_mesh((1,), ("model",))):
        ...     out = dot_general_out(None, "model")(jnp.ones((4, 3)), jnp.ones((3, 2)), (((1,), (0,)), ((), ())))
        >>> str(out.sharding.spec)
        "P(None, 'model')"
    """

    def dot_general(*args: Any, **kwargs: Any) -> Any:
        """Run `jax.lax.dot_general` with the captured output sharding, whatever the caller asked for."""
        return jax.lax.dot_general(*args, **{**kwargs, "out_sharding": jax.sharding.PartitionSpec(*spec)})

    return dot_general


__all__ = ["donate_argnames", "dot_general_out", "get_jax_device", "get_jax_devices"]


if not TYPE_CHECKING:
    import sys

    from structcast.utils.lazy_import import LazySelectedImporter

    sys.modules[__name__] = LazySelectedImporter(__name__, globals())
