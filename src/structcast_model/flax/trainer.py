"""Trainer helpers for Flax models."""

from collections import OrderedDict
from collections.abc import Mapping
from functools import lru_cache
from typing import TYPE_CHECKING, Any

import jax
import numpy as np
from pydantic import TypeAdapter, ValidationError

# Protocol and runtime_checkable come from typing_extensions so that isinstance checks use
# inspect.getattr_static on Python 3.11 as well (backported from 3.12), as in base_trainer.
from typing_extensions import Protocol, runtime_checkable

from structcast_model.builders.schema import TensorSpec, TensorSpecTree
from structcast_model.utils.base import resolve_input_shapes, resolve_tensor_initializer

DTYPES = {
    "float32": jax.numpy.float32,
    "float16": jax.numpy.float16,
    "bfloat16": jax.numpy.bfloat16,
    "int32": jax.numpy.int32,
    "int64": jax.numpy.int64,
}
"""JAX element types of the supported tensor element types.

`int64` is truncated to `int32` unless JAX is configured with `jax_enable_x64`.
"""


@runtime_checkable
class TensorInitializer(Protocol):
    """Callable creating a dummy JAX array, called as `initializer(size, dtype=...)`."""

    def __call__(self, size: tuple[int, ...], *, dtype: Any) -> Any:
        """Create an array of the given size and element type."""
        ...


def random_array(size: tuple[int, ...], *, dtype: Any) -> Any:
    """Create a uniformly distributed random JAX array, the default initializer for floating point types.

    `jax.random` cannot be used as an initializer directly, since it requires an explicit key.

    Args:
        size (tuple[int, ...]): The size of the array, including the batch dimension.
        dtype (Any): The element type of the array.

    Returns:
        Any: The created array.
    """
    return jax.numpy.array(np.random.random(size), dtype=dtype)


def create_jax_inputs(shape: Any, *, batch_size: int = 1) -> Any:
    """Create dummy JAX inputs based on the provided shape.

    Args:
        shape (Any): The shape of the inputs to create. This can be a tensor specification,
            which is a tuple of integers or a mapping with the `_SHAPE_` key,
            a dictionary of shapes, or a list of shapes.
        batch_size (int): The batch size to use for the inputs.
            This will be prepended to the shape of every tensor specification.

    Returns:
        Any: The created inputs, which can be a JAX array, a dictionary of arrays, or a list of arrays.

    Raises:
        ValueError: If the shape is neither a tensor specification nor a dictionary or list nesting more of them.
    """
    try:
        node: TensorSpecTree = TypeAdapter(TensorSpecTree).validate_python(shape)
    except ValidationError:
        raise ValueError(f"Invalid tensor shape: {shape}") from None
    if isinstance(node, TensorSpec):
        initializer = resolve_tensor_initializer(
            node.INIT,
            node.DTYPE,
            float_default=random_array,
            int_default=jax.numpy.zeros,
            protocol=TensorInitializer,
        )
        return initializer((batch_size, *node.SHAPE), dtype=DTYPES[node.DTYPE])
    if isinstance(node, Mapping):
        return {key: create_jax_inputs(value, batch_size=batch_size) for key, value in node.items()}
    return [create_jax_inputs(value, batch_size=batch_size) for value in node]


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


__all__ = ["TensorInitializer", "create_jax_inputs", "get_jax_device", "get_jax_devices", "resolve_input_shapes"]


if not TYPE_CHECKING:
    import sys

    from structcast.utils.lazy_import import LazySelectedImporter

    sys.modules[__name__] = LazySelectedImporter(__name__, globals())
