"""Trainer helpers for Flax models."""

from collections import OrderedDict
from collections.abc import Mapping
from functools import lru_cache
from typing import TYPE_CHECKING, Any

import jax
import numpy as np
from pydantic import TypeAdapter, ValidationError


def create_jax_inputs(shape: Any, *, batch_size: int = 1) -> Any:
    """Create dummy JAX inputs based on the provided shape.

    Args:
        shape (Any): The shape of the inputs to create. This can be a tuple of integers,
            a dictionary of shapes, or a list of shapes.
        batch_size (int): The batch size to use for the inputs.

    Returns:
        Any: The created inputs, which can be a JAX array, a dictionary of arrays, or a list of arrays.
    """
    try:
        shape = TypeAdapter(tuple[int, ...]).validate_python(shape)
        return jax.numpy.array(np.random.rand(batch_size, *shape).astype(np.float32))
    except ValidationError:
        pass
    if isinstance(shape, Mapping):
        return {key: create_jax_inputs(value, batch_size=batch_size) for key, value in shape.items()}
    if isinstance(shape, (list, tuple)):
        return [create_jax_inputs(value, batch_size=batch_size) for value in shape]
    raise ValueError(f"Invalid tensor shape: {shape}")


@lru_cache(maxsize=1)
def get_jax_devices() -> OrderedDict[str, jax.Device]:
    """Get a mapping of available JAX devices.

    Returns:
        OrderedDict[str, jax.Device]: An ordered dictionary mapping device strings (e.g., "cpu:0", "gpu:0") to
            JAX Device objects.
    """
    return OrderedDict((f"{d.platform}:{d.id}", d) for d in jax.devices())


def get_jax_device(device: str | None = None) -> jax.Device:
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


__all__ = ["create_jax_inputs"]


if not TYPE_CHECKING:
    import sys

    from structcast.utils.lazy_import import LazySelectedImporter

    sys.modules[__name__] = LazySelectedImporter(__name__, globals())
