"""Shared JAX device helpers."""

from collections import OrderedDict
from functools import lru_cache
from typing import TYPE_CHECKING

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


__all__ = ["get_jax_device", "get_jax_devices"]


if not TYPE_CHECKING:
    import sys

    from structcast.utils.lazy_import import LazySelectedImporter

    sys.modules[__name__] = LazySelectedImporter(__name__, globals())
