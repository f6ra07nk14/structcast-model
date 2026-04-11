"""Unit tests for structcast_model.flax.trainer."""

from __future__ import annotations

from collections import OrderedDict

import jax
import jax.numpy as jnp
import pytest

from structcast_model.flax.trainer import create_jax_inputs, get_jax_device, get_jax_devices


def test_create_jax_inputs_from_int_tuple_returns_array() -> None:
    """A tuple of ints produces a float32 JAX array with batch dimension 1."""
    result = create_jax_inputs((3, 4))
    assert result.shape == (1, 3, 4)
    assert result.dtype == jnp.float32


def test_create_jax_inputs_from_list_returns_list() -> None:
    """A list of shapes returns a list of JAX arrays."""
    result = create_jax_inputs([(3,), (4,)])
    assert isinstance(result, list)
    assert len(result) == 2
    assert all(getattr(item, "shape", None) is not None for item in result)


def test_create_jax_inputs_from_dict_returns_dict() -> None:
    """A dict of shapes returns a dict of JAX arrays."""
    result = create_jax_inputs({"image": (3, 4), "mask": (1, 4)})
    assert isinstance(result, dict)
    assert set(result.keys()) == {"image", "mask"}
    assert all(getattr(item, "shape", None) is not None for item in result.values())


def test_create_jax_inputs_invalid_shape_raises() -> None:
    """A non-shape scalar raises ValueError."""
    with pytest.raises(ValueError, match="Invalid tensor shape"):
        create_jax_inputs("not_a_shape")


def test_create_jax_inputs_custom_batch_size() -> None:
    """Custom batch_size is prepended to the shape."""
    result = create_jax_inputs((5,), batch_size=4)
    assert result.shape == (4, 5)


# ---------------------------------------------------------------------------
# get_jax_devices
# ---------------------------------------------------------------------------


def test_get_jax_devices_returns_ordered_dict() -> None:
    """get_jax_devices returns an OrderedDict of JAX devices."""
    devices = get_jax_devices()
    assert isinstance(devices, OrderedDict)
    assert len(devices) > 0
    for key, dev in devices.items():
        assert isinstance(key, str)
        assert isinstance(dev, jax.Device)


def test_get_jax_devices_keys_match_platform_id() -> None:
    """Each key has the form 'platform:id'."""
    devices = get_jax_devices()
    for key, dev in devices.items():
        assert key == f"{dev.platform}:{dev.id}"


# ---------------------------------------------------------------------------
# get_jax_device
# ---------------------------------------------------------------------------


def test_get_jax_device_default() -> None:
    """get_jax_device with no arg returns the first available device."""
    device = get_jax_device()
    assert isinstance(device, jax.Device)
    first_key = next(iter(get_jax_devices()))
    assert device is get_jax_devices()[first_key]


def test_get_jax_device_explicit_valid() -> None:
    """get_jax_device returns the requested device when it exists."""
    devices = get_jax_devices()
    key = next(iter(devices))
    device = get_jax_device(key)
    assert device is devices[key]


def test_get_jax_device_invalid_raises() -> None:
    """get_jax_device raises ValueError for a non-existent device string."""
    with pytest.raises(ValueError, match="not available"):
        get_jax_device("nonexistent:99")
