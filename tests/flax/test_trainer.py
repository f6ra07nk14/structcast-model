"""Unit tests for structcast_model.flax.trainer."""

from __future__ import annotations

from collections import OrderedDict
import logging

import jax
import jax.numpy as jnp
import pytest

from structcast_model.flax.trainer import create_jax_inputs, get_jax_device, get_jax_devices


def test_create_jax_inputs_from_int_tuple_returns_array() -> None:
    """A tuple of ints produces a bfloat16 JAX array with batch dimension 1, bfloat16 being the default dtype."""
    result = create_jax_inputs((3, 4))
    assert result.shape == (1, 3, 4)
    assert result.dtype == jnp.bfloat16


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


def test_create_jax_inputs_int_dtype_falls_back_to_zeros_with_warning(caplog: pytest.LogCaptureFixture) -> None:
    """An integer dtype without an initializer falls back to zeros, because random floats cannot be integers.

    The fallback is a guess about the caller's intent, so it must be reported.
    JAX truncates `int64` to `int32` unless `jax_enable_x64` is set, so only the kind of the dtype is asserted.
    """
    with caplog.at_level(logging.WARNING):
        result = create_jax_inputs({"_SHAPE_": [5], "_DTYPE_": "int64"})
    assert jnp.issubdtype(result.dtype, jnp.integer)
    assert jnp.array_equal(result, jnp.zeros((1, 5), dtype=result.dtype))
    assert "Falling back to zeros" in caplog.text


def test_create_jax_inputs_honours_explicit_initializer() -> None:
    """An explicit `_INIT_` address replaces the dtype-based default initializer."""
    result = create_jax_inputs({"_SHAPE_": [4], "_INIT_": "jax.numpy.ones"})
    assert jnp.array_equal(result, jnp.ones((1, 4), dtype=jnp.bfloat16))


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
