"""Unit tests for structcast_model.flax.trainer."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from structcast_model.flax.trainer import create_jax_inputs


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
