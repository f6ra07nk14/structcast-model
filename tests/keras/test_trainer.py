"""Unit tests for structcast_model.keras.trainer."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import keras
from structcast_model.keras.trainer import (
    create_keras_inputs,
    create_numpy_inputs,
    get_keras_device,
    initial_model,
)


def test_create_numpy_inputs_from_int_tuple_returns_array() -> None:
    """A tuple of ints produces a float32 NumPy array with batch dimension 1."""
    result = create_numpy_inputs((3, 4))
    assert isinstance(result, np.ndarray)
    assert result.shape == (1, 3, 4)
    assert result.dtype == np.float32


def test_create_numpy_inputs_from_dict_returns_dict() -> None:
    """A dict of shapes returns a dict of NumPy arrays."""
    result = create_numpy_inputs({"image": (3, 4), "mask": (1, 4)})
    assert isinstance(result, dict)
    assert set(result.keys()) == {"image", "mask"}
    assert all(isinstance(value, np.ndarray) for value in result.values())


def test_create_numpy_inputs_invalid_shape_raises() -> None:
    """A non-shape scalar raises ValueError."""
    with pytest.raises(ValueError, match="Invalid tensor shape"):
        create_numpy_inputs("not_a_shape")


def test_create_keras_inputs_from_dict_returns_dict() -> None:
    """A dict of shapes returns a dict of Keras input tensors."""
    result = create_keras_inputs({"x": (3,), "y": (2,)})
    assert isinstance(result, dict)
    assert set(result.keys()) == {"x", "y"}
    assert tuple(result["x"].shape) == (None, 3)
    assert tuple(result["y"].shape) == (None, 2)


def test_create_keras_inputs_invalid_shape_raises() -> None:
    """A non-shape scalar raises ValueError."""
    with pytest.raises(ValueError, match="Invalid tensor shape"):
        create_keras_inputs("not_a_shape")


def test_initial_model_returns_existing_model_when_shapes_is_none() -> None:
    """An existing Keras model is returned unchanged when no shapes are provided."""
    inputs = keras.Input(shape=(3,))
    outputs = keras.layers.Lambda(lambda x: x)(inputs)
    model = keras.Model(inputs=inputs, outputs=outputs)
    assert initial_model(model, shapes=None) is model


def test_initial_model_raises_for_layer_without_shapes() -> None:
    """A non-model Keras callable requires shapes for initialization."""
    with pytest.raises(ValueError, match="Input shapes are required"):
        initial_model(keras.layers.Dense(2), shapes=None)


def test_initial_model_builds_model_from_symbolic_inputs() -> None:
    """A Keras layer is wrapped into a built Keras model using symbolic inputs."""

    class AddLayer(keras.layers.Layer):
        def call(self, x: Any, y: Any) -> Any:
            """Add two inputs."""
            return x + y

    model = initial_model(AddLayer(), {"x": (3,), "y": (3,)})
    assert isinstance(model, keras.Model)

    outputs = model(
        {
            "x": np.ones((1, 3), dtype=np.float32),
            "y": np.full((1, 3), 2.0, dtype=np.float32),
        }
    )
    np.testing.assert_allclose(np.array(outputs), np.full((1, 3), 3.0, dtype=np.float32))


# ---------------------------------------------------------------------------
# create_numpy_inputs — additional branches
# ---------------------------------------------------------------------------


def test_create_numpy_inputs_from_list_returns_list() -> None:
    """A list of shapes produces a list of NumPy arrays."""
    result = create_numpy_inputs([(3,), (4, 5)])
    assert isinstance(result, list)
    assert len(result) == 2
    assert result[0].shape == (1, 3)
    assert result[1].shape == (1, 4, 5)


def test_create_numpy_inputs_custom_batch_size() -> None:
    """Custom batch_size is respected."""
    result = create_numpy_inputs((2,), batch_size=8)
    assert result.shape == (8, 2)


# ---------------------------------------------------------------------------
# create_keras_inputs — additional branches
# ---------------------------------------------------------------------------


def test_create_keras_inputs_from_tuple_returns_tensor() -> None:
    """A tuple of ints produces a single Keras Input tensor."""
    result = create_keras_inputs((3, 4))
    assert tuple(result.shape) == (None, 3, 4)


def test_create_keras_inputs_from_list_returns_list() -> None:
    """A list of shapes produces a list of Keras Input tensors."""
    result = create_keras_inputs([(3,), (4,)])
    assert isinstance(result, list)
    assert len(result) == 2
    assert tuple(result[0].shape) == (None, 3)
    assert tuple(result[1].shape) == (None, 4)


def test_create_keras_inputs_with_batch_size() -> None:
    """Batch size is attached to symbolic input when specified."""
    result = create_keras_inputs((5,), batch_size=4)
    assert tuple(result.shape) == (4, 5)


# ---------------------------------------------------------------------------
# initial_model — list inputs
# ---------------------------------------------------------------------------


def test_initial_model_with_list_inputs() -> None:
    """A layer accepting positional args is wrapped via list shaped inputs."""

    class ConcatLayer(keras.layers.Layer):
        def call(self, a: Any, b: Any) -> Any:
            """Concatenate two inputs."""
            return keras.ops.concatenate([a, b], axis=-1)

    model = initial_model(ConcatLayer(), [(3,), (2,)])
    assert isinstance(model, keras.Model)
    out = model([np.ones((1, 3), dtype=np.float32), np.ones((1, 2), dtype=np.float32)])
    assert np.array(out).shape == (1, 5)


# ---------------------------------------------------------------------------
# get_keras_device
# ---------------------------------------------------------------------------


def test_get_keras_device_default() -> None:
    """get_keras_device with no arg returns a device from the available list."""
    device = get_keras_device()
    assert isinstance(device, str)
    assert device in keras.distribution.list_devices()


def test_get_keras_device_explicit_valid() -> None:
    """get_keras_device returns the specified device when it exists."""
    available = keras.distribution.list_devices()
    device = get_keras_device(available[0])
    assert device == available[0]


def test_get_keras_device_invalid_raises() -> None:
    """get_keras_device raises ValueError for a non-existent device."""
    with pytest.raises(ValueError, match="not available"):
        get_keras_device("nonexistent_device:99")
