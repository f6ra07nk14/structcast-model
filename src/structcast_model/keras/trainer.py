"""Trainer helpers for Keras models."""

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import numpy as np
from pydantic import TypeAdapter, ValidationError

import keras


def create_numpy_inputs(shape: Any, *, batch_size: int = 1) -> Any:
    """Create dummy NumPy inputs based on the provided shape.

    Args:
        shape (Any): The shape of the inputs to create. This can be a tuple of integers,
            a dictionary of shapes, or a list of shapes.
        batch_size (int): The batch size to use for the inputs.

    Returns:
        Any: The created inputs, which can be a NumPy array, a dictionary of arrays, or a list of arrays.
    """
    try:
        shape = TypeAdapter(tuple[int, ...]).validate_python(shape)
        return np.random.rand(batch_size, *shape).astype(np.float32)
    except ValidationError:
        pass
    if isinstance(shape, Mapping):
        return {k: create_numpy_inputs(v, batch_size=batch_size) for k, v in shape.items()}
    if isinstance(shape, (tuple, list)):
        return [create_numpy_inputs(v, batch_size=batch_size) for v in shape]
    raise ValueError(f"Invalid tensor shape: {shape}")


def create_keras_inputs(shape: Any, *, batch_size: int | None = None, name: str | None = None, **kwargs: Any) -> Any:
    """Create symbolic Keras inputs based on the provided shape.

    Args:
        shape (Any): The shape of the inputs to create. This can be a tuple of integers,
            a dictionary of shapes, or a list of shapes.
        batch_size (int | None): The optional batch size to bind to the symbolic inputs.
        name (str | None): The optional base name to use for the input tensors.
        **kwargs (Any): Additional keyword arguments to pass to keras.Input.

    Returns:
        Any: The created inputs, which can be a Keras tensor, a dictionary of tensors, or a list of tensors.
    """
    try:
        shape = TypeAdapter(tuple[int, ...]).validate_python(shape)
        return keras.Input(shape=shape, batch_size=batch_size, name=name, **kwargs)
    except ValidationError:
        pass
    if isinstance(shape, Mapping):
        return {
            k: create_keras_inputs(v, batch_size=batch_size, name=k if name is None else f"{name}_{k}", **kwargs)
            for k, v in shape.items()
        }
    if isinstance(shape, (tuple, list)):
        return [
            create_keras_inputs(v, batch_size=batch_size, name=f"{name or 'input'}_{i}", **kwargs)
            for i, v in enumerate(shape)
        ]
    raise ValueError(f"Invalid tensor shape: {shape}")


def initial_model(model: Any, shapes: Any) -> Any:
    """Initialize a Keras model by tracing it with symbolic inputs.

    Args:
        model: The model or layer to initialize.
        shapes: A dictionary mapping input names to their tensor shapes.

    Returns:
        A built Keras model.

    Raises:
        ValueError: If shapes are not provided for a non-Model callable.
    """
    if isinstance(model, keras.Model):
        return model
    if shapes is None:
        raise ValueError("Input shapes are required to initialize a Keras layer into a model.")
    inputs = create_keras_inputs(shapes)
    if isinstance(inputs, Mapping):
        outputs = model(**inputs)
    elif isinstance(inputs, (tuple, list)):
        outputs = model(*inputs)
    else:
        outputs = model(inputs)
    return keras.Model(inputs=inputs, outputs=outputs)


def get_keras_device(device: str | None = None) -> str:
    """Get a list of available Keras devices."""
    devices = keras.distribution.list_devices()
    if not devices:
        raise ValueError("No Keras devices are available.")
    if device is None:
        device = next(iter(devices))
    if device in devices:
        return device
    devices_str = ", ".join(f"{d!r}" for d in devices)
    raise ValueError(f"Specified device {device!r} is not available. Available devices: {devices_str}")


__all__ = ["create_keras_inputs", "create_numpy_inputs", "get_keras_device", "initial_model"]


if not TYPE_CHECKING:
    import sys

    from structcast.utils.lazy_import import LazySelectedImporter

    sys.modules[__name__] = LazySelectedImporter(__name__, globals())
