"""Trainer helpers for Keras models."""

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import ml_dtypes
import numpy as np
from pydantic import TypeAdapter, ValidationError

# Protocol and runtime_checkable come from typing_extensions so that isinstance checks use
# inspect.getattr_static on Python 3.11 as well (backported from 3.12), as in base_trainer.
from typing_extensions import Protocol, runtime_checkable

import keras
from structcast_model.builders.schema import TensorSpec, TensorSpecTree
from structcast_model.utils.base import resolve_input_shapes, resolve_tensor_initializer

DTYPES = {
    "float32": np.float32,
    "float16": np.float16,
    "bfloat16": ml_dtypes.bfloat16,
    "int32": np.int32,
    "int64": np.int64,
}
"""NumPy element types of the supported tensor element types.

NumPy has no native `bfloat16`, so the type registered by `ml_dtypes`, a hard dependency of Keras, is used.
"""


@runtime_checkable
class TensorInitializer(Protocol):
    """Callable creating a dummy NumPy array, called as `initializer(size, dtype=...)`."""

    def __call__(self, size: tuple[int, ...], *, dtype: Any) -> Any:
        """Create an array of the given size and element type."""
        ...


def random_array(size: tuple[int, ...], *, dtype: Any) -> Any:
    """Create a uniformly distributed random NumPy array, the default initializer for floating point types.

    `numpy.random.rand` cannot be used as an initializer directly,
    since it takes the size as variadic arguments and has no `dtype` argument.

    Args:
        size (tuple[int, ...]): The size of the array, including the batch dimension.
        dtype (Any): The element type of the array.

    Returns:
        Any: The created array.
    """
    return np.random.random(size).astype(dtype)


def create_numpy_inputs(shape: Any, *, batch_size: int = 1) -> Any:
    """Create dummy NumPy inputs based on the provided shape.

    Args:
        shape (Any): The shape of the inputs to create. This can be a tensor specification,
            which is a tuple of integers or a mapping with the `_SHAPE_` key,
            a dictionary of shapes, or a list of shapes.
        batch_size (int): The batch size to use for the inputs.
            This will be prepended to the shape of every tensor specification.

    Returns:
        Any: The created inputs, which can be a NumPy array, a dictionary of arrays, or a list of arrays.

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
            int_default=np.zeros,
            protocol=TensorInitializer,
        )
        return initializer((batch_size, *node.SHAPE), dtype=DTYPES[node.DTYPE])
    if isinstance(node, Mapping):
        return {k: create_numpy_inputs(v, batch_size=batch_size) for k, v in node.items()}
    return [create_numpy_inputs(v, batch_size=batch_size) for v in node]


def create_keras_inputs(shape: Any, *, batch_size: int | None = None, name: str | None = None, **kwargs: Any) -> Any:
    """Create symbolic Keras inputs based on the provided shape.

    Args:
        shape (Any): The shape of the inputs to create. This can be a tensor specification,
            which is a tuple of integers or a mapping with the `_SHAPE_` key,
            a dictionary of shapes, or a list of shapes.
        batch_size (int | None): The optional batch size to bind to the symbolic inputs.
        name (str | None): The optional base name to use for the input tensors.
        **kwargs (Any): Additional keyword arguments to pass to keras.Input,
            taking precedence over the element type of the tensor specification.

    Returns:
        Any: The created inputs, which can be a Keras tensor, a dictionary of tensors, or a list of tensors.

    Raises:
        ValueError: If the shape is neither a tensor specification nor a dictionary or list nesting more of them.
    """
    try:
        node: TensorSpecTree = TypeAdapter(TensorSpecTree).validate_python(shape)
    except ValidationError:
        raise ValueError(f"Invalid tensor shape: {shape}") from None
    if isinstance(node, TensorSpec):
        kwargs.setdefault("dtype", node.DTYPE)
        return keras.Input(shape=node.SHAPE, batch_size=batch_size, name=name, **kwargs)
    if isinstance(node, Mapping):
        return {
            k: create_keras_inputs(v, batch_size=batch_size, name=k if name is None else f"{name}_{k}", **kwargs)
            for k, v in node.items()
        }
    return [
        create_keras_inputs(v, batch_size=batch_size, name=f"{name or 'input'}_{i}", **kwargs)
        for i, v in enumerate(node)
    ]


def initial_model(model: Any, shapes: Any) -> Any:
    """Initialize a Keras model by tracing it with symbolic inputs.

    Args:
        model: The model or layer to initialize.
        shapes: A dictionary mapping input names to their tensor shapes.
            If empty or None, the shapes declared by the model itself are used.

    Returns:
        A built Keras model.

    Raises:
        ValueError: If shapes are neither provided nor declared by a non-Model callable.
    """
    if isinstance(model, keras.Model):
        return model
    shapes = resolve_input_shapes(model, shapes)
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


__all__ = [
    "TensorInitializer",
    "create_keras_inputs",
    "create_numpy_inputs",
    "get_keras_device",
    "initial_model",
    "resolve_input_shapes",
]


if not TYPE_CHECKING:
    import sys

    from structcast.utils.lazy_import import LazySelectedImporter

    sys.modules[__name__] = LazySelectedImporter(__name__, globals())
