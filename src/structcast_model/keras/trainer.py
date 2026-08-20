"""Trainer helpers for Keras models."""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import ml_dtypes
import numpy as np
from pydantic import TypeAdapter, ValidationError

# Protocol and runtime_checkable come from typing_extensions so that isinstance checks use
# inspect.getattr_static on Python 3.11 as well (backported from 3.12), as in base_trainer.
from typing_extensions import Protocol, runtime_checkable

import keras
from structcast_model.base_trainer import BaseInfo, BaseTrainer, BestCriterion
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


@dataclass(kw_only=True, slots=True)
class KerasTracker:
    """Running mean of the criteria of one training or validation split.

    The sums stay device-side tensors, so accumulating a step is one backend `add` on whichever
    backend `keras.ops` dispatches to. The means come back as Python floats, not tensors:
    `BaseTrainer.tracker` is typed `Callable[..., dict[str, float]]` and everything downstream --
    the epoch history, the `BestCriterion` comparison against a float infinity, and the loggers'
    `log_metric(value: float)` -- consumes that contract, as the torch tracker's `.item()` does.
    `keras.ops.convert_to_numpy` is the backend-neutral host read, and it lands inside the region
    `BaseTrainer.train` times, where the torch loop blocks on `torch.cuda.synchronize()` for the
    same reason.

    Unlike the torch tracker there is no all-reduce: a distributed step's criteria are reduced by
    the backend adapter that ran the step, so what reaches this tracker is already one value
    (`docs/adr/0016`).

    Example:
        >>> import keras
        >>> from structcast_model.keras.trainer import KerasTracker
        >>> tracker = KerasTracker.from_criteria(["loss"])
        >>> tracker(loss=keras.ops.convert_to_tensor(1.0))
        {'loss': 1.0}
        >>> tracker(loss=keras.ops.convert_to_tensor(3.0))
        {'loss': 2.0}
        >>> tracker.reset()
        >>> tracker.logs()
        {}
    """

    criteria: tuple[str, ...]
    """Names of the criteria to track; a step must report every one of them."""

    sums: dict[str, Any] = field(init=False)
    """Device-side sum of each criterion over the steps seen since the last reset."""

    count: int = field(default=0, init=False)
    """Number of steps summed since the last reset."""

    def __post_init__(self) -> None:
        """Start every criterion at a zero sum."""
        self.reset()

    def reset(self) -> None:
        """Zero the sums and the step count."""
        # float32 whatever the criteria are: a run under a float16 or bfloat16 policy would
        # otherwise accumulate an epoch of steps in the reduced type and lose the small ones.
        self.sums = {criterion: keras.ops.zeros((), dtype="float32") for criterion in self.criteria}
        self.count = 0

    def on_training_begin(self, info: BaseInfo[Any]) -> None:
        """Reset the tracker so an epoch's training averages start empty."""
        self.reset()

    def on_validation_begin(self, info: BaseInfo[Any]) -> None:
        """Reset the tracker so validation averages do not carry training values."""
        self.reset()

    def __call__(self, **criteria: Any) -> dict[str, float]:
        """Add one step's criteria to the sums and return the running means."""
        for criterion in self.criteria:
            self.sums[criterion] = keras.ops.add(self.sums[criterion], criteria[criterion])
        self.count += 1
        return self.logs()

    def logs(self) -> dict[str, float]:
        """Return the running mean of each criterion, or an empty mapping before the first step."""
        if not self.count:
            return {}
        return {name: float(keras.ops.convert_to_numpy(total)) / self.count for name, total in self.sums.items()}

    @classmethod
    def from_criteria(cls, outputs: Iterable[str]) -> "KerasTracker":
        """Create a tracker for the named criteria, mirroring `TorchTracker.from_criteria`."""
        return cls(criteria=tuple(outputs))


@dataclass(kw_only=True)
class KerasTrainer(BaseTrainer[Any]):
    """Trainer for Keras models.

    `BaseTrainer.sync` stays the inherited no-op: `KerasTracker.logs` converts the criteria to
    NumPy on every step, which already waits for the step's computation on every backend, and the
    device a run uses is chosen by the backend adapter, not by a trainer field.

    The model type stays `Any`: keras ships no py.typed, so `keras.Model` is `Any` to a type
    checker anyway.
    """


@dataclass(kw_only=True, slots=True)
class KerasBestCriterion(BestCriterion[Any]):
    """A callback tracking the best value of a criterion during training or validation for Keras models.

    The twin of `structcast_model.torch.trainer.TorchBestCriterion`, duplicated rather than shared
    because importing either module imports its framework. The `from_criteria` builder of the twins
    also saves the best states, which needs the Keras state strategy, so it arrives with it.
    """


__all__ = [
    "KerasBestCriterion",
    "KerasTracker",
    "KerasTrainer",
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
