"""Trainer for PyTorch models."""

from collections.abc import Callable, Iterable, Mapping
from contextlib import AbstractContextManager, suppress
from dataclasses import dataclass, field
from functools import partial
from logging import getLogger
from typing import TYPE_CHECKING, Any, TypeVar

from pydantic import TypeAdapter, ValidationError
from timm.data import Mixup
from timm.utils import ModelEmaV3
from torch.nn import Module
from torch.utils.data import DataLoader

from structcast_model.base_trainer import GLOBAL_CALLBACKS, BaseInfo, BaseTrainer, Callback, invoke_callback
from structcast_model.torch.layers.criteria_tracker import CriteriaTracker
from structcast_model.torch.types import Tensor
from torch import autocast, bfloat16, cuda, device, float16, float32, no_grad, rand

logger = getLogger(__name__)

DTYPES = {
    "float32": float32,
    "float16": float16,
    "bfloat16": bfloat16,
}

T = TypeVar("T")


def create_torch_inputs(shape: Any) -> Any:
    """Create dummy inputs based on the provided shape."""
    try:
        return rand((1, *TypeAdapter(tuple[int, ...]).validate_python(shape)), dtype=float32)
    except ValidationError:
        pass
    if isinstance(shape, dict):
        return {k: create_torch_inputs(v) for k, v in shape.items()}
    if isinstance(shape, (list, tuple)):
        return [create_torch_inputs(v) for v in shape]
    raise ValueError(f"Invalid tensor shape: {shape}")


def get_torch_device(device: str | None = None) -> str:
    """Get the device to run the model on."""
    if device is None:
        return "cuda" if cuda.is_available() else "cpu"
    if device not in ["cpu", "cuda"]:
        raise ValueError(f'Only "cpu" and "cuda" are supported. Got invalid device: {device}')
    if device == "cuda" and not cuda.is_available():
        logger.warning("CUDA is not available. Using CPU instead.")
        return "cpu"
    return device


def initial_model(
    model: T,
    shapes: dict[str, Any] | None = None,
    compile_fn: Callable[[Module], Module] | None = None,
) -> tuple[T, Any, Any]:
    """Initialize the model by creating dummy inputs based on the provided shapes and running a forward pass.

    Args:
        model (T): The model to initialize. Can be any nested structure containing PyTorch modules.
        shapes (dict[str, Any] | None): A dictionary mapping module names to their input shapes.
            If None, the model will not be initialized with dummy inputs.
        compile_fn (Callable[[Module], Module] | None): An optional function to compile the model after initialization.
            If None, the model will not be compiled.

    Returns:
        A tuple containing the initialized (and optionally compiled) model, the dummy inputs used for initialization,
            and the outputs of the forward pass.
    """
    inputs = None if shapes is None else create_torch_inputs(shapes)
    outputs = {}

    def _init(raw: Any) -> Any:
        if isinstance(raw, Module):
            outputs[raw] = None if inputs is None else raw(**inputs)
            return raw if compile_fn is None else compile_fn(raw)
        if isinstance(raw, Mapping):
            res = {k: _init(v) for k, v in raw.items()}
            return res if (cls := type(raw)) is dict else cls(**res)
        if isinstance(raw, (list, tuple)):
            return type(raw)(_init(v) for v in raw)
        return raw

    def _construct_outputs(raw: Any) -> Any:
        if isinstance(raw, Module):
            return outputs[raw]
        if isinstance(raw, Mapping):
            res = {k: _construct_outputs(v) for k, v in raw.items()}
            return res if (cls := type(raw)) is dict else cls(**res)
        if isinstance(raw, (list, tuple)):
            return type(raw)(_construct_outputs(v) for v in raw)
        return raw

    return _init(model), inputs, _construct_outputs(model)


def get_autocast(mixed_precision_type: str | None, device: str | None) -> Callable[[], AbstractContextManager[None]]:
    """Get the appropriate autocast context manager based on the device and mixed precision type."""
    if mixed_precision_type is None:
        return suppress
    return partial(autocast, device_type=get_torch_device(device), dtype=DTYPES[mixed_precision_type])


@dataclass(kw_only=True, slots=True)
class TrainingStep:
    """A training step for a PyTorch model."""

    models: list[str]

    losses: Module
    """A module that computes the losses for the model."""

    metrics: Module | None = None
    """A module that computes the metrics for the model."""

    autocast: Callable[[], AbstractContextManager[None]] = suppress
    """A context manager for automatic mixed precision (AMP). By default, it does nothing."""

    def __call__(self, inputs: dict[str, Any], **models: Module) -> dict[str, Tensor]:
        """Perform the forward pass for the given inputs and return the outputs and any additional information."""
        with self.autocast():
            outputs = inputs.copy()
            for name in self.models:
                outputs.update(models[name](**outputs))
            criteria = self.losses(**outputs)
            if self.metrics is None:
                return criteria
            with no_grad():
                criteria.update(self.metrics(**outputs))
            return criteria


@dataclass(kw_only=True, slots=True)
class ValidationStep(TrainingStep):
    """A validation step for a PyTorch model."""

    def __call__(self, inputs: dict[str, Any], **models: Module) -> dict[str, Tensor]:
        """Perform the forward pass for the given inputs and return the outputs and any additional information."""
        with no_grad():
            with self.autocast():
                outputs = inputs.copy()
                for name in self.models:
                    outputs.update(models[name](**outputs))
                criteria = self.losses(**outputs)
                if self.metrics is None:
                    return criteria
                criteria.update(self.metrics(**outputs))
                return criteria


@dataclass(kw_only=True, slots=True)
class TorchTracker:
    """A tracker for PyTorch models."""

    losses_tracker: CriteriaTracker
    """A tracker for the losses of the model."""

    metrics_tracker: CriteriaTracker | None = None
    """A tracker for the metrics of the model."""

    def __post_init__(self) -> None:
        """Post-initialization."""
        GLOBAL_CALLBACKS.on_training_begin.append(lambda _: self.losses_tracker.reset())  # type: ignore[arg-type]
        if self.metrics_tracker is not None:
            GLOBAL_CALLBACKS.on_training_begin.append(lambda _: self.metrics_tracker.reset())  # type: ignore[arg-type]

    def __call__(self, **criteria: Tensor) -> dict[str, float]:
        """Log the criteria and return the average values."""
        res: dict[str, Tensor] = self.losses_tracker({k: criteria[k] for k in self.losses_tracker.criteria})
        if self.metrics_tracker is not None:
            res.update(self.metrics_tracker({k: criteria[k] for k in self.metrics_tracker.criteria}))
        return {k: v.item() for k, v in res.items()}

    @classmethod
    def from_criteria(
        cls,
        loss_outputs: list[str],
        metric_outputs: list[str] | None = None,
        compile_fn: Callable[[Module], Module] | None = None,
    ) -> "TorchTracker":
        """Create a tracker from the given loss and metric modules.

        Args:
            loss_outputs (list[str]): The outputs to track for the loss module.
            metric_outputs (list[str] | None): The outputs to track for the metric module.
            compile_fn (Callable[[Module], Module] | None): An optional function to compile the loss and metric modules.

        Returns:
            A TorchTracker instance with the specified loss and metric trackers.
        """
        losses_tracker = CriteriaTracker(loss_outputs)
        metrics_tracker = None if metric_outputs is None else CriteriaTracker(metric_outputs)
        if compile_fn is not None:
            losses_tracker = compile_fn(losses_tracker)
            if metrics_tracker is not None:
                metrics_tracker = compile_fn(metrics_tracker)
        return cls(losses_tracker=losses_tracker, metrics_tracker=metrics_tracker)


@dataclass(kw_only=True, slots=True)
class TimmEmaUpdater:
    """A callback that updates the EMA model from the timm library."""

    name: str
    """The name of the model to update the EMA for."""

    ema: ModelEmaV3
    """The EMA model."""

    def __call__(self, info: BaseInfo, **models: Module) -> None:
        """Update the EMA model."""
        self.ema.update(models[self.name], step=info.update)


@dataclass(kw_only=True, slots=True)
class TimmEmaWrapper:
    """An inference wrapper that returns the EMA model from the timm library."""

    ema: dict[str, ModelEmaV3]
    """The EMA model."""

    callbacks: list[Callback[Module]] = field(default_factory=list)
    """The callbacks to invoke when the wrapper is called."""

    def __post_init__(self) -> None:
        """Post-initialization."""
        GLOBAL_CALLBACKS.on_update += [TimmEmaUpdater(name=n, ema=e) for n, e in self.ema.items()]

    def __call__(self, info: BaseInfo, **models: Module) -> dict[str, Any]:
        """Return the EMA model."""
        models = {
            n: o.module if n in self.ema and (o := self.ema[n]).device == m.device else m for n, m in models.items()
        }
        invoke_callback(self.callbacks, info, **models)
        return models

    @classmethod
    def from_models(
        cls,
        models: dict[str, Module],
        callbacks: list[Callback[Module]] | None = None,
        device: device | None = None,
        **kwargs: Any,
    ) -> "TimmEmaWrapper":
        """Create a TimmEmaWrapper from the given models.

        Args:
            models (dict[str, Module]): The models to create the EMA wrapper for.
            callbacks (list[Callback[Module]] | None): The callbacks to invoke when the wrapper is called.
                If None, no callbacks will be invoked.
            device (device | None): The device to move the EMA models to. If None, the EMA models will not be moved.
            **kwargs: Additional keyword arguments to pass to the ModelEmaV3 constructor.

        Returns:
            A TimmEmaWrapper instance with the specified EMA models and callbacks.
        """
        ema = {n: ModelEmaV3(m, device=device, **kwargs) for n, m in models.items()}
        return cls(ema=ema, callbacks=callbacks or [])


@dataclass(kw_only=True)
class TorchTrainer(BaseTrainer[Module]):
    """Trainer for PyTorch models."""


@dataclass(kw_only=True, slots=True)
class _LoaderWrapper:
    """A wrapper for the data loader that moves the data to the specified device and dtype."""

    loader: DataLoader
    """The data loader."""

    def __len__(self) -> int:
        """Return the number of batches in the data loader."""
        return len(self.loader)

    def __getattribute__(self, name) -> Any:
        loader = super(_LoaderWrapper, self).__getattribute__("loader")
        return loader if name == "loader" else loader.__getattribute__(name)


@dataclass(kw_only=True, slots=True)
class TimmLoaderWrapper(_LoaderWrapper):
    """A wrapper for the data loader that applies mixup and moves the data to the specified device and dtype."""

    mixup: Mixup
    """The mixup function."""

    device: str
    """The device to move the data to."""

    dtype: str
    """The data type to move the data to."""

    def __iter__(self) -> Iterable[dict[str, Any]]:
        """Iterate over the data loader and yield the data with the corresponding names."""
        device, dtype, mixup = get_torch_device(self.device), DTYPES[self.dtype], self.mixup
        for input, target in self.loader:
            input, target = input.to(device=device, dtype=dtype), target.to(device=device)
            yield mixup(input, target)


@dataclass(kw_only=True, slots=True)
class NamedData(_LoaderWrapper):
    """A wrapper for the data loader that yields the data with the corresponding names."""

    names: list[str]
    """The names of the data."""

    def __iter__(self) -> Iterable[dict[str, Any]]:
        """Iterate over the data loader and yield the data with the corresponding names."""
        for data in self.loader:
            yield dict(zip(self.names, data, strict=True))


__all__ = [
    "CriteriaTracker",
    "NamedData",
    "TimmEmaUpdater",
    "TimmEmaWrapper",
    "TimmLoaderWrapper",
    "TorchTracker",
    "TorchTrainer",
    "TrainingStep",
    "ValidationStep",
    "create_torch_inputs",
    "get_autocast",
    "get_torch_device",
    "initial_model",
]


if not TYPE_CHECKING:
    import sys

    from structcast.utils.lazy_import import LazySelectedImporter

    sys.modules[__name__] = LazySelectedImporter(__name__, globals())
