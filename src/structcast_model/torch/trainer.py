"""Trainer for PyTorch models."""

from collections.abc import Callable, Iterable
from contextlib import AbstractContextManager, suppress
from dataclasses import dataclass
from logging import getLogger
from typing import TYPE_CHECKING, Any

from pydantic import TypeAdapter, ValidationError
from timm.data import Mixup
from timm.utils import ModelEmaV3
from torch.nn import Module
from torch.utils.data import DataLoader, Dataset, Sampler

from structcast_model.base_trainer import GLOBAL_CALLBACKS, BaseInfo, BaseTrainer
from structcast_model.torch.layers.criteria_tracker import CriteriaTracker
from structcast_model.torch.types import Tensor
from torch import bfloat16, cuda, float16, float32, no_grad, rand

logger = getLogger(__name__)

DTYPES = {
    "float32": float32,
    "float16": float16,
    "bfloat16": bfloat16,
}


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


@dataclass(kw_only=True, slots=True)
class TrainingStep:
    """A training step for a PyTorch model."""

    models: list[str]

    losses: Module
    """A module that computes the losses for the model."""

    metrics: Module | None = None
    """A module that computes the metrics for the model."""

    autocast: Callable[[], AbstractContextManager[None]] = suppress

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
class TorchLogger:
    """A logger for PyTorch models."""

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
    """A wrapper for the EMA model from the timm library."""

    ema_on_cpu: bool
    """Whether the EMA model is on CPU or not."""

    ema: dict[str, ModelEmaV3]
    """The EMA model."""

    def __post_init__(self) -> None:
        """Post-initialization."""
        GLOBAL_CALLBACKS.on_update += [TimmEmaUpdater(name=n, ema=e) for n, e in self.ema.items()]

    def __call__(self, **models: Module) -> dict[str, Any]:
        """Return the EMA model."""
        return models if self.ema_on_cpu else {k: self.ema[k].module for k in models}


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

    @property
    def sampler(self) -> Sampler | Iterable:
        """Return the sampler of the data loader."""
        return self.loader.sampler

    @property
    def dataset(self) -> Dataset:
        """Return the dataset of the data loader."""
        return self.loader.dataset


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
    "NamedData",
    "TimmEmaUpdater",
    "TimmEmaWrapper",
    "TimmLoaderWrapper",
    "TorchLogger",
    "TorchTrainer",
    "TrainingStep",
    "ValidationStep",
    "create_torch_inputs",
    "get_torch_device",
]


if not TYPE_CHECKING:
    import sys

    from structcast.utils.lazy_import import LazySelectedImporter

    sys.modules[__name__] = LazySelectedImporter(__name__, globals())
