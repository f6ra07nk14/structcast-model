"""Trainer for PyTorch models."""

from collections.abc import Callable
from contextlib import AbstractContextManager, suppress
from dataclasses import dataclass
from logging import getLogger

from pydantic import TypeAdapter, ValidationError
from timm.utils import ModelEmaV3
from torch.nn import Module

from structcast_model.base_trainer import Any, BaseTrainer
from structcast_model.torch.layers.criteria_tracker import CriteriaTracker
from structcast_model.torch.types import Tensor
from torch import cuda, float32, no_grad, rand

logger = getLogger(__name__)


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

    losses: Module
    """A module that computes the losses for the model."""

    metrics: Module | None = None
    """A module that computes the metrics for the model."""

    autocast: Callable[[], AbstractContextManager[None]] = suppress

    def __call__(self, inputs: dict[str, Any], model: Module) -> dict[str, Tensor]:
        """Perform the forward pass for the given inputs and return the outputs and any additional information."""
        with self.autocast():
            outputs = model(**inputs)
            losses = self.losses(**inputs, **outputs)
            if self.metrics is None:
                return losses
            with no_grad():
                metrics = self.metrics(**inputs, **outputs)
            return {**losses, **metrics}


@dataclass(kw_only=True, slots=True)
class ValidationStep(TrainingStep):
    """A validation step for a PyTorch model."""

    def __call__(self, inputs: dict[str, Any], model: Module) -> dict[str, Tensor]:
        """Perform the forward pass for the given inputs and return the outputs and any additional information."""
        with no_grad():
            with self.autocast():
                outputs = model(**inputs)
                losses = self.losses(**inputs, **outputs)
                if self.metrics is None:
                    return losses
                return {**losses, **self.metrics(**inputs, **outputs)}


@dataclass(kw_only=True, slots=True)
class TorchLogger:
    """A logger for PyTorch models."""

    losses_tracker: CriteriaTracker
    """A tracker for the losses of the model."""

    metrics_tracker: CriteriaTracker | None = None
    """A tracker for the metrics of the model."""

    def __call__(self, **criteria: Tensor) -> dict[str, Tensor]:
        """Log the criteria and return the average values."""
        res: dict[str, Tensor] = self.losses_tracker({k: criteria[k] for k in self.losses_tracker.criteria})
        if self.metrics_tracker is not None:
            res.update(self.metrics_tracker({k: criteria[k] for k in self.metrics_tracker.criteria}))
        return {k: v.item() for k, v in res.items()}


@dataclass(kw_only=True, slots=True)
class TimmEmaWrapper:
    """A wrapper for the EMA model from the timm library."""

    ema: dict[str, ModelEmaV3]
    """The EMA model."""

    def __call__(self, **models: Module) -> dict[str, Any]:
        """Return the EMA model."""
        return {k: self.ema[k].module for k in models}


@dataclass(kw_only=True)
class TorchTrainer(BaseTrainer[Module]):
    """Trainer for PyTorch models."""
