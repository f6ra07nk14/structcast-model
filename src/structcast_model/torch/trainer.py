from collections.abc import Callable
from contextlib import AbstractContextManager, suppress
from dataclasses import dataclass

from torch.nn import Module

from structcast_model.base_trainer import Any, BaseTrainer
from structcast_model.torch.layers.criteria_tracker import CriteriaTracker
from structcast_model.torch.types import Tensor
from torch import no_grad


@dataclass(kw_only=True, slots=True)
class TorchForward:
    """A forward pass for a PyTorch model."""

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


@dataclass(kw_only=True)
class TorchTrainer(BaseTrainer[Module]):
    """Trainer for PyTorch models."""
