"""Shared interface of the loggers recording a training run to an experiment tracking service."""

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Protocol, cast, runtime_checkable

from structcast_model.base_trainer import BaseInfo, BaseTrainer


@runtime_checkable
class Logger(Protocol):
    """Protocol for the object recording a training run to an experiment tracking service.

    A logger owns the run: entering it starts the run, leaving it ends the run. It is also a
    callback reacting to the end of each epoch, so passing it to a trainer logs the epoch metrics.
    """

    def __enter__(self) -> "Logger":
        """Start the run and return the logger recording it."""

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        """End the run, marking it failed when an exception is propagating."""

    def log_params(self, params: Mapping[str, Any]) -> None:
        """Log the run parameters."""

    def log_dict(self, data: Mapping[str, Any], name: str) -> None:
        """Log a dictionary as an artifact under the given file name."""

    def log_artifact(self, path: str) -> None:
        """Log a local file as an artifact."""

    def log_metric(self, name: str, value: float, step: int) -> None:
        """Log one metric value at the given step."""

    def log_metrics(self, metrics: Mapping[str, float], step: int) -> None:
        """Log several metric values at the given step."""

    def log_state_dict(self, states: Mapping[str, Any], name: str) -> None:
        """Log a state dictionary under the given artifact name."""

    def on_epoch_end(self, info: BaseInfo, **models: Any) -> None:
        """Log the criteria and learning rates of the finished epoch."""


def _epoch_metrics(info: BaseInfo) -> dict[str, Any]:
    """Merge the learning rates reported by the learner into the criteria of the current epoch.

    Schedules step in the learner's own on_epoch_end hooks, which the trainer dispatches before the
    logger's, so the recorded learning rate is the one the NEXT epoch will use -- the same one-epoch
    offset the pre-redesign global callbacks produced.
    """
    return {**cast("BaseTrainer[Any]", info).learner.learning_rates, **info.logs()}


# `_epoch_metrics` is listed because the LazySelectedImporter tail below only exposes the names in
# `__all__`, and the two logger backends import it from here.
__all__ = ["Logger", "_epoch_metrics"]


if not TYPE_CHECKING:
    import sys

    from structcast.utils.lazy_import import LazySelectedImporter

    sys.modules[__name__] = LazySelectedImporter(__name__, globals())
