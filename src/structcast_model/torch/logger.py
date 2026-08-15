"""Shared interface of the loggers recording a training run to an experiment tracking service."""

from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from typing_extensions import Protocol, runtime_checkable

from structcast_model.base_trainer import BaseInfo, BaseTrainer
import torch


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

    def fetch_training_state(self, reference: str) -> dict[str, Any] | None:
        """Fetch the training state the reference points to; None when the logger records nothing."""

    def on_epoch_end(self, info: BaseInfo, **models: Any) -> None:
        """Log the criteria and learning rates of the finished epoch."""


class NullLogger(Logger):
    """Logger that records nothing.

    The write-side null object for ranks that must run the collective checkpoint production but
    own no experiment-tracking run (every rank except rank 0): callbacks call it unconditionally
    and only the ranks holding a real logger persist anything. Test fakes subclass it and override
    the one method they observe, satisfying the full protocol without ceremony.
    """

    def __enter__(self) -> "Logger":
        """Start nothing and hand back the logger, mirroring a real run's context shape."""
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        """End nothing."""

    def log_params(self, params: Mapping[str, Any]) -> None:
        """Discard the run parameters."""

    def log_dict(self, data: Mapping[str, Any], name: str) -> None:
        """Discard the dictionary."""

    def log_artifact(self, path: str) -> None:
        """Discard the artifact."""

    def log_metric(self, name: str, value: float, step: int) -> None:
        """Discard the metric value."""

    def log_metrics(self, metrics: Mapping[str, float], step: int) -> None:
        """Discard the metric values."""

    def log_state_dict(self, states: Mapping[str, Any], name: str) -> None:
        """Discard the state dictionary."""

    def fetch_training_state(self, reference: str) -> None:
        """Fetch nothing: the non-main ranks receive the state through the strategy broadcast."""

    def on_epoch_end(self, info: BaseInfo, **models: Any) -> None:
        """React to nothing."""


def _epoch_metrics(info: BaseInfo) -> dict[str, Any]:
    """Merge the learner's learning rates and decay values into the criteria of the current epoch.

    Schedules step in the learner's own on_epoch_end hooks, which the trainer dispatches before the
    logger's, so the recorded learning rate is the one the NEXT epoch will use -- the same one-epoch
    offset the pre-redesign global callbacks produced. ``weight_decays`` is an optional learner
    member (generated learners flatten it from `create_opt`'s parameter groups), so weight and
    layer decay dynamics land in the same run.
    """
    learner = cast("BaseTrainer[Any]", info).learner
    return {**learner.learning_rates, **getattr(learner, "weight_decays", {}), **info.logs()}


def _local_training_state(reference: str, expected_form: str) -> dict[str, Any]:
    """Load a training state from an existing local path, naming the logger's accepted forms otherwise."""
    path = Path(reference)
    if not path.exists():
        raise ValueError(
            f'Cannot fetch a training state from "{reference}": expected {expected_form} or an existing local path.'
        )
    # `weights_only` because the reference is user input, and an unpickled checkpoint executes code.
    return torch.load(path, map_location="cpu", weights_only=True)


# `_epoch_metrics` and `_local_training_state` are listed because the LazySelectedImporter tail below
# only exposes the names in `__all__`, and the two logger backends import them from here.
__all__ = ["Logger", "NullLogger", "_epoch_metrics", "_local_training_state"]


if not TYPE_CHECKING:
    import sys

    from structcast.utils.lazy_import import LazySelectedImporter

    sys.modules[__name__] = LazySelectedImporter(__name__, globals())
