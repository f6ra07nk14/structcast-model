"""Logger recording a training run to MLflow."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from structcast.utils.lazy_import import try_import

from structcast_model.base_trainer import BaseInfo
from structcast_model.torch.logger import Logger, _epoch_metrics

with try_import() as _imports:
    import mlflow
    import mlflow.pytorch


@dataclass(kw_only=True, slots=True)
class MLflowLogger(Logger):
    """Logger recording a run to MLflow.

    The logger owns the run: entering it starts the run, leaving it ends the run. It also reacts to
    the end of each epoch, so passing it to a trainer logs the epoch metrics.
    """

    experiment: str
    """The experiment the run is recorded under."""

    def __post_init__(self) -> None:
        """Fail with an explanatory error when mlflow is not installed."""
        _imports.check()

    def __enter__(self) -> "MLflowLogger":
        """Start a run in the configured experiment."""
        mlflow.set_experiment(self.experiment)
        mlflow.start_run()
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        """End the run, marking it failed when an exception is propagating."""
        mlflow.end_run(status="FINISHED" if exc_type is None else "FAILED")

    def log_params(self, params: Mapping[str, Any]) -> None:
        """Log the run parameters."""
        for key, value in params.items():
            mlflow.log_param(key, value)

    def log_dict(self, data: Mapping[str, Any], name: str) -> None:
        """Log a dictionary as an artifact under the given file name."""
        mlflow.log_dict(dict(data), name)

    def log_artifact(self, path: str) -> None:
        """Log a local file as an artifact."""
        mlflow.log_artifact(path)

    def log_metric(self, name: str, value: float, step: int) -> None:
        """Log one metric value at the given step."""
        mlflow.log_metric(name, value, step=step)

    def log_metrics(self, metrics: Mapping[str, float], step: int) -> None:
        """Log several metric values at the given step."""
        mlflow.log_metrics(dict(metrics), step=step)

    def log_state_dict(self, states: Mapping[str, Any], name: str) -> None:
        """Log a state dictionary under the given artifact name."""
        mlflow.pytorch.log_state_dict(dict(states), name)

    def on_epoch_end(self, info: BaseInfo, **models: Any) -> None:
        """Log the criteria and learning rates of the finished epoch."""
        self.log_metrics(_epoch_metrics(info), step=info.epoch)


__all__ = ["MLflowLogger"]


if not TYPE_CHECKING:
    import sys

    from structcast.utils.lazy_import import LazySelectedImporter

    sys.modules[__name__] = LazySelectedImporter(__name__, globals())
