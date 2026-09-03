"""Logger recording a training run to MLflow."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any

from structcast.utils.lazy_import import try_import

from structcast_model.base_trainer import BaseInfo
from structcast_model.loggers.base import Logger, _epoch_metrics, _local_training_state
from structcast_model.loggers.state_backends import StateBackend, TorchStateBackend

with try_import() as _imports:
    import mlflow


@dataclass(kw_only=True, slots=True)
class MLflowLogger(Logger):
    """Logger recording a run to MLflow.

    The logger owns the run: entering it starts the run, leaving it ends the run. It also reacts to
    the end of each epoch, so passing it to a trainer logs the epoch metrics.
    """

    experiment: str
    """The experiment the run is recorded under."""

    state_backend: StateBackend = field(default_factory=TorchStateBackend)
    """The format training states are written and read in; the torch one unless a run asks otherwise."""

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
        """Log a state dictionary as one artifact file, named after the backend's format.

        The artifact is `<name><suffix>`, not the directory `mlflow.pytorch` used to write: one
        format for both backends, and the fetch path below still accepts the old layout.
        """
        with TemporaryDirectory() as directory:
            self.log_artifact(str(self.state_backend.save(states, Path(directory), name)))

    def fetch_training_state(self, reference: str) -> dict[str, Any]:
        """Load a saved training state from an MLflow `runs:/` URI or a local path.

        Args:
            reference (str): The training state location: `runs:/<run_id>/<artifact>` or a local path.

        Returns:
            dict[str, Any]: The loaded training state.

        Raises:
            ValueError: If the reference is neither a `runs:/` URI nor an existing local path, or if
                a downloaded artifact directory holds no state file.
        """
        if reference.startswith("runs:/"):
            path = Path(mlflow.artifacts.download_artifacts(artifact_uri=reference))
            if path.is_dir():
                # A directory artifact is what `mlflow.pytorch.log_state_dict` used to write, and
                # what a backend saving a directory would write next: take this backend's format
                # first, then the torch-flavored `state_dict.pth` of the runs recorded before.
                states = sorted(path.glob(f"*{self.state_backend.suffix}"))
                if states:
                    return self.state_backend.load(states[0])
                legacy = sorted(path.glob("*.pth"))
                if not legacy:
                    raise ValueError(
                        f'No "*{self.state_backend.suffix}" or legacy "*.pth" training state found in the '
                        f'downloaded MLflow artifact "{path}".'
                    )
                # `*.pth` is a torch pickle whatever this logger's backend is, so it is read as one.
                return TorchStateBackend().load(legacy[0])
            return self.state_backend.load(path)
        return _local_training_state(reference, 'a "runs:/<run_id>/<artifact>" URI', self.state_backend)

    def on_epoch_end(self, info: BaseInfo) -> None:
        """Log the criteria and learning rates of the finished epoch."""
        self.log_metrics(_epoch_metrics(info), step=info.epoch)


__all__ = ["MLflowLogger"]


if not TYPE_CHECKING:
    import sys

    from structcast.utils.lazy_import import LazySelectedImporter

    sys.modules[__name__] = LazySelectedImporter(__name__, globals())
