"""Logger recording a training run to Weights & Biases."""

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from structcast.utils.base import dump_yaml
from structcast.utils.lazy_import import try_import

from structcast_model.base_trainer import BaseInfo
from structcast_model.torch.logger import Logger, _epoch_metrics
import torch

with try_import() as _imports:
    import wandb


@dataclass(kw_only=True, slots=True)
class WandbLogger(Logger):
    """Logger recording a run to Weights & Biases, with the same interface as `MLflowLogger`."""

    experiment: str
    """The project the run is recorded under."""

    def __post_init__(self) -> None:
        """Fail with an explanatory error when wandb is not installed."""
        _imports.check()

    def __enter__(self) -> "WandbLogger":
        """Start a run in the project named after the experiment."""
        wandb.init(project=self.experiment)
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        """Finish the run, marking it failed when an exception is propagating."""
        wandb.finish(exit_code=0 if exc_type is None else 1)

    def log_params(self, params: Mapping[str, Any]) -> None:
        """Log the run parameters."""
        wandb.config.update(dict(params))

    def log_dict(self, data: Mapping[str, Any], name: str) -> None:
        """Write a dictionary into the run directory as YAML, matching what MLflow stores."""
        dump_yaml(dict(data), Path(wandb.run.dir) / name)

    def log_artifact(self, path: str) -> None:
        """Log a local file as an artifact."""
        wandb.save(path)

    def log_metric(self, name: str, value: float, step: int) -> None:
        """Log one metric value at the given step."""
        wandb.log({name: value}, step=step)

    def log_metrics(self, metrics: Mapping[str, float], step: int) -> None:
        """Log several metric values at the given step."""
        wandb.log(dict(metrics), step=step)

    def log_state_dict(self, states: Mapping[str, Any], name: str) -> None:
        """Save a state dictionary into the run directory."""
        torch.save(dict(states), Path(wandb.run.dir) / f"{name}.pt")

    def on_epoch_end(self, info: BaseInfo, **models: Any) -> None:
        """Log the criteria and learning rates of the finished epoch."""
        self.log_metrics(_epoch_metrics(info), step=info.epoch)


__all__ = ["WandbLogger"]


if not TYPE_CHECKING:
    import sys

    from structcast.utils.lazy_import import LazySelectedImporter

    sys.modules[__name__] = LazySelectedImporter(__name__, globals())
