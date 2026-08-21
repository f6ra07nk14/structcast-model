"""Loggers recording a training run to an experiment tracking service."""

from typing import TYPE_CHECKING

__all__ = [
    "FlaxStateBackend",
    "Logger",
    "MLflowLogger",
    "NullLogger",
    "StateBackend",
    "TorchStateBackend",
    "WandbLogger",
]

if TYPE_CHECKING:
    from structcast_model.loggers.base import Logger, NullLogger
    from structcast_model.loggers.mlflow import MLflowLogger
    from structcast_model.loggers.state_backends import FlaxStateBackend, StateBackend, TorchStateBackend
    from structcast_model.loggers.wandb import WandbLogger
else:
    import sys

    from structcast.utils.lazy_import import LazySelectedImporter

    # Public symbols only: base's private helpers stay reachable by importing the module itself.
    import_structure = {
        "base": ["Logger", "NullLogger"],
        "mlflow": ["MLflowLogger"],
        "state_backends": ["FlaxStateBackend", "StateBackend", "TorchStateBackend"],
        "wandb": ["WandbLogger"],
    }
    sys.modules[__name__] = LazySelectedImporter(__name__, globals(), import_structure)
