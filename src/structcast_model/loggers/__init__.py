"""Loggers recording a training run to an experiment tracking service."""

from typing import TYPE_CHECKING

__all__ = ["base", "mlflow", "wandb"]

if TYPE_CHECKING:
    from structcast_model.loggers import base, mlflow, wandb
else:
    import sys

    from structcast.utils.lazy_import import LazySelectedImporter

    import_structure = {
        "base": [],
        "mlflow": [],
        "wandb": [],
    }
    sys.modules[__name__] = LazySelectedImporter(__name__, globals(), import_structure)
