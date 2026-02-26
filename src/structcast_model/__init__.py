"""StructCast-Model: Construct neural network models and training workflows by structcast package."""

from typing import TYPE_CHECKING

__version__ = "0.0.0"
__all__ = ["builders", "torch", "utils"]

if TYPE_CHECKING:
    from structcast_model import builders, torch, utils
else:
    import sys

    from structcast.utils.lazy_import import LazySelectedImporter

    import_structure = {
        "builders": [],
        "torch": [],
        "utils": [],
    }
    sys.modules[__name__] = LazySelectedImporter(__name__, globals(), import_structure)
