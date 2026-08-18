"""Torch Extensions for StructCast-Model."""

from typing import TYPE_CHECKING

__all__ = ["distributed", "layers", "optimizers", "trainer", "types", "utils"]

if TYPE_CHECKING:
    from structcast_model.torch import distributed, layers, optimizers, trainer, types, utils
else:
    import sys

    from structcast.utils.lazy_import import LazySelectedImporter

    import_structure = {
        "distributed": [],
        "layers": [],
        "optimizers": [],
        "trainer": [],
        "types": [],
        "utils": [],
    }
    sys.modules[__name__] = LazySelectedImporter(__name__, globals(), import_structure)
