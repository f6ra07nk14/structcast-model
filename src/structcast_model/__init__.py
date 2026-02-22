"""StructCast-Model: Construct neural network models and training workflows by structcast package."""

from typing import TYPE_CHECKING

__version__ = "0.0.0"

if TYPE_CHECKING:
    from structcast_model import builders, torch, utils

    __all__ = ["builders", "torch", "utils"]
else:
    import sys

    from structcast_model.utils.lazy_import import LazySelectedImporter

    _import_structure = {}
    _skip_modules = list(_import_structure)
    _import_structure["builders"] = []
    _import_structure["torch"] = []
    _import_structure["utils"] = []
    sys.modules[__name__] = LazySelectedImporter(
        __name__,
        globals()["__file__"],
        _import_structure,
        skip_modules=_skip_modules,
        extra_objects={"__version__": __version__},
    )
