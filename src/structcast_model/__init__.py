"""StructCast-Model: Construct neural network models and training workflows by structcast package."""

from typing import TYPE_CHECKING

__version__ = "0.0.0"
__all__ = ["builders", "torch", "utils"]

if TYPE_CHECKING:
    from structcast_model import builders, torch, utils
else:
    import sys

    from structcast.utils.security import get_default_dir

    from structcast_model.utils.lazy_import import LazySelectedImporter

    sys.modules[__name__] = LazySelectedImporter(
        name=__name__,
        module_file=globals()["__file__"],
        import_structure={
            "builders": [],
            "torch": [],
            "utils": [],
        },
        extra={k: globals().get(k) for k in get_default_dir(globals())},
    )
