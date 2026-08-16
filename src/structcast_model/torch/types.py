"""Types for torch module."""

from typing import TYPE_CHECKING

# Protocol and runtime_checkable come from typing_extensions so that isinstance checks use
# inspect.getattr_static on Python 3.11 as well (backported from 3.12), as in base_trainer.
from typing_extensions import Protocol, runtime_checkable

if TYPE_CHECKING:
    from typing import TypeAlias

    from torch import Tensor as _Tensor, device as _device, dtype as _dtype

    DeviceLike: TypeAlias = _device | str
    """Device like type."""

    DType = _dtype
    """Data type."""

    Tensor = _Tensor
    """Tensor type."""
else:
    from typing import Any

    DeviceLike = Any
    """Device like type."""

    DType = Any
    """Data type."""

    Tensor = Any
    """Tensor type."""


@runtime_checkable
class TensorInitializer(Protocol):
    """Callable creating a dummy tensor of the given size and element type, e.g. `torch.rand`.

    Note:
        Being runtime-checkable, `isinstance` only verifies that `__call__` exists;
        a mismatched signature is only detected when the initializer is called.
    """

    # The aliases above are assigned in both branches of the `TYPE_CHECKING` split, so within this module mypy
    # sees them as variables rather than types. The annotations are quoted because the underlying torch names
    # are only imported while type checking.
    def __call__(self, size: tuple[int, ...], *, dtype: "_dtype") -> "_Tensor":
        """Create a tensor of the given size and element type."""
        ...


__all__ = ["DType", "DeviceLike", "Tensor", "TensorInitializer"]


if not TYPE_CHECKING:
    import sys

    from structcast.utils.lazy_import import LazySelectedImporter

    sys.modules[__name__] = LazySelectedImporter(__name__, globals())
