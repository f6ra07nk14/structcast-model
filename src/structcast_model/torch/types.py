"""Types for torch module."""

from typing import TYPE_CHECKING

from structcast.utils.security import get_default_dir

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

__all__ = ["DType", "DeviceLike", "Tensor"]


def __dir__() -> list[str]:
    return get_default_dir(globals())
