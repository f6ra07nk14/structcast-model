"""Scale Identity layer."""

from typing import Any

from structcast.utils.security import get_default_dir
from torch.nn import Module, Parameter

from structcast_model.torch.types import DeviceLike, DType, Tensor
from torch import tensor


class ScaleIdentity(Module):
    """Scale Identity layer."""

    scale: Tensor

    def __init__(
        self,
        scale: float,
        trainable: bool = False,
        device: DeviceLike | None = None,
        dtype: DType | None = None,
    ) -> None:
        """Initialize Scale Identity layer."""
        super().__init__()
        factory_kwargs: dict[str, Any] = {"device": device, "dtype": dtype}
        self.scale = Parameter(tensor(scale, **factory_kwargs), requires_grad=trainable)

    def forward(self, input: Tensor) -> Tensor:
        """Forward method."""
        return input.mul(self.scale)


__all__ = ["ScaleIdentity"]


def __dir__() -> list[str]:
    return get_default_dir(globals())
