"""Multiply layer for PyTorch."""

from structcast.utils.security import get_default_dir
from torch.nn import Module

from structcast_model.torch.types import Tensor


class Multiply(Module):
    """Multiply layer."""

    def forward(self, inputs: list[Tensor]) -> Tensor:
        """Forward pass."""
        out = inputs[0]
        for inp in inputs[1:]:
            out = out.mul(inp)
        return out


__all__ = ["Multiply"]


def __dir__() -> list[str]:
    return get_default_dir(globals())
