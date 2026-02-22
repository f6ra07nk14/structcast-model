"""Concatenate layer for PyTorch."""

from structcast.utils.security import get_default_dir
from torch.nn import Module

from structcast_model.torch.types import Tensor
from torch import cat


class Concatenate(Module):
    """Concatenate module for PyTorch."""

    __constants__ = ["dim"]
    dim: int

    def __init__(self, dim: int = -1) -> None:
        """Initialize the layer.

        Args:
            dim (int, optional): The dimension to concatenate. Defaults to 1.
        """
        super().__init__()
        self.dim = dim

    def forward(self, inputs: list[Tensor]) -> Tensor:
        """Forward pass.

        Args:
            inputs (list[Tensor]): The input tensors.

        Returns:
            Tensor: The concatenated tensor.
        """
        return cat(inputs, dim=self.dim)

    def extra_repr(self) -> str:
        """Extra representation of the layer.

        Returns:
            str: The extra representation.
        """
        return f"dim={self.dim}"


Concat = Concatenate
"""Alias for Concatenate."""

__all__ = ["Concat", "Concatenate"]


def __dir__() -> list[str]:
    return get_default_dir(globals())
