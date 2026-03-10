"""Reduce layers for PyTorch backend."""

from torch.nn import Module

from structcast_model.torch.types import Tensor


class ReduceSum(Module):
    """Reduce sum layer."""

    dim: int | tuple[int, ...] | None
    keepdim: bool

    def __init__(self, dim: int | tuple[int, ...] | None = None, keepdim: bool = False) -> None:
        """Initialize."""
        super().__init__()
        self.dim = dim
        self.keepdim = keepdim

    def forward(self, input: Tensor) -> Tensor:
        """Forward pass."""
        if self.dim is None:
            return input.sum()
        return input.sum(self.dim, keepdim=self.keepdim)
