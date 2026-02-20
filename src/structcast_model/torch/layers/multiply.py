"""Multiply layer for PyTorch."""

from torch.nn import Module

from structcast_model.torch.layers.types import Tensor


class Multiply(Module):
    """Multiply layer."""

    def forward(self, inputs: list[Tensor]) -> Tensor:
        """Forward pass."""
        out = inputs[0]
        for inp in inputs[1:]:
            out = out.mul(inp)
        return out
