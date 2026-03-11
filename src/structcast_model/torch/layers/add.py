"""Add layer for PyTorch."""

from torch.nn import Module

from structcast_model.torch.types import Tensor


class Add(Module):
    """Add layer."""

    def forward(self, inputs: list[Tensor]) -> Tensor:
        """Forward pass."""
        out = inputs[0]
        for inp in inputs[1:]:
            out = out.add(inp)
        return out
