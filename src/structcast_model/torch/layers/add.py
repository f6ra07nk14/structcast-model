"""Add layer for PyTorch."""

from typing import TYPE_CHECKING

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


__all__ = ["Add"]


if not TYPE_CHECKING:
    import sys

    from structcast.utils.lazy_import import LazySelectedImporter

    sys.modules[__name__] = LazySelectedImporter(__name__, globals())
