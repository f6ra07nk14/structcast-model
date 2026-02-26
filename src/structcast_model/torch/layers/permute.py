"""Permute layer for PyTorch."""

from typing import TYPE_CHECKING

from torch.nn import Module
from torch.nn.modules.lazy import LazyModuleMixin

from structcast_model.torch.types import Tensor


class Permute(Module):
    """Split module for PyTorch."""

    __constants__ = ["dims"]
    dims: tuple[int, ...]

    def __init__(self, dims: tuple[int, ...]) -> None:
        """Initialize the layer.

        Args:
            dims (Tuple[int]): The dimensions to permute.
        """
        super().__init__()
        self.dims = dims if isinstance(dims, tuple) else tuple(dims)

    def forward(self, input: Tensor) -> Tensor:
        """Forward pass.

        Args:
            input (Tensor): The input tensor.

        Returns:
            Tensor: The permuted tensor.
        """
        return input.permute((0,) + self.dims)

    def extra_repr(self) -> str:
        """Extra representation of the layer.

        Returns:
            str: The extra representation.
        """
        return f"dims={self.dims}"


class ToChannelLast(LazyModuleMixin, Permute):
    """Lazy version of :class:`Permute`."""

    cls_to_become = Permute  # type: ignore[assignment]

    def __init__(self) -> None:
        """Initialize ToChannelLast layer."""
        super().__init__((0,))

    def initialize_parameters(self, input: Tensor) -> None:
        """Initialize parameters based on input tensor shape."""
        if self.dims[0] == 0:
            self.dims = (*range(2, len(input.shape[1:-1]) + 2), 1)


class ToChannelFirst(LazyModuleMixin, Permute):
    """Lazy version of :class:`Permute`."""

    cls_to_become = Permute  # type: ignore[assignment]

    def __init__(self) -> None:
        """Initialize ToChannelFirst layer."""
        super().__init__((0,))

    def initialize_parameters(self, input: Tensor) -> None:
        """Initialize parameters based on input tensor shape."""
        if self.dims[0] == 0:
            size = len(input.shape[1:-1]) + 1
            self.dims = (size, *range(1, size))


__all__ = ["Permute", "ToChannelFirst", "ToChannelLast"]


if not TYPE_CHECKING:
    import sys

    from structcast.utils.lazy_import import LazySelectedImporter

    sys.modules[__name__] = LazySelectedImporter(__name__, globals())
