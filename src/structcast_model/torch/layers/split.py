"""Split layer for PyTorch."""

from torch.nn import Module

from structcast_model.torch.types import Tensor


class Split(Module):
    """Split module for PyTorch."""

    __constants__ = ["split_size_or_sections", "dim"]
    split_size_or_sections: int | list[int]
    dim: int

    def __init__(self, split_size_or_sections: int | list[int], dim: int = -1) -> None:
        """Initialize the layer.

        Args:
            split_size_or_sections (Union[int, List[int]]): The size or sections to split.
            dim (int, optional): The dimension to split. Defaults to 0.
        """
        super().__init__()
        self.split_size_or_sections = split_size_or_sections
        self.dim = dim

    def forward(self, input: Tensor) -> tuple[Tensor, ...]:  # pylint: disable=redefined-builtin
        """Forward pass.

        Args:
            input (Tensor): The input tensor.

        Returns:
            List[Tensor]: The list of tensors.
        """
        return input.split(self.split_size_or_sections, dim=self.dim)

    def extra_repr(self) -> str:
        """Extra representation of the layer.

        Returns:
            str: The extra representation.
        """
        return f"split_size_or_sections={self.split_size_or_sections}, dim={self.dim}"


__all__ = ["Split"]
