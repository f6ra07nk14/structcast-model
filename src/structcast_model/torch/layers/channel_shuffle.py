"""Channel shuffle layer."""

from typing import TYPE_CHECKING

from torch.nn import Module

from structcast_model.torch.types import Tensor


class ChannelLastShuffle(Module):
    """Divides and rearranges the last channels in a tensor."""

    __constants__ = ["groups"]
    groups: int

    def __init__(self, groups: int) -> None:
        """Initialize the layer."""
        super().__init__()
        self.groups = groups

    def forward(self, input: Tensor) -> Tensor:  # pylint: disable=redefined-builtin
        """Rearrange the last channels in a tensor."""
        prefix = input.shape[:-1]
        dim = input.shape[-1]
        return input.view(prefix + (self.groups, dim // self.groups)).mT.reshape(prefix + (dim,))

    def extra_repr(self) -> str:
        """Extra representation of the layer."""
        return f"groups={self.groups}"


__all__ = ["ChannelLastShuffle"]


if not TYPE_CHECKING:
    import sys

    from structcast.utils.lazy_import import LazySelectedImporter

    sys.modules[__name__] = LazySelectedImporter(__name__, globals())
