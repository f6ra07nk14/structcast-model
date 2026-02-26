"""Lazy normalization layer."""

from typing import TYPE_CHECKING, Any

from torch.nn import LayerNorm, RMSNorm, UninitializedParameter
from torch.nn.modules.lazy import LazyModuleMixin

from structcast_model.torch.types import DeviceLike, DType, Tensor
from torch import no_grad


class LazyRMSNorm(LazyModuleMixin, RMSNorm):
    """Lazy version of :class:`torch.nn.RMSNorm`.

    Attributes:
        channels: Number of channels.
        eps: A value added to the denominator for numerical stability. Default: 1e-5
        elementwise_affine: Whether to learn a separate scale and bias for each channel
    """

    cls_to_become = RMSNorm  # type: ignore[assignment]
    weight: UninitializedParameter  # type: ignore[assignment]
    channels: int = 1

    def __init__(
        self,
        channels: int = 1,
        eps: float | None = None,
        elementwise_affine: bool = True,
        device: DeviceLike | None = None,
        dtype: DType | None = None,
    ) -> None:
        """Initialize LazyRMSNorm layer."""
        factory_kwargs: dict[str, Any] = {"device": device, "dtype": dtype}
        super().__init__(0, eps, False)
        self.channels = channels
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.weight = UninitializedParameter(**factory_kwargs)

    def reset_parameters(self) -> None:
        """Reset parameters."""
        if not self.has_uninitialized_params() and self.normalized_shape[-1] != 0:
            super().reset_parameters()

    def initialize_parameters(self, input: Tensor) -> None:
        """Initialize parameters based on input tensor shape."""
        if self.has_uninitialized_params():
            with no_grad():
                self.normalized_shape = input.shape[-self.channels :]
                if self.elementwise_affine:
                    self.weight.materialize(self.normalized_shape)
                self.reset_parameters()


class LazyLayerNorm(LazyModuleMixin, LayerNorm):
    """Lazy version of :class:`torch.nn.LayerNorm`."""

    cls_to_become = LayerNorm  # type: ignore[assignment]
    weight: UninitializedParameter  # type: ignore[assignment]
    bias: UninitializedParameter  # type: ignore[assignment]
    channels: int = 1

    def __init__(
        self,
        channels: int = 1,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        bias: bool = True,
        device: DeviceLike | None = None,
        dtype: DType | None = None,
    ) -> None:
        """Initialize LazyLayerNorm layer."""
        factory_kwargs: dict[str, Any] = {"device": device, "dtype": dtype}
        super().__init__(0, eps, False, False)
        self.channels = channels
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.weight = UninitializedParameter(**factory_kwargs)
            if bias:
                self.bias = UninitializedParameter(**factory_kwargs)

    def reset_parameters(self) -> None:
        """Reset parameters."""
        if not self.has_uninitialized_params() and self.normalized_shape[-1] != 0:
            super().reset_parameters()

    def initialize_parameters(self, input: Tensor) -> None:
        """Initialize parameters based on input tensor shape."""
        if self.has_uninitialized_params():
            with no_grad():
                self.normalized_shape = input.shape[-self.channels :]
                if self.elementwise_affine:
                    self.weight.materialize(self.normalized_shape)
                    if self.bias is not None:
                        self.bias.materialize(self.normalized_shape)
                self.reset_parameters()


__all__ = ["LazyLayerNorm", "LazyRMSNorm"]


if not TYPE_CHECKING:
    import sys

    from structcast.utils.lazy_import import LazySelectedImporter

    sys.modules[__name__] = LazySelectedImporter(__name__, globals())
