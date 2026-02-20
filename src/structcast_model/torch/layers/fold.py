"""Fold and Unfold Layers."""

from torch.nn import Fold, Unfold
from torch.nn.modules.lazy import LazyModuleMixin

from structcast_model.torch.layers.types import Tensor


def _compute_spatial_shape(
    spatial_shape: tuple[int, ...],
    kernel_size: int | tuple[int, ...],
    dilation: int | tuple[int, ...],
    padding: int | tuple[int, ...],
    stride: int | tuple[int, ...],
) -> tuple[int, ...]:
    """Compute the spatial shape."""
    shape_size = len(spatial_shape)
    return tuple(
        (v + 2 * p - (d * (k - 1) + 1)) // s + 1
        for v, k, d, p, s in zip(
            spatial_shape,
            (kernel_size,) * shape_size if isinstance(kernel_size, int) else kernel_size,
            (dilation,) * shape_size if isinstance(dilation, int) else dilation,
            (padding,) * shape_size if isinstance(padding, int) else padding,
            (stride,) * shape_size if isinstance(stride, int) else stride,
            strict=False,
        )
    )


def _compute_transposed_spatial_shape(
    spatial_shape: tuple[int, ...],
    kernel_size: int | tuple[int, ...],
    dilation: int | tuple[int, ...],
    padding: int | tuple[int, ...],
    stride: int | tuple[int, ...],
    output_padding: int | tuple[int, ...],
) -> tuple[int, ...]:
    """Compute the transposed spatial shape."""
    shape_size = len(spatial_shape)
    return tuple(
        (v - 1) * s - 2 * p + d * (k - 1) + o + 1
        for v, k, d, p, s, o in zip(
            spatial_shape,
            (kernel_size,) * shape_size if isinstance(kernel_size, int) else kernel_size,
            (dilation,) * shape_size if isinstance(dilation, int) else dilation,
            (padding,) * shape_size if isinstance(padding, int) else padding,
            (stride,) * shape_size if isinstance(stride, int) else stride,
            (output_padding,) * shape_size if isinstance(output_padding, int) else output_padding,
            strict=False,
        )
    )


class UnfoldExt(LazyModuleMixin, Unfold):
    """Extended Unfold Layer (Channel)."""

    cls_to_become = Unfold  # type: ignore[assignment]
    input_size: tuple[int, ...]
    output_size: tuple[int, ...]

    def __init__(
        self,
        kernel_size: int | tuple[int, ...] = 1,
        dilation: int | tuple[int, ...] = 1,
        padding: int | tuple[int, ...] = 0,
        stride: int | tuple[int, ...] = 1,
    ) -> None:
        """Initialize UnfoldExt layer."""
        super().__init__(kernel_size, dilation, padding, stride)
        self.input_size = (0,)
        self.output_size = (0,)

    def initialize_parameters(self, input: Tensor) -> None:
        """Initialize parameters based on input tensor shape."""
        if self.input_size[0] == 0:
            self.input_size = input.shape[2:]
            self.output_size = _compute_spatial_shape(
                self.input_size,
                kernel_size=self.kernel_size,
                dilation=self.dilation,
                padding=self.padding,
                stride=self.stride,
            )


class FoldExt(LazyModuleMixin, Fold):
    """Extended Fold Layer (Channel)."""

    cls_to_become = Fold  # type: ignore[assignment]
    input_size: tuple[int, ...]
    output_padding: int | tuple[int, ...]

    def __init__(
        self,
        kernel_size: int | tuple[int, ...] = 1,
        dilation: int | tuple[int, ...] = 1,
        padding: int | tuple[int, ...] = 0,
        stride: int | tuple[int, ...] = 1,
        output_padding: int | tuple[int, ...] = 0,
    ) -> None:
        """Initialize FoldExt layer."""
        super().__init__((0,), kernel_size, dilation, padding, stride)
        self.input_size = (0,)
        self.output_padding = output_padding

    def initialize_parameters(self, input: Tensor) -> None:
        """Initialize parameters based on input tensor shape."""
        if self.input_size[0] == 0:
            self.input_size = input.shape[2:]
            self.output_size = _compute_transposed_spatial_shape(
                self.input_size,
                kernel_size=self.kernel_size,
                dilation=self.dilation,
                padding=self.padding,
                stride=self.stride,
                output_padding=self.output_padding,
            )
