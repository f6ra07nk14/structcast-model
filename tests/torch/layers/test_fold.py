"""Tests for fold and unfold extensions."""

from structcast_model.torch.layers.fold import (
    FoldExt,
    UnfoldExt,
    compute_spatial_shape,
    compute_transposed_spatial_shape,
)
import torch


def test_compute_spatial_shape_with_int_parameters() -> None:
    """Compute output spatial shape with scalar convolution parameters."""
    assert compute_spatial_shape((8, 8), kernel_size=3, dilation=1, padding=1, stride=2) == (4, 4)


def test_compute_transposed_spatial_shape_with_tuple_parameters() -> None:
    """Compute transposed output spatial shape with tuple parameters."""
    assert compute_transposed_spatial_shape(
        (4, 4),
        kernel_size=(3, 3),
        dilation=(1, 1),
        padding=(1, 1),
        stride=(2, 2),
        output_padding=(1, 1),
    ) == (8, 8)


def test_unfold_ext_initialize_parameters_only_once() -> None:
    """Initialize UnfoldExt input and output shape lazily one time."""
    layer = UnfoldExt(kernel_size=2, stride=2)
    layer.initialize_parameters(torch.randn(1, 3, 8, 8))
    first_state = (layer.input_size, layer.output_size)
    layer.initialize_parameters(torch.randn(1, 3, 10, 10))
    assert layer.input_size == (8, 8)
    assert layer.output_size == (4, 4)
    assert (layer.input_size, layer.output_size) == first_state


def test_fold_ext_initialize_parameters_only_once() -> None:
    """Initialize FoldExt input and output shape lazily one time."""
    layer = FoldExt(kernel_size=2, stride=2, output_padding=0)
    layer.initialize_parameters(torch.randn(1, 3, 4, 4))
    first_state = (layer.input_size, layer.output_size)
    layer.initialize_parameters(torch.randn(1, 3, 6, 6))
    assert layer.input_size == (4, 4)
    assert layer.output_size == (8, 8)
    assert (layer.input_size, layer.output_size) == first_state
