"""Tests for lazy normalization layers."""

from structcast_model.torch.layers.lazy_norm import LazyLayerNorm, LazyRMSNorm
import torch


def test_lazy_rms_norm_initializes_parameters_with_affine() -> None:
    """Materialize weight and normalized shape from input tensor."""
    layer = LazyRMSNorm(channels=2, elementwise_affine=True)
    layer.initialize_parameters(torch.randn(2, 3, 4, 5))
    assert layer.normalized_shape == (4, 5)
    assert layer.weight.shape == (4, 5)


def test_lazy_rms_norm_without_affine_keeps_no_weight() -> None:
    """Skip parameter materialization when affine is disabled."""
    layer = LazyRMSNorm(channels=1, elementwise_affine=False)
    layer.initialize_parameters(torch.randn(2, 3, 4))
    assert layer.normalized_shape == (0,)
    assert layer.weight is None


def test_lazy_layer_norm_initializes_weight_and_bias() -> None:
    """Materialize layer norm parameters with affine and bias."""
    layer = LazyLayerNorm(channels=1, elementwise_affine=True, bias=True)
    layer.initialize_parameters(torch.randn(2, 3, 4))
    assert layer.normalized_shape == (4,)
    assert layer.weight.shape == (4,)
    assert layer.bias is not None
    assert layer.bias.shape == (4,)


def test_lazy_layer_norm_without_bias() -> None:
    """Keep bias disabled while still materializing weight."""
    layer = LazyLayerNorm(channels=2, elementwise_affine=True, bias=False)
    layer.initialize_parameters(torch.randn(2, 3, 4, 5))
    assert layer.normalized_shape == (4, 5)
    assert layer.weight.shape == (4, 5)
    assert layer.bias is None
