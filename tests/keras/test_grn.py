"""Unit tests for structcast_model.keras.layers.grn using timm as reference."""

from __future__ import annotations

import numpy as np
import pytest
from timm.layers.grn import GlobalResponseNorm as TimmGRN

import keras
from structcast_model.keras.layers.grn import GlobalResponseNormalization
import torch


def _run_timm_grn(x_np: np.ndarray, *, dim: int, eps: float = 1e-6) -> np.ndarray:
    """Run timm GRN with scale=1, bias=0 and return NumPy result."""
    grn = TimmGRN(dim=dim, eps=eps, channels_last=True)
    with torch.no_grad():
        grn.weight.fill_(1.0)
        grn.bias.fill_(0.0)
        out = grn(torch.from_numpy(x_np).float())
    return out.numpy()


def _run_keras_grn(x_np: np.ndarray, *, eps: float = 1e-6) -> np.ndarray:
    """Run Keras GRN with default initializers (scale=1, bias=0) and return NumPy result."""
    layer = GlobalResponseNormalization(epsilon=eps)
    layer.build(x_np.shape)
    out = layer(x_np.astype(np.float32))
    # stop_gradient is the backend-neutral detach: the torch backend refuses numpy() on a tensor
    # that requires grad.
    return np.asarray(keras.ops.convert_to_numpy(keras.ops.stop_gradient(out)))


def test_grn_matches_timm_simple() -> None:
    """Keras GRN output matches timm GlobalResponseNorm on a simple input."""
    rng = np.random.RandomState(42)
    x = rng.randn(1, 4, 4, 8).astype(np.float32)
    expected = _run_timm_grn(x, dim=8)
    actual = _run_keras_grn(x)
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)


def test_grn_matches_timm_large_batch() -> None:
    """Keras GRN matches timm on a larger batch."""
    rng = np.random.RandomState(123)
    x = rng.randn(4, 8, 8, 16).astype(np.float32)
    expected = _run_timm_grn(x, dim=16)
    actual = _run_keras_grn(x)
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)


def test_grn_matches_timm_single_spatial() -> None:
    """Keras GRN matches timm when spatial dims are 1x1."""
    rng = np.random.RandomState(7)
    x = rng.randn(2, 1, 1, 4).astype(np.float32)
    expected = _run_timm_grn(x, dim=4)
    actual = _run_keras_grn(x)
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)


def test_grn_custom_epsilon() -> None:
    """Keras GRN matches timm when custom epsilon is used."""
    rng = np.random.RandomState(99)
    x = rng.randn(1, 2, 2, 3).astype(np.float32)
    expected = _run_timm_grn(x, dim=3, eps=1e-3)
    actual = _run_keras_grn(x, eps=1e-3)
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize(("spatial_h", "spatial_w", "channels"), [(2, 3, 5), (6, 6, 32), (3, 3, 1)])
def test_grn_matches_timm_various_shapes(spatial_h: int, spatial_w: int, channels: int) -> None:
    """Keras GRN matches timm across various spatial/channel sizes."""
    rng = np.random.RandomState(spatial_h + spatial_w + channels)
    x = rng.randn(2, spatial_h, spatial_w, channels).astype(np.float32)
    expected = _run_timm_grn(x, dim=channels)
    actual = _run_keras_grn(x)
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)


def test_grn_build_sets_weights() -> None:
    """Build properly creates scale and bias parameters."""
    layer = GlobalResponseNormalization()
    layer.build((1, 4, 4, 8))
    assert layer.scale.shape == (8,)
    assert layer.bias.shape == (8,)


def test_grn_feature_axes_tuple() -> None:
    """Feature axes as a tuple produces correct parameter shape."""
    layer = GlobalResponseNormalization(feature_axes=(-1,))
    layer.build((1, 4, 4, 8))
    assert layer.scale.shape == (8,)
