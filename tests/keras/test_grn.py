"""Unit tests for structcast_model.keras.layers.grn using timm as reference."""

from __future__ import annotations

import jax
import numpy as np
import pytest
import tensorflow as tf
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


def _timm_grn_input_grad(x_np: np.ndarray, *, dim: int, eps: float = 1e-6) -> np.ndarray:
    """Return d(sum(timm_grn(x)))/dx with scale=1, bias=0, as NumPy."""
    grn = TimmGRN(dim=dim, eps=eps, channels_last=True)
    with torch.no_grad():
        grn.weight.fill_(1.0)
        grn.bias.fill_(0.0)
    x = torch.from_numpy(x_np).float().requires_grad_()
    grn(x).sum().backward()
    assert x.grad is not None
    return x.grad.numpy()


def _zero_channel_input() -> np.ndarray:
    """Random input in which two channels are exactly zero over every spatial position."""
    rng = np.random.RandomState(2024)
    x = rng.randn(2, 4, 4, 8).astype(np.float32)
    x[0, :, :, 3] = 0.0
    x[1, :, :, 6] = 0.0
    return x


def _input_gradient(layer: GlobalResponseNormalization, x_np: np.ndarray) -> np.ndarray:
    """Return d(sum(layer(x)))/dx as NumPy, on whichever backend Keras resolved.

    Keras has no backend-neutral gradient API, so each of the three tox backends needs its own call.
    """
    backend = keras.backend.backend()
    if backend == "jax":
        return np.asarray(jax.grad(lambda v: keras.ops.sum(layer(v)))(jax.numpy.asarray(x_np)), dtype=np.float32)
    if backend == "tensorflow":
        x_tf = tf.convert_to_tensor(x_np)
        with tf.GradientTape() as tape:
            tape.watch(x_tf)
            loss = keras.ops.sum(layer(x_tf))
        return np.asarray(tape.gradient(loss, x_tf), dtype=np.float32)
    x_torch = torch.from_numpy(x_np).requires_grad_()
    keras.ops.sum(layer(x_torch)).backward()
    assert x_torch.grad is not None
    return np.asarray(x_torch.grad.float().numpy(), dtype=np.float32)


def _run_keras_grn(x_np: np.ndarray, *, eps: float = 1e-6) -> np.ndarray:
    """Run Keras GRN with scale=1, bias=0 and return NumPy result."""
    layer = GlobalResponseNormalization(epsilon=eps, gamma_initializer="ones")
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
    np.testing.assert_array_equal(keras.ops.convert_to_numpy(layer.scale), np.zeros(8))
    np.testing.assert_array_equal(keras.ops.convert_to_numpy(layer.bias), np.zeros(8))


def test_grn_feature_axes_tuple() -> None:
    """Feature axes as a tuple produces correct parameter shape."""
    layer = GlobalResponseNormalization(feature_axes=(-1,))
    layer.build((1, 4, 4, 8))
    assert layer.scale.shape == (8,)


def test_grn_zero_channel_gradient_matches_timm() -> None:
    """An all-zero channel keeps the gradient finite and equal to timm's.

    Over such a channel `sum(x*x)` is 0, and `sqrt` has an infinite derivative there, so the naive
    norm back-propagates `inf * 0 = NaN` -- the failure that NaN'd two real ConvNeXt V2 ImageNet
    trainings. timm's `x.norm(p=2)` defines the sub-gradient at zero norm as 0, and timm is the
    reference this layer is held to, so the gradient has to be 0 there, not merely finite.
    """
    x = _zero_channel_input()
    np.testing.assert_allclose(_run_keras_grn(x), _run_timm_grn(x, dim=8), rtol=1e-5, atol=1e-6)

    layer = GlobalResponseNormalization(gamma_initializer="ones")
    layer.build(x.shape)
    grad = _input_gradient(layer, x)
    assert np.isfinite(grad).all()
    np.testing.assert_allclose(grad, _timm_grn_input_grad(x, dim=8), rtol=1e-5, atol=1e-6)


def test_grn_bfloat16_zero_channel_gradient_is_finite() -> None:
    """Under `mixed_bfloat16` the layer tracks float32 and keeps the zero-channel gradient finite.

    The campaign trains under `mixed_bfloat16`, so bfloat16 -- not float32 -- is where the zero-norm
    guard has to hold. The guard is dtype-neutral, so the only claim here is that half precision tracks
    the float32 layer to bfloat16 tolerance and that the all-zero channel still back-propagates a finite
    gradient.
    """
    x = _zero_channel_input()
    reference = keras.ops.convert_to_numpy(keras.ops.cast(keras.ops.cast(_run_keras_grn(x), "bfloat16"), "float32"))
    previous = keras.mixed_precision.global_policy()
    try:
        keras.mixed_precision.set_global_policy("mixed_bfloat16")
        layer = GlobalResponseNormalization(gamma_initializer="ones")
        layer.build(x.shape)
        out = layer(x)
        assert keras.backend.standardize_dtype(out.dtype) == "bfloat16"
        actual = keras.ops.convert_to_numpy(keras.ops.cast(keras.ops.stop_gradient(out), "float32"))
        np.testing.assert_allclose(actual, reference, rtol=1e-2, atol=1e-2)
        assert np.isfinite(_input_gradient(layer, x)).all()
    finally:
        keras.mixed_precision.set_global_policy(previous)


def test_grn_default_is_identity_like_timm() -> None:
    """Zero-initialized GRN leaves the residual branch unchanged, matching timm."""
    x = _zero_channel_input()
    layer = GlobalResponseNormalization()
    layer.build(x.shape)
    expected = TimmGRN(dim=8)(torch.from_numpy(x)).detach().numpy()
    actual = keras.ops.convert_to_numpy(keras.ops.stop_gradient(layer(x)))
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(expected, x)
    np.testing.assert_array_equal(_input_gradient(layer, x), np.ones_like(x))
