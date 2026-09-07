"""Unit tests for structcast_model.flax.layers.grn using timm as reference."""

from __future__ import annotations

from flax.nnx import Rngs
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from timm.layers.grn import GlobalResponseNorm as TimmGRN

from structcast_model.flax.layers.grn import GlobalResponseNorm as FlaxGRN
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


def _run_flax_grn(x_np: np.ndarray, *, dim: int, eps: float = 1e-6) -> np.ndarray:
    """Run Flax GRN with scale=1, bias=0 and return NumPy result."""
    grn = FlaxGRN(num_features=dim, epsilon=eps, rngs=Rngs(0))
    out = grn(jnp.array(x_np))
    return np.array(out)


def test_grn_matches_timm_simple() -> None:
    """Flax GRN output matches timm GlobalResponseNorm on a simple input."""
    rng = np.random.RandomState(42)
    x = rng.randn(1, 4, 4, 8).astype(np.float32)
    expected = _run_timm_grn(x, dim=8)
    actual = _run_flax_grn(x, dim=8)
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)


def test_grn_matches_timm_large_batch() -> None:
    """Flax GRN matches timm on a larger batch."""
    rng = np.random.RandomState(123)
    x = rng.randn(4, 8, 8, 16).astype(np.float32)
    expected = _run_timm_grn(x, dim=16)
    actual = _run_flax_grn(x, dim=16)
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)


def test_grn_matches_timm_single_spatial() -> None:
    """Flax GRN matches timm when spatial dims are 1x1."""
    rng = np.random.RandomState(7)
    x = rng.randn(2, 1, 1, 4).astype(np.float32)
    expected = _run_timm_grn(x, dim=4)
    actual = _run_flax_grn(x, dim=4)
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)


def test_grn_custom_epsilon() -> None:
    """Flax GRN matches timm when custom epsilon is used."""
    rng = np.random.RandomState(99)
    x = rng.randn(1, 2, 2, 3).astype(np.float32)
    expected = _run_timm_grn(x, dim=3, eps=1e-3)
    actual = _run_flax_grn(x, dim=3, eps=1e-3)
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize(("spatial_h", "spatial_w", "channels"), [(2, 3, 5), (6, 6, 32), (3, 3, 1)])
def test_grn_matches_timm_various_shapes(spatial_h: int, spatial_w: int, channels: int) -> None:
    """Flax GRN matches timm across various spatial/channel sizes."""
    rng = np.random.RandomState(spatial_h + spatial_w + channels)
    x = rng.randn(2, spatial_h, spatial_w, channels).astype(np.float32)
    expected = _run_timm_grn(x, dim=channels)
    actual = _run_flax_grn(x, dim=channels)
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)


def test_grn_initializes_scale_and_bias() -> None:
    """Scale and bias parameters are initialized with expected shapes and values."""
    grn = FlaxGRN(num_features=8, rngs=Rngs(0))
    assert grn.scale[...].shape == (8,)
    assert grn.bias[...].shape == (8,)
    np.testing.assert_allclose(np.array(grn.scale[...]), np.ones(8), atol=1e-7)
    np.testing.assert_allclose(np.array(grn.bias[...]), np.zeros(8), atol=1e-7)


def test_grn_output_shape_matches_input() -> None:
    """GRN output shape matches the input shape."""
    grn = FlaxGRN(num_features=4, rngs=Rngs(0))
    x = jnp.ones((2, 3, 3, 4))
    out = grn(x)
    assert out.shape == x.shape


def test_grn_zero_channel_gradient_matches_timm() -> None:
    """An all-zero channel keeps the gradient finite and equal to timm's.

    Over such a channel `sum(x*x)` is 0, and `sqrt` has an infinite derivative there, so the naive
    norm back-propagates `inf * 0 = NaN` -- the failure that NaN'd two real ConvNeXt V2 ImageNet
    trainings. timm's `x.norm(p=2)` defines the sub-gradient at zero norm as 0, and timm is the
    reference this layer is held to, so the gradient has to be 0 there, not merely finite.
    """
    x = _zero_channel_input()
    np.testing.assert_allclose(_run_flax_grn(x, dim=8), _run_timm_grn(x, dim=8), rtol=1e-5, atol=1e-6)

    grn = FlaxGRN(num_features=8, epsilon=1e-6, rngs=Rngs(0))
    grad = np.array(jax.grad(lambda v: grn(v).sum())(jnp.array(x)))
    assert np.isfinite(grad).all()
    np.testing.assert_allclose(grad, _timm_grn_input_grad(x, dim=8), rtol=1e-5, atol=1e-6)


def test_grn_bfloat16_zero_channel_gradient_is_finite() -> None:
    """A bfloat16 layer stays close to the float32 layer and keeps the zero-channel gradient finite.

    The campaign trains in bfloat16, so bfloat16 -- not float32 -- is where the zero-norm guard has to
    hold. The guard is dtype-neutral, so the only claim here is that half precision tracks the float32
    layer to bfloat16 tolerance and that the all-zero channel still back-propagates a finite gradient.
    """
    x_bf16 = jnp.array(_zero_channel_input(), dtype=jnp.bfloat16)
    grn = FlaxGRN(num_features=8, dtype=jnp.bfloat16, rngs=Rngs(0))
    out = grn(x_bf16)
    assert out.dtype == jnp.bfloat16

    reference = FlaxGRN(num_features=8, rngs=Rngs(0))(x_bf16.astype(jnp.float32)).astype(jnp.bfloat16)
    np.testing.assert_allclose(
        np.array(out.astype(jnp.float32)), np.array(reference.astype(jnp.float32)), rtol=1e-2, atol=1e-2
    )

    grad = jax.grad(lambda v: grn(v).sum())(x_bf16)
    assert np.isfinite(np.array(grad.astype(jnp.float32))).all()
