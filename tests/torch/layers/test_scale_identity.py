"""Tests for scale identity layer."""

from structcast_model.torch.layers.scale_identity import ScaleIdentity
import torch


def test_scale_identity_forward_with_fixed_scale() -> None:
    """Scale input tensor with non-trainable parameter."""
    layer = ScaleIdentity(scale=2.5, trainable=False)
    output = layer(torch.tensor([1.0, -2.0]))
    assert torch.equal(output, torch.tensor([2.5, -5.0]))
    assert layer.scale.requires_grad is False


def test_scale_identity_trainable_scale() -> None:
    """Create a trainable scale parameter when requested."""
    assert ScaleIdentity(scale=1.5, trainable=True).scale.requires_grad is True
