"""Tests for concatenate layer."""

from structcast_model.torch.layers.concatenate import Concat, Concatenate
import torch


def test_concatenate_forward_with_custom_dim() -> None:
    """Concatenate tensors along the configured dimension."""
    output = Concatenate(dim=1)([torch.ones(2, 1), torch.zeros(2, 2)])
    assert output.shape == (2, 3)
    assert torch.equal(output[:, :1], torch.ones(2, 1))


def test_concatenate_alias_matches_class() -> None:
    """Provide Concat alias for Concatenate."""
    assert Concat is Concatenate


def test_concatenate_extra_repr() -> None:
    """Include concat dimension in extra representation."""
    assert Concatenate(dim=-1).extra_repr() == "dim=-1"
