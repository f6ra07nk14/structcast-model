"""Tests for permute layers."""

from structcast_model.torch.layers.permute import Permute, ToChannelFirst, ToChannelLast
import torch


def test_permute_forward() -> None:
    """Permute dimensions with leading batch axis preserved."""
    assert Permute((2, 1, 3))(torch.randn(2, 3, 4, 5)).shape == (2, 4, 3, 5)


def test_permute_extra_repr() -> None:
    """Include permute dimensions in extra representation."""
    assert Permute((2, 3, 1)).extra_repr() == "dims=(2, 3, 1)"


def test_to_channel_last_initialization_and_forward() -> None:
    """Infer channel-last permutation order from input shape."""
    layer = ToChannelLast()
    output = layer(torch.randn(2, 3, 4, 5))
    assert layer.dims == (2, 3, 1)
    assert output.shape == (2, 4, 5, 3)


def test_to_channel_first_initialization_and_forward() -> None:
    """Infer channel-first permutation order from input shape."""
    layer = ToChannelFirst()
    output = layer(torch.randn(2, 4, 5, 3))

    assert layer.dims == (3, 1, 2)
    assert output.shape == (2, 3, 4, 5)
