"""Tests for channel shuffle layer."""

from structcast_model.torch.layers.channel_shuffle import ChannelLastShuffle
import torch


def test_channel_last_shuffle_forward() -> None:
    """Shuffle channels along the last dimension by groups."""
    output = ChannelLastShuffle(groups=2)(torch.tensor([[1, 2, 3, 4]], dtype=torch.float32))
    expected = torch.tensor([[1, 3, 2, 4]], dtype=torch.float32)
    assert torch.equal(output, expected)


def test_channel_last_shuffle_extra_repr() -> None:
    """Return configured group count in extra representation."""
    assert ChannelLastShuffle(groups=4).extra_repr() == "groups=4"
