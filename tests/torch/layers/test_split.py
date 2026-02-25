"""Tests for split layer."""

from structcast_model.torch.layers.split import Split
import torch


def test_split_forward_with_fixed_size() -> None:
    """Split tensor into equal chunks by size."""
    outputs = Split(2, dim=1)(torch.arange(12, dtype=torch.float32).view(2, 6))
    assert len(outputs) == 3
    assert outputs[0].shape == (2, 2)


def test_split_forward_with_sections() -> None:
    """Split tensor into configured sections."""
    outputs = Split([1, 3, 2], dim=1)(torch.arange(12, dtype=torch.float32).view(2, 6))
    assert [part.shape[1] for part in outputs] == [1, 3, 2]


def test_split_extra_repr() -> None:
    """Include split configuration in extra representation."""
    assert Split(3, dim=-1).extra_repr() == "split_size_or_sections=3, dim=-1"
