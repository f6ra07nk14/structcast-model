"""Tests for add layer."""

from structcast_model.torch.layers.add import Add
import torch


def test_add_forward_multiple_inputs() -> None:
    """Add tensors element-wise across multiple inputs."""
    output = Add()([torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0]), torch.tensor([5.0, 6.0])])
    assert torch.equal(output, torch.tensor([9.0, 12.0]))


def test_add_forward_single_input() -> None:
    """Return the original tensor when a single input is provided."""
    tensor = torch.tensor([1.0, 2.0])
    assert torch.equal(Add()([tensor]), tensor)
