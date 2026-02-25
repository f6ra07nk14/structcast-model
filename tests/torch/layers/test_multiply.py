"""Tests for multiply layer."""

from structcast_model.torch.layers.multiply import Multiply
import torch


def test_multiply_forward_multiple_inputs() -> None:
    """Multiply tensors element-wise across multiple inputs."""
    output = Multiply()([torch.tensor([2.0, 3.0]), torch.tensor([4.0, 5.0]), torch.tensor([0.5, 2.0])])
    assert torch.equal(output, torch.tensor([4.0, 30.0]))


def test_multiply_forward_single_input() -> None:
    """Return the original tensor when a single input is provided."""
    tensor = torch.tensor([2.0, 3.0])
    assert torch.equal(Multiply()([tensor]), tensor)
