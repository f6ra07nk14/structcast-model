"""Tests for reduce layers."""

from structcast_model.torch.layers.reduce import ReduceSum
import torch


def test_reduce_sum_all_dimensions() -> None:
    """Reduce all dimensions when dim is None."""
    assert torch.isclose(ReduceSum()(torch.tensor([[1.0, 2.0], [3.0, 4.0]])), torch.tensor(10.0))


def test_reduce_sum_specific_dims_keepdim() -> None:
    """Reduce specified dimensions while preserving shape."""
    output = ReduceSum(dim=(1, 2), keepdim=True)(torch.ones(2, 3, 4))
    assert output.shape == (2, 1, 1)
    assert torch.equal(output, torch.full((2, 1, 1), 12.0))
