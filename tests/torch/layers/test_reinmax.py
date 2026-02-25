"""Tests for reinmax."""

import pytest

from structcast_model.torch.layers.reinmax import reinmax
import torch


def test_reinmax_raises_when_tau_below_one() -> None:
    """Raise an error for unsupported temperature values."""
    with pytest.raises(ValueError, match="temperature"):
        reinmax(torch.randn(2, 3), tau=0.5)


def test_reinmax_outputs_shapes_and_probabilities() -> None:
    """Return hard and soft samples with valid shape and distributions."""
    torch.manual_seed(7)
    logits = torch.randn(4, 5)
    y_hard, y_soft = reinmax(logits, tau=1.0)
    assert y_hard.shape == logits.shape
    assert y_soft.shape == logits.shape
    assert torch.equal(y_hard.sum(dim=-1), torch.ones(4))
    assert torch.allclose(y_soft.sum(dim=-1), torch.ones(4), atol=1e-6)


def test_reinmax_backward_produces_finite_gradients() -> None:
    """Backpropagate through ReinMax outputs."""
    torch.manual_seed(13)
    logits = torch.randn(3, 4, requires_grad=True)
    y_hard, y_soft = reinmax(logits, tau=1.0)
    loss = (y_hard + y_soft).sum()
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
