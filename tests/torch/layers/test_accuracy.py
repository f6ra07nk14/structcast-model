"""Tests for accuracy layers."""

from structcast_model.torch.layers.accuracy import sparse_categorical_accuracy, sparse_top_k_categorical_accuracy
import torch


def test_sparse_categorical_accuracy_with_index_labels() -> None:
    """Compute sparse categorical accuracy from index labels."""
    y_true = torch.tensor([0, 2])
    y_pred = torch.tensor([[0.9, 0.1, 0.0], [0.1, 0.2, 0.7]])
    assert torch.isclose(sparse_categorical_accuracy(y_true, y_pred), torch.tensor(1.0))


def test_sparse_categorical_accuracy_with_one_hot_labels() -> None:
    """Compute sparse categorical accuracy from one-hot labels."""
    y_true = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    y_pred = torch.tensor([[0.8, 0.2, 0.0], [0.4, 0.1, 0.5]])
    assert torch.isclose(sparse_categorical_accuracy(y_true, y_pred), torch.tensor(1.0))


def test_sparse_categorical_accuracy_with_last_dim_singleton() -> None:
    """Compute sparse categorical accuracy when labels use last singleton dimension."""
    y_true = torch.tensor([[0], [2]])
    y_pred = torch.tensor([[0.3, 0.7, 0.0], [0.1, 0.3, 0.6]])
    assert torch.isclose(sparse_categorical_accuracy(y_true, y_pred), torch.tensor(0.5))


def test_sparse_top_k_categorical_accuracy_with_index_labels() -> None:
    """Compute sparse top-k categorical accuracy from index labels."""
    y_true = torch.tensor([0, 2])
    y_pred = torch.tensor([[0.5, 0.4, 0.1], [0.1, 0.7, 0.2]])
    assert torch.isclose(sparse_top_k_categorical_accuracy(y_true, y_pred, k=2), torch.tensor(1.0))


def test_sparse_top_k_categorical_accuracy_with_one_hot_labels() -> None:
    """Compute sparse top-k categorical accuracy from one-hot labels."""
    y_true = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    y_pred = torch.tensor([[0.6, 0.2, 0.2], [0.1, 0.8, 0.1]])
    assert torch.isclose(sparse_top_k_categorical_accuracy(y_true, y_pred, k=2), torch.tensor(0.5))


def test_sparse_top_k_categorical_accuracy_with_last_dim_singleton() -> None:
    """Compute sparse top-k categorical accuracy when labels use last singleton dimension."""
    y_true = torch.tensor([[0], [2]])
    y_pred = torch.tensor([[0.9, 0.1, 0.0], [0.2, 0.5, 0.3]])
    assert torch.isclose(sparse_top_k_categorical_accuracy(y_true, y_pred, k=2), torch.tensor(1.0))
