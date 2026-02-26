"""Accuracy."""

from typing import TYPE_CHECKING

import torch


def sparse_categorical_accuracy(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """Sparse categorical accuracy."""
    if y_true.ndim == y_pred.ndim:
        if y_true.shape[-1] == 1:
            y_true.squeeze_()
        else:
            y_true = y_true.argmax(dim=-1)
    return torch.eq(torch.argmax(y_pred, dim=-1), y_true).float().mean()


def sparse_top_k_categorical_accuracy(y_true: torch.Tensor, y_pred: torch.Tensor, k: int = 5) -> torch.Tensor:
    """Sparse top k categorical accuracy."""
    if y_true.ndim == y_pred.ndim:
        if y_true.shape[-1] == 1:
            expanded_y = y_true.expand(-1, k)
        else:
            expanded_y = y_true.argmax(dim=-1, keepdim=True).expand(-1, k)
    else:
        expanded_y = y_true.view(-1, 1).expand(-1, k)
    _, sorted_indices = torch.topk(y_pred, k=k, dim=-1)
    return torch.eq(sorted_indices, expanded_y).float().sum(dim=-1).mean()


__all__ = ["sparse_categorical_accuracy", "sparse_top_k_categorical_accuracy"]

if not TYPE_CHECKING:
    import sys

    from structcast.utils.lazy_import import LazySelectedImporter

    sys.modules[__name__] = LazySelectedImporter(__name__, globals())
