"""Tests for the causal self-attention layer."""

import pytest

from structcast_model.torch.layers.attention import CausalSelfAttention
import torch


def test_causal_self_attention_ignores_future_tokens() -> None:
    """Rewrite the future of a sequence and every earlier position must keep its output.

    This is what makes the layer usable for next-token prediction: if a position could see the
    tokens after it, the training loss would be computed on answers the model was shown.
    """
    torch.manual_seed(0)
    layer = CausalSelfAttention(embed_dim=8, num_heads=2).eval()
    sequence = torch.randn(1, 6, 8)
    rewritten = sequence.clone()
    rewritten[:, 3:] = torch.randn(1, 3, 8)

    output, changed = layer(sequence), layer(rewritten)

    assert torch.allclose(output[:, :3], changed[:, :3], atol=1e-6)
    assert not torch.allclose(output[:, 3:], changed[:, 3:], atol=1e-6)


def test_causal_self_attention_rejects_indivisible_embedding_dimension() -> None:
    """Refuse a head count the embedding dimension cannot be split into, instead of silently reshaping."""
    with pytest.raises(ValueError, match="divisible"):
        CausalSelfAttention(embed_dim=9, num_heads=2)
