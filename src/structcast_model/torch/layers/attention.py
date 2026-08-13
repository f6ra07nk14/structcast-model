"""Causal self-attention layer for PyTorch."""

from torch.nn import Dropout, Linear, Module
from torch.nn.functional import scaled_dot_product_attention

from structcast_model.torch.types import Tensor


class CausalSelfAttention(Module):
    """Multi-head self-attention over a sequence, masked so a position never attends to later ones.

    The mask is not materialized: `scaled_dot_product_attention(..., is_causal=True)` applies it
    internally, which keeps the memory cost independent of the sequence length.
    """

    __constants__ = ["embed_dim", "num_heads", "head_dim"]
    embed_dim: int
    num_heads: int
    head_dim: int

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0, bias: bool = True) -> None:
        """Initialize the layer.

        Args:
            embed_dim (int): The size of the embedding dimension, which must be divisible by `num_heads`.
            num_heads (int): The number of attention heads.
            dropout (float, optional): The dropout probability of the attention weights and the output.
                Defaults to 0.0.
            bias (bool, optional): Whether the projections have a bias. Defaults to True.

        Raises:
            ValueError: If `embed_dim` is not divisible by `num_heads`.
        """
        super().__init__()
        if embed_dim % num_heads:
            raise ValueError(f"embed_dim {embed_dim} must be divisible by num_heads {num_heads}.")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout
        self.qkv = Linear(embed_dim, 3 * embed_dim, bias=bias)
        self.proj = Linear(embed_dim, embed_dim, bias=bias)
        self.proj_drop = Dropout(dropout)

    def forward(self, input: Tensor) -> Tensor:  # pylint: disable=redefined-builtin
        """Forward pass.

        Args:
            input (Tensor): The input tensor of shape `(batch, sequence, embed_dim)`.

        Returns:
            Tensor: The attended tensor, of the same shape as the input.
        """
        batch, seq, _ = input.shape
        qkv = self.qkv(input).reshape(batch, seq, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        query, key, value = qkv[0], qkv[1], qkv[2]
        attended = scaled_dot_product_attention(
            query, key, value, dropout_p=self.dropout if self.training else 0.0, is_causal=True
        )
        return self.proj_drop(self.proj(attended.transpose(1, 2).reshape(batch, seq, self.embed_dim)))

    def extra_repr(self) -> str:
        """Extra representation of the layer.

        Returns:
            str: The extra representation.
        """
        return f"embed_dim={self.embed_dim}, num_heads={self.num_heads}, dropout={self.dropout}"


__all__ = ["CausalSelfAttention"]
