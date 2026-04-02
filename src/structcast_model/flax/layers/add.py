"""Add layer for Flax nnx."""


class Add:
    """Element-wise addition of a list of arrays.

    Accepts a list of arrays and returns their element-wise sum.  Intended as
    a drop-in counterpart to :class:`structcast_model.torch.layers.Add` for
    use in generated Flax model scripts.
    """

    def __call__(self, tensors: list[object], **kwargs: object) -> object:
        """Return the element-wise sum of *tensors*.

        Args:
            tensors: A list of arrays with the same shape.
            **kwargs: Ignored; accepted for API compatibility.

        Returns:
            The element-wise sum of all arrays in *tensors*.
        """
        import jax.numpy as jnp  # noqa: PLC0415

        return jnp.sum(jnp.stack(tensors, axis=0), axis=0)


__all__ = ["Add"]
