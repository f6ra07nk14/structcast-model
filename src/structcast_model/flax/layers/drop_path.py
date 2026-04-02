"""Stochastic depth (DropPath) layer for Flax nnx."""


class DropPath:
    """Stochastic depth (DropPath) regularisation.

    During training a random subset of samples in the batch is zeroed out
    with probability ``drop_prob``.  During inference the layer acts as the
    identity.  Intended as a lightweight counterpart to
    ``timm.layers.DropPath`` for use in generated Flax model scripts.

    Args:
        drop_prob: Probability of dropping a sample path.  Set to ``0.0``
            to disable stochastic depth.
    """

    def __init__(self, drop_prob: float = 0.0) -> None:
        """Initialise DropPath with the given drop probability."""
        self.drop_prob = drop_prob

    def __call__(self, x: object, *, training: bool = True, **kwargs: object) -> object:
        """Apply stochastic depth to *x*.

        Args:
            x: Input array of shape ``(B, ...)``.
            training: When ``False`` (inference mode) the identity is returned.
            **kwargs: Ignored; accepted for API compatibility.

        Returns:
            Array of the same shape as *x* with paths randomly zeroed during
            training.

        Note:
            This is a simplified implementation that generates the random key
            from a hash of the input sum.  For production use, inject a proper
            ``jax.random.PRNGKey`` via an ``rngs`` argument or replace this
            layer with a ``flax.nnx``-aware implementation.
        """
        import jax  # noqa: PLC0415
        import jax.numpy as jnp  # noqa: PLC0415

        if not training or self.drop_prob == 0.0:
            return x
        keep_prob = 1.0 - self.drop_prob
        # Shape (B, 1, ..., 1): one mask value per sample, broadcast over spatial/channel dims.
        shape = (jnp.shape(x)[0],) + (1,) * (jnp.ndim(x) - 1)  # type: ignore[arg-type]
        seed = int(jnp.sum(jnp.asarray(x, dtype=jnp.float32)).item()) % (2**31)  # type: ignore[arg-type]
        rng = jax.random.PRNGKey(seed)
        random_tensor = jax.random.bernoulli(rng, keep_prob, shape=shape)
        return jnp.where(random_tensor, x / keep_prob, jnp.zeros_like(x))  # type: ignore[arg-type]


__all__ = ["DropPath"]
