"""Global average pooling layer for Flax nnx."""


class GlobalAveragePool2D:
    """Global average pooling over spatial (H, W) axes.

    Reduces a ``(B, H, W, C)`` array to ``(B, C)`` by averaging over the
    height and width dimensions.  Intended as a lightweight counterpart to
    ``keras.layers.GlobalAveragePooling2D`` for use in generated Flax model
    scripts.
    """

    def __call__(self, x: object, **kwargs: object) -> object:
        """Average *x* over the spatial axes.

        Args:
            x: Array of shape ``(B, H, W, C)``.
            **kwargs: Ignored; accepted for API compatibility.

        Returns:
            Array of shape ``(B, C)``.
        """
        import jax.numpy as jnp  # noqa: PLC0415

        return jnp.mean(x, axis=(1, 2))  # type: ignore[arg-type]


__all__ = ["GlobalAveragePool2D"]
