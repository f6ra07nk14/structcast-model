"""Global Response Normalization (GRN) layer implementation in Keras."""

from typing import Any

import keras
from keras import ops


class GlobalResponseNormalization(keras.layers.Layer):
    """Global Response Normalization (GRN) layer.

    GRN is a normalization technique that normalizes the input across the feature dimension.
    It is designed to improve the training of deep neural networks by stabilizing the activations and gradients.

    Args:
        reduction_axes (int | tuple[int, ...]): The axes along which to compute the L2 norm for normalization.
        feature_axes (int | tuple[int, ...]): The axes along which to compute the mean for normalization.
        epsilon (float): A small constant added to the denominator for numerical stability.
        beta_initializer: Initializer for the bias parameter.
        gamma_initializer: Initializer for the scale parameter.
        beta_regularizer: Regularizer for the bias parameter.
        gamma_regularizer: Regularizer for the scale parameter.
        beta_constraint: Constraint for the bias parameter.
        gamma_constraint: Constraint for the scale parameter.
    """

    def __init__(
        self,
        reduction_axes: int | tuple[int, ...] = (1, 2),
        feature_axes: int | tuple[int, ...] = -1,
        epsilon: float = 1e-6,
        beta_initializer: Any = "zeros",
        gamma_initializer: Any = "ones",
        beta_regularizer: Any | None = None,
        gamma_regularizer: Any | None = None,
        beta_constraint: Any | None = None,
        gamma_constraint: Any | None = None,
        **kwargs: Any,
    ) -> None:
        """Initializes the GRN layer."""
        super().__init__(**kwargs)
        self.reduction_axes = reduction_axes
        self.feature_axes = feature_axes
        self.epsilon = epsilon
        self.beta_initializer = keras.initializers.get(beta_initializer)
        self.gamma_initializer = keras.initializers.get(gamma_initializer)
        self.beta_regularizer = keras.regularizers.get(beta_regularizer)
        self.gamma_regularizer = keras.regularizers.get(gamma_regularizer)
        self.beta_constraint = keras.constraints.get(beta_constraint)
        self.gamma_constraint = keras.constraints.get(gamma_constraint)

    def build(self, input_shape: tuple[int | None, ...]) -> None:
        """Builds the layer's parameters."""
        if isinstance(self.feature_axes, (tuple, list)):
            param_shape = tuple(input_shape[axis] for axis in self.feature_axes)
        else:
            param_shape = (input_shape[self.feature_axes],)
        self.scale = self.add_weight(
            name="scale",
            shape=param_shape,
            initializer=self.gamma_initializer,
            regularizer=self.gamma_regularizer,
            constraint=self.gamma_constraint,
            trainable=True,
            dtype=self.dtype,
        )
        self.bias = self.add_weight(
            name="bias",
            shape=param_shape,
            initializer=self.beta_initializer,
            regularizer=self.beta_regularizer,
            constraint=self.beta_constraint,
            trainable=True,
            dtype=self.dtype,
        )

    def call(self, inputs: keras.KerasTensor) -> keras.KerasTensor:
        """Applies Global Response Normalization to the input."""
        # `sqrt` has an infinite derivative at 0, so an all-zero channel back-propagates `inf * 0 = NaN`.
        # This mirrors `optax.safe_norm`: mask the zero vector to ones and renorm, so the untaken branch
        # stays finite and the zero-norm sub-gradient is 0, matching PyTorch `norm_backward` (timm). See
        # https://github.com/google-deepmind/optax/blob/main/optax/_src/numerics.py#L48-L83 and
        # https://docs.jax.dev/en/latest/faq.html#gradients-contain-nan-where-using-where on the inner `where`.
        norm = ops.sqrt(ops.sum(ops.square(inputs), axis=self.reduction_axes, keepdims=True))
        masked = ops.where(ops.less_equal(norm, 0.0), ops.ones_like(inputs), inputs)
        masked_norm = ops.sqrt(ops.sum(ops.square(masked), axis=self.reduction_axes, keepdims=True))
        x_g = ops.where(ops.less_equal(norm, 0.0), 0.0, masked_norm)
        x_n = x_g / (ops.mean(x_g, axis=self.feature_axes, keepdims=True) + self.epsilon)
        return inputs + (inputs * x_n) * self.scale + self.bias
