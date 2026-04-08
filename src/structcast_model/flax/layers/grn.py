"""Global Response Normalization (GRN) layer implementation in Flax."""

from types import MappingProxyType
import typing as tp

from flax.nnx import Module, Param, rnglib
from flax.nnx.nn import dtypes, initializers
from flax.typing import Axes, Dtype, Initializer, PromoteDtypeFn
import jax
import jax.numpy as jnp


class GlobalResponseNorm(Module):
    """Global Response Normalization (GRN) layer."""

    def __init__(
        self,
        num_features: int,
        *,
        epsilon: float = 1e-6,
        dtype: Dtype | None = None,
        param_dtype: Dtype = jnp.float32,
        bias_init: Initializer = initializers.zeros_init(),  # noqa: B008
        scale_init: Initializer = initializers.ones_init(),  # noqa: B008
        reduction_axes: Axes = (1, 2),
        feature_axes: Axes = -1,
        promote_dtype: PromoteDtypeFn = dtypes.promote_dtype,
        rngs: rnglib.Rngs,
        bias_metadata: tp.Mapping[str, tp.Any] = MappingProxyType({}),
        scale_metadata: tp.Mapping[str, tp.Any] = MappingProxyType({}),
    ) -> None:
        """Initializes the GRN layer."""
        feature_shape = (num_features,)
        self.scale = Param(scale_init(rngs.params(), feature_shape, param_dtype), **scale_metadata)
        self.bias = Param(bias_init(rngs.params(), feature_shape, param_dtype), **bias_metadata)
        self.num_features = num_features
        self.epsilon = epsilon
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.reduction_axes = reduction_axes
        self.feature_axes = feature_axes
        self.promote_dtype = promote_dtype

    def __call__(self, x: jax.Array) -> jax.Array:
        """Applies Global Response Normalization to the input."""
        x, scale, bias = self.promote_dtype((x, self.scale, self.bias), dtype=self.dtype)
        x_g = jax.lax.rsqrt((x * x).sum(axis=self.reduction_axes, keepdims=True))
        x_n = x_g / (x_g.mean(axis=self.feature_axes, keepdims=True) + self.epsilon)
        return x + (x * x_n) * scale + bias
