"""Trainer helpers for Flax models."""

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import jax
import numpy as np
from pydantic import TypeAdapter, ValidationError


def create_jax_inputs(shape: Any, *, batch_size: int = 1) -> Any:
    """Create dummy JAX inputs based on the provided shape.

    Args:
        shape (Any): The shape of the inputs to create. This can be a tuple of integers,
            a dictionary of shapes, or a list of shapes.
        batch_size (int): The batch size to use for the inputs.

    Returns:
        Any: The created inputs, which can be a JAX array, a dictionary of arrays, or a list of arrays.
    """
    try:
        shape = TypeAdapter(tuple[int, ...]).validate_python(shape)
        return jax.numpy.array(np.random.rand(batch_size, *shape).astype(np.float32))
    except ValidationError:
        pass
    if isinstance(shape, Mapping):
        return {key: create_jax_inputs(value, batch_size=batch_size) for key, value in shape.items()}
    if isinstance(shape, (list, tuple)):
        return [create_jax_inputs(value, batch_size=batch_size) for value in shape]
    raise ValueError(f"Invalid tensor shape: {shape}")


__all__ = ["create_jax_inputs"]


if not TYPE_CHECKING:
    import sys

    from structcast.utils.lazy_import import LazySelectedImporter

    sys.modules[__name__] = LazySelectedImporter(__name__, globals())
