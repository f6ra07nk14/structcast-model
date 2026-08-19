"""Trainer helpers for Flax models."""

from collections import OrderedDict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from functools import lru_cache
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import numpy as np
from pydantic import TypeAdapter, ValidationError

# Protocol and runtime_checkable come from typing_extensions so that isinstance checks use
# inspect.getattr_static on Python 3.11 as well (backported from 3.12), as in base_trainer.
from typing_extensions import Protocol, runtime_checkable

from flax import nnx
from structcast_model.base_trainer import BaseInfo, BaseTrainer
from structcast_model.builders.schema import TensorSpec, TensorSpecTree
from structcast_model.utils.base import resolve_input_shapes, resolve_tensor_initializer

DTYPES = {
    "float32": jax.numpy.float32,
    "float16": jax.numpy.float16,
    "bfloat16": jax.numpy.bfloat16,
    "int32": jax.numpy.int32,
    "int64": jax.numpy.int64,
}
"""JAX element types of the supported tensor element types.

`int64` is truncated to `int32` unless JAX is configured with `jax_enable_x64`.
"""


@runtime_checkable
class TensorInitializer(Protocol):
    """Callable creating a dummy JAX array, called as `initializer(size, dtype=...)`."""

    def __call__(self, size: tuple[int, ...], *, dtype: Any) -> Any:
        """Create an array of the given size and element type."""
        ...


def random_array(size: tuple[int, ...], *, dtype: Any) -> Any:
    """Create a uniformly distributed random JAX array, the default initializer for floating point types.

    `jax.random` cannot be used as an initializer directly, since it requires an explicit key.

    Args:
        size (tuple[int, ...]): The size of the array, including the batch dimension.
        dtype (Any): The element type of the array.

    Returns:
        Any: The created array.
    """
    return jax.numpy.array(np.random.random(size), dtype=dtype)


def create_jax_inputs(shape: Any, *, batch_size: int = 1) -> Any:
    """Create dummy JAX inputs based on the provided shape.

    Args:
        shape (Any): The shape of the inputs to create. This can be a tensor specification,
            which is a tuple of integers or a mapping with the `_SHAPE_` key,
            a dictionary of shapes, or a list of shapes.
        batch_size (int): The batch size to use for the inputs.
            This will be prepended to the shape of every tensor specification.

    Returns:
        Any: The created inputs, which can be a JAX array, a dictionary of arrays, or a list of arrays.

    Raises:
        ValueError: If the shape is neither a tensor specification nor a dictionary or list nesting more of them.
    """
    try:
        node: TensorSpecTree = TypeAdapter(TensorSpecTree).validate_python(shape)
    except ValidationError:
        raise ValueError(f"Invalid tensor shape: {shape}") from None
    if isinstance(node, TensorSpec):
        initializer = resolve_tensor_initializer(
            node.INIT,
            node.DTYPE,
            float_default=random_array,
            int_default=jax.numpy.zeros,
            protocol=TensorInitializer,
        )
        return initializer((batch_size, *node.SHAPE), dtype=DTYPES[node.DTYPE])
    if isinstance(node, Mapping):
        return {key: create_jax_inputs(value, batch_size=batch_size) for key, value in node.items()}
    return [create_jax_inputs(value, batch_size=batch_size) for value in node]


# `jax.Device` is Any to mypy: jaxlib re-exports it from its `_jax` C extension, which ships no stubs.
@lru_cache(maxsize=1)
def get_jax_devices() -> OrderedDict[str, jax.Device]:  # type: ignore[no-any-unimported]
    """Get a mapping of available JAX devices.

    Returns:
        OrderedDict[str, jax.Device]: An ordered dictionary mapping device strings (e.g., "cpu:0", "gpu:0") to
            JAX Device objects.
    """
    return OrderedDict((f"{d.platform}:{d.id}", d) for d in jax.devices())


# `jax.Device` is Any to mypy, as above.
def get_jax_device(device: str | None = None) -> jax.Device:  # type: ignore[no-any-unimported]
    """Get a JAX device based on the provided device string.

    Args:
        device (str | None): The device string to look for (e.g., "cpu:0", "gpu:0").
            If None, the first available device will be returned.

    Returns:
        jax.Device: The JAX device corresponding to the provided device string.
    """
    devices = get_jax_devices()
    if not devices:
        raise ValueError("No JAX devices are available.")
    if device is None:
        device = next(iter(devices))
    if device in devices:
        return devices[device]
    devices_str = ", ".join(f"{d!r}" for d in devices)
    raise ValueError(f"Specified device {device!r} is not available. Available devices: {devices_str}")


@dataclass(kw_only=True, slots=True)
class FlaxTracker:
    """Running mean of the criteria of one training or validation split.

    The device-side sums are plain JAX arrays, so accumulating a step costs one asynchronous add.
    The means come back as Python floats, not arrays: `BaseTrainer.tracker` is typed
    `Callable[..., dict[str, float]]` and everything downstream -- the epoch history, the
    `BestCriterion` comparison against a float infinity, and the loggers' `log_metric(value: float)`
    -- consumes that contract, as the torch tracker's `.item()` does. The host read is a single
    `jax.device_get` over all criteria at once, and it lands inside the region `BaseTrainer.train`
    times, where the torch loop blocks on `torch.cuda.synchronize()` for the same reason.

    Unlike the torch tracker there is no all-reduce: JAX is single-controller, so a criterion
    computed from a sharded batch is already the global value.

    Example:
        >>> import jax.numpy as jnp
        >>> from structcast_model.flax.trainer import FlaxTracker
        >>> tracker = FlaxTracker.from_criteria(["loss"])
        >>> tracker(loss=jnp.asarray(1.0))
        {'loss': 1.0}
        >>> tracker(loss=jnp.asarray(3.0))
        {'loss': 2.0}
        >>> tracker.reset()
        >>> tracker.logs()
        {}
    """

    criteria: tuple[str, ...]
    """Names of the criteria to track; a step must report every one of them."""

    sums: dict[str, jax.Array] = field(init=False)
    """Device-side sum of each criterion over the steps seen since the last reset."""

    count: int = field(default=0, init=False)
    """Number of steps summed since the last reset."""

    def __post_init__(self) -> None:
        """Start every criterion at a zero sum."""
        self.reset()

    def reset(self) -> None:
        """Zero the sums and the step count."""
        self.sums = {criterion: jnp.zeros((), jnp.float32) for criterion in self.criteria}
        self.count = 0

    def on_training_begin(self, info: BaseInfo[Any]) -> None:
        """Reset the tracker so an epoch's training averages start empty."""
        self.reset()

    def on_validation_begin(self, info: BaseInfo[Any]) -> None:
        """Reset the tracker so validation averages do not carry training values."""
        self.reset()

    def __call__(self, **criteria: Any) -> dict[str, float]:
        """Add one step's criteria to the sums and return the running means."""
        for criterion in self.criteria:
            self.sums[criterion] += criteria[criterion]
        self.count += 1
        return self.logs()

    def logs(self) -> dict[str, float]:
        """Return the running mean of each criterion, or an empty mapping before the first step."""
        if not self.count:
            return {}
        return {name: float(total) / self.count for name, total in jax.device_get(self.sums).items()}

    @classmethod
    def from_criteria(cls, outputs: Iterable[str]) -> "FlaxTracker":
        """Create a tracker for the named criteria, mirroring `TorchTracker.from_criteria`."""
        return cls(criteria=tuple(outputs))


@dataclass(kw_only=True)
class FlaxTrainer(BaseTrainer[nnx.Module]):
    """Trainer for Flax (nnx) models.

    `BaseTrainer.sync` stays the inherited no-op: `FlaxTracker.logs` reads the criteria to host
    memory on every step, which already waits for the step's program, and the device a run uses is
    the strategy's mesh, not a trainer field.
    """


__all__ = [
    "FlaxTracker",
    "FlaxTrainer",
    "TensorInitializer",
    "create_jax_inputs",
    "get_jax_device",
    "get_jax_devices",
    "resolve_input_shapes",
]


if not TYPE_CHECKING:
    import sys

    from structcast.utils.lazy_import import LazySelectedImporter

    sys.modules[__name__] = LazySelectedImporter(__name__, globals())
