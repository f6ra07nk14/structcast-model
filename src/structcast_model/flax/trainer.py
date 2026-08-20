"""Trainer helpers for Flax models."""

from collections import OrderedDict
from collections.abc import Callable, Collection, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from logging import getLogger
from typing import TYPE_CHECKING, Any, Self, cast

import jax
import jax.numpy as jnp
import numpy as np
from pydantic import TypeAdapter, ValidationError

# Protocol and runtime_checkable come from typing_extensions so that isinstance checks use
# inspect.getattr_static on Python 3.11 as well (backported from 3.12), as in base_trainer.
from typing_extensions import Protocol, runtime_checkable

from flax import nnx
from structcast_model.base_trainer import (
    EVENTS,
    BaseInfo,
    BaseTrainer,
    BestCriterion,
    DatasetLike,
    get_dataset,
    get_dataset_size,
)
from structcast_model.builders.schema import TensorSpec, TensorSpecTree
from structcast_model.loggers.base import Logger
from structcast_model.utils.base import resolve_input_shapes, resolve_tensor_initializer

if TYPE_CHECKING:
    # Only for the annotations: `flax.distributed` imports this module, so importing it back here
    # at runtime would close the cycle.
    from structcast_model.flax.distributed import FlaxDistributedStrategy

# `_logger`, not the usual `logger`: `restore_training_state` takes a `Logger` parameter named `logger`.
_logger = getLogger(__name__)

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


@dataclass(frozen=True)
class ShardedDataset:
    """A dataset whose batches are placed across the strategy's mesh as they are read.

    Built once per run and handed to the data provider, whose properties must return the same object
    on every read. Counting an epoch goes to the wrapped dataset, so it places nothing.
    """

    dataset: DatasetLike | Callable[[], DatasetLike]
    """The dataset producing the batches."""

    strategy: Any
    """The strategy whose mesh the batches are placed on."""

    def __iter__(self) -> Iterator[dict[str, Any]]:
        """Yield every batch of one epoch, placed on the mesh."""
        return (self.strategy.shard_batch(batch) for batch in get_dataset(self.dataset))

    def __len__(self) -> int:
        """Return the number of batches in one epoch."""
        return get_dataset_size(self.dataset)

    def __post_init__(self) -> None:
        """Take over the wrapped dataset's event methods, so the trainer still routes them to it.

        Copied onto the instance rather than forwarded from `__getattr__`: the trainer selects the
        participants of an event with `isinstance` against a protocol, and a protocol check looks its
        attributes up statically, which never reaches a `__getattr__`.
        """
        for event in EVENTS:
            method = getattr(self.dataset, event, None)
            if method is not None:
                object.__setattr__(self, event, method)


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


@dataclass(kw_only=True, slots=True)
class FlaxBestCriterion(BestCriterion[nnx.Module]):
    """A callback tracking the best value of a criterion during training or validation for Flax models.

    The twin of `structcast_model.torch.trainer.TorchBestCriterion`, duplicated rather than shared
    because importing either module imports its framework.
    """

    @classmethod
    def from_criteria(
        cls,
        higher_criteria: Sequence[str],
        lower_criteria: Sequence[str],
        save_criteria: Collection[str],
        logger: Logger,
        strategy: "FlaxDistributedStrategy",
    ) -> list[Self]:
        """Build one monitor per criterion, each logging its best value through *logger*.

        Criteria named in *save_criteria* also save the model states that reached the best value,
        produced through *strategy*.
        """
        monitors: list[Self] = []
        for target in higher_criteria:
            best = cls(target=target, mode="max")
            best.callbacks.append(_BestLogger(logger=logger, save=target in save_criteria, strategy=strategy))
            monitors.append(best)
        for target in lower_criteria:
            best = cls(target=target, mode="min")
            best.callbacks.append(_BestLogger(logger=logger, save=target in save_criteria, strategy=strategy))
            monitors.append(best)
        return monitors


@dataclass(kw_only=True, slots=True)
class _BestLogger:
    """Log the best value of a criterion, and save the models that reached it when asked to."""

    logger: Logger
    """The logger the best values and model states are written through."""

    save: bool
    """Whether to also save the model states that reached the best value."""

    strategy: "FlaxDistributedStrategy"
    """The strategy producing the model states to save."""

    def on_best(self, info: BaseInfo[nnx.Module], best: BestCriterion[nnx.Module]) -> None:
        """Log the best value, and save the states of the info's models when this epoch reached it."""
        name = f"best_{best.target}"
        self.logger.log_metric(name, best.value, step=info.epoch)
        if self.save and info.step == best.step:
            self.logger.log_state_dict(self.strategy.state_dict(dict(info.models))["models"], name)


@dataclass(kw_only=True, slots=True)
class FlaxTrainingStateSaver:
    """Callback saving models, optimizers and loop counters through a logger.

    The twin of `structcast_model.torch.trainer.TrainingStateSaver`, minus the gradient scalers Flax
    has none of; the payload keeps their (always empty) slot so both frameworks resume from the same
    shape (`docs/adr/0015`).
    """

    logger: Logger
    """The logger the training-state artifacts are written through."""

    strategy: "FlaxDistributedStrategy"
    """The strategy producing the model and optimizer states to save."""

    extra_meta: Mapping[str, Any] = field(default_factory=dict)
    """Run facts the loop does not know, recorded next to the counters: seed, configuration and
    optimizer hashes, which the resume path compares against the rebuilt configuration."""

    def on_epoch_end(self, info: BaseInfo[nnx.Module]) -> None:
        """Save the full training state of the finished epoch, so a run can be resumed from it."""
        learner = cast("FlaxTrainer", info).learner
        states = self.strategy.state_dict(dict(info.models), learner.optimizers, learner.optimizer_models)
        states.setdefault("optimizers", {})
        states["grad_scalers"] = {}
        states["meta"] = {"epoch": info.epoch, "step": info.step, "update": info.update, **dict(self.extra_meta)}
        self.logger.log_state_dict(states, "training_state")


def restore_training_state(
    *,
    resume: str,
    strategy: "FlaxDistributedStrategy",
    models: Mapping[str, nnx.Module],
    learner: Any,
    start_epoch: int,
    logger: Logger,
    optimizer_hashes: Mapping[str, str] | None = None,
    config_hash: str | None = None,
    is_main: bool = True,
) -> int:
    """Load the resumed state into the models and optimizers, and return the epoch to continue at.

    The saved epoch wins over *start_epoch*, as in the torch loader (`docs/adr/0005`). *optimizer_hashes*
    are the hashes of the optimizer patterns the current configuration rebuilt: optax builds `tx`
    from configuration and the state restore cannot see it, so a schedule swapped between save and
    resume would silently continue the new schedule from the old count. A mismatch warns rather than
    refuses -- extending a schedule or lowering the rate of a fine-tune is legitimate.

    Args:
        resume (str): The training state reference, in whatever form *logger* accepts.
        strategy (FlaxDistributedStrategy): The strategy placing the restored arrays.
        models (Mapping[str, nnx.Module]): The live models to restore into.
        learner (Any): The learner owning the optimizers to restore into.
        start_epoch (int): The epoch the command line asked for, reported when the state overrides it.
        logger (Logger): The logger the state is fetched through.
        optimizer_hashes (Mapping[str, str] | None): Hashes of the rebuilt optimizer patterns, by segment.
        config_hash (str | None): Digest of what this run trains, compared with the saved one.
        is_main (bool): Whether this process logs the override message.

    Returns:
        int: The epoch to continue at: the saved one plus one.
    """
    state = strategy.load_state_dict(
        models, learner.optimizers, learner.optimizer_models, logger.fetch_training_state(resume)
    )
    meta = state["meta"]
    saved_hashes = meta.get("optimizer_hashes", {})
    for segment, digest in (optimizer_hashes or {}).items():
        if segment in saved_hashes and saved_hashes[segment] != digest:
            _logger.warning(
                'The optimizer of segment "%s" is not the one the state was saved with: optax rebuilds '
                "it from the configuration, so the run continues with the new one from the saved step count.",
                segment,
            )
    if config_hash is not None and meta.get("config_hash", config_hash) != config_hash:
        _logger.warning(
            "The state was saved from a different model, learner or shape configuration: the arrays it holds "
            "are restored into whatever the current one built, wherever the two still line up."
        )
    resumed_epoch = int(meta["epoch"]) + 1
    if start_epoch != 1 and is_main:
        _logger.info("Ignoring --start-epoch %s: the resumed state continues at epoch %s.", start_epoch, resumed_epoch)
    return resumed_epoch


__all__ = [
    "FlaxBestCriterion",
    "FlaxTracker",
    "FlaxTrainer",
    "FlaxTrainingStateSaver",
    "ShardedDataset",
    "TensorInitializer",
    "create_jax_inputs",
    "get_jax_device",
    "get_jax_devices",
    "resolve_input_shapes",
    "restore_training_state",
]


if not TYPE_CHECKING:
    import sys

    from structcast.utils.lazy_import import LazySelectedImporter

    sys.modules[__name__] = LazySelectedImporter(__name__, globals())
