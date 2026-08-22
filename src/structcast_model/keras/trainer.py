"""Trainer helpers for Keras models."""

from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from logging import getLogger
from typing import TYPE_CHECKING, Any, Self, cast

import ml_dtypes
import numpy as np
from pydantic import TypeAdapter, ValidationError

# Protocol and runtime_checkable come from typing_extensions so that isinstance checks use
# inspect.getattr_static on Python 3.11 as well (backported from 3.12), as in base_trainer.
from typing_extensions import Protocol, runtime_checkable

import keras
from structcast_model.base_trainer import BaseInfo, BaseTrainer, BestCriterion
from structcast_model.builders.schema import TensorSpec, TensorSpecTree
from structcast_model.keras.distributed import KerasDistributedStrategy
from structcast_model.loggers.base import Logger
from structcast_model.utils.base import resolve_input_shapes, resolve_tensor_initializer

# `_logger`, not the usual `logger`: `restore_training_state` takes a `Logger` parameter named
# `logger`, as in the flax twin.
_logger = getLogger(__name__)

DTYPES = {
    "float32": np.float32,
    "float16": np.float16,
    "bfloat16": ml_dtypes.bfloat16,
    "int32": np.int32,
    "int64": np.int64,
}
"""NumPy element types of the supported tensor element types.

NumPy has no native `bfloat16`, so the type registered by `ml_dtypes`, a hard dependency of Keras, is used.
"""


@runtime_checkable
class TensorInitializer(Protocol):
    """Callable creating a dummy NumPy array, called as `initializer(size, dtype=...)`."""

    def __call__(self, size: tuple[int, ...], *, dtype: Any) -> Any:
        """Create an array of the given size and element type."""
        ...


def random_array(size: tuple[int, ...], *, dtype: Any) -> Any:
    """Create a uniformly distributed random NumPy array, the default initializer for floating point types.

    `numpy.random.rand` cannot be used as an initializer directly,
    since it takes the size as variadic arguments and has no `dtype` argument.

    Args:
        size (tuple[int, ...]): The size of the array, including the batch dimension.
        dtype (Any): The element type of the array.

    Returns:
        Any: The created array.
    """
    return np.random.random(size).astype(dtype)


def create_numpy_inputs(shape: Any, *, batch_size: int = 1) -> Any:
    """Create dummy NumPy inputs based on the provided shape.

    Args:
        shape (Any): The shape of the inputs to create. This can be a tensor specification,
            which is a tuple of integers or a mapping with the `_SHAPE_` key,
            a dictionary of shapes, or a list of shapes.
        batch_size (int): The batch size to use for the inputs.
            This will be prepended to the shape of every tensor specification.

    Returns:
        Any: The created inputs, which can be a NumPy array, a dictionary of arrays, or a list of arrays.

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
            int_default=np.zeros,
            protocol=TensorInitializer,
        )
        return initializer((batch_size, *node.SHAPE), dtype=DTYPES[node.DTYPE])
    if isinstance(node, Mapping):
        return {k: create_numpy_inputs(v, batch_size=batch_size) for k, v in node.items()}
    return [create_numpy_inputs(v, batch_size=batch_size) for v in node]


def create_keras_inputs(shape: Any, *, batch_size: int | None = None, name: str | None = None, **kwargs: Any) -> Any:
    """Create symbolic Keras inputs based on the provided shape.

    Args:
        shape (Any): The shape of the inputs to create. This can be a tensor specification,
            which is a tuple of integers or a mapping with the `_SHAPE_` key,
            a dictionary of shapes, or a list of shapes.
        batch_size (int | None): The optional batch size to bind to the symbolic inputs.
        name (str | None): The optional base name to use for the input tensors.
        **kwargs (Any): Additional keyword arguments to pass to keras.Input,
            taking precedence over the element type of the tensor specification.

    Returns:
        Any: The created inputs, which can be a Keras tensor, a dictionary of tensors, or a list of tensors.

    Raises:
        ValueError: If the shape is neither a tensor specification nor a dictionary or list nesting more of them.
    """
    try:
        node: TensorSpecTree = TypeAdapter(TensorSpecTree).validate_python(shape)
    except ValidationError:
        raise ValueError(f"Invalid tensor shape: {shape}") from None
    if isinstance(node, TensorSpec):
        kwargs.setdefault("dtype", node.DTYPE)
        return keras.Input(shape=node.SHAPE, batch_size=batch_size, name=name, **kwargs)
    if isinstance(node, Mapping):
        return {
            k: create_keras_inputs(v, batch_size=batch_size, name=k if name is None else f"{name}_{k}", **kwargs)
            for k, v in node.items()
        }
    return [
        create_keras_inputs(v, batch_size=batch_size, name=f"{name or 'input'}_{i}", **kwargs)
        for i, v in enumerate(node)
    ]


def initial_model(model: Any, shapes: Any) -> Any:
    """Initialize a Keras model by tracing it with symbolic inputs.

    Args:
        model: The model or layer to initialize.
        shapes: A dictionary mapping input names to their tensor shapes.
            If empty or None, the shapes declared by the model itself are used.

    Returns:
        A built Keras model.

    Raises:
        ValueError: If shapes are neither provided nor declared by a non-Model callable.
    """
    if isinstance(model, keras.Model):
        return model
    shapes = resolve_input_shapes(model, shapes)
    if shapes is None:
        raise ValueError("Input shapes are required to initialize a Keras layer into a model.")
    inputs = create_keras_inputs(shapes)
    if isinstance(inputs, Mapping):
        outputs = model(**inputs)
    elif isinstance(inputs, (tuple, list)):
        outputs = model(*inputs)
    else:
        outputs = model(inputs)
    return keras.Model(inputs=inputs, outputs=outputs)


@dataclass(kw_only=True, slots=True)
class KerasTracker:
    """Running mean of the criteria of one training or validation split.

    The sums stay device-side tensors, so accumulating a step is one backend `add` on whichever
    backend `keras.ops` dispatches to. The means come back as Python floats, not tensors:
    `BaseTrainer.tracker` is typed `Callable[..., dict[str, float]]` and everything downstream --
    the epoch history, the `BestCriterion` comparison against a float infinity, and the loggers'
    `log_metric(value: float)` -- consumes that contract, as the torch tracker's `.item()` does.
    `keras.ops.convert_to_numpy` is the backend-neutral host read, and it lands inside the region
    `BaseTrainer.train` times, where the torch loop blocks on `torch.cuda.synchronize()` for the
    same reason.

    Unlike the torch tracker there is no all-reduce: a distributed step's criteria are reduced
    before they get here — by the distributed strategy's step wrapper on the tensorflow and torch
    backends, and by XLA's global reduction of sharded arrays on jax — so what reaches this tracker
    is already one value (`docs/adr/0016`).

    Example:
        >>> import keras
        >>> from structcast_model.keras.trainer import KerasTracker
        >>> tracker = KerasTracker.from_criteria(["loss"])
        >>> tracker(loss=keras.ops.convert_to_tensor(1.0))
        {'loss': 1.0}
        >>> tracker(loss=keras.ops.convert_to_tensor(3.0))
        {'loss': 2.0}
        >>> tracker.reset()
        >>> tracker.logs()
        {}
    """

    criteria: tuple[str, ...]
    """Names of the criteria to track; a step must report every one of them."""

    sums: dict[str, Any] = field(init=False)
    """Device-side sum of each criterion over the steps seen since the last reset."""

    count: int = field(default=0, init=False)
    """Number of steps summed since the last reset."""

    def __post_init__(self) -> None:
        """Start every criterion at a zero sum."""
        self.reset()

    def reset(self) -> None:
        """Zero the sums and the step count."""
        # float32 whatever the criteria are: a run under a float16 or bfloat16 policy would
        # otherwise accumulate an epoch of steps in the reduced type and lose the small ones.
        self.sums = {criterion: keras.ops.zeros((), dtype="float32") for criterion in self.criteria}
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
            # The torch backend hands criteria still attached to the autograd graph; detaching
            # keeps the epoch sum from retaining every step's graph and lets `logs` reach numpy.
            value = keras.ops.stop_gradient(criteria[criterion])
            self.sums[criterion] = keras.ops.add(self.sums[criterion], value)
        self.count += 1
        return self.logs()

    def logs(self) -> dict[str, float]:
        """Return the running mean of each criterion, or an empty mapping before the first step."""
        if not self.count:
            return {}
        return {name: float(keras.ops.convert_to_numpy(total)) / self.count for name, total in self.sums.items()}

    @classmethod
    def from_criteria(cls, outputs: Iterable[str]) -> "KerasTracker":
        """Create a tracker for the named criteria, mirroring `TorchTracker.from_criteria`."""
        return cls(criteria=tuple(outputs))


@dataclass(kw_only=True)
class KerasTrainer(BaseTrainer[Any]):
    """Trainer for Keras models.

    `BaseTrainer.sync` stays the inherited no-op: `KerasTracker.logs` converts the criteria to
    NumPy on every step, which already waits for the step's computation on every backend, and the
    device a run uses is chosen by the backend adapter, not by a trainer field.

    The model type stays `Any`: keras ships no py.typed, so `keras.Model` is `Any` to a type
    checker anyway.
    """


def restore_training_state(
    *,
    resume: str,
    strategy: KerasDistributedStrategy,
    models: Mapping[str, Any],
    learner: Any,
    start_epoch: int,
    logger: Logger,
    optimizer_hashes: Mapping[str, str] | None = None,
    config_hash: str | None = None,
    is_main: bool = True,
) -> int:
    """Load the resumed state into the models and optimizers, and return the epoch to continue at.

    The saved epoch wins over *start_epoch*, as in the torch and Flax loaders (`docs/adr/0005`).
    Three things are checked against the state, and only one of them refuses the run:

    - The Keras backend it was written on. Normalization statistics and RNG trajectories are not
      verified equivalent across backends, so a state from another one is refused rather than
      silently continued (`docs/adr/0016`). It is checked before anything is assigned, so a refused
      resume leaves the freshly built run untouched.
    - The optimizer patterns, which the learner rebuilt from configuration: a schedule swapped
      between save and resume would continue the new schedule from the old step count. That warns
      rather than refuses -- extending a schedule or lowering a fine-tune's rate is legitimate.
    - The configuration digest of what the run trains, which warns for the same reason.

    Args:
        resume (str): The training state reference, in whatever form *logger* accepts.
        strategy (KerasDistributedStrategy): The strategy loading the state into the live variables.
        models (Mapping[str, Any]): The live models to restore into.
        learner (Any): The learner owning the optimizers to restore into.
        start_epoch (int): The epoch the command line asked for, reported when the state overrides it.
        logger (Logger): The logger the state is fetched through.
        optimizer_hashes (Mapping[str, str] | None): Hashes of the rebuilt optimizer patterns, by segment.
        config_hash (str | None): Digest of what this run trains, compared with the saved one.
        is_main (bool): Whether this process logs the override message.

    Returns:
        int: The epoch to continue at: the saved one plus one.

    Raises:
        ValueError: If the state was saved on another Keras backend.
    """
    saved = logger.fetch_training_state(resume)
    backend = keras.backend.backend()
    written = (saved or {}).get("meta", {}).get("backend")
    if written is not None and written != backend:
        raise ValueError(
            f'The training state was saved on the "{written}" Keras backend and this run is on "{backend}". '
            "Normalization statistics and RNG trajectories are not verified equivalent across the Keras "
            f"backends, so the run would silently continue from something else: resume it with --backend "
            f"{written}, or start it from scratch."
        )
    state = strategy.load_state_dict(models, learner.optimizers, learner.optimizer_models, saved)
    meta = state["meta"]
    saved_hashes = meta.get("optimizer_hashes", {})
    for segment, digest in (optimizer_hashes or {}).items():
        if segment in saved_hashes and saved_hashes[segment] != digest:
            _logger.warning(
                'The optimizer of segment "%s" is not the one the state was saved with: the learner '
                "rebuilds it from the configuration, so the run continues with the new one from the saved "
                "step count.",
                segment,
            )
    if config_hash is not None and meta.get("config_hash", config_hash) != config_hash:
        _logger.warning(
            "The state was saved from a different model, learner or shape configuration: the arrays it holds "
            "are restored into whatever the current one built, wherever the two still line up."
        )
    # Seed the learner's counters from the meta, so the step, update and accumulation clocks
    # continue where the saved run left off (docs/adr/0018).
    learner.restore_counters(int(meta["step"]), int(meta["update"]))
    resumed_epoch = int(meta["epoch"]) + 1
    if start_epoch != 1 and is_main:
        _logger.info("Ignoring --start-epoch %s: the resumed state continues at epoch %s.", start_epoch, resumed_epoch)
    return resumed_epoch


@dataclass(kw_only=True, slots=True)
class KerasBestCriterion(BestCriterion[Any]):
    """A callback tracking the best value of a criterion during training or validation for Keras models.

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
        strategy: KerasDistributedStrategy,
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

    strategy: KerasDistributedStrategy
    """The strategy producing the model states to save."""

    def on_best(self, info: BaseInfo[Any], best: BestCriterion[Any]) -> None:
        """Log the best value, and save the states of the info's models when this epoch reached it."""
        name = f"best_{best.target}"
        self.logger.log_metric(name, best.value, step=info.epoch)
        if self.save and info.step == best.step:
            # The models alone, as the torch and flax twins save: best-value weights are for
            # inference, so they carry no optimizer state, no counters and no wrapper key.
            self.logger.log_state_dict(self.strategy.state_dict(dict(info.models))["models"], name)


@dataclass(kw_only=True, slots=True)
class KerasTrainingStateSaver:
    """Callback saving models, optimizers and loop counters through a logger.

    The twin of `structcast_model.torch.trainer.TrainingStateSaver`, minus the gradient scalers:
    Keras loss scaling lives inside the optimizer, so its state is already in the optimizer's
    variables, and the payload keeps their (always empty) slot so every framework resumes from the
    same shape (`docs/adr/0015`).
    """

    logger: Logger
    """The logger the training-state artifacts are written through."""

    strategy: KerasDistributedStrategy
    """The strategy producing the model and optimizer states to save."""

    extra_meta: Mapping[str, Any] = field(default_factory=dict)
    """Run facts the loop does not know, recorded next to the counters, e.g. the run seed."""

    def on_epoch_end(self, info: BaseInfo[Any]) -> None:
        """Save the full training state of the finished epoch, so a run can be resumed from it."""
        learner = cast("KerasTrainer", info).learner
        states = self.strategy.state_dict(dict(info.models), learner.optimizers, learner.optimizer_models)
        states["grad_scalers"] = {}
        states["meta"] = {
            "epoch": info.epoch,
            "step": info.step,
            "update": info.update,
            # Load-bearing: normalization statistics and RNG trajectories are not verified
            # equivalent across the Keras backends, so a resume refuses a mismatch (`docs/adr/0016`).
            "backend": keras.backend.backend(),
            **dict(self.extra_meta),
        }
        self.logger.log_state_dict(states, "training_state")


__all__ = [
    "KerasBestCriterion",
    "KerasTracker",
    "KerasTrainer",
    "KerasTrainingStateSaver",
    "TensorInitializer",
    "create_keras_inputs",
    "create_numpy_inputs",
    "initial_model",
    "resolve_input_shapes",
    "restore_training_state",
]


if not TYPE_CHECKING:
    import sys

    from structcast.utils.lazy_import import LazySelectedImporter

    sys.modules[__name__] = LazySelectedImporter(__name__, globals())
