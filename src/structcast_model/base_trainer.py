"""Base trainer for training a model."""

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from logging import getLogger
from math import inf
from operator import gt, lt
from time import time
from typing import TYPE_CHECKING, Any, Generic, Literal, Protocol, TypeAlias, TypeVar, runtime_checkable

logger = getLogger(__name__)

ModelT_contra = TypeVar("ModelT_contra", contravariant=True)

DatasetLike: TypeAlias = Iterable[dict[str, Any]]
"""Dataset-like object."""


def get_dataset(dataset: DatasetLike | Callable[[], DatasetLike]) -> Iterable[dict[str, Any]]:
    """Get the dataset."""
    return dataset() if callable(dataset) else dataset


def get_dataset_size(dataset: DatasetLike | Callable[[], DatasetLike]) -> int:
    """Get the size of the dataset."""
    dataset = get_dataset(dataset)
    if hasattr(dataset, "__len__"):
        return dataset.__len__()
    return sum(1 for _ in dataset)


@runtime_checkable
class Learner(Protocol):
    """Protocol for the object that owns the models and defines how they learn.

    A learner decides when an update should happen, how a training step runs, and how an
    inference step runs.
    """

    @property
    def models(self) -> dict[str, Any]:
        """The models to train."""

    def update(self, step: int) -> bool:
        """Determine whether to update the model based on the current step and any internal state."""

    def training_step(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """Perform the training step for the given criteria."""

    def inference_step(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """Perform the inference step for the given criteria."""


@runtime_checkable
class DataProvider(Protocol):
    """Protocol supplying the datasets of a whole training run."""

    @property
    def training_dataset(self) -> DatasetLike | Callable[[], DatasetLike]:
        """The dataset used for training."""

    @property
    def validation_dataset(self) -> DatasetLike | Callable[[], DatasetLike] | None:
        """The dataset used for validation, or None to skip validation."""


@dataclass
class SimpleDataProvider:
    """Data provider holding an already-built training dataset and an optional validation dataset.

    Example:
        >>> provider = SimpleDataProvider([{"x": 1}])
        >>> provider.validation_dataset is None
        True
    """

    training_dataset: DatasetLike | Callable[[], DatasetLike]
    """The dataset used for training."""

    validation_dataset: DatasetLike | Callable[[], DatasetLike] | None = None
    """The dataset used for validation, or None to skip validation."""


@dataclass(kw_only=True)
class BaseInfo:
    """Base information for building a model."""

    step: int = 0
    """The current training step."""

    update: int = 0
    """The number of times the model has been updated."""

    epoch: int = 0
    """The current epoch."""

    history: dict[int, dict[str, Any]] = field(default_factory=dict)
    """History of training and validation logs."""

    def logs(self, epoch: int | None = None) -> dict[str, Any]:
        """Get the log for the given epoch."""
        if epoch is None:
            return self.history.setdefault(self.epoch, {})
        if epoch in self.history:
            return self.history[epoch]
        raise KeyError(f"No logs found for key: {epoch}.")


@runtime_checkable
class Callback(Protocol, Generic[ModelT_contra]):
    """Protocol for callbacks."""

    def __call__(self, info: BaseInfo, **models: ModelT_contra) -> None:
        """Call the callback with the given information."""


@runtime_checkable
class BestCallback(Protocol[ModelT_contra]):
    """Protocol for best criterion callback."""

    def __call__(self, info: BaseInfo, best: "BestCriterion", **models: ModelT_contra) -> None:
        """Call the callback with the given info, target criterion, and best value."""


def invoke_callback(
    callbacks: Sequence[Callable[..., None]],
    info: BaseInfo,
    *args: Any,
    **models: ModelT_contra,
) -> None:
    """Invoke callback."""
    for callback in callbacks:
        callback(info, *args, **models)


EVENTS: tuple[str, ...] = (
    "on_update",
    "on_training_begin",
    "on_training_end",
    "on_training_step_begin",
    "on_training_step_end",
    "on_validation_begin",
    "on_validation_end",
    "on_validation_step_begin",
    "on_validation_step_end",
    "on_epoch_begin",
    "on_epoch_end",
)
"""Names of the lifecycle events a trainer dispatches."""


@runtime_checkable
class OnUpdate(Protocol, Generic[ModelT_contra]):
    """Protocol for objects reacting after each update."""

    def on_update(self, info: "BaseInfo", **models: ModelT_contra) -> None:
        """React to the models having just been updated."""


@runtime_checkable
class OnTrainingBegin(Protocol, Generic[ModelT_contra]):
    """Protocol for objects reacting at the beginning of training."""

    def on_training_begin(self, info: "BaseInfo", **models: ModelT_contra) -> None:
        """React to training being about to start."""


@runtime_checkable
class OnTrainingEnd(Protocol, Generic[ModelT_contra]):
    """Protocol for objects reacting at the end of training."""

    def on_training_end(self, info: "BaseInfo", **models: ModelT_contra) -> None:
        """React to training having finished."""


@runtime_checkable
class OnTrainingStepBegin(Protocol, Generic[ModelT_contra]):
    """Protocol for objects reacting at the beginning of each training step."""

    def on_training_step_begin(self, info: "BaseInfo", **models: ModelT_contra) -> None:
        """React to a training step being about to start."""


@runtime_checkable
class OnTrainingStepEnd(Protocol, Generic[ModelT_contra]):
    """Protocol for objects reacting at the end of each training step."""

    def on_training_step_end(self, info: "BaseInfo", **models: ModelT_contra) -> None:
        """React to a training step having finished."""


@runtime_checkable
class OnValidationBegin(Protocol, Generic[ModelT_contra]):
    """Protocol for objects reacting at the beginning of validation."""

    def on_validation_begin(self, info: "BaseInfo", **models: ModelT_contra) -> None:
        """React to validation being about to start."""


@runtime_checkable
class OnValidationEnd(Protocol, Generic[ModelT_contra]):
    """Protocol for objects reacting at the end of validation."""

    def on_validation_end(self, info: "BaseInfo", **models: ModelT_contra) -> None:
        """React to validation having finished."""


@runtime_checkable
class OnValidationStepBegin(Protocol, Generic[ModelT_contra]):
    """Protocol for objects reacting at the beginning of each validation step."""

    def on_validation_step_begin(self, info: "BaseInfo", **models: ModelT_contra) -> None:
        """React to a validation step being about to start."""


@runtime_checkable
class OnValidationStepEnd(Protocol, Generic[ModelT_contra]):
    """Protocol for objects reacting at the end of each validation step."""

    def on_validation_step_end(self, info: "BaseInfo", **models: ModelT_contra) -> None:
        """React to a validation step having finished."""


@runtime_checkable
class OnEpochBegin(Protocol, Generic[ModelT_contra]):
    """Protocol for objects reacting at the beginning of each epoch."""

    def on_epoch_begin(self, info: "BaseInfo", **models: ModelT_contra) -> None:
        """React to an epoch being about to start."""


@runtime_checkable
class OnEpochEnd(Protocol, Generic[ModelT_contra]):
    """Protocol for objects reacting at the end of each epoch."""

    def on_epoch_end(self, info: "BaseInfo", **models: ModelT_contra) -> None:
        """React to an epoch having finished."""


EVENT_PROTOCOLS: Mapping[str, type] = {
    "on_update": OnUpdate,
    "on_training_begin": OnTrainingBegin,
    "on_training_end": OnTrainingEnd,
    "on_training_step_begin": OnTrainingStepBegin,
    "on_training_step_end": OnTrainingStepEnd,
    "on_validation_begin": OnValidationBegin,
    "on_validation_end": OnValidationEnd,
    "on_validation_step_begin": OnValidationStepBegin,
    "on_validation_step_end": OnValidationStepEnd,
    "on_epoch_begin": OnEpochBegin,
    "on_epoch_end": OnEpochEnd,
}
"""Event name to the protocol an object must implement to receive that event."""


@dataclass(kw_only=True)
class BaseTrainer(BaseInfo, Generic[ModelT_contra]):
    """Base trainer for training a model.

    Every participant given to the trainer -- the learner, its optimizers, the tracker, the data
    provider, and the explicit callbacks -- is scanned once at construction and routed into the
    lifecycle events whose protocol it implements.
    """

    learner: Learner
    """The learner owning the models and the step definitions."""

    tracker: Callable[..., dict[str, float]]
    """The tracker to log training and validation information."""

    data: DataProvider | None = None
    """The provider of the training and validation datasets."""

    callbacks: Sequence[Any] = ()
    """Objects routed into the events whose protocol they implement."""

    training_prefix: str = ""
    """ Prefix for training logs. """

    validation_prefix: str = "val_"
    """ Prefix for validation logs. """

    history: dict[int, dict[str, Any]] = field(default_factory=dict)
    """History of training and validation logs."""

    _events: dict[str, list[tuple[str, Callable[..., None]]]] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        """Route every participant into the events whose protocol it implements."""
        candidates: list[Any] = [self.learner]
        optimizers = getattr(self.learner, "optimizers", None)
        if isinstance(optimizers, Mapping):
            candidates.extend(optimizers.values())
        candidates.append(self.tracker)
        if self.data is not None:
            candidates.append(self.data)
        candidates.extend(self.callbacks)
        self._events = {event: [] for event in EVENTS}
        registered: dict[str, set[int]] = {event: set() for event in EVENTS}
        for candidate in candidates:
            for event, protocol in EVENT_PROTOCOLS.items():
                if isinstance(candidate, protocol) and id(candidate) not in registered[event]:
                    registered[event].add(id(candidate))
                    self._events[event].append((type(candidate).__name__, getattr(candidate, event)))

    def describe(self) -> dict[str, list[str]]:
        """Return a mapping of event name to registered callback display names.

        Returns:
            A dict keyed by event name (e.g. ``"on_epoch_end"``) whose values are
            lists of display names.  Events with no registered callbacks are omitted.
        """
        return {event: [name for name, _ in registered] for event, registered in self._events.items() if registered}

    def _dispatch(self, event: str, **models: Any) -> None:
        """Call every callback registered for *event* with this trainer and the models."""
        for _, callback in self._events[event]:
            callback(self, **models)

    def sync(self) -> None:
        """Synchronize the device if necessary. This is a no-op by default, but can be overridden by subclasses."""

    def update_models(self, __inputs__: Any) -> tuple[bool, dict[str, Any]]:
        """Perform a training step and update the models.

        Args:
            __inputs__ (Any): The inputs for the training step.

        Returns:
            tuple[bool, dict[str, Any]]: A tuple containing a boolean indicating whether the model was updated and
                a dictionary of criteria for tracking.
        """
        return self.learner.update(self.step), self.learner.training_step(**__inputs__)

    def train(self, dataset: DatasetLike | Callable[[], DatasetLike]) -> Mapping[str, Any]:
        """Train the model on the given dataset.

        Args:
            dataset (DatasetLike | Callable[[], DatasetLike]): The dataset to train on,
                which can be an iterable of input dictionaries or a callable that returns such an iterable.

        Returns:
            Mapping[str, Any]: The logs from training, which may include metrics and other information.
        """
        models = self.learner.models
        self._dispatch("on_training_begin", **models)
        elapsed_time = 0.0
        for index, inputs in enumerate(get_dataset(dataset), start=1):
            self.step += 1
            self._dispatch("on_training_step_begin", **models)
            elapsed_time -= time()
            updated, criteria = self.update_models(inputs)
            logs = self.tracker(**criteria)
            self.sync()
            elapsed_time += time()
            logs["elapsed_time"] = elapsed_time / index
            if self.training_prefix:
                logs = {f"{self.training_prefix}{k}": v for k, v in logs.items()}
            self.logs().update(logs)
            if updated:
                self.update += 1
                self._dispatch("on_update", **models)
            self._dispatch("on_training_step_end", **models)
        self._dispatch("on_training_end", **models)
        return logs

    def evaluate(self, dataset: DatasetLike | Callable[[], DatasetLike]) -> Mapping[str, Any]:
        """Evaluate the model on the given dataset.

        Args:
            dataset (DatasetLike | Callable[[], DatasetLike]): The dataset to evaluate on,
                which can be an iterable of input dictionaries or a callable that returns such an iterable.

        Returns:
            Mapping[str, Any]: The logs from evaluation, which may include metrics and other information.
        """
        models = self.learner.models
        self._dispatch("on_validation_begin", **models)
        elapsed_time = 0.0
        for index, data in enumerate(get_dataset(dataset), start=1):
            self._dispatch("on_validation_step_begin", **models)
            elapsed_time -= time()
            logs = self.tracker(**self.learner.inference_step(**data))
            self.sync()
            elapsed_time += time()
            logs["elapsed_time"] = elapsed_time / index
            if self.validation_prefix:
                logs = {f"{self.validation_prefix}{k}": v for k, v in logs.items()}
            self.logs().update(logs)
            self._dispatch("on_validation_step_end", **models)
        self._dispatch("on_validation_end", **models)
        return logs

    def fit(
        self,
        epochs: int,
        start_epoch: int = 1,
        validation_frequency: int = 1,
    ) -> dict[int, dict[str, Any]]:
        """Fit the model on the datasets of the data provider.

        Args:
            epochs (int): Number of epochs to train.
            start_epoch (int, optional): Epoch to start training from. Defaults to 1.
            validation_frequency (int, optional): Frequency of validation. Defaults to 1.

        Returns:
            History of training and validation logs.
        """
        if validation_frequency < 1:
            raise ValueError("Validation frequency must be at least 1.")
        if start_epoch < 1:
            raise ValueError(f"Start epoch must be at least 1: {start_epoch}")
        if start_epoch > epochs:
            raise ValueError(f"Start epoch must be less than or equal to epochs: {start_epoch} > {epochs}")
        if self.data is None:
            raise ValueError("No data provider was given to the trainer: fit() needs one, use train() instead.")
        training_dataset = self.data.training_dataset
        validation_dataset = self.data.validation_dataset
        models = self.learner.models
        for epoch in range(start_epoch, epochs + 1):
            self.epoch = epoch
            self._dispatch("on_epoch_begin", **models)
            self.train(training_dataset)
            if validation_dataset is not None and epoch % validation_frequency == 0:
                self.evaluate(validation_dataset)
            self._dispatch("on_epoch_end", **models)
        return self.history


@dataclass(kw_only=True, slots=True)
class BestCriterion(Generic[ModelT_contra]):
    """Callback to track the best criterion during training or validation."""

    target: str
    """The target criterion to monitor."""

    mode: Literal["min", "max"] = "min"
    """The mode to monitor the criterion. Either 'min' or 'max'."""

    on_best: list[BestCallback[ModelT_contra]] = field(default_factory=list)
    """Callbacks to be called when a new best criterion is found."""

    _step: int = field(default=0, repr=False)
    _best: float = field(init=False, repr=False)
    _compare: Callable[[float, float], bool] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Post initialization."""
        self._compare = lt if self.mode == "min" else gt
        self._best = inf if self.mode == "min" else -inf

    @property
    def step(self) -> int:
        """Get the step at which the best criterion was found."""
        return self._step

    @property
    def value(self) -> float:
        """Get the best criterion value found."""
        return self._best

    def on_epoch_end(self, info: BaseInfo, **models: ModelT_contra) -> None:
        """Check and update the best criterion."""
        current: float | None = info.logs().get(self.target, None)
        if current is not None:
            if self._compare(current, self._best):
                self._step = info.step
                self._best = current
            invoke_callback(self.on_best, info, self, **models)


def _format_criteria(info: BaseInfo) -> str:
    """Format the criteria of the current epoch as indented ``key: value`` lines.

    The learner's learning rates are prepended when *info* exposes a learner that reports them.
    """
    learner = getattr(info, "learner", None)
    learning_rates = getattr(learner, "learning_rates", None)
    values: dict[str, Any] = dict(learning_rates) if isinstance(learning_rates, Mapping) else {}
    values.update(info.logs())
    return "\n".join([f"epoch: {info.epoch}", *(f"  {key}: {value}" for key, value in values.items())])


class ProgressBar:
    """Callback showing training and validation progress on a ``tqdm`` bar."""

    def __init__(
        self,
        steps_per_epoch: int,
        validation_steps: int = 0,
        training_criteria: Sequence[str] = (),
        validation_criteria: Sequence[str] = (),
    ) -> None:
        """Create the progress bar.

        Args:
            steps_per_epoch: Number of training steps in one epoch.
            validation_steps: Number of validation steps in one epoch.
            training_criteria: Log keys shown next to the bar during training.
            validation_criteria: Log keys shown next to the bar during validation.
        """
        # tqdm is not a declared dependency of this package: importing it here keeps the module
        # importable without it, so a top-level import is not an option.
        from tqdm import tqdm  # noqa: PLC0415

        self.steps_per_epoch = steps_per_epoch
        self.validation_steps = validation_steps
        self.training_criteria = training_criteria
        self.validation_criteria = validation_criteria
        self.bar = tqdm(total=steps_per_epoch, unit="batch")

    def _set_postfix(self, info: BaseInfo, criteria: Sequence[str]) -> None:
        """Show the values of *criteria* that the current epoch has produced."""
        logs = info.logs()
        self.bar.set_postfix({key: logs[key] for key in criteria if key in logs})

    def on_training_begin(self, info: BaseInfo, **models: Any) -> None:
        """Restart the bar for the training steps of a new epoch."""
        self.bar.reset(total=self.steps_per_epoch)

    def on_training_step_end(self, info: BaseInfo, **models: Any) -> None:
        """Advance the bar by one training step."""
        self.bar.update(1)
        self._set_postfix(info, self.training_criteria)

    def on_training_end(self, info: BaseInfo, **models: Any) -> None:
        """Flush the bar after the last training step."""
        self.bar.refresh()

    def on_validation_begin(self, info: BaseInfo, **models: Any) -> None:
        """Restart the bar for the validation steps of the current epoch."""
        self.bar.reset(total=self.validation_steps)

    def on_validation_step_end(self, info: BaseInfo, **models: Any) -> None:
        """Advance the bar by one validation step."""
        self.bar.update(1)
        self._set_postfix(info, self.validation_criteria)

    def on_validation_end(self, info: BaseInfo, **models: Any) -> None:
        """Flush the bar after the last validation step."""
        self.bar.refresh()

    def on_epoch_end(self, info: BaseInfo, **models: Any) -> None:
        """Write the criteria of the finished epoch above the bar."""
        self.bar.write(_format_criteria(info))


class Printer:
    """Callback printing the criteria of each epoch, for environments without a terminal."""

    def on_epoch_end(self, info: BaseInfo, **models: Any) -> None:
        """Print the criteria of the finished epoch."""
        print(_format_criteria(info))


__all__ = [
    "EVENTS",
    "EVENT_PROTOCOLS",
    "BaseInfo",
    "BaseTrainer",
    "BestCallback",
    "BestCriterion",
    "Callback",
    "DataProvider",
    "DatasetLike",
    "Learner",
    "OnEpochBegin",
    "OnEpochEnd",
    "OnTrainingBegin",
    "OnTrainingEnd",
    "OnTrainingStepBegin",
    "OnTrainingStepEnd",
    "OnUpdate",
    "OnValidationBegin",
    "OnValidationEnd",
    "OnValidationStepBegin",
    "OnValidationStepEnd",
    "Printer",
    "ProgressBar",
    "SimpleDataProvider",
    "get_dataset",
    "get_dataset_size",
    "invoke_callback",
]


if not TYPE_CHECKING:
    import sys

    from structcast.utils.lazy_import import LazySelectedImporter

    sys.modules[__name__] = LazySelectedImporter(__name__, globals())
