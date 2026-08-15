"""Base trainer for training a model."""

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from logging import getLogger
from math import inf
from operator import gt, lt
from time import time
from typing import TYPE_CHECKING, Any, Generic, Literal, TypeAlias, TypeVar, cast

# Protocol and runtime_checkable come from typing_extensions so that isinstance checks use
# inspect.getattr_static on Python 3.11 as well (backported from 3.12): probing a protocol member
# must not execute a property getter, which for a data provider may build a real data loader.
from typing_extensions import Protocol, runtime_checkable

if TYPE_CHECKING:
    import tqdm
else:
    from structcast.utils.lazy_import import LazyModuleImporter

    tqdm = LazyModuleImporter("tqdm")

logger = getLogger(__name__)

ModelT = TypeVar("ModelT")

DatasetLike: TypeAlias = Iterable[dict[str, Any]]
"""Dataset-like object."""


def get_dataset(dataset: DatasetLike | Callable[[], DatasetLike]) -> Iterable[dict[str, Any]]:
    """Get the dataset."""
    return dataset() if callable(dataset) else dataset


def get_dataset_size(dataset: DatasetLike | Callable[[], DatasetLike]) -> int:
    """Get the size of the dataset.

    A ``__len__`` on the object itself wins, even for a callable dataset, so counting a loader
    wrapper never materializes an epoch of data. Iterating is the last resort and consumes a
    one-shot iterable.
    """
    if hasattr(dataset, "__len__"):
        return dataset.__len__()
    dataset = get_dataset(dataset)
    if hasattr(dataset, "__len__"):
        return dataset.__len__()
    return sum(1 for _ in dataset)


@runtime_checkable
class Learner(Protocol, Generic[ModelT]):
    """Protocol for the object that owns the models and defines how they learn.

    A learner decides when an update should happen, how a training step runs, and how an
    inference step runs.
    """

    @property
    def models(self) -> dict[str, ModelT]:
        """The models to train."""

    @property
    def optimizers(self) -> dict[str, Any]:
        """The optimizers by name; members implementing event protocols are routed by the trainer."""

    @property
    def learning_rates(self) -> dict[str, float]:
        """The current learning rate of each optimizer, for display and logging."""

    def update(self, step: int) -> bool:
        """Determine whether to update the model based on the current step and any internal state."""

    def training_step(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """Perform the training step for the given criteria."""

    def inference_step(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """Perform the inference step for the given criteria."""


@runtime_checkable
class DataProvider(Protocol):
    """Protocol supplying the datasets of a whole training run.

    Both dataset properties must return the same object on every read: the trainer reads them for
    the event-protocol scan and again in ``fit()``, so a getter that builds a fresh dataset per
    read would train a different object than the one receiving events.
    """

    @property
    def training_dataset(self) -> DatasetLike | Callable[[], DatasetLike]:
        """The dataset used for training."""

    @property
    def validation_dataset(self) -> DatasetLike | Callable[[], DatasetLike] | None:
        """The dataset used for validation, or None to skip validation."""

    @property
    def steps_per_epoch(self) -> int:
        """Number of training steps in one epoch."""

    @property
    def validation_steps(self) -> int:
        """Number of validation steps in one epoch, 0 when there is no validation dataset."""


@dataclass(kw_only=True, slots=True)
class SimpleDataProvider:
    """Data provider holding an already-built training dataset and an optional validation dataset.

    The step counts come from :func:`get_dataset_size`, computed on the first read and cached for
    the rest of the run. The first read of a dataset exposing no ``__len__`` counts it by
    iterating, which consumes a one-shot iterable.

    Example:
        >>> provider = SimpleDataProvider(training_dataset=[{"x": 1}])
        >>> provider.validation_dataset is None
        True
        >>> provider.steps_per_epoch, provider.validation_steps
        (1, 0)
    """

    training_dataset: DatasetLike | Callable[[], DatasetLike]
    """The dataset used for training."""

    validation_dataset: DatasetLike | Callable[[], DatasetLike] | None = None
    """The dataset used for validation, or None to skip validation."""

    _steps_per_epoch: int | None = field(default=None, init=False, repr=False)
    """Cache of steps_per_epoch, counted on the first read."""

    _validation_steps: int | None = field(default=None, init=False, repr=False)
    """Cache of validation_steps, counted on the first read."""

    @property
    def steps_per_epoch(self) -> int:
        """Number of training steps in one epoch, counted from the training dataset on the first read."""
        if self._steps_per_epoch is None:
            self._steps_per_epoch = get_dataset_size(self.training_dataset)
        return self._steps_per_epoch

    @property
    def validation_steps(self) -> int:
        """Number of validation steps in one epoch, counted on the first read, 0 without a validation dataset."""
        if self._validation_steps is None:
            self._validation_steps = 0 if self.validation_dataset is None else get_dataset_size(self.validation_dataset)
        return self._validation_steps


@dataclass(kw_only=True)
class BaseInfo(Generic[ModelT]):
    """Base information for building a model."""

    step: int = 0
    """The current training step."""

    update: int = 0
    """The number of times the model has been updated."""

    epoch: int = 0
    """The current epoch."""

    history: dict[int, dict[str, Any]] = field(default_factory=dict)
    """History of training and validation logs."""

    @property
    def models(self) -> dict[str, ModelT]:
        """The models by name; a bare info holds none, a trainer delegates to its learner."""
        return {}

    def logs(self, epoch: int | None = None) -> dict[str, Any]:
        """Get the log for the given epoch."""
        if epoch is None:
            return self.history.setdefault(self.epoch, {})
        if epoch in self.history:
            return self.history[epoch]
        raise KeyError(f"No logs found for key: {epoch}.")


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
class OnUpdate(Protocol, Generic[ModelT]):
    """Protocol for objects reacting after each update."""

    def on_update(self, info: "BaseInfo[ModelT]") -> None:
        """React to the models having just been updated."""


@runtime_checkable
class OnTrainingBegin(Protocol, Generic[ModelT]):
    """Protocol for objects reacting at the beginning of training."""

    def on_training_begin(self, info: "BaseInfo[ModelT]") -> None:
        """React to training being about to start."""


@runtime_checkable
class OnTrainingEnd(Protocol, Generic[ModelT]):
    """Protocol for objects reacting at the end of training."""

    def on_training_end(self, info: "BaseInfo[ModelT]") -> None:
        """React to training having finished."""


@runtime_checkable
class OnTrainingStepBegin(Protocol, Generic[ModelT]):
    """Protocol for objects reacting at the beginning of each training step."""

    def on_training_step_begin(self, info: "BaseInfo[ModelT]") -> None:
        """React to a training step being about to start."""


@runtime_checkable
class OnTrainingStepEnd(Protocol, Generic[ModelT]):
    """Protocol for objects reacting at the end of each training step."""

    def on_training_step_end(self, info: "BaseInfo[ModelT]") -> None:
        """React to a training step having finished."""


@runtime_checkable
class OnValidationBegin(Protocol, Generic[ModelT]):
    """Protocol for objects reacting at the beginning of validation."""

    def on_validation_begin(self, info: "BaseInfo[ModelT]") -> None:
        """React to validation being about to start."""


@runtime_checkable
class OnValidationEnd(Protocol, Generic[ModelT]):
    """Protocol for objects reacting at the end of validation."""

    def on_validation_end(self, info: "BaseInfo[ModelT]") -> None:
        """React to validation having finished."""


@runtime_checkable
class OnValidationStepBegin(Protocol, Generic[ModelT]):
    """Protocol for objects reacting at the beginning of each validation step."""

    def on_validation_step_begin(self, info: "BaseInfo[ModelT]") -> None:
        """React to a validation step being about to start."""


@runtime_checkable
class OnValidationStepEnd(Protocol, Generic[ModelT]):
    """Protocol for objects reacting at the end of each validation step."""

    def on_validation_step_end(self, info: "BaseInfo[ModelT]") -> None:
        """React to a validation step having finished."""


@runtime_checkable
class OnEpochBegin(Protocol, Generic[ModelT]):
    """Protocol for objects reacting at the beginning of each epoch."""

    def on_epoch_begin(self, info: "BaseInfo[ModelT]") -> None:
        """React to an epoch being about to start."""


@runtime_checkable
class OnEpochEnd(Protocol, Generic[ModelT]):
    """Protocol for objects reacting at the end of each epoch."""

    def on_epoch_end(self, info: "BaseInfo[ModelT]") -> None:
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
class BaseTrainer(BaseInfo[ModelT]):
    """Base trainer for training a model.

    Every participant given to the trainer -- the learner, its optimizers, the tracker, the data
    provider and its datasets, and the explicit callbacks -- is scanned once when first used (the
    first dispatched event) and routed into the lifecycle events whose protocol it implements.
    """

    learner: Learner[ModelT]
    """The learner owning the models and the step definitions."""

    tracker: Callable[..., dict[str, float]]
    """The tracker to log training and validation information."""

    data: DataProvider
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
        """Extension hook kept for subclasses; the participant scan runs lazily via ``_scan``."""

    @property
    def models(self) -> dict[str, ModelT]:
        """The learner's models, read on every access rather than snapshotted."""
        return self.learner.models

    def _routed_events(self) -> dict[str, list[tuple[str, Callable[..., None]]]]:
        """Return event name to participants, scanning the current candidates.

        The datasets join the scan so hooks such as a distributed sampler's set_epoch fire on
        every rank without explicit registration.
        """
        candidates: list[Any] = [
            self.learner,
            *self.learner.optimizers.values(),
            self.tracker,
            self.data,
            self.data.training_dataset,
            self.data.validation_dataset,
            *self.callbacks,
        ]
        events: dict[str, list[tuple[str, Callable[..., None]]]] = {event: [] for event in EVENTS}
        registered: dict[str, set[int]] = {event: set() for event in EVENTS}
        for candidate in candidates:
            for event, protocol in EVENT_PROTOCOLS.items():
                if isinstance(candidate, protocol) and id(candidate) not in registered[event]:
                    registered[event].add(id(candidate))
                    events[event].append((type(candidate).__name__, getattr(candidate, event)))
        return events

    def _scan(self) -> None:
        """Freeze the participant routing at the first dispatched event.

        Deferring past construction lets callbacks appended to the given sequence afterwards (the
        CLI builds its display callbacks from the constructed trainer's prefixes) take part; the
        dead-callback warning also fires here.
        """
        self._events = self._routed_events()
        # The learner/tracker/data participants legitimately may implement no event, but an entry of
        # the explicit callbacks sequence that matches nothing is almost certainly a typo'd hook name.
        for callback in self.callbacks:
            if not any(isinstance(callback, protocol) for protocol in EVENT_PROTOCOLS.values()):
                logger.warning(
                    f'Callback "{type(callback).__name__}" implements no event protocol '
                    f"({', '.join(EVENTS)}) and will never be called."
                )

    def describe(self) -> dict[str, list[str]]:
        """Return a mapping of event name to registered callback display names.

        Before the first dispatched event this previews the current participants without freezing
        the scan, so callbacks appended after a ``describe()`` call still take part.

        Returns:
            A dict keyed by event name (e.g. ``"on_epoch_end"``) whose values are
            lists of display names.  Events with no registered callbacks are omitted.
        """
        events = self._events or self._routed_events()
        return {event: [name for name, _ in registered] for event, registered in events.items() if registered}

    def _dispatch(self, event: str) -> None:
        """Call every callback registered for *event* with this trainer as the info."""
        if not self._events:
            self._scan()
        for _, callback in self._events[event]:
            callback(self)

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
        self._dispatch("on_training_begin")
        elapsed_time = 0.0
        for index, inputs in enumerate(get_dataset(dataset), start=1):
            self.step += 1
            self._dispatch("on_training_step_begin")
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
                self._dispatch("on_update")
            self._dispatch("on_training_step_end")
        self._dispatch("on_training_end")
        return logs

    def evaluate(self, dataset: DatasetLike | Callable[[], DatasetLike]) -> Mapping[str, Any]:
        """Evaluate the model on the given dataset.

        Args:
            dataset (DatasetLike | Callable[[], DatasetLike]): The dataset to evaluate on,
                which can be an iterable of input dictionaries or a callable that returns such an iterable.

        Returns:
            Mapping[str, Any]: The logs from evaluation, which may include metrics and other information.
        """
        self._dispatch("on_validation_begin")
        elapsed_time = 0.0
        for index, data in enumerate(get_dataset(dataset), start=1):
            self._dispatch("on_validation_step_begin")
            elapsed_time -= time()
            logs = self.tracker(**self.learner.inference_step(**data))
            self.sync()
            elapsed_time += time()
            logs["elapsed_time"] = elapsed_time / index
            if self.validation_prefix:
                logs = {f"{self.validation_prefix}{k}": v for k, v in logs.items()}
            self.logs().update(logs)
            self._dispatch("on_validation_step_end")
        self._dispatch("on_validation_end")
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
        training_dataset = self.data.training_dataset
        validation_dataset = self.data.validation_dataset
        for epoch in range(start_epoch, epochs + 1):
            self.epoch = epoch
            self._dispatch("on_epoch_begin")
            self.train(training_dataset)
            if validation_dataset is not None and epoch % validation_frequency == 0:
                self.evaluate(validation_dataset)
            self._dispatch("on_epoch_end")
        return self.history


@runtime_checkable
class OnBest(Protocol, Generic[ModelT]):
    """Protocol for participants notified after a monitored criterion has been checked."""

    def on_best(self, info: BaseInfo[ModelT], best: "BestCriterion[ModelT]") -> None:
        """React to the check of *best* for the current epoch."""


@dataclass(kw_only=True, slots=True)
class BestCriterion(Generic[ModelT]):
    """Callback to track the best criterion during training or validation."""

    target: str
    """The target criterion to monitor."""

    mode: Literal["min", "max"] = "min"
    """The mode to monitor the criterion. Either 'min' or 'max'."""

    callbacks: list[OnBest[ModelT]] = field(default_factory=list)
    """Participants notified whenever the target was produced, the way a trainer routes events:
    each implements the ``OnBest`` protocol and receives this criterion alongside the info.

    Named ``callbacks`` (not ``on_best``) so the field cannot shadow the protocol method name in
    an isinstance check."""

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

    def on_epoch_end(self, info: BaseInfo[ModelT]) -> None:
        """Check and update the best criterion."""
        current: float | None = info.logs().get(self.target, None)
        if current is not None:
            if self._compare(current, self._best):
                self._step = info.step
                self._best = current
            for callback in self.callbacks:
                callback.on_best(info, self)


def _format_criteria(info: BaseInfo) -> str:
    """Format the criteria of the current epoch as indented ``key: value`` lines.

    Trainers dispatch themselves as *info*, so the learner's learning rates are read directly.
    """
    values: dict[str, Any] = dict(cast("BaseTrainer[Any]", info).learner.learning_rates)
    values.update(info.logs())
    return "\n".join([f"epoch: {info.epoch}", *(f"  {key}: {value}" for key, value in values.items())])


@dataclass(kw_only=True, slots=True)
class ProgressBar:
    """Callback showing training and validation progress on a ``tqdm`` bar."""

    steps_per_epoch: int
    """Number of training steps in one epoch."""

    validation_steps: int = 0
    """Number of validation steps in one epoch."""

    training_criteria: Sequence[str] = ()
    """Log keys shown next to the bar during training."""

    validation_criteria: Sequence[str] = ()
    """Log keys shown next to the bar during validation."""

    bar: "tqdm.tqdm" = field(init=False, repr=False)
    """The underlying bar, created at construction."""

    def __post_init__(self) -> None:
        """Create the bar sized to one training epoch."""
        self.bar = tqdm.tqdm(total=self.steps_per_epoch, unit="batch")

    def _set_postfix(self, info: BaseInfo, criteria: Sequence[str]) -> None:
        """Show the values of *criteria* that the current epoch has produced."""
        logs = info.logs()
        self.bar.set_postfix({key: logs[key] for key in criteria if key in logs})

    def on_training_begin(self, info: BaseInfo) -> None:
        """Restart the bar for the training steps of a new epoch."""
        self.bar.reset(total=self.steps_per_epoch)

    def on_training_step_end(self, info: BaseInfo) -> None:
        """Advance the bar by one training step."""
        self.bar.update(1)
        self._set_postfix(info, self.training_criteria)

    def on_training_end(self, info: BaseInfo) -> None:
        """Flush the bar after the last training step."""
        self.bar.refresh()

    def on_validation_begin(self, info: BaseInfo) -> None:
        """Restart the bar for the validation steps of the current epoch."""
        self.bar.reset(total=self.validation_steps)

    def on_validation_step_end(self, info: BaseInfo) -> None:
        """Advance the bar by one validation step."""
        self.bar.update(1)
        self._set_postfix(info, self.validation_criteria)

    def on_validation_end(self, info: BaseInfo) -> None:
        """Flush the bar after the last validation step."""
        self.bar.refresh()

    def on_epoch_end(self, info: BaseInfo) -> None:
        """Write the criteria of the finished epoch above the bar."""
        self.bar.write(_format_criteria(info))


@dataclass(kw_only=True, slots=True)
class Printer:
    """Callback printing the criteria of each epoch, for environments without a terminal."""

    def on_epoch_end(self, info: BaseInfo) -> None:
        """Print the criteria of the finished epoch."""
        print(_format_criteria(info))


__all__ = [
    "EVENTS",
    "EVENT_PROTOCOLS",
    "BaseInfo",
    "BaseTrainer",
    "BestCriterion",
    "DataProvider",
    "DatasetLike",
    "Learner",
    "OnBest",
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
]


if not TYPE_CHECKING:
    import sys

    from structcast.utils.lazy_import import LazySelectedImporter

    sys.modules[__name__] = LazySelectedImporter(__name__, globals())
