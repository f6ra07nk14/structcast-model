"""Base trainer for training a model."""

from collections.abc import Callable, Generator, Iterable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from logging import getLogger
from math import inf
from operator import gt, lt
from time import time
from typing import TYPE_CHECKING, Any, Generic, Literal, Protocol, TypeAlias, TypeVar

logger = getLogger(__name__)

ModelT_contra = TypeVar("ModelT_contra", contravariant=True)
_NCL_T = TypeVar("_NCL_T")

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


class Forward(Protocol[ModelT_contra]):
    """Protocol for forward pass configuration."""

    def __call__(self, inputs: Any, **models: ModelT_contra) -> dict[str, Any]:
        """Perform the forward pass for the given inputs and return the outputs and any additional information."""


class Backward(Protocol):
    """Protocol for backward pass configuration."""

    def update(self, step: int) -> bool:
        """Determine whether to update the model based on the current step and any internal state."""

    def __call__(self, *args: Any, **kwargs: Any) -> None:
        """Perform the backward pass for the given losses."""


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


class Callback(Protocol, Generic[ModelT_contra]):
    """Protocol for callbacks."""

    def __call__(self, info: BaseInfo, **models: ModelT_contra) -> None:
        """Call the callback with the given information."""


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


_CALLBACK_ATTRS: tuple[str, ...] = (
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


def _callback_name(callback: Any) -> str:
    """Derive a display name from a callback object."""
    name = getattr(callback, "__name__", None)
    if name is None:
        func = getattr(callback, "__func__", None)
        name = getattr(func, "__name__", None)
    return name or type(callback).__name__


class NamedCallbackList(list[_NCL_T]):
    """A generic list of callbacks that supports named registration for clearer introspection.

    Callbacks appended via :meth:`append` or :meth:`extend` receive an auto-generated
    display name inferred from the callable.  Use :meth:`register` to supply an
    explicit human-readable name.

    Example:
        >>> from typing import Any
        >>> ncl: NamedCallbackList[Any] = NamedCallbackList()
        >>> ncl.register("log_metrics", lambda i, **kw: None)
        >>> ncl.names()
        ['log_metrics']
    """

    _names: list[str]

    def __init__(self) -> None:
        """Initialize an empty list with an accompanying name registry."""
        super().__init__()
        self._names = []

    def __class_getitem__(cls, item: Any) -> type:  # type: ignore[override]
        """Support generic-alias syntax ``NamedCallbackList[X]`` in annotations."""
        return cls

    def append(self, callback: _NCL_T) -> None:  # type: ignore[override]
        """Append *callback*, using its inferred name as the display name."""
        super().append(callback)
        self._names.append(_callback_name(callback))

    def extend(self, callbacks: Iterable[_NCL_T]) -> None:  # type: ignore[override]
        """Extend the list, deriving display names automatically."""
        for cb in callbacks:
            self.append(cb)

    def register(self, name: str, callback: _NCL_T) -> None:
        """Register *callback* with an explicit display *name*.

        Args:
            name: Human-readable label shown when describing registered callbacks.
            callback: The callable to register.
        """
        super().append(callback)
        self._names.append(name)

    def clear(self) -> None:
        """Clear all callbacks and their display names."""
        super().clear()
        self._names.clear()

    def names(self) -> list[str]:
        """Return the display names of all registered callbacks.

        Returns:
            A new list containing the display name of each registered callback,
            in registration order.
        """
        return list(self._names)


@dataclass(kw_only=True)
class Callbacks(Generic[ModelT_contra]):
    """Callbacks."""

    on_update: NamedCallbackList[Callback[ModelT_contra]] = field(default_factory=NamedCallbackList)
    """Callbacks to call after each update."""

    on_training_begin: NamedCallbackList[Callback[ModelT_contra]] = field(default_factory=NamedCallbackList)
    """Callbacks to call at the beginning of training."""

    on_training_end: NamedCallbackList[Callback[ModelT_contra]] = field(default_factory=NamedCallbackList)
    """Callbacks to call at the end of training."""

    on_training_step_begin: NamedCallbackList[Callback[ModelT_contra]] = field(default_factory=NamedCallbackList)
    """Callbacks to be called at the beginning of each training step."""

    on_training_step_end: NamedCallbackList[Callback[ModelT_contra]] = field(default_factory=NamedCallbackList)
    """Callbacks to be called at the end of each training step."""

    on_validation_begin: NamedCallbackList[Callback[ModelT_contra]] = field(default_factory=NamedCallbackList)
    """Callbacks to be called at the beginning of validation."""

    on_validation_end: NamedCallbackList[Callback[ModelT_contra]] = field(default_factory=NamedCallbackList)
    """Callbacks to be called at the end of validation."""

    on_validation_step_begin: NamedCallbackList[Callback[ModelT_contra]] = field(default_factory=NamedCallbackList)
    """Callbacks to be called at the beginning of each validation step."""

    on_validation_step_end: NamedCallbackList[Callback[ModelT_contra]] = field(default_factory=NamedCallbackList)
    """Callbacks to be called at the end of each validation step."""

    on_epoch_begin: NamedCallbackList[Callback[ModelT_contra]] = field(default_factory=NamedCallbackList)
    """Callbacks to be called at the beginning of each epoch."""

    on_epoch_end: NamedCallbackList[Callback[ModelT_contra]] = field(default_factory=NamedCallbackList)
    """Callbacks to be called at the end of each epoch."""

    add_global_callbacks: bool = True
    """Whether to add global callbacks."""

    def __post_init__(self) -> None:
        """Post initialization."""
        if self.add_global_callbacks:
            for attr in _CALLBACK_ATTRS:
                src: NamedCallbackList = getattr(GLOBAL_CALLBACKS, attr)
                dst: NamedCallbackList = getattr(self, attr)
                for name, cb in zip(src._names, src, strict=True):
                    dst.register(name, cb)

    def clear(self) -> None:
        """Reset all callback lists to empty."""
        for attr in _CALLBACK_ATTRS:
            getattr(self, attr).clear()

    def describe(self) -> dict[str, list[str]]:
        """Return a mapping of event name to registered callback display names.

        Returns:
            A dict keyed by event name (e.g. ``"on_epoch_end"``) whose values are
            lists of display names.  Events with no registered callbacks are omitted.
        """
        return {attr: getattr(self, attr).names() for attr in _CALLBACK_ATTRS if getattr(self, attr)}


GLOBAL_CALLBACKS = Callbacks[Any](add_global_callbacks=False)
"""Global callbacks."""


@contextmanager
def callbacks_session() -> Generator[None, None, None]:
    """Context manager that clears GLOBAL_CALLBACKS on entry and exit.

    Use this to scope callback registrations to a single training session,
    preventing accumulation of stale callbacks across multiple runs.

    Example:
        >>> with callbacks_session():
        ...     # register callbacks and run training
        ...     pass
    """
    GLOBAL_CALLBACKS.clear()
    try:
        yield
    finally:
        GLOBAL_CALLBACKS.clear()


class InferenceWrapper(Protocol[ModelT_contra]):
    """Protocol for inference wrapper."""

    def __call__(self, info: BaseInfo, **models: ModelT_contra) -> dict[str, Any]:
        """Wrap the model for inference, e.g., for quantization or ONNX export."""


@dataclass(kw_only=True)
class BaseTrainer(BaseInfo, Callbacks[ModelT_contra]):
    """Base trainer for training a model."""

    training_step: Forward[ModelT_contra]
    """The forward pass configuration for training."""

    backward: Backward
    """The backward pass configuration."""

    tracker: Callable[..., dict[str, float]]
    """The tracker to log training and validation information."""

    inference_wrapper: InferenceWrapper[ModelT_contra] | None = None
    """An optional wrapper to apply to the model during inference, e.g., for quantization or ONNX export."""

    validation_step: Forward[ModelT_contra] | None = None
    """The forward pass configuration for validation."""

    training_prefix: str = ""
    """ Prefix for training logs. """

    validation_prefix: str = "val_"
    """ Prefix for validation logs. """

    history: dict[int, dict[str, Any]] = field(default_factory=dict)
    """History of training and validation logs."""

    def sync(self) -> None:
        """Synchronize the device if necessary. This is a no-op by default, but can be overridden by subclasses."""

    def update_models(self, __inputs__: Any, **models: ModelT_contra) -> tuple[bool, dict[str, Any]]:
        """Perform a training step and update the models.

        Args:
            __inputs__ (Any): The inputs for the training step.
            **models (ModelT): The models to update.

        Returns:
            tuple[bool, dict[str, Any]]: A tuple containing a boolean indicating whether the model was updated and
                a dictionary of criteria for tracking.
        """
        criteria = self.training_step(__inputs__, **models)
        updated = self.backward.update(self.step)
        self.backward(**criteria)
        return updated, criteria

    def train(self, dataset: DatasetLike | Callable[[], DatasetLike], **models: ModelT_contra) -> Mapping[str, Any]:
        """Train the model on the given dataset.

        Args:
            dataset (DatasetLike | Callable[[], DatasetLike]): The dataset to train on,
                which can be an iterable of input dictionaries or a callable that returns such an iterable.
            **models (ModelT): The models to train.

        Returns:
            Mapping[str, Any]: The logs from training, which may include metrics and other information.
        """
        invoke_callback(self.on_training_begin, self, **models)
        elapsed_time = 0.0
        for index, inputs in enumerate(get_dataset(dataset), start=1):
            self.step += 1
            invoke_callback(self.on_training_step_begin, self, **models)
            elapsed_time -= time()
            updated, criteria = self.update_models(inputs, **models)
            self.sync()
            elapsed_time += time()
            logs = self.tracker(**criteria) | {"elapsed_time": elapsed_time / index}
            if self.training_prefix:
                logs = {f"{self.training_prefix}{k}": v for k, v in logs.items()}
            self.logs().update(logs)
            if updated:
                self.update += 1
                invoke_callback(self.on_update, self, **models)
            invoke_callback(self.on_training_step_end, self, **models)
        invoke_callback(self.on_training_end, self, **models)
        return logs

    def evaluate(self, dataset: DatasetLike | Callable[[], DatasetLike], **models: ModelT_contra) -> Mapping[str, Any]:
        """Evaluate the model on the given dataset.

        Args:
            dataset (DatasetLike | Callable[[], DatasetLike]): The dataset to evaluate on,
                which can be an iterable of input dictionaries or a callable that returns such an iterable.
            **models (ModelT): The models to evaluate.

        Returns:
            Mapping[str, Any]: The logs from evaluation, which may include metrics and other information.
        """
        if self.validation_step is None:
            logger.warning("Validation step is not defined. Skipping evaluation.")
            return {}
        if self.inference_wrapper is not None:
            models = self.inference_wrapper(self, **models)
        invoke_callback(self.on_validation_begin, self, **models)
        elapsed_time = 0.0
        for index, data in enumerate(get_dataset(dataset), start=1):
            invoke_callback(self.on_validation_step_begin, self, **models)
            elapsed_time -= time()
            criteria = self.validation_step(data, **models)
            self.sync()
            elapsed_time += time()
            logs = self.tracker(**criteria) | {"elapsed_time": elapsed_time / index}
            if self.validation_prefix:
                logs = {f"{self.validation_prefix}{k}": v for k, v in logs.items()}
            self.logs().update(logs)
            invoke_callback(self.on_validation_step_end, self, **models)
        invoke_callback(self.on_validation_end, self, **models)
        return logs

    def fit(
        self,
        epochs: int,
        training_dataset: DatasetLike | Callable[[], DatasetLike],
        validation_dataset: DatasetLike | Callable[[], DatasetLike] | None = None,
        start_epoch: int = 1,
        validation_frequency: int = 1,
        **models: ModelT_contra,
    ) -> dict[int, dict[str, Any]]:
        """Fit the model.

        Args:
            epochs (int): Number of epochs to train.
            training_dataset (DatasetLike | Callable[[], DatasetLike]): Training dataset.
            validation_dataset (DatasetLike | Callable[[], DatasetLike] | None, optional): Validation dataset.
                Defaults to None.
            start_epoch (int, optional): Epoch to start training from. Defaults to 1.
            validation_frequency (int, optional): Frequency of validation. Defaults to 1.
            **models (ModelT): The models to train and validate.

        Returns:
            History of training and validation logs.
        """
        if validation_frequency < 1:
            raise ValueError("Validation frequency must be at least 1.")
        if start_epoch < 1:
            raise ValueError(f"Start epoch must be at least 1: {start_epoch}")
        if start_epoch > epochs:
            raise ValueError(f"Start epoch must be less than or equal to epochs: {start_epoch} > {epochs}")
        for epoch in range(start_epoch, epochs + 1):
            self.epoch = epoch
            invoke_callback(self.on_epoch_begin, self, **models)
            self.train(training_dataset, **models)
            if validation_dataset is not None and epoch % validation_frequency == 0:
                self.evaluate(validation_dataset, **models)
            invoke_callback(self.on_epoch_end, self, **models)
        return self.history


@dataclass(kw_only=True, slots=True)
class BestCriterion(Generic[ModelT_contra]):
    """Callback to track the best criterion during training or validation."""

    target: str
    """The target criterion to monitor."""

    mode: Literal["min", "max"] = "min"
    """The mode to monitor the criterion. Either 'min' or 'max'."""

    on_best: NamedCallbackList[BestCallback[ModelT_contra]] = field(default_factory=NamedCallbackList)  # type: ignore[assignment]
    """Callbacks to be called when a new best criterion is found."""

    _step: int = field(default=0, repr=False)
    _best: float = field(init=False, repr=False)
    _compare: Callable[[float, float], bool] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Post initialization."""
        self._compare = lt if self.mode == "min" else gt
        self._best = inf if self.mode == "min" else -inf
        if not isinstance(self.on_best, NamedCallbackList):
            raw = list(self.on_best)
            ncl: NamedCallbackList = NamedCallbackList()
            for cb in raw:
                ncl.append(cb)
            self.on_best = ncl

    @property
    def step(self) -> int:
        """Get the step at which the best criterion was found."""
        return self._step

    @property
    def value(self) -> float:
        """Get the best criterion value found."""
        return self._best

    def __call__(self, info: BaseInfo, **models: ModelT_contra) -> None:
        """Check and update the best criterion."""
        current: float | None = info.logs().get(self.target, None)
        if current is not None:
            if self._compare(current, self._best):
                self._step = info.step
                self._best = current
            invoke_callback(self.on_best, info, self, **models)


__all__ = [
    "GLOBAL_CALLBACKS",
    "Backward",
    "BaseInfo",
    "BaseTrainer",
    "BestCallback",
    "BestCriterion",
    "Callback",
    "Callbacks",
    "DatasetLike",
    "Forward",
    "InferenceWrapper",
    "NamedCallbackList",
    "callbacks_session",
    "get_dataset",
    "get_dataset_size",
    "invoke_callback",
]


if not TYPE_CHECKING:
    import sys

    from structcast.utils.lazy_import import LazySelectedImporter

    sys.modules[__name__] = LazySelectedImporter(__name__, globals())
