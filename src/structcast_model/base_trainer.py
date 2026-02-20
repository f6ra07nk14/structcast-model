"""Base trainer for training a model."""

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from math import inf
from operator import gt, lt
from time import time
from typing import Any, Generic, Literal, Protocol, TypeAlias, TypeVar

ModelT_contra = TypeVar("ModelT_contra", contravariant=True)

DatasetLike: TypeAlias = Iterable[dict[str, Any]]
"""Dataset-like object."""


def get_dataset(dataset: DatasetLike | Callable[[], DatasetLike]) -> Iterable[dict[str, Any]]:
    """Get the dataset."""
    return dataset() if callable(dataset) else dataset


def get_dataset_size(dataset: DatasetLike | Callable[[], DatasetLike]) -> int:
    """Get the size of the dataset."""
    dataset = get_dataset(dataset)
    if isinstance(dataset, Sequence):
        return len(dataset)
    return sum(1 for _ in dataset)


class Forward(Protocol[ModelT_contra]):
    """Protocol for forward pass configuration."""

    def __call__(self, inputs: Any, **models: ModelT_contra) -> dict[str, Any]:
        """Perform the forward pass for the given inputs and return the outputs and any additional information."""


class Backward(Protocol):
    """Protocol for backward pass configuration."""

    def __call__(self, step: int, *args: Any, **kwargs: Any) -> bool:
        """Perform the backward pass for the given step and losses, and return whether the model has been updated."""


class InferenceWrapper(Protocol[ModelT_contra]):
    """Protocol for inference wrapper."""

    def __call__(self, **models: ModelT_contra) -> dict[str, Any]:
        """Wrap the model for inference, e.g., for quantization or ONNX export."""


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

    def __call__(self, info: BaseInfo, target: str, best: float, **models: ModelT_contra) -> None:
        """Call the callback with the given info, target criterion, and best value."""


def invoke_callback(callbacks: list[Callable[..., None]], info: BaseInfo, *args: Any, **models: ModelT_contra) -> None:
    """Invoke callback."""
    for callback in callbacks:
        callback(info, *args, **models)


@dataclass(kw_only=True)
class Callbacks(Generic[ModelT_contra]):
    """Callbacks."""

    on_update: list[Callback[ModelT_contra]] = field(default_factory=list)
    """Callbacks to call after each update."""

    on_training_begin: list[Callback[ModelT_contra]] = field(default_factory=list)
    """Callbacks to call at the beginning of training."""

    on_training_end: list[Callback[ModelT_contra]] = field(default_factory=list)
    """Callbacks to call at the end of training."""

    on_training_step_begin: list[Callback[ModelT_contra]] = field(default_factory=list)
    """Callbacks to be called at the beginning of each training step."""

    on_training_step_end: list[Callback[ModelT_contra]] = field(default_factory=list)
    """Callbacks to be called at the end of each training step."""

    on_validation_begin: list[Callback[ModelT_contra]] = field(default_factory=list)
    """Callbacks to be called at the beginning of validation."""

    on_validation_end: list[Callback[ModelT_contra]] = field(default_factory=list)
    """Callbacks to be called at the end of validation."""

    on_validation_step_begin: list[Callback[ModelT_contra]] = field(default_factory=list)
    """Callbacks to be called at the beginning of each validation step."""

    on_validation_step_end: list[Callback[ModelT_contra]] = field(default_factory=list)
    """Callbacks to be called at the end of each validation step."""

    on_epoch_begin: list[Callback[ModelT_contra]] = field(default_factory=list)
    """Callbacks to be called at the beginning of each epoch."""

    on_epoch_end: list[Callback[ModelT_contra]] = field(default_factory=list)
    """Callbacks to be called at the end of each epoch."""


GLOBAL_CALLBACKS = Callbacks[Any]()
"""Global callbacks."""


@dataclass(kw_only=True)
class BaseTrainer(BaseInfo, Callbacks[ModelT_contra]):
    """Base trainer for training a model."""

    training_step: Forward[ModelT_contra]
    """The forward pass configuration for training."""

    backward: Backward
    """The backward pass configuration."""

    logger: Callable[..., dict[str, Any]]
    """The logger to log training and validation information."""

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

    add_global_callbacks: bool = False
    """Whether to add global callbacks."""

    def __post_init__(self) -> None:
        """Post initialization."""
        if self.add_global_callbacks:
            self.on_update.extend(GLOBAL_CALLBACKS.on_update)
            self.on_training_begin.extend(GLOBAL_CALLBACKS.on_training_begin)
            self.on_training_end.extend(GLOBAL_CALLBACKS.on_training_end)
            self.on_training_step_begin.extend(GLOBAL_CALLBACKS.on_training_step_begin)
            self.on_training_step_end.extend(GLOBAL_CALLBACKS.on_training_step_end)
            self.on_validation_begin.extend(GLOBAL_CALLBACKS.on_validation_begin)
            self.on_validation_end.extend(GLOBAL_CALLBACKS.on_validation_end)
            self.on_validation_step_begin.extend(GLOBAL_CALLBACKS.on_validation_step_begin)
            self.on_validation_step_end.extend(GLOBAL_CALLBACKS.on_validation_step_end)
            self.on_epoch_begin.extend(GLOBAL_CALLBACKS.on_epoch_begin)
            self.on_epoch_end.extend(GLOBAL_CALLBACKS.on_epoch_end)

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
        logger, training_step, backward, elapsed_time = self.logger, self.training_step, self.backward, 0.0
        for index, inputs in enumerate(get_dataset(dataset), start=1):
            self.step += 1
            invoke_callback(self.on_training_step_begin, self, **models)
            elapsed_time -= time()
            criteria = training_step(inputs, **models)
            should_update = backward(self.step, **criteria)
            elapsed_time += time()
            logs = logger(**criteria) | {"elapsed_time": elapsed_time / index}
            if self.training_prefix:
                logs = {f"{self.training_prefix}{k}": v for k, v in logs.items()}
            self.logs().update(logs)
            if should_update:
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
        if self.inference_wrapper is not None:
            models = self.inference_wrapper(**models)
        invoke_callback(self.on_validation_begin, self, **models)
        logger, elapsed_time = self.logger, 0.0
        validation_step = self.training_step if self.validation_step is None else self.validation_step
        for index, data in enumerate(get_dataset(dataset), start=1):
            invoke_callback(self.on_validation_step_begin, self, **models)
            elapsed_time -= time()
            criteria = validation_step(data, **models)
            elapsed_time += time()
            logs = logger(**criteria) | {"elapsed_time": elapsed_time / index}
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
    """Save the best criterion."""

    target: str
    """The target criterion to monitor."""

    mode: Literal["min", "max"] = "min"
    """The mode to monitor the criterion. Either 'min' or 'max'."""

    on_best: list[BestCallback[ModelT_contra]] = field(default_factory=list)
    """Callbacks to be called when a new best criterion is found."""

    _best: float = field(init=False, repr=False)
    _compare: Callable[[float, float], bool] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Post initialization."""
        self._compare = lt if self.mode == "min" else gt
        self._best = inf if self.mode == "min" else -inf

    def __call__(self, info: BaseInfo, **models: ModelT_contra) -> None:
        """Check and update the best criterion."""
        current: float | None = info.logs().get(self.target, None)
        if current is not None:
            if self._compare(current, self._best):
                self._best = current
            invoke_callback(self.on_best, info, self.target, self._best, **models)
