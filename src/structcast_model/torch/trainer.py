"""Trainer for PyTorch models."""

from collections.abc import Callable, Collection, Mapping, Sequence
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass, field
from logging import getLogger
from typing import TYPE_CHECKING, Any, Self, TypeVar, cast

from pydantic import TypeAdapter, ValidationError

from structcast_model.base_trainer import BaseInfo, BaseTrainer, BestCriterion
from structcast_model.builders.schema import TensorSpec, TensorSpecTree
from structcast_model.torch.distributed import DistributedStrategy, initial_distributed_env
from structcast_model.torch.layers.criteria_tracker import CriteriaTracker
from structcast_model.torch.logger import Logger
from structcast_model.torch.types import Tensor, TensorInitializer
from structcast_model.torch.utils import get_torch_device, get_torch_device_type
from structcast_model.utils.base import resolve_input_shapes, resolve_tensor_initializer
import torch

logger = getLogger(__name__)

DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "int32": torch.int32,
    "int64": torch.int64,
}

T = TypeVar("T")


def create_torch_inputs(shape: Any, *, batch_size: int = 1) -> Any:
    """Create dummy inputs based on the provided shape.

    Args:
        shape (Any): The shape of the inputs to create. This can be a tensor specification,
            which is a tuple of integers or a mapping with the `_SHAPE_` key,
            a dictionary of shapes, or a list/tuple of shapes.
        batch_size (int): The batch size to use for the inputs.
            This will be prepended to the shape of every tensor specification.

    Returns:
        Any: The created inputs, which can be a tensor, a dictionary of tensors, or a list of tensors.

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
            float_default=torch.rand,
            int_default=torch.zeros,
            protocol=TensorInitializer,
        )
        return initializer((batch_size, *node.SHAPE), dtype=DTYPES[node.DTYPE])
    if isinstance(node, Mapping):
        return {k: create_torch_inputs(v, batch_size=batch_size) for k, v in node.items()}
    return [create_torch_inputs(v, batch_size=batch_size) for v in node]


def _low_precision_dtype(inputs: Any) -> Any:
    """Return the first `float16` or `bfloat16` element type found in the inputs, or `None` if there is none."""
    if isinstance(inputs, torch.Tensor):
        return inputs.dtype if inputs.dtype in (torch.float16, torch.bfloat16) else None
    if isinstance(inputs, Mapping):
        inputs = inputs.values()
    elif not isinstance(inputs, (list, tuple)):
        return None
    return next((dtype for value in inputs if (dtype := _low_precision_dtype(value)) is not None), None)


def autocast_inputs(inputs: Any, device_type: str) -> AbstractContextManager[Any]:
    """Get the autocast context to run a model on the given dummy inputs in.

    Tensor specifications declare `bfloat16` by default while model parameters are created as `float32`,
    so running a model on the dummy inputs directly would fail on mismatched element types.
    Autocast resolves this the same way mixed precision training does.

    Args:
        inputs (Any): The dummy inputs, which can be a tensor, a dictionary of tensors, or a list of tensors.
        device_type (str): The device type to autocast on, e.g. "cpu" or "cuda".

    Returns:
        AbstractContextManager[Any]: An autocast context for the element type of the inputs,
            or a null context when the inputs contain no low precision floating point tensor.
    """
    dtype = _low_precision_dtype(inputs)
    return nullcontext() if dtype is None else torch.autocast(device_type, dtype=dtype)


def initial_model(model: Any, shapes: dict[str, Any] | None = None) -> tuple[Any, Any]:
    """Initialize the model by creating dummy inputs based on the provided shapes and running a forward pass.

    Args:
        model (Any): The model to initialize. Can be any nested structure containing PyTorch modules.
        shapes (dict[str, Any] | None): A dictionary mapping module names to their input shapes.
            If empty or None, the shapes declared by the model itself are used, and the model
            will not be initialized with dummy inputs when it declares none either.

    Returns:
        A tuple containing the inputs created based on the shapes,
            and the outputs forwarded through the model using the dummy inputs.
    """
    shapes = resolve_input_shapes(model, shapes)
    inputs = None if shapes is None else create_torch_inputs(shapes)
    device_type = torch.get_default_device().type

    def _init(raw: Any) -> Any:
        if isinstance(raw, torch.nn.Module):
            if inputs is None:
                return None
            with autocast_inputs(inputs, device_type):
                return raw(**inputs)
        if isinstance(raw, Mapping):
            res = {k: _init(v) for k, v in raw.items()}
            return res if (cls := type(raw)) is dict else cls(**res)
        if isinstance(raw, (list, tuple)):
            return type(raw)(_init(v) for v in raw)
        return raw

    return inputs, _init(model)


@dataclass(kw_only=True, slots=True)
class TorchTracker:
    """A tracker for PyTorch models."""

    tracker: CriteriaTracker
    """The tracker to use for tracking the criteria."""

    distributed: bool = field(default_factory=torch.distributed.is_initialized)
    """Whether the tracker is being used in a distributed training environment."""

    def on_training_begin(self, info: BaseInfo) -> None:
        """Reset the tracker so an epoch's training averages start empty."""
        self.tracker.reset()

    def on_validation_begin(self, info: BaseInfo) -> None:
        """Reset the tracker so validation averages do not carry training values."""
        self.tracker.reset()

    def __call__(self, **criteria: Tensor) -> dict[str, float]:
        """Log the criteria and return the average values."""
        res: dict[str, Tensor] = self.tracker(criteria)
        if self.distributed:
            for key, tensor in res.items():
                new_tensor = tensor.clone()
                torch.distributed.all_reduce(new_tensor, op=torch.distributed.ReduceOp.AVG)
                res[key] = new_tensor
        return {k: v.item() for k, v in res.items()}

    @classmethod
    def from_criteria(
        cls,
        outputs: list[str],
        compile_fn: Callable[[torch.nn.Module], torch.nn.Module] | None = None,
        distributed: bool | None = None,
    ) -> "TorchTracker":
        """Create a tracker from the given loss and metric modules.

        Args:
            outputs (list[str]): The names of the outputs to track from the loss and metric modules.
            compile_fn (Callable[[torch.nn.Module], torch.nn.Module] | None):
                An optional function to compile the loss and metric modules.
            distributed (bool | None): Whether the tracker will be used in a distributed training environment.

        Returns:
            A TorchTracker instance with the specified loss and metric trackers.
        """
        tracker = CriteriaTracker(outputs)
        if compile_fn is not None:
            # torch.compile returns an OptimizedModule proxying the tracker, typed as a plain Module.
            tracker = cast("CriteriaTracker", compile_fn(tracker))
        if distributed is None:
            distributed = torch.distributed.is_initialized()
        return cls(tracker=tracker, distributed=distributed)


@dataclass(kw_only=True)
class TorchTrainer(BaseTrainer[torch.nn.Module]):
    """Trainer for PyTorch models."""

    device: str
    """Device to run the model on, e.g., 'cuda' or 'cpu'."""

    def sync(self) -> None:
        """Synchronize the device if it is a CUDA device."""
        if "cuda" in self.device:
            torch.cuda.synchronize()


@dataclass(kw_only=True, slots=True)
class TorchBestCriterion(BestCriterion[torch.nn.Module]):
    """A callback to track the best criterion during training or validation for PyTorch models."""

    @classmethod
    def from_criteria(
        cls,
        higher_criteria: Sequence[str],
        lower_criteria: Sequence[str],
        save_criteria: Collection[str],
        logger: Logger,
        strategy: DistributedStrategy,
    ) -> list[Self]:
        """Build one monitor per criterion, each logging its best value through *logger*.

        Criteria named in *save_criteria* also save the model states that reached the best value,
        produced through *strategy*. Ranks that write nothing pass a :class:`NullLogger`.
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
    """The logger the best values and model states are written through; a NullLogger on non-writing ranks."""

    save: bool
    """Whether to also save the model states that reached the best value."""

    strategy: DistributedStrategy
    """The strategy producing the model states to save."""

    def on_best(self, info: BaseInfo[torch.nn.Module], best: BestCriterion[torch.nn.Module]) -> None:
        """Log the best value, and save the states of the info's models when this epoch reached it."""
        name = f"best_{best.target}"
        self.logger.log_metric(name, best.value, step=info.epoch)
        if self.save and info.step == best.step:
            # Producing the states is a collective, so every rank must reach it. That the ranks agree
            # on whether this epoch is the best is guaranteed by the tracker values being all-reduced.
            self.logger.log_state_dict(self.strategy.state_dict(dict(info.models))["models"], name)


@dataclass(kw_only=True, slots=True)
class TrainingStateSaver:
    """Callback saving models, optimizers, grad scalers, and loop counters through a logger."""

    logger: Logger
    """The logger the training-state artifacts are written through; a NullLogger on non-writing ranks."""

    strategy: DistributedStrategy
    """The strategy producing the model and optimizer states to save."""

    def on_epoch_end(self, info: BaseInfo[torch.nn.Module]) -> None:
        """Save the full training state of the finished epoch, so a run can be resumed from it."""
        learner = cast("TorchTrainer", info).learner
        # Producing the states is a collective: every rank runs it, the null-logger ranks discard it.
        states = self.strategy.state_dict(dict(info.models), learner.optimizers, learner.optimizer_models)
        states.setdefault("optimizers", {})
        states["grad_scalers"] = {n: s.state_dict() for n, s in getattr(learner, "grad_scalers", {}).items()}
        states["meta"] = {"epoch": info.epoch, "step": info.step, "update": info.update}
        self.logger.log_state_dict(states, "training_state")


__all__ = [
    "CriteriaTracker",
    "TorchBestCriterion",
    "TorchTracker",
    "TorchTrainer",
    "TrainingStateSaver",
    "autocast_inputs",
    "create_torch_inputs",
    "get_torch_device",
    "get_torch_device_type",
    "initial_distributed_env",
    "initial_model",
    "resolve_input_shapes",
]


if not TYPE_CHECKING:
    import sys

    from structcast.utils.lazy_import import LazySelectedImporter

    sys.modules[__name__] = LazySelectedImporter(__name__, globals())
