"""Trainer for PyTorch models."""

from collections.abc import Callable, Collection, Generator, Mapping, Sequence
from contextlib import AbstractContextManager, contextmanager, nullcontext
from dataclasses import dataclass, field
from logging import getLogger
import os
from typing import TYPE_CHECKING, Any, Literal, Self, TypeVar, cast, overload

from pydantic import TypeAdapter, ValidationError
from timm.utils.distributed import init_distributed_device_so, is_distributed_env, world_info_from_env

from structcast_model.base_trainer import BaseInfo, BaseTrainer, BestCriterion
from structcast_model.builders.schema import TensorSpec, TensorSpecTree
from structcast_model.torch.layers.criteria_tracker import CriteriaTracker
from structcast_model.torch.logger import Logger
from structcast_model.torch.types import Tensor, TensorInitializer
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
        node = TypeAdapter(TensorSpecTree).validate_python(shape)
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


def get_torch_device(device: str | None = None) -> str:
    """Get the device to run the model on."""
    if device is None:
        return "cuda" if torch.cuda.is_available() else "cpu"
    if "cpu" in device:
        return device
    if "cuda" in device:
        if torch.cuda.is_available():
            return device
        logger.warning("CUDA is not available. Using CPU instead.")
        return "cpu"
    raise ValueError(f'Only "cpu" and "cuda" (with optional rank suffix) are supported. Got invalid device: {device}')


def get_torch_device_type(device: str | None = None) -> str:
    """Get the device type (cpu or cuda) from the device string."""
    return get_torch_device(device).split(":")[0]


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


@overload
def initial_distributed_env(
    device: str | None = None,
    dist_backend: str | None = None,
    dist_url: str | None = None,
    *,
    return_dict: Literal[True] = True,
) -> dict[str, Any]: ...


@overload
def initial_distributed_env(
    device: str | None = None,
    dist_backend: str | None = None,
    dist_url: str | None = None,
    *,
    return_dict: Literal[False] = False,
) -> tuple[str, int, int, int, bool]: ...


def initial_distributed_env(
    device: str | None = None,
    dist_backend: str | None = None,
    dist_url: str | None = None,
    *,
    return_dict: bool = True,
) -> dict[str, Any] | tuple[str, int, int, int, bool]:
    """Initialize the distributed environment.

    Args:
        device (str | None): The device to run the model on, e.g., 'cuda' or 'cpu'.
        dist_backend (str | None): The backend to use for distributed training.
            If None, the backend will be automatically selected based on the device.
        dist_url (str | None): The URL to use for distributed training initialization.
            If None, the URL will be automatically generated based on the environment.
        return_dict (bool): Whether to return the result as a dictionary.

    Returns:
        If return_dict is False, returns a tuple of (device, global_rank, local_rank, world_size, distributed).
        If return_dict is True, returns a dictionary with device, global_rank, local_rank, world_size, distributed keys.
    """
    if is_distributed_env() and torch.distributed.is_initialized():
        if "SLURM_PROCID" in os.environ:
            local_rank, global_rank, world_size = world_info_from_env()
        else:
            local_rank, _, _ = world_info_from_env()
            world_size = torch.distributed.get_world_size()
            global_rank = torch.distributed.get_rank()
        device_type = get_torch_device_type(device)
        result = {
            "device": f"{device_type}:{local_rank}" if device_type != "cpu" else "cpu",
            "global_rank": global_rank,
            "local_rank": local_rank,
            "world_size": world_size,
            "distributed": True,
        }
    else:
        device = get_torch_device(device)
        result = init_distributed_device_so(device=device, dist_backend=dist_backend, dist_url=dist_url)
    if return_dict:
        return result
    return result["device"], result["global_rank"], result["local_rank"], result["world_size"], result["distributed"]


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

    def on_training_begin(self, info: BaseInfo, **models: torch.nn.Module) -> None:
        """Reset the tracker so an epoch's training averages start empty."""
        self.tracker.reset()

    def on_validation_begin(self, info: BaseInfo, **models: torch.nn.Module) -> None:
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
            tracker = compile_fn(tracker)
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

    @contextmanager
    def no_sync(self, __updated__: bool) -> Generator[None, None, None]:
        """Context manager to disable gradient synchronizations for DistributedDataParallel models when not updating.

        Args:
            __updated__ (bool): Whether the model is being updated.
        """
        if __updated__:
            yield
        else:
            models, old_values = self.learner.models, {}
            try:
                for name, model in models.items():
                    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                        old_values[name] = model.require_backward_grad_sync
                        model.require_backward_grad_sync = False
                yield
            finally:
                for name, value in old_values.items():
                    models[name].require_backward_grad_sync = value

    def update_models(self, __inputs__: Any) -> tuple[bool, dict[str, Any]]:
        """Perform a training step and update the models.

        Args:
            __inputs__ (Any): The inputs for the training step.

        Returns:
            tuple[bool, dict[str, Any]]: A tuple containing a boolean indicating whether the model was updated and
                a dictionary of criteria for tracking.
        """
        with self.no_sync(updated := self.learner.update(self.step)):
            return updated, self.learner.training_step(**__inputs__)


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
    ) -> list[Self]:
        """Build one monitor per criterion, each logging its best value through *logger*.

        Criteria named in *save_criteria* also save the model states that reached the best value.
        """
        monitors: list[Self] = []
        for target in higher_criteria:
            best = cls(target=target, mode="max")
            best.on_best.append(_BestLogger(logger=logger, save=target in save_criteria))
            monitors.append(best)
        for target in lower_criteria:
            best = cls(target=target, mode="min")
            best.on_best.append(_BestLogger(logger=logger, save=target in save_criteria))
            monitors.append(best)
        return monitors


@dataclass(kw_only=True, slots=True)
class _BestLogger:
    """Log the best value of a criterion, and save the models that reached it when asked to."""

    logger: Logger
    """The logger the best values and model states are written through."""

    save: bool
    """Whether to also save the model states that reached the best value."""

    def on_best(self, info: BaseInfo, best: BestCriterion[torch.nn.Module], **models: torch.nn.Module) -> None:
        """Log the best value, and save the states of *models* when this epoch reached it."""
        name = f"best_{best.target}"
        self.logger.log_metric(name, best.value, step=info.epoch)
        if self.save and info.step == best.step:
            self.logger.log_state_dict(_get_state_dict(_unwrap_ddp(models)), name)


def _get_state_dict(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Return a mapping of name to state dict for all given modules."""
    return {n: m.state_dict() for n, m in kwargs.items()}


def _unwrap_ddp(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Return a mapping of name to module for all given modules, unwrapping DistributedDataParallel if necessary."""
    return {n: m.module if isinstance(m, torch.nn.parallel.DistributedDataParallel) else m for n, m in kwargs.items()}


@dataclass(kw_only=True, slots=True)
class TrainingStateSaver:
    """Callback saving models, optimizers, grad scalers, and loop counters through a logger."""

    logger: Logger
    """The logger the training-state artifacts are written through."""

    def on_epoch_end(self, info: BaseInfo, **kwargs: Any) -> None:
        """Save the full training state of the finished epoch, so a run can be resumed from it."""
        learner = cast("TorchTrainer", info).learner
        states: dict[str, Any] = {
            "models": _get_state_dict(_unwrap_ddp(kwargs)),
            "optimizers": _get_state_dict(getattr(learner, "optimizers", {})),
            "grad_scalers": _get_state_dict(getattr(learner, "grad_scalers", {})),
            "meta": {"epoch": info.epoch, "step": info.step, "update": info.update},
        }
        self.logger.log_state_dict(states, "training_state")


# `_get_state_dict` and `_unwrap_ddp` are listed because the LazySelectedImporter tail below only
# exposes the names in `__all__`, and the unit tests import them directly.
__all__ = [
    "CriteriaTracker",
    "TorchBestCriterion",
    "TorchTracker",
    "TorchTrainer",
    "TrainingStateSaver",
    "_get_state_dict",
    "_unwrap_ddp",
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
