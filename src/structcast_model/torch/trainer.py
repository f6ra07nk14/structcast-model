"""Trainer for PyTorch models."""

from collections import OrderedDict
from collections.abc import Callable, Generator, Mapping
from contextlib import AbstractContextManager, contextmanager, nullcontext
from dataclasses import dataclass, field
from functools import cached_property, partial
from logging import getLogger
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, TypeVar, cast, overload

from pydantic import TypeAdapter, ValidationError
from structcast.core.instantiator import ObjectPattern, instantiate
from structcast.utils.base import dump_yaml
from timm.utils.distributed import init_distributed_device_so, is_distributed_env, world_info_from_env

from structcast_model.base_trainer import BaseInfo, BaseTrainer, BestCriterion
from structcast_model.builders.schema import TensorSpec, TensorSpecTree
from structcast_model.torch.layers.criteria_tracker import CriteriaTracker
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


@dataclass(kw_only=True)
class TorchLearnerFactory:
    """Build the models and the learner of a training run from object patterns.

    The tracker and any `DistributedDataParallel` wrapping are deliberately left to the caller:
    the tracker is a metrics concern with its own trainer field, and the wrapping depends on the
    distributed environment the caller set up.
    """

    model_patterns: list[dict[str, Any]]
    """Object patterns of the models, each entry mapping exactly one model name to its pattern."""

    learner_pattern: Any
    """Object pattern of the learner, called with the instantiated models as keyword arguments."""

    compile_pattern: dict[str, Any] | None = None
    """Keyword arguments for `torch.compile`, or None to leave the step functions uncompiled."""

    initializer_patterns: list[dict[str, Any]] | None = None
    """Object patterns of the initializers, each entry mapping model names to an initializer."""

    shapes: dict[str, Any] | None = None
    """Input shapes used to create the dummy inputs, overriding the shapes declared by the models."""

    input_shapes: dict[str, Any] = field(default_factory=dict, init=False)
    """The input shapes actually used, resolved during the last call."""

    @cached_property
    def compile_fn(self) -> Callable[[Any], Any]:
        """Compile a module or a function, or return it unchanged when no compile pattern was given."""
        if self.compile_pattern is None:
            return lambda module: module
        return partial(torch.compile, **instantiate(self.compile_pattern))

    @staticmethod
    def _instantiate_pattern(raw: Any) -> Any:
        """Instantiate a validated object pattern, raising on malformed input instead of passing it through."""
        return ObjectPattern.model_validate(raw).build().runs[0]

    def _instantiate_models(self) -> "OrderedDict[str, Any]":
        """Instantiate the models from the name-pattern mappings, in the order they were given."""
        models: OrderedDict[str, Any] = OrderedDict()
        for raw in self.model_patterns:
            if len(raw) != 1:
                raise ValueError(f"Each model pattern should contain exactly one model definition. Got: {raw}")
            name, pattern = next(iter(raw.items()))
            models[name] = self._instantiate_pattern(pattern)
        return models

    def __call__(self, device: str, *, apply_initializers: bool = True) -> tuple["OrderedDict[str, Any]", Any]:
        """Build the models and the learner on the given device.

        Args:
            device (str): The device to create the models on, e.g. "cpu" or "cuda:0".
            apply_initializers (bool): Whether to apply the initializers. Distributed runs apply them
                on the main process only and broadcast the result.

        Returns:
            The models by name, and the learner built from them.
        """
        with torch.device(device):
            models = self._instantiate_models()
            self.input_shapes = resolve_input_shapes(models, self.shapes) or {}
            initial_model(models, self.input_shapes)
            if apply_initializers and self.initializer_patterns:
                initializers = instantiate({k: v for raw in self.initializer_patterns for k, v in raw.items()})
                for name, model in models.items():
                    if name in initializers:
                        model.apply(initializers[name])
            learner = self._instantiate_pattern(self.learner_pattern)(**models)
        if hasattr(learner, "forward_training_step"):
            learner.forward_training_step = self.compile_fn(learner.forward_training_step)
        if hasattr(learner, "forward_inference_step"):
            learner.forward_inference_step = self.compile_fn(learner.forward_inference_step)
        return models, learner


def _epoch_metrics(info: BaseInfo) -> dict[str, Any]:
    """Merge the learning rates reported by the learner into the criteria of the current epoch.

    Schedules step in the learner's own on_epoch_end hooks, which the trainer dispatches before the
    logger's, so the recorded learning rate is the one the NEXT epoch will use -- the same one-epoch
    offset the pre-redesign global callbacks produced.
    """
    return {**cast("BaseTrainer[Any]", info).learner.learning_rates, **info.logs()}


class MLflowLogger:
    """Logger recording a run to MLflow.

    The logger owns the run: entering it starts the run, leaving it ends the run. It also reacts to
    the end of each epoch, so passing it to a trainer logs the epoch metrics.
    """

    def __init__(self, experiment: str) -> None:
        """Create the logger for the given experiment, without starting a run yet."""
        # mlflow is an optional extra: importing it here keeps this module importable without it.
        import mlflow  # noqa: PLC0415
        import mlflow.pytorch  # noqa: PLC0415

        self.experiment = experiment
        self.mlflow = mlflow

    def __enter__(self) -> "MLflowLogger":
        """Start a run in the configured experiment."""
        self.mlflow.set_experiment(self.experiment)
        self.mlflow.start_run()
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        """End the run, marking it failed when an exception is propagating."""
        self.mlflow.end_run(status="FINISHED" if exc_type is None else "FAILED")

    def log_params(self, params: Mapping[str, Any]) -> None:
        """Log the run parameters."""
        for key, value in params.items():
            self.mlflow.log_param(key, value)

    def log_dict(self, data: Mapping[str, Any], name: str) -> None:
        """Log a dictionary as an artifact under the given file name."""
        self.mlflow.log_dict(dict(data), name)

    def log_artifact(self, path: str) -> None:
        """Log a local file as an artifact."""
        self.mlflow.log_artifact(path)

    def log_metric(self, name: str, value: float, step: int) -> None:
        """Log one metric value at the given step."""
        self.mlflow.log_metric(name, value, step=step)

    def log_metrics(self, metrics: Mapping[str, float], step: int) -> None:
        """Log several metric values at the given step."""
        self.mlflow.log_metrics(dict(metrics), step=step)

    def log_state_dict(self, states: Mapping[str, Any], name: str) -> None:
        """Log a state dictionary under the given artifact name."""
        self.mlflow.pytorch.log_state_dict(dict(states), name)

    def on_epoch_end(self, info: BaseInfo, **models: Any) -> None:
        """Log the criteria and learning rates of the finished epoch."""
        self.log_metrics(_epoch_metrics(info), step=info.epoch)


class WandbLogger:
    """Logger recording a run to Weights & Biases, with the same interface as `MLflowLogger`."""

    def __init__(self, experiment: str) -> None:
        """Create the logger for the given experiment, without starting a run yet."""
        # wandb is an optional extra: importing it here keeps this module importable without it.
        import wandb  # noqa: PLC0415

        self.experiment = experiment
        self.wandb = wandb

    def __enter__(self) -> "WandbLogger":
        """Start a run in the project named after the experiment."""
        self.wandb.init(project=self.experiment)
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        """Finish the run, marking it failed when an exception is propagating."""
        self.wandb.finish(exit_code=0 if exc_type is None else 1)

    def log_params(self, params: Mapping[str, Any]) -> None:
        """Log the run parameters."""
        self.wandb.config.update(dict(params))

    def log_dict(self, data: Mapping[str, Any], name: str) -> None:
        """Write a dictionary into the run directory as YAML, matching what MLflow stores."""
        dump_yaml(dict(data), Path(self.wandb.run.dir) / name)

    def log_artifact(self, path: str) -> None:
        """Log a local file as an artifact."""
        self.wandb.save(path)

    def log_metric(self, name: str, value: float, step: int) -> None:
        """Log one metric value at the given step."""
        self.wandb.log({name: value}, step=step)

    def log_metrics(self, metrics: Mapping[str, float], step: int) -> None:
        """Log several metric values at the given step."""
        self.wandb.log(dict(metrics), step=step)

    def log_state_dict(self, states: Mapping[str, Any], name: str) -> None:
        """Save a state dictionary into the run directory."""
        torch.save(dict(states), Path(self.wandb.run.dir) / f"{name}.pt")

    def on_epoch_end(self, info: BaseInfo, **models: Any) -> None:
        """Log the criteria and learning rates of the finished epoch."""
        self.log_metrics(_epoch_metrics(info), step=info.epoch)


__all__ = [
    "CriteriaTracker",
    "MLflowLogger",
    "TorchBestCriterion",
    "TorchLearnerFactory",
    "TorchTracker",
    "TorchTrainer",
    "WandbLogger",
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
