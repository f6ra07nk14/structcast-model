"""Trainer for PyTorch models."""

from collections.abc import Callable, Generator, Iterable, Mapping
from contextlib import AbstractContextManager, contextmanager, suppress
from dataclasses import dataclass, field
from functools import cached_property, partial
from logging import getLogger
import os
from typing import TYPE_CHECKING, Any, Literal, TypeVar, overload

from pydantic import Field, TypeAdapter, ValidationError
from structcast.core.base import WithExtra
from structcast.core.specifier import FlexSpec
from timm.data import (
    IMAGENET_DEFAULT_MEAN,
    IMAGENET_DEFAULT_STD,
    AugMixDataset,
    FastCollateMixup,
    Mixup,
    create_dataset,
    create_loader,
)
from timm.utils import ModelEmaV3
from timm.utils.distributed import init_distributed_device_so, is_distributed_env, world_info_from_env
from torch.utils.data import DataLoader

from structcast_model.base_trainer import GLOBAL_CALLBACKS, BaseInfo, BaseTrainer, BestCriterion
from structcast_model.torch.layers.criteria_tracker import CriteriaTracker
from structcast_model.torch.types import Tensor
import torch

logger = getLogger(__name__)

DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}

T = TypeVar("T")


def create_torch_inputs(shape: Any) -> Any:
    """Create dummy inputs based on the provided shape."""
    try:
        return torch.rand((1, *TypeAdapter(tuple[int, ...]).validate_python(shape)), dtype=torch.float32)
    except ValidationError:
        pass
    if isinstance(shape, dict):
        return {k: create_torch_inputs(v) for k, v in shape.items()}
    if isinstance(shape, (list, tuple)):
        return [create_torch_inputs(v) for v in shape]
    raise ValueError(f"Invalid tensor shape: {shape}")


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
            If None, the model will not be initialized with dummy inputs.

    Returns:
        A tuple containing the inputs created based on the shapes,
            and the outputs forwarded through the model using the dummy inputs.
    """
    inputs = None if shapes is None else create_torch_inputs(shapes)

    def _init(raw: Any) -> Any:
        if isinstance(raw, torch.nn.Module):
            return None if inputs is None else raw(**inputs)
        if isinstance(raw, Mapping):
            res = {k: _init(v) for k, v in raw.items()}
            return res if (cls := type(raw)) is dict else cls(**res)
        if isinstance(raw, (list, tuple)):
            return type(raw)(_init(v) for v in raw)
        return raw

    return inputs, _init(model)


def get_autocast(mixed_precision_type: str | None, device: str | None) -> Callable[[], AbstractContextManager[None]]:
    """Get the appropriate autocast context manager based on the device and mixed precision type."""
    if mixed_precision_type is None:
        return suppress
    return partial(torch.autocast, device_type=get_torch_device_type(device), dtype=DTYPES[mixed_precision_type])


@dataclass(kw_only=True, slots=True)
class TrainingStep:
    """A training step for a PyTorch model."""

    models: list[str]
    """The names of the models to use for the training step."""

    losses: torch.nn.Module
    """A module that computes the losses for the model."""

    metrics: torch.nn.Module | None = None
    """A module that computes the metrics for the model."""

    autocast: Callable[[], AbstractContextManager[None]] = suppress
    """A context manager for automatic mixed precision (AMP). By default, it does nothing."""

    def __call__(self, inputs: dict[str, Any], **models: torch.nn.Module) -> dict[str, Tensor]:
        """Perform the forward pass for the given inputs and return the outputs and any additional information."""
        with self.autocast():
            outputs = inputs.copy()
            for name in self.models:
                outputs.update(models[name](**outputs))
            criteria = self.losses(**outputs)
            if self.metrics is None:
                return criteria
            with torch.no_grad():
                criteria.update(self.metrics(**outputs))
            return criteria


@dataclass(kw_only=True, slots=True)
class ValidationStep(TrainingStep):
    """A validation step for a PyTorch model."""

    def __call__(self, inputs: dict[str, Any], **models: torch.nn.Module) -> dict[str, Tensor]:
        """Perform the forward pass for the given inputs and return the outputs and any additional information."""
        with torch.no_grad():
            with self.autocast():
                outputs = inputs.copy()
                for name in self.models:
                    outputs.update(models[name](**outputs))
                criteria = self.losses(**outputs)
                if self.metrics is None:
                    return criteria
                criteria.update(self.metrics(**outputs))
                return criteria


@dataclass(kw_only=True, slots=True)
class TorchTracker:
    """A tracker for PyTorch models."""

    tracker: CriteriaTracker
    """The tracker to use for tracking the criteria."""

    distributed: bool = field(default_factory=torch.distributed.is_initialized)
    """Whether the tracker is being used in a distributed training environment."""

    def __post_init__(self) -> None:
        """Post-initialization."""
        GLOBAL_CALLBACKS.on_training_begin.register("reset_tracker", self.reset)
        GLOBAL_CALLBACKS.on_validation_begin.register("reset_tracker", self.reset)

    def reset(self, info: BaseInfo, **models: torch.nn.Module) -> None:
        """Reset the trackers at the beginning of training."""
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


@dataclass(kw_only=True, slots=True)
class TimmEmaWrapper:
    """An inference wrapper that returns the EMA model from the timm library."""

    is_cross_device: dict[str, bool]
    """A dictionary mapping model names to a boolean indicating whether
    the EMA model is on a different device than the original model."""

    ema: dict[str, ModelEmaV3]
    """The EMA model."""

    distributed: bool = field(default_factory=torch.distributed.is_initialized)
    """Whether the wrapper is being used in a distributed training environment."""

    def __post_init__(self) -> None:
        """Post-initialization."""
        GLOBAL_CALLBACKS.on_update.register("ema_update", self.update)

    def update(self, info: BaseInfo, **models: torch.nn.Module) -> None:
        """Update the EMA model."""
        if self.distributed:
            models = {n: getattr(m, "module", m) for n, m in models.items()}  # unwrap DDP for EMA update
        for name, ema in self.ema.items():
            ema.update(models[name], step=info.update)

    def __call__(self, info: BaseInfo, **models: torch.nn.Module) -> dict[str, Any]:
        """Return the EMA model."""
        return {n: m if n in self.ema and self.is_cross_device[n] else self.ema[n] for n, m in models.items()}

    @property
    def models(self) -> dict[str, torch.nn.Module]:
        """Return the EMA models."""
        return {n: ema.module for n, ema in self.ema.items()}

    @classmethod
    def from_models(
        cls,
        models: dict[str, torch.nn.Module],
        device: torch.device | None = None,
        compile_fn: Callable[[torch.nn.Module], torch.nn.Module] | None = None,
        distributed: bool | None = None,
        **kwargs: Any,
    ) -> "TimmEmaWrapper":
        """Create a TimmEmaWrapper from the given models.

        Args:
            models (dict[str, torch.nn.Module]): The models to create the EMA wrapper for.
            device (torch.device | None): The device to move the EMA models to.
                If None, the EMA models will not be moved.
            compile_fn (Callable[[torch.nn.Module], torch.nn.Module] | None):
                An optional function to compile the EMA models.
            distributed (bool | None): Whether the wrapper will be used in a distributed training environment.
            **kwargs: Additional keyword arguments to pass to the ModelEmaV3 constructor.

        Returns:
            A TimmEmaWrapper instance with the specified EMA models and callbacks.
        """

        def _get_device(model: torch.nn.Module) -> str:
            return next(model.parameters()).device.type

        ema, is_cross_device = {}, {}
        for name, model in models.items():
            ema_model = ModelEmaV3(model, device=device, **kwargs)
            if compile_fn is not None:
                ema_model = compile_fn(ema_model)
            ema[name] = ema_model
            is_cross_device[name] = _get_device(model) != _get_device(ema_model.module)
        if distributed is None:
            distributed = torch.distributed.is_initialized()
        return cls(ema=ema, is_cross_device=is_cross_device, distributed=distributed)


def _model_train(info: BaseInfo, **models: torch.nn.Module) -> None:
    """A callback to synchronize the device at the beginning of training."""
    for model in models.values():
        model.train()


def _model_eval(info: BaseInfo, **models: torch.nn.Module) -> None:
    """A callback to set the model to evaluation mode at the beginning of validation."""
    for model in models.values():
        model.eval()


@dataclass(kw_only=True)
class TorchTrainer(BaseTrainer[torch.nn.Module]):
    """Trainer for PyTorch models."""

    device: str
    """Device to run the model on, e.g., 'cuda' or 'cpu'."""

    def __post_init__(self) -> None:
        """Post-initialization."""
        super().__post_init__()
        self.on_training_begin.register("model_train", _model_train)
        self.on_validation_begin.register("model_eval", _model_eval)

    def sync(self) -> None:
        """Synchronize the device if it is a CUDA device."""
        if "cuda" in self.device:
            torch.cuda.synchronize()

    @contextmanager
    def no_sync(self, __updated__: bool, **models: torch.nn.Module) -> Generator[None, None, None]:
        """Context manager to disable gradient synchronizations for DistributedDataParallel models when not updating.

        Args:
            __updated__ (bool): Whether the model is being updated.
            **models (torch.nn.Module): The models to potentially disable gradient synchronization for.
        """
        if __updated__:
            yield
        else:
            old_values = {}
            try:
                for name, model in models.items():
                    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                        old_values[name] = model.require_backward_grad_sync
                        model.require_backward_grad_sync = False
                yield
            finally:
                for name, value in old_values.items():
                    models[name].require_backward_grad_sync = value

    def update_models(self, __inputs__: Any, **models: torch.nn.Module) -> tuple[bool, dict[str, Any]]:
        """Perform a training step and update the models.

        Args:
            __inputs__ (Any): The inputs for the training step.
            **models (torch.nn.Module): The models to update.

        Returns:
            tuple[bool, dict[str, Any]]: A tuple containing a boolean indicating whether the model was updated and
                a dictionary of criteria for tracking.
        """
        updated = self.backward.update(self.step)
        with self.no_sync(__updated__=updated, **models):
            criteria = self.training_step(__inputs__, **models)
            self.backward(**criteria)
            return updated, criteria


@dataclass(kw_only=True, slots=True)
class TorchBestCriterion(BestCriterion[torch.nn.Module]):
    """A callback to track the best criterion during training or validation for PyTorch models."""


class TimmDatasetWrapper(WithExtra):
    """A wrapper for timm data loaders to be used in the training loop."""

    batch_size: int = 128
    """Batch size hint for iterable datasets (TFDS, WDS, HFIDS)."""

    name: str = "imagenet"
    """Dataset name, empty is okay for folder based datasets."""

    root: str | None = None
    """Root folder of dataset (All)."""

    input_img_mode: Literal[
        "1",
        "CMYK",
        "F",
        "HSV",
        "I",
        "I;16",
        "I;16B",
        "I;16L",
        "I;16N",
        "L",
        "LA",
        "La",
        "LAB",
        "P",
        "PA",
        "RGB",
        "RGBA",
        "RGBa",
        "RGBX",
        "YCbCr",
    ] = "RGB"
    """The image mode to use for the input images. This should be a valid mode supported by the PIL library."""

    input_key: str | None = None
    """Dataset key for input images."""

    target_key: str | None = None
    """Dataset key for target labels."""

    class_map: dict[str, Any] | None = None
    """A mapping from class names to indices for the dataset.
    This is optional and can be used to remap the class labels if needed."""

    seed: int = 42
    """The random seed to use for shuffling the dataset and any other random operations.
    This ensures reproducibility of the training process."""

    repeats: int = 0
    """Epoch repeat multiplier (number of times to repeat dataset epoch per train epoch)."""

    download: bool = False
    """Allow download of dataset for torch/ and tfds/ datasets that support it."""

    trust_remote_code: bool = False
    """Allow huggingface dataset import to execute code downloaded from the dataset's repo."""

    is_training: bool = False
    """Create dataset in train mode, this is different from the split.
    For Iterable / TDFS it enables shuffle, ignored for other datasets. (TFDS, WDS, HFIDS)"""

    split: str = "validation"
    """The dataset split to use for training or validation.
    This should be a valid split supported by the dataset, such as "train", "validation", or "test"."""

    num_samples: int | None = None
    """Manually specify num samples in target split, for IterableDatasets."""

    @property
    def default_kwargs(self) -> dict[str, Any]:
        """Default kwargs for the dataset."""
        return {
            "name": self.name,
            "root": self.root,
            "class_map": self.class_map,
            "download": self.download,
            "batch_size": self.batch_size,
            "seed": self.seed,
            "repeats": self.repeats,
            "input_img_mode": self.input_img_mode,
            "input_key": self.input_key,
            "target_key": self.target_key,
            "trust_remote_code": self.trust_remote_code,
            "is_training": self.is_training,
            "split": self.split,
            "num_samples": self.num_samples,
        }

    @cached_property
    def dataset(self) -> Any:
        """Create a dataset using the timm library."""
        return create_dataset(**self.default_kwargs, **self.model_extra)


class TimmDataLoaderWrapper(WithExtra):
    """A wrapper for timm data loaders to be used in the training loop."""

    spec: FlexSpec | None = None
    """An optional FlexSpec to apply to the data loader outputs, for flexible input mapping to the model."""

    dataset: TimmDatasetWrapper = Field(default_factory=TimmDatasetWrapper)
    """The dataset to create the data loader for."""

    channels_last: bool = False
    """Use channels_last memory format for inputs."""

    # for distributed training, will be passed to initial_distributed_env:

    device: str = "cpu"
    """Device to move data to after loading, e.g. 'cuda' or 'cpu'. If None, data will not be moved."""

    dist_backend: str | None = None
    """The backend to use for distributed training.
    If None, the backend will be automatically selected based on the device."""

    dist_url: str | None = None
    """The URL to use for distributed training initialization.
    If None, the URL will be automatically generated based on the environment."""

    # for mixup

    use_prefetcher: bool = True
    """Use efficient pre-fetcher to load samples onto device."""

    mixup_alpha: float = 0.0
    """Mixup alpha value, mixup enabled if > 0.0."""

    cutmix_alpha: float = 0.0
    """CutMix alpha value, CutMix enabled if > 0.0."""

    cutmix_minmax: tuple[float, float] | None = None
    """cutmix min/max ratio, overrides alpha and enables cutmix if set."""

    mixup_prob: float = 1.0
    """Probability of performing mixup or cutmix when either/both is enabled."""

    mixup_switch_prob: float = 0.5
    """Probability of switching to cutmix when both mixup and cutmix enabled."""

    mixup_mode: Literal["batch", "pair", "elem"] = "batch"
    """Mode of applying mixup or cutmix."""

    mixup_off_epoch: int = 0
    """Turn off mixup after this epoch, disabled if 0 (default: 0)"""

    label_smoothing: float = 0.0
    """Label smoothing value."""

    num_classes: int = 1000
    """Number of label classes in dataset."""

    # for create_loader

    input_size: int | tuple[int, int] | tuple[int, int, int] = (3, 224, 224)
    """Target input size (channels, height, width) tuple or size scalar."""

    interpolation: Literal["random", "nearest", "bilinear", "bicubic", "box", "hamming", "lanczos"] = "bicubic"
    """Interpolation method for resizing images.
    Can be 'random', 'nearest', 'bilinear', 'bicubic', 'box', 'hamming', or 'lanczos'."""

    mean: tuple[float, float, float] = IMAGENET_DEFAULT_MEAN
    """Mean for image normalization, as a tuple of (R, G, B) values."""

    std: tuple[float, float, float] = IMAGENET_DEFAULT_STD
    """Standard deviation for image normalization, as a tuple of (R, G, B) values."""

    image_dtype: Literal["float32", "float16", "bfloat16"] = "float32"
    """Data type for the input images. Can be 'float32', 'float16', or 'bfloat16'."""

    num_workers: int = 1
    """Num worker processes per DataLoader."""

    pin_memory: bool = False
    """Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU."""

    # only for training / is_training=True kwargs:

    no_aug: bool = False
    """Disable augmentation for training (useful for debug)."""

    re_prob: float = 0.0
    """Random erasing probability."""

    re_mode: Literal["const", "pixel", "rand"] = "const"
    """Random erasing fill mode."""

    re_count: int = 1
    """Number of random erasing regions."""

    re_split: bool = False
    """Control split of random erasing across batch size."""

    train_crop_mode: Literal["rrc", "rkrc", "rkrr"] = "rrc"
    """Random cropping mode for training.
    Options are 'rrc' (random resized crop), 'rkrc' (random resized crop with scale and ratio),
    and 'rkrr' (random resized crop with scale, ratio, and interpolation)."""

    scale: tuple[float, float] = (0.08, 1.0)
    """Random resized crop scale range."""

    ratio: tuple[float, float] = (3.0 / 4.0, 4.0 / 3.0)
    """Random resized crop aspect ratio range."""

    hflip: float = 0.5
    """Horizontal flip probability."""

    vflip: float = 0.0
    """Vertical flip probability."""

    color_jitter: float = 0.4
    """Random color jitter component factors (brightness, contrast, saturation, hue).
    Scalar is applied as (scalar,) * 3 (no hue)."""

    color_jitter_prob: float | None = None
    """Apply color jitter with this probability if not None (for SimlCLR-like augmentation)."""

    grayscale_prob: float = 0.0
    """Random grayscale probability."""

    gaussian_blur_prob: float = 0.0
    """Random Gaussian blur probability."""

    auto_augment: str | None = None
    """Auto augmentation policy. Can be one of the policies in the timm library,
    such as 'v0', 'original', 'rand-m9-mstd0.5-inc1', etc."""

    num_aug_repeats: int = 0
    """Number of augmentation repetitions (distributed training only) (default: 0)"""

    num_aug_splits: int = 0
    """Number of augmentation splits (default: 0, valid: 0 or >=2)"""

    use_multi_epochs_loader: bool = False
    """use the multi-epochs-loader to save time at the beginning of every epoch."""

    worker_seeding: Literal["all", "part"] = "all"
    """Control worker random seeding at init."""

    # only for validation / is_training=False kwargs:

    crop_pct: float = 0.875
    """Inference crop percentage (output size / resize size)."""

    @property
    def mixup_active(self) -> bool:
        """Whether mixup or cutmix is active based on the provided parameters."""
        return self.mixup_alpha > 0.0 or self.cutmix_alpha > 0.0 or self.cutmix_minmax is not None

    @property
    def mixup_kwargs(self) -> dict[str, Any]:
        """Mixup kwargs for the data loader."""
        return {
            "mixup_alpha": self.mixup_alpha,
            "cutmix_alpha": self.cutmix_alpha,
            "cutmix_minmax": self.cutmix_minmax,
            "prob": self.mixup_prob,
            "switch_prob": self.mixup_switch_prob,
            "mode": self.mixup_mode,
            "label_smoothing": self.label_smoothing,
            "num_classes": self.num_classes,
        }

    @cached_property
    def distributed_results(self) -> dict[str, Any]:
        """Distributed results for the data loader."""
        return initial_distributed_env(device=self.device, dist_backend=self.dist_backend, dist_url=self.dist_url)

    @cached_property
    def distributed(self) -> bool:
        """Whether the data loader is distributed."""
        return self.distributed_results["distributed"]

    @cached_property
    def default_kwargs(self) -> dict[str, Any]:
        """Default kwargs for the data loader."""
        kwargs: dict[str, Any] = {}
        kwargs["input_size"] = self.input_size
        kwargs["interpolation"] = self.interpolation
        kwargs["num_workers"] = self.num_workers
        kwargs["pin_memory"] = self.pin_memory
        kwargs["mean"] = self.mean
        kwargs["std"] = self.std
        kwargs["img_dtype"] = DTYPES[self.image_dtype]
        kwargs["device"] = torch.device(self.distributed_results["device"])
        kwargs["distributed"] = self.distributed
        kwargs["use_prefetcher"] = self.use_prefetcher
        if self.dataset.is_training:
            kwargs["no_aug"] = self.no_aug
            kwargs["re_prob"] = self.re_prob
            kwargs["re_mode"] = self.re_mode
            kwargs["re_count"] = self.re_count
            kwargs["re_split"] = self.re_split
            kwargs["train_crop_mode"] = self.train_crop_mode
            kwargs["scale"] = self.scale
            kwargs["ratio"] = self.ratio
            kwargs["hflip"] = self.hflip
            kwargs["vflip"] = self.vflip
            kwargs["color_jitter"] = self.color_jitter
            kwargs["color_jitter_prob"] = self.color_jitter_prob
            kwargs["grayscale_prob"] = self.grayscale_prob
            kwargs["gaussian_blur_prob"] = self.gaussian_blur_prob
            kwargs["auto_augment"] = self.auto_augment
            kwargs["num_aug_repeats"] = self.num_aug_repeats
            kwargs["num_aug_splits"] = self.num_aug_splits
            kwargs["use_multi_epochs_loader"] = self.use_multi_epochs_loader
            kwargs["worker_seeding"] = self.worker_seeding
        else:
            kwargs["crop_pct"] = self.crop_pct
        return kwargs

    @cached_property
    def mixup(self) -> Mixup:
        """Create a Mixup function if mixup or cutmix is active."""
        if self.mixup_active:
            return (FastCollateMixup if self.use_prefetcher else Mixup)(**self.mixup_kwargs)
        raise ValueError("Mixup is not active, cannot create mixup function.")

    def disable_mixup(self, info: BaseInfo, **models: torch.nn.Module) -> None:
        """Disable mixup after the specified epoch."""
        if info.epoch >= self.mixup_off_epoch:
            self.mixup.mixup_enabled = False

    @cached_property
    def dataset_wrapper(self) -> TimmDatasetWrapper:
        """Return the dataset wrapper."""
        dataset = self.dataset.dataset
        if self.dataset.is_training and self.num_aug_splits > 1:
            dataset = AugMixDataset(dataset, num_splits=self.num_aug_splits)
        return dataset

    def set_dataset_epoch(self, info: BaseInfo, **models: torch.nn.Module) -> None:
        """Set the epoch for the dataset if it has a set_epoch method."""
        self.dataset_wrapper.set_epoch(info.epoch - 1)

    def set_dataloader_epoch(self, info: BaseInfo, **models: torch.nn.Module) -> None:
        """Set the epoch for the data loader if it has a set_epoch method."""
        self.dataloader.sampler.set_epoch(info.epoch - 1)

    @cached_property
    def dataloader(self) -> DataLoader:
        """Create a data loader using the timm library."""
        collate_fn, dataset = None, self.dataset_wrapper
        if self.dataset.is_training:
            if self.mixup_active:
                if self.use_prefetcher:
                    collate_fn = self.mixup
                if self.mixup_off_epoch:
                    GLOBAL_CALLBACKS.on_training_begin.register("disable_mixup", self.disable_mixup)
        loader = create_loader(
            dataset=dataset,
            batch_size=self.dataset.batch_size,
            is_training=self.dataset.is_training,
            collate_fn=collate_fn,
            **self.default_kwargs,
            **self.model_extra,
        )
        if self.dataset.is_training:
            if hasattr(dataset, "set_epoch"):
                GLOBAL_CALLBACKS.on_epoch_begin.register("set_dataset_epoch", self.set_dataset_epoch)
            elif self.distributed and hasattr(loader.sampler, "set_epoch"):
                GLOBAL_CALLBACKS.on_epoch_begin.register("set_dataloader_epoch", self.set_dataloader_epoch)
        return loader

    def __len__(self) -> int:
        """Return the length of the data loader."""
        return len(self.dataloader)

    def _call(self) -> Iterable[tuple[Tensor, Tensor]]:
        """Return the data loader."""
        if self.use_prefetcher:
            if self.channels_last:
                for inp, target in self.dataloader:
                    yield inp.contiguous(memory_format=torch.channels_last), target
            else:
                yield from self.dataloader
        else:
            device, dtype = self.default_kwargs["device"], self.default_kwargs["img_dtype"]
            mixup = self.mixup if self.mixup_active else None
            for inp, target in self.dataloader:
                inp, target = inp.to(device=device, dtype=dtype), target.to(device=device)
                if mixup is not None:
                    inp, target = mixup(inp, target)
                if self.channels_last:
                    inp = inp.contiguous(memory_format=torch.channels_last)
                yield inp, target

    def __call__(self) -> Any:
        """Return the data loader outputs, optionally applying a FlexSpec to map the outputs to the model inputs."""
        if self.spec is None:
            yield from self._call()
        else:
            yield from map(self.spec, self._call())


__all__ = [
    "CriteriaTracker",
    "TimmDataLoaderWrapper",
    "TimmDatasetWrapper",
    "TimmEmaWrapper",
    "TorchBestCriterion",
    "TorchTracker",
    "TorchTrainer",
    "TrainingStep",
    "ValidationStep",
    "create_torch_inputs",
    "get_autocast",
    "get_torch_device",
    "initial_distributed_env",
    "initial_model",
]


if not TYPE_CHECKING:
    import sys

    from structcast.utils.lazy_import import LazySelectedImporter

    sys.modules[__name__] = LazySelectedImporter(__name__, globals())
