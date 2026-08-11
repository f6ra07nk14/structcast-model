"""Example timm data integration for PyTorch training runs.

The package knows nothing about timm datasets: `scm torch train` instantiates whatever the
`--training-dataset` / `--validation-dataset` patterns describe and routes any dataset implementing
an event protocol into the trainer's callbacks. The timm wrappers below are therefore example code,
referenced from a configuration by file path:

```yaml
_obj_:
  - _addr_: TimmDataLoaderWrapper
    _file_: examples/torch/data.py
  - _attr_: model_validate
  - - _call_
    - dataset: {name: torch/cifar100, root: data, is_training: true, split: train}
      input_size: [3, 224, 224]
```

`TimmDataLoaderWrapper` implements `on_epoch_begin` and `on_training_begin`, so the trainer reshuffles
the distributed sampler and turns mixup off at its cutoff epoch without any registration.
`TimmDataProvider` is the programmatic counterpart: give it to a trainer as `data=` and it forwards
both events to the training wrapper.
"""

from collections.abc import Iterable
from functools import cached_property
from typing import Any, Literal

from pydantic import BaseModel, Field
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
from torch.utils.data import DataLoader

from structcast_model.base_trainer import BaseInfo
from structcast_model.torch.trainer import initial_distributed_env
from structcast_model.torch.types import Tensor
import torch

DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "int32": torch.int32,
    "int64": torch.int64,
}


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

    def on_training_begin(self, info: BaseInfo, **models: torch.nn.Module) -> None:
        """Disable mixup once the configured epoch is reached.

        Safe to call on every training begin: it is a no-op unless this is a training split whose
        active mixup is configured to stop at `mixup_off_epoch`.
        """
        if (
            self.dataset.is_training
            and self.mixup_active
            and self.mixup_off_epoch
            and info.epoch >= self.mixup_off_epoch
        ):
            self.mixup.mixup_enabled = False

    @cached_property
    def dataset_wrapper(self) -> TimmDatasetWrapper:
        """Return the dataset wrapper."""
        dataset = self.dataset.dataset
        if self.dataset.is_training and self.num_aug_splits > 1:
            dataset = AugMixDataset(dataset, num_splits=self.num_aug_splits)
        return dataset

    def on_epoch_begin(self, info: BaseInfo, **models: torch.nn.Module) -> None:
        """Tell the dataset or the distributed sampler which epoch is starting, so shuffling varies.

        Safe to call on every epoch begin: it is a no-op unless this is a training split whose
        dataset or sampler supports `set_epoch`. Epochs are 1-based here and 0-based there.
        """
        if not self.dataset.is_training:
            return
        if hasattr(self.dataset_wrapper, "set_epoch"):
            self.dataset_wrapper.set_epoch(info.epoch - 1)
        elif self.distributed and hasattr(self.dataloader.sampler, "set_epoch"):
            self.dataloader.sampler.set_epoch(info.epoch - 1)

    @cached_property
    def dataloader(self) -> DataLoader:
        """Create a data loader using the timm library."""
        collate_fn, dataset = None, self.dataset_wrapper
        if self.dataset.is_training and self.mixup_active and self.use_prefetcher:
            collate_fn = self.mixup
        return create_loader(
            dataset=dataset,
            batch_size=self.dataset.batch_size,
            is_training=self.dataset.is_training,
            collate_fn=collate_fn,
            **self.default_kwargs,
            **self.model_extra,
        )

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


class TimmDataProvider(BaseModel):
    """Data provider supplying timm data loaders and keeping the training split in sync with the loop.

    The trainer scans the provider for event protocols, so the epoch synchronization and the mixup
    cutoff of the training wrapper run without any registration.
    """

    training: TimmDataLoaderWrapper
    """The wrapper producing the training dataset."""

    validation: Any = None
    """The validation dataset: a timm wrapper, any other dataset object, or None to skip validation.

    Only the training wrapper needs per-epoch hooks, so the validation side accepts any dataset."""

    @property
    def training_dataset(self) -> TimmDataLoaderWrapper:
        """The dataset used for training: the wrapper is callable and yields the loader outputs."""
        return self.training

    @property
    def validation_dataset(self) -> Any:
        """The dataset used for validation, or None when no validation dataset was given."""
        return self.validation

    def on_epoch_begin(self, info: BaseInfo, **models: torch.nn.Module) -> None:
        """Forward the new epoch to the training wrapper."""
        self.training.on_epoch_begin(info, **models)

    def on_training_begin(self, info: BaseInfo, **models: torch.nn.Module) -> None:
        """Let the training wrapper turn mixup off once its cutoff epoch is reached."""
        self.training.on_training_begin(info, **models)
