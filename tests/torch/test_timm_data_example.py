"""Unit tests for the timm data integration example in examples/torch/data.py."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
from PIL import Image
import pytest
from timm.data import AugMixDataset, FastCollateMixup, ImageDataset, Mixup

from structcast_model.base_trainer import BaseInfo, DataProvider, SimpleDataProvider
from structcast_model.torch.trainer import TorchTracker, TorchTrainer
import torch


def _load_example_module() -> Any:
    """Load the example module the way a configuration does: by file path, not by import name."""
    path = Path(__file__).resolve().parents[2] / "examples" / "torch" / "data.py"
    spec = importlib.util.spec_from_file_location("example_data", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_EXAMPLE = _load_example_module()
TimmDatasetWrapper = _EXAMPLE.TimmDatasetWrapper
TimmDataLoaderWrapper = _EXAMPLE.TimmDataLoaderWrapper
TimmDataProvider = _EXAMPLE.TimmDataProvider


class _StubLearner:
    """A minimal stub implementing the Learner protocol, for the trainer routing test."""

    models: dict[str, Any] = {}
    optimizers: dict[str, Any] = {}
    learning_rates: dict[str, float] = {}

    def update(self, step: int) -> bool:
        """Always signal that an update should occur."""
        return True

    def training_step(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """No-op training step."""
        return {}

    def inference_step(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """No-op inference step."""
        return {}


def _populate_image_folder(root: Path, *, num_classes: int = 2, images_per_class: int = 4) -> Path:
    """Create an ImageFolder-compatible directory tree with random PNG images.

    Returns the *root* path so callers can pass it directly to ``TimmDatasetWrapper(root=...)``.
    """
    rng = np.random.default_rng(0)
    for cls_idx in range(num_classes):
        cls_dir = root / f"class_{cls_idx}"
        cls_dir.mkdir(parents=True, exist_ok=True)
        for img_idx in range(images_per_class):
            arr = rng.integers(0, 255, (32, 32, 3), dtype=np.uint8)
            Image.fromarray(arr).save(cls_dir / f"{img_idx}.png")
    return root


def _training_wrapper(**kwargs: Any) -> Any:
    """Return a wrapper whose dataset is a training split, without touching any real data."""
    return TimmDataLoaderWrapper(dataset=TimmDatasetWrapper(is_training=True), **kwargs)


# ---------------------------------------------------------------------------
# TimmDatasetWrapper
# ---------------------------------------------------------------------------


def test_timm_dataset_wrapper_default_kwargs_contains_all_keys() -> None:
    """default_kwargs exposes all fields required by create_dataset."""
    ds = TimmDatasetWrapper()
    kwargs = ds.default_kwargs
    for key in (
        "name",
        "root",
        "split",
        "is_training",
        "seed",
        "batch_size",
        "class_map",
        "download",
        "repeats",
        "input_img_mode",
        "input_key",
        "target_key",
        "trust_remote_code",
        "num_samples",
    ):
        assert key in kwargs, f"Missing key: {key}"


def test_timm_dataset_wrapper_dataset_calls_create_dataset(tmp_path: Path) -> None:
    """Dataset cached_property delegates to create_dataset."""
    _populate_image_folder(tmp_path)
    ds = TimmDatasetWrapper(name="", root=str(tmp_path))
    assert isinstance(ds.dataset, ImageDataset)
    assert len(ds.dataset) == 8  # 2 classes × 4 images


# ---------------------------------------------------------------------------
# TimmDataLoaderWrapper – utility properties
# ---------------------------------------------------------------------------


def test_timm_dataloader_mixup_active_false_by_default() -> None:
    """mixup_active returns False when all alpha/cutmix values are at defaults."""
    assert TimmDataLoaderWrapper().mixup_active is False


def test_timm_dataloader_mixup_active_true_with_mixup_alpha() -> None:
    """mixup_active returns True when mixup_alpha > 0."""
    assert TimmDataLoaderWrapper(mixup_alpha=0.2).mixup_active is True


def test_timm_dataloader_mixup_active_true_with_cutmix_alpha() -> None:
    """mixup_active returns True when cutmix_alpha > 0."""
    assert TimmDataLoaderWrapper(cutmix_alpha=0.2).mixup_active is True


def test_timm_dataloader_mixup_kwargs_contains_expected_keys() -> None:
    """mixup_kwargs exposes all fields expected by timm mixup constructors."""
    kwargs = TimmDataLoaderWrapper().mixup_kwargs
    for key in (
        "mixup_alpha",
        "cutmix_alpha",
        "cutmix_minmax",
        "prob",
        "switch_prob",
        "mode",
        "label_smoothing",
        "num_classes",
    ):
        assert key in kwargs, f"Missing mixup kwarg: {key}"


# ---------------------------------------------------------------------------
# TimmDataLoaderWrapper – distributed_results & default_kwargs
# ---------------------------------------------------------------------------


def test_timm_dataloader_distributed_results() -> None:
    """distributed_results calls init_distributed_device_so."""
    result = TimmDataLoaderWrapper().distributed_results
    assert result["device"] == "cpu"
    assert result["distributed"] is False


def test_timm_dataloader_default_kwargs_validation_branch() -> None:
    """default_kwargs includes crop_pct (not training kwargs) when is_training=False."""
    kwargs = TimmDataLoaderWrapper().default_kwargs
    assert "crop_pct" in kwargs
    assert "no_aug" not in kwargs


def test_timm_dataloader_default_kwargs_training_branch() -> None:
    """default_kwargs includes training-specific keys when is_training=True."""
    kwargs = _training_wrapper().default_kwargs
    assert "no_aug" in kwargs
    assert "re_prob" in kwargs
    assert "auto_augment" in kwargs
    assert "crop_pct" not in kwargs


# ---------------------------------------------------------------------------
# TimmDataLoaderWrapper – mixup cached_property
# ---------------------------------------------------------------------------


def test_timm_dataloader_mixup_raises_when_inactive() -> None:
    """Accessing mixup when mixup is not active raises ValueError."""
    with pytest.raises(ValueError, match="Mixup is not active"):
        _ = TimmDataLoaderWrapper().mixup


def test_timm_dataloader_mixup_returns_fast_collate_with_prefetcher() -> None:
    """With use_prefetcher=True and mixup_alpha>0, mixup returns FastCollateMixup."""
    assert isinstance(TimmDataLoaderWrapper(mixup_alpha=0.4, use_prefetcher=True).mixup, FastCollateMixup)


def test_timm_dataloader_mixup_returns_mixup_without_prefetcher() -> None:
    """With use_prefetcher=False and mixup_alpha>0, mixup returns Mixup."""
    assert isinstance(TimmDataLoaderWrapper(mixup_alpha=0.4, use_prefetcher=False).mixup, Mixup)


# ---------------------------------------------------------------------------
# TimmDataLoaderWrapper – on_training_begin (the mixup cutoff)
# ---------------------------------------------------------------------------


def test_timm_dataloader_on_training_begin_disables_mixup_when_epoch_reached() -> None:
    """Mixup stops at mixup_off_epoch so the last epochs train on unmixed samples."""
    wrapper = _training_wrapper(mixup_alpha=0.5, mixup_off_epoch=3)
    wrapper.on_training_begin(BaseInfo(epoch=3))
    assert wrapper.mixup.mixup_enabled is False


def test_timm_dataloader_on_training_begin_is_a_noop_before_the_cutoff() -> None:
    """Before the cutoff epoch mixup stays on."""
    wrapper = _training_wrapper(mixup_alpha=0.5, mixup_off_epoch=5)
    wrapper.on_training_begin(BaseInfo(epoch=2))
    assert wrapper.mixup.mixup_enabled is True


@pytest.mark.parametrize(
    "kwargs",
    [
        {"mixup_alpha": 0.5, "mixup_off_epoch": 0},
        {"mixup_alpha": 0.0, "mixup_off_epoch": 3},
    ],
    ids=["no_cutoff_epoch", "mixup_inactive"],
)
def test_timm_dataloader_on_training_begin_is_safe_without_a_mixup_cutoff(kwargs: dict[str, Any]) -> None:
    """The trainer calls this on every training begin, so the guards must keep it a no-op, not an error."""
    _training_wrapper(**kwargs).on_training_begin(BaseInfo(epoch=9))


def test_timm_dataloader_on_training_begin_ignores_a_validation_split() -> None:
    """Mixup is a training-only augmentation: a validation wrapper must not build one."""
    wrapper = TimmDataLoaderWrapper(mixup_alpha=0.5, mixup_off_epoch=1)
    wrapper.on_training_begin(BaseInfo(epoch=9))
    assert wrapper.mixup.mixup_enabled is True


# ---------------------------------------------------------------------------
# TimmDataLoaderWrapper – on_epoch_begin (the reshuffling hook)
# ---------------------------------------------------------------------------


def test_timm_dataloader_on_epoch_begin_forwards_zero_based_epoch_to_the_dataset() -> None:
    """Datasets reshuffle per epoch; the trainer counts from 1 and the dataset from 0."""
    seen: list[int] = []
    wrapper = _training_wrapper()
    wrapper.__dict__["dataset_wrapper"] = SimpleNamespace(set_epoch=seen.append)
    wrapper.on_epoch_begin(BaseInfo(epoch=4))
    assert seen == [3]


def test_timm_dataloader_on_epoch_begin_falls_back_to_the_distributed_sampler() -> None:
    """Without dataset support, the distributed sampler is what keeps ranks shuffling in step."""
    seen: list[int] = []
    wrapper = _training_wrapper()
    wrapper.__dict__["dataset_wrapper"] = object()
    wrapper.__dict__["distributed"] = True
    wrapper.__dict__["dataloader"] = SimpleNamespace(sampler=SimpleNamespace(set_epoch=seen.append))
    wrapper.on_epoch_begin(BaseInfo(epoch=1))
    assert seen == [0]


def test_timm_dataloader_on_epoch_begin_ignores_a_validation_split() -> None:
    """A validation split is not reshuffled, so its dataset must never be told the epoch."""
    seen: list[int] = []
    wrapper = TimmDataLoaderWrapper()
    wrapper.__dict__["dataset_wrapper"] = SimpleNamespace(set_epoch=seen.append)
    wrapper.on_epoch_begin(BaseInfo(epoch=4))
    assert seen == []


def test_timm_dataloader_is_routed_into_the_epoch_events_by_the_trainer() -> None:
    """The renamed hooks are what let the trainer pick the dataset up from the provider scan."""
    trainer = TorchTrainer(
        device="cpu",
        learner=_StubLearner(),
        tracker=TorchTracker.from_criteria(["loss"], distributed=False),
        data=SimpleDataProvider(training_dataset=_training_wrapper()),
    )
    described = trainer.describe()
    assert described["on_epoch_begin"] == ["TimmDataLoaderWrapper"]
    assert described["on_training_begin"] == ["TorchTracker", "TimmDataLoaderWrapper"]


# ---------------------------------------------------------------------------
# TimmDataLoaderWrapper – dataloader, __len__, _call, __call__
# ---------------------------------------------------------------------------


_LOADER_BASE_KWARGS: dict[str, Any] = {
    "input_size": (3, 32, 32),
    "num_workers": 0,
    "persistent_workers": False,
}
"""Shared kwargs for all ``TimmDataLoaderWrapper`` instances in tests that need a real loader."""


@pytest.fixture
def image_folder(tmp_path: Path) -> Path:
    """Create a minimal ImageFolder tree and return its root."""
    return _populate_image_folder(tmp_path)


def test_timm_dataloader_wrapper_dataloader_validation(image_folder: Path) -> None:
    """Dataloader property returns the object from create_loader in validation mode."""
    loader = TimmDataLoaderWrapper(
        dataset=TimmDatasetWrapper(name="", root=str(image_folder), batch_size=2),
        **_LOADER_BASE_KWARGS,
    ).dataloader
    assert len(loader) > 0


def test_timm_dataloader_wrapper_dataloader_training_no_mixup(image_folder: Path) -> None:
    """Dataloader is obtained in training mode without mixup."""
    loader = TimmDataLoaderWrapper(
        dataset=TimmDatasetWrapper(name="", root=str(image_folder), batch_size=2, is_training=True),
        **_LOADER_BASE_KWARGS,
    ).dataloader
    assert len(loader) > 0


def test_timm_dataloader_wrapper_dataloader_training_with_mixup_collates_with_mixup(image_folder: Path) -> None:
    """With the prefetcher, mixup happens in the collate function of the loader."""
    loader = TimmDataLoaderWrapper(
        dataset=TimmDatasetWrapper(name="", root=str(image_folder), batch_size=2, is_training=True),
        mixup_alpha=0.5,
        mixup_off_epoch=3,
        use_prefetcher=True,
        num_classes=2,
        **_LOADER_BASE_KWARGS,
    ).dataloader
    assert isinstance(loader.loader.collate_fn, FastCollateMixup)


def test_timm_dataloader_wrapper_dataloader_with_aug_splits(image_folder: Path) -> None:
    """num_aug_splits>1 wraps the dataset in AugMixDataset."""
    wrapper = TimmDataLoaderWrapper(
        dataset=TimmDatasetWrapper(name="", root=str(image_folder), batch_size=2, is_training=True),
        num_aug_splits=2,
        **_LOADER_BASE_KWARGS,
    )
    assert isinstance(wrapper.dataset_wrapper, AugMixDataset)


def test_timm_dataloader_wrapper_len(image_folder: Path) -> None:
    """__len__ delegates to the underlying dataloader."""
    wrapper = TimmDataLoaderWrapper(
        dataset=TimmDatasetWrapper(name="", root=str(image_folder), batch_size=2),
        **_LOADER_BASE_KWARGS,
    )
    assert len(wrapper) == len(wrapper.dataloader)


def test_timm_dataloader_call_prefetcher_no_channels_last(image_folder: Path) -> None:
    """_call with prefetcher=True channels_last=False yields from dataloader directly."""
    wrapper = TimmDataLoaderWrapper(
        dataset=TimmDatasetWrapper(name="", root=str(image_folder), batch_size=2),
        use_prefetcher=True,
        channels_last=False,
        **_LOADER_BASE_KWARGS,
    )
    batches = list(wrapper._call())
    assert len(batches) > 0
    inp, _ = batches[0]
    assert inp.shape[1:] == (3, 32, 32)


def test_timm_dataloader_call_prefetcher_channels_last(image_folder: Path) -> None:
    """_call with prefetcher=True channels_last=True yields channels_last tensors."""
    wrapper = TimmDataLoaderWrapper(
        dataset=TimmDatasetWrapper(name="", root=str(image_folder), batch_size=2),
        use_prefetcher=True,
        channels_last=True,
        **_LOADER_BASE_KWARGS,
    )
    batches = list(wrapper._call())
    assert len(batches) > 0
    inp, _ = batches[0]
    assert inp.is_contiguous(memory_format=torch.channels_last)


def test_timm_dataloader_call_no_prefetcher(image_folder: Path) -> None:
    """_call with prefetcher=False moves tensors to device/dtype."""
    wrapper = TimmDataLoaderWrapper(
        dataset=TimmDatasetWrapper(name="", root=str(image_folder), batch_size=2),
        use_prefetcher=False,
        **_LOADER_BASE_KWARGS,
    )
    batches = list(wrapper._call())
    assert len(batches) > 0


def test_timm_dataloader_call_no_prefetcher_with_mixup(image_folder: Path) -> None:
    """_call with prefetcher=False and mixup_alpha>0 applies Mixup to each batch."""
    wrapper = TimmDataLoaderWrapper(
        dataset=TimmDatasetWrapper(name="", root=str(image_folder), batch_size=2, is_training=True),
        use_prefetcher=False,
        mixup_alpha=0.4,
        num_classes=2,
        **_LOADER_BASE_KWARGS,
    )
    batches = list(wrapper._call())
    assert len(batches) > 0


def test_timm_dataloader_call_no_prefetcher_channels_last(image_folder: Path) -> None:
    """_call with prefetcher=False channels_last=True applies channels_last format."""
    wrapper = TimmDataLoaderWrapper(
        dataset=TimmDatasetWrapper(name="", root=str(image_folder), batch_size=2),
        use_prefetcher=False,
        channels_last=True,
        **_LOADER_BASE_KWARGS,
    )
    inp, _ = next(iter(wrapper._call()))
    assert inp.is_contiguous(memory_format=torch.channels_last)


def test_timm_dataloader_dunder_call_no_spec(image_folder: Path) -> None:
    """__call__ with spec=None yields raw (inp, target) pairs."""
    wrapper = TimmDataLoaderWrapper(
        dataset=TimmDatasetWrapper(name="", root=str(image_folder), batch_size=2),
        spec=None,
        **_LOADER_BASE_KWARGS,
    )
    batches = list(wrapper())
    assert len(batches) > 0
    inp, target = batches[0]
    assert isinstance(inp, torch.Tensor)


def test_timm_dataloader_dunder_call_with_spec(image_folder: Path) -> None:
    """__call__ with a spec applies map(spec, _call())."""
    wrapper = TimmDataLoaderWrapper(
        dataset=TimmDatasetWrapper(name="", root=str(image_folder), batch_size=2),
        spec=None,
        **_LOADER_BASE_KWARGS,
    )
    results: list[Any] = []

    def fake_spec(x: Any) -> Any:
        results.append(x)
        return x

    # bypass Pydantic validation to set a plain callable
    wrapper.__dict__["spec"] = fake_spec
    list(wrapper())
    assert len(results) > 0


# ---------------------------------------------------------------------------
# TimmDataProvider
# ---------------------------------------------------------------------------


def test_timm_data_provider_satisfies_the_data_provider_protocol() -> None:
    """The trainer accepts any DataProvider; the provider must qualify without inheriting anything."""
    provider = TimmDataProvider(training=_training_wrapper())
    assert isinstance(provider, DataProvider)
    assert provider.training_dataset is provider.training
    assert provider.validation_dataset is None


def test_timm_data_provider_exposes_the_validation_wrapper_as_dataset() -> None:
    """A validation wrapper is what makes fit() evaluate, so it must surface as the dataset."""
    validation = TimmDataLoaderWrapper()
    provider = TimmDataProvider(training=_training_wrapper(), validation=validation)
    assert provider.validation_dataset is validation


def test_timm_data_provider_counts_validation_steps_of_a_plain_dataset() -> None:
    """Anything with a length works as validation data, so the count must not require a wrapper."""
    provider = TimmDataProvider(training=_training_wrapper(), validation=[{"x": 1}, {"x": 2}])
    assert provider.validation_steps == 2


def test_timm_data_provider_reports_zero_validation_steps_when_absent() -> None:
    """No validation dataset means fit() skips validation, so the count must be 0, not an error."""
    assert TimmDataProvider(training=_training_wrapper()).validation_steps == 0


def test_timm_data_provider_wrappers_are_routed_into_the_epoch_events_by_the_trainer() -> None:
    """Passing the provider to the trainer is the whole wiring: its datasets are scanned directly.

    The provider forwards nothing. A validation wrapper registers too -- its hooks no-op
    internally (see the split-guard tests above).
    """
    trainer = TorchTrainer(
        device="cpu",
        learner=_StubLearner(),
        tracker=TorchTracker.from_criteria(["loss"], distributed=False),
        data=TimmDataProvider(training=_training_wrapper(), validation=TimmDataLoaderWrapper()),
    )
    described = trainer.describe()
    assert described["on_epoch_begin"] == ["TimmDataLoaderWrapper", "TimmDataLoaderWrapper"]
    assert described["on_training_begin"] == ["TorchTracker", "TimmDataLoaderWrapper", "TimmDataLoaderWrapper"]
