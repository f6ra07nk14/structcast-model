"""Unit tests for structcast_model.torch.trainer - utility functions and classes."""

from __future__ import annotations

import logging
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any

import mlflow
import numpy as np
from PIL import Image
import pytest
from structcast.utils.base import configure_security
from timm.data import AugMixDataset, FastCollateMixup, ImageDataset, Mixup
from torch.nn import Module

from structcast_model.base_trainer import BaseInfo, DataProvider
from structcast_model.torch.trainer import (
    MLflowLogger,
    TimmDataLoaderWrapper,
    TimmDataProvider,
    TimmDatasetWrapper,
    TorchLearnerFactory,
    TorchTracker,
    TorchTrainer,
    WandbLogger,
    autocast_inputs,
    create_torch_inputs,
    get_torch_device,
    initial_distributed_env,
    initial_model,
)
import torch

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def allow_module_imports() -> Any:
    """Allow `_INIT_` addresses to be imported, then restore the default security settings."""
    configure_security()
    yield
    configure_security()


class _IdentityModel(Module):
    """A model that passes all inputs through unchanged."""

    def forward(self, **kwargs: Any) -> dict[str, Any]:
        return {}


class _LossModule(Module):
    """A loss module that always returns a fixed loss tensor."""

    def forward(self, **kwargs: Any) -> dict[str, torch.Tensor]:
        return {"loss": torch.tensor(0.5)}


class _StubLearner:
    """A minimal stub implementing the Learner protocol for tests that don't exercise a real step."""

    def __init__(self, models: dict[str, Any] | None = None, learning_rates: dict[str, float] | None = None) -> None:
        """Initialize with optional models and the learning rates a real learner would report."""
        self._models = models or {}
        self.learning_rates = learning_rates or {}

    @property
    def models(self) -> dict[str, Any]:
        """Return the models dict."""
        return self._models

    def update(self, step: int) -> bool:
        """Always signal that an update should occur."""
        return True

    def training_step(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """No-op training step."""
        return {}

    def inference_step(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """No-op inference step."""
        return {}


class _MetricModule(Module):
    """A metric module that always returns a fixed accuracy tensor."""

    def forward(self, **kwargs: Any) -> dict[str, torch.Tensor]:
        return {"acc": torch.tensor(0.9)}


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


# ---------------------------------------------------------------------------
# create_torch_inputs
# ---------------------------------------------------------------------------


def test_create_torch_inputs_from_int_tuple_returns_tensor() -> None:
    """A tuple of ints produces a bfloat16 tensor with batch dimension 1, bfloat16 being the default dtype."""
    result = create_torch_inputs((3, 4))
    assert isinstance(result, torch.Tensor)
    assert result.shape == (1, 3, 4)
    assert result.dtype == torch.bfloat16


def test_create_torch_inputs_from_list_returns_list() -> None:
    """A list of shapes returns a list of tensors."""
    result = create_torch_inputs([(3,), (4,)])
    assert isinstance(result, list)
    assert len(result) == 2
    assert all(isinstance(t, torch.Tensor) for t in result)


def test_create_torch_inputs_from_dict_returns_dict() -> None:
    """A dict of shapes returns a dict of tensors."""
    result = create_torch_inputs({"image": (3, 4), "mask": (1, 4)})
    assert isinstance(result, dict)
    assert set(result.keys()) == {"image", "mask"}
    assert all(isinstance(v, torch.Tensor) for v in result.values())


def test_create_torch_inputs_invalid_shape_raises() -> None:
    """A non-shape scalar raises ValueError."""
    with pytest.raises(ValueError, match="Invalid tensor shape"):
        create_torch_inputs("not_a_shape")


def test_create_torch_inputs_int_dtype_falls_back_to_zeros_with_warning(caplog: pytest.LogCaptureFixture) -> None:
    """An integer dtype without an initializer falls back to zeros, because rand cannot produce integers.

    The fallback is a guess about the caller's intent, so it must be reported.
    """
    with caplog.at_level(logging.WARNING):
        result = create_torch_inputs({"_SHAPE_": [5], "_DTYPE_": "int64"})
    assert result.dtype == torch.int64
    assert torch.equal(result, torch.zeros((1, 5), dtype=torch.int64))
    assert "Falling back to zeros" in caplog.text


def test_create_torch_inputs_honours_explicit_initializer(allow_module_imports: None) -> None:
    """An explicit `_INIT_` address replaces the dtype-based default initializer."""
    result = create_torch_inputs({"_SHAPE_": [4], "_INIT_": "torch.ones"})
    assert torch.equal(result, torch.ones((1, 4), dtype=torch.bfloat16))


def test_create_torch_inputs_rejects_non_callable_initializer(allow_module_imports: None) -> None:
    """A `_INIT_` address resolving to a non-callable is rejected, instead of failing later at call time."""
    with pytest.raises(TypeError, match="not callable as a tensor initializer"):
        create_torch_inputs({"_SHAPE_": [4], "_INIT_": "torch.pi"})


# ---------------------------------------------------------------------------
# get_torch_device
# ---------------------------------------------------------------------------


def test_get_torch_device_returns_cpu_when_explicit() -> None:
    """Passing 'cpu' always returns 'cpu'."""
    assert get_torch_device("cpu") == "cpu"


def test_get_torch_device_returns_cpu_for_none_when_cuda_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    """Returns 'cpu' when device=None and CUDA is unavailable."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert get_torch_device(None) == "cpu"


def test_get_torch_device_returns_cuda_for_none_when_cuda_available(monkeypatch: pytest.MonkeyPatch) -> None:
    """Returns 'cuda' when device=None and CUDA is available."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert get_torch_device(None) == "cuda"


def test_get_torch_device_cuda_falls_back_to_cpu_when_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    """'cuda' requested but unavailable falls back to 'cpu' with a warning."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    result = get_torch_device("cuda")
    assert result == "cpu"


def test_get_torch_device_raises_for_invalid_device() -> None:
    """Unsupported device string raises ValueError."""
    with pytest.raises(ValueError, match="invalid device"):
        get_torch_device("tpu")


# ---------------------------------------------------------------------------
# initial_model
# ---------------------------------------------------------------------------


def test_initial_model_returns_model_none_inputs_when_shapes_is_none() -> None:
    """When shapes=None, inputs is None and the model is returned unchanged."""
    model = _IdentityModel()
    inputs, outputs = initial_model(model, shapes=None)
    assert inputs is None


def test_initial_model_runs_forward_when_shapes_provided() -> None:
    """With shapes provided, forward() is called and inputs are returned."""

    class SimpleModel(Module):
        def forward(self, x: torch.Tensor) -> dict[str, Any]:
            return {}

    model = SimpleModel()
    inputs, outputs = initial_model(model, shapes={"x": (3,)})
    assert inputs is not None
    assert "x" in inputs


def test_initial_model_applies_compile_fn() -> None:
    """initial_model returns 2-tuple without compile_fn parameter."""
    model = _IdentityModel()
    inputs, outputs = initial_model(model, shapes=None)
    assert inputs is None


def test_initial_model_handles_dict_of_modules() -> None:
    """A dict of modules is handled correctly."""
    models = {"a": _IdentityModel(), "b": _IdentityModel()}
    inputs, outputs = initial_model(models, shapes=None)
    assert inputs is None


def test_initial_model_handles_list_of_modules() -> None:
    """A list of modules is handled correctly."""
    models = [_IdentityModel(), _IdentityModel()]
    inputs, outputs = initial_model(models, shapes=None)
    assert inputs is None


def test_initial_model_falls_back_to_the_shapes_declared_by_the_model() -> None:
    """Without requested shapes, the model is initialized from the `input_shapes` the builder emitted."""

    class DeclaredModel(Module):
        input_shapes = {"x": (3,)}

        def forward(self, x: torch.Tensor) -> dict[str, Any]:
            return {"x": x}

    inputs, outputs = initial_model(DeclaredModel())
    assert inputs["x"].shape == (1, 3)
    assert outputs["x"].shape == (1, 3)


def test_initial_model_runs_low_precision_inputs_through_float32_parameters() -> None:
    """Dummy inputs default to `bfloat16` while parameters stay `float32`, so the forward pass needs autocast."""
    model = torch.nn.Linear(4, 2)
    inputs, outputs = initial_model(model, shapes={"input": [4]})
    assert inputs["input"].dtype is torch.bfloat16
    assert next(model.parameters()).dtype is torch.float32
    assert outputs.dtype is torch.bfloat16


def test_autocast_inputs_is_a_null_context_for_float32_inputs() -> None:
    """Inputs that already match `float32` parameters must keep running without autocast."""
    with autocast_inputs({"x": torch.rand((1, 4), dtype=torch.float32)}, "cpu"):
        assert not torch.is_autocast_enabled("cpu")


# ---------------------------------------------------------------------------
# TorchTracker
# ---------------------------------------------------------------------------


def test_torch_tracker_from_criteria_creates_tracker() -> None:
    """TorchTracker.from_criteria returns a valid TorchTracker."""
    tracker = TorchTracker.from_criteria(["loss"])
    assert isinstance(tracker, TorchTracker)


def test_torch_tracker_from_criteria_with_metric_outputs() -> None:
    """TorchTracker.from_criteria accepts a combined outputs list."""
    tracker = TorchTracker.from_criteria(["loss", "acc"])
    assert isinstance(tracker, TorchTracker)


def test_torch_tracker_call_returns_float_values() -> None:
    """__call__ returns a dict of float values from Tensor criteria."""
    tracker = TorchTracker.from_criteria(["loss"])
    result = tracker(loss=torch.tensor(0.42))
    assert "loss" in result
    assert isinstance(result["loss"], float)
    assert result["loss"] == pytest.approx(0.42)


def test_torch_tracker_call_with_metrics() -> None:
    """__call__ includes metric values when combined outputs list is used."""
    tracker = TorchTracker.from_criteria(["loss", "acc"])
    result = tracker(loss=torch.tensor(0.4), acc=torch.tensor(0.8))
    assert "loss" in result
    assert "acc" in result


def test_torch_tracker_is_routed_into_the_reset_events_by_the_trainer() -> None:
    """The tracker is scanned like any participant: its reset methods alone put it on both events."""
    tracker = TorchTracker.from_criteria(["loss"], distributed=False)
    trainer = TorchTrainer(device="cpu", learner=_StubLearner(), tracker=tracker)
    assert trainer.describe() == {
        "on_training_begin": ["TorchTracker"],
        "on_validation_begin": ["TorchTracker"],
    }


def test_torch_tracker_reset_clears_the_running_average_between_splits() -> None:
    """Without the reset, validation averages would carry the training values of the same epoch."""
    tracker = TorchTracker.from_criteria(["loss"], distributed=False)
    tracker(loss=torch.tensor(1.0))
    tracker.on_validation_begin(BaseInfo())
    assert tracker(loss=torch.tensor(0.0))["loss"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# TorchTrainer.sync
# ---------------------------------------------------------------------------


def test_torch_trainer_sync_cpu_is_noop() -> None:
    """sync() on a CPU trainer should not raise."""
    trainer = TorchTrainer(
        device="cpu",
        learner=_StubLearner(),
        tracker=TorchTracker.from_criteria(["loss"]),
    )
    trainer.sync()  # should not raise


# ---------------------------------------------------------------------------
# initial_model – non-Module, non-Mapping, non-list/tuple passthrough (lines 97, 107)
# ---------------------------------------------------------------------------


def test_initial_model_non_module_passthrough() -> None:
    """A plain scalar passes through _init unchanged."""
    inputs, outputs = initial_model(42)
    assert inputs is None
    assert outputs == 42


# ---------------------------------------------------------------------------
# TorchTracker.from_criteria – compile_fn branches (lines 211–213)
# ---------------------------------------------------------------------------


def test_torch_tracker_from_criteria_applies_compile_fn_to_losses() -> None:
    """compile_fn is invoked on losses_tracker (line 211) when provided."""
    compiled: list[Any] = []

    def fake_compile(m: torch.nn.Module) -> torch.nn.Module:
        compiled.append(m)
        return m

    tracker = TorchTracker.from_criteria(["loss"], compile_fn=fake_compile)
    assert len(compiled) == 1
    assert isinstance(tracker, TorchTracker)


def test_torch_tracker_from_criteria_applies_compile_fn_to_both_trackers() -> None:
    """compile_fn is applied to the tracker when provided."""
    compiled: list[Any] = []

    def fake_compile(m: torch.nn.Module) -> torch.nn.Module:
        compiled.append(m)
        return m

    tracker = TorchTracker.from_criteria(["loss", "acc"], compile_fn=fake_compile)
    assert len(compiled) == 1
    assert isinstance(tracker, TorchTracker)


# ---------------------------------------------------------------------------
# TorchTrainer.sync – CUDA path (line 287)
# ---------------------------------------------------------------------------


def test_torch_trainer_sync_cuda_calls_synchronize(monkeypatch: pytest.MonkeyPatch) -> None:
    """sync() calls torch.cuda.synchronize() when device contains 'cuda' (line 287)."""
    synced: list[bool] = []
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: synced.append(True))
    trainer = TorchTrainer(
        device="cuda",
        learner=_StubLearner(),
        tracker=TorchTracker.from_criteria(["loss"]),
    )
    trainer.sync()
    assert synced == [True]


# ---------------------------------------------------------------------------
# TimmDatasetWrapper (lines 368, 388)
# ---------------------------------------------------------------------------


def test_timm_dataset_wrapper_default_kwargs_contains_all_keys() -> None:
    """default_kwargs exposes all fields required by create_dataset (line 368)."""
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


def test_timm_dataset_wrapper_dataset_calls_create_dataset(
    tmp_path: Path,
) -> None:
    """Dataset cached_property delegates to create_dataset (line 388)."""
    _populate_image_folder(tmp_path)
    ds = TimmDatasetWrapper(name="", root=str(tmp_path))
    assert isinstance(ds.dataset, ImageDataset)
    assert len(ds.dataset) == 8  # 2 classes × 4 images


# ---------------------------------------------------------------------------
# TimmDataLoaderWrapper – utility properties (lines 533, 538)
# ---------------------------------------------------------------------------


def test_timm_dataloader_mixup_active_false_by_default() -> None:
    """mixup_active returns False when all alpha/cutmix values are at defaults (line 533)."""
    assert TimmDataLoaderWrapper().mixup_active is False


def test_timm_dataloader_mixup_active_true_with_mixup_alpha() -> None:
    """mixup_active returns True when mixup_alpha > 0 (line 533)."""
    assert TimmDataLoaderWrapper(mixup_alpha=0.2).mixup_active is True


def test_timm_dataloader_mixup_active_true_with_cutmix_alpha() -> None:
    """mixup_active returns True when cutmix_alpha > 0 (line 533)."""
    assert TimmDataLoaderWrapper(cutmix_alpha=0.2).mixup_active is True


def test_timm_dataloader_mixup_kwargs_contains_expected_keys() -> None:
    """mixup_kwargs exposes all fields expected by timm mixup constructors (line 538)."""
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
# TimmDataLoaderWrapper – distributed_results & default_kwargs (lines 552, 557–590)
# ---------------------------------------------------------------------------


def test_timm_dataloader_distributed_results() -> None:
    """distributed_results calls init_distributed_device_so (line 552)."""
    result = TimmDataLoaderWrapper().distributed_results
    assert result["device"] == "cpu"
    assert result["distributed"] is False


def test_timm_dataloader_default_kwargs_validation_branch() -> None:
    """default_kwargs includes crop_pct (not training kwargs) when is_training=False (lines 557–568, 589)."""
    kwargs = TimmDataLoaderWrapper().default_kwargs
    assert "crop_pct" in kwargs
    assert "no_aug" not in kwargs


def test_timm_dataloader_default_kwargs_training_branch() -> None:
    """default_kwargs includes training-specific keys when is_training=True (lines 568–587)."""
    kwargs = TimmDataLoaderWrapper(dataset=TimmDatasetWrapper(is_training=True)).default_kwargs
    assert "no_aug" in kwargs
    assert "re_prob" in kwargs
    assert "auto_augment" in kwargs
    assert "crop_pct" not in kwargs


# ---------------------------------------------------------------------------
# TimmDataLoaderWrapper – mixup cached_property (lines 595–597)
# ---------------------------------------------------------------------------


def test_timm_dataloader_mixup_raises_when_inactive() -> None:
    """Accessing mixup when mixup is not active raises ValueError (line 597)."""
    with pytest.raises(ValueError, match="Mixup is not active"):
        _ = TimmDataLoaderWrapper().mixup


def test_timm_dataloader_mixup_returns_fast_collate_with_prefetcher() -> None:
    """With use_prefetcher=True and mixup_alpha>0, mixup returns FastCollateMixup (lines 595–596)."""
    assert isinstance(TimmDataLoaderWrapper(mixup_alpha=0.4, use_prefetcher=True).mixup, FastCollateMixup)


def test_timm_dataloader_mixup_returns_mixup_without_prefetcher() -> None:
    """With use_prefetcher=False and mixup_alpha>0, mixup returns Mixup (lines 595–596)."""
    assert isinstance(TimmDataLoaderWrapper(mixup_alpha=0.4, use_prefetcher=False).mixup, Mixup)


# ---------------------------------------------------------------------------
# TimmDataLoaderWrapper – disable_mixup (lines 601–602)
# ---------------------------------------------------------------------------


def _training_wrapper(**kwargs: Any) -> TimmDataLoaderWrapper:
    """Return a wrapper whose dataset is a training split, without touching any real data."""
    return TimmDataLoaderWrapper(dataset=TimmDatasetWrapper(is_training=True), **kwargs)


def test_timm_dataloader_disable_mixup_disables_when_epoch_reached() -> None:
    """Mixup stops at mixup_off_epoch so the last epochs train on unmixed samples."""
    wrapper = _training_wrapper(mixup_alpha=0.5, mixup_off_epoch=3)
    wrapper.disable_mixup(BaseInfo(epoch=3))
    assert wrapper.mixup.mixup_enabled is False


def test_timm_dataloader_disable_mixup_noop_before_epoch() -> None:
    """Before the cutoff epoch mixup stays on."""
    wrapper = _training_wrapper(mixup_alpha=0.5, mixup_off_epoch=5)
    wrapper.disable_mixup(BaseInfo(epoch=2))
    assert wrapper.mixup.mixup_enabled is True


@pytest.mark.parametrize(
    "kwargs",
    [
        {"mixup_alpha": 0.5, "mixup_off_epoch": 0},
        {"mixup_alpha": 0.0, "mixup_off_epoch": 3},
    ],
    ids=["no_cutoff_epoch", "mixup_inactive"],
)
def test_timm_dataloader_disable_mixup_is_safe_without_a_mixup_cutoff(kwargs: dict[str, Any]) -> None:
    """The provider calls this unconditionally, so the guards must keep it a no-op, not an error."""
    _training_wrapper(**kwargs).disable_mixup(BaseInfo(epoch=9))


def test_timm_dataloader_disable_mixup_ignores_a_validation_split() -> None:
    """Mixup is a training-only augmentation: a validation wrapper must not build one."""
    wrapper = TimmDataLoaderWrapper(mixup_alpha=0.5, mixup_off_epoch=1)
    wrapper.disable_mixup(BaseInfo(epoch=9))
    assert wrapper.mixup.mixup_enabled is True


def test_timm_dataloader_set_epoch_forwards_zero_based_epoch_to_the_dataset() -> None:
    """Datasets reshuffle per epoch; the trainer counts from 1 and the dataset from 0."""
    seen: list[int] = []
    wrapper = _training_wrapper()
    wrapper.__dict__["dataset_wrapper"] = SimpleNamespace(set_epoch=seen.append)
    wrapper.set_epoch(BaseInfo(epoch=4))
    assert seen == [3]


def test_timm_dataloader_set_epoch_falls_back_to_the_distributed_sampler() -> None:
    """Without dataset support, the distributed sampler is what keeps ranks shuffling in step."""
    seen: list[int] = []
    wrapper = _training_wrapper()
    wrapper.__dict__["dataset_wrapper"] = object()
    wrapper.__dict__["distributed"] = True
    wrapper.__dict__["dataloader"] = SimpleNamespace(sampler=SimpleNamespace(set_epoch=seen.append))
    wrapper.set_epoch(BaseInfo(epoch=1))
    assert seen == [0]


def test_timm_dataloader_set_epoch_ignores_a_validation_split() -> None:
    """A validation split is not reshuffled, so its dataset must never be told the epoch."""
    seen: list[int] = []
    wrapper = TimmDataLoaderWrapper()
    wrapper.__dict__["dataset_wrapper"] = SimpleNamespace(set_epoch=seen.append)
    wrapper.set_epoch(BaseInfo(epoch=4))
    assert seen == []


# ---------------------------------------------------------------------------
# TimmDataLoaderWrapper – dataloader, __len__, _call, __call__
# (lines 607–616, 627, 631–646, 650–653)
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
    """Dataloader property returns the object from create_loader in validation mode (lines 607–608)."""
    loader = TimmDataLoaderWrapper(
        dataset=TimmDatasetWrapper(name="", root=str(image_folder), batch_size=2),
        **_LOADER_BASE_KWARGS,
    ).dataloader
    assert len(loader) > 0


def test_timm_dataloader_wrapper_dataloader_training_no_mixup(
    image_folder: Path,
) -> None:
    """Dataloader is obtained in training mode without mixup (line 608)."""
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
    """num_aug_splits>1 wraps the dataset in AugMixDataset (lines 614–615)."""
    wrapper = TimmDataLoaderWrapper(
        dataset=TimmDatasetWrapper(name="", root=str(image_folder), batch_size=2, is_training=True),
        num_aug_splits=2,
        **_LOADER_BASE_KWARGS,
    )
    assert isinstance(wrapper.dataset_wrapper, AugMixDataset)


def test_timm_dataloader_wrapper_len(image_folder: Path) -> None:
    """__len__ delegates to the underlying dataloader (line 627)."""
    wrapper = TimmDataLoaderWrapper(
        dataset=TimmDatasetWrapper(name="", root=str(image_folder), batch_size=2),
        **_LOADER_BASE_KWARGS,
    )
    assert len(wrapper) == len(wrapper.dataloader)


def test_timm_dataloader_call_prefetcher_no_channels_last(image_folder: Path) -> None:
    """_call with prefetcher=True channels_last=False yields from dataloader directly (line 636)."""
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
    """_call with prefetcher=True channels_last=True yields channels_last tensors (lines 633–634)."""
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
    """_call with prefetcher=False moves tensors to device/dtype (lines 638–641, 646)."""
    wrapper = TimmDataLoaderWrapper(
        dataset=TimmDatasetWrapper(name="", root=str(image_folder), batch_size=2),
        use_prefetcher=False,
        **_LOADER_BASE_KWARGS,
    )
    batches = list(wrapper._call())
    assert len(batches) > 0


def test_timm_dataloader_call_no_prefetcher_with_mixup(image_folder: Path) -> None:
    """_call with prefetcher=False and mixup_alpha>0 applies Mixup to each batch (lines 639, 642–643)."""
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
    """_call with prefetcher=False channels_last=True applies channels_last format (lines 644–645)."""
    wrapper = TimmDataLoaderWrapper(
        dataset=TimmDatasetWrapper(name="", root=str(image_folder), batch_size=2),
        use_prefetcher=False,
        channels_last=True,
        **_LOADER_BASE_KWARGS,
    )
    inp, _ = next(iter(wrapper._call()))
    assert inp.is_contiguous(memory_format=torch.channels_last)


def test_timm_dataloader_dunder_call_no_spec(image_folder: Path) -> None:
    """__call__ with spec=None yields raw (inp, target) pairs (lines 650–651)."""
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
    """__call__ with a spec applies map(spec, _call()) (lines 652–653)."""
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
# Strategy 2: real gloo process group (single-process distributed tests)
# ---------------------------------------------------------------------------

# --- initial_distributed_env ---


def test_initial_distributed_env_non_distributed_returns_cpu() -> None:
    """Non-distributed env returns cpu device, rank=0, world_size=1, distributed=False."""
    result = initial_distributed_env(device="cpu")
    assert result["device"] == "cpu"
    assert result["global_rank"] == 0
    assert result["world_size"] == 1
    assert result["distributed"] is False


def test_initial_distributed_env_return_dict_false_returns_tuple() -> None:
    """return_dict=False returns a 5-element tuple."""
    result = initial_distributed_env(device="cpu", return_dict=False)
    assert isinstance(result, tuple)
    assert len(result) == 5
    device, global_rank, local_rank, world_size, distributed = result
    assert device == "cpu"
    assert distributed is False


def test_initial_distributed_env_with_gloo_non_slurm(
    single_process_gloo: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With gloo PG initialized and env vars set, detects distributed (non-SLURM path)."""
    # WORLD_SIZE must be > 1 for timm's is_distributed_env() to return True
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.delenv("SLURM_PROCID", raising=False)
    monkeypatch.delenv("SLURM_NTASKS", raising=False)
    result = initial_distributed_env(device="cpu")
    assert result["distributed"] is True
    assert result["device"] == "cpu"
    # world_size and rank come from the actual PG (world_size=1)
    assert result["world_size"] == 1
    assert result["global_rank"] == 0
    assert result["local_rank"] == 0


def test_initial_distributed_env_with_gloo_slurm_path(
    single_process_gloo: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With gloo PG initialized and SLURM env vars, takes the SLURM branch."""
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("SLURM_PROCID", "0")
    monkeypatch.setenv("SLURM_NTASKS", "2")
    result = initial_distributed_env(device="cpu")
    assert result["distributed"] is True
    assert result["device"] == "cpu"


def test_initial_distributed_env_with_gloo_return_tuple(
    single_process_gloo: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """return_dict=False with active gloo PG returns a distributed tuple."""
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.delenv("SLURM_PROCID", raising=False)
    monkeypatch.delenv("SLURM_NTASKS", raising=False)
    device, global_rank, local_rank, world_size, distributed = initial_distributed_env(device="cpu", return_dict=False)
    assert distributed is True
    assert world_size == 1


# --- get_torch_device_type ---


def test_get_torch_device_type_cpu() -> None:
    """get_torch_device split returns 'cpu' for cpu device."""
    assert get_torch_device("cpu").split(":")[0] == "cpu"


def test_get_torch_device_type_cuda_with_rank(monkeypatch: pytest.MonkeyPatch) -> None:
    """get_torch_device split returns 'cuda' for 'cuda:0'."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert get_torch_device("cuda:0").split(":")[0] == "cuda"


# --- TorchTracker distributed all_reduce ---


def test_torch_tracker_distributed_all_reduce_identity(single_process_gloo: None) -> None:
    """all_reduce(AVG) with world_size=1 is identity; verifies distributed branch."""
    tracker = TorchTracker.from_criteria(["loss"], distributed=True)
    result = tracker(loss=torch.tensor(0.5))
    assert "loss" in result
    assert result["loss"] == pytest.approx(0.5)


def test_torch_tracker_distributed_all_reduce_multiple_criteria(single_process_gloo: None) -> None:
    """Distributed all_reduce works for multiple criteria simultaneously."""
    tracker = TorchTracker.from_criteria(["loss", "acc"], distributed=True)
    result = tracker(loss=torch.tensor(0.3), acc=torch.tensor(0.85))
    assert result["loss"] == pytest.approx(0.3)
    assert result["acc"] == pytest.approx(0.85)


def test_torch_tracker_from_criteria_auto_detects_distributed(single_process_gloo: None) -> None:
    """from_criteria with distributed=None auto-detects is_initialized() → True."""
    tracker = TorchTracker.from_criteria(["loss"], distributed=None)
    assert tracker.distributed is True


def test_torch_tracker_from_criteria_auto_detects_non_distributed() -> None:
    """from_criteria with distributed=None when no PG returns distributed=False."""
    tracker = TorchTracker.from_criteria(["loss"], distributed=None)
    assert tracker.distributed is False


# --- TorchTrainer.no_sync ---


def test_torch_trainer_no_sync_disables_grad_sync_for_ddp(single_process_gloo: None) -> None:
    """no_sync(__updated__=False) sets require_backward_grad_sync=False on DDP models."""
    model = torch.nn.Linear(2, 2)
    ddp_model = torch.nn.parallel.DistributedDataParallel(model)
    trainer = TorchTrainer(
        device="cpu",
        learner=_StubLearner(models={"m": ddp_model}),
        tracker=TorchTracker.from_criteria(["loss"], distributed=False),
    )
    assert ddp_model.require_backward_grad_sync is True
    with trainer.no_sync(False):
        assert ddp_model.require_backward_grad_sync is False
    # restored after exiting
    assert ddp_model.require_backward_grad_sync is True


def test_torch_trainer_no_sync_yields_directly_when_updated(single_process_gloo: None) -> None:
    """no_sync(__updated__=True) yields without touching DDP grad sync flag."""
    model = torch.nn.Linear(2, 2)
    ddp_model = torch.nn.parallel.DistributedDataParallel(model)
    trainer = TorchTrainer(
        device="cpu",
        learner=_StubLearner(models={"m": ddp_model}),
        tracker=TorchTracker.from_criteria(["loss"], distributed=False),
    )
    with trainer.no_sync(True):
        assert ddp_model.require_backward_grad_sync is True


def test_torch_trainer_no_sync_restores_on_exception(single_process_gloo: None) -> None:
    """no_sync restores require_backward_grad_sync even when the body raises."""
    model = torch.nn.Linear(2, 2)
    ddp_model = torch.nn.parallel.DistributedDataParallel(model)
    trainer = TorchTrainer(
        device="cpu",
        learner=_StubLearner(models={"m": ddp_model}),
        tracker=TorchTracker.from_criteria(["loss"], distributed=False),
    )
    with pytest.raises(RuntimeError, match="boom"):
        with trainer.no_sync(False):
            raise RuntimeError("boom")
    assert ddp_model.require_backward_grad_sync is True


def test_torch_trainer_no_sync_ignores_non_ddp_model() -> None:
    """no_sync leaves non-DDP models untouched when __updated__=False."""
    model = torch.nn.Linear(2, 2)
    trainer = TorchTrainer(
        device="cpu",
        learner=_StubLearner(models={"m": model}),
        tracker=TorchTracker.from_criteria(["loss"]),
    )
    with trainer.no_sync(False):
        assert not hasattr(model, "require_backward_grad_sync")


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


def test_timm_data_provider_forwards_the_epoch_only_to_the_training_wrapper() -> None:
    """Only the training split is reshuffled per epoch; forwarding to validation would be wrong."""
    training_epochs: list[int] = []
    validation_epochs: list[int] = []
    training, validation = _training_wrapper(), TimmDataLoaderWrapper()
    training.__dict__["dataset_wrapper"] = SimpleNamespace(set_epoch=training_epochs.append)
    validation.__dict__["dataset_wrapper"] = SimpleNamespace(set_epoch=validation_epochs.append)
    TimmDataProvider(training=training, validation=validation).on_epoch_begin(BaseInfo(epoch=2))
    assert training_epochs == [1]
    assert validation_epochs == []


def test_timm_data_provider_disables_mixup_of_the_training_wrapper() -> None:
    """The mixup cutoff used to be a globally registered callback; the provider now carries it."""
    training = _training_wrapper(mixup_alpha=0.5, mixup_off_epoch=2)
    TimmDataProvider(training=training).on_training_begin(BaseInfo(epoch=2))
    assert training.mixup.mixup_enabled is False


def test_timm_data_provider_is_routed_into_the_epoch_events_by_the_trainer() -> None:
    """Passing the provider to the trainer is the whole wiring: no registration call is involved."""
    trainer = TorchTrainer(
        device="cpu",
        learner=_StubLearner(),
        tracker=TorchTracker.from_criteria(["loss"], distributed=False),
        data=TimmDataProvider(training=_training_wrapper()),
    )
    described = trainer.describe()
    assert described["on_epoch_begin"] == ["TimmDataProvider"]
    assert described["on_training_begin"] == ["TorchTracker", "TimmDataProvider"]


# ---------------------------------------------------------------------------
# TorchLearnerFactory
# ---------------------------------------------------------------------------


_LEARNER_SOURCE = '''"""Tiny learner module instantiated by object patterns in the factory tests."""


class TinyLearner:
    """Records the models it was built from, like a real learner keeps them for its steps."""

    def __init__(self, **models):
        self.models = models

    def forward_training_step(self, **inputs):
        return {}


def zero_weights(module):
    """Zero every weight, so an applied initializer is visible in the parameters."""
    if hasattr(module, "weight"):
        module.weight.data.zero_()
'''


@pytest.fixture
def learner_module(tmp_path: Path) -> Path:
    """Write the learner module the patterns refer to by file path, and return that path."""
    path = tmp_path / "tiny_learner.py"
    path.write_text(_LEARNER_SOURCE)
    return path


def _linear_pattern(**kwargs: Any) -> list[Any]:
    """Return the object pattern of a `torch.nn.Linear`."""
    return ["_obj_", {"_addr_": "torch.nn.Linear"}, {"_call_": kwargs}]


def _factory(learner_module: Path, **kwargs: Any) -> TorchLearnerFactory:
    """Return a factory building one linear model and the tiny learner from the temporary module."""
    kwargs.setdefault("model_patterns", [{"encoder": _linear_pattern(in_features=4, out_features=2)}])
    return TorchLearnerFactory(
        learner_pattern=["_obj_", {"_addr_": "TinyLearner", "_file_": str(learner_module)}], **kwargs
    )


def test_torch_learner_factory_builds_the_models_and_hands_them_to_the_learner(
    learner_module: Path, allow_module_imports: None
) -> None:
    """The learner is constructed from the models by name: that binding is the factory's job."""
    models, learner = _factory(learner_module)("cpu")
    assert list(models) == ["encoder"]
    assert learner.models["encoder"] is models["encoder"]


def test_torch_learner_factory_resolves_shapes_and_initializes_lazy_models(
    learner_module: Path, allow_module_imports: None
) -> None:
    """A model with deferred parameters only gets them from a forward pass on the resolved shapes."""
    factory = _factory(
        learner_module,
        model_patterns=[{"encoder": ["_obj_", {"_addr_": "torch.nn.LazyLinear"}, {"_call_": {"out_features": 2}}]}],
        shapes={"input": [4]},
    )
    models, _ = factory("cpu")
    assert factory.input_shapes == {"input": [4]}
    assert tuple(models["encoder"].weight.shape) == (2, 4)


def test_torch_learner_factory_applies_the_initializers(learner_module: Path, allow_module_imports: None) -> None:
    """Initializers are matched to models by name, so only the named model is reinitialized."""
    factory = _factory(
        learner_module,
        initializer_patterns=[{"encoder": ["_obj_", {"_addr_": "zero_weights", "_file_": str(learner_module)}]}],
    )
    models, _ = factory("cpu")
    assert torch.count_nonzero(models["encoder"].weight) == 0


def test_torch_learner_factory_can_skip_the_initializers(learner_module: Path, allow_module_imports: None) -> None:
    """Distributed runs initialize on the main rank only, so skipping must leave the weights alone."""
    factory = _factory(
        learner_module,
        initializer_patterns=[{"encoder": ["_obj_", {"_addr_": "zero_weights", "_file_": str(learner_module)}]}],
    )
    models, _ = factory("cpu", apply_initializers=False)
    assert torch.count_nonzero(models["encoder"].weight) > 0


def test_torch_learner_factory_compiles_the_step_functions(learner_module: Path, allow_module_imports: None) -> None:
    """The step functions are the hot path, so a compile pattern has to reach them, not just the models."""
    _, learner = _factory(learner_module, compile_pattern={"dynamic": False})("cpu")
    assert hasattr(learner.forward_training_step, "_torchdynamo_orig_callable")


def test_torch_learner_factory_leaves_the_learner_uncompiled_without_a_pattern(
    learner_module: Path, allow_module_imports: None
) -> None:
    """No compile pattern means no compilation: `torch.compile` must not be applied by default."""
    _, learner = _factory(learner_module)("cpu")
    assert not hasattr(learner.forward_training_step, "_torchdynamo_orig_callable")


def test_torch_learner_factory_rejects_a_pattern_naming_several_models(
    learner_module: Path, allow_module_imports: None
) -> None:
    """A two-key entry hides which name belongs to which pattern, so it is a configuration error."""
    factory = _factory(
        learner_module, model_patterns=[{"a": _linear_pattern(in_features=1, out_features=1), "b": None}]
    )
    with pytest.raises(ValueError, match="exactly one model definition"):
        factory("cpu")


# ---------------------------------------------------------------------------
# MLflowLogger / WandbLogger
# ---------------------------------------------------------------------------


def _info_with_metrics() -> TorchTrainer:
    """Return a trainer carrying one epoch of criteria and a learner reporting learning rates."""
    trainer = TorchTrainer(
        device="cpu",
        learner=_StubLearner(learning_rates={"lr": 0.1}),
        tracker=TorchTracker.from_criteria(["loss"], distributed=False),
    )
    trainer.epoch = 1
    trainer.logs()["loss"] = 0.5
    return trainer


@pytest.fixture
def mlflow_store(tmp_path: Path) -> Any:
    """Point MLflow at a temporary store, so the tests exercise the real client, and restore it after."""
    previous = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri((tmp_path / "mlruns").as_uri())
    yield mlflow
    mlflow.set_tracking_uri(previous)


def test_mlflow_logger_owns_the_run_and_logs_the_epoch_metrics(mlflow_store: Any) -> None:
    """No event fires once per fit, so the run lifecycle lives in the context manager, not a callback."""
    with MLflowLogger("phase-two") as logger:
        run_id = mlflow_store.active_run().info.run_id
        logger.log_params({"epochs": 1})
        logger.log_metric("best_loss", 0.25, step=1)
        logger.on_epoch_end(_info_with_metrics())
    assert mlflow_store.active_run() is None
    run = mlflow_store.get_run(run_id)
    assert run.data.params == {"epochs": "1"}
    assert run.data.metrics == pytest.approx({"best_loss": 0.25, "loss": 0.5, "lr": 0.1})


def test_mlflow_logger_stores_dicts_states_and_files_as_artifacts(mlflow_store: Any, tmp_path: Path) -> None:
    """Arguments, model states and config files must survive the run for it to be reproducible."""
    artifact = tmp_path / "config.yaml"
    artifact.write_text("epochs: 1\n")
    with MLflowLogger("phase-two") as logger:
        run_id = mlflow_store.active_run().info.run_id
        logger.log_dict({"epochs": 1}, "arguments.yaml")
        logger.log_artifact(str(artifact))
        logger.log_state_dict({"weight": torch.zeros(2)}, "training_state")
    names = {item.path for item in mlflow_store.artifacts.list_artifacts(run_id=run_id)}
    assert {"arguments.yaml", "config.yaml", "training_state"} <= names


def test_wandb_logger_requires_the_optional_dependency() -> None:
    """Wandb is an optional extra: the failure must be a plain import error, not a missing attribute."""
    with pytest.raises(ImportError, match="wandb"):
        WandbLogger("phase-two")


class _FakeWandbRun:
    """Stand-in for `wandb.run`, exposing only the run directory the logger writes into."""

    def __init__(self, directory: Path) -> None:
        """Remember the directory reported as the run directory."""
        self.dir = str(directory)


class _FakeWandb:
    """Stand-in for the wandb module, recording what the logger asks it to do."""

    def __init__(self, directory: Path) -> None:
        """Create the fake module with a run directory and empty call records."""
        self.run = _FakeWandbRun(directory)
        self.projects: list[str] = []
        self.finished = 0
        self.params: dict[str, Any] = {}
        self.logged: list[tuple[dict[str, Any], int]] = []
        self.saved: list[str] = []
        self.config = SimpleNamespace(update=self.params.update)

    def init(self, project: str) -> None:
        """Record the started project."""
        self.projects.append(project)

    def finish(self) -> None:
        """Record that the run was finished."""
        self.finished += 1

    def log(self, values: dict[str, Any], step: int) -> None:
        """Record logged metrics."""
        self.logged.append((values, step))

    def save(self, path: str) -> None:
        """Record a saved file."""
        self.saved.append(path)


def test_wandb_logger_records_a_run_through_the_wandb_module(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The wandb backend must offer the same lifecycle and calls as MLflow, so phase 3 can swap them."""
    fake = _FakeWandb(tmp_path)
    monkeypatch.setitem(sys.modules, "wandb", fake)
    with WandbLogger("phase-two") as logger:
        logger.log_params({"epochs": 1})
        logger.log_dict({"epochs": 1}, "arguments.yaml")
        logger.log_artifact("config.yaml")
        logger.log_state_dict({"weight": torch.zeros(2)}, "training_state")
        logger.on_epoch_end(_info_with_metrics())
    assert fake.projects == ["phase-two"]
    assert fake.finished == 1
    assert fake.params == {"epochs": 1}
    assert fake.saved == ["config.yaml"]
    assert fake.logged == [({"lr": 0.1, "loss": 0.5}, 1)]
    assert "epochs: 1" in (tmp_path / "arguments.yaml").read_text()
    assert torch.load(tmp_path / "training_state.pt")["weight"].tolist() == [0.0, 0.0]


def test_the_two_loggers_expose_the_same_interface() -> None:
    """The CLI picks a backend by name, so any member missing on one of them breaks that choice."""
    members = {name for name in vars(MLflowLogger) if not name.startswith("_")}
    assert members == {name for name in vars(WandbLogger) if not name.startswith("_")}
