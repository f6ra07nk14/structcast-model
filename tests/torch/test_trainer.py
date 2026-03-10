"""Unit tests for structcast_model.torch.trainer – utility functions and classes."""

from __future__ import annotations

from contextlib import suppress
import functools
from typing import Any
from unittest.mock import MagicMock

import pytest
from timm.data import FastCollateMixup, Mixup
from timm.utils import ModelEmaV3
from torch.nn import Module

from structcast_model.base_trainer import GLOBAL_CALLBACKS, BaseInfo
from structcast_model.torch.trainer import (
    TimmDataLoaderWrapper,
    TimmDatasetWrapper,
    TimmEmaWrapper,
    TorchTracker,
    TorchTrainer,
    TrainingStep,
    ValidationStep,
    create_torch_inputs,
    get_autocast,
    get_torch_device,
    initial_model,
)
import torch

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_global_callbacks() -> Any:
    """Restore GLOBAL_CALLBACKS to its pre-test state after each test."""
    _attrs = (
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
    saved = {a: list(getattr(GLOBAL_CALLBACKS, a)) for a in _attrs}
    yield
    for a, v in saved.items():
        setattr(GLOBAL_CALLBACKS, a, v)


class _IdentityModel(Module):
    """A model that passes all inputs through unchanged."""

    def forward(self, **kwargs: Any) -> dict[str, Any]:
        return {}


class _LossModule(Module):
    """A loss module that always returns a fixed loss tensor."""

    def forward(self, **kwargs: Any) -> dict[str, torch.Tensor]:
        return {"loss": torch.tensor(0.5)}


class _MetricModule(Module):
    """A metric module that always returns a fixed accuracy tensor."""

    def forward(self, **kwargs: Any) -> dict[str, torch.Tensor]:
        return {"acc": torch.tensor(0.9)}


def _patch_global(monkeypatch: pytest.MonkeyPatch, func: Any, name: str, value: Any) -> None:
    """Patch a global referenced by a function.

    The trainer module is exposed via a lazy module wrapper, so patching module
    attributes by string path is unreliable for non-exported names.
    """
    monkeypatch.setitem(func.__globals__, name, value)


# ---------------------------------------------------------------------------
# create_torch_inputs
# ---------------------------------------------------------------------------


def test_create_torch_inputs_from_int_tuple_returns_tensor() -> None:
    """A tuple of ints produces a float32 tensor with batch dimension 1."""
    result = create_torch_inputs((3, 4))
    assert isinstance(result, torch.Tensor)
    assert result.shape == (1, 3, 4)
    assert result.dtype == torch.float32


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
    result_model, inputs, outputs = initial_model(model, shapes=None)
    assert result_model is model
    assert inputs is None


def test_initial_model_runs_forward_when_shapes_provided() -> None:
    """With shapes provided, forward() is called and inputs are returned."""

    class SimpleModel(Module):
        def forward(self, x: torch.Tensor) -> dict[str, Any]:
            return {}

    model = SimpleModel()
    result_model, inputs, outputs = initial_model(model, shapes={"x": (3,)})
    assert result_model is model
    assert inputs is not None
    assert "x" in inputs


def test_initial_model_applies_compile_fn() -> None:
    """compile_fn is applied to each Module in the structure."""
    model = _IdentityModel()
    compiled = []

    def fake_compile(m: Module) -> Module:
        compiled.append(m)
        return m

    initial_model(model, shapes=None, compile_fn=fake_compile)
    assert model in compiled


def test_initial_model_handles_dict_of_modules() -> None:
    """A dict of modules is handled correctly."""
    models = {"a": _IdentityModel(), "b": _IdentityModel()}
    result, inputs, outputs = initial_model(models, shapes=None)
    assert isinstance(result, dict)
    assert set(result.keys()) == {"a", "b"}


def test_initial_model_handles_list_of_modules() -> None:
    """A list of modules is handled correctly."""
    models = [_IdentityModel(), _IdentityModel()]
    result, inputs, outputs = initial_model(models, shapes=None)
    assert isinstance(result, list)
    assert len(result) == 2


# ---------------------------------------------------------------------------
# get_autocast
# ---------------------------------------------------------------------------


def test_get_autocast_returns_suppress_when_no_precision() -> None:
    """Returns contextlib.suppress when mixed_precision_type is None."""
    fn = get_autocast(None, "cpu")
    assert fn is suppress


def test_get_autocast_returns_partial_autocast_when_precision_set() -> None:
    """Returns a partial wrapping torch.autocast when mixed precision is set."""
    fn = get_autocast("bfloat16", "cpu")
    assert isinstance(fn, functools.partial)


# ---------------------------------------------------------------------------
# TrainingStep
# ---------------------------------------------------------------------------


def test_training_step_returns_loss_criteria() -> None:
    """TrainingStep runs models and loss module, returning criteria dict."""
    model = _IdentityModel()
    loss = _LossModule()
    step = TrainingStep(models=["m"], losses=loss)
    criteria = step({"x": torch.zeros(1)}, m=model)
    assert "loss" in criteria
    assert isinstance(criteria["loss"], torch.Tensor)


def test_training_step_includes_metrics_when_provided() -> None:
    """TrainingStep also computes metrics when metrics module is given."""
    model = _IdentityModel()
    loss = _LossModule()
    metric = _MetricModule()
    step = TrainingStep(models=["m"], losses=loss, metrics=metric)
    criteria = step({"x": torch.zeros(1)}, m=model)
    assert "loss" in criteria
    assert "acc" in criteria


def test_training_step_no_model_keys() -> None:
    """TrainingStep with empty models list still runs the loss."""
    loss = _LossModule()
    step = TrainingStep(models=[], losses=loss)
    criteria = step({"x": torch.zeros(1)})
    assert "loss" in criteria


# ---------------------------------------------------------------------------
# ValidationStep
# ---------------------------------------------------------------------------


def test_validation_step_returns_loss_criteria() -> None:
    """ValidationStep runs under no_grad and returns criteria."""
    model = _IdentityModel()
    loss = _LossModule()
    step = ValidationStep(models=["m"], losses=loss)
    criteria = step({"x": torch.zeros(1)}, m=model)
    assert "loss" in criteria


def test_validation_step_includes_metrics_when_provided() -> None:
    """ValidationStep also computes metrics when provided."""
    model = _IdentityModel()
    loss = _LossModule()
    metric = _MetricModule()
    step = ValidationStep(models=["m"], losses=loss, metrics=metric)
    criteria = step({"x": torch.zeros(1)}, m=model)
    assert "loss" in criteria
    assert "acc" in criteria


# ---------------------------------------------------------------------------
# TorchTracker
# ---------------------------------------------------------------------------


def test_torch_tracker_from_criteria_creates_tracker() -> None:
    """TorchTracker.from_criteria returns a valid TorchTracker."""
    tracker = TorchTracker.from_criteria(["loss"])
    assert isinstance(tracker, TorchTracker)
    assert tracker.metrics_tracker is None


def test_torch_tracker_from_criteria_with_metric_outputs() -> None:
    """TorchTracker.from_criteria creates metric tracker when metric_outputs given."""
    tracker = TorchTracker.from_criteria(["loss"], metric_outputs=["acc"])
    assert tracker.metrics_tracker is not None


def test_torch_tracker_call_returns_float_values() -> None:
    """__call__ returns a dict of float values from Tensor criteria."""
    tracker = TorchTracker.from_criteria(["loss"])
    result = tracker(loss=torch.tensor(0.42))
    assert "loss" in result
    assert isinstance(result["loss"], float)
    assert result["loss"] == pytest.approx(0.42)


def test_torch_tracker_call_with_metrics() -> None:
    """__call__ includes metric values when metrics_tracker is present."""
    tracker = TorchTracker.from_criteria(["loss"], metric_outputs=["acc"])
    result = tracker(loss=torch.tensor(0.4), acc=torch.tensor(0.8))
    assert "loss" in result
    assert "acc" in result


def test_torch_tracker_post_init_registers_reset_callback() -> None:
    """Creating a TorchTracker registers a reset callback in GLOBAL_CALLBACKS."""
    before = len(GLOBAL_CALLBACKS.on_training_begin)
    TorchTracker.from_criteria(["loss"])
    assert len(GLOBAL_CALLBACKS.on_training_begin) > before


# ---------------------------------------------------------------------------
# TorchTrainer.sync
# ---------------------------------------------------------------------------


def test_torch_trainer_sync_cpu_is_noop() -> None:
    """sync() on a CPU trainer should not raise."""
    loss = _LossModule()
    trainer = TorchTrainer(
        device="cpu",
        training_step=TrainingStep(models=[], losses=loss),
        backward=lambda step, **kw: True,
        tracker=TorchTracker.from_criteria(["loss"]),
        add_global_callbacks=False,
    )
    trainer.sync()  # should not raise


# ---------------------------------------------------------------------------
# initial_model – non-Module, non-Mapping, non-list/tuple passthrough (lines 97, 107)
# ---------------------------------------------------------------------------


def test_initial_model_non_module_passthrough() -> None:
    """A plain scalar passes through _init (line 97) and _construct_outputs (line 107) unchanged."""
    result, inputs, outputs = initial_model(42)
    assert result == 42
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
    """compile_fn is applied to both trackers (lines 211–213) when metric_outputs given."""
    compiled: list[Any] = []

    def fake_compile(m: torch.nn.Module) -> torch.nn.Module:
        compiled.append(m)
        return m

    tracker = TorchTracker.from_criteria(["loss"], metric_outputs=["acc"], compile_fn=fake_compile)
    assert len(compiled) == 2
    assert tracker.metrics_tracker is not None


# ---------------------------------------------------------------------------
# TorchTrainer.sync – CUDA path (line 287)
# ---------------------------------------------------------------------------


def test_torch_trainer_sync_cuda_calls_synchronize(monkeypatch: pytest.MonkeyPatch) -> None:
    """sync() calls torch.cuda.synchronize() when device contains 'cuda' (line 287)."""
    synced: list[bool] = []
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: synced.append(True))
    loss = _LossModule()
    trainer = TorchTrainer(
        device="cuda",
        training_step=TrainingStep(models=[], losses=loss),
        backward=lambda step, **kw: True,
        tracker=TorchTracker.from_criteria(["loss"]),
        add_global_callbacks=False,
    )
    trainer.sync()
    assert synced == [True]


# ---------------------------------------------------------------------------
# TimmEmaWrapper (lines 230, 234–235, 239, 244, 267–274)
# ---------------------------------------------------------------------------


class _ParamModel(Module):
    """A tiny model with trainable parameters for EMA tests."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = torch.nn.Linear(2, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


def test_timm_ema_wrapper_from_models_registers_update_callback() -> None:
    """from_models triggers __post_init__ which appends self.update to GLOBAL_CALLBACKS.on_update (line 230)."""
    before = len(GLOBAL_CALLBACKS.on_update)
    TimmEmaWrapper.from_models({"m": _ParamModel()})
    assert len(GLOBAL_CALLBACKS.on_update) == before + 1


def test_timm_ema_wrapper_from_models_creates_ema_entry() -> None:
    """from_models produces a ModelEmaV3 for every provided model (lines 267–274)."""
    wrapper = TimmEmaWrapper.from_models({"m": _ParamModel()})
    assert "m" in wrapper.ema
    assert isinstance(wrapper.ema["m"], ModelEmaV3)


def test_timm_ema_wrapper_from_models_with_compile_fn() -> None:
    """from_models applies compile_fn to each EMA model (lines 270–271)."""
    compiled: list[Any] = []

    def fake_compile(m: Module) -> Module:
        compiled.append(m)
        return m

    TimmEmaWrapper.from_models({"m": _ParamModel()}, compile_fn=fake_compile)
    assert len(compiled) == 1


def test_timm_ema_wrapper_update_advances_ema() -> None:
    """update() calls ema.update for each model (lines 234–235)."""
    model = _ParamModel()
    wrapper = TimmEmaWrapper.from_models({"m": model})
    info = BaseInfo(update=3)
    wrapper.update(info, m=model)  # must not raise


def test_timm_ema_wrapper_call_returns_original_when_not_cross_device() -> None:
    """__call__ returns original model when is_cross_device is False (line 239)."""
    model = _ParamModel()
    wrapper = TimmEmaWrapper.from_models({"m": model})  # device=None → is_cross_device=False
    info = BaseInfo()
    result = wrapper(info, m=model)
    assert result["m"] is model


def test_timm_ema_wrapper_call_returns_ema_when_cross_device() -> None:
    """__call__ returns the EMA wrapper when is_cross_device is True (line 239)."""
    model = _ParamModel()
    ema_model = ModelEmaV3(model)
    wrapper = TimmEmaWrapper(ema={"m": ema_model}, is_cross_device={"m": True})
    info = BaseInfo()
    result = wrapper(info, m=model)
    assert result["m"] is ema_model


def test_timm_ema_wrapper_models_property_returns_ema_modules() -> None:
    """Models property returns the underlying nn.Module for each EMA entry (line 244)."""
    model = _ParamModel()
    wrapper = TimmEmaWrapper.from_models({"m": model})
    ema_models = wrapper.models
    assert "m" in ema_models
    assert isinstance(ema_models["m"], Module)


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
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Dataset cached_property delegates to create_dataset (line 388)."""
    sentinel = object()
    calls: list[dict[str, Any]] = []

    def fake_create_dataset(**kw: Any) -> object:
        calls.append(kw)
        return sentinel

    _patch_global(monkeypatch, TimmDatasetWrapper.dataset.func, "create_dataset", fake_create_dataset)
    ds = TimmDatasetWrapper()
    assert ds.dataset is sentinel
    assert len(calls) == 1
    assert calls[0]["name"] == "imagenet"


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


def test_timm_dataloader_distributed_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """distributed_results calls init_distributed_device_so (line 552)."""
    _patch_global(
        monkeypatch,
        TimmDataLoaderWrapper.distributed_results.func,
        "init_distributed_device_so",
        lambda device: {"device": "cpu", "distributed": False},
    )
    result = TimmDataLoaderWrapper().distributed_results
    assert result["device"] == "cpu"
    assert result["distributed"] is False


def test_timm_dataloader_default_kwargs_validation_branch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """default_kwargs includes crop_pct (not training kwargs) when is_training=False (lines 557–568, 589)."""
    _patch_global(
        monkeypatch,
        TimmDataLoaderWrapper.distributed_results.func,
        "init_distributed_device_so",
        lambda device: {"device": "cpu", "distributed": False},
    )
    kwargs = TimmDataLoaderWrapper().default_kwargs
    assert "crop_pct" in kwargs
    assert "no_aug" not in kwargs


def test_timm_dataloader_default_kwargs_training_branch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """default_kwargs includes training-specific keys when is_training=True (lines 568–587)."""
    _patch_global(
        monkeypatch,
        TimmDataLoaderWrapper.distributed_results.func,
        "init_distributed_device_so",
        lambda device: {"device": "cpu", "distributed": False},
    )
    wrapper = TimmDataLoaderWrapper(dataset=TimmDatasetWrapper(is_training=True))
    kwargs = wrapper.default_kwargs
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
    assert isinstance(
        TimmDataLoaderWrapper(mixup_alpha=0.4, use_prefetcher=True).mixup,
        FastCollateMixup,
    )


def test_timm_dataloader_mixup_returns_mixup_without_prefetcher() -> None:
    """With use_prefetcher=False and mixup_alpha>0, mixup returns Mixup (lines 595–596)."""
    assert isinstance(
        TimmDataLoaderWrapper(mixup_alpha=0.4, use_prefetcher=False).mixup,
        Mixup,
    )


# ---------------------------------------------------------------------------
# TimmDataLoaderWrapper – disable_mixup (lines 601–602)
# ---------------------------------------------------------------------------


def test_timm_dataloader_disable_mixup_disables_when_epoch_reached() -> None:
    """disable_mixup sets mixup.mixup_enabled=False when epoch >= mixup_off_epoch (lines 601–602)."""
    wrapper = TimmDataLoaderWrapper(mixup_alpha=0.5, mixup_off_epoch=3)
    _ = wrapper.mixup  # initialise cached_property
    wrapper.disable_mixup(BaseInfo(epoch=3))
    assert wrapper.mixup.mixup_enabled is False


def test_timm_dataloader_disable_mixup_noop_before_epoch() -> None:
    """disable_mixup is a no-op when epoch < mixup_off_epoch (line 601 – False branch)."""
    wrapper = TimmDataLoaderWrapper(mixup_alpha=0.5, mixup_off_epoch=5)
    _ = wrapper.mixup
    wrapper.disable_mixup(BaseInfo(epoch=2))
    assert wrapper.mixup.mixup_enabled is True


# ---------------------------------------------------------------------------
# TimmDataLoaderWrapper – dataloader, __len__, _call, __call__
# (lines 607–616, 627, 631–646, 650–653)
# ---------------------------------------------------------------------------


class _FakeLoader:
    """Minimal DataLoader stand-in that yields one fixed batch."""

    _data = [(torch.zeros(2, 3, 4, 4), torch.zeros(2, dtype=torch.long))]

    def __len__(self) -> int:
        return len(self._data)

    def __iter__(self) -> Any:
        return iter(self._data)


@pytest.fixture
def patch_timm_io(monkeypatch: pytest.MonkeyPatch) -> _FakeLoader:
    """Monkeypatch timm I/O so no real dataset or DataLoader is created."""
    loader = _FakeLoader()
    _patch_global(
        monkeypatch,
        TimmDataLoaderWrapper.distributed_results.func,
        "init_distributed_device_so",
        lambda device: {"device": "cpu", "distributed": False},
    )
    _patch_global(
        monkeypatch,
        TimmDatasetWrapper.dataset.func,
        "create_dataset",
        lambda **kw: MagicMock(),
    )
    _patch_global(
        monkeypatch,
        TimmDataLoaderWrapper.dataloader.func,
        "create_loader",
        lambda **kw: loader,
    )
    return loader


def test_timm_dataloader_wrapper_dataloader_validation(
    patch_timm_io: _FakeLoader,
) -> None:
    """Dataloader property returns the object from create_loader in validation mode (lines 607–608)."""
    wrapper = TimmDataLoaderWrapper()
    assert wrapper.dataloader is patch_timm_io


def test_timm_dataloader_wrapper_dataloader_training_no_mixup(
    patch_timm_io: _FakeLoader,
) -> None:
    """Dataloader is obtained in training mode without mixup (line 608)."""
    wrapper = TimmDataLoaderWrapper(dataset=TimmDatasetWrapper(is_training=True))
    assert wrapper.dataloader is patch_timm_io


def test_timm_dataloader_wrapper_dataloader_training_with_mixup_off_epoch(
    patch_timm_io: _FakeLoader,
) -> None:
    """Dataloader with mixup and mixup_off_epoch>0 registers disable_mixup callback (lines 609–613)."""
    before = len(GLOBAL_CALLBACKS.on_training_begin)
    wrapper = TimmDataLoaderWrapper(
        dataset=TimmDatasetWrapper(is_training=True),
        mixup_alpha=0.5,
        mixup_off_epoch=3,
        use_prefetcher=True,
    )
    _ = wrapper.dataloader
    assert len(GLOBAL_CALLBACKS.on_training_begin) > before


def test_timm_dataloader_wrapper_dataloader_with_aug_splits(
    patch_timm_io: _FakeLoader,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """num_aug_splits>1 wraps the dataset in AugMixDataset (lines 614–615)."""
    aug_ds_created: list[tuple[Any, int]] = []

    class _FakeAugMix:
        def __init__(self, dataset: Any, num_splits: int) -> None:
            aug_ds_created.append((dataset, num_splits))

    _patch_global(monkeypatch, TimmDataLoaderWrapper.dataloader.func, "AugMixDataset", _FakeAugMix)
    wrapper = TimmDataLoaderWrapper(
        dataset=TimmDatasetWrapper(is_training=True),
        num_aug_splits=2,
    )
    _ = wrapper.dataloader
    assert len(aug_ds_created) == 1
    assert aug_ds_created[0][1] == 2


def test_timm_dataloader_wrapper_len(patch_timm_io: _FakeLoader) -> None:
    """__len__ delegates to the underlying dataloader (line 627)."""
    assert len(TimmDataLoaderWrapper()) == len(patch_timm_io)


def test_timm_dataloader_call_prefetcher_no_channels_last(
    patch_timm_io: _FakeLoader,
) -> None:
    """_call with prefetcher=True channels_last=False yields from dataloader directly (line 636)."""
    wrapper = TimmDataLoaderWrapper(use_prefetcher=True, channels_last=False)
    batches = list(wrapper._call())
    assert len(batches) == 1
    inp, _ = batches[0]
    assert inp.shape == (2, 3, 4, 4)


def test_timm_dataloader_call_prefetcher_channels_last(
    patch_timm_io: _FakeLoader,
) -> None:
    """_call with prefetcher=True channels_last=True yields channels_last tensors (lines 633–634)."""
    wrapper = TimmDataLoaderWrapper(use_prefetcher=True, channels_last=True)
    batches = list(wrapper._call())
    assert len(batches) == 1
    inp, _ = batches[0]
    assert inp.is_contiguous(memory_format=torch.channels_last)


def test_timm_dataloader_call_no_prefetcher(
    patch_timm_io: _FakeLoader,
) -> None:
    """_call with prefetcher=False moves tensors to device/dtype (lines 638–641, 646)."""
    wrapper = TimmDataLoaderWrapper(use_prefetcher=False)
    batches = list(wrapper._call())
    assert len(batches) == 1


def test_timm_dataloader_call_no_prefetcher_with_mixup(
    patch_timm_io: _FakeLoader,
) -> None:
    """_call with prefetcher=False and mixup_alpha>0 applies Mixup to each batch (lines 639, 642–643)."""
    wrapper = TimmDataLoaderWrapper(use_prefetcher=False, mixup_alpha=0.4)
    batches = list(wrapper._call())
    assert len(batches) == 1


def test_timm_dataloader_call_no_prefetcher_channels_last(
    patch_timm_io: _FakeLoader,
) -> None:
    """_call with prefetcher=False channels_last=True applies channels_last format (lines 644–645)."""
    wrapper = TimmDataLoaderWrapper(use_prefetcher=False, channels_last=True)
    inp, _ = next(iter(wrapper._call()))
    assert inp.is_contiguous(memory_format=torch.channels_last)


def test_timm_dataloader_dunder_call_no_spec(
    patch_timm_io: _FakeLoader,
) -> None:
    """__call__ with spec=None yields raw (inp, target) pairs (lines 650–651)."""
    wrapper = TimmDataLoaderWrapper(spec=None)
    batches = list(wrapper())
    assert len(batches) == 1
    inp, target = batches[0]
    assert isinstance(inp, torch.Tensor)


def test_timm_dataloader_dunder_call_with_spec(
    patch_timm_io: _FakeLoader,
) -> None:
    """__call__ with a spec applies map(spec, _call()) (lines 652–653)."""
    wrapper = TimmDataLoaderWrapper(spec=None)
    results: list[Any] = []

    def fake_spec(x: Any) -> Any:
        results.append(x)
        return x

    # bypass Pydantic validation to set a plain callable
    wrapper.__dict__["spec"] = fake_spec
    list(wrapper())
    assert len(results) == 1
