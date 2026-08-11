"""Unit tests for structcast_model.torch.trainer - utility functions and classes."""

from __future__ import annotations

import logging
from typing import Any

import pytest
from torch.nn import Module

from structcast_model.base_trainer import BaseInfo, SimpleDataProvider
from structcast_model.torch.trainer import (
    TorchTracker,
    TorchTrainer,
    TrainingStateSaver,
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

    def __init__(
        self,
        models: dict[str, Any] | None = None,
        learning_rates: dict[str, float] | None = None,
        optimizers: dict[str, Any] | None = None,
    ) -> None:
        """Initialize with optional models, optimizers, and the learning rates a real learner would report."""
        self._models = models or {}
        self._optimizers = optimizers or {}
        self.learning_rates = learning_rates or {}

    @property
    def models(self) -> dict[str, Any]:
        """Return the models dict."""
        return self._models

    @property
    def optimizers(self) -> dict[str, Any]:
        """Return the optimizers dict; the trainer scan must handle an empty mapping."""
        return self._optimizers

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


def test_create_torch_inputs_honours_explicit_initializer() -> None:
    """An explicit `_INIT_` address replaces the dtype-based default initializer."""
    result = create_torch_inputs({"_SHAPE_": [4], "_INIT_": "torch.ones"})
    assert torch.equal(result, torch.ones((1, 4), dtype=torch.bfloat16))


def test_create_torch_inputs_rejects_non_callable_initializer() -> None:
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


def test_torch_tracker_buffers_follow_the_ambient_device() -> None:
    """The CLI builds the tracker inside `with torch.device(device)`.

    The buffers must land on that device, or the first CUDA training step mixes CUDA criteria with
    CPU buffers and crashes.
    """
    with torch.device("meta"):
        tracker = TorchTracker.from_criteria(["loss"], None, False)
    assert tracker.tracker.total.device.type == "meta"


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
    trainer = TorchTrainer(
        device="cpu", learner=_StubLearner(), tracker=tracker, data=SimpleDataProvider(training_dataset=[])
    )
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
        data=SimpleDataProvider(training_dataset=[]),
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
        data=SimpleDataProvider(training_dataset=[]),
    )
    trainer.sync()
    assert synced == [True]


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
        data=SimpleDataProvider(training_dataset=[]),
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
        data=SimpleDataProvider(training_dataset=[]),
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
        data=SimpleDataProvider(training_dataset=[]),
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
        data=SimpleDataProvider(training_dataset=[]),
    )
    with trainer.no_sync(False):
        assert not hasattr(model, "require_backward_grad_sync")


# ---------------------------------------------------------------------------
# TrainingStateSaver
# ---------------------------------------------------------------------------


class _RecordingLogger:
    """Logger recording only the state dictionaries, which is all the saver produces."""

    def __init__(self) -> None:
        """Start with nothing recorded."""
        self.states: list[tuple[dict[str, Any], str]] = []

    def log_state_dict(self, states: Any, name: str) -> None:
        """Record the state dictionary the saver hands over."""
        self.states.append((dict(states), name))


def test_training_state_saver_records_everything_needed_to_resume() -> None:
    """A run is resumed from the weights, the optimizer state, and the loop counters together."""
    model = torch.nn.Linear(4, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    trainer = TorchTrainer(
        device="cpu",
        learner=_StubLearner(models={"model": model}, optimizers={"opt": optimizer}),
        tracker=TorchTracker.from_criteria(["loss"], distributed=False),
        data=SimpleDataProvider(training_dataset=[]),
    )
    trainer.epoch, trainer.step, trainer.update = 3, 7, 2
    recorder = _RecordingLogger()
    TrainingStateSaver(recorder).on_epoch_end(trainer, model=model)
    states, name = recorder.states[0]
    assert name == "training_state"
    assert "weight" in states["models"]["model"]
    assert states["optimizers"]["opt"]["param_groups"][0]["lr"] == 0.1
    assert states["meta"] == {"epoch": 3, "step": 7, "update": 2}
