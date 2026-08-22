"""Unit tests for structcast_model.torch.trainer - utility functions and classes."""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
from typing import Any

import pytest
from torch.nn import Module

from structcast_model.base_trainer import BaseInfo, SimpleDataProvider
from structcast_model.loggers.base import NullLogger
from structcast_model.torch.distributed import SingleDeviceStrategy
from structcast_model.torch.trainer import (
    TorchBestCriterion,
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


@dataclass(kw_only=True)
class _InfoWithModels(BaseInfo[torch.nn.Module]):
    """Info carrying models without a trainer, for callbacks tested outside a training loop.

    ``BaseInfo.step`` is a read-only view of the learner's counter, so this learner-less info
    overrides it with the settable ``current_step``.
    """

    named_models: dict[str, torch.nn.Module] = field(default_factory=dict)
    """The models the property hands out."""

    current_step: int = 0
    """The step count the ``step`` property reports."""

    @property
    def models(self) -> dict[str, torch.nn.Module]:
        """Return the models this info was built with."""
        return self.named_models

    @property
    def step(self) -> int:
        """Report the driven step, standing in for a trainer's learner-backed count."""
        return self.current_step


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
        self.steps = 0
        self.updates = 0
        self.has_updated = False

    @property
    def models(self) -> dict[str, Any]:
        """Return the models dict."""
        return self._models

    @property
    def optimizers(self) -> dict[str, Any]:
        """Return the optimizers dict; the trainer scan must handle an empty mapping."""
        return self._optimizers

    @property
    def optimizer_models(self) -> dict[str, list[str]]:
        """Pair every optimizer with every model, which is the pairing a single-model learner reports."""
        return {name: list(self._models) for name in self._optimizers}

    def restore_counters(self, steps: int, updates: int) -> None:
        """Seed the counters, the way a resume path would."""
        self.steps = steps
        self.updates = updates

    def training_step(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """Count one step that always lands an update, returning no criteria."""
        self.steps += 1
        self.updates += 1
        self.has_updated = True
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


# ---------------------------------------------------------------------------
# TrainingStateSaver
# ---------------------------------------------------------------------------


class _RecordingLogger(NullLogger):
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
    learner = _StubLearner(models={"model": model}, optimizers={"opt": optimizer})
    trainer = TorchTrainer(
        device="cpu",
        learner=learner,
        tracker=TorchTracker.from_criteria(["loss"], distributed=False),
        data=SimpleDataProvider(training_dataset=[]),
    )
    trainer.epoch = 3
    learner.restore_counters(steps=7, updates=2)
    recorder = _RecordingLogger()
    TrainingStateSaver(logger=recorder, strategy=SingleDeviceStrategy(device="cpu")).on_epoch_end(trainer)
    states, name = recorder.states[0]
    assert name == "training_state"
    assert "weight" in states["models"]["model"]
    assert states["optimizers"]["opt"]["param_groups"][0]["lr"] == 0.1
    assert states["meta"] == {"epoch": 3, "step": 7, "update": 2}


class _RecordingStrategy(SingleDeviceStrategy):
    """A strategy recording that its collective state production ran, returning a recognisable state."""

    def __init__(self) -> None:
        """Start with nothing recorded."""
        super().__init__(device="cpu")
        self.calls: list[dict[str, Any]] = []
        self.pairings: list[Any] = []

    def state_dict(self, models: Any, optimizers: Any = None, optimizer_models: Any = None) -> dict[str, Any]:
        """Record the models and pairing handed over, returning a state no plain `state_dict()` could produce."""
        self.calls.append(dict(models))
        self.pairings.append(optimizer_models)
        return {"models": {"gathered": True}, "optimizers": {}}


def test_training_state_saver_produces_state_on_null_logger_ranks() -> None:
    """Producing the state is a collective, so a rank that writes nothing must still take part in it.

    Skipping the producer on the null-logger ranks hangs the job under FSDP2.
    """
    model = torch.nn.Linear(4, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    learner = _StubLearner(models={"model": model}, optimizers={"opt": optimizer})
    trainer = TorchTrainer(
        device="cpu",
        learner=learner,
        tracker=TorchTracker.from_criteria(["loss"], distributed=False),
        data=SimpleDataProvider(training_dataset=[]),
    )
    strategy = _RecordingStrategy()
    TrainingStateSaver(logger=NullLogger(), strategy=strategy).on_epoch_end(trainer)
    assert strategy.calls == [{"model": model}]
    # The learner's own pairing must reach the strategy: without it the strategy cannot key sharded
    # optimizer state by parameter name and falls back to unloadable plain state dicts.
    assert strategy.pairings == [learner.optimizer_models] == [{"opt": ["model"]}]


# ---------------------------------------------------------------------------
# TorchBestCriterion.from_criteria
# ---------------------------------------------------------------------------


class _BestRecordingLogger(NullLogger):
    """Logger recording the metrics and state-dict names the best monitors produce."""

    def __init__(self) -> None:
        """Start with nothing recorded."""
        self.metrics: list[tuple[str, float, int]] = []
        self.states: list[str] = []
        self.payloads: list[Any] = []

    def log_metric(self, name: str, value: float, step: int) -> None:
        """Record a logged best value."""
        self.metrics.append((name, value, step))

    def log_state_dict(self, states: Any, name: str) -> None:
        """Record the name a state dictionary was saved under, and the state itself."""
        self.states.append(name)
        self.payloads.append(states)


def test_from_criteria_builds_one_wired_monitor_per_criterion() -> None:
    """The CLI hands its criteria lists straight to this factory, so the modes must map correctly."""
    monitors = TorchBestCriterion.from_criteria(
        ["acc"], ["loss"], ["acc"], _BestRecordingLogger(), SingleDeviceStrategy(device="cpu")
    )
    assert [(monitor.target, monitor.mode) for monitor in monitors] == [("acc", "max"), ("loss", "min")]
    assert all(len(monitor.callbacks) == 1 for monitor in monitors)


def test_from_criteria_monitor_logs_the_best_value_each_epoch() -> None:
    """The best value must land in the run as a metric: that is what makes a run comparable afterwards."""
    logger = _BestRecordingLogger()
    (monitor,) = TorchBestCriterion.from_criteria([], ["val_loss"], [], logger, SingleDeviceStrategy(device="cpu"))
    info = _InfoWithModels(named_models={"model": torch.nn.Linear(4, 2)})
    info.epoch, info.current_step = 1, 5
    info.history[1] = {"val_loss": 0.3}
    monitor.on_epoch_end(info)
    assert logger.metrics == [("best_val_loss", 0.3, 1)]
    assert logger.states == []


@pytest.mark.parametrize(("save_criteria", "expected"), [(["val_loss"], ["best_val_loss"]), ([], [])])
def test_from_criteria_saves_the_state_only_for_save_criteria(save_criteria: list[str], expected: list[str]) -> None:
    """Weights are only written for criteria the user asked to save, since every save costs disk space."""
    logger = _BestRecordingLogger()
    (monitor,) = TorchBestCriterion.from_criteria(
        [], ["val_loss"], save_criteria, logger, SingleDeviceStrategy(device="cpu")
    )
    info = _InfoWithModels(named_models={"model": torch.nn.Linear(4, 2)})
    info.epoch, info.current_step = 1, 5
    info.history[1] = {"val_loss": 0.3}
    monitor.on_epoch_end(info)
    assert logger.states == expected


def test_from_criteria_does_not_save_a_stale_best() -> None:
    """A best reached at an earlier step must not overwrite the saved weights of the current models."""
    logger = _BestRecordingLogger()
    (monitor,) = TorchBestCriterion.from_criteria(
        [], ["val_loss"], ["val_loss"], logger, SingleDeviceStrategy(device="cpu")
    )
    info = _InfoWithModels(named_models={"model": torch.nn.Linear(4, 2)})
    info.epoch, info.current_step = 1, 5
    info.history[1] = {"val_loss": 0.3}
    monitor.on_epoch_end(info)
    info.epoch, info.current_step = 2, 9
    info.history[2] = {"val_loss": 0.9}
    monitor.on_epoch_end(info)
    assert logger.states == ["best_val_loss"]


def test_from_criteria_saves_the_states_the_strategy_produced() -> None:
    """The saved weights must come from the strategy, which is what makes them wrapper-free and gathered.

    Calling `state_dict()` on the models directly would write shards under FSDP2 and `module.*` keys
    under DDP, neither of which any loader accepts.
    """
    logger = _BestRecordingLogger()
    strategy = _RecordingStrategy()
    model = torch.nn.Linear(4, 2)
    (monitor,) = TorchBestCriterion.from_criteria([], ["val_loss"], ["val_loss"], logger, strategy)
    info = _InfoWithModels(named_models={"model": model})
    info.epoch, info.current_step = 1, 5
    info.history[1] = {"val_loss": 0.3}
    monitor.on_epoch_end(info)
    assert strategy.calls == [{"model": model}]
    assert logger.payloads == [{"gathered": True}]
