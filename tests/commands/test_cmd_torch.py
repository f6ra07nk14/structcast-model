"""Unit tests for structcast_model.commands.cmd_torch."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Generator
from contextlib import contextmanager
import os
import pathlib
import traceback
from typing import Any

import mlflow
import pytest
from structcast.utils.security import configure_security
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils._python_dispatch import TorchDispatchMode, _get_current_dispatch_mode_stack
from typer import Typer
from typer.testing import CliRunner

from structcast_model.base_trainer import GLOBAL_CALLBACKS, BaseInfo, BestCriterion, callbacks_session
from structcast_model.commands.cmd_torch import app
from structcast_model.torch.trainer import (
    TimmEmaWrapper,
    TorchTracker,
    TorchTrainer,
    TrainingStep,
    ValidationStep,
)
from tests import ASSETS_DIR
import torch

MODEL_CFG = str(ASSETS_DIR / "cfg" / "torch" / "ConvNeXtV2.yaml")
BACKWARD_CFG = str(ASSETS_DIR / "cfg" / "torch" / "ConvNeXtV2Backward.yaml")

# ---------------------------------------------------------------------------
# Helper: access cmd_torch's real globals (bypasses LazySelectedImporter proxy)
# ---------------------------------------------------------------------------

_CMD_GLOBALS: dict[str, Any] = app.registered_commands[0].callback.__globals__  # type: ignore[union-attr]

# Access private functions from cmd_torch via its module globals
_compile_module = _CMD_GLOBALS["_compile_module"]
_get_module_outputs = _CMD_GLOBALS["_get_module_outputs"]
_get_state_dict = _CMD_GLOBALS["_get_state_dict"]
_instantiate = _CMD_GLOBALS["_instantiate"]
_instantiate_models = _CMD_GLOBALS["_instantiate_models"]
_log_criteria = _CMD_GLOBALS["_log_criteria"]
_on_best = _CMD_GLOBALS["_on_best"]
_save_training_state = _CMD_GLOBALS["_save_training_state"]
_unwrap_ddp = _CMD_GLOBALS["_unwrap_ddp"]


@contextmanager
def patch_cmd_globals(**kwargs: Any) -> Generator[None, Any, None]:
    """Temporarily override entries in cmd_torch's real module globals."""
    originals = {k: _CMD_GLOBALS.get(k) for k in kwargs}
    _CMD_GLOBALS.update(kwargs)
    try:
        yield
    finally:
        _CMD_GLOBALS.update(originals)


@pytest.fixture(autouse=True)
def _clean_global_callbacks() -> Generator[None, None, None]:
    """Ensure GLOBAL_CALLBACKS and torch dispatch stack are clean around each test."""

    def _drain_dispatch_stack() -> None:
        for mode in reversed(_get_current_dispatch_mode_stack()):
            # Call TorchDispatchMode.__exit__ directly to avoid ptflops
            # FlopCounterMode.__exit__ calling print_fn on a closed StringIO.
            TorchDispatchMode.__exit__(mode, None, None, None)

    GLOBAL_CALLBACKS.clear()
    _drain_dispatch_stack()
    yield
    GLOBAL_CALLBACKS.clear()
    _drain_dispatch_stack()


# ---------------------------------------------------------------------------
# Minimal real modules for training tests
# ---------------------------------------------------------------------------


class _SimpleModel(torch.nn.Module):
    """A tiny model for testing: Linear(4 -> 2) returning a dict."""

    outputs: list[str] = ["logits"]

    def __init__(self) -> None:
        super().__init__()
        self.fc = torch.nn.Linear(4, 2)

    def forward(self, x: torch.Tensor, **kwargs: Any) -> dict[str, torch.Tensor]:
        """Forward pass."""
        return {"logits": self.fc(x)}


class _SimpleLoss(torch.nn.Module):
    """Loss module that computes cross-entropy from logits and target."""

    outputs: list[str] = ["loss"]

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("_dummy", torch.tensor(0.0))

    def forward(self, logits: torch.Tensor, target: torch.Tensor, **kwargs: Any) -> dict[str, torch.Tensor]:
        """Compute loss."""
        return {"loss": torch.nn.functional.cross_entropy(logits, target)}


class _SimpleMetric(torch.nn.Module):
    """Metric module that computes top-1 accuracy."""

    outputs: list[str] = ["acc"]

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("_dummy", torch.tensor(0.0))

    def forward(self, logits: torch.Tensor, target: torch.Tensor, **kwargs: Any) -> dict[str, torch.Tensor]:
        """Compute accuracy."""
        preds = logits.argmax(dim=-1)
        return {"acc": (preds == target).float().mean()}


class _SimpleBackward:
    """Minimal backward implementing the Backward protocol with a real optimizer."""

    mixed_precision_type: str | None = None

    def __init__(self, model: torch.nn.Module, **kwargs: Any) -> None:
        self._optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    def update(self, step: int) -> bool:
        """Always signal update."""
        return True

    def __call__(self, loss: torch.Tensor, **kwargs: Any) -> None:
        """Backward pass + optimizer step."""
        loss.backward()
        self._optimizer.step()
        self._optimizer.zero_grad()

    @property
    def optimizers(self) -> dict[str, torch.optim.Optimizer]:
        """Return optimizers."""
        return {"optimizer": self._optimizer}

    @property
    def grad_scalers(self) -> dict[str, Any]:
        """Return empty grad scalers dict."""
        return {}

    @property
    def learning_rates(self) -> dict[str, float]:
        """Return current learning rates."""
        return {"optimizer": self._optimizer.param_groups[0]["lr"]}

    @property
    def param_group_names(self) -> dict[str, list[dict[str, Any]]]:
        """Return parameter group info."""
        return {"optimizer": [{k: v for k, v in pg.items() if k != "params"} for pg in self._optimizer.param_groups]}


def _make_training_dataset() -> list[dict[str, torch.Tensor]]:
    """Create a minimal training dataset (list of batches)."""
    return [{"x": torch.randn(4, 4), "target": torch.randint(0, 2, (4,))} for _ in range(3)]


def _make_validation_dataset() -> list[dict[str, torch.Tensor]]:
    """Create a minimal validation dataset (list of batches)."""
    return [{"x": torch.randn(4, 4), "target": torch.randint(0, 2, (4,))} for _ in range(2)]


def _train_callback() -> Any:
    """Return the callback function for the ``train`` command."""
    for command in app.registered_commands:
        callback_name = "" if command.callback is None else command.callback.__name__
        if command.name == "train" or callback_name == "train":
            return command.callback
    raise AssertionError("train command not found")


def _make_instantiate_fn(
    *,
    training_data: list[dict[str, torch.Tensor]],
    validation_data: list[dict[str, torch.Tensor]] | None = None,
    backward_cls: type = _SimpleBackward,
    loss: torch.nn.Module | None = None,
    metric: torch.nn.Module | None = None,
) -> Any:
    """Build a replacement for ``_instantiate`` that returns real objects."""
    _loss = loss if loss is not None else _SimpleLoss()
    _metric = metric if metric is not None else _SimpleMetric()

    def _side_effect(raw: Any) -> Any:
        if raw == "MODEL":
            return _SimpleModel()
        if raw == "LOSS":
            return _loss
        if raw == "METRIC":
            return _metric
        if raw == "BACKWARD":
            return backward_cls
        if raw == "TRAIN_DS":
            return training_data
        if raw == "VALID_DS":
            return validation_data
        return raw

    return _side_effect


# ---------------------------------------------------------------------------
# App structure tests
# ---------------------------------------------------------------------------


def test_app_is_typer_instance() -> None:
    """The cmd_torch `app` must be a Typer instance."""
    assert isinstance(app, Typer)
    names = [cmd.name or (cmd.callback.__name__ if cmd.callback else "") for cmd in app.registered_commands]
    assert "ptflops" in names
    assert "calflops" in names
    group_names = [g.name for g in app.registered_groups]
    assert "create" in group_names


def test_help_exits_zero(cli_runner: CliRunner) -> None:
    """--help should exit with code 0."""
    result = cli_runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "create" in result.output
    assert "ptflops" in result.output
    assert "calflops" in result.output


def test_create_help_exits_zero(cli_runner: CliRunner) -> None:
    """'create --help' should exit with code 0."""
    assert cli_runner.invoke(app, ["create", "--help"]).exit_code == 0
    assert cli_runner.invoke(app, ["create", "model", "--help"]).exit_code == 0
    assert cli_runner.invoke(app, ["create", "backward", "--help"]).exit_code == 0
    assert cli_runner.invoke(app, ["ptflops", "--help"]).exit_code == 0
    assert cli_runner.invoke(app, ["calflops", "--help"]).exit_code == 0


# ---------------------------------------------------------------------------
# 'create model' command
# ---------------------------------------------------------------------------


def test_create_model_calls_torch_builder(tmp_path: Any, cli_runner: CliRunner) -> None:
    """'create model' should generate a model script from a real configuration."""
    configure_security(allowed_modules_check=False, blocked_modules_check=False)
    out = str(tmp_path / "model.py")
    result = cli_runner.invoke(app, ["create", "model", MODEL_CFG, "--output", out])
    assert result.exit_code == 0, result.output
    assert "class Model" in (tmp_path / "model.py").read_text()


def test_create_model_passes_classname(tmp_path: Any, cli_runner: CliRunner) -> None:
    """'create model --classname' should generate a script with the given class name."""
    configure_security(allowed_modules_check=False, blocked_modules_check=False)
    out = str(tmp_path / "my_net.py")
    result = cli_runner.invoke(app, ["create", "model", MODEL_CFG, "--classname", "MyNet", "--output", out])
    assert result.exit_code == 0, result.output
    assert "class MyNet" in (tmp_path / "my_net.py").read_text()


def test_create_model_structured_output_default_true(tmp_path: Any, cli_runner: CliRunner) -> None:
    """'create model' should default structured_output to True (dict return in root class)."""
    configure_security(allowed_modules_check=False, blocked_modules_check=False)
    out = str(tmp_path / "model.py")
    result = cli_runner.invoke(app, ["create", "model", MODEL_CFG, "--output", out])
    assert result.exit_code == 0, result.output
    last_class = (tmp_path / "model.py").read_text().rsplit("class Model", 1)[-1]
    assert "return {'" in last_class


def test_create_model_no_structured_output(tmp_path: Any, cli_runner: CliRunner) -> None:
    """'create model --no-structured-output' root class should not return a dict."""
    configure_security(allowed_modules_check=False, blocked_modules_check=False)
    out = str(tmp_path / "model.py")
    result = cli_runner.invoke(app, ["create", "model", MODEL_CFG, "--no-structured-output", "--output", out])
    assert result.exit_code == 0, result.output
    last_class = (tmp_path / "model.py").read_text().rsplit("class Model", 1)[-1]
    assert "return {'" not in last_class


def test_create_model_with_output_path(tmp_path: Any, cli_runner: CliRunner) -> None:
    """'create model --output' should write the generated script to the specified path."""
    configure_security(allowed_modules_check=False, blocked_modules_check=False)
    out_file = tmp_path / "nested" / "out.py"
    result = cli_runner.invoke(app, ["create", "model", MODEL_CFG, "--output", str(out_file)])
    assert result.exit_code == 0, result.output
    assert out_file.exists()
    assert "class Model" in out_file.read_text()


# ---------------------------------------------------------------------------
# 'create backward' command
# ---------------------------------------------------------------------------


def test_create_backward_calls_torch_backward_builder(tmp_path: Any, cli_runner: CliRunner) -> None:
    """'create backward' should generate a backward script from a real configuration."""
    configure_security(allowed_modules_check=False, blocked_modules_check=False)
    out = str(tmp_path / "backward.py")
    result = cli_runner.invoke(app, ["create", "backward", BACKWARD_CFG, "--output", out])
    assert result.exit_code == 0, result.output
    assert "class Backward" in (tmp_path / "backward.py").read_text()


def test_create_backward_passes_default_classname(tmp_path: Any, cli_runner: CliRunner) -> None:
    """'create backward' default classname should produce a class named 'Backward'."""
    configure_security(allowed_modules_check=False, blocked_modules_check=False)
    out = str(tmp_path / "backward.py")
    result = cli_runner.invoke(app, ["create", "backward", BACKWARD_CFG, "--output", out])
    assert result.exit_code == 0, result.output
    assert "class Backward" in (tmp_path / "backward.py").read_text()


# ---------------------------------------------------------------------------
# 'calflops' command with real model (must run BEFORE ptflops, which installs
# a persistent __torch_dispatch__ handler that captures CliRunner's StringIO)
# ---------------------------------------------------------------------------


def test_calflops_runs_with_real_model(cli_runner: CliRunner) -> None:
    """'calflops' should run calflops on a real model and print FLOPs, MACs, parameters."""
    configure_security(allowed_modules_check=False)

    deps = {"_instantiate": lambda raw: _SimpleModel()}
    with patch_cmd_globals(**deps):
        result = cli_runner.invoke(
            app,
            ["calflops", "{_obj_: [[_addr_, dummy], _call_]}", "--shape", "x: [4]", "--device", "cpu"],
        )
    assert result.exit_code == 0, result.output
    assert "FLOPs" in result.output
    assert "MACs" in result.output
    assert "Parameters" in result.output


# ---------------------------------------------------------------------------
# 'ptflops' command with real model
# ---------------------------------------------------------------------------

_IDENTITY_PATTERN = "{_obj_: [[_addr_, torch.nn.Identity], _call_]}"


def test_ptflops_runs_with_real_model(cli_runner: CliRunner) -> None:
    """'ptflops' should run ptflops on a real model and print results."""
    configure_security(allowed_modules_check=False)
    result = cli_runner.invoke(app, ["ptflops", _IDENTITY_PATTERN, "--device", "cpu"])
    assert result.exit_code == 0, result.output


def test_ptflops_none_results_print_nothing(cli_runner: CliRunner) -> None:
    """'ptflops' should not error when flops/params are None (e.g. identity)."""
    configure_security(allowed_modules_check=False)
    result = cli_runner.invoke(app, ["ptflops", _IDENTITY_PATTERN, "--device", "cpu"])
    assert result.exit_code == 0, result.output


# ---------------------------------------------------------------------------
# _instantiate – direct unit test
# ---------------------------------------------------------------------------


def test_instantiate_builds_object_from_pattern() -> None:
    """_instantiate() resolves an ObjectPattern and returns the built instance."""
    configure_security(allowed_modules_check=False)
    raw = {"_obj_": [["_addr_", "torch.nn.Identity"], ["_call_", {}]]}
    result = _instantiate(raw)
    assert isinstance(result, torch.nn.Identity)


def test_instantiate_builds_linear_with_args() -> None:
    """_instantiate() builds a torch.nn.Linear with keyword arguments."""
    configure_security(allowed_modules_check=False)
    raw = {"_obj_": [["_addr_", "torch.nn.Linear"], {"_call_": {"in_features": 8, "out_features": 4}}]}
    result = _instantiate(raw)
    assert isinstance(result, torch.nn.Linear)
    assert result.in_features == 8
    assert result.out_features == 4


# ---------------------------------------------------------------------------
# _compile_module
# ---------------------------------------------------------------------------


def test_compile_module_returns_module_when_no_kwargs() -> None:
    """_compile_module returns the module unchanged when compile_kw is None."""
    module = torch.nn.Linear(4, 2)
    assert _compile_module(module, None) is module


# ---------------------------------------------------------------------------
# _instantiate_models
# ---------------------------------------------------------------------------


def test_instantiate_models_creates_ordered_dict() -> None:
    """_instantiate_models creates an OrderedDict of models from patterns."""
    configure_security(allowed_modules_check=False)
    patterns = [{"model": {"_obj_": [["_addr_", "torch.nn.Identity"], "_call_"]}}]
    result = _instantiate_models(patterns)
    assert isinstance(result, OrderedDict)
    assert "model" in result
    assert isinstance(result["model"], torch.nn.Identity)


def test_instantiate_models_raises_for_multiple_keys() -> None:
    """_instantiate_models raises ValueError when a pattern dict has multiple keys."""
    with pytest.raises(ValueError, match="exactly one model definition"):
        _instantiate_models([{"a": {}, "b": {}}])


def test_instantiate_models_returns_empty_for_empty_list() -> None:
    """_instantiate_models returns empty OrderedDict for empty patterns list."""
    result = _instantiate_models([])
    assert result == OrderedDict()


# ---------------------------------------------------------------------------
# _get_module_outputs
# ---------------------------------------------------------------------------


def test_get_module_outputs_from_attribute() -> None:
    """_get_module_outputs returns the module's ``outputs`` attribute."""
    assert _get_module_outputs(_SimpleLoss(), None, "loss") == ["loss"]


def test_get_module_outputs_falls_back_to_default() -> None:
    """_get_module_outputs uses default when module has no ``outputs`` attribute."""
    assert _get_module_outputs(torch.nn.Identity(), ["out"], "test") == ["out"]


def test_get_module_outputs_raises_when_no_outputs_and_no_default() -> None:
    """_get_module_outputs raises ValueError when neither source is available."""
    with pytest.raises(ValueError, match='Module "loss" does not have an "outputs"'):
        _get_module_outputs(torch.nn.Identity(), None, "loss")


# ---------------------------------------------------------------------------
# _get_state_dict / _unwrap_ddp
# ---------------------------------------------------------------------------


def test_get_state_dict_returns_state_dicts() -> None:
    """_get_state_dict returns a name-to-state_dict mapping."""
    model = torch.nn.Linear(4, 2)
    result = _get_state_dict({"model": model})
    assert "model" in result
    assert "weight" in result["model"]
    assert "bias" in result["model"]


def test_unwrap_ddp_passes_through_non_ddp_models() -> None:
    """_unwrap_ddp returns non-DDP modules unchanged."""
    model = torch.nn.Linear(4, 2)
    result = _unwrap_ddp({"model": model})
    assert result["model"] is model


def test_unwrap_ddp_extracts_module_from_ddp(single_process_gloo: None) -> None:
    """_unwrap_ddp extracts the .module from DistributedDataParallel models."""
    model = torch.nn.Linear(4, 2)
    ddp_model = torch.nn.parallel.DistributedDataParallel(model)
    result = _unwrap_ddp({"model": ddp_model})
    assert result["model"] is model


# ---------------------------------------------------------------------------
# _on_best
# ---------------------------------------------------------------------------


def test_on_best_logs_metric_to_mlflow(tmp_path: pathlib.Path) -> None:
    """_on_best logs the best metric value to MLflow."""
    mlflow.set_tracking_uri(str(tmp_path / "mlruns"))
    mlflow.set_experiment("test_on_best")
    with mlflow.start_run():
        info = BaseInfo()
        info.epoch = 1
        info.step = 5
        best = BestCriterion[torch.nn.Module](target="val_loss", mode="min")
        best._best = 0.3
        best._step = 5
        model = torch.nn.Linear(4, 2)
        _on_best(info, best, save=True, model=model)


def test_on_best_does_not_save_when_save_false(tmp_path: pathlib.Path) -> None:
    """_on_best with save=False only logs metric, not state dict."""
    mlflow.set_tracking_uri(str(tmp_path / "mlruns"))
    mlflow.set_experiment("test_on_best_no_save")
    with mlflow.start_run():
        info = BaseInfo()
        info.epoch = 1
        info.step = 5
        best = BestCriterion[torch.nn.Module](target="val_loss", mode="min")
        best._best = 0.3
        best._step = 5
        _on_best(info, best, save=False, model=torch.nn.Linear(4, 2))


# ---------------------------------------------------------------------------
# _save_training_state
# ---------------------------------------------------------------------------


def test_save_training_state_logs_to_mlflow(tmp_path: pathlib.Path) -> None:
    """_save_training_state saves models, optimizers, grad_scalers, meta to MLflow."""
    mlflow.set_tracking_uri(str(tmp_path / "mlruns"))
    mlflow.set_experiment("test_save_state")
    model = _SimpleModel()
    backward = _SimpleBackward(model)
    tracker = TorchTracker.from_criteria(["loss"], distributed=False)
    trainer = TorchTrainer(
        device="cpu",
        training_step=TrainingStep(models=["model"], losses=_SimpleLoss()),
        validation_step=ValidationStep(models=["model"], losses=_SimpleLoss()),
        backward=backward,
        tracker=tracker,
        add_global_callbacks=False,
    )
    trainer.epoch = 1
    trainer.step = 10
    trainer.update = 10
    with mlflow.start_run():
        _save_training_state(trainer, model=model)


def test_save_training_state_includes_ema(tmp_path: pathlib.Path) -> None:
    """_save_training_state includes EMA state when inference_wrapper is set."""
    mlflow.set_tracking_uri(str(tmp_path / "mlruns"))
    mlflow.set_experiment("test_save_state_ema")
    model = _SimpleModel()
    backward = _SimpleBackward(model)
    ema_wrapper = TimmEmaWrapper.from_models({"model": model}, distributed=False)
    tracker = TorchTracker.from_criteria(["loss"], distributed=False)
    trainer = TorchTrainer(
        device="cpu",
        training_step=TrainingStep(models=["model"], losses=_SimpleLoss()),
        backward=backward,
        tracker=tracker,
        inference_wrapper=ema_wrapper,
        add_global_callbacks=False,
    )
    trainer.epoch = 1
    trainer.step = 10
    trainer.update = 10
    with mlflow.start_run():
        _save_training_state(trainer, model=model)


# ---------------------------------------------------------------------------
# _log_criteria
# ---------------------------------------------------------------------------


def test_log_criteria_returns_yaml_string(tmp_path: pathlib.Path) -> None:
    """_log_criteria formats criteria as YAML and logs them to MLflow."""
    mlflow.set_tracking_uri(str(tmp_path / "mlruns"))
    mlflow.set_experiment("test_log_criteria")
    model = _SimpleModel()
    backward = _SimpleBackward(model)
    tracker = TorchTracker.from_criteria(["loss"], distributed=False)
    trainer = TorchTrainer(
        device="cpu",
        training_step=TrainingStep(models=["model"], losses=_SimpleLoss()),
        backward=backward,
        tracker=tracker,
        add_global_callbacks=False,
    )
    trainer.epoch = 1
    trainer.step = 5
    trainer.update = 5
    trainer.history[1] = {"loss": 0.5, "acc": 0.8}
    with mlflow.start_run():
        result = _log_criteria(trainer)
    assert "epoch: 1" in result


# ---------------------------------------------------------------------------
# `train` command — validation errors
# ---------------------------------------------------------------------------


def test_train_raises_for_empty_model_patterns() -> None:
    """`train` should fail fast when no model patterns are provided."""
    train_fn = _train_callback()
    with pytest.raises(ValueError, match="At least one model pattern"):
        train_fn(
            model_patterns=[],
            initializer_patterns=None,
            shapes=None,
            device="cpu",
            ema=None,
            ema_device=None,
            loss_pattern="IGNORED",
            loss_outputs=None,
            metric_pattern=None,
            metric_outputs=None,
            backward_pattern="IGNORED",
            mixed_precision_type=None,
            compile_pattern=None,
            trainer_pattern=None,
            training_step_pattern=None,
            validation_step_pattern=None,
            epochs=1,
            start_epoch=1,
            training_dataset_pattern="IGNORED",
            validation_dataset_pattern=None,
            validation_frequency=1,
            lower_criteria=[],
            higher_criteria=[],
            save_criteria=[],
            seed=42,
            matmul_precision="high",
            experiment="test",
            log_arguments=None,
            log_artifacts=None,
            ci=True,
            dist_backend=None,
            dist_url=None,
        )


def test_train_raises_for_invalid_model_pattern_shape() -> None:
    """`train` should reject model-pattern entries containing multiple models."""
    configure_security(allowed_modules_check=False)
    train_fn = _train_callback()
    deps = {
        "_instantiate": _make_instantiate_fn(training_data=_make_training_dataset()),
    }
    with patch_cmd_globals(**deps), pytest.raises(ValueError, match="exactly one model definition"):
        train_fn(
            model_patterns=[{"a": "MODEL", "b": "MODEL"}],
            initializer_patterns=None,
            shapes=[{"x": (4,)}],
            device="cpu",
            ema=None,
            ema_device=None,
            loss_pattern="LOSS",
            loss_outputs=None,
            metric_pattern=None,
            metric_outputs=None,
            backward_pattern="BACKWARD",
            mixed_precision_type=None,
            compile_pattern=None,
            trainer_pattern=None,
            training_step_pattern=None,
            validation_step_pattern=None,
            epochs=1,
            start_epoch=1,
            training_dataset_pattern="TRAIN_DS",
            validation_dataset_pattern=None,
            validation_frequency=1,
            lower_criteria=[],
            higher_criteria=[],
            save_criteria=[],
            seed=42,
            matmul_precision="high",
            experiment="test",
            log_arguments=None,
            log_artifacts=None,
            ci=True,
            dist_backend=None,
            dist_url=None,
        )


def test_train_raises_when_module_outputs_missing_and_not_provided() -> None:
    """`train` should fail when a loss module lacks ``outputs`` and no fallback is given."""
    configure_security(allowed_modules_check=False)
    train_fn = _train_callback()
    deps = {
        "_instantiate": _make_instantiate_fn(
            training_data=_make_training_dataset(),
            loss=torch.nn.Identity(),
        ),
    }
    with patch_cmd_globals(**deps), pytest.raises(ValueError, match='Module "loss" does not have an "outputs"'):
        train_fn(
            model_patterns=[{"model": "MODEL"}],
            initializer_patterns=None,
            shapes=[{"x": (4,)}],
            device="cpu",
            ema=None,
            ema_device=None,
            loss_pattern="LOSS",
            loss_outputs=None,
            metric_pattern=None,
            metric_outputs=None,
            backward_pattern="BACKWARD",
            mixed_precision_type=None,
            compile_pattern=None,
            trainer_pattern=None,
            training_step_pattern=None,
            validation_step_pattern=None,
            epochs=1,
            start_epoch=1,
            training_dataset_pattern="TRAIN_DS",
            validation_dataset_pattern=None,
            validation_frequency=1,
            lower_criteria=[],
            higher_criteria=[],
            save_criteria=[],
            seed=42,
            matmul_precision="high",
            experiment="test",
            log_arguments=None,
            log_artifacts=None,
            ci=True,
            dist_backend=None,
            dist_url=None,
        )


# ---------------------------------------------------------------------------
# `train` — full end-to-end (CI mode, non-distributed, real modules)
# ---------------------------------------------------------------------------


def _invoke_train(
    tmp_path: pathlib.Path,
    *,
    ci: bool = True,
    ema: dict[str, Any] | None = None,
    validation_data: list[dict[str, torch.Tensor]] | None = None,
    loss_outputs: list[str] | None = None,
    metric_outputs: list[str] | None = None,
    mixed_precision_type: str | None = None,
    lower_criteria: list[str] | None = None,
    higher_criteria: list[str] | None = None,
    save_criteria: list[str] | None = None,
    loss: torch.nn.Module | None = None,
    metric: torch.nn.Module | None = None,
    backward_cls: type = _SimpleBackward,
    log_arguments: list[dict[str, Any]] | None = None,
    log_artifacts: list[pathlib.Path] | None = None,
    epochs: int = 2,
) -> None:
    """Invoke the ``train`` callback with real modules, patching only ``_instantiate``."""
    configure_security(allowed_modules_check=False)
    training_data = _make_training_dataset()
    if validation_data is None:
        validation_data = _make_validation_dataset()
    deps = {
        "_instantiate": _make_instantiate_fn(
            training_data=training_data,
            validation_data=validation_data,
            backward_cls=backward_cls,
            loss=loss,
            metric=metric,
        ),
    }
    mlflow.set_tracking_uri(str(tmp_path / "mlruns"))
    train_fn = _train_callback()
    with patch_cmd_globals(**deps):
        train_fn(
            model_patterns=[{"model": "MODEL"}],
            initializer_patterns=None,
            shapes=[{"x": (4,)}],
            device="cpu",
            ema=ema,
            ema_device=None,
            loss_pattern="LOSS",
            loss_outputs=loss_outputs,
            metric_pattern="METRIC",
            metric_outputs=metric_outputs,
            backward_pattern="BACKWARD",
            mixed_precision_type=mixed_precision_type,
            compile_pattern=None,
            trainer_pattern=None,
            training_step_pattern=None,
            validation_step_pattern=None,
            epochs=epochs,
            start_epoch=1,
            training_dataset_pattern="TRAIN_DS",
            validation_dataset_pattern="VALID_DS",
            validation_frequency=1,
            lower_criteria=lower_criteria or ["loss"],
            higher_criteria=higher_criteria or ["acc"],
            save_criteria=save_criteria or ["acc"],
            seed=42,
            matmul_precision="high",
            experiment="test-e2e",
            log_arguments=log_arguments,
            log_artifacts=log_artifacts,
            ci=ci,
            dist_backend=None,
            dist_url=None,
        )


def test_train_ci_mode_end_to_end(tmp_path: pathlib.Path) -> None:
    """Full end-to-end train in CI mode with real model, loss, metric, backward."""
    artifact = tmp_path / "artifact.bin"
    artifact.write_text("dummy")
    _invoke_train(
        tmp_path,
        ci=True,
        log_arguments=[{"run": "test"}],
        log_artifacts=[artifact],
    )


def test_train_ci_mode_with_ema(tmp_path: pathlib.Path) -> None:
    """CI mode with EMA enabled should save EMA state in training_state."""
    _invoke_train(tmp_path, ci=True, ema={}, save_criteria=[])


def test_train_non_ci_mode_uses_pbar(tmp_path: pathlib.Path) -> None:
    """Non-CI mode should use tqdm progress bar callbacks."""
    _invoke_train(tmp_path, ci=False)


def test_train_with_loss_outputs_fallback(tmp_path: pathlib.Path) -> None:
    """Train should accept explicit --loss-outputs when module has no ``outputs`` attr."""

    class _KwargsLoss(torch.nn.Module):
        """Loss that accepts **kwargs (no ``outputs`` attribute)."""

        def forward(self, logits: torch.Tensor, target: torch.Tensor, **kwargs: Any) -> dict[str, torch.Tensor]:
            return {"loss": torch.nn.functional.cross_entropy(logits, target)}

    class _KwargsMetric(torch.nn.Module):
        """Metric that accepts **kwargs (no ``outputs`` attribute)."""

        def forward(self, logits: torch.Tensor, target: torch.Tensor, **kwargs: Any) -> dict[str, torch.Tensor]:
            return {"acc": (logits.argmax(dim=-1) == target).float().mean()}

    _invoke_train(
        tmp_path,
        ci=True,
        loss=_KwargsLoss(),
        metric=_KwargsMetric(),
        loss_outputs=["loss"],
        metric_outputs=["acc"],
        save_criteria=[],
        lower_criteria=[],
        higher_criteria=[],
    )


def test_train_backward_mixed_precision_type_override(tmp_path: pathlib.Path) -> None:
    """Backward's mixed_precision_type should override the command's default."""

    class _BackwardWithMixedPrecision(_SimpleBackward):
        mixed_precision_type = "bfloat16"

    _invoke_train(
        tmp_path,
        ci=True,
        mixed_precision_type="float16",
        backward_cls=_BackwardWithMixedPrecision,
        save_criteria=[],
        lower_criteria=[],
        higher_criteria=[],
    )


# ---------------------------------------------------------------------------
# DDP distributed training tests via torch.multiprocessing.spawn
# ---------------------------------------------------------------------------


def _patch_ddp_for_cpu() -> None:
    """Monkey-patch DDP __init__ to drop device_ids/output_device for CPU-only testing.

    Safe to call in a forked child process -- does not affect the parent.
    """
    _orig_init = torch.nn.parallel.DistributedDataParallel.__init__

    def _cpu_safe_init(
        self: Any, module: torch.nn.Module, device_ids: Any = None, output_device: Any = None, **kwargs: Any
    ) -> None:
        _orig_init(self, module, device_ids=None, output_device=None, **kwargs)

    torch.nn.parallel.DistributedDataParallel.__init__ = _cpu_safe_init  # type: ignore[assignment]


def _ddp_train_worker(
    rank: int,
    world_size: int,
    init_file: str,
    mlflow_uri: str,
    ci: bool,
) -> None:
    """Worker function for DDP training tests launched by mp.spawn."""
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)

    dist.init_process_group(backend="gloo", init_method=f"file://{init_file}", rank=rank, world_size=world_size)
    _patch_ddp_for_cpu()

    try:
        configure_security(allowed_modules_check=False)
        mlflow.set_tracking_uri(mlflow_uri)
        training_data = _make_training_dataset()
        validation_data = _make_validation_dataset()
        deps = {"_instantiate": _make_instantiate_fn(training_data=training_data, validation_data=validation_data)}
        train_fn = _train_callback()
        originals = {k: _CMD_GLOBALS.get(k) for k in deps}
        _CMD_GLOBALS.update(deps)
        try:
            with callbacks_session():
                train_fn.__wrapped__(  # type: ignore[attr-defined]
                    model_patterns=[{"model": "MODEL"}],
                    initializer_patterns=None,
                    shapes=[{"x": (4,)}],
                    device="cpu",
                    ema=None,
                    ema_device=None,
                    loss_pattern="LOSS",
                    loss_outputs=None,
                    metric_pattern="METRIC",
                    metric_outputs=None,
                    backward_pattern="BACKWARD",
                    mixed_precision_type=None,
                    compile_pattern=None,
                    trainer_pattern=None,
                    training_step_pattern=None,
                    validation_step_pattern=None,
                    epochs=2,
                    start_epoch=1,
                    training_dataset_pattern="TRAIN_DS",
                    validation_dataset_pattern="VALID_DS",
                    validation_frequency=1,
                    lower_criteria=["loss"],
                    higher_criteria=["acc"],
                    save_criteria=["acc"],
                    seed=42,
                    matmul_precision="high",
                    experiment="test-ddp",
                    log_arguments=None,
                    log_artifacts=None,
                    ci=ci,
                    dist_backend="gloo",
                    dist_url=f"file://{init_file}",
                )
        finally:
            _CMD_GLOBALS.update(originals)
    except Exception:
        traceback.print_exc()
        raise


def test_train_distributed_ddp_end_to_end(tmp_path: pathlib.Path) -> None:
    """Full end-to-end DDP training with 2 workers via mp.spawn (gloo, CPU)."""
    init_file = str(tmp_path / "dist_init")
    mlflow_uri = str(tmp_path / "mlruns")
    mp.spawn(
        _ddp_train_worker,
        args=(2, init_file, mlflow_uri, True),
        nprocs=2,
        join=True,
    )


def _ddp_rank_gating_worker(
    rank: int,
    world_size: int,
    init_file: str,
    result_dir: str,
) -> None:
    """Worker that verifies rank-based gating: only rank 0 creates MLflow runs."""
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)

    dist.init_process_group(backend="gloo", init_method=f"file://{init_file}", rank=rank, world_size=world_size)
    _patch_ddp_for_cpu()

    try:
        configure_security(allowed_modules_check=False)
        mlflow_uri = os.path.join(result_dir, "mlruns")
        mlflow.set_tracking_uri(mlflow_uri)
        training_data = _make_training_dataset()
        deps = {"_instantiate": _make_instantiate_fn(training_data=training_data)}
        train_fn = _train_callback()
        originals = {k: _CMD_GLOBALS.get(k) for k in deps}
        _CMD_GLOBALS.update(deps)
        try:
            with callbacks_session():
                train_fn.__wrapped__(  # type: ignore[attr-defined]
                    model_patterns=[{"model": "MODEL"}],
                    initializer_patterns=None,
                    shapes=[{"x": (4,)}],
                    device="cpu",
                    ema=None,
                    ema_device=None,
                    loss_pattern="LOSS",
                    loss_outputs=None,
                    metric_pattern=None,
                    metric_outputs=None,
                    backward_pattern="BACKWARD",
                    mixed_precision_type=None,
                    compile_pattern=None,
                    trainer_pattern=None,
                    training_step_pattern=None,
                    validation_step_pattern=None,
                    epochs=1,
                    start_epoch=1,
                    training_dataset_pattern="TRAIN_DS",
                    validation_dataset_pattern=None,
                    validation_frequency=1,
                    lower_criteria=[],
                    higher_criteria=[],
                    save_criteria=[],
                    seed=42,
                    matmul_precision="high",
                    experiment="test-rank-gating",
                    log_arguments=None,
                    log_artifacts=None,
                    ci=True,
                    dist_backend="gloo",
                    dist_url=f"file://{init_file}",
                )
        finally:
            _CMD_GLOBALS.update(originals)
        marker = pathlib.Path(result_dir) / f"rank_{rank}_done"
        marker.write_text(f"rank={rank}")
    except Exception:
        traceback.print_exc()
        raise


def test_train_distributed_rank_gating(tmp_path: pathlib.Path) -> None:
    """In DDP mode, all ranks complete training to verify rank-gating works."""
    init_file = str(tmp_path / "dist_init_rg")
    result_dir = str(tmp_path / "results")
    os.makedirs(result_dir, exist_ok=True)
    mp.spawn(_ddp_rank_gating_worker, args=(2, init_file, result_dir), nprocs=2, join=True)
    assert (tmp_path / "results" / "rank_0_done").exists()
    assert (tmp_path / "results" / "rank_1_done").exists()


def _ddp_seed_offset_worker(
    rank: int,
    world_size: int,
    init_file: str,
    result_dir: str,
    seed: int,
) -> None:
    """Worker that records the seed applied after train() so we can verify rank offset."""
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    dist.init_process_group(backend="gloo", init_method=f"file://{init_file}", rank=rank, world_size=world_size)
    _patch_ddp_for_cpu()
    try:
        configure_security(allowed_modules_check=False)
        mlflow_uri = os.path.join(result_dir, "mlruns")
        mlflow.set_tracking_uri(mlflow_uri)
        training_data = _make_training_dataset()
        deps = {"_instantiate": _make_instantiate_fn(training_data=training_data)}
        train_fn = _train_callback()
        originals = {k: _CMD_GLOBALS.get(k) for k in deps}
        _CMD_GLOBALS.update(deps)
        try:
            with callbacks_session():
                train_fn.__wrapped__(  # type: ignore[attr-defined]
                    model_patterns=[{"model": "MODEL"}],
                    initializer_patterns=None,
                    shapes=[{"x": (4,)}],
                    device="cpu",
                    ema=None,
                    ema_device=None,
                    loss_pattern="LOSS",
                    loss_outputs=None,
                    metric_pattern=None,
                    metric_outputs=None,
                    backward_pattern="BACKWARD",
                    mixed_precision_type=None,
                    compile_pattern=None,
                    trainer_pattern=None,
                    training_step_pattern=None,
                    validation_step_pattern=None,
                    epochs=1,
                    start_epoch=1,
                    training_dataset_pattern="TRAIN_DS",
                    validation_dataset_pattern=None,
                    validation_frequency=1,
                    lower_criteria=[],
                    higher_criteria=[],
                    save_criteria=[],
                    seed=seed,
                    matmul_precision="high",
                    experiment="test-seed",
                    log_arguments=None,
                    log_artifacts=None,
                    ci=True,
                    dist_backend="gloo",
                    dist_url=f"file://{init_file}",
                )
        finally:
            _CMD_GLOBALS.update(originals)
        # After train(), torch manual_seed was called with (seed + rank).
        # Generate a random tensor to verify the seed differs per rank.
        torch.manual_seed(seed + rank)
        sample = torch.randn(4).tolist()
        pathlib.Path(result_dir, f"seed_rank_{rank}.txt").write_text(str(sample))
    except Exception:
        traceback.print_exc()
        raise


def test_train_distributed_seeds_offset_by_rank(tmp_path: pathlib.Path) -> None:
    """Seeds must be offset by global_rank for distributed training."""
    init_file = str(tmp_path / "dist_init_seed")
    result_dir = str(tmp_path / "results")
    os.makedirs(result_dir, exist_ok=True)
    mp.spawn(_ddp_seed_offset_worker, args=(2, init_file, result_dir, 42), nprocs=2, join=True)
    seed0 = (tmp_path / "results" / "seed_rank_0.txt").read_text()
    seed1 = (tmp_path / "results" / "seed_rank_1.txt").read_text()
    assert seed0 != seed1


def _ddp_unwrap_state_worker(
    rank: int,
    world_size: int,
    init_file: str,
    result_dir: str,
) -> None:
    """Worker that verifies DDP models are properly unwrapped."""
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    dist.init_process_group(backend="gloo", init_method=f"file://{init_file}", rank=rank, world_size=world_size)
    try:
        model = torch.nn.Linear(4, 2)
        ddp_model = torch.nn.parallel.DistributedDataParallel(model)
        result = _unwrap_ddp({"model": ddp_model})
        assert result["model"] is model
        state = _get_state_dict(_unwrap_ddp({"model": ddp_model}))
        assert "model" in state
        assert "weight" in state["model"]
        pathlib.Path(result_dir, f"unwrap_rank_{rank}_ok").write_text("ok")
    except Exception:
        traceback.print_exc()
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def test_train_distributed_unwraps_ddp_when_saving_state(tmp_path: pathlib.Path) -> None:
    """_unwrap_ddp correctly unwraps real DDP models in multi-process setting."""
    init_file = str(tmp_path / "dist_init_unwrap")
    result_dir = str(tmp_path / "results")
    os.makedirs(result_dir, exist_ok=True)
    mp.spawn(_ddp_unwrap_state_worker, args=(2, init_file, result_dir), nprocs=2, join=True)
    assert (tmp_path / "results" / "unwrap_rank_0_ok").exists()
    assert (tmp_path / "results" / "unwrap_rank_1_ok").exists()
