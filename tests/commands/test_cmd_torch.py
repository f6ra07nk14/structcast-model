"""Unit tests for structcast_model.commands.cmd_torch."""

from collections.abc import Generator
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
from structcast.utils.security import configure_security
from typer import Typer
from typer.testing import CliRunner

from structcast_model.base_trainer import NamedCallbackList
from structcast_model.commands.cmd_torch import app

# ---------------------------------------------------------------------------
# Helper: patch the real module globals (bypasses LazySelectedImporter proxy)
# ---------------------------------------------------------------------------


@contextmanager
def patch_cmd_globals(**kwargs: Any) -> Generator[None, Any, None]:
    """Temporarily override entries in cmd_torch's real module globals."""
    real = app.registered_commands[0].callback.__globals__
    originals = {k: real.get(k) for k in kwargs}
    real.update(kwargs)
    try:
        yield
    finally:
        real.update(originals)


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


def test_create_model_calls_torch_builder(tmp_path: Any, cli_runner: CliRunner) -> None:
    """'create model' should delegate to TorchBuilder.from_path."""
    cfg_file = tmp_path / "model.yaml"
    cfg_file.write_text("layer: Linear\n")
    mock_builder = MagicMock()
    with patch_cmd_globals(torch_builder=mock_builder):
        assert cli_runner.invoke(app, ["create", "model", str(cfg_file)]).exit_code == 0
    mock_builder.TorchBuilder.from_path.assert_called_once_with(str(cfg_file))


def test_create_model_passes_classname(tmp_path: Any, cli_runner: CliRunner) -> None:
    """'create model --classname' should forward the classname to the builder."""
    cfg_file = tmp_path / "model.yaml"
    cfg_file.write_text("layer: Linear\n")
    mock_builder = MagicMock()
    with patch_cmd_globals(torch_builder=mock_builder):
        assert cli_runner.invoke(app, ["create", "model", str(cfg_file), "--classname", "MyNet"]).exit_code == 0
    assert mock_builder.TorchBuilder.from_path.return_value.call_args[1]["classname"] == "MyNet"


def test_create_model_structured_output_default_true(tmp_path: Any, cli_runner: CliRunner) -> None:
    """'create model' should default structured_output to True."""
    cfg_file = tmp_path / "model.yaml"
    cfg_file.write_text("layer: Linear\n")
    mock_builder = MagicMock()
    with patch_cmd_globals(torch_builder=mock_builder):
        assert cli_runner.invoke(app, ["create", "model", str(cfg_file)]).exit_code == 0
    assert mock_builder.TorchBuilder.from_path.return_value.call_args[1]["forced_structured_output"] is True


def test_create_model_no_structured_output(tmp_path: Any, cli_runner: CliRunner) -> None:
    """'create model --no-structured-output' should pass False to the builder."""
    cfg_file = tmp_path / "model.yaml"
    cfg_file.write_text("layer: Linear\n")
    mock_builder = MagicMock()
    with patch_cmd_globals(torch_builder=mock_builder):
        assert cli_runner.invoke(app, ["create", "model", str(cfg_file), "--no-structured-output"]).exit_code == 0
    assert mock_builder.TorchBuilder.from_path.return_value.call_args[1]["forced_structured_output"] is False


def test_create_model_with_output_path(tmp_path: Any, cli_runner: CliRunner) -> None:
    """'create model --output' should forward the output path to the builder result."""
    cfg_file = tmp_path / "model.yaml"
    cfg_file.write_text("layer: Linear\n")
    out_file = str(tmp_path / "out.py")
    mock_builder = MagicMock()
    with patch_cmd_globals(torch_builder=mock_builder):
        assert cli_runner.invoke(app, ["create", "model", str(cfg_file), "--output", out_file]).exit_code == 0
    # The output path should be passed to the final call: builder(...)(...)(output)
    assert out_file in mock_builder.TorchBuilder.from_path.return_value.return_value.call_args[0]


# ---------------------------------------------------------------------------
# 'create backward' command execution
# ---------------------------------------------------------------------------


def test_create_backward_calls_torch_backward_builder(tmp_path: Any, cli_runner: CliRunner) -> None:
    """'create backward' should delegate to TorchBackwardBuilder.from_path."""
    cfg_file = tmp_path / "backward.yaml"
    cfg_file.write_text("layer: Linear\n")
    mock_builder = MagicMock()
    with patch_cmd_globals(torch_builder=mock_builder):
        assert cli_runner.invoke(app, ["create", "backward", str(cfg_file)]).exit_code == 0
    mock_builder.TorchBackwardBuilder.from_path.assert_called_once_with(str(cfg_file))


def test_create_backward_passes_default_classname(tmp_path: Any, cli_runner: CliRunner) -> None:
    """'create backward' default classname should be 'Backward'."""
    cfg_file = tmp_path / "backward.yaml"
    cfg_file.write_text("layer: Linear\n")
    mock_builder = MagicMock()
    with patch_cmd_globals(torch_builder=mock_builder):
        assert cli_runner.invoke(app, ["create", "backward", str(cfg_file)]).exit_code == 0
    assert mock_builder.TorchBackwardBuilder.from_path.return_value.call_args[1]["classname"] == "Backward"


# ---------------------------------------------------------------------------
# 'ptflops' command execution
# ---------------------------------------------------------------------------


def _make_torch_mock() -> MagicMock:
    """Create a torch mock that supports use as a context manager via torch.device(...)."""
    torch_mock = MagicMock()
    # torch.device(...) used as context manager via `with torch.device(...):`
    torch_mock.device.return_value.__enter__ = MagicMock(return_value=None)
    torch_mock.device.return_value.__exit__ = MagicMock(return_value=False)
    return torch_mock


MODEL_PATTERN_ARG = "{_obj_: [[_addr_, some.module.Model], _call_]}"


def test_ptflops_calls_instantiate(cli_runner: CliRunner) -> None:
    """'ptflops' should call _instantiate to build the model instance."""
    mock_instantiate = MagicMock(return_value=MagicMock())
    mock_torch_trainer = MagicMock()
    mock_torch_trainer.get_torch_device.return_value = "cpu"
    mock_torch_trainer.create_torch_inputs.return_value = {}
    mock_torch_trainer.initial_model.return_value = ({}, None)
    mock_ptflops = MagicMock()
    mock_ptflops.get_model_complexity_info.return_value = ("2.5 GMac", "3.0 M")
    with patch_cmd_globals(
        _instantiate=mock_instantiate,
        torch=_make_torch_mock(),
        torch_trainer=mock_torch_trainer,
        ptflops=mock_ptflops,
    ):
        result = cli_runner.invoke(app, ["ptflops", MODEL_PATTERN_ARG])
        assert result.exit_code == 0
    assert "2.5 GMac" in result.output
    assert "3.0 M" in result.output


def test_ptflops_none_results_print_nothing(cli_runner: CliRunner) -> None:
    """'ptflops' should not error when flops/params are None."""
    mock_instantiate = MagicMock(return_value=MagicMock())
    mock_torch_trainer = MagicMock()
    mock_torch_trainer.get_torch_device.return_value = "cpu"
    mock_torch_trainer.create_torch_inputs.return_value = {}
    mock_torch_trainer.initial_model.return_value = ({}, None)
    mock_ptflops = MagicMock()
    mock_ptflops.get_model_complexity_info.return_value = (None, None)
    with patch_cmd_globals(
        _instantiate=mock_instantiate,
        torch=_make_torch_mock(),
        torch_trainer=mock_torch_trainer,
        ptflops=mock_ptflops,
    ):
        assert cli_runner.invoke(app, ["ptflops", MODEL_PATTERN_ARG]).exit_code == 0


# ---------------------------------------------------------------------------
# 'calflops' command execution
# ---------------------------------------------------------------------------


def test_calflops_calls_instantiate(cli_runner: CliRunner) -> None:
    """'calflops' should call _instantiate to build the model instance."""
    mock_instantiate = MagicMock(return_value=MagicMock())
    mock_torch_trainer = MagicMock()
    mock_torch_trainer.get_torch_device.return_value = "cpu"
    mock_torch_trainer.create_torch_inputs.return_value = {}
    mock_torch_trainer.initial_model.return_value = ({}, None)
    mock_calflops = MagicMock()
    mock_calflops.calculate_flops.return_value = ("1.0 GFLOPs", "500 MMac", "1.0 M")
    with patch_cmd_globals(
        _instantiate=mock_instantiate,
        torch=_make_torch_mock(),
        torch_trainer=mock_torch_trainer,
        calflops=mock_calflops,
    ):
        assert cli_runner.invoke(app, ["calflops", MODEL_PATTERN_ARG]).exit_code == 0
    mock_instantiate.assert_called_once()


def test_calflops_prints_flops_macs_params(cli_runner: CliRunner) -> None:
    """'calflops' should print FLOPs, MACs, and parameter counts."""
    mock_instantiate = MagicMock(return_value=MagicMock())
    mock_torch_trainer = MagicMock()
    mock_torch_trainer.get_torch_device.return_value = "cpu"
    mock_torch_trainer.create_torch_inputs.return_value = {}
    mock_torch_trainer.initial_model.return_value = ({}, None)
    mock_calflops = MagicMock()
    mock_calflops.calculate_flops.return_value = ("4.2 GFLOPs", "2.1 GMac", "5.0 M")
    with patch_cmd_globals(
        _instantiate=mock_instantiate,
        torch=_make_torch_mock(),
        torch_trainer=mock_torch_trainer,
        calflops=mock_calflops,
    ):
        result = cli_runner.invoke(app, ["calflops", MODEL_PATTERN_ARG])
        assert result.exit_code == 0
    assert "4.2 GFLOPs" in result.output
    assert "2.1 GMac" in result.output
    assert "5.0 M" in result.output


# ---------------------------------------------------------------------------
# _instantiate – direct unit test (covers cmd_torch.py line 119)
# ---------------------------------------------------------------------------


def test_instantiate_builds_object_from_pattern() -> None:
    """_instantiate() resolves an ObjectPattern and returns the built instance."""
    configure_security(allowed_modules_check=False)
    # LazySelectedImporter only exposes __all__; access _instantiate via globals.
    _instantiate = app.registered_commands[0].callback.__globals__["_instantiate"]

    raw = {
        "_obj_": [
            ["_addr_", "torch.nn.Identity"],
            ["_call_", {}],
        ]
    }
    result = _instantiate(raw)
    import torch  # noqa: PLC0415

    assert isinstance(result, torch.nn.Identity)


# ---------------------------------------------------------------------------
# `train` command execution
# ---------------------------------------------------------------------------


def _train_callback() -> Any:
    """Return the callback function for the `train` command."""
    for command in app.registered_commands:
        callback_name = "" if command.callback is None else command.callback.__name__
        if command.name == "train" or callback_name == "train":
            return command.callback
    raise AssertionError("train command not found")


class _FakeParameter:
    def __init__(self, size: int, requires_grad: bool = True) -> None:
        self._size = size
        self.requires_grad = requires_grad

    def numel(self) -> int:
        return self._size


class _FakeModule:
    def __init__(self, *, outputs: list[str] | None = None, param_size: int = 3) -> None:
        if outputs is not None:
            self.outputs = outputs
        self._param_size = param_size

    def parameters(self) -> list[_FakeParameter]:
        return [_FakeParameter(self._param_size), _FakeParameter(99, requires_grad=False)]

    def state_dict(self) -> dict[str, int]:
        return {"param_size": self._param_size}


class _FakeBestCriterion:
    def __init__(self, target: str, mode: str, on_best: list[Any] | None = None) -> None:
        self.target = target
        self.mode = mode
        self.on_best: NamedCallbackList = NamedCallbackList()
        if on_best:
            for cb in on_best:
                self.on_best.append(cb)
        self._step = 0
        self._value = 0.5

    @property
    def step(self) -> int:
        return self._step

    @property
    def value(self) -> float:
        return self._value

    def __call__(self, info: Any, **kwargs: _FakeModule) -> None:
        self._step = info.step
        for callback in self.on_best:
            callback(info, self, **kwargs)


class _FakeInfo:
    def __init__(self, trainer: Any, logs: dict[str, float]) -> None:
        self.epoch = 1
        self.step = 2
        self.update = 3
        self.backward = trainer.backward
        self.inference_wrapper = trainer.inference_wrapper
        self._logs = logs

    def logs(self) -> dict[str, float]:
        return dict(self._logs)


class _FakeTrainer:
    training_prefix = "train/"
    validation_prefix = "valid/"

    def __init__(self, **kwargs: Any) -> None:
        self.backward = kwargs["backward"]
        self.inference_wrapper = kwargs["inference_wrapper"]
        self.on_epoch_end: NamedCallbackList = NamedCallbackList()
        self.on_training_begin: NamedCallbackList = NamedCallbackList()
        self.on_training_step_end: NamedCallbackList = NamedCallbackList()
        self.on_training_end: NamedCallbackList = NamedCallbackList()
        self.on_validation_begin: NamedCallbackList = NamedCallbackList()
        self.on_validation_step_end: NamedCallbackList = NamedCallbackList()
        self.on_validation_end: NamedCallbackList = NamedCallbackList()

    def describe(self) -> dict[str, list[str]]:
        """Return registered callback names for display."""
        attrs = (
            "on_epoch_end",
            "on_training_begin",
            "on_training_step_end",
            "on_training_end",
            "on_validation_begin",
            "on_validation_step_end",
            "on_validation_end",
        )
        return {a: getattr(self, a).names() for a in attrs if getattr(self, a)}

    def fit(self, **kwargs: Any) -> None:
        modules = {name: module for name, module in kwargs.items() if hasattr(module, "state_dict")}
        info = _FakeInfo(
            trainer=self,
            logs={
                "train/loss": 0.11,
                "valid/loss": 0.10,
                "valid/acc": 0.90,
            },
        )
        for callback in self.on_training_begin:
            callback(info, **modules)
        for callback in self.on_training_step_end:
            callback(info, **modules)
        for callback in self.on_training_end:
            callback(info, **modules)
        for callback in self.on_validation_begin:
            callback(info, **modules)
        for callback in self.on_validation_step_end:
            callback(info, **modules)
        for callback in self.on_validation_end:
            callback(info, **modules)
        for callback in self.on_epoch_end:
            callback(info, **modules)


def _build_train_args(tmp_path: Any, **overrides: Any) -> dict[str, Any]:
    artifact = tmp_path / "artifact.bin"
    artifact.write_text("dummy")
    args = {
        "model_patterns": [{"model": "MODEL_PATTERN"}],
        "shapes": [{"image": (3, 32, 32)}],
        "device": None,
        "ema": None,
        "ema_device": None,
        "loss_pattern": "LOSS_PATTERN",
        "loss_outputs": None,
        "metric_pattern": "METRIC_PATTERN",
        "metric_outputs": None,
        "backward_pattern": "BACKWARD_PATTERN",
        "mixed_precision_type": "float16",
        "compile_pattern": {"fullgraph": True},
        "epochs": 2,
        "start_epoch": 1,
        "training_dataset_pattern": "TRAIN_DATASET_PATTERN",
        "validation_dataset_pattern": "VALIDATION_DATASET_PATTERN",
        "validation_frequency": 1,
        "lower_criteria": ["valid/loss"],
        "higher_criteria": ["valid/acc"],
        "save_criteria": ["valid/acc"],
        "seed": 123,
        "matmul_precision": "high",
        "experiment": "unit-test-exp",
        "log_arguments": [{"run": "test"}],
        "log_artifacts": [artifact],
        "ci": False,
    }
    args.update(overrides)
    return args


def _build_train_deps(
    *,
    loss_module: _FakeModule,
    metric_module: _FakeModule | None,
    backward: Any,
    use_ema: bool,
) -> tuple[dict[str, Any], MagicMock, Any, Any, Any]:
    dataset_train = object()
    dataset_valid = object()

    def _instantiate_side_effect(raw: Any) -> Any:
        if raw == "MODEL_PATTERN":
            return _FakeModule(param_size=7)
        if raw == "LOSS_PATTERN":
            return loss_module
        if raw == "METRIC_PATTERN":
            return metric_module
        if raw == "BACKWARD_PATTERN":
            return lambda **_: backward
        if raw == "TRAIN_DATASET_PATTERN":
            return dataset_train
        if raw == "VALIDATION_DATASET_PATTERN":
            return dataset_valid
        return raw

    mlflow_mock = MagicMock()
    mlflow_mock.pytorch = MagicMock()
    mlflow_mock.start_run.return_value.__enter__ = MagicMock(return_value=None)
    mlflow_mock.start_run.return_value.__exit__ = MagicMock(return_value=False)

    pbar = MagicMock()
    tqdm_mock = MagicMock()
    tqdm_mock.tqdm.return_value = pbar

    torch_mock = _make_torch_mock()
    torch_mock.backends = SimpleNamespace(cudnn=SimpleNamespace(benchmark=False))
    torch_mock.set_float32_matmul_precision = MagicMock()
    torch_mock.manual_seed = MagicMock()
    torch_mock.compile = MagicMock(side_effect=lambda module, **_: module)
    torch_mock.version = SimpleNamespace(cuda="12.4")
    torch_mock.__version__ = "2.6.0"
    # _unwrap_ddp uses isinstance(..., torch.nn.parallel.DistributedDataParallel)
    # so we need a real type, not a MagicMock
    torch_mock.nn.parallel.DistributedDataParallel = type("DistributedDataParallel", (), {})

    np_mock = MagicMock()
    np_mock.random = MagicMock()

    trainer_ns = MagicMock()
    trainer_ns.get_torch_device.return_value = "cpu"
    trainer_ns.initial_distributed_env.return_value = ("cpu", 0, 0, 1, False)
    trainer_ns.initial_model.side_effect = lambda models, _shapes: ({}, None)
    trainer_ns.TorchTracker.from_criteria.return_value = "tracker"
    trainer_ns.get_autocast.return_value = "autocast"
    trainer_ns.TrainingStep.side_effect = SimpleNamespace
    trainer_ns.ValidationStep.side_effect = SimpleNamespace
    trainer_ns.TorchTrainer.side_effect = _FakeTrainer
    trainer_ns.TorchBestCriterion.side_effect = lambda target, mode: _FakeBestCriterion(
        target=target, mode=mode, on_best=[]
    )
    trainer_ns.TimmEmaWrapper.from_models.side_effect = lambda models, **_: (
        SimpleNamespace(models=models) if use_ema else None
    )

    deps = {
        "_instantiate": MagicMock(side_effect=_instantiate_side_effect),
        "configure_security": MagicMock(),
        "get_dataset_size": MagicMock(side_effect=lambda ds: 8 if ds is dataset_train else 3),
        "instantiator": MagicMock(instantiate=MagicMock(side_effect=lambda raw: {} if raw is None else raw)),
        "mlflow": mlflow_mock,
        "np": np_mock,
        "timm": SimpleNamespace(__version__="1.0.0"),
        "torch": torch_mock,
        "torch_trainer": trainer_ns,
        "tqdm": tqdm_mock,
    }
    return deps, mlflow_mock, pbar, trainer_ns, torch_mock


def test_train_raises_for_empty_model_patterns(tmp_path: Any) -> None:
    """`train` should fail fast when no model patterns are provided."""
    train_fn = _train_callback()
    args = _build_train_args(tmp_path, model_patterns=[])
    with pytest.raises(ValueError, match="At least one model pattern"):
        train_fn(**args)


def test_train_raises_for_invalid_model_pattern_shape(tmp_path: Any) -> None:
    """`train` should reject model-pattern entries containing multiple models."""
    train_fn = _train_callback()
    args = _build_train_args(tmp_path, model_patterns=[{"a": "MODEL_PATTERN", "b": "MODEL_PATTERN"}])

    backward = SimpleNamespace(
        learning_rates={"lr": 0.1},
        optimizers={"opt": _FakeModule()},
        grad_scalers={"scaler": _FakeModule()},
    )
    deps, *_ = _build_train_deps(
        loss_module=_FakeModule(outputs=["loss"]), metric_module=None, backward=backward, use_ema=False
    )

    with patch_cmd_globals(**deps), pytest.raises(ValueError, match="exactly one model definition"):
        train_fn(**args)


def test_train_runs_non_ci_flow_and_best_criteria_logging(tmp_path: Any) -> None:
    """Non-CI mode should wire progress callbacks and log best/training states."""
    train_fn = _train_callback()
    args = _build_train_args(tmp_path, ci=False, ema=None, loss_outputs=None, metric_outputs=None)

    backward = SimpleNamespace(
        learning_rates={"lr": 0.01},
        optimizers={"opt": _FakeModule()},
        grad_scalers={"scaler": _FakeModule()},
        param_group_names={"group0": ["model.weight"]},
    )
    deps, mlflow_mock, pbar, trainer_ns, torch_mock = _build_train_deps(
        loss_module=_FakeModule(outputs=["loss"]),
        metric_module=_FakeModule(outputs=["acc"]),
        backward=backward,
        use_ema=False,
    )

    with patch_cmd_globals(**deps):
        train_fn(**args)

    assert pbar.reset.called
    assert pbar.set_postfix.called
    assert pbar.write.called
    assert trainer_ns.get_autocast.call_count == 1
    assert torch_mock.compile.call_count >= 2
    assert mlflow_mock.log_metrics.called
    assert mlflow_mock.log_metric.called
    assert mlflow_mock.log_artifact.called
    assert mlflow_mock.log_dict.call_args_list[0].args[1] == "param_groups.yaml"

    state_dict_calls = mlflow_mock.pytorch.log_state_dict.call_args_list
    artifact_paths = [call.kwargs.get("artifact_path") for call in state_dict_calls if "artifact_path" in call.kwargs]
    assert "training_state" in artifact_paths
    assert any(call.args[1] == "best_valid/acc" for call in state_dict_calls if len(call.args) >= 2)


def test_train_runs_ci_with_ema_and_default_outputs(tmp_path: Any) -> None:
    """CI mode should print criteria and include EMA state when EMA is enabled."""
    train_fn = _train_callback()
    args = _build_train_args(
        tmp_path,
        ci=True,
        ema={},
        loss_outputs=["loss"],
        metric_outputs=["acc"],
        mixed_precision_type="float16",
    )

    backward = SimpleNamespace(
        mixed_precision_type="bfloat16",
        learning_rates={"lr": 0.02},
        optimizers={"opt": _FakeModule()},
        grad_scalers={"scaler": _FakeModule()},
    )
    deps, mlflow_mock, _pbar, trainer_ns, _torch_mock = _build_train_deps(
        loss_module=_FakeModule(outputs=None),
        metric_module=_FakeModule(outputs=None),
        backward=backward,
        use_ema=True,
    )

    with patch_cmd_globals(**deps):
        train_fn(**args)

    trainer_ns.get_autocast.assert_called_once_with("bfloat16", "cpu")
    assert mlflow_mock.log_metrics.called
    state_calls = mlflow_mock.pytorch.log_state_dict.call_args_list
    training_state = [call for call in state_calls if call.kwargs.get("artifact_path") == "training_state"]
    assert training_state
    assert "ema" in training_state[0].args[0]


def test_train_raises_when_module_outputs_missing_and_not_provided(tmp_path: Any) -> None:
    """`train` should fail when a module lacks `outputs` and no fallback outputs are given."""
    train_fn = _train_callback()
    args = _build_train_args(tmp_path, loss_outputs=None)

    backward = SimpleNamespace(
        learning_rates={"lr": 0.01},
        optimizers={"opt": _FakeModule()},
        grad_scalers={"scaler": _FakeModule()},
    )
    deps, *_ = _build_train_deps(
        loss_module=_FakeModule(outputs=None),
        metric_module=_FakeModule(outputs=["acc"]),
        backward=backward,
        use_ema=False,
    )

    with patch_cmd_globals(**deps), pytest.raises(ValueError, match='Module "loss" does not have an "outputs"'):
        train_fn(**args)
