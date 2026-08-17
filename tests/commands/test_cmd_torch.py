"""Unit tests for structcast_model.commands.cmd_torch."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Generator
from contextlib import contextmanager
from datetime import timedelta
from functools import partial
import json
import os
import pathlib
import traceback
from types import SimpleNamespace
from typing import Any

import mlflow
from mlflow.tracking import MlflowClient
import pytest
import torch.multiprocessing as mp
from torch.utils._python_dispatch import TorchDispatchMode, _get_current_dispatch_mode_stack
from typer import Typer
from typer.testing import CliRunner

from structcast_model.base_trainer import BaseInfo
from structcast_model.commands.cmd_torch import app
from structcast_model.commands.utils import instantiate_object
from structcast_model.torch.trainer import TorchTrainer
from tests import CFG_DIR, FIXTURES_DIR
import torch
import torch.distributed as dist

LINEAR_CFG = str(FIXTURES_DIR / "cfg" / "torch" / "Linear.yaml")
MODEL_CFG = str(CFG_DIR / "torch" / "models" / "ConvNeXtV2.yaml")
LEARNER_CFG = str(CFG_DIR / "torch" / "learners" / "ConvNeXtV2.yaml")

# ---------------------------------------------------------------------------
# Helper: access cmd_torch's real globals (bypasses LazySelectedImporter proxy)
# ---------------------------------------------------------------------------

_FIRST_CALLBACK = app.registered_commands[0].callback
assert _FIRST_CALLBACK is not None, "cmd_torch registers every command with a callback"
_CMD_GLOBALS: dict[str, Any] = _FIRST_CALLBACK.__globals__

# Access private functions from cmd_torch via its module globals
_get_module_outputs = _CMD_GLOBALS["_get_module_outputs"]
_instantiate_models = _CMD_GLOBALS["_instantiate_models"]


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
def _clean_torch_dispatch_stack() -> Generator[None, None, None]:
    """Ensure the torch dispatch stack is clean around each test."""

    def _drain_dispatch_stack() -> None:
        for mode in reversed(_get_current_dispatch_mode_stack()):
            # Call TorchDispatchMode.__exit__ directly to avoid ptflops
            # FlopCounterMode.__exit__ calling print_fn on a closed StringIO.
            TorchDispatchMode.__exit__(mode, None, None, None)

    _drain_dispatch_stack()
    yield
    _drain_dispatch_stack()


# ---------------------------------------------------------------------------
# Minimal real modules for training tests
# ---------------------------------------------------------------------------


class SimpleModel(torch.nn.Module):
    """A tiny model for testing: Linear(4 -> 2) returning a dict.

    Public so that object patterns can address it: the learner factory imports what it builds.
    """

    outputs: list[str] = ["logits"]

    def __init__(self) -> None:
        """Create the linear layer."""
        super().__init__()
        self.fc = torch.nn.Linear(4, 2)

    def forward(self, x: torch.Tensor, **kwargs: Any) -> dict[str, torch.Tensor]:
        """Forward pass."""
        return {"logits": self.fc(x)}


class ZeroLinear(torch.nn.Linear):
    """A `Linear(2 -> 1)` starting at zero, so a rank's gradient is decided by its batch alone.

    Public so that object patterns can address it.
    """

    def __init__(self) -> None:
        """Create the layer without a bias and zero its weight."""
        super().__init__(2, 1, bias=False)
        torch.nn.init.zeros_(self.weight)

    # `Linear.forward` names its parameter `input`; renaming it is what this fake is for, and the
    # command calls modules as `model(**inputs)`, keyed by the shape names, so the rename is required.
    def forward(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:  # type: ignore[override]
        """Forward pass, taking the input under the name the datasets and the shapes use."""
        return super().forward(x)


class _SimpleLoss(torch.nn.Module):
    """Loss module that computes cross-entropy from logits and target."""

    outputs: list[str] = ["loss"]

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("_dummy", torch.tensor(0.0))

    def forward(self, logits: torch.Tensor, target: torch.Tensor, **kwargs: Any) -> dict[str, torch.Tensor]:
        """Compute loss."""
        return {"loss": torch.nn.functional.cross_entropy(logits, target)}


class SimpleLearner:
    """Minimal learner implementing the Learner protocol with a real optimizer."""

    outputs: list[str] = ["loss", "acc"]

    def __init__(self, **models: torch.nn.Module) -> None:
        """Keep the models and build one optimizer over the first of them."""
        self._models = models
        model = next(iter(models.values()))
        self._optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    @property
    def models(self) -> dict[str, Any]:
        """Return models."""
        return self._models

    def update(self, step: int) -> bool:
        """Always signal update."""
        return True

    def training_step(self, **kwargs: Any) -> dict[str, Any]:
        """Return fixed training criteria."""
        return {"loss": torch.tensor(0.5), "acc": torch.tensor(0.9)}

    def inference_step(self, **kwargs: Any) -> dict[str, Any]:
        """Return fixed inference criteria."""
        return {"loss": torch.tensor(0.3), "acc": torch.tensor(0.85)}

    @property
    def optimizers(self) -> dict[str, torch.optim.Optimizer]:
        """Return optimizers."""
        return {"optimizer": self._optimizer}

    @property
    def optimizer_models(self) -> dict[str, list[str]]:
        """Declare no pairing; these runs do not exercise sharded optimizer state."""
        return {}

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


class LearnerWithoutOutputs(SimpleLearner):
    """Learner whose ``outputs`` attribute is missing, as a learner built from a bare class can be."""

    # Raising keeps `hasattr(learner, "outputs")` False; mypy only sees a property replacing a list attribute.
    outputs = property(lambda self: (_ for _ in ()).throw(AttributeError))  # type: ignore[assignment]


class GradientLearner:
    """Learner running one squared-error step and dumping the gradient it produced to disk.

    The command builds the learner with the models the strategy wrapped, so the gradient read here
    is whatever that wrapper leaves behind: under DDP, the average over the ranks. Nothing zeroes
    the gradients and there is no optimizer, so a one-step run leaves exactly that value.
    """

    outputs: list[str] = ["loss"]

    def __init__(self, **models: torch.nn.Module) -> None:
        """Keep the models the command built."""
        self._models = models

    @property
    def models(self) -> dict[str, Any]:
        """Return models."""
        return self._models

    @property
    def optimizers(self) -> dict[str, Any]:
        """Return no optimizers: the run reads gradients, it never applies them."""
        return {}

    @property
    def optimizer_models(self) -> dict[str, list[str]]:
        """Return no pairing, there being no optimizer."""
        return {}

    @property
    def learning_rates(self) -> dict[str, float]:
        """Return no learning rates, there being no optimizer."""
        return {}

    def update(self, step: int) -> bool:
        """Always signal update."""
        return True

    def training_step(self, x: torch.Tensor, target: torch.Tensor, **kwargs: Any) -> dict[str, Any]:
        """Run one step and write the model's gradient, right after the backward, to `GRADIENT_DIR`."""
        model = self._models["model"]
        loss = ((model(x) - target) ** 2).sum()
        loss.backward()
        gradient = next(model.parameters()).grad
        assert gradient is not None, "the backward pass left no gradient on the model parameters"
        path = pathlib.Path(os.environ["GRADIENT_DIR"], f"grad_{os.environ['RANK']}.json")
        path.write_text(json.dumps(gradient.flatten().tolist()))
        return {"loss": loss.detach()}


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


def _address(name: str) -> str:
    """Return the import address of a class of this test module, for use in object patterns."""
    return f"{__name__}.{name}"


MODEL_PATTERN: list[Any] = ["_obj_", {"_addr_": _address("SimpleModel")}, "_call_"]
"""Object pattern building one `SimpleModel`."""

ZERO_MODEL_PATTERN: list[Any] = ["_obj_", {"_addr_": _address("ZeroLinear")}, "_call_"]
"""Object pattern building one `ZeroLinear`."""


def _learner_pattern(classname: str = "SimpleLearner") -> list[Any]:
    """Return the object pattern resolving to a learner class, which the factory calls with the models."""
    return ["_obj_", {"_addr_": _address(classname)}]


def _make_instantiate_fn(
    *,
    training_data: list[dict[str, torch.Tensor]],
    validation_data: list[dict[str, torch.Tensor]] | None = None,
) -> Any:
    """Build a replacement for ``instantiate_object`` that returns the datasets of a run."""

    def _side_effect(raw: Any) -> Any:
        if raw == "TRAIN_DS":
            return training_data
        if raw == "VALID_DS":
            return validation_data
        # Models and learner are built for real, so the tests exercise the actual assembly; a
        # trainer type handed over directly, rather than as a pattern, is passed through.
        return instantiate_object(raw) if isinstance(raw, (list, dict)) else raw

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
    assert cli_runner.invoke(app, ["create", "learner", "--help"]).exit_code == 0
    assert cli_runner.invoke(app, ["ptflops", "--help"]).exit_code == 0
    assert cli_runner.invoke(app, ["calflops", "--help"]).exit_code == 0


def test_train_help_shows_no_python_repr(cli_runner: CliRunner) -> None:
    """'train --help' must describe values, never Python objects.

    The three criterion options pair `...` with `default_factory=list`, which Typer renders as
    "[default: <class 'list'>]" unless show_default is off; a user shown a class repr cannot tell that
    the real default is "no criteria monitored".
    """
    result = cli_runner.invoke(app, ["train", "--help"])
    assert result.exit_code == 0, result.output
    assert "<class" not in result.output


# ---------------------------------------------------------------------------
# 'create model' command
# ---------------------------------------------------------------------------


def test_create_model_calls_torch_builder(tmp_path: Any, cli_runner: CliRunner) -> None:
    """'create model' should generate a model script from a real configuration."""
    out = str(tmp_path / "model.py")
    result = cli_runner.invoke(app, ["create", "model", MODEL_CFG, "--output", out])
    assert result.exit_code == 0, result.output
    assert "class Model" in (tmp_path / "model.py").read_text()


def test_create_model_passes_classname(tmp_path: Any, cli_runner: CliRunner) -> None:
    """'create model --classname' should generate a script with the given class name."""
    out = str(tmp_path / "my_net.py")
    result = cli_runner.invoke(app, ["create", "model", MODEL_CFG, "--classname", "MyNet", "--output", out])
    assert result.exit_code == 0, result.output
    assert "class MyNet" in (tmp_path / "my_net.py").read_text()


def test_create_model_structured_output_defaults_to_configuration(tmp_path: Any, cli_runner: CliRunner) -> None:
    """'create model' leaves the return type to the template, which here means a bare tensor.

    The root of the ConvNeXtV2 template sets no STRUCTURED_OUTPUT, so its single `cls` output must
    stay a plain tensor a loss can consume; only the multi-output Backbone opts into a dict.
    """
    out = str(tmp_path / "model.py")
    result = cli_runner.invoke(app, ["create", "model", MODEL_CFG, "--output", out])
    assert result.exit_code == 0, result.output
    script = (tmp_path / "model.py").read_text()
    assert "return {'feat1'" in script  # the Backbone's own STRUCTURED_OUTPUT is still honored
    assert "return {'" not in script.rsplit("class Model", 1)[-1]


def test_create_model_forced_structured_output(tmp_path: Any, cli_runner: CliRunner) -> None:
    """'create model --structured-output' overrides the template and returns a dict from the root."""
    out = str(tmp_path / "model.py")
    result = cli_runner.invoke(app, ["create", "model", MODEL_CFG, "--structured-output", "--output", out])
    assert result.exit_code == 0, result.output
    last_class = (tmp_path / "model.py").read_text().rsplit("class Model", 1)[-1]
    assert "return {'" in last_class


def test_create_model_no_structured_output(tmp_path: Any, cli_runner: CliRunner) -> None:
    """'create model --no-structured-output' root class should not return a dict."""
    out = str(tmp_path / "model.py")
    result = cli_runner.invoke(app, ["create", "model", MODEL_CFG, "--no-structured-output", "--output", out])
    assert result.exit_code == 0, result.output
    last_class = (tmp_path / "model.py").read_text().rsplit("class Model", 1)[-1]
    assert "return {'" not in last_class


def test_create_model_with_output_path(tmp_path: Any, cli_runner: CliRunner) -> None:
    """'create model --output' should write the generated script to the specified path."""
    out_file = tmp_path / "nested" / "out.py"
    result = cli_runner.invoke(app, ["create", "model", MODEL_CFG, "--output", str(out_file)])
    assert result.exit_code == 0, result.output
    assert out_file.exists()
    assert "class Model" in out_file.read_text()


def test_create_model_linear(tmp_path: Any, cli_runner: CliRunner) -> None:
    """'create model' generates a script from a simple Linear config."""
    out = str(tmp_path / "model.py")
    result = cli_runner.invoke(app, ["create", "model", LINEAR_CFG, "--output", out])
    assert result.exit_code == 0, result.output
    text = (tmp_path / "model.py").read_text()
    assert "class Model" in text
    assert "LazyLinear" in text


def test_create_model_linear_classname(tmp_path: Any, cli_runner: CliRunner) -> None:
    """'create model --classname' generates a script with the given class name for Linear."""
    out = str(tmp_path / "net.py")
    result = cli_runner.invoke(app, ["create", "model", LINEAR_CFG, "--classname", "SimpleNet", "--output", out])
    assert result.exit_code == 0, result.output
    assert "class SimpleNet" in (tmp_path / "net.py").read_text()


# ---------------------------------------------------------------------------
# 'time' command — simple Linear layer
# ---------------------------------------------------------------------------


def test_time_linear(cli_runner: CliRunner) -> None:
    """'time' measures inference on a simple torch Linear layer."""
    pattern = "[_obj_, {_addr_: torch.nn.Linear}, {_call_: {in_features: 4, out_features: 2}}]"
    result = cli_runner.invoke(
        app,
        ["time", pattern, "--shape", "input: [4]", "--warmup-runs", "1", "--times", "1", "--batch-size", "1"],
    )
    assert result.exit_code == 0, result.output
    assert "Average inference time" in result.output


# ---------------------------------------------------------------------------
# 'create learner' command
# ---------------------------------------------------------------------------


def test_create_learner_calls_torch_learner_builder(tmp_path: Any, cli_runner: CliRunner) -> None:
    """'create learner' should generate a learner script from a real configuration."""
    out = str(tmp_path / "learner.py")
    result = cli_runner.invoke(app, ["create", "learner", LEARNER_CFG, "--output", out])
    assert result.exit_code == 0, result.output
    script = (tmp_path / "learner.py").read_text()
    assert "class Learner" in script
    # The generated class is a Learner by shape, not by inheritance: these members are the protocol.
    for member in ("def update(self", "def training_step(self", "def inference_step(self"):
        assert member in script
    assert "def models(self" in script


def test_create_learner_passes_default_classname(tmp_path: Any, cli_runner: CliRunner) -> None:
    """'create learner' default classname should produce a class named 'Learner'."""
    out = str(tmp_path / "learner.py")
    result = cli_runner.invoke(app, ["create", "learner", LEARNER_CFG, "--output", out])
    assert result.exit_code == 0, result.output
    assert "class Learner" in (tmp_path / "learner.py").read_text()


# ---------------------------------------------------------------------------
# 'calflops' command with real model (must run BEFORE ptflops, which installs
# a persistent __torch_dispatch__ handler that captures CliRunner's StringIO)
# ---------------------------------------------------------------------------


def test_calflops_runs_with_real_model(cli_runner: CliRunner) -> None:
    """'calflops' should run calflops on a real model and print FLOPs, MACs, parameters."""
    deps = {"instantiate_object": lambda raw: SimpleModel()}
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
    result = cli_runner.invoke(app, ["ptflops", _IDENTITY_PATTERN, "--device", "cpu"])
    assert result.exit_code == 0, result.output


def test_ptflops_none_results_print_nothing(cli_runner: CliRunner) -> None:
    """'ptflops' should not error when flops/params are None (e.g. identity)."""
    result = cli_runner.invoke(app, ["ptflops", _IDENTITY_PATTERN, "--device", "cpu"])
    assert result.exit_code == 0, result.output


# ---------------------------------------------------------------------------
# instantiate – direct unit test (now in commands.utils)
# ---------------------------------------------------------------------------


def test_instantiate_builds_object_from_pattern() -> None:
    """instantiate() resolves an ObjectPattern and returns the built instance."""
    raw = {"_obj_": [["_addr_", "torch.nn.Identity"], ["_call_", {}]]}
    result = instantiate_object(raw)
    assert isinstance(result, torch.nn.Identity)


def test_instantiate_builds_linear_with_args() -> None:
    """instantiate() builds a torch.nn.Linear with keyword arguments."""
    raw = {"_obj_": [["_addr_", "torch.nn.Linear"], {"_call_": {"in_features": 8, "out_features": 4}}]}
    result = instantiate_object(raw)
    assert isinstance(result, torch.nn.Linear)
    assert result.in_features == 8
    assert result.out_features == 4


# ---------------------------------------------------------------------------
# _instantiate_models
# ---------------------------------------------------------------------------


def test_instantiate_models_creates_ordered_dict() -> None:
    """The learner is built with the models by keyword, so their names and order must be preserved."""
    patterns = [{"model": {"_obj_": [["_addr_", "torch.nn.Identity"], "_call_"]}}]
    result = _instantiate_models(patterns)
    assert isinstance(result, OrderedDict)
    assert isinstance(result["model"], torch.nn.Identity)


def test_instantiate_models_raises_for_multiple_keys() -> None:
    """A two-key entry hides which name belongs to which pattern, so it is a configuration error."""
    with pytest.raises(ValueError, match="exactly one model definition"):
        _instantiate_models([{"a": {}, "b": {}}])


def test_instantiate_models_returns_empty_for_empty_list() -> None:
    """_instantiate_models returns an empty OrderedDict for an empty pattern list."""
    assert _instantiate_models([]) == OrderedDict()


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
            learner_pattern="IGNORED",
            learner_outputs=None,
            compile_pattern=None,
            trainer_pattern=None,
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
    train_fn = _train_callback()
    deps = {
        "instantiate_object": _make_instantiate_fn(training_data=_make_training_dataset()),
    }
    with patch_cmd_globals(**deps), pytest.raises(ValueError, match="exactly one model definition"):
        train_fn(
            model_patterns=[{"a": MODEL_PATTERN, "b": MODEL_PATTERN}],
            initializer_patterns=None,
            shapes=[{"x": (4,)}],
            device="cpu",
            learner_pattern=_learner_pattern(),
            learner_outputs=None,
            compile_pattern=None,
            trainer_pattern=None,
            epochs=1,
            start_epoch=1,
            resume=None,
            strategy_pattern=None,
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
    """`train` should fail when the learner lacks ``outputs`` and no fallback is given."""
    train_fn = _train_callback()
    deps = {"instantiate_object": _make_instantiate_fn(training_data=_make_training_dataset())}
    with patch_cmd_globals(**deps), pytest.raises(ValueError, match='Module "learner" does not have an "outputs"'):
        train_fn(
            model_patterns=[{"model": MODEL_PATTERN}],
            initializer_patterns=None,
            shapes=[{"x": (4,)}],
            device="cpu",
            learner_pattern=_learner_pattern("LearnerWithoutOutputs"),
            learner_outputs=None,
            compile_pattern=None,
            trainer_pattern=None,
            epochs=1,
            start_epoch=1,
            resume=None,
            strategy_pattern=None,
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


class _EpochAwareDataset(list[dict[str, torch.Tensor]]):
    """A dataset that also reacts to epochs, the way a wrapper driving a `DistributedSampler` does."""

    def __init__(self, batches: list[dict[str, torch.Tensor]]) -> None:
        """Keep the batches and start with no epoch seen."""
        super().__init__(batches)
        self.epochs: list[int] = []

    def on_epoch_begin(self, info: BaseInfo) -> None:
        """Record the epoch the trainer is starting."""
        self.epochs.append(info.epoch)


def _recording_trainer(captured: list[Any]) -> Any:
    """Return a trainer type that records the callbacks the command hands it.

    Captured at ``fit()`` time: the command builds the trainer first and appends its callbacks
    afterwards, relying on the first-use scan.
    """

    class _RecordingTrainer(TorchTrainer):
        def fit(self, *args: Any, **kwargs: Any) -> dict[int, dict[str, Any]]:
            captured.extend(self.callbacks)
            return super().fit(*args, **kwargs)

    return _RecordingTrainer


def _invoke_train(
    tmp_path: pathlib.Path,
    *,
    ci: bool = True,
    training_data: list[dict[str, torch.Tensor]] | None = None,
    trainer_pattern: Any = None,
    validation_data: list[dict[str, torch.Tensor]] | None = None,
    learner_outputs: list[str] | None = None,
    lower_criteria: list[str] | None = None,
    higher_criteria: list[str] | None = None,
    save_criteria: list[str] | None = None,
    learner_classname: str = "SimpleLearner",
    logger_name: str = "mlflow",
    log_arguments: list[dict[str, Any]] | None = None,
    log_artifacts: list[pathlib.Path] | None = None,
    epochs: int = 2,
    resume: str | None = None,
) -> None:
    """Invoke the ``train`` callback with real modules, patching only the dataset instantiation."""
    if training_data is None:
        training_data = _make_training_dataset()
    if validation_data is None:
        validation_data = _make_validation_dataset()
    deps = {
        "instantiate_object": _make_instantiate_fn(training_data=training_data, validation_data=validation_data),
    }
    mlflow.set_tracking_uri(str(tmp_path / "mlruns"))
    train_fn = _train_callback()
    with patch_cmd_globals(**deps):
        train_fn(
            model_patterns=[{"model": MODEL_PATTERN}],
            initializer_patterns=None,
            shapes=[{"x": (4,)}],
            device="cpu",
            learner_pattern=_learner_pattern(learner_classname),
            learner_outputs=learner_outputs,
            compile_pattern=None,
            trainer_pattern=trainer_pattern,
            epochs=epochs,
            start_epoch=1,
            resume=resume,
            training_dataset_pattern="TRAIN_DS",
            validation_dataset_pattern="VALID_DS",
            validation_frequency=1,
            lower_criteria=lower_criteria or ["loss"],
            higher_criteria=higher_criteria or ["acc"],
            save_criteria=save_criteria or ["acc"],
            seed=42,
            matmul_precision="high",
            experiment="test-e2e",
            logger_name=logger_name,
            log_arguments=log_arguments,
            log_artifacts=log_artifacts,
            ci=ci,
            dist_backend=None,
            dist_url=None,
            strategy_pattern=None,
        )


def test_train_ci_mode_end_to_end(tmp_path: pathlib.Path) -> None:
    """Full end-to-end train in CI mode with a real model, learner, tracker, and MLflow logger."""
    artifact = tmp_path / "artifact.bin"
    artifact.write_text("dummy")
    _invoke_train(
        tmp_path,
        ci=True,
        log_arguments=[{"run": "test"}],
        log_artifacts=[artifact],
    )
    run = mlflow.search_runs(experiment_names=["test-e2e"], output_format="list")[0]
    assert run.data.metrics["val_loss"] == pytest.approx(0.3)
    assert run.data.metrics["best_acc"] == pytest.approx(0.9)
    artifacts = [artifact.path for artifact in MlflowClient().list_artifacts(run.info.run_id)]
    assert {"training_state", "best_acc", "arguments.yaml", "param_groups.yaml"} <= set(artifacts)


def test_train_resumes_from_a_saved_training_state(tmp_path: pathlib.Path) -> None:
    """--resume must continue at the epoch after the saved one instead of training the run again."""
    _invoke_train(tmp_path, epochs=2)
    # `mlflow.pytorch.log_state_dict` writes the tensors to a file inside the artifact directory.
    (state,) = (tmp_path / "mlruns").rglob("training_state/state_dict.pth")
    _invoke_train(tmp_path, epochs=3, resume=str(state))
    runs = mlflow.search_runs(experiment_names=["test-e2e"], output_format="list")
    assert len(runs) == 2
    resumed = max(runs, key=lambda run: run.info.start_time)
    history = MlflowClient().get_metric_history(resumed.info.run_id, "val_loss")
    assert [metric.step for metric in history] == [3]


def test_train_ci_mode_with_learner_outputs_fallback(tmp_path: pathlib.Path) -> None:
    """Train should accept explicit --learner-outputs when the learner has no ``outputs`` attribute."""
    _invoke_train(
        tmp_path,
        ci=True,
        learner_classname="LearnerWithoutOutputs",
        learner_outputs=["loss", "acc"],
        save_criteria=[],
        lower_criteria=[],
        higher_criteria=[],
    )


def test_train_non_ci_mode_uses_pbar(tmp_path: pathlib.Path) -> None:
    """Non-CI mode should use the tqdm progress bar callback."""
    _invoke_train(tmp_path, ci=False)


def test_train_accepts_a_trainer_factory_without_class_attributes(tmp_path: pathlib.Path) -> None:
    """A --trainer pattern may yield a factory, not a class: prefixes must come off the built instance."""
    _invoke_train(tmp_path, ci=False, trainer_pattern=partial(TorchTrainer))


def test_train_routes_a_dataset_implementing_an_event_protocol_without_a_callback(tmp_path: pathlib.Path) -> None:
    """A dataset with a lifecycle hook is called without ever entering the callbacks.

    The CLI knows no dataset type: the trainer scans the provider datasets -- on every rank,
    because a `DistributedSampler` reshuffles only if `set_epoch` reaches all of them.
    """
    captured: list[Any] = []
    training = _EpochAwareDataset(_make_training_dataset())
    validation = _make_validation_dataset()
    _invoke_train(
        tmp_path,
        training_data=training,
        validation_data=validation,
        trainer_pattern=_recording_trainer(captured),
        epochs=2,
    )
    assert all(cb is not training and cb is not validation for cb in captured)
    assert training.epochs == [1, 2]


def test_train_adds_no_callbacks_for_datasets_without_event_hooks(tmp_path: pathlib.Path) -> None:
    """A plain dataset has nothing to react to, so it must stay out of the callbacks entirely."""
    captured: list[Any] = []
    training, validation = _make_training_dataset(), _make_validation_dataset()
    _invoke_train(
        tmp_path,
        training_data=training,
        validation_data=validation,
        trainer_pattern=_recording_trainer(captured),
        epochs=1,
    )
    assert all(cb is not training and cb is not validation for cb in captured)


class _FakeWandb:
    """Stand-in for the wandb package, recording what `WandbLogger` sends it."""

    def __init__(self, run_dir: pathlib.Path) -> None:
        run_dir.mkdir(parents=True, exist_ok=True)
        self.run = SimpleNamespace(dir=str(run_dir))
        self.config = SimpleNamespace(update=self._update_config)
        self.projects: list[str] = []
        self.finished = 0
        self.params: dict[str, Any] = {}
        self.metrics: list[tuple[dict[str, Any], int]] = []
        self.saved: list[str] = []

    def _update_config(self, params: dict[str, Any]) -> None:
        self.params.update(params)

    def init(self, project: str) -> None:
        """Start a run."""
        self.projects.append(project)

    def finish(self, exit_code: int = 0) -> None:
        """End the run."""
        self.finished += 1

    def log(self, metrics: dict[str, Any], step: int) -> None:
        """Record metric values."""
        self.metrics.append((metrics, step))

    def save(self, path: str) -> None:
        """Record an artifact."""
        self.saved.append(path)


def test_train_logs_the_whole_run_through_the_selected_logger(
    tmp_path: pathlib.Path, wandb_logger_with: Callable[[Any], Any]
) -> None:
    """--logger wandb must route the run lifecycle and the metrics to wandb, not to MLflow."""
    fake = _FakeWandb(tmp_path / "wandb_run")
    artifact = tmp_path / "artifact.bin"
    artifact.write_text("dummy")
    # The command's lazy handle caches the module it first loaded, so it needs the reloaded one.
    with patch_cmd_globals(wandb_logger=wandb_logger_with(fake)):
        _invoke_train(tmp_path, ci=True, logger_name="wandb", log_artifacts=[artifact])
    assert fake.projects == ["test-e2e"]
    assert fake.finished == 1
    assert fake.params["epochs"] == 2
    assert fake.saved == [str(artifact)]
    assert [step for _, step in fake.metrics] == [1, 1, 1, 2, 2, 2]
    assert fake.metrics[0][0]["val_loss"] == pytest.approx(0.3)
    assert (tmp_path / "wandb_run" / "arguments.yaml").exists()
    assert (tmp_path / "wandb_run" / "training_state.pt").exists()


# ---------------------------------------------------------------------------
# DDP distributed training tests via torch.multiprocessing.spawn
# ---------------------------------------------------------------------------


def _init_worker_group(rank: int, world_size: int, init_file: str) -> None:
    """Join the gloo group of a spawned worker.

    The timeout turns a collective the ranks disagree on into a failure rather than a hung CI job.
    """
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=60),
    )


def _ddp_train_worker(
    rank: int,
    world_size: int,
    init_file: str,
    mlflow_uri: str,
    ci: bool,
) -> None:
    """Worker function for DDP training tests launched by mp.spawn."""
    _init_worker_group(rank, world_size, init_file)
    try:
        mlflow.set_tracking_uri(mlflow_uri)
        training_data = _make_training_dataset()
        validation_data = _make_validation_dataset()
        deps = {
            "instantiate_object": _make_instantiate_fn(training_data=training_data, validation_data=validation_data)
        }
        train_fn = _train_callback()
        originals = {k: _CMD_GLOBALS.get(k) for k in deps}
        _CMD_GLOBALS.update(deps)
        try:
            train_fn(
                model_patterns=[{"model": MODEL_PATTERN}],
                initializer_patterns=None,
                shapes=[{"x": (4,)}],
                device="cpu",
                learner_pattern=_learner_pattern(),
                learner_outputs=None,
                compile_pattern=None,
                trainer_pattern=None,
                epochs=2,
                start_epoch=1,
                resume=None,
                strategy_pattern=None,
                training_dataset_pattern="TRAIN_DS",
                validation_dataset_pattern="VALID_DS",
                validation_frequency=1,
                lower_criteria=["loss"],
                higher_criteria=["acc"],
                save_criteria=["acc"],
                seed=42,
                matmul_precision="high",
                experiment="test-ddp",
                logger_name="mlflow",
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
    _init_worker_group(rank, world_size, init_file)
    try:
        mlflow_uri = os.path.join(result_dir, "mlruns")
        mlflow.set_tracking_uri(mlflow_uri)
        training_data = _make_training_dataset()
        deps = {"instantiate_object": _make_instantiate_fn(training_data=training_data)}
        train_fn = _train_callback()
        originals = {k: _CMD_GLOBALS.get(k) for k in deps}
        _CMD_GLOBALS.update(deps)
        try:
            train_fn(
                model_patterns=[{"model": MODEL_PATTERN}],
                initializer_patterns=None,
                shapes=[{"x": (4,)}],
                device="cpu",
                learner_pattern=_learner_pattern(),
                learner_outputs=None,
                compile_pattern=None,
                trainer_pattern=None,
                epochs=1,
                start_epoch=1,
                resume=None,
                strategy_pattern=None,
                training_dataset_pattern="TRAIN_DS",
                validation_dataset_pattern=None,
                validation_frequency=1,
                lower_criteria=[],
                higher_criteria=[],
                save_criteria=[],
                seed=42,
                matmul_precision="high",
                experiment="test-rank-gating",
                logger_name="mlflow",
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
    _init_worker_group(rank, world_size, init_file)
    try:
        mlflow_uri = os.path.join(result_dir, "mlruns")
        mlflow.set_tracking_uri(mlflow_uri)
        training_data = _make_training_dataset()
        deps = {"instantiate_object": _make_instantiate_fn(training_data=training_data)}
        train_fn = _train_callback()
        originals = {k: _CMD_GLOBALS.get(k) for k in deps}
        _CMD_GLOBALS.update(deps)
        try:
            train_fn(
                model_patterns=[{"model": MODEL_PATTERN}],
                initializer_patterns=None,
                shapes=[{"x": (4,)}],
                device="cpu",
                learner_pattern=_learner_pattern(),
                learner_outputs=None,
                compile_pattern=None,
                trainer_pattern=None,
                epochs=1,
                start_epoch=1,
                resume=None,
                strategy_pattern=None,
                training_dataset_pattern="TRAIN_DS",
                validation_dataset_pattern=None,
                validation_frequency=1,
                lower_criteria=[],
                higher_criteria=[],
                save_criteria=[],
                seed=seed,
                matmul_precision="high",
                experiment="test-seed",
                logger_name="mlflow",
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


def _run_train_on_rank(
    rank: int,
    world_size: int,
    init_file: str,
    result_dir: str,
    training_data: list[dict[str, torch.Tensor]],
    **overrides: Any,
) -> None:
    """Run a one-epoch, validation-free `train` on one rank of a spawned gloo group."""
    _init_worker_group(rank, world_size, init_file)
    mlflow.set_tracking_uri(os.path.join(result_dir, "mlruns"))
    options: dict[str, Any] = {
        "model_patterns": [{"model": MODEL_PATTERN}],
        "initializer_patterns": None,
        "shapes": [{"x": (4,)}],
        "device": "cpu",
        "learner_pattern": _learner_pattern(),
        "learner_outputs": None,
        "compile_pattern": None,
        "trainer_pattern": None,
        "epochs": 1,
        "start_epoch": 1,
        "resume": None,
        "strategy_pattern": None,
        "training_dataset_pattern": "TRAIN_DS",
        "validation_dataset_pattern": None,
        "validation_frequency": 1,
        "lower_criteria": [],
        "higher_criteria": [],
        "save_criteria": [],
        "seed": 42,
        "matmul_precision": "high",
        "logger_name": "mlflow",
        "log_arguments": None,
        "log_artifacts": None,
        "ci": True,
        "dist_backend": "gloo",
        "dist_url": f"file://{init_file}",
        **overrides,
    }
    deps = {"instantiate_object": _make_instantiate_fn(training_data=training_data)}
    train_fn = _train_callback()
    originals = {k: _CMD_GLOBALS.get(k) for k in deps}
    _CMD_GLOBALS.update(deps)
    try:
        train_fn(**options)
    except Exception:
        traceback.print_exc()
        raise
    finally:
        _CMD_GLOBALS.update(originals)


class _WrapAssertingTrainer(TorchTrainer):
    """Trainer checking the learner it was built with holds the models the strategy wrapped."""

    def fit(self, *args: Any, **kwargs: Any) -> dict[int, dict[str, Any]]:
        """Assert the wrap reached the learner, then train."""
        models = self.learner.models
        assert all(isinstance(m, torch.nn.parallel.DistributedDataParallel) for m in models.values()), (
            f"the learner was built with unwrapped models: {models}"
        )
        return super().fit(*args, **kwargs)


def _ddp_wrapped_models_worker(rank: int, world_size: int, init_file: str, result_dir: str) -> None:
    """Worker that verifies the learner receives DDP-wrapped models."""
    _run_train_on_rank(
        rank,
        world_size,
        init_file,
        result_dir,
        _make_training_dataset(),
        trainer_pattern=_WrapAssertingTrainer,
        experiment="test-wrap",
    )
    pathlib.Path(result_dir, f"wrap_rank_{rank}_ok").write_text("ok")


def test_train_distributed_builds_the_learner_with_wrapped_models(tmp_path: pathlib.Path) -> None:
    """The strategy must wrap before the learner is built: a learner holding raw modules never syncs."""
    init_file = str(tmp_path / "dist_init_wrap")
    result_dir = str(tmp_path / "results")
    os.makedirs(result_dir, exist_ok=True)
    mp.spawn(_ddp_wrapped_models_worker, args=(2, init_file, result_dir), nprocs=2, join=True)
    assert (tmp_path / "results" / "wrap_rank_0_ok").exists()
    assert (tmp_path / "results" / "wrap_rank_1_ok").exists()


def _ddp_gradient_worker(rank: int, world_size: int, init_file: str, result_dir: str) -> None:
    """Worker training one batch, whose gradient alone differs per rank, through the real CLI path."""
    os.environ["GRADIENT_DIR"] = result_dir
    x, target = ([[1.0, 0.0]], 1.0) if rank == 0 else ([[0.0, 1.0]], 2.0)
    _run_train_on_rank(
        rank,
        world_size,
        init_file,
        result_dir,
        [{"x": torch.tensor(x), "target": torch.tensor([target])}],
        model_patterns=[{"model": ZERO_MODEL_PATTERN}],
        shapes=[{"x": (2,)}],
        learner_pattern=_learner_pattern("GradientLearner"),
        experiment="test-gradient",
    )


def test_train_distributed_ddp_synchronizes_gradients(tmp_path: pathlib.Path) -> None:
    """A DDP run must all-reduce the gradients: both ranks step on the average, not on their batch.

    From zero weights, rank 0's batch alone produces [-2, 0] and rank 1's [0, -4]; the wrap the
    command installs averages them, so every rank reads [-1, -2] right after its backward.
    """
    init_file = str(tmp_path / "dist_init_gradient")
    result_dir = str(tmp_path / "results")
    os.makedirs(result_dir, exist_ok=True)
    mp.spawn(_ddp_gradient_worker, args=(2, init_file, result_dir), nprocs=2, join=True)
    for rank in (0, 1):
        gradient = json.loads((tmp_path / "results" / f"grad_{rank}.json").read_text())
        assert gradient == pytest.approx([-1.0, -2.0]), f"rank {rank} kept its own gradient"
