"""Unit tests for structcast_model.commands.cmd_flax."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterator, Mapping, Sequence
from importlib.util import module_from_spec, spec_from_file_location
import json
import logging
from math import isfinite
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import jax
import jax.numpy as jnp
import mlflow
from mlflow.tracking import MlflowClient
import optax
import pytest
from typer import Typer
from typer.testing import CliRunner
import wandb
from yaml import safe_load

from flax import nnx
from structcast_model.builders.flax import FlaxBuilder, FlaxLearnerBuilder
from structcast_model.commands.cmd_flax import app
from structcast_model.flax.distributed import FlaxDistributedStrategy
from structcast_model.flax.trainer import FlaxTrainer
from structcast_model.loggers.state_backends import FlaxStateBackend
from structcast_model.utils.base import load_any
from tests import CFG_DIR, FIXTURES_DIR
from tests.fakes import CountingLearner

LINEAR_CFG = str(FIXTURES_DIR / "cfg" / "flax" / "Linear.yaml")
LEARNER_CFG = str(FIXTURES_DIR / "cfg" / "flax" / "LinearLearner.yaml")
MODEL_CFG = str(CFG_DIR / "flax" / "models" / "ConvNeXtV2.yaml")

# The module is published through a lazy importer exposing `__all__` alone, so the private helper
# under test is reached through the globals its command callbacks were defined in, as in cmd_torch.
_FIRST_CALLBACK = app.registered_commands[0].callback
assert _FIRST_CALLBACK is not None, "cmd_flax registers every command with a callback"
_CMD_GLOBALS: dict[str, Any] = _FIRST_CALLBACK.__globals__
_optimizer_hashes = _CMD_GLOBALS["_optimizer_hashes"]

# ---------------------------------------------------------------------------
# app structure
# ---------------------------------------------------------------------------


def test_app_is_typer_instance() -> None:
    """The cmd_flax `app` must be a Typer instance."""
    assert isinstance(app, Typer)
    names = [cmd.name or (cmd.callback.__name__ if cmd.callback else "") for cmd in app.registered_commands]
    assert "time" in names
    group_names = [g.name for g in app.registered_groups]
    assert "create" in group_names


def test_help_exits_zero(cli_runner: CliRunner) -> None:
    """--help should exit with code 0."""
    result = cli_runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "create" in result.output
    assert "time" in result.output


def test_time_help_exits_zero(cli_runner: CliRunner) -> None:
    """Time subcommand --help should exit with code 0."""
    result = cli_runner.invoke(app, ["time", "--help"])
    assert result.exit_code == 0


# ---------------------------------------------------------------------------
# 'create model' command — simple Linear layer
# ---------------------------------------------------------------------------


def test_create_model_linear(tmp_path: Any, cli_runner: CliRunner) -> None:
    """'create model' generates a script from a simple Linear config."""
    out = str(tmp_path / "model.py")
    result = cli_runner.invoke(app, ["create", "model", LINEAR_CFG, "--output", out])
    assert result.exit_code == 0, result.output
    text = (tmp_path / "model.py").read_text()
    assert "class Model" in text
    assert "Linear" in text


def test_create_model_linear_classname(tmp_path: Any, cli_runner: CliRunner) -> None:
    """'create model --classname' honours the custom class name."""
    out = str(tmp_path / "net.py")
    result = cli_runner.invoke(app, ["create", "model", LINEAR_CFG, "--classname", "MyLinear", "--output", out])
    assert result.exit_code == 0, result.output
    assert "class MyLinear" in (tmp_path / "net.py").read_text()


def test_create_model_no_structured_output(tmp_path: Any, cli_runner: CliRunner) -> None:
    """'create model --no-structured-output' does not return a dict."""
    out = str(tmp_path / "model.py")
    result = cli_runner.invoke(app, ["create", "model", LINEAR_CFG, "--no-structured-output", "--output", out])
    assert result.exit_code == 0, result.output
    body = (tmp_path / "model.py").read_text().rsplit("class Model", 1)[-1]
    assert "return {'" not in body


def test_create_model_convnextv2(tmp_path: Any, cli_runner: CliRunner) -> None:
    """'create model' generates a script from the ConvNeXtV2 config."""
    out = str(tmp_path / "model.py")
    result = cli_runner.invoke(app, ["create", "model", MODEL_CFG, "--output", out])
    assert result.exit_code == 0, result.output
    assert "class Model" in (tmp_path / "model.py").read_text()


# ---------------------------------------------------------------------------
# 'time' command — simple Linear layer
# ---------------------------------------------------------------------------


def test_time_linear(cli_runner: CliRunner) -> None:
    """'time' measures inference on a simple flax Linear layer."""
    pattern = (
        "[_obj_, {_addr_: flax.nnx.Linear},"
        " {_call_: {in_features: 4, out_features: 2,"
        " rngs: [_obj_, {_addr_: flax.nnx.Rngs}, {_call_: [0]}]}}]"
    )
    result = cli_runner.invoke(
        app,
        ["time", pattern, "--shape", "inputs: [4]", "--warmup-runs", "1", "--times", "1", "--batch-size", "1"],
    )
    assert result.exit_code == 0, result.output
    assert "Average inference time" in result.output


def test_time_linear_training_mode_kwargs_mapping(cli_runner: CliRunner) -> None:
    """'time --training-mode-kwargs' accepts a plain mapping of "nnx.view" flags, as its help documents."""
    pattern = (
        "[_obj_, {_addr_: flax.nnx.Linear},"
        " {_call_: {in_features: 4, out_features: 2,"
        " rngs: [_obj_, {_addr_: flax.nnx.Rngs}, {_call_: [0]}]}}]"
    )
    result = cli_runner.invoke(
        app,
        [
            "time",
            pattern,
            "--shape",
            "inputs: [4]",
            "--training-mode-kwargs",
            "{deterministic: true}",
            "--warmup-runs",
            "1",
            "--times",
            "1",
            "--batch-size",
            "1",
        ],
    )
    assert result.exit_code == 0, result.output


# ---------------------------------------------------------------------------
# 'create learner' command — the linear fixture
# ---------------------------------------------------------------------------


def _load(path: Path, name: str) -> Any:
    """Load a generated module by file path, the way a configuration does: not by import name."""
    spec = spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_create_learner_writes_an_importable_class(tmp_path: Path, cli_runner: CliRunner) -> None:
    """'create learner' generates a learner module that imports and exposes the named class."""
    out = tmp_path / "my_learner.py"
    result = cli_runner.invoke(
        app,
        [
            "create",
            "learner",
            LEARNER_CFG,
            "--classname",
            "MyLearner",
            "--parameter",
            "DEFAULT: {accumulate_gradients: 2}",
            "--output",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.output
    module = _load(out, "generated_learner")
    assert hasattr(module, "MyLearner")
    # The parameter reached the template: the MultiSteps wrapper carries the window on the device,
    # and the step detects the updates it applies by reading the count back across its own update.
    text = out.read_text()
    assert "MultiSteps" in text
    assert "_before = gradient_steps(optimizer)" in text
    assert "_has_updated = True if _before is None else gradient_steps(optimizer) > _before" in text


# ---------------------------------------------------------------------------
# 'train' command — end to end over the generated linear model and learner
# ---------------------------------------------------------------------------


def linear_batches(count: int = 3, size: int = 2) -> list[dict[str, Any]]:
    """Return a fixed dataset of *count* batches of *size* rows, so criteria depend on the seed alone.

    Public and addressable: the runs build it through an object pattern, as a real dataset is built.
    """
    return [{"x": jnp.full((size, 4), float(index + 1)), "y": jnp.zeros((size, 2))} for index in range(count)]


DATASET = f"[_obj_, {{_addr_: {__name__}.linear_batches}}, _call_]"
"""Object pattern building the training dataset."""

VALIDATION_DATASET = f"[_obj_, {{_addr_: {__name__}.linear_batches}}, {{_call_: {{count: 2}}}}]"
"""Object pattern building a shorter validation dataset."""

EPOCHS_SEEN: list[int] = []
"""Epochs the event method of `epoch_aware_batches` was called with; cleared by the test that reads it."""


class _EpochAwareDataset(list[dict[str, Any]]):
    """A dataset reacting to epochs, the way a wrapper driving a per-rank sampler does."""

    def on_epoch_begin(self, info: Any) -> None:
        """Record the epoch the trainer is starting."""
        EPOCHS_SEEN.append(info.epoch)


def epoch_aware_batches() -> _EpochAwareDataset:
    """Return the fixed training batches as a dataset that is also an event callback."""
    return _EpochAwareDataset(linear_batches())


EPOCH_AWARE_DATASET = f"[_obj_, {{_addr_: {__name__}.epoch_aware_batches}}, _call_]"
"""Object pattern building the training dataset that reacts to epochs."""


class PrebuiltModel(nnx.Module):
    """A module a pattern can build on its own, standing in for a pattern that carries "_call_"."""

    def __call__(self, x: jax.Array) -> jax.Array:
        """Never reached: the command refuses the pattern before any step runs."""
        return x


class NamelessLearner(CountingLearner):
    """A hand-written learner that trains but declares no criterion names, as the protocol allows."""

    def __init__(self, model: nnx.Module) -> None:
        """Keep *model* and the optimizer over it."""
        super().__init__()
        self._models = {"model": model}
        self._optimizers = {"optimizer": nnx.Optimizer(model, tx=optax.sgd(0.1), wrt=nnx.Param)}

    @property
    def models(self) -> dict[str, Any]:
        """The single model the learner was built over."""
        return self._models

    @property
    def optimizers(self) -> dict[str, Any]:
        """The single optimizer."""
        return self._optimizers

    @property
    def optimizer_models(self) -> dict[str, list[str]]:
        """The optimizer owns the only model."""
        return {"optimizer": ["model"]}

    @property
    def learning_rates(self) -> dict[str, float]:
        """The rate the optimizer was built with."""
        return {"optimizer": 0.1}

    def training_step(self, x: jax.Array, y: jax.Array) -> dict[str, Any]:
        """Take one plain gradient step, count it as one Update, and report the squared error."""
        model, optimizer = self._models["model"], self._optimizers["optimizer"]
        loss, grads = nnx.value_and_grad(lambda m: jnp.mean((m(x) - y) ** 2))(model)
        optimizer.update(model, grads)
        self.count_step()
        return {"loss": loss}

    def inference_step(self, x: jax.Array, y: jax.Array) -> dict[str, Any]:
        """Report the squared error without touching any state."""
        return {"loss": jnp.mean((self._models["model"](x) - y) ** 2)}


COMPILE_CALLS: list[tuple[str, dict[str, Any]]] = []
"""What the recording strategy below was asked to compile; cleared by the test that reads it."""


class RecordingStrategy(FlaxDistributedStrategy):
    """A strategy recording the compilation contract the command hands each flow function."""

    def compile(self, module: Any, compile_kw: Mapping[str, Any] | None) -> Any:
        """Record the arguments, then compile exactly as the shipped strategy does."""
        COMPILE_CALLS.append((getattr(module, "__name__", repr(module)), dict(compile_kw or {})))
        return super().compile(module, compile_kw)


class RecordingTrainer(FlaxTrainer):
    """A trainer recording that --trainer reached the run."""

    def fit(self, *args: Any, **kwargs: Any) -> Any:
        """Note the run before handing over to the shared loop."""
        TRAINERS_USED.append(type(self).__name__)
        return super().fit(*args, **kwargs)


TRAINERS_USED: list[str] = []
"""Names of the trainers `RecordingTrainer` ran under; cleared by the test that reads it."""


class ReplacingStrategy(FlaxDistributedStrategy):
    """A strategy whose `wrap` hands back other modules, as the protocol allows a user factory to."""

    def wrap(self, models: OrderedDict[str, nnx.Module]) -> OrderedDict[str, nnx.Module]:
        """Return zeroed clones of *models*, leaving the given modules untouched."""
        clones = OrderedDict((name, nnx.clone(model)) for name, model in models.items())
        for clone in clones.values():
            state = nnx.state(clone, nnx.Param)
            nnx.replace_by_pure_dict(state, jax.tree.map(jnp.zeros_like, nnx.to_pure_dict(state)))
            nnx.update(clone, state)
        return clones


@pytest.fixture(autouse=True)
def _clear_mesh() -> Iterator[None]:
    """Unset the mesh a strategy activated, so it does not leak into unrelated tests.

    ``jax.set_mesh`` is process-wide, as in ``tests/flax/test_distributed``.
    """
    yield
    jax.set_mesh(None)


@pytest.fixture(scope="module")
def patterns(tmp_path_factory: pytest.TempPathFactory) -> tuple[str, str]:
    """Return the model and learner patterns of the generated linear fixture, built once."""
    directory = tmp_path_factory.mktemp("generated")
    FlaxBuilder.from_path(FIXTURES_DIR / "cfg" / "flax" / "Linear.yaml")()(directory / "model.py")
    FlaxLearnerBuilder.from_path(FIXTURES_DIR / "cfg" / "flax" / "LinearLearner.yaml")()(directory / "learner.py")
    return (
        f"model: [_obj_, {{_addr_: Model, _file_: {directory / 'model.py'}}}]",
        f"[_obj_, {{_addr_: Learner, _file_: {directory / 'learner.py'}}}]",
    )


def _train(
    cli_runner: CliRunner,
    patterns: tuple[str, str],
    tmp_path: Path,
    *,
    experiment: str,
    epochs: int = 2,
    extra: Sequence[str] = (),
) -> Any:
    """Run `train` over the generated fixture, recording to an MLflow store under *tmp_path*."""
    mlflow.set_tracking_uri(str(tmp_path / "mlruns"))
    model_pattern, learner_pattern = patterns
    result = cli_runner.invoke(
        app,
        [
            "train",
            model_pattern,
            "--learner",
            learner_pattern,
            "--training-dataset",
            DATASET,
            "--validation-dataset",
            VALIDATION_DATASET,
            "--epochs",
            str(epochs),
            "--lower-criterion",
            "loss",
            "--lower-criterion",
            "val_loss",
            "--save-criterion",
            "val_loss",
            "--experiment",
            experiment,
            "--ci",
            *extra,
        ],
    )
    assert result.exit_code == 0, f"{result.output}\n{result.exception}"
    return result


def _runs(experiment: str) -> list[Any]:
    """Return the runs of *experiment*, oldest first."""
    runs = mlflow.search_runs(experiment_names=[experiment], output_format="list")
    return sorted(runs, key=lambda run: run.info.start_time)


def test_train_end_to_end(tmp_path: Path, cli_runner: CliRunner, patterns: tuple[str, str]) -> None:
    """A full run trains, validates, prints its routed events and records everything it produced."""
    result = _train(cli_runner, patterns, tmp_path, experiment="flax-e2e")
    assert "Registered callbacks:" in result.output
    assert "FlaxTrainingStateSaver" in result.output
    (run,) = _runs("flax-e2e")
    assert isfinite(run.data.metrics["loss"])
    assert isfinite(run.data.metrics["val_loss"])
    # The learning rate the injected optimizer reports, which is NaN when the injection is lost.
    assert run.data.metrics["optimizer"] == pytest.approx(0.1)
    history = MlflowClient().get_metric_history(run.info.run_id, "loss")
    assert [metric.step for metric in history] == [1, 2]
    assert history[1].value < history[0].value
    artifacts = {artifact.path for artifact in MlflowClient().list_artifacts(run.info.run_id)}
    assert {"training_state.tar.gz", "best_val_loss.tar.gz", "arguments.yaml"} <= artifacts
    # Plain YAML, and the two entries the command computes rather than echoes back.
    (recorded,) = (tmp_path / "mlruns").rglob("arguments.yaml")
    arguments = safe_load(recorded.read_text())
    assert arguments["parameters"] == {"model": 10}
    assert arguments["mesh"] == {"data": 1}


def test_train_without_compilation(tmp_path: Path, cli_runner: CliRunner, patterns: tuple[str, str]) -> None:
    """--compile none runs the steps eagerly, and must reach the same loss as the compiled run."""
    _train(cli_runner, patterns, tmp_path, experiment="flax-eager", epochs=1, extra=["--compile", "none"])
    _train(cli_runner, patterns, tmp_path, experiment="flax-eager", epochs=1)
    eager, compiled = _runs("flax-eager")
    assert eager.data.metrics["loss"] == pytest.approx(compiled.data.metrics["loss"], rel=1e-6)


def test_train_repeats_a_seeded_run(tmp_path: Path, cli_runner: CliRunner, patterns: tuple[str, str]) -> None:
    """--seed decides the initialization: one seed repeats, another does not, so the seed is used."""
    for _ in range(2):
        _train(cli_runner, patterns, tmp_path, experiment="flax-seed", epochs=1, extra=["--seed", "7"])
    _train(cli_runner, patterns, tmp_path, experiment="flax-seed", epochs=1, extra=["--seed", "8"])
    first, second, other = _runs("flax-seed")
    assert first.data.metrics["loss"] == second.data.metrics["loss"]
    assert other.data.metrics["loss"] != first.data.metrics["loss"]


def test_train_keeps_the_contract_compilation_arguments(
    tmp_path: Path, cli_runner: CliRunner, patterns: tuple[str, str]
) -> None:
    """--compile cannot redefine what is static and what is donated, in either spelling.

    `static_argnums: 0` would freeze the models the training step rewrites, and `donate_argnums: 0`
    would donate the buffers the inference views share with them: both are dropped, so the run is
    the plain compiled one.
    """
    override = "{static_argnums: 0, donate_argnums: 0}"
    _train(cli_runner, patterns, tmp_path, experiment="flax-compile-kw", epochs=1, extra=["--compile", override])
    _train(cli_runner, patterns, tmp_path, experiment="flax-compile-kw", epochs=1)
    overridden, plain = _runs("flax-compile-kw")
    assert overridden.data.metrics["loss"] == plain.data.metrics["loss"]


def test_train_routes_the_events_of_the_dataset_it_shards(
    tmp_path: Path, cli_runner: CliRunner, patterns: tuple[str, str]
) -> None:
    """Placing the batches wraps the dataset, which must not hide the events the trainer scans for."""
    EPOCHS_SEEN.clear()
    _train(
        cli_runner,
        patterns,
        tmp_path,
        experiment="flax-dataset-events",
        epochs=2,
        extra=["--training-dataset", EPOCH_AWARE_DATASET],
    )
    assert EPOCHS_SEEN == [1, 2]


def test_train_learns_from_the_models_the_strategy_returns(
    tmp_path: Path, cli_runner: CliRunner, patterns: tuple[str, str]
) -> None:
    """A strategy replacing the models it wraps must be the one the learner is built over.

    The shipped strategy places the arrays in place and hands the same modules back, so only a
    strategy returning others tells the two apart: these are zeroed, which pins the loss at zero.
    """
    strategy = f"[_obj_, {{_addr_: {__name__}.ReplacingStrategy}}]"
    _train(cli_runner, patterns, tmp_path, experiment="flax-replaced", epochs=1, extra=["--strategy", strategy])
    (run,) = _runs("flax-replaced")
    assert run.data.metrics["loss"] == 0.0


def test_train_with_a_strategy_pattern(tmp_path: Path, cli_runner: CliRunner, patterns: tuple[str, str]) -> None:
    """--strategy accepts a path to an object pattern, here the shipped data-parallel template."""
    strategy = str(CFG_DIR / "flax" / "strategies" / "dp.yaml")
    _train(cli_runner, patterns, tmp_path, experiment="flax-dp", epochs=1, extra=["--strategy", strategy])
    (run,) = _runs("flax-dp")
    assert isfinite(run.data.metrics["loss"])


def test_train_resumes_from_a_saved_training_state(
    tmp_path: Path, cli_runner: CliRunner, patterns: tuple[str, str], caplog: pytest.LogCaptureFixture
) -> None:
    """--resume continues at the epoch after the saved one, with the state the first run left."""
    _train(cli_runner, patterns, tmp_path, experiment="flax-resume", epochs=2)
    (state,) = (tmp_path / "mlruns").rglob("training_state.tar.gz")
    with caplog.at_level(logging.INFO, logger="structcast_model.flax.trainer"):
        _train(
            cli_runner,
            patterns,
            tmp_path,
            experiment="flax-resume",
            epochs=3,
            extra=["--resume", str(state), "--start-epoch", "2"],
        )
    assert "Ignoring --start-epoch 2: the resumed state continues at epoch 3." in caplog.text
    first, resumed = _runs("flax-resume")
    history = MlflowClient().get_metric_history(resumed.info.run_id, "loss")
    assert [metric.step for metric in history] == [3]
    # Training continued from the restored weights rather than from a fresh initialization.
    assert history[0].value < first.data.metrics["loss"]


def test_train_resumes_the_averaged_shadow_models_the_learner_declares(
    tmp_path: Path, cli_runner: CliRunner, patterns: tuple[str, str]
) -> None:
    """A resume restores everything the learner calls a model, which is more than the command built.

    The command builds the models named on its command line; the average is the learner's own, and
    the saver writes it because it writes `learner.models`. Restoring the command's mapping instead
    would leave the average at its construction value -- a checkpoint that saves what it cannot
    resume. One batch per epoch makes the blend exact: the resumed run's average has to be the saved
    one blended once with the parameters that epoch left behind.
    """
    raw = {**load_any(LEARNER_CFG), "EMA": {"model": {"decay": 0.5}}}
    FlaxLearnerBuilder(raw=raw, current_path=LEARNER_CFG)()(tmp_path / "averaging.py")
    model_pattern, _ = patterns
    averaging = (model_pattern, f"[_obj_, {{_addr_: Learner, _file_: {tmp_path / 'averaging.py'}}}]")
    one_batch = ["--training-dataset", f"[_obj_, {{_addr_: {__name__}.linear_batches}}, {{_call_: {{count: 1}}}}]"]
    _train(cli_runner, averaging, tmp_path, experiment="flax-ema-resume", epochs=1, extra=one_batch)
    (first,) = (tmp_path / "mlruns").rglob("training_state.tar.gz")
    saved = FlaxStateBackend().load(first)["models"]

    _train(
        cli_runner,
        averaging,
        tmp_path,
        experiment="flax-ema-resume",
        epochs=2,
        extra=[*one_batch, "--resume", str(first)],
    )

    path = next(p for p in (tmp_path / "mlruns").rglob("training_state.tar.gz") if p != first)
    resumed = FlaxStateBackend().load(path)["models"]
    expected = 0.5 * jnp.asarray(saved["ema_model"]["fc"]["kernel"]) + 0.5 * jnp.asarray(
        resumed["model"]["fc"]["kernel"]
    )
    assert jnp.allclose(jnp.asarray(resumed["ema_model"]["fc"]["kernel"]), expected, rtol=1e-6)


def test_train_warns_when_the_optimizer_pattern_changed_between_save_and_resume(
    tmp_path: Path, cli_runner: CliRunner, patterns: tuple[str, str], caplog: pytest.LogCaptureFixture
) -> None:
    """A resumed run rebuilds `tx` from the configuration, and a changed schedule must be reported.

    The restored optimizer state carries the step count but not the transformation, so a run resumed
    against another rate silently continues the new schedule from the old count. The digests the
    generated learner emits are what makes that visible, end to end: saved with one rate, resumed
    with another, the loader names the segment.
    """
    _train(cli_runner, patterns, tmp_path, experiment="flax-rebuilt-optimizer", epochs=1)
    (state,) = (tmp_path / "mlruns").rglob("training_state.tar.gz")
    raw = load_any(LEARNER_CFG)
    raw["LEARNERS"][0]["OPTIMIZER"][2] = {
        "_bind_": {"tx": ["_obj_", {"_addr_": "optax.sgd"}, {"_call_": {"learning_rate": 0.2}}]}
    }
    FlaxLearnerBuilder(raw=raw, current_path=LEARNER_CFG)()(tmp_path / "faster_learner.py")
    rebuilt = (patterns[0], f"[_obj_, {{_addr_: Learner, _file_: {tmp_path / 'faster_learner.py'}}}]")

    with caplog.at_level(logging.WARNING):
        _train(
            cli_runner,
            rebuilt,
            tmp_path,
            experiment="flax-rebuilt-optimizer",
            epochs=2,
            extra=["--resume", str(state)],
        )

    assert 'optimizer of segment "optimizer" is not the one the state was saved with' in caplog.text


# ---------------------------------------------------------------------------
# 'train' command — argument validation
# ---------------------------------------------------------------------------


UNUSED_PATTERN = "[_obj_, {_addr_: builtins.dict}]"
"""A well-formed pattern for the options a validation failure is reached before using."""


def _train_error(cli_runner: CliRunner, arguments: Sequence[str]) -> BaseException:
    """Invoke `train` with *arguments* and return the exception it failed with."""
    result = cli_runner.invoke(app, ["train", *arguments])
    assert result.exit_code != 0
    assert result.exception is not None, result.output
    return result.exception


def test_train_refuses_an_empty_model_pattern_list() -> None:
    """Without a model there is nothing to give the learner, and the failure must say so here.

    Typer rejects a missing positional argument itself, so the guard is reached only by calling the
    command as a function -- which is how a caller embedding the CLI reaches it too.
    """
    (command,) = (cmd for cmd in app.registered_commands if cmd.callback and cmd.callback.__name__ == "train")
    assert command.callback is not None

    with pytest.raises(ValueError, match="At least one model pattern"):
        command.callback(model_patterns=[])


def test_train_refuses_a_pattern_naming_two_models(cli_runner: CliRunner, patterns: tuple[str, str]) -> None:
    """One argument is one named model: two entries would silently drop one of them."""
    model_pattern, learner_pattern = patterns
    body = model_pattern.split("model: ", 1)[1]
    doubled = f"{{model: {body}, second: {body}}}"
    error = _train_error(cli_runner, [doubled, "--learner", learner_pattern, "--training-dataset", DATASET])

    assert isinstance(error, ValueError)
    assert "exactly one model definition" in str(error)


def test_train_refuses_a_model_pattern_that_builds_the_module_itself(cli_runner: CliRunner) -> None:
    """A pattern carrying "_call_" hands back a built module, so the run's seeded RNG never reaches it.

    Left through, the command calls the module -- running its forward pass with a `rngs` keyword --
    and the failure surfaces from inside the generated model instead of from the pattern.
    """
    pattern = f"model: [_obj_, {{_addr_: {__name__}.PrebuiltModel}}, _call_]"
    error = _train_error(cli_runner, [pattern, "--learner", UNUSED_PATTERN, "--training-dataset", DATASET])

    assert isinstance(error, ValueError)
    assert 'Drop the "_call_" entry' in str(error)


def test_train_refuses_a_learner_that_names_no_criteria(cli_runner: CliRunner, patterns: tuple[str, str]) -> None:
    """The criterion names build the tracker, so a learner declaring none must be named on the command line."""
    model_pattern, _ = patterns
    learner = f"[_obj_, {{_addr_: {__name__}.NamelessLearner}}]"
    error = _train_error(cli_runner, [model_pattern, "--learner", learner, "--training-dataset", DATASET])

    assert isinstance(error, ValueError)
    assert 'Module "learner" does not have an "outputs" attribute' in str(error)


# ---------------------------------------------------------------------------
# 'train' command — --gpu-memory-fraction
# ---------------------------------------------------------------------------


XLA_MEMORY_VARIABLES = ("XLA_PYTHON_CLIENT_MEM_FRACTION", "XLA_PYTHON_CLIENT_PREALLOCATE")
"""The pair `--gpu-memory-fraction` writes, which the tests below own for their duration."""


class _JaxTouched(Exception):
    """Ends the run at its first `jax` attribute access, once the probe has read the environment."""


class _JaxProbe:
    """Stands in for cmd_flax's `jax`, recording the environment as it was when JAX is first used.

    JAX reads the memory-fraction variables while it brings up a backend, which cannot happen before
    the module is first used for anything -- so the environment at the first attribute access is
    exactly what XLA gets to see, and a cap written after this point would reach nothing.
    """

    def __init__(self) -> None:
        """Start with nothing recorded."""
        self.environment: dict[str, str] = {}

    def __getattr__(self, name: str) -> Any:
        """Record the environment, then stop the run before it reaches the real JAX."""
        self.environment = dict(os.environ)
        raise _JaxTouched(name)


@pytest.fixture
def xla_memory_environment() -> Iterator[None]:
    """Run the test with both XLA memory variables unset, restoring whatever was there afterwards."""
    saved = {name: os.environ.pop(name, None) for name in XLA_MEMORY_VARIABLES}
    yield
    for name, value in saved.items():
        os.environ.pop(name, None)
        if value is not None:
            os.environ[name] = value


@pytest.mark.parametrize(
    ("preset", "flag", "expected"),
    [
        pytest.param(None, "0.25", "0.25", id="flag-alone"),
        pytest.param("0.3", None, "0.3", id="environment-alone"),
        pytest.param("0.3", "0.25", "0.25", id="flag-over-environment"),
        pytest.param(None, None, None, id="neither"),
    ],
)
def test_train_exports_the_xla_cap_before_anything_touches_jax(
    cli_runner: CliRunner,
    monkeypatch: pytest.MonkeyPatch,
    xla_memory_environment: None,
    preset: str | None,
    flag: str | None,
    expected: str | None,
) -> None:
    """The cap has to be in the environment before JAX is used, and the flag has to win over it.

    Written afterwards it would be read by nothing and the run would take XLA's default share of
    every device while reporting the cap it was given, so the probe standing in for `jax` snapshots
    the environment at the first access and ends the run there. Preallocation is turned off with the
    fraction, which is also what tells a cap the command applied apart from one it merely inherited.
    """
    if preset is not None:
        os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = preset
    probe = _JaxProbe()
    monkeypatch.setitem(_CMD_GLOBALS, "jax", probe)
    fraction = [] if flag is None else ["--gpu-memory-fraction", flag]

    result = cli_runner.invoke(
        app,
        ["train", f"model: {UNUSED_PATTERN}", "--learner", UNUSED_PATTERN, "--training-dataset", DATASET, *fraction],
    )

    assert isinstance(result.exception, _JaxTouched), result.output
    if expected is None:
        assert not [name for name in XLA_MEMORY_VARIABLES if name in probe.environment]
    else:
        assert probe.environment["XLA_PYTHON_CLIENT_MEM_FRACTION"] == expected
        assert probe.environment["XLA_PYTHON_CLIENT_PREALLOCATE"] == "false"


@pytest.mark.parametrize("fraction", ["0", "-0.5", "1.5"])
def test_train_refuses_a_memory_fraction_outside_the_unit_interval(
    cli_runner: CliRunner, xla_memory_environment: None, fraction: str
) -> None:
    """A fraction at 0 or outside (0, 1] caps nothing, and would otherwise be exported as if it did.

    XLA silently ignores a value it cannot use, so a run given one would take its default share of
    every device while reporting the cap it was asked for.
    """
    error = _train_error(
        cli_runner,
        [
            f"model: {UNUSED_PATTERN}",
            "--learner",
            UNUSED_PATTERN,
            "--training-dataset",
            DATASET,
            f"--gpu-memory-fraction={fraction}",
        ],
    )

    assert isinstance(error, ValueError)
    assert "must be in (0, 1]" in str(error)
    assert "XLA_PYTHON_CLIENT_MEM_FRACTION" not in os.environ


def test_train_names_the_criteria_from_learner_outputs(
    tmp_path: Path, cli_runner: CliRunner, patterns: tuple[str, str]
) -> None:
    """--learner-outputs is the only source of names for a learner declaring none, and it must be used."""
    model_pattern, _ = patterns
    learner = f"[_obj_, {{_addr_: {__name__}.NamelessLearner}}]"
    _train(
        cli_runner,
        (model_pattern, learner),
        tmp_path,
        experiment="flax-outputs",
        epochs=1,
        extra=["--learner-outputs", "loss"],
    )

    (run,) = _runs("flax-outputs")
    assert isfinite(run.data.metrics["loss"])


# ---------------------------------------------------------------------------
# 'train' command — the seams the options are supposed to reach
# ---------------------------------------------------------------------------


def test_train_compiles_each_flow_under_its_own_contract(
    tmp_path: Path, cli_runner: CliRunner, patterns: tuple[str, str], recwarn: pytest.WarningsRecorder
) -> None:
    """--compile is what binds the steps, and the training step's state parameters alone are donated.

    The donated names are read off the step's signature, so they are exactly the models and
    optimizers the generated learner names there -- the keyword-only batch is never donated.
    Comparing losses cannot see any of this: an ignored --compile reaches the same numbers eagerly.
    Donating the inference views -- which share their arrays with the models -- or failing to donate
    the training state shows up only as a JAX warning, so the recorded warnings are checked too.
    """
    COMPILE_CALLS.clear()
    strategy = f"[_obj_, {{_addr_: {__name__}.RecordingStrategy}}]"
    _train(cli_runner, patterns, tmp_path, experiment="flax-compile", epochs=1, extra=["--strategy", strategy])

    contracts = dict(COMPILE_CALLS)
    assert contracts["_training_step"] == {
        "donate_argnames": ("model", "optimizer"),
    }
    # Every other flow is an inference one: no static flag, and nothing donated.
    assert [arguments for name, arguments in COMPILE_CALLS if name != "_training_step"] == [{}] * (
        len(COMPILE_CALLS) - 1
    )
    assert not [warning for warning in recwarn.list if "donated" in str(warning.message)]


def test_the_optimizer_digests_are_read_off_the_learner_class(patterns: tuple[str, str]) -> None:
    """The generated learner declares its digests as a class attribute, and nothing else may be read.

    A generated learner class is loaded from a file rather than imported by name, so its module
    never lands in `sys.modules`; the class is the only handle on the digests a resume compares.
    A hand-written learner declares none, and the resume check skips what it cannot find.
    """
    _, learner_pattern = patterns
    path = Path(learner_pattern.split("_file_: ", 1)[1].rstrip("}]"))
    learner_type = _load(path, "digest_learner").Learner

    hashes = _optimizer_hashes(learner_type.__new__(learner_type))

    assert hashes == learner_type.OPTIMIZER_HASHES
    assert set(hashes) == {"optimizer"}
    assert _optimizer_hashes(NamelessLearner.__new__(NamelessLearner)) == {}


def test_train_without_compilation_binds_nothing(
    tmp_path: Path, cli_runner: CliRunner, patterns: tuple[str, str]
) -> None:
    """--compile none must leave the steps as the learner defined them, not compile them anyway."""
    COMPILE_CALLS.clear()
    strategy = f"[_obj_, {{_addr_: {__name__}.RecordingStrategy}}]"
    _train(
        cli_runner,
        patterns,
        tmp_path,
        experiment="flax-no-compile",
        epochs=1,
        extra=["--strategy", strategy, "--compile", "none"],
    )

    assert COMPILE_CALLS == []


def test_train_runs_the_trainer_the_option_names(
    tmp_path: Path, cli_runner: CliRunner, patterns: tuple[str, str]
) -> None:
    """--trainer replaces the loop driver, so a run must go through the named class, not FlaxTrainer."""
    TRAINERS_USED.clear()
    trainer = f"[_obj_, {{_addr_: {__name__}.RecordingTrainer}}]"
    _train(cli_runner, patterns, tmp_path, experiment="flax-trainer", epochs=1, extra=["--trainer", trainer])

    assert TRAINERS_USED == ["RecordingTrainer"]


def test_train_shows_a_progress_bar_without_ci(
    tmp_path: Path, cli_runner: CliRunner, patterns: tuple[str, str]
) -> None:
    """Without --ci the run must register the bar; the routed-callback map is where that is visible."""
    mlflow.set_tracking_uri(str(tmp_path / "mlruns"))
    model_pattern, learner_pattern = patterns
    result = cli_runner.invoke(
        app,
        [
            "train",
            model_pattern,
            "--learner",
            learner_pattern,
            "--training-dataset",
            DATASET,
            "--epochs",
            "1",
            "--experiment",
            "flax-bar",
        ],
    )

    assert result.exit_code == 0, f"{result.output}\n{result.exception}"
    assert "ProgressBar" in result.output
    assert "Printer" not in result.output


def test_train_records_a_run_through_the_wandb_backend(
    tmp_path: Path, cli_runner: CliRunner, patterns: tuple[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """--logger wandb has to carry the Flax state backend, or the run writes a torch pickle of jax arrays."""
    monkeypatch.setenv("WANDB_MODE", "offline")
    monkeypatch.setenv("WANDB_DIR", str(tmp_path))
    monkeypatch.setenv("WANDB_SILENT", "true")
    monkeypatch.setenv("WANDB_CONSOLE", "off")

    _train(cli_runner, patterns, tmp_path, experiment="flax-wandb", epochs=1, extra=["--logger", "wandb"])

    assert wandb.run is None, "the command must close the run it opened"
    assert list(tmp_path.rglob("training_state.tar.gz"))
    assert list(tmp_path.rglob("arguments.yaml"))


# ---------------------------------------------------------------------------
# 'train' command — more devices than a running JAX process can gain
# ---------------------------------------------------------------------------

MULTI_DEVICE_SCRIPT = """
import json, sys
import jax
import mlflow
from typer.testing import CliRunner

from structcast_model.commands.cmd_flax import app

directory = sys.argv[1]
mlflow.set_tracking_uri(directory + "/mlruns")


def dataset(size):
    return "[_obj_, {_addr_: batches, _file_: " + directory + "/data.py}, {_call_: {size: " + str(size) + "}}]"


def run(experiment, strategy, size):
    # The mesh is process-wide and each run activates its own, so the previous one has to go first.
    jax.set_mesh(None)
    result = CliRunner().invoke(app, [
        "train",
        *("%s: [_obj_, {_addr_: Model, _file_: %s/model.py}]" % (name, directory) for name in "abc"),
        "--learner", "[_obj_, {_addr_: Learner, _file_: " + directory + "/learner.py}]",
        "--training-dataset", dataset(size),
        "--epochs", "1",
        "--experiment", experiment,
        "--strategy", strategy,
        "--ci",
    ])
    if result.exit_code != 0:
        return {"error": str(result.exception)}
    run, = mlflow.search_runs(experiment_names=[experiment], output_format="list")
    return {name: run.data.metrics[name] for name in ("loss_ab", "loss_c")}


print(json.dumps({
    "devices": jax.device_count(),
    "single": run("single", "single", 8),
    "dp": run("dp", "dp", 8),
    "indivisible": run("odd", "dp", 2),
}))
"""

DATA_MODULE = '''
"""The dataset the multi-device run trains on, addressed by file from an object pattern."""

import jax.numpy as jnp


def batches(size):
    """Return three fixed batches of *size* rows."""
    return [{"x": jnp.full((size, 4), float(i + 1)), "y": jnp.zeros((size, 2))} for i in range(3)]
'''


def test_train_splits_a_batch_across_four_devices_without_changing_the_criteria(tmp_path: Path) -> None:
    """The CLI's own seam -- strategy, sharded loader, learner -- has to hold on more than one device.

    Two segments, so a per-segment optimizer that picked up the other segment's gradients under
    sharding would move the wrong parameters and shift the criterion that reports them. The batch
    that the mesh cannot split must be refused by name rather than silently padded or replicated.
    """
    FlaxBuilder.from_path(FIXTURES_DIR / "cfg" / "flax" / "Linear.yaml")()(tmp_path / "model.py")
    FlaxLearnerBuilder.from_path(FIXTURES_DIR / "cfg" / "flax" / "TwoSegmentLearner.yaml")()(tmp_path / "learner.py")
    (tmp_path / "data.py").write_text(DATA_MODULE)
    script = tmp_path / "script.py"
    script.write_text(f"import jax\njax.config.update('jax_num_cpu_devices', 4)\n{MULTI_DEVICE_SCRIPT}")

    process = subprocess.run(
        [sys.executable, str(script), str(tmp_path)], capture_output=True, text=True, timeout=600, check=False
    )
    assert process.returncode == 0, process.stderr
    result = json.loads(process.stdout.splitlines()[-1])

    assert result["devices"] == 4
    for criterion in ("loss_ab", "loss_c"):
        assert result["dp"][criterion] == pytest.approx(result["single"][criterion], rel=1e-6)
    assert '"x"' in result["indivisible"]["error"]
