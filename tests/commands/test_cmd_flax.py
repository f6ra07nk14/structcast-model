"""Unit tests for structcast_model.commands.cmd_flax."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterator, Sequence
from importlib.util import module_from_spec, spec_from_file_location
from math import isfinite
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import mlflow
from mlflow.tracking import MlflowClient
import pytest
from typer import Typer
from typer.testing import CliRunner
from yaml import safe_load

from flax import nnx
from structcast_model.builders.flax import FlaxBuilder, FlaxLearnerBuilder
from structcast_model.commands.cmd_flax import app
from structcast_model.flax.distributed import FlaxDistributedStrategy
from tests import CFG_DIR, FIXTURES_DIR

LINEAR_CFG = str(FIXTURES_DIR / "cfg" / "flax" / "Linear.yaml")
LEARNER_CFG = str(FIXTURES_DIR / "cfg" / "flax" / "LinearLearner.yaml")
MODEL_CFG = str(CFG_DIR / "flax" / "models" / "ConvNeXtV2.yaml")

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
    # The parameter reached the template: accumulation makes the learner update every second step.
    assert "(step + 1) % 2 == 0" in out.read_text()


# ---------------------------------------------------------------------------
# 'train' command — end to end over the generated linear model and learner
# ---------------------------------------------------------------------------


def linear_batches(count: int = 3) -> list[dict[str, Any]]:
    """Return a fixed dataset, so a run's criteria depend on the seed alone.

    Public and addressable: the runs build it through an object pattern, as a real dataset is built.
    """
    return [{"x": jnp.full((2, 4), float(index + 1)), "y": jnp.zeros((2, 2))} for index in range(count)]


DATASET = f"[_obj_, {{_addr_: {__name__}.linear_batches}}, _call_]"
"""Object pattern building the training dataset."""

VALIDATION_DATASET = f"[_obj_, {{_addr_: {__name__}.linear_batches}}, {{_call_: {{count: 2}}}}]"
"""Object pattern building a shorter validation dataset."""

EPOCHS_SEEN: list[int] = []
"""Epochs the hook of `epoch_aware_batches` was called with; cleared by the test that reads it."""


class _EpochAwareDataset(list[dict[str, Any]]):
    """A dataset reacting to epochs, the way a wrapper driving a per-rank sampler does."""

    def on_epoch_begin(self, info: Any) -> None:
        """Record the epoch the trainer is starting."""
        EPOCHS_SEEN.append(info.epoch)


def epoch_aware_batches() -> _EpochAwareDataset:
    """Return the fixed training batches as a dataset carrying a lifecycle hook."""
    return _EpochAwareDataset(linear_batches())


EPOCH_AWARE_DATASET = f"[_obj_, {{_addr_: {__name__}.epoch_aware_batches}}, _call_]"
"""Object pattern building the training dataset that reacts to epochs."""


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


def test_train_routes_the_lifecycle_hooks_of_the_dataset_it_shards(
    tmp_path: Path, cli_runner: CliRunner, patterns: tuple[str, str]
) -> None:
    """Placing the batches wraps the dataset, which must not hide the hooks the trainer scans for."""
    EPOCHS_SEEN.clear()
    _train(
        cli_runner,
        patterns,
        tmp_path,
        experiment="flax-dataset-hook",
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
    tmp_path: Path, cli_runner: CliRunner, patterns: tuple[str, str]
) -> None:
    """--resume continues at the epoch after the saved one, with the state the first run left."""
    _train(cli_runner, patterns, tmp_path, experiment="flax-resume", epochs=2)
    (state,) = (tmp_path / "mlruns").rglob("training_state.tar.gz")
    result = _train(
        cli_runner,
        patterns,
        tmp_path,
        experiment="flax-resume",
        epochs=3,
        extra=["--resume", str(state), "--start-epoch", "2"],
    )
    assert "Ignoring --start-epoch 2: the resumed state continues at epoch 3." in result.output
    first, resumed = _runs("flax-resume")
    history = MlflowClient().get_metric_history(resumed.info.run_id, "loss")
    assert [metric.step for metric in history] == [3]
    # Training continued from the restored weights rather than from a fresh initialization.
    assert history[0].value < first.data.metrics["loss"]
