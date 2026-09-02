"""Unit tests for structcast_model.commands.cmd_keras."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from importlib.util import module_from_spec, spec_from_file_location
import inspect
import logging
from math import isfinite
import os
from pathlib import Path
import subprocess
import sys
from types import ModuleType
from typing import Any

import mlflow
from mlflow.tracking import MlflowClient
import numpy as np
import pytest
from typer import Typer
from typer.testing import CliRunner

import keras
from structcast_model.builders.keras import KerasBuilder, KerasLearnerBuilder
from structcast_model.commands.cmd_keras import app
from structcast_model.keras.adapters import select_backend_adapter
from structcast_model.keras.distributed import KerasDistributedStrategy
from structcast_model.keras.trainer import KerasTrainer
from structcast_model.loggers.state_backends import KerasStateBackend
from structcast_model.utils.base import load_any
from tests import CFG_DIR, FIXTURES_DIR

LINEAR_CFG = str(FIXTURES_DIR / "cfg" / "keras" / "Linear.yaml")
LEARNER_CFG = FIXTURES_DIR / "cfg" / "keras" / "LinearLearner.yaml"
MODEL_CFG = str(CFG_DIR / "keras" / "models" / "ConvNeXtV2.yaml")

BACKEND = keras.backend.backend()
"""The backend this session resolved Keras on; every run below has to be told the same one."""

# ---------------------------------------------------------------------------
# app structure
# ---------------------------------------------------------------------------


def test_app_is_typer_instance() -> None:
    """The cmd_keras `app` must be a Typer instance."""
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


def test_time_help_documents_both_pattern_spellings_and_optimizer_constraint() -> None:
    """'time' must document both object pattern spellings and that "optimizer" cannot be passed to --compile.

    Both spellings are accepted by the instantiator, so documenting only one hides a valid input, and
    `--compile "{optimizer: adam}"` raises a TypeError because the command already passes `optimizer=None`.
    """
    command = next(cmd for cmd in app.registered_commands if cmd.name == "time")
    assert command.callback is not None
    params = inspect.signature(command.callback).parameters
    pattern_help = params["model_pattern"].default.help
    assert "[_obj_, {_addr_: my_package.MyModel, _file_: my_package.py}, {_call_: {...}}]" in pattern_help
    assert "[_obj_, [_addr_, my_package.MyModel, my_package.py], {_call_: {...}}]" in pattern_help
    assert '"optimizer" is always passed as None' in params["compile_pattern"].default.help


# ---------------------------------------------------------------------------
# 'create model' command — simple Linear layer
# ---------------------------------------------------------------------------


def test_create_model_linear(tmp_path: Any, cli_runner: CliRunner) -> None:
    """'create model' generates a script from a simple Dense config."""
    out = str(tmp_path / "model.py")
    result = cli_runner.invoke(app, ["create", "model", LINEAR_CFG, "--output", out])
    assert result.exit_code == 0, result.output
    text = (tmp_path / "model.py").read_text()
    assert "class Model" in text
    assert "Dense" in text


def test_create_model_linear_classname(tmp_path: Any, cli_runner: CliRunner) -> None:
    """'create model --classname' honours the custom class name."""
    out = str(tmp_path / "net.py")
    result = cli_runner.invoke(app, ["create", "model", LINEAR_CFG, "--classname", "MyDense", "--output", out])
    assert result.exit_code == 0, result.output
    assert "class MyDense" in (tmp_path / "net.py").read_text()


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
# 'time' command — simple Dense layer
# ---------------------------------------------------------------------------


def test_time_dense(cli_runner: CliRunner) -> None:
    """'time' measures inference on a simple keras Dense layer, saying which backend ran it.

    This is the one command that takes no --backend and inherits the ambient one (`docs/adr/0016`),
    so the number is only attributable if it names the backend that produced it.
    """
    pattern = "[_obj_, {_addr_: keras.layers.Dense}, {_call_: {units: 2}}]"
    result = cli_runner.invoke(
        app,
        ["time", pattern, "--shape", "inputs: [4]", "--warmup-runs", "1", "--times", "1", "--batch-size", "1"],
    )
    assert result.exit_code == 0, result.output
    assert f'Timing on the "{BACKEND}" Keras backend' in result.output
    assert "Average inference time" in result.output


# ---------------------------------------------------------------------------
# 'create learner' command
# ---------------------------------------------------------------------------


def _load(path: Path, name: str) -> ModuleType:
    """Load a generated module by file path, the way a configuration does: not by import name."""
    spec = spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_create_learner_writes_an_importable_class(tmp_path: Path, cli_runner: CliRunner) -> None:
    """'create learner' generates a learner module that imports and exposes the named class.

    The mixed-precision constants come with it, on the class: the training command reads them off
    the class it holds before it instantiates it, because the policy has to be set before the models
    are built.
    """
    out = tmp_path / "my_learner.py"
    result = cli_runner.invoke(
        app,
        [
            "create",
            "learner",
            str(LEARNER_CFG),
            "--classname",
            "MyLearner",
            "--parameter",
            "DEFAULT: {accumulate_gradients: 3}",
            "--output",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.output
    module = _load(out, "generated_keras_learner")
    assert hasattr(module, "MyLearner")
    assert module.MyLearner.MIXED_PRECISION is False
    assert module.MyLearner.MIXED_PRECISION_TYPE is None
    # The parameter reached the template: the window is the OPTIMIZER pattern's keyword, and the
    # written learner reads the optimizer's own counter back after each step (`docs/adr/0018`).
    text = out.read_text()
    assert "gradient_accumulation_steps=3" in text
    assert "current = int(keras.ops.convert_to_numpy(" in text


def test_create_learner_takes_no_backend_because_it_imports_no_keras(tmp_path: Path) -> None:
    """Writing a learner script never touches Keras, so the command must not demand a backend.

    A fresh interpreter without KERAS_BACKEND is the only honest check: this session has Keras
    imported and the variable set long before the test runs, so an accidental keras import here
    would be invisible.
    """
    out = tmp_path / "learner.py"
    script = (
        "from typer.testing import CliRunner; from structcast_model.commands.cmd_keras import app; "
        f"result = CliRunner().invoke(app, ['create', 'learner', {str(LEARNER_CFG)!r}, '--output', {str(out)!r}]); "
        "import sys; print(result.output, result.exception); "
        "raise SystemExit(result.exit_code or ('keras' in sys.modules) * 3)"
    )
    environment = {key: value for key, value in os.environ.items() if key != "KERAS_BACKEND"}

    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=False, env=environment
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "class Learner:" in out.read_text()


# ---------------------------------------------------------------------------
# 'train' command — end to end over the generated linear model and learner
# ---------------------------------------------------------------------------


def linear_batches(count: int = 3, size: int = 2) -> list[dict[str, Any]]:
    """Return a fixed dataset of *count* batches of *size* rows, so criteria depend on the seed alone.

    Public and addressable: the runs build it through an object pattern, as a real dataset is built.
    """
    return [
        {"x": np.full((size, 4), float(index + 1), dtype="float32"), "y": np.zeros((size, 2), dtype="float32")}
        for index in range(count)
    ]


DATASET = f"[_obj_, {{_addr_: {__name__}.linear_batches}}, _call_]"
"""Object pattern building the training dataset."""

VALIDATION_DATASET = f"[_obj_, {{_addr_: {__name__}.linear_batches}}, {{_call_: {{count: 2}}}}]"
"""Object pattern building a shorter validation dataset."""

POLICIES: list[str] = []
"""Dtype policies of the models `RecordingTrainer` trained; cleared by the test that reads them."""

CALLBACKS: list[str] = []
"""Callback types `RecordingTrainer` was wired with; cleared by the test that reads them."""


class RecordingTrainer(KerasTrainer):
    """A trainer recording what it was handed: the models' dtype policy, and its own callbacks."""

    def fit(self, *args: Any, **kwargs: Any) -> Any:
        """Note what the models compute in and what was wired in, then hand over to the shared loop."""
        POLICIES.extend(model.dtype_policy.name for model in self.learner.models.values())
        CALLBACKS.extend(type(callback).__name__ for callback in self.callbacks)
        return super().fit(*args, **kwargs)


@dataclass(kw_only=True)
class NonMainStrategy(KerasDistributedStrategy):
    """A strategy reporting a non-zero rank, as every torchrun worker but the first does.

    The rank is the one fact a worker differs by, so it is the only thing overridden: the state
    collection the saver drives stays the real strategy's, which is what makes the run reach the
    callbacks at all. Reaching a real second process would need a launcher, which a unit test has
    no business starting.
    """

    def __post_init__(self) -> None:
        """Validate the preset as the real strategy does, then take a worker's rank."""
        super().__post_init__()
        self._rank = 1


@pytest.fixture(scope="module")
def patterns(tmp_path_factory: pytest.TempPathFactory) -> tuple[str, str]:
    """Return the model and learner patterns of the generated linear fixture, built once."""
    directory = tmp_path_factory.mktemp("generated")
    KerasBuilder.from_path(FIXTURES_DIR / "cfg" / "keras" / "Linear.yaml")()(directory / "model.py")
    KerasLearnerBuilder.from_path(LEARNER_CFG)()(directory / "learner.py")
    return (
        f"model: [_obj_, {{_addr_: Model, _file_: {directory / 'model.py'}}}, _call_]",
        f"[_obj_, {{_addr_: Learner, _file_: {directory / 'learner.py'}}}]",
    )


def _train(
    cli_runner: CliRunner,
    patterns: tuple[str, str],
    tmp_path: Path,
    *,
    experiment: str,
    epochs: int = 2,
    extra: list[str] | None = None,
    shape: str | None = "x: [4]",
) -> Any:
    """Run `train` over the generated fixture, recording to an MLflow store under *tmp_path*.

    The Keras session is cleared first, because a run is a fresh process everywhere but here: layer
    names carry a per-process counter, and a state saved by one run only restores into another whose
    variables are named the same way.
    """
    keras.backend.clear_session()
    mlflow.set_tracking_uri(str(tmp_path / "mlruns"))
    model_pattern, learner_pattern = patterns
    result = cli_runner.invoke(
        app,
        [
            "train",
            model_pattern,
            "--backend",
            BACKEND,
            *([] if shape is None else ["--shape", shape]),
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
            *(extra or []),
        ],
    )
    assert result.exit_code == 0, f"{result.output}\n{result.exception}"
    return result


def test_train_end_to_end(tmp_path: Path, cli_runner: CliRunner, patterns: tuple[str, str]) -> None:
    """A full run trains, validates, prints its routed events and records everything it produced."""
    result = _train(cli_runner, patterns, tmp_path, experiment="keras-e2e")
    assert "Registered callbacks:" in result.output
    assert "KerasTrainingStateSaver" in result.output
    (run,) = mlflow.search_runs(experiment_names=["keras-e2e"], output_format="list")
    assert isfinite(run.data.metrics["loss"])
    assert isfinite(run.data.metrics["val_loss"])
    # The learning rate the generated learner reports, which is where a lost optimizer shows up.
    assert run.data.metrics["optimizer"] == pytest.approx(0.1)
    history = MlflowClient().get_metric_history(run.info.run_id, "loss")
    assert [metric.step for metric in history] == [1, 2]
    assert history[1].value < history[0].value
    artifacts = {artifact.path for artifact in MlflowClient().list_artifacts(run.info.run_id)}
    assert {"training_state.npz", "best_val_loss.npz", "arguments.yaml"} <= artifacts
    assert run.data.params["keras_backend"] == BACKEND


@pytest.fixture
def built_steps(monkeypatch: pytest.MonkeyPatch) -> list[tuple[dict[str, Any] | None, bool]]:
    """Watch how each run's learner builds its training step: the choice it read, and what it got.

    Reading `compile_kw` off the adapter after a run proves nothing about the seam, because the
    attribute reads the same whether it was set before the learner was built -- the only moment that
    decides anything -- or uselessly after it. The build call is that moment, so it is what is
    recorded, together with whether the step handed back is a traced one: a `tf.function` carries the
    `python_function` it was built from, an eager step does not.
    """
    adapter: Any = select_backend_adapter()
    built: list[tuple[dict[str, Any] | None, bool]] = []
    build = adapter.build_train_step

    def recording(segments: Any) -> Any:
        step = build(segments)
        choice = None if adapter.compile_kw is None else dict(adapter.compile_kw)
        built.append((choice, hasattr(step, "python_function")))
        return step

    monkeypatch.setattr(adapter, "build_train_step", recording)
    return built


@pytest.mark.skipif(BACKEND == "torch", reason="The torch backend builds no compiled step; it refuses the flag below.")
def test_train_compiles_the_learners_steps_only_while_compile_asks_for_it(
    tmp_path: Path,
    cli_runner: CliRunner,
    patterns: tuple[str, str],
    built_steps: list[tuple[dict[str, Any] | None, bool]],
) -> None:
    """--compile must reach the adapter before the learner builds its steps, and only for its own run.

    Three runs in one process, because no one of them shows the seam on its own: the criteria are
    identical compiled or eager, an assignment made after the learner was built reads back exactly
    like one made in time, and the two eager runs are what prove the compiled run's choice does not
    stay behind on an adapter cached for the whole process. `--compile null` and an omitted flag are
    the two spellings of off; on tensorflow the step itself is checked too, which is what a backend
    compiling unconditionally again -- the behaviour this replaced -- would fail on.
    """
    _train(cli_runner, patterns, tmp_path, experiment="keras-compiled", epochs=1, extra=["--compile", "true"])
    assert select_backend_adapter().compile_kw is None
    _train(cli_runner, patterns, tmp_path, experiment="keras-null", epochs=1, extra=["--compile", "null"])
    _train(cli_runner, patterns, tmp_path, experiment="keras-eager", epochs=1)

    assert [choice for choice, _ in built_steps] == [{}, None, None]
    if BACKEND == "tensorflow":
        assert [traced for _, traced in built_steps] == [True, False, False]
    (run,) = mlflow.search_runs(experiment_names=["keras-eager"], output_format="list")
    assert isfinite(run.data.metrics["loss"])


def test_train_refuses_none_as_a_spelling_of_an_eager_run(
    tmp_path: Path, cli_runner: CliRunner, patterns: tuple[str, str]
) -> None:
    """The bare word "none" was a local hack here, and YAML reads it as a string: a path that is not.

    Off is spelled the way YAML spells it -- omitted, `null`, `~` or `false`, all asserted eager
    above -- and the run this would otherwise have made is a full, successful, silently eager one:
    the arguments below are exactly the ones a passing run is given. Invoked directly rather than
    through the helper, which asserts success.
    """
    keras.backend.clear_session()
    mlflow.set_tracking_uri(str(tmp_path / "mlruns"))
    model_pattern, learner_pattern = patterns

    result = cli_runner.invoke(
        app,
        [
            "train",
            model_pattern,
            "--backend",
            BACKEND,
            "--shape",
            "x: [4]",
            "--learner",
            learner_pattern,
            "--training-dataset",
            DATASET,
            "--epochs",
            "1",
            "--lower-criterion",
            "loss",
            "--experiment",
            "keras-compile-none",
            "--ci",
            "--compile",
            "none",
        ],
    )

    assert result.exit_code != 0
    # The word itself, not only a non-zero exit: a typer that no longer knew `--compile` at all
    # would also exit non-zero, and this run has to fail on the value it was handed.
    assert "Path does not exist: none" in str(result.exception)
    assert not mlflow.search_runs(experiment_names=["keras-compile-none"], output_format="list")


@pytest.mark.skipif(BACKEND != "torch", reason="Only the torch backend has no compiler to refuse for.")
def test_train_refuses_compilation_on_the_backend_that_has_none(cli_runner: CliRunner) -> None:
    """A --compile the backend cannot honor must abort the run, not train eagerly and report success.

    Before the models and the learner are built, not down in the adapter: everything downstream
    would look like a healthy run, and only the missing speed would say otherwise. The model and the
    learner below name files that do not exist, so a refusal arriving with the first built step --
    where a hand-written learner still meets it -- would be drowned out by an import error instead.
    """
    result = cli_runner.invoke(
        app,
        [
            "train",
            "model: [_obj_, {_addr_: Model, _file_: /nonexistent.py}]",
            "--backend",
            "torch",
            "--learner",
            "[_obj_, {_addr_: nothing.Learner}]",
            "--training-dataset",
            DATASET,
            "--compile",
            "true",
        ],
    )

    assert result.exit_code != 0
    assert "builds no compiled step" in str(result.exception)


def test_train_records_the_shapes_the_models_declared_when_none_are_given(
    tmp_path: Path, cli_runner: CliRunner
) -> None:
    """--shape is optional, and the shapes a model declared have to reach the run's arguments anyway.

    A layer's INPUT_SHAPES is what it was traced with, and it does not survive the functional wrap
    `initial_model` applies -- so read back off the built models it would be lost, and the run would
    record `shapes: {}` while having been traced with something.
    """
    raw = {**load_any(LINEAR_CFG), "INPUT_SHAPES": {"x": [4]}}
    KerasBuilder(raw=raw, current_path=LINEAR_CFG)()(tmp_path / "model.py")
    KerasLearnerBuilder.from_path(LEARNER_CFG)()(tmp_path / "learner.py")
    patterns = (
        f"model: [_obj_, {{_addr_: Model, _file_: {tmp_path / 'model.py'}}}, _call_]",
        f"[_obj_, {{_addr_: Learner, _file_: {tmp_path / 'learner.py'}}}]",
    )

    _train(cli_runner, patterns, tmp_path, experiment="keras-declared-shapes", epochs=1, shape=None)

    (arguments,) = (tmp_path / "mlruns").rglob("arguments.yaml")
    assert load_any(arguments)["shapes"] == {"x": [4]}


def _learner_module(patterns: tuple[str, str]) -> ModuleType:
    """Load the generated learner module the *patterns* fixture wrote, to read its class constants."""
    return _load(Path(patterns[1].split("_file_: ", 1)[1].rstrip("}]")), "generated_learner_constants")


def _leaf_names(tree: Any) -> list[str]:
    """Return the name of every array in a path-keyed state tree, whatever Keras named the layers."""
    names: list[str] = []
    for key, value in tree.items():
        names.extend(_leaf_names(value) if isinstance(value, dict) else [key])
    return names


def test_train_records_the_backend_that_wrote_the_state(
    tmp_path: Path, cli_runner: CliRunner, patterns: tuple[str, str]
) -> None:
    """The saved state names its Keras backend, which is what a resume refuses a mismatch on.

    Nothing else in the archive identifies it -- the arrays are backend-portable -- so a state saved
    without the field would resume silently on a backend whose statistics and RNG stream differ.
    """
    _train(cli_runner, patterns, tmp_path, experiment="keras-state", epochs=1)

    (saved,) = (tmp_path / "mlruns").rglob("training_state.npz")
    state = KerasStateBackend().load(saved)
    meta = state["meta"]
    assert {key: meta[key] for key in ("epoch", "step", "update", "backend", "seed")} == {
        "epoch": 1,
        "step": 3,
        "update": 3,
        "backend": BACKEND,
        "seed": 42,
    }
    # The two digests a resume compares against: the configuration the run trains, and the optimizer
    # pattern each segment was built from -- the learner rebuilds the optimizer, so a swapped
    # schedule is only visible through the digest the generated learner emits.
    assert len(meta["config_hash"]) == 64
    assert meta["optimizer_hashes"] == {"optimizer": _learner_module(patterns).Learner.OPTIMIZER_HASHES["optimizer"]}
    assert state["grad_scalers"] == {}
    # Both halves travel, each under the name the run gave it and the paths Keras gave its
    # variables -- which the layer counter of the process decides, so only the leaves are asserted.
    assert sorted(_leaf_names(state["models"]["model"])) == ["bias", "kernel"]
    assert sorted(_leaf_names(state["optimizers"]["optimizer"])) == ["iteration", "learning_rate"]


def test_train_without_a_backend_names_both_ways_to_give_one(cli_runner: CliRunner) -> None:
    """There is no default backend, and the error has to say what to do about it.

    Keras would otherwise silently take the backend from ~/.keras/keras.json, deciding what the run
    computes on by a file nobody looked at.
    """
    result = cli_runner.invoke(app, ["train", "--help"], env={"KERAS_BACKEND": None})
    assert "KERAS_BACKEND" in result.output

    result = cli_runner.invoke(app, ["train", "model: {}"], env={"KERAS_BACKEND": None})

    assert result.exit_code == 2
    assert "--backend" in result.output
    assert "KERAS_BACKEND" in result.output


def test_train_refuses_a_backend_other_than_the_one_keras_already_runs(tmp_path: Path) -> None:
    """Keras resolves its backend once, at import: asking for another one has to fail, not be ignored.

    It has to fail on the already-imported Keras, before KERAS_BACKEND is set to a value that can
    never take effect -- a variable left pointing at a backend the process does not run is what any
    later reader, the run's own record included, would believe. Hence the assertions on the message
    of that first check and on the variable the failed command left behind.

    Only a fresh interpreter can exercise it -- this session imported Keras long before the test --
    so the check runs in a subprocess whose KERAS_BACKEND is pinned to the session's backend and
    whose command asks for a different one.
    """
    other = "jax" if BACKEND != "jax" else "tensorflow"
    script = (
        "import keras, os; from typer.testing import CliRunner; "
        "from structcast_model.commands.cmd_keras import app; "
        "result = CliRunner().invoke(app, ['train', 'model: [_obj_, {_addr_: builtins.dict}, _call_]', "
        f"'--backend', {other!r}, '--learner', '[_obj_, {{_addr_: builtins.dict}}]', "
        "'--training-dataset', '[_obj_, {_addr_: builtins.list}, _call_]']); "
        "print(type(result.exception).__name__, result.exception); "
        "print('KERAS_BACKEND=' + os.environ['KERAS_BACKEND'])"
    )
    environment = {**os.environ, "KERAS_BACKEND": BACKEND}

    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=False, env=environment
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "ValueError" in result.stdout
    assert BACKEND in result.stdout
    assert other in result.stdout
    assert "already running on" in result.stdout
    assert "fresh process" in result.stdout
    assert f"KERAS_BACKEND={BACKEND}" in result.stdout


@pytest.mark.parametrize(
    ("mixed_precision", "precision_type", "policy"),
    [(True, "bfloat16", "mixed_bfloat16"), ({}, "float16", "mixed_float16")],
    ids=["enabled", "empty-keyword-arguments"],
)
def test_train_sets_the_global_mixed_precision_policy_before_it_builds_the_models(
    tmp_path: Path, cli_runner: CliRunner, mixed_precision: Any, precision_type: str, policy: str
) -> None:
    """A learner declaring MIXED_PRECISION trains under the policy it names.

    The generated learner deliberately does not set it (`docs/adr/0016`): it receives models that
    are already built, and a policy set then would apply to nothing while looking like it had. So
    the models' own dtype policy -- not the global one, which any later call could have set -- is
    what proves the CLI set it at the only moment that works.

    An empty mapping is a mapping, and the adapter reads it as enabled -- it wraps every optimizer
    in a `LossScaleOptimizer` -- so a CLI that read it as disabled would loss-scale the gradients of
    a float32 model, with nothing to say so.
    """
    POLICIES.clear()
    raw = {**load_any(LEARNER_CFG), "MIXED_PRECISION": mixed_precision, "MIXED_PRECISION_TYPE": precision_type}
    KerasBuilder.from_path(FIXTURES_DIR / "cfg" / "keras" / "Linear.yaml")()(tmp_path / "model.py")
    KerasLearnerBuilder(raw=raw, current_path=str(LEARNER_CFG))()(tmp_path / "learner.py")
    patterns = (
        f"model: [_obj_, {{_addr_: Model, _file_: {tmp_path / 'model.py'}}}, _call_]",
        f"[_obj_, {{_addr_: Learner, _file_: {tmp_path / 'learner.py'}}}]",
    )
    try:
        _train(
            cli_runner,
            patterns,
            tmp_path,
            experiment=f"keras-mixed-{precision_type}",
            epochs=1,
            extra=["--trainer", f"[_obj_, {{_addr_: {__name__}.RecordingTrainer}}]"],
        )
    finally:
        # The policy is process-wide state the run leaves behind.
        keras.mixed_precision.set_global_policy("float32")

    assert POLICIES == [policy]


def test_train_starts_keras_on_the_backend_the_flag_names(tmp_path: Path) -> None:
    """The flag has to reach `KERAS_BACKEND` before anything imports Keras, not after.

    Set afterwards it would be ignored -- Keras reads the variable once -- and the run would train
    on whatever ~/.keras/keras.json happens to name while reporting the requested backend. The
    subprocess starts without KERAS_BACKEND so the flag is the only thing that can decide it, and
    the command is left to fail on its placeholder patterns afterwards: the import is what is under
    test.
    """
    script = (
        "from typer.testing import CliRunner; from structcast_model.commands.cmd_keras import app; "
        "CliRunner().invoke(app, ['train', 'model: [_obj_, {_addr_: builtins.dict}, _call_]', "
        "'--backend', 'jax', '--learner', '[_obj_, {_addr_: builtins.dict}]', "
        "'--training-dataset', '[_obj_, {_addr_: builtins.list}, _call_]']); "
        "import os, sys; print(sys.modules['keras'].backend.backend(), os.environ['KERAS_BACKEND'])"
    )
    environment = {key: value for key, value in os.environ.items() if key != "KERAS_BACKEND"}

    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=False, env=environment
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip().endswith("jax jax")


def test_train_caps_the_jax_memory_fraction_before_it_imports_keras(tmp_path: Path) -> None:
    """XLA reads its share of the GPU once, while JAX starts, so the cap has to be set before that.

    Set afterwards it would be ignored and the run would take XLA's default share of every device
    while reporting the cap it was given. The subprocess starts without either variable, so what it
    prints was written by the command, and it is left to fail on its placeholder patterns
    afterwards: the environment it prepared before importing Keras is what is under test.
    """
    script = (
        "from typer.testing import CliRunner; from structcast_model.commands.cmd_keras import app; "
        "CliRunner().invoke(app, ['train', 'model: [_obj_, {_addr_: builtins.dict}, _call_]', "
        "'--backend', 'jax', '--gpu-memory-fraction', '0.25', "
        "'--learner', '[_obj_, {_addr_: builtins.dict}]', "
        "'--training-dataset', '[_obj_, {_addr_: builtins.list}, _call_]']); "
        "import os, sys; print(os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'], "
        "os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'], sys.modules['keras'].backend.backend())"
    )
    scrubbed = {"KERAS_BACKEND", "XLA_PYTHON_CLIENT_MEM_FRACTION", "XLA_PYTHON_CLIENT_PREALLOCATE"}
    environment = {key: value for key, value in os.environ.items() if key not in scrubbed}

    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=False, env=environment
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip().endswith("0.25 false jax")


def test_train_asks_tensorflow_for_growth_because_it_has_no_fraction(tmp_path: Path) -> None:
    """TensorFlow has no fraction cap, so the option must not pretend it applied one.

    Growth on demand is the nearest thing it has -- the run takes what it uses instead of the whole
    device -- and it is read while TensorFlow starts, so it goes in before the import like the JAX
    one. A fraction variable TensorFlow never reads would leave the cap silently unapplied, so the
    subprocess also reports that none was written.
    """
    script = (
        "from typer.testing import CliRunner; from structcast_model.commands.cmd_keras import app; "
        "CliRunner().invoke(app, ['train', 'model: [_obj_, {_addr_: builtins.dict}, _call_]', "
        "'--backend', 'tensorflow', '--gpu-memory-fraction', '0.25', "
        "'--learner', '[_obj_, {_addr_: builtins.dict}]', "
        "'--training-dataset', '[_obj_, {_addr_: builtins.list}, _call_]']); "
        "import os, sys; print(os.environ['TF_FORCE_GPU_ALLOW_GROWTH'], "
        "'XLA_PYTHON_CLIENT_MEM_FRACTION' in os.environ, sys.modules['keras'].backend.backend())"
    )
    scrubbed = {"KERAS_BACKEND", "TF_FORCE_GPU_ALLOW_GROWTH", "XLA_PYTHON_CLIENT_MEM_FRACTION"}
    environment = {key: value for key, value in os.environ.items() if key not in scrubbed}

    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=False, env=environment
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip().endswith("true False tensorflow")


def test_train_caps_the_torch_backend_on_every_visible_device(tmp_path: Path) -> None:
    """The torch backend has no environment variable for this, so the cap is one call per device.

    It also has to come after the import, `torch.cuda` being what applies it. A cap put on device 0
    alone would leave every other rank of a torchrun launch uncapped, which is the whole point of
    asking for a fraction on a shared machine. The subprocess stands in for the GPUs this machine
    may not have -- the calls are what is under test, not CUDA -- and the command is left to fail on
    its placeholder patterns afterwards.
    """
    script = (
        "import torch; "
        "torch.cuda.is_available = lambda: True; "
        "torch.cuda.device_count = lambda: 2; "
        "capped = []; "
        "torch.cuda.set_per_process_memory_fraction = lambda fraction, device: capped.append((fraction, device)); "
        "from typer.testing import CliRunner; from structcast_model.commands.cmd_keras import app; "
        "CliRunner().invoke(app, ['train', 'model: [_obj_, {_addr_: builtins.dict}, _call_]', "
        "'--backend', 'torch', '--gpu-memory-fraction', '0.5', "
        "'--learner', '[_obj_, {_addr_: builtins.dict}]', "
        "'--training-dataset', '[_obj_, {_addr_: builtins.list}, _call_]']); "
        # Printed after the invoke: the runner captures whatever the command writes to stdout.
        "print('capped', capped)"
    )
    environment = {key: value for key, value in os.environ.items() if key != "KERAS_BACKEND"}

    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=False, env=environment
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip().endswith("capped [(0.5, 0), (0.5, 1)]")


# ---------------------------------------------------------------------------
# 'train' command — distributed strategy and resume
# ---------------------------------------------------------------------------


def _train_error(cli_runner: CliRunner, arguments: Sequence[str]) -> BaseException:
    """Invoke `train` with *arguments* and return the exception it failed with."""
    result = cli_runner.invoke(app, ["train", *arguments])
    assert result.exit_code != 0
    assert result.exception is not None, result.output
    return result.exception


def test_train_accepts_the_single_strategy_preset(
    tmp_path: Path, cli_runner: CliRunner, patterns: tuple[str, str]
) -> None:
    """The default preset must go through the strategy, not around it, or nothing else would."""
    _train(cli_runner, patterns, tmp_path, experiment="keras-single", epochs=1, extra=["--strategy", "single"])

    (run,) = mlflow.search_runs(experiment_names=["keras-single"], output_format="list")
    assert isfinite(run.data.metrics["loss"])


@pytest.mark.skipif(
    BACKEND == "torch", reason="torch data parallelism needs a launcher-provided process group; refused below."
)
def test_train_accepts_a_strategy_pattern(tmp_path: Path, cli_runner: CliRunner, patterns: tuple[str, str]) -> None:
    """--strategy also takes a path to an object pattern, here the shipped data-parallel template.

    On this lane's backend the template spans whatever single device the session has, so the run is
    a one-replica data-parallel run: what is under test is that the pattern is resolved, activated
    and trained through -- the per-backend mechanics have their own tests.
    """
    strategy = str(CFG_DIR / "keras" / "strategies" / "dp.yaml")
    _train(cli_runner, patterns, tmp_path, experiment="keras-dp", epochs=1, extra=["--strategy", strategy])

    (run,) = mlflow.search_runs(experiment_names=["keras-dp"], output_format="list")
    assert isfinite(run.data.metrics["loss"])


def test_train_refuses_a_memory_fraction_outside_the_unit_interval(
    cli_runner: CliRunner, patterns: tuple[str, str]
) -> None:
    """A fraction above 1 or at 0 caps nothing and would be applied as if it did."""
    error = _train_error(
        cli_runner,
        [
            patterns[0],
            "--backend",
            BACKEND,
            "--shape",
            "x: [4]",
            "--learner",
            patterns[1],
            "--training-dataset",
            DATASET,
            "--gpu-memory-fraction",
            "1.5",
        ],
    )

    assert isinstance(error, ValueError)
    assert "must be in (0, 1]" in str(error)


def test_train_leaves_the_run_and_the_display_to_the_main_rank(
    tmp_path: Path, cli_runner: CliRunner, patterns: tuple[str, str]
) -> None:
    """Under torchrun every rank runs this command, and only the first may own the run.

    A worker that built a real logger would open a tracking run of its own and write its own
    checkpoint next to the first rank's, which is what a multi-rank run on real hardware showed; a
    worker that kept the display would print a second progress bar over it. The saver and the
    best-criterion monitors stay on every rank, as they do in `scm torch train`: they read the state
    through the strategy, and their writes land in the null logger. See `docs/adr/0005`.
    """
    CALLBACKS.clear()

    result = _train(
        cli_runner,
        patterns,
        tmp_path,
        experiment="keras-worker",
        epochs=1,
        extra=[
            "--strategy",
            f"[_obj_, {{_addr_: {__name__}.NonMainStrategy}}]",
            "--trainer",
            f"[_obj_, {{_addr_: {__name__}.RecordingTrainer}}]",
        ],
    )

    assert CALLBACKS == ["NullLogger", "KerasTrainingStateSaver", "KerasBestCriterion", "KerasBestCriterion"]
    assert mlflow.get_experiment_by_name("keras-worker") is None
    assert "Registered callbacks:" not in result.output
    assert "Training dataset size:" not in result.output


@pytest.mark.skipif(BACKEND == "jax", reason="fsdp is supported on the jax backend.")
def test_train_refuses_the_fsdp_preset_on_a_backend_that_cannot_shard(
    tmp_path: Path, cli_runner: CliRunner, patterns: tuple[str, str]
) -> None:
    """The rejected cells must stop the command, with the reason, before it trains anything."""
    model_pattern, learner_pattern = patterns
    error = _train_error(
        cli_runner,
        [
            model_pattern,
            "--backend",
            BACKEND,
            "--shape",
            "x: [4]",
            "--learner",
            learner_pattern,
            "--training-dataset",
            DATASET,
            "--strategy",
            "fsdp",
        ],
    )

    assert isinstance(error, ValueError)
    assert 'The "fsdp" preset is not available' in str(error)


@pytest.mark.skipif(BACKEND != "torch", reason="Only the torch backend takes its ranks from a process group.")
def test_train_refuses_the_dp_preset_without_a_process_group(cli_runner: CliRunner, patterns: tuple[str, str]) -> None:
    """On torch the replicas are ranks, not devices, so a run outside a group has none to spread over.

    Training the whole batch on this one process instead would look like a successful data-parallel
    run while delivering none of it, so the command stops and names the launcher it needs.
    """
    error = _train_error(
        cli_runner,
        [
            patterns[0],
            "--backend",
            BACKEND,
            "--shape",
            "x: [4]",
            "--learner",
            patterns[1],
            "--training-dataset",
            DATASET,
            "--strategy",
            str(CFG_DIR / "keras" / "strategies" / "dp.yaml"),
        ],
    )

    assert isinstance(error, RuntimeError)
    assert "launch the command with torchrun" in str(error)


def test_train_resumes_from_a_saved_training_state(
    tmp_path: Path, cli_runner: CliRunner, patterns: tuple[str, str], caplog: pytest.LogCaptureFixture
) -> None:
    """--resume continues at the epoch after the saved one, with the state the first run left."""
    _train(cli_runner, patterns, tmp_path, experiment="keras-resume", epochs=2)
    (state,) = (tmp_path / "mlruns").rglob("training_state.npz")

    with caplog.at_level(logging.INFO, logger="structcast_model.keras.trainer"):
        _train(
            cli_runner,
            patterns,
            tmp_path,
            experiment="keras-resume",
            epochs=3,
            extra=["--resume", str(state), "--start-epoch", "2"],
        )

    assert "Ignoring --start-epoch 2: the resumed state continues at epoch 3." in caplog.text
    first, resumed = mlflow.search_runs(experiment_names=["keras-resume"], output_format="list")[::-1]
    history = MlflowClient().get_metric_history(resumed.info.run_id, "loss")
    assert [metric.step for metric in history] == [3]
    # Training continued from the restored weights rather than from a fresh initialization.
    assert history[0].value < first.data.metrics["loss"]


def test_train_refuses_a_state_written_on_another_keras_backend(
    tmp_path: Path, cli_runner: CliRunner, patterns: tuple[str, str]
) -> None:
    """A resume across backends is refused by the command, not only by the loader in isolation."""
    _train(cli_runner, patterns, tmp_path, experiment="keras-foreign", epochs=1)
    (saved,) = (tmp_path / "mlruns").rglob("training_state.npz")
    backend = KerasStateBackend()
    state = backend.load(saved)
    state["meta"]["backend"] = "jax" if BACKEND != "jax" else "torch"
    backend.save(state, tmp_path, "foreign")

    error = _train_error(
        cli_runner,
        [
            patterns[0],
            "--backend",
            BACKEND,
            "--shape",
            "x: [4]",
            "--learner",
            patterns[1],
            "--training-dataset",
            DATASET,
            "--resume",
            str(tmp_path / "foreign.npz"),
        ],
    )

    assert isinstance(error, ValueError)
    assert "Keras backend" in str(error)
