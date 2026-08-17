"""Unit tests for structcast_model.commands.cmd_flax."""

from __future__ import annotations

from typing import Any

from typer import Typer
from typer.testing import CliRunner

from structcast_model.commands.cmd_flax import app
from tests import CFG_DIR, FIXTURES_DIR

LINEAR_CFG = str(FIXTURES_DIR / "cfg" / "flax" / "Linear.yaml")
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
