"""Unit tests for structcast_model.commands.main."""

from collections.abc import Generator
from contextlib import contextmanager
from typing import Any
from unittest.mock import MagicMock

from jinja2 import UndefinedError
from structcast.utils.security import register_dir, unregister_dir
from typer.testing import CliRunner

from structcast_model.commands.main import app


def test_app_no_args_is_help(cli_runner: CliRunner) -> None:
    """Calling the app with no arguments should display help text (exit 0 or 2)."""
    result = cli_runner.invoke(app, [])
    assert result.exit_code == 2
    assert "torch" in result.output


def test_app_help(cli_runner: CliRunner) -> None:
    """Invoking the app with --help should exit with code 0."""
    result = cli_runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "torch" in result.output


def test_app_torch_help(cli_runner: CliRunner) -> None:
    """Invoking the 'torch' subgroup with --help should exit with code 0."""
    result = cli_runner.invoke(app, ["torch", "--help"])
    assert result.exit_code == 0
    for subcmd in ("create", "ptflops", "calflops"):
        assert subcmd in result.output


# ---------------------------------------------------------------------------
# 'format' command
# ---------------------------------------------------------------------------


def test_format_template_prints_output_to_stdout(tmp_path: Any, cli_runner: CliRunner) -> None:
    """'format' with no --output prints the formatted YAML to stdout."""
    register_dir(tmp_path)
    try:
        cfg = tmp_path / "tmpl.yaml"
        cfg.write_text("key: value\ncount: 42\n")
        result = cli_runner.invoke(app, ["format", str(cfg)])
        assert result.exit_code == 0, result.output
        assert "key" in result.output
    finally:
        unregister_dir(tmp_path)


def test_format_template_writes_to_output_file(tmp_path: Any, cli_runner: CliRunner) -> None:
    """'format' with --output writes the result to the specified file."""
    register_dir(tmp_path)
    try:
        cfg = tmp_path / "tmpl.yaml"
        cfg.write_text("x: 1\ny: 2\n")
        out_file = tmp_path / "out.yaml"
        result = cli_runner.invoke(app, ["format", str(cfg), "--output", str(out_file)])
        assert result.exit_code == 0, result.output
        assert out_file.exists()
        content = out_file.read_text()
        assert "x" in content
    finally:
        unregister_dir(tmp_path)


def test_format_template_with_parameters(tmp_path: Any, cli_runner: CliRunner) -> None:
    """'format' accepts --parameter flags and renders the template."""
    register_dir(tmp_path)
    try:
        cfg = tmp_path / "tmpl.yaml"
        cfg.write_text("value: 10\n")
        result = cli_runner.invoke(app, ["format", str(cfg), "--parameter", "default: {a: 1}"])
        assert result.exit_code == 0, result.output
    finally:
        unregister_dir(tmp_path)


def test_format_help_exits_zero(cli_runner: CliRunner) -> None:
    """'format --help' exits with code 0."""
    result = cli_runner.invoke(app, ["format", "--help"])
    assert result.exit_code == 0


def test_format_template_undefined_variable_gives_helpful_error(tmp_path: Any, cli_runner: CliRunner) -> None:
    """'format' exits non-zero with a helpful hint when Jinja2 raises UndefinedError."""
    cfg = tmp_path / "tmpl.yaml"
    cfg.write_text("key: value\n")

    @contextmanager
    def _patch_format_globals(**kwargs: Any) -> Generator[None, Any, None]:
        """Temporarily patch globals of the format_template callback."""
        for cmd in app.registered_commands:
            if cmd.name == "format" and cmd.callback is not None:
                real = cmd.callback.__globals__
                originals = {k: real.get(k) for k in kwargs}
                real.update(kwargs)
                try:
                    yield
                finally:
                    real.update(originals)
                return
        raise AssertionError("format command not found")

    mock_schema = MagicMock()
    mock_schema.Template.from_path.return_value.side_effect = UndefinedError("'missing_var' is undefined")

    with _patch_format_globals(schema=mock_schema):
        result = cli_runner.invoke(app, ["format", str(cfg)])

    assert result.exit_code != 0
    output = str(result.exception or "") + str(result.output or "")
    assert "missing_var" in output or "Template rendering failed" in output
