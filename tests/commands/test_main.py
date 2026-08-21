"""Unit tests for structcast_model.commands.main."""

from collections.abc import Callable, Iterator
import inspect
import subprocess
import sys
from typing import Any

import pytest
from structcast.utils.base import register_dir, unregister_dir
from typer import Typer
from typer.models import ArgumentInfo, OptionInfo
from typer.testing import CliRunner

from structcast_model.commands.main import app


def _iter_commands(typer_app: Typer, path: tuple[str, ...] = ()) -> Iterator[tuple[str, Callable[..., Any]]]:
    """Yield ``(command path, callback)`` for every command registered under `typer_app`, recursively."""
    if typer_app.registered_callback is not None and typer_app.registered_callback.callback is not None:
        yield " ".join((*path, "(callback)")), typer_app.registered_callback.callback
    for command in typer_app.registered_commands:
        assert command.callback is not None
        yield " ".join((*path, command.name or command.callback.__name__)), command.callback
    for group in typer_app.registered_groups:
        assert group.typer_instance is not None
        yield from _iter_commands(group.typer_instance, (*path, group.name or ""))


def test_every_cli_parameter_has_help() -> None:
    """Every option and argument of every command must document itself, or `--help` shows a blank description."""
    missing = [
        f"{command_path} / {param_name}"
        for command_path, callback in _iter_commands(app)
        for param_name, param in inspect.signature(callback).parameters.items()
        if isinstance(param.default, OptionInfo | ArgumentInfo) and not (param.default.help or "").strip()
    ]
    assert not missing, f"CLI parameters without help text: {missing}"


def test_short_flags_are_globally_unique() -> None:
    """One letter, one meaning: a short flag reused for a different long option would silently change meaning."""
    meanings: dict[str, str] = {}
    collisions: list[str] = []
    for command_path, callback in _iter_commands(app):
        for param_name, param in inspect.signature(callback).parameters.items():
            if not isinstance(param.default, OptionInfo):
                continue
            decls = list(param.default.param_decls or ())
            long = next((decl for decl in decls if decl.startswith("--")), param_name)
            shorts = [decl for decl in decls if decl.startswith("-") and not decl.startswith("--")]
            for short in shorts:
                if meanings.setdefault(short, long) != long:
                    collisions.append(f"{short}: {meanings[short]} vs {long} (at {command_path} / {param_name})")
    assert not collisions, f"Short flags meaning more than one long option: {collisions}"


@pytest.mark.parametrize("module", ["cmd_flax", "cmd_keras", "cmd_torch"])
def test_importing_a_command_module_imports_no_framework(module: str) -> None:
    """Building the CLI must cost no framework: `main` imports all three command modules at once.

    Every framework a command drives is reached through a lazy router or a `LazyModuleImporter`, so
    that `--help` and the framework-free commands stay fast and a run pays only for the framework it
    asked for. A single eager import -- a plain `from structcast_model.torch import ...` here, or one
    added to a shim on the way -- would silently make every invocation import torch, JAX and Keras.
    A subprocess is the only honest check: the test session has all three imported long before this
    runs.
    """
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            f"import structcast_model.commands.{module}; import sys; "
            "leaked = [name for name in ('torch', 'tensorflow', 'jax', 'keras', 'flax') if name in sys.modules]; "
            "raise SystemExit(f'imported: {leaked}' if leaked else 0)",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr


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
    register_dir(tmp_path)
    try:
        cfg = tmp_path / "tmpl.yaml"
        cfg.write_text('_jinja_: "{{ missing_var }}"\n')
        result = cli_runner.invoke(app, ["format", str(cfg)])
    finally:
        unregister_dir(tmp_path)

    assert result.exit_code != 0
    output = str(result.exception or "") + str(result.output or "")
    assert "missing_var" in output or "Template rendering failed" in output
