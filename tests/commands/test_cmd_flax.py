"""Unit tests for structcast_model.commands.cmd_flax."""

from __future__ import annotations

from typing import Any

from typer import Typer
from typer.testing import CliRunner

from structcast_model.commands.cmd_flax import app

# ---------------------------------------------------------------------------
# Helper: access cmd_flax's real globals (bypasses LazySelectedImporter proxy)
# ---------------------------------------------------------------------------

_CMD_GLOBALS: dict[str, Any] = app.registered_commands[0].callback.__globals__


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
