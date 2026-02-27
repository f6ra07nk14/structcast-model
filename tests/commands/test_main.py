"""Unit tests for structcast_model.commands.main."""

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
