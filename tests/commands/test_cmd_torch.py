"""Unit tests for structcast_model.commands.cmd_torch."""

from collections.abc import Generator
from contextlib import contextmanager
from typing import Any
from unittest.mock import MagicMock

from typer import Typer
from typer.testing import CliRunner

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
    mock_torch_trainer.initial_model.return_value = (MagicMock(), {}, None)
    mock_ptflops = MagicMock()
    mock_ptflops.get_model_complexity_info.return_value = ("1.0 GMac", "1.0 M")
    with patch_cmd_globals(
        _instantiate=mock_instantiate,
        torch=_make_torch_mock(),
        torch_trainer=mock_torch_trainer,
        ptflops=mock_ptflops,
    ):
        assert cli_runner.invoke(app, ["ptflops", MODEL_PATTERN_ARG]).exit_code == 0
    mock_instantiate.assert_called_once()


def test_ptflops_prints_flops_and_params(cli_runner: CliRunner) -> None:
    """'ptflops' should print FLOPs and parameter counts to stdout."""
    mock_instantiate = MagicMock(return_value=MagicMock())
    mock_torch_trainer = MagicMock()
    mock_torch_trainer.get_torch_device.return_value = "cpu"
    mock_torch_trainer.create_torch_inputs.return_value = {}
    mock_torch_trainer.initial_model.return_value = (MagicMock(), {}, None)
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
    mock_torch_trainer.initial_model.return_value = (MagicMock(), {}, None)
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
    mock_torch_trainer.initial_model.return_value = (MagicMock(), {}, None)
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
    mock_torch_trainer.initial_model.return_value = (MagicMock(), {}, None)
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
