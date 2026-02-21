"""PyTorch related commands for the StructCast Model CLI application."""

from pathlib import Path
from typing import TYPE_CHECKING

from typer import Argument, Option, Typer

from structcast_model.commands.utils import dict_parser, reduce_dict
from structcast_model.utils.lazy_import import LazyModuleImporter

if TYPE_CHECKING:
    from structcast_model.builders import torch_builder
else:
    torch_builder = LazyModuleImporter("structcast_model.builders.torch_builder")


app = Typer(no_args_is_help=True)
creator = Typer(no_args_is_help=True)
app.add_typer(creator, name="create", help="Commands for creating PyTorch models and backward classes.")

param = Option(
    None,
    "--parameter",
    "-p",
    parser=dict_parser,
    help="Parameters to format the template configuration file with. Can be specified multiple times. "
    'Each parameter should be in the format of "key: {...}", where `key` is the name of the parameter group, '
    "and the value is a dictionary of keyword arguments for formatting the template. "
    'For example: --parameter "model: {input_size: 128, output_size: 10}" --parameter "optimizer: {lr: 0.001}"',
)
output_script_path = Option(None, "--output", "-o", help="Output script path (Python).")


@creator.command(name="model")
def create_model(
    cfg_path: Path = Argument(..., help="Path to the model configuration file."),
    output: Path | None = output_script_path,
    param: list[dict] | None = param,
    class_name: str = Option("Model", "--class-name", "-c", help="Name the model class."),
    structured_output: bool = Option(True, help="Enable structured output for the model."),
) -> None:
    """Create a PyTorch model from the given configuration file and parameters."""
    torch_builder.TorchBuilder.from_path(cfg_path)(
        parameters=reduce_dict(param),
        classname=class_name,
        forced_structured_output=structured_output,
    )(output)


@creator.command(name="backward")
def create_backward(
    cfg_path: Path = Argument(..., help="Path to the backward configuration file."),
    output: Path | None = output_script_path,
    param: list[dict] | None = param,
    class_name: str = Option("Backward", "--class-name", "-c", help="Name the backward class."),
) -> None:
    """Create a PyTorch backward class from the given configuration file and parameters."""
    torch_builder.TorchBackwardBuilder.from_path(cfg_path)(parameters=reduce_dict(param), classname=class_name)(output)
