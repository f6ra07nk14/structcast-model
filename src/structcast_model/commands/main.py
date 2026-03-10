"""Main entry point for the StructCast Model CLI application."""

from typing import TYPE_CHECKING, Any

from structcast.utils.base import dump_yaml, dump_yaml_to_string
from typer import Argument, Option, Typer

from structcast_model.commands import cmd_torch
from structcast_model.commands.utils import dict_parser, reduce_dict

if TYPE_CHECKING:
    from structcast_model.builders import schema
else:
    from structcast.utils.lazy_import import LazyModuleImporter

    schema = LazyModuleImporter("structcast_model.builders.schema")


app = Typer(invoke_without_command=True, no_args_is_help=True, help="StructCast Model CLI application.")
app.add_typer(cmd_torch.app, name="torch", help="PyTorch related commands.")


@app.command(name="format")
def format_template(
    cfg_path: str = Argument(..., help="Path to the template file."),
    output: str | None = Option(
        None,
        "--output",
        "-o",
        help="Path to save the formatted template. If not provided, the formatted template will be printed to stdout.",
    ),
    parameters: list[dict] | None = Option(
        None,
        "--parameter",
        "-p",
        parser=dict_parser,
        help="Parameters to format the template configuration file with. "
        'Each parameter should be in the format of "key: {...}", where `key` is the name of the parameter group, '
        "and the value is a dictionary of keyword arguments for formatting the template. "
        'For example: -p "default: {a: 1, b: 2}" -p "SHARED: {c: 3}" -p "extra: {d: 4}"',
    ),
) -> None:
    """Format a template configuration file with the provided parameters and print or save the result."""
    res: Any = schema.Template.from_path(cfg_path)(reduce_dict(parameters)).model_dump(mode="json")
    raw = dump_yaml_to_string(res)
    if output is None:
        print(raw)
    else:
        with open(output, "w") as f:
            dump_yaml(res, f)


__all__ = ["app"]


if not TYPE_CHECKING:
    import sys

    from structcast.utils.lazy_import import LazySelectedImporter

    sys.modules[__name__] = LazySelectedImporter(__name__, globals())
