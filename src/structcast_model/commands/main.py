"""Main entry point for the StructCast Model CLI application."""

import re
from typing import TYPE_CHECKING, Any

from structcast.utils.base import dump_yaml, dump_yaml_to_string
from typer import Argument, Option, Typer

from structcast_model.commands import cmd_flax, cmd_keras, cmd_torch
from structcast_model.commands.shared_args import template_param_option
from structcast_model.commands.utils import reduce_dict

if TYPE_CHECKING:
    import jinja2

    from structcast_model.builders import schema
else:
    from structcast.utils.lazy_import import LazyModuleImporter

    jinja2 = LazyModuleImporter("jinja2")
    schema = LazyModuleImporter("structcast_model.builders.schema")


app = Typer(invoke_without_command=True, no_args_is_help=True, help="StructCast Model CLI application.")
app.add_typer(cmd_torch.app, name="torch", help="PyTorch related commands.")
app.add_typer(cmd_keras.app, name="keras", help="Keras related commands.")
app.add_typer(cmd_flax.app, name="flax", help="Flax nnx related commands.")


@app.command(name="format")
def format_template(
    cfg_path: str = Argument(..., help="Path to the template file."),
    output: str | None = Option(
        None,
        "--output",
        "-o",
        help="Path to save the formatted template. If not provided, the formatted template will be printed to stdout.",
    ),
    parameters: list[dict] | None = template_param_option(
        'For example: -p "default: {a: 1, b: 2}" -p "SHARED: {c: 3}" -p "extra: {d: 4}"'
    ),
) -> None:
    """Format a template configuration file with the provided parameters and print or save the result."""
    try:
        res: Any = schema.Template.from_path(cfg_path)(reduce_dict(parameters)).model_dump(mode="json")
    except jinja2.UndefinedError as exc:
        match = re.search(r"'([^']+)' is undefined", str(exc))
        var_hint = f" '{match.group(1)}'" if match else ""
        missing_hint = f' Provide it with: --parameter "{match.group(1)}: {{...}}"' if match else ""
        raise SystemExit(
            f"Template rendering failed: {exc}\nHint: missing template variable{var_hint}.{missing_hint}"
        ) from exc
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

if __name__ == "__main__":
    # `python -m structcast_model.commands.main` (the documented torchrun launch) imports this
    # module as __main__, which the lazy-import tail does not dispatch to the Typer app.
    app()
