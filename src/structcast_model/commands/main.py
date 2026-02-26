"""Main entry point for the StructCast Model CLI application."""

from typing import TYPE_CHECKING

from typer import Typer

from structcast_model.commands import cmd_torch

# allowed_modules = ["torch", "timm", "torchvision", "jax", "flax", "transformers", "keras", "tensorflow", "numpy"]
# configure_security(allowed_modules=deepcopy(DEFAULT_ALLOWED_MODULES) | dict.fromkeys(allowed_modules))

app = Typer(invoke_without_command=True, no_args_is_help=True, help="StructCast Model CLI application.")
app.add_typer(cmd_torch.app, name="torch", help="PyTorch related commands.")

__all__ = ["app"]


if not TYPE_CHECKING:
    import sys

    from structcast.utils.lazy_import import LazySelectedImporter

    sys.modules[__name__] = LazySelectedImporter(__name__, globals())
