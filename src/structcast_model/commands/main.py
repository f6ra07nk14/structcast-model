"""Main entry point for the SMB2 CLI application."""

import typer

from structcast_model.commands import cmd_torch

app = typer.Typer(invoke_without_command=True, no_args_is_help=True, help="SMB2 Command Line Interface (CLI).")
app.add_typer(cmd_torch.app, name="torch", help="PyTorch related commands.")
