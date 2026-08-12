"""Pytest configuration."""

from collections.abc import Callable, Generator
import importlib
import pathlib
import sys
from typing import Any

import pytest
import torch.distributed as dist
from typer.testing import CliRunner

WANDB_LOGGER = "structcast_model.torch.wandb_logger"


@pytest.fixture
def cli_runner() -> CliRunner:
    """Fixture that provides a Typer CliRunner for testing."""
    return CliRunner()


@pytest.fixture
def single_process_gloo(tmp_path: pathlib.Path) -> Generator[None, None, None]:
    """Initialize a single-process gloo distributed group for testing."""
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{tmp_path / 'dist_init'}",
        rank=0,
        world_size=1,
    )
    yield
    dist.destroy_process_group()


@pytest.fixture
def wandb_logger_with(monkeypatch: pytest.MonkeyPatch) -> Generator[Callable[[Any], Any], None, None]:
    """Return a factory publishing a fake `wandb` and handing back the logger module that sees it.

    `structcast_model.torch.wandb_logger` records the missing-wandb import failure once, while it is
    imported, so a fake injected into `sys.modules` later stays invisible until the module is
    reloaded. The teardown reloads it again without the fake, restoring the recorded failure for the
    tests that assert on it.
    """

    def _reload_with(fake: Any) -> Any:
        monkeypatch.setitem(sys.modules, "wandb", fake)
        return importlib.reload(importlib.import_module(WANDB_LOGGER))

    yield _reload_with
    monkeypatch.undo()
    if WANDB_LOGGER in sys.modules:
        importlib.reload(sys.modules[WANDB_LOGGER])
