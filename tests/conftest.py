"""Pytest configuration."""

from collections.abc import Callable, Generator
import importlib
import os
import pathlib
import sys
from typing import Any

import pytest
from typer.testing import CliRunner

import torch.distributed as dist

WANDB_LOGGER = "structcast_model.loggers.wandb"

# MLflow 3.15 put the filesystem tracking backend into maintenance mode and refuses it unless this
# opt-out is set. The MLflow tests point at a temporary file store on purpose, so they exercise the
# real client without a server; production callers pick their own backend URI and are unaffected.
# Set here rather than in a fixture because the distributed tests spawn workers that inherit it.
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"


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

    `structcast_model.loggers.wandb` records the missing-wandb import failure once, while it is
    imported, so a fake injected into `sys.modules` later stays invisible until the module is
    reloaded. The teardown reloads it again without the fake, restoring the recorded failure for the
    tests that assert on it.
    """

    def _reload() -> Any:
        """Reload the logger module and re-point its package shim at the reloaded object.

        `structcast_model.loggers` is a `LazySelectedImporter`, which caches the submodule object it
        first resolved, while `importlib.reload` leaves a *new* object in `sys.modules`. Without the
        re-point, `loggers.wandb` -- the attribute the CLI reaches the logger through -- would keep
        serving the module resolved before the reload.
        """
        reloaded = importlib.reload(importlib.import_module(WANDB_LOGGER))
        package, _, attribute = WANDB_LOGGER.rpartition(".")
        setattr(importlib.import_module(package), attribute, reloaded)
        return reloaded

    def _reload_with(fake: Any) -> Any:
        monkeypatch.setitem(sys.modules, "wandb", fake)
        return _reload()

    yield _reload_with
    monkeypatch.undo()
    if WANDB_LOGGER in sys.modules:
        _reload()
