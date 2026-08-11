"""Unit tests for structcast_model.torch.wandb_logger."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from structcast_model.torch.logger import Logger
from structcast_model.torch.trainer import TorchTrainer
from structcast_model.torch.wandb_logger import WandbLogger
import torch


class _FakeWandb:
    """Stand-in for the wandb module, recording what the logger asks it to do."""

    def __init__(self, directory: Path) -> None:
        """Create the fake module with a run directory and empty call records."""
        self.run = SimpleNamespace(dir=str(directory))
        self.projects: list[str] = []
        self.finished = 0
        self.params: dict[str, Any] = {}
        self.logged: list[tuple[dict[str, Any], int]] = []
        self.saved: list[str] = []
        self.config = SimpleNamespace(update=self.params.update)

    def init(self, project: str) -> None:
        """Record the started project."""
        self.projects.append(project)

    def finish(self, exit_code: int = 0) -> None:
        """Record that the run was finished."""
        self.finished += 1

    def log(self, values: dict[str, Any], step: int) -> None:
        """Record logged metrics."""
        self.logged.append((values, step))

    def save(self, path: str) -> None:
        """Record a saved file."""
        self.saved.append(path)


def test_wandb_logger_requires_the_optional_dependency() -> None:
    """Wandb is an optional extra: the failure must name the missing package, not a missing attribute."""
    with pytest.raises(ImportError, match="Tried to import 'wandb'"):
        WandbLogger("phase-two")


def test_wandb_logger_records_a_run_through_the_wandb_module(
    wandb_logger_with: Callable[[Any], Any], tmp_path: Path, epoch_info: TorchTrainer
) -> None:
    """The wandb backend must offer the same lifecycle and calls as MLflow, so the CLI can swap them."""
    fake = _FakeWandb(tmp_path)
    module = wandb_logger_with(fake)
    with module.WandbLogger("phase-two") as logger:
        logger.log_params({"epochs": 1})
        logger.log_dict({"epochs": 1}, "arguments.yaml")
        logger.log_artifact("config.yaml")
        logger.log_state_dict({"weight": torch.zeros(2)}, "training_state")
        logger.on_epoch_end(epoch_info)
    assert fake.projects == ["phase-two"]
    assert fake.finished == 1
    assert fake.params == {"epochs": 1}
    assert fake.saved == ["config.yaml"]
    assert fake.logged == [({"lr": 0.1, "loss": 0.5}, 1)]
    assert "epochs: 1" in (tmp_path / "arguments.yaml").read_text()
    assert torch.load(tmp_path / "training_state.pt")["weight"].tolist() == [0.0, 0.0]


def test_wandb_logger_satisfies_the_logger_protocol(wandb_logger_with: Callable[[Any], Any], tmp_path: Path) -> None:
    """The CLI picks a backend by name, so a member missing from one of them breaks that choice."""
    module = wandb_logger_with(_FakeWandb(tmp_path))
    assert isinstance(module.WandbLogger("phase-two"), Logger)
