"""Unit tests for structcast_model.loggers.wandb."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from structcast_model.loggers.base import Logger
from structcast_model.loggers.wandb import WandbLogger
from structcast_model.torch.trainer import TorchTrainer
import torch


class _FakeApi:
    """Stand-in for `wandb.Api()`, serving one recorded training state as a run file."""

    def __init__(self, state: dict[str, Any]) -> None:
        """Hold the state the fake serves and start empty records of what was asked for."""
        self.state = state
        self.requested: list[str] = []
        self.filename = ""

    def run(self, path: str) -> _FakeApi:
        """Record the requested "<entity>/<project>/<run_id>" path."""
        self.requested.append(path)
        return self

    def file(self, name: str) -> _FakeApi:
        """Record the requested file name."""
        self.filename = name
        return self

    def download(self, root: str, replace: bool) -> None:
        """Write the recorded state where wandb would have downloaded the file."""
        torch.save(self.state, Path(root) / self.filename)


class _FakeWandb:
    """Stand-in for the wandb module, recording what the logger asks it to do."""

    def __init__(self, directory: Path, state: dict[str, Any] | None = None) -> None:
        """Create the fake module with a run directory, a downloadable state and empty call records."""
        self.run = SimpleNamespace(dir=str(directory))
        self.api = _FakeApi(state or {})
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

    def Api(self) -> _FakeApi:
        """Hand back the fake API client, under the name the logger calls: `wandb.Api()`."""
        return self.api


def test_wandb_logger_requires_the_optional_dependency() -> None:
    """Wandb is an optional extra: the failure must name the missing package, not a missing attribute."""
    with pytest.raises(ImportError, match="Tried to import 'wandb'"):
        WandbLogger(experiment="phase-two")


def test_wandb_logger_records_a_run_through_the_wandb_module(
    wandb_logger_with: Callable[[Any], Any], tmp_path: Path, epoch_info: TorchTrainer
) -> None:
    """The wandb backend must offer the same lifecycle and calls as MLflow, so the CLI can swap them."""
    fake = _FakeWandb(tmp_path)
    module = wandb_logger_with(fake)
    with module.WandbLogger(experiment="phase-two") as logger:
        logger.log_params({"epochs": 1})
        logger.log_dict({"epochs": 1}, "arguments.yaml")
        logger.log_artifact("config.yaml")
        logger.log_state_dict({"weight": torch.zeros(2)}, "training_state")
        logger.on_epoch_end(epoch_info)
    assert fake.projects == ["phase-two"]
    assert fake.finished == 1
    assert fake.params == {"epochs": 1}
    assert fake.saved == ["config.yaml"]
    assert fake.logged == [({"lr": 0.1, "opt_group0_weight_decay": 0.05, "loss": 0.5}, 1)]
    assert "epochs: 1" in (tmp_path / "arguments.yaml").read_text()
    assert torch.load(tmp_path / "training_state.pt")["weight"].tolist() == [0.0, 0.0]


def test_wandb_logger_fetches_a_training_state_from_a_run_reference(
    wandb_logger_with: Callable[[Any], Any], tmp_path: Path
) -> None:
    """--resume names the run that produced the state, so the logger must download it from its own run."""
    fake = _FakeWandb(tmp_path, {"weight": torch.ones(2)})
    module = wandb_logger_with(fake)
    state = module.WandbLogger(experiment="phase-two").fetch_training_state(
        "wandb://entity/project/run-id/training_state.pt"
    )
    assert state["weight"].tolist() == [1.0, 1.0]
    assert fake.api.requested == ["entity/project/run-id"]
    assert fake.api.filename == "training_state.pt"


def test_wandb_logger_fetches_a_training_state_from_a_local_path(
    wandb_logger_with: Callable[[Any], Any], tmp_path: Path
) -> None:
    """A state copied out of the run directory must still resume, so a plain path stays a valid reference."""
    module = wandb_logger_with(_FakeWandb(tmp_path))
    path = tmp_path / "training_state.pt"
    torch.save({"weight": torch.zeros(2)}, path)
    state = module.WandbLogger(experiment="phase-two").fetch_training_state(str(path))
    assert state["weight"].tolist() == [0.0, 0.0]


@pytest.mark.parametrize(("reference", "local"), [("missing.pt", True), ("runs:/run-id/training_state", False)])
def test_wandb_logger_rejects_a_reference_it_cannot_resolve(
    wandb_logger_with: Callable[[Any], Any], tmp_path: Path, reference: str, local: bool
) -> None:
    """A logger only knows its own scheme, so a foreign reference must fail naming the wandb form."""
    module = wandb_logger_with(_FakeWandb(tmp_path))
    with pytest.raises(ValueError, match="wandb://"):
        module.WandbLogger(experiment="phase-two").fetch_training_state(
            str(tmp_path / reference) if local else reference
        )


def test_wandb_logger_satisfies_the_logger_protocol(wandb_logger_with: Callable[[Any], Any], tmp_path: Path) -> None:
    """The CLI picks a backend by name, so a member missing from one of them breaks that choice."""
    module = wandb_logger_with(_FakeWandb(tmp_path))
    assert isinstance(module.WandbLogger(experiment="phase-two"), Logger)
