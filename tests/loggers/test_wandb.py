"""Unit tests for structcast_model.loggers.wandb."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import wandb

from structcast_model.loggers.base import Logger
from structcast_model.loggers.wandb import WandbLogger
from structcast_model.torch.trainer import TorchTrainer
import torch


@pytest.fixture
def wandb_offline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    """Point wandb at offline mode in a temp directory, so tests exercise the real SDK without network."""
    monkeypatch.setenv("WANDB_MODE", "offline")
    monkeypatch.setenv("WANDB_DIR", str(tmp_path))
    monkeypatch.setenv("WANDB_SILENT", "true")
    monkeypatch.setenv("WANDB_CONSOLE", "off")
    yield wandb
    if wandb.run is not None:
        wandb.finish()


def test_wandb_logger_run_lifecycle_and_content(wandb_offline: Any, tmp_path: Path, epoch_info: TorchTrainer) -> None:
    """The wandb backend must offer the same lifecycle and calls as MLflow, so the CLI can swap them."""
    artifact = tmp_path / "config.yaml"
    artifact.write_text("key: value\n")

    with WandbLogger(experiment="phase-two") as logger:
        assert wandb_offline.run is not None
        run_dir = Path(wandb_offline.run.dir)

        logger.log_params({"epochs": 1})
        assert wandb_offline.config["epochs"] == 1

        logger.log_dict({"epochs": 1}, "arguments.yaml")
        logger.log_artifact(str(artifact))
        logger.log_state_dict({"weight": torch.zeros(2)}, "training_state")
        logger.on_epoch_end(epoch_info)
        # wandb.run.summary is empty in offline mode (wandb>=0.24) because the internal process
        # that writes summary data does not flush synchronously; no epoch-metric assertion here.

    assert wandb_offline.run is None
    assert "epochs: 1" in (run_dir / "arguments.yaml").read_text()
    assert torch.load(run_dir / "training_state.pt", weights_only=True)["weight"].tolist() == [0.0, 0.0]
    assert any(run_dir.rglob("config.yaml"))


def test_wandb_logger_rejects_file_writes_outside_a_run() -> None:
    """Run-directory writes need an active run, so misuse must fail with a clear error, not a crash."""
    with pytest.raises(RuntimeError, match="No active wandb run"):
        WandbLogger(experiment="phase-two").log_dict({"epochs": 1}, "arguments.yaml")


def test_wandb_logger_fetches_a_training_state_from_a_local_path(tmp_path: Path) -> None:
    """A state copied out of the run directory must still resume, so a plain path stays a valid reference."""
    path = tmp_path / "training_state.pt"
    torch.save({"weight": torch.zeros(2)}, path)
    state = WandbLogger(experiment="phase-two").fetch_training_state(str(path))
    assert state["weight"].tolist() == [0.0, 0.0]


@pytest.mark.parametrize(("reference", "local"), [("missing.pt", True), ("runs:/run-id/training_state", False)])
def test_wandb_logger_rejects_a_reference_it_cannot_resolve(tmp_path: Path, reference: str, local: bool) -> None:
    """A logger only knows its own scheme, so a foreign reference must fail naming the wandb form."""
    with pytest.raises(ValueError, match="wandb://"):
        WandbLogger(experiment="phase-two").fetch_training_state(str(tmp_path / reference) if local else reference)


def test_wandb_logger_satisfies_the_logger_protocol() -> None:
    """The CLI picks a backend by name, so a member missing from one of them breaks that choice."""
    assert isinstance(WandbLogger(experiment="phase-two"), Logger)
