"""Unit tests for structcast_model.loggers.mlflow."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import mlflow
import pytest

from structcast_model.loggers.base import Logger
from structcast_model.loggers.mlflow import MLflowLogger
from structcast_model.loggers.state_backends import FlaxStateBackend
from structcast_model.torch.trainer import TorchTrainer
import torch


@pytest.fixture
def mlflow_store(tmp_path: Path) -> Any:
    """Point MLflow at a temporary store, so the tests exercise the real client, and restore it after."""
    previous = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri((tmp_path / "mlruns").as_uri())
    yield mlflow
    mlflow.set_tracking_uri(previous)


def test_mlflow_logger_owns_the_run_and_logs_the_epoch_metrics(mlflow_store: Any, epoch_info: TorchTrainer) -> None:
    """No event fires once per fit, so the run lifecycle lives in the context manager, not a callback."""
    with MLflowLogger(experiment="phase-two") as logger:
        run_id = mlflow_store.active_run().info.run_id
        logger.log_params({"epochs": 1})
        logger.log_metric("best_loss", 0.25, step=1)
        logger.on_epoch_end(epoch_info)
    assert mlflow_store.active_run() is None
    run = mlflow_store.get_run(run_id)
    assert run.data.params == {"epochs": "1"}
    assert run.data.metrics == pytest.approx(
        {"best_loss": 0.25, "loss": 0.5, "lr": 0.1, "opt_group0_weight_decay": 0.05}
    )


def test_mlflow_logger_stores_dicts_states_and_files_as_artifacts(mlflow_store: Any, tmp_path: Path) -> None:
    """Arguments, model states and config files must survive the run for it to be reproducible."""
    artifact = tmp_path / "config.yaml"
    artifact.write_text("epochs: 1\n")
    with MLflowLogger(experiment="phase-two") as logger:
        run_id = mlflow_store.active_run().info.run_id
        logger.log_dict({"epochs": 1}, "arguments.yaml")
        logger.log_artifact(str(artifact))
        logger.log_state_dict({"weight": torch.zeros(2)}, "training_state")
    names = {item.path for item in mlflow_store.artifacts.list_artifacts(run_id=run_id)}
    # One artifact file named after the backend's format, where `mlflow.pytorch` wrote a directory.
    assert {"arguments.yaml", "config.yaml", "training_state.pt"} <= names


def test_mlflow_logger_fetches_a_training_state_from_a_run_uri(mlflow_store: Any) -> None:
    """--resume names the run that produced the state, so the logger must read back its own URI."""
    with MLflowLogger(experiment="phase-two") as logger:
        run_id = mlflow_store.active_run().info.run_id
        logger.log_state_dict({"weight": torch.zeros(2)}, "training_state")
    state = MLflowLogger(experiment="phase-two").fetch_training_state(f"runs:/{run_id}/training_state.pt")
    assert state["weight"].tolist() == [0.0, 0.0]


def test_mlflow_logger_reads_the_legacy_torch_state_directory(mlflow_store: Any, tmp_path: Path) -> None:
    """A checkpoint from before the backend seam must still resume.

    Those runs wrote `state_dict.pth` into a directory artifact, and a `--resume` pointing at one
    keeps working whatever format this run's own backend writes.
    """
    legacy = tmp_path / "state_dict.pth"
    torch.save({"weight": torch.ones(2)}, legacy)
    with MLflowLogger(experiment="phase-two"):
        run_id = mlflow_store.active_run().info.run_id
        mlflow_store.log_artifact(str(legacy), artifact_path="training_state")

    logger = MLflowLogger(experiment="phase-two", state_backend=FlaxStateBackend())
    assert logger.fetch_training_state(f"runs:/{run_id}/training_state")["weight"].tolist() == [1.0, 1.0]


def test_mlflow_logger_fetches_a_training_state_from_a_local_path(tmp_path: Path) -> None:
    """A state copied out of the tracker must still resume, so a plain path stays a valid reference."""
    path = tmp_path / "state_dict.pth"
    torch.save({"weight": torch.ones(2)}, path)
    assert MLflowLogger(experiment="phase-two").fetch_training_state(str(path))["weight"].tolist() == [1.0, 1.0]


def test_mlflow_logger_rejects_an_artifact_directory_holding_no_state(mlflow_store: Any) -> None:
    """A directory artifact that never held a state file must fail loudly, not hand torch.load a directory."""
    with MLflowLogger(experiment="phase-two") as logger:
        run_id = mlflow_store.active_run().info.run_id
        logger.log_dict({"a": 1}, "state_dir/config.json")
    with pytest.raises(ValueError, match='No "'):
        MLflowLogger(experiment="phase-two").fetch_training_state(f"runs:/{run_id}/state_dir")


@pytest.mark.parametrize(
    ("reference", "local"), [("missing.pth", True), ("wandb://entity/project/run/state.pt", False)]
)
def test_mlflow_logger_rejects_a_reference_it_cannot_resolve(tmp_path: Path, reference: str, local: bool) -> None:
    """A logger only knows its own scheme, so a foreign reference must fail naming the MLflow form."""
    with pytest.raises(ValueError, match="runs:/"):
        MLflowLogger(experiment="phase-two").fetch_training_state(str(tmp_path / reference) if local else reference)


def test_mlflow_logger_satisfies_the_logger_protocol() -> None:
    """The CLI picks a backend by name, so a member missing from one of them breaks that choice."""
    assert isinstance(MLflowLogger(experiment="phase-two"), Logger)
