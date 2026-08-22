"""Pytest configuration shared by the logger tests."""

from typing import Any

import pytest

from structcast_model.base_trainer import SimpleDataProvider
from structcast_model.torch.trainer import TorchTracker, TorchTrainer


class _LearningRateLearner:
    """Minimal learner reporting one learning rate and one decay value, which a logger merges into the epoch."""

    @property
    def models(self) -> dict[str, Any]:
        """No models: the loggers under test never touch them."""
        return {}

    @property
    def optimizers(self) -> dict[str, Any]:
        """No optimizers: the trainer scan must handle an empty mapping."""
        return {}

    @property
    def optimizer_models(self) -> dict[str, list[str]]:
        """No pairing, there being no optimizer."""
        return {}

    @property
    def flow_functions(self) -> dict[str, Any]:
        """No separable flows: nothing here compiles or replicates a step."""
        return {}

    @property
    def learning_rates(self) -> dict[str, float]:
        """The rate the finished epoch is logged with."""
        return {"lr": 0.1}

    @property
    def weight_decays(self) -> dict[str, float]:
        """The optional decay metrics the finished epoch is logged with."""
        return {"opt_group0_weight_decay": 0.05}

    @property
    def steps(self) -> int:
        """No steps taken: the loggers under test never run the loop."""
        return 0

    @property
    def updates(self) -> int:
        """No updates applied: the loggers under test never run the loop."""
        return 0

    @property
    def has_updated(self) -> bool:
        """No step just finished, so nothing was applied."""
        return False

    def restore_counters(self, steps: int, updates: int) -> None:
        """Nothing to seed: the loggers under test never resume."""

    def training_step(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """No-op training step."""
        return {}

    def inference_step(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """No-op inference step."""
        return {}


@pytest.fixture
def epoch_info() -> TorchTrainer:
    """Return a trainer carrying one epoch of criteria and a learner reporting learning rates."""
    trainer = TorchTrainer(
        device="cpu",
        learner=_LearningRateLearner(),
        tracker=TorchTracker.from_criteria(["loss"], distributed=False),
        data=SimpleDataProvider(training_dataset=[]),
    )
    trainer.epoch = 1
    trainer.logs()["loss"] = 0.5
    return trainer
