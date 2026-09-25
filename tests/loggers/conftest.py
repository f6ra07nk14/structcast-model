"""Pytest configuration shared by the logger tests."""

import pytest

from structcast_model.base_trainer import SimpleDataProvider
from structcast_model.torch.trainer import TorchTracker, TorchTrainer
from tests.fakes import CountingLearner


class _LearningRateLearner(CountingLearner):
    """Minimal learner reporting one learning rate and one decay value, which a logger merges into the epoch."""

    @property
    def learning_rates(self) -> dict[str, float]:
        """The rate the finished epoch is logged with."""
        return {"lr": 0.1}

    @property
    def weight_decays(self) -> dict[str, float]:
        """The optional decay metrics the finished epoch is logged with."""
        return {"opt_group0_weight_decay": 0.05}


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
