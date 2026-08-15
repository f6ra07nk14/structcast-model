"""Unit tests for structcast_model.torch.logger."""

from __future__ import annotations

from structcast_model.torch.logger import Logger, NullLogger


def test_null_logger_fetches_no_training_state() -> None:
    """Only rank 0 owns a run: the other ranks must fetch nothing and take the state from the broadcast."""
    logger: Logger = NullLogger()
    assert logger.fetch_training_state("anything") is None
