"""Tests for the Keras training-state callbacks.

The payload is produced by real collaborators here -- a learner generated from a fixture
configuration, its Keras optimizer and the backend adapter that runs the steps -- because every part
of it only takes its final shape when those actually run.
"""

from __future__ import annotations

from collections.abc import Callable
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import keras
from structcast_model.base_trainer import SimpleDataProvider
from structcast_model.builders.keras import KerasBuilder, KerasLearnerBuilder
from structcast_model.keras.trainer import (
    KerasBestCriterion,
    KerasTracker,
    KerasTrainer,
    KerasTrainingStateSaver,
    initial_model,
)
from structcast_model.loggers.base import NullLogger
from tests import FIXTURES_DIR

CFG_DIR = FIXTURES_DIR / "cfg" / "keras"

X = np.asarray([[1.0, 0.5, -0.5, 2.0], [0.0, 1.0, 1.0, -1.0]], dtype="float32")
"""One fixed batch, as in the learner runtime tests."""

Y = np.asarray([[1.0, -1.0], [0.5, 0.25]], dtype="float32")


def _load(path: Path, name: str) -> Any:
    """Load a generated module by file path, the way a configuration does: not by import name."""
    spec = spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def make_learner(tmp_path_factory: pytest.TempPathFactory) -> Callable[..., Any]:
    """Return a factory building a fresh learner from the generated linear fixture."""
    directory = tmp_path_factory.mktemp("generated")
    KerasBuilder.from_path(CFG_DIR / "Linear.yaml")()(directory / "model.py")
    KerasLearnerBuilder.from_path(CFG_DIR / "LinearLearner.yaml")()(directory / "learner.py")
    model_type = _load(directory / "model.py", "generated_model").Model
    learner_type = _load(directory / "learner.py", "generated_learner").Learner

    def _build(seed: int = 0) -> Any:
        """Build a learner over a model initialized from *seed*, as the training CLI builds it."""
        keras.utils.set_random_seed(seed)
        return learner_type(model=initial_model(model_type(), {"x": (4,)}))

    return _build


class _RecordingLogger(NullLogger):
    """Logger recording the state dictionaries and best values the callbacks produce."""

    def __init__(self) -> None:
        """Start with nothing recorded."""
        self.states: list[tuple[dict[str, Any], str]] = []
        self.metrics: list[tuple[str, float, int]] = []

    def log_state_dict(self, states: Any, name: str) -> None:
        """Record the state dictionary and the name it is saved under."""
        self.states.append((dict(states), name))

    def log_metric(self, name: str, value: float, step: int) -> None:
        """Record a logged best value."""
        self.metrics.append((name, value, step))


def _trainer(learner: Any, callbacks: list[Any]) -> KerasTrainer:
    """Build a trainer running one batch per epoch through *learner*."""
    return KerasTrainer(
        learner=learner,
        tracker=KerasTracker.from_criteria(["loss"]),
        data=SimpleDataProvider(training_dataset=[{"x": X, "y": Y}]),
        callbacks=callbacks,
    )


def test_the_saver_writes_one_resumable_state_per_epoch(make_learner: Callable[..., Any]) -> None:
    """A run resumes from the weights, the optimizer state and the loop counters together.

    The payload keys are the torch ones -- an empty `grad_scalers` slot included, since Keras loss
    scaling lives inside the optimizer -- so a resume path reads the same shape whichever framework
    wrote the state, plus the backend a Keras resume refuses a mismatch on.
    """
    recorder = _RecordingLogger()
    saver = KerasTrainingStateSaver(logger=recorder, extra_meta={"seed": 42})

    _trainer(make_learner(), [saver]).fit(epochs=2)

    assert [name for _, name in recorder.states] == ["training_state", "training_state"]
    states, _ = recorder.states[-1]
    assert sorted(states) == ["grad_scalers", "meta", "models", "optimizers"]
    assert states["grad_scalers"] == {}
    assert states["meta"] == {
        "epoch": 2,
        "step": 2,
        "update": 2,
        "backend": keras.backend.backend(),
        "seed": 42,
    }
    assert list(states["models"]) == ["model"]
    assert list(states["optimizers"]) == ["optimizer"]


def test_the_best_criterion_saves_the_models_alone(make_learner: Callable[..., Any]) -> None:
    """Best-value weights are for inference, so they carry no optimizer state and no counters.

    The twin of the flax and torch artifacts, down to the nesting: a consumer reading best weights
    across frameworks must not have to strip a Keras-only wrapper key first.
    """
    recorder = _RecordingLogger()
    monitors = KerasBestCriterion.from_criteria([], ["loss"], ["loss"], logger=recorder)

    assert [(monitor.target, monitor.mode) for monitor in monitors] == [("loss", "min")]

    _trainer(make_learner(), list(monitors)).fit(epochs=1)

    assert [(name, step) for name, _, step in recorder.metrics] == [("best_loss", 1)]
    payload, name = recorder.states[0]
    assert name == "best_loss"
    assert list(payload) == ["model"]
