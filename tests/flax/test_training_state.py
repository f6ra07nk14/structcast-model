"""Tests for the Flax training-state callbacks and the resume helper.

The state a run is resumed from is produced by real collaborators here -- a learner generated from a
fixture configuration, its nnx optimizer and the single-device strategy -- because every part of the
payload only takes its final shape when those actually run.
"""

from __future__ import annotations

from collections.abc import Callable
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import Any
from warnings import catch_warnings, simplefilter

import jax
import jax.numpy as jnp
import pytest

from flax import nnx
from structcast_model.base_trainer import SimpleDataProvider
from structcast_model.builders.flax import FlaxBuilder, FlaxLearnerBuilder
from structcast_model.flax.distributed import FlaxDistributedStrategy
from structcast_model.flax.trainer import (
    FlaxBestCriterion,
    FlaxTracker,
    FlaxTrainer,
    FlaxTrainingStateSaver,
    restore_training_state,
)
from structcast_model.loggers.base import NullLogger
from tests import FIXTURES_DIR

CFG_DIR = FIXTURES_DIR / "cfg" / "flax"

X = jnp.asarray([[1.0, 0.5, -0.5, 2.0], [0.0, 1.0, 1.0, -1.0]])
"""One fixed batch, as in the learner runtime tests."""

Y = jnp.asarray([[1.0, -1.0], [0.5, 0.25]])


@pytest.fixture(autouse=True)
def _clear_mesh() -> Any:
    """Unset the mesh a strategy activated, so it does not leak into unrelated tests.

    ``jax.set_mesh`` is process-wide, as in ``tests/flax/test_distributed``.
    """
    yield
    jax.set_mesh(None)


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
    FlaxBuilder.from_path(CFG_DIR / "Linear.yaml")()(directory / "model.py")
    FlaxLearnerBuilder.from_path(CFG_DIR / "LinearLearner.yaml")()(directory / "learner.py")
    model_type = _load(directory / "model.py", "generated_model").Model
    learner_type = _load(directory / "learner.py", "generated_learner").Learner

    def _build(seed: int = 0) -> Any:
        """Build a learner over a model initialized from *seed*."""
        return learner_type(model_type(rngs=nnx.Rngs(seed)))

    return _build


@pytest.fixture
def strategy() -> FlaxDistributedStrategy:
    """The single-device strategy every test here produces and places states through."""
    return FlaxDistributedStrategy(preset="single")


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


class _StateLogger(NullLogger):
    """Logger handing back one prepared state, standing in for a fetch from a tracking service."""

    def __init__(self, state: dict[str, Any]) -> None:
        """Remember the state to hand back."""
        self.state = state

    # `Any`, not `dict[str, Any]`: `NullLogger` fetches nothing and narrows the return to None.
    def fetch_training_state(self, reference: str) -> Any:
        """Return the prepared state whatever the reference is."""
        return self.state


def _trainer(learner: Any, callbacks: list[Any]) -> FlaxTrainer:
    """Build a trainer running one batch per epoch through *learner*."""
    return FlaxTrainer(
        learner=learner,
        tracker=FlaxTracker.from_criteria(["loss"]),
        data=SimpleDataProvider(training_dataset=[{"x": X, "y": Y}]),
        callbacks=callbacks,
    )


def _saved_state(strategy: FlaxDistributedStrategy, learner: Any, **meta: Any) -> dict[str, Any]:
    """Produce the payload a saver would have written for *learner*, with *meta* merged in."""
    states = strategy.state_dict(dict(learner.models), learner.optimizers, learner.optimizer_models)
    states["grad_scalers"] = {}
    states["meta"] = {"epoch": 4, "step": 8, "update": 8, **meta}
    return states


def test_the_saver_writes_one_resumable_state_per_epoch(
    make_learner: Callable[..., Any], strategy: FlaxDistributedStrategy
) -> None:
    """A run resumes from the weights, the optimizer state and the loop counters together.

    The payload keys are the torch ones -- an empty `grad_scalers` slot included -- so a resume path
    reads the same shape whichever framework wrote the state.
    """
    recorder = _RecordingLogger()
    saver = FlaxTrainingStateSaver(logger=recorder, strategy=strategy, extra_meta={"seed": 42})

    _trainer(make_learner(), [saver]).fit(epochs=2)

    assert [name for _, name in recorder.states] == ["training_state", "training_state"]
    states, _ = recorder.states[-1]
    assert sorted(states) == ["grad_scalers", "meta", "models", "optimizers"]
    assert states["grad_scalers"] == {}
    assert states["meta"] == {"epoch": 2, "step": 2, "update": 2, "seed": 42}
    assert list(states["models"]) == ["model"]
    assert list(states["optimizers"]) == ["optimizer"]


def test_the_best_criterion_saves_the_models_alone(
    make_learner: Callable[..., Any], strategy: FlaxDistributedStrategy
) -> None:
    """Best-value weights are for inference, so they carry no optimizer state and no counters."""
    recorder = _RecordingLogger()
    monitors = FlaxBestCriterion.from_criteria([], ["loss"], ["loss"], recorder, strategy)

    assert [(monitor.target, monitor.mode) for monitor in monitors] == [("loss", "min")]

    _trainer(make_learner(), list(monitors)).fit(epochs=1)

    assert [(name, step) for name, _, step in recorder.metrics] == [("best_loss", 1)]
    payload, name = recorder.states[0]
    assert name == "best_loss"
    assert list(payload) == ["model"]


def test_restoring_continues_the_run_the_saved_state_left_off(
    make_learner: Callable[..., Any], strategy: FlaxDistributedStrategy
) -> None:
    """A resumed run must continue the old one: same weights, same optimizer step, next epoch."""
    recorder = _RecordingLogger()
    trained = make_learner()
    _trainer(trained, [FlaxTrainingStateSaver(logger=recorder, strategy=strategy)]).fit(epochs=2)
    saved, _ = recorder.states[-1]

    resumed = make_learner(seed=7)
    epoch = restore_training_state(
        resume="whatever",
        strategy=strategy,
        models=resumed.models,
        learner=resumed,
        start_epoch=1,
        logger=_StateLogger(saved),
    )

    assert epoch == 3
    assert jnp.array_equal(resumed.models["model"].fc.kernel[...], trained.models["model"].fc.kernel[...])
    assert nnx.state(resumed.optimizers["optimizer"]).step[...] == nnx.state(trained.optimizers["optimizer"]).step[...]


def test_the_saved_epoch_overrides_start_epoch_and_says_so(
    make_learner: Callable[..., Any], strategy: FlaxDistributedStrategy, capsys: pytest.CaptureFixture[str]
) -> None:
    """Resuming into a different epoch silently would misalign every schedule the run reports."""
    learner = make_learner()

    epoch = restore_training_state(
        resume="whatever",
        strategy=strategy,
        models=learner.models,
        learner=learner,
        start_epoch=9,
        logger=_StateLogger(_saved_state(strategy, learner)),
    )

    assert epoch == 5
    assert "Ignoring --start-epoch 9" in capsys.readouterr().out


def test_an_optimizer_rebuilt_differently_warns_naming_the_segment(
    make_learner: Callable[..., Any], strategy: FlaxDistributedStrategy
) -> None:
    """A rebuilt optimizer must be reported, not accepted in silence.

    optax rebuilds `tx` from configuration, so a swapped schedule restores cleanly and continues the
    new one from the old step count: the loader says so rather than refusing, per `docs/adr/0015`.
    """
    learner = make_learner()
    state = _saved_state(strategy, learner, optimizer_hashes={"optimizer": "saved-digest"})

    with pytest.warns(UserWarning, match='segment "optimizer"'):
        restore_training_state(
            resume="whatever",
            strategy=strategy,
            models=learner.models,
            learner=learner,
            start_epoch=1,
            logger=_StateLogger(state),
            optimizer_hashes={"optimizer": "rebuilt-digest"},
        )


def test_a_state_saved_from_another_configuration_warns(
    make_learner: Callable[..., Any], strategy: FlaxDistributedStrategy
) -> None:
    """Restoring arrays into another model, learner or shape configuration must be reported.

    The digest is recorded exactly so a resume can say so; a mismatch warns rather than refuses,
    because restoring what still lines up is how a run is continued with a widened input.
    """
    learner = make_learner()
    state = _saved_state(strategy, learner, config_hash="saved-digest")

    def _restore(config_hash: str) -> None:
        """Resume the prepared state as a run built from *config_hash* would."""
        restore_training_state(
            resume="whatever",
            strategy=strategy,
            models=learner.models,
            learner=learner,
            start_epoch=1,
            logger=_StateLogger(state),
            config_hash=config_hash,
        )

    with pytest.warns(UserWarning, match="different model, learner or shape configuration"):
        _restore("rebuilt-digest")

    with catch_warnings():
        simplefilter("error")
        _restore("saved-digest")


@pytest.mark.parametrize(
    "hashes", [None, {"optimizer": "saved-digest"}], ids=["state-without-hashes", "matching-hashes"]
)
def test_an_unchanged_optimizer_resumes_without_a_warning(
    make_learner: Callable[..., Any], strategy: FlaxDistributedStrategy, hashes: dict[str, str] | None
) -> None:
    """States written before the hashes existed, and runs that changed nothing, must stay quiet."""
    learner = make_learner()
    state = _saved_state(strategy, learner, **({} if hashes is None else {"optimizer_hashes": hashes}))

    with catch_warnings():
        simplefilter("error")
        restore_training_state(
            resume="whatever",
            strategy=strategy,
            models=learner.models,
            learner=learner,
            start_epoch=1,
            logger=_StateLogger(state),
            optimizer_hashes={"optimizer": "saved-digest"},
        )
