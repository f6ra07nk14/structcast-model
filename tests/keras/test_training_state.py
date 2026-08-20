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
from warnings import catch_warnings, simplefilter

import numpy as np
import pytest

import keras
from structcast_model.base_trainer import SimpleDataProvider
from structcast_model.builders.keras import KerasBuilder, KerasLearnerBuilder
from structcast_model.keras.distributed import KerasDistributedStrategy
from structcast_model.keras.trainer import (
    KerasBestCriterion,
    KerasTracker,
    KerasTrainer,
    KerasTrainingStateSaver,
    initial_model,
    restore_training_state,
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
        """Build a learner over a model initialized from *seed*, as the training CLI builds it.

        The session is cleared first because a Keras state is keyed by `variable.path`, and a path
        carries the layer counter of the *process*: a second model built in the same one is named
        `model_1/dense_1/...` and would not line up with a state saved from the first. A resumed run
        is a fresh process, where the counter starts over -- which is what this reproduces.
        """
        keras.backend.clear_session()
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
    saver = KerasTrainingStateSaver(logger=recorder, strategy=KerasDistributedStrategy(), extra_meta={"seed": 42})

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
    monitors = KerasBestCriterion.from_criteria(
        [], ["loss"], ["loss"], logger=recorder, strategy=KerasDistributedStrategy()
    )

    assert [(monitor.target, monitor.mode) for monitor in monitors] == [("loss", "min")]

    _trainer(make_learner(), list(monitors)).fit(epochs=1)

    assert [(name, step) for name, _, step in recorder.metrics] == [("best_loss", 1)]
    payload, name = recorder.states[0]
    assert name == "best_loss"
    assert list(payload) == ["model"]


class _StateLogger(NullLogger):
    """Logger handing back one prepared state, standing in for a fetch from a tracking service."""

    def __init__(self, state: dict[str, Any]) -> None:
        """Remember the state to hand back."""
        self.state = state

    # `Any`, not `dict[str, Any]`: `NullLogger` fetches nothing and narrows the return to None.
    def fetch_training_state(self, reference: str) -> Any:
        """Return the prepared state whatever the reference is."""
        return self.state


def _saved_state(learner: Any, **meta: Any) -> dict[str, Any]:
    """Produce the payload a saver would have written for *learner*, with *meta* merged in."""
    states = KerasDistributedStrategy().state_dict(dict(learner.models), learner.optimizers)
    states["grad_scalers"] = {}
    states["meta"] = {"epoch": 4, "step": 8, "update": 8, "backend": keras.backend.backend(), **meta}
    return states


def _restore(learner: Any, state: dict[str, Any], **kwargs: Any) -> int:
    """Restore *state* into *learner* the way the training command does."""
    return restore_training_state(
        resume="whatever",
        strategy=KerasDistributedStrategy(),
        models=learner.models,
        learner=learner,
        start_epoch=kwargs.pop("start_epoch", 1),
        logger=_StateLogger(state),
        **kwargs,
    )


def test_restoring_continues_the_run_the_saved_state_left_off(make_learner: Callable[..., Any]) -> None:
    """A resumed run must continue the old one: same weights, same optimizer counter, next epoch.

    The counter matters as much as the weights: it is what the optimizer's schedule and its
    accumulation window are indexed by, so a run that restored the weights alone would take its
    next step as if it were the first.
    """
    recorder = _RecordingLogger()
    trained = make_learner()
    _trainer(trained, [KerasTrainingStateSaver(logger=recorder, strategy=KerasDistributedStrategy())]).fit(epochs=2)
    saved, _ = recorder.states[-1]

    resumed = make_learner(seed=7)
    before = float(keras.ops.convert_to_numpy(resumed.models["model"].variables[0].value)[0, 0])

    epoch = _restore(resumed, saved)

    assert epoch == 3
    assert before != float(keras.ops.convert_to_numpy(trained.models["model"].variables[0].value)[0, 0])
    for restored, original in zip(resumed.models["model"].variables, trained.models["model"].variables, strict=True):
        assert np.array_equal(keras.ops.convert_to_numpy(restored.value), keras.ops.convert_to_numpy(original.value))
    assert int(keras.ops.convert_to_numpy(resumed.optimizers["optimizer"].iterations)) == 2


def test_the_saved_epoch_overrides_start_epoch_and_says_so(
    make_learner: Callable[..., Any], capsys: pytest.CaptureFixture[str]
) -> None:
    """Resuming into a different epoch silently would misalign every schedule the run reports."""
    learner = make_learner()

    epoch = _restore(learner, _saved_state(learner), start_epoch=9)

    assert epoch == 5
    assert "Ignoring --start-epoch 9" in capsys.readouterr().out


def test_a_state_written_on_another_keras_backend_is_refused(make_learner: Callable[..., Any]) -> None:
    """Backends are not interchangeable mid-run, and the arrays alone cannot tell them apart.

    Normalization statistics and RNG trajectories are not verified equivalent across the Keras
    backends (`docs/adr/0016`), so a state written on one must not quietly continue on another --
    and nothing is assigned before the check, so a refused resume leaves the fresh run untouched.
    """
    learner = make_learner()
    active = keras.backend.backend()
    other = "jax" if active != "jax" else "torch"
    original = np.asarray(keras.ops.convert_to_numpy(learner.models["model"].variables[0].value)).copy()
    state = _saved_state(learner, backend=other)
    # Emptied, so a restore that got as far as assigning would fail on the missing paths instead:
    # the refusal has to come before anything is written.
    state["models"]["model"] = {}

    with pytest.raises(ValueError, match=f'saved on the "{other}" Keras backend') as error:
        _restore(learner, state)

    assert f'this run is on "{active}"' in str(error.value)
    assert np.array_equal(keras.ops.convert_to_numpy(learner.models["model"].variables[0].value), original)


def test_an_optimizer_rebuilt_differently_warns_naming_the_segment(make_learner: Callable[..., Any]) -> None:
    """A rebuilt optimizer must be reported, not accepted in silence.

    The learner rebuilds its optimizer from the configuration, so a swapped schedule restores
    cleanly and continues the new one from the old step count: the loader says so rather than
    refusing, per `docs/adr/0015`.
    """
    learner = make_learner()
    state = _saved_state(learner, optimizer_hashes={"optimizer": "saved-digest"})

    with pytest.warns(UserWarning, match='segment "optimizer"'):
        _restore(learner, state, optimizer_hashes={"optimizer": "rebuilt-digest"})


def test_a_state_saved_from_another_configuration_warns(make_learner: Callable[..., Any]) -> None:
    """Restoring arrays into another model, learner or shape configuration must be reported."""
    learner = make_learner()
    state = _saved_state(learner, config_hash="saved-digest")

    with pytest.warns(UserWarning, match="different model, learner or shape configuration"):
        _restore(learner, state, config_hash="rebuilt-digest")

    with catch_warnings():
        simplefilter("error")
        _restore(learner, state, config_hash="saved-digest")


@pytest.mark.parametrize(
    "hashes", [None, {"optimizer": "saved-digest"}], ids=["state-without-hashes", "matching-hashes"]
)
def test_an_unchanged_optimizer_resumes_without_a_warning(
    make_learner: Callable[..., Any], hashes: dict[str, str] | None
) -> None:
    """States written before the hashes existed, and runs that changed nothing, must stay quiet."""
    learner = make_learner()
    state = _saved_state(learner, **({} if hashes is None else {"optimizer_hashes": hashes}))

    with catch_warnings():
        simplefilter("error")
        _restore(learner, state, optimizer_hashes={"optimizer": "saved-digest"})


def test_a_state_missing_a_live_variable_is_refused(make_learner: Callable[..., Any]) -> None:
    """A silently partial restore trains half-resumed weights, so the missing path is named."""
    learner = make_learner()
    state = _saved_state(learner)
    state["models"]["model"] = {}

    with pytest.raises(ValueError, match="holds no value for"):
        _restore(learner, state)
