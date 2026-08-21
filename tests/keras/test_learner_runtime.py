"""Runtime tests for the learner modules `KerasLearnerBuilder` generates.

The generated script is exec'd from a file, the way a run would import it, and driven with real
Keras models on whichever backend `KERAS_BACKEND` selects (the conftest defaults it to tensorflow):
one backend-neutral learner has to train identically on all three, and everything asserted here --
who owns which variables, when an update lands, what an inference step touches -- is only decided
when the emitted code actually runs.

The loss assertions reuse one fixed batch. A moving batch hides a dead optimizer: the loss wanders
on its own and a step that updates nothing still produces a plausible-looking curve.
"""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest

import keras
from structcast_model.base_trainer import BaseInfo, Learner, SimpleDataProvider
from structcast_model.builders.keras import KerasBuilder, KerasLearnerBuilder
from structcast_model.keras.trainer import KerasTracker, KerasTrainer, initial_model
from structcast_model.utils.base import load_any
from tests import FIXTURES_DIR

CFG_DIR = FIXTURES_DIR / "cfg" / "keras"
MODEL_YAML = CFG_DIR / "Linear.yaml"
LEARNER_YAML = CFG_DIR / "LinearLearner.yaml"
SEGMENTS_YAML = CFG_DIR / "TwoSegmentLearner.yaml"

X = np.asarray([[1.0, 0.5, -0.5, 2.0], [0.0, 1.0, 1.0, -1.0]], dtype="float32")
"""One fixed batch, so a loss that moves can only be the optimizer's doing."""

Y = np.asarray([[1.0, -1.0], [0.5, 0.25]], dtype="float32")

BATCH = {"x": X, "y": Y}


def _load(path: Path, name: str) -> ModuleType:
    """Load a generated module by file path, the way a configuration does: not by import name."""
    spec = spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _model_type(tmp_path: Path) -> Any:
    """Generate the tiny linear layer the learner fixtures are written against."""
    KerasBuilder.from_path(MODEL_YAML)()(tmp_path / "model.py")
    return _load(tmp_path / "model.py", "generated_model").Model


def _models(tmp_path: Path, count: int = 1) -> list[Any]:
    """Build *count* seeded copies of the fixture model, the way the training CLI builds them."""
    model_type = _model_type(tmp_path)
    models = []
    for seed in range(count):
        keras.utils.set_random_seed(seed)
        models.append(initial_model(model_type(), {"x": (4,)}))
    return models


def _learner_type(tmp_path: Path, path: Path = LEARNER_YAML, name: str = "learner", **kwargs: Any) -> Any:
    """Generate a learner from a fixture configuration and return its class."""
    KerasLearnerBuilder.from_path(path)(**kwargs)(tmp_path / f"{name}.py")
    return _load(tmp_path / f"{name}.py", f"generated_{name}").Learner


def _values(variables: Any) -> list[np.ndarray]:
    """Read the host-side values of variables (or tensors), in the order they were given."""
    return [np.asarray(keras.ops.convert_to_numpy(getattr(v, "value", v))) for v in variables]


def _moved(before: list[np.ndarray], after: list[np.ndarray]) -> float:
    """Return the largest absolute change between two reads of the same variables."""
    return max(float(np.abs(a - b).max()) for a, b in zip(before, after, strict=True))


class _MovementRecorder:
    """Records, per training step the trainer runs, whether the watched variables moved."""

    def __init__(self, variables: Any) -> None:
        """Read the values the first step is compared against."""
        self.variables = variables
        self.previous = _values(variables)
        self.moved: list[bool] = []

    def on_training_step_end(self, info: BaseInfo) -> None:
        """Record whether the step that just ended moved the variables."""
        current = _values(self.variables)
        self.moved.append(_moved(self.previous, current) > 0.0)
        self.previous = current


def _sgd_step(model: Any, learning_rate: float) -> tuple[np.ndarray, np.ndarray]:
    """Return the kernel and bias one plain SGD step on the fixed batch should produce.

    The fixture model is a single `keras.layers.Dense`, so the expected update is analytic: it goes
    through neither the generated code, nor the optimizer, nor the adapter.
    """
    kernel, bias = _values(model.trainable_variables)
    difference = X @ kernel + bias - Y
    scale = 2.0 / difference.size
    return kernel - learning_rate * scale * (X.T @ difference), bias - learning_rate * scale * difference.sum(axis=0)


def test_generated_learner_satisfies_the_trainer_learner_protocol(tmp_path: Path) -> None:
    """The trainer drives learners through the protocol only, so a missing member breaks every run."""
    learner = _learner_type(tmp_path)(*_models(tmp_path))

    assert learner.outputs == ["loss"]
    assert learner.optimizer_models == {"optimizer": ["model"]}
    assert learner.learning_rates == {"optimizer": pytest.approx(0.1)}
    assert isinstance(learner, Learner)


def test_training_step_lowers_the_loss_it_reports_and_moves_the_weights(tmp_path: Path) -> None:
    """Three steps on one batch must bring the reported loss down and actually move a kernel.

    A learner that differentiated the wrong variables, or handed the adapter a segment whose
    variables belong to a copy of the model, still returns finite losses -- they just would not move.
    """
    model = _models(tmp_path)[0]
    learner = _learner_type(tmp_path)(model)
    before = _values(model.trainable_variables)

    losses = []
    for step in range(3):
        assert learner.update(step) is True
        losses.append(float(keras.ops.convert_to_numpy(learner.training_step(**BATCH)["loss"])))

    assert losses == sorted(losses, reverse=True)
    assert _moved(before, _values(model.trainable_variables)) > 0.0


def test_each_optimizer_moves_only_the_models_it_owns(tmp_path: Path) -> None:
    """Two segments, two variable lists: each optimizer applies its own rate to its own variables.

    The first optimizer owns two models and the second owns one; a variable list that leaked into
    the other segment would show up as the wrong rate on the wrong model.
    """
    models = _models(tmp_path, 3)
    expected = [_sgd_step(models[0], 0.1), _sgd_step(models[1], 0.1), _sgd_step(models[2], 0.01)]
    learner = _learner_type(tmp_path, SEGMENTS_YAML, name="segments")(*models)

    criteria = learner.training_step(**BATCH)

    assert sorted(criteria) == ["loss_ab", "loss_c"]
    for model, (kernel, bias) in zip(models, expected, strict=True):
        actual_kernel, actual_bias = _values(model.trainable_variables)
        assert np.allclose(actual_kernel, kernel, atol=1e-6)
        assert np.allclose(actual_bias, bias, atol=1e-6)


def test_accumulated_gradients_reach_the_optimizer_and_delay_the_update(tmp_path: Path) -> None:
    """`ACCUMULATE_GRADIENTS: 3` is the Keras optimizer's own window, and it has to be its own.

    Keras accumulates the gradients and gates the update inside `optimizer.apply`, so the variables
    may only move on every third step; a learner that dropped the setting would move them on every
    one, and `update` stays true throughout because the gate is no longer the learner's.
    """
    model = _models(tmp_path)[0]
    learner = _learner_type(tmp_path, name="accumulated", parameters={"DEFAULT": {"accumulate_gradients": 3}})(model)
    before = _values(model.trainable_variables)

    assert learner.optimizers["optimizer"].gradient_accumulation_steps == 3
    for step in range(2):
        assert learner.update(step) is True
        learner.training_step(**BATCH)
        assert _moved(before, _values(model.trainable_variables)) == 0.0

    assert learner.update(2) is True
    learner.training_step(**BATCH)

    assert _moved(before, _values(model.trainable_variables)) > 0.0


def test_accumulated_gradients_gate_the_variables_but_not_the_update_counter(tmp_path: Path) -> None:
    """Driven by the trainer, `ACCUMULATE_GRADIENTS: 2` moves the variables on every second step only.

    The counter is the documented divergence (`docs/adr/0016`): the gate belongs to the Keras
    optimizer, so `update()` stays true and `BaseInfo.update` counts steps rather than optimizer
    applications -- 4 over an epoch of 4 steps, where the torch and flax learners would report 2.
    Re-deriving the gate in the learner to make the counters agree is exactly what the ADR refuses,
    so this pins both halves at once: the real gate is the optimizer's, the counter is not it.
    """
    model = _models(tmp_path)[0]
    learner = _learner_type(tmp_path, name="gated", parameters={"DEFAULT": {"accumulate_gradients": 2}})(model)
    recorder = _MovementRecorder(model.trainable_variables)
    trainer = KerasTrainer(
        learner=learner,
        tracker=KerasTracker.from_criteria(["loss"]),
        data=SimpleDataProvider(training_dataset=[BATCH] * 4),
        callbacks=[recorder],
    )

    trainer.fit(epochs=1)

    assert recorder.moved == [False, True, False, True]
    assert trainer.update == 4


def test_the_optimizer_pattern_carries_the_clipping_that_replaces_clip(tmp_path: Path) -> None:
    """A Keras optimizer clips its own gradients, which is why the learner schema rejects `CLIP`.

    The rejection only helps if the substitute it names actually arrives: the OPTIMIZER pattern is
    emitted as written, so a builder that reformatted its keyword arguments would drop the clipping
    silently and the guidance would send readers to a dead end (`docs/adr/0016`).
    """
    raw = load_any(LEARNER_YAML)
    raw["LEARNERS"][0]["OPTIMIZER"] = [
        "_obj_",
        {"_addr_": "keras.optimizers.SGD"},
        {"_call_": {"learning_rate": 0.1, "global_clipnorm": 1.0}},
    ]
    KerasLearnerBuilder(raw=raw, current_path=str(LEARNER_YAML))()(tmp_path / "clipped.py")
    model = _models(tmp_path)[0]

    learner = _load(tmp_path / "clipped.py", "clipped_learner").Learner(model)

    assert learner.optimizers["optimizer"].global_clipnorm == pytest.approx(1.0)


def test_float16_mixed_precision_wraps_every_optimizer_and_still_trains(tmp_path: Path) -> None:
    """A float16 policy needs the loss scaled, which is the `LossScaleOptimizer` `prepare` wraps in.

    The wrapper replaces the optimizer object, so a learner that reported the one it constructed
    would hand the trainer -- and later the checkpoints -- an optimizer that never applied anything.
    The global policy is left alone on purpose: setting it is the training CLI's job, before the
    models are built, and the wrapping must not depend on it.
    """
    raw = {**load_any(LEARNER_YAML), "MIXED_PRECISION": True, "MIXED_PRECISION_TYPE": "float16"}
    KerasLearnerBuilder(raw=raw, current_path=str(LEARNER_YAML))()(tmp_path / "scaled.py")
    model = _models(tmp_path)[0]
    learner = _load(tmp_path / "scaled.py", "scaled_learner").Learner(model)

    assert isinstance(learner.optimizers["optimizer"], keras.optimizers.LossScaleOptimizer)
    # The rate is the inner optimizer's, which the wrapper delegates to; a NaN here would log NaN.
    assert learner.learning_rates == {"optimizer": pytest.approx(0.1)}

    losses = [float(keras.ops.convert_to_numpy(learner.training_step(**BATCH)["loss"])) for _ in range(3)]

    assert losses[-1] < losses[0]


def test_inference_step_reports_the_criteria_and_mutates_nothing(tmp_path: Path) -> None:
    """Evaluation must be repeatable and must not train.

    Neither the model variables nor the optimizer's own state may move: an inference step that
    stepped the optimizer would leave the weights alone on this fixture and only surface later, as a
    schedule that drifted for no reason.
    """
    model = _models(tmp_path)[0]
    learner = _learner_type(tmp_path)(model)
    optimizer = learner.optimizers["optimizer"]
    before, optimizer_before = _values(model.trainable_variables), _values(optimizer.variables)

    first = learner.inference_step(**BATCH)
    second = learner.inference_step(**BATCH)

    assert sorted(first) == ["loss"]
    assert float(keras.ops.convert_to_numpy(first["loss"])) == float(keras.ops.convert_to_numpy(second["loss"]))
    assert _moved(before, _values(model.trainable_variables)) == 0.0
    assert _moved(optimizer_before, _values(optimizer.variables)) == 0.0

    learner.training_step(**BATCH)

    # ... and a training step does move both, so the comparisons above can fail.
    assert _moved(before, _values(model.trainable_variables)) > 0.0
    assert _moved(optimizer_before, _values(optimizer.variables)) > 0.0


def test_inference_step_sees_what_the_last_training_step_wrote(tmp_path: Path) -> None:
    """The steps share the models, so validation reads the weights the last update produced.

    A compiled inference step that captured its variables as constants would keep reporting the
    criteria of the weights it was first traced on -- a validation curve that never moves.
    """
    model = _models(tmp_path)[0]
    learner = _learner_type(tmp_path)(model)

    before = float(keras.ops.convert_to_numpy(learner.inference_step(**BATCH)["loss"]))
    for _ in range(3):
        learner.training_step(**BATCH)
    after = float(keras.ops.convert_to_numpy(learner.inference_step(**BATCH)["loss"]))

    assert after < before
