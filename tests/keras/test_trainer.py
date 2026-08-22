"""Unit tests for structcast_model.keras.trainer."""

from __future__ import annotations

from functools import partial
import logging
from typing import Any

import ml_dtypes
import numpy as np
import pytest

import keras
from structcast_model.base_trainer import EVENTS, BaseInfo, BestCriterion, SimpleDataProvider
from structcast_model.keras.trainer import (
    KerasBestCriterion,
    KerasTracker,
    KerasTrainer,
    create_keras_inputs,
    create_numpy_inputs,
    initial_model,
)
from structcast_model.keras.utils import get_keras_device


def test_create_numpy_inputs_from_int_tuple_returns_array() -> None:
    """A tuple of ints produces a bfloat16 NumPy array with batch dimension 1, bfloat16 being the default dtype."""
    result = create_numpy_inputs((3, 4))
    assert isinstance(result, np.ndarray)
    assert result.shape == (1, 3, 4)
    assert result.dtype == ml_dtypes.bfloat16


def test_create_numpy_inputs_from_dict_returns_dict() -> None:
    """A dict of shapes returns a dict of NumPy arrays."""
    result = create_numpy_inputs({"image": (3, 4), "mask": (1, 4)})
    assert isinstance(result, dict)
    assert set(result.keys()) == {"image", "mask"}
    assert all(isinstance(value, np.ndarray) for value in result.values())


def test_create_numpy_inputs_invalid_shape_raises() -> None:
    """A non-shape scalar raises ValueError."""
    with pytest.raises(ValueError, match="Invalid tensor shape"):
        create_numpy_inputs("not_a_shape")


def test_create_keras_inputs_from_dict_returns_dict() -> None:
    """A dict of shapes returns a dict of Keras input tensors."""
    result = create_keras_inputs({"x": (3,), "y": (2,)})
    assert isinstance(result, dict)
    assert set(result.keys()) == {"x", "y"}
    assert tuple(result["x"].shape) == (None, 3)
    assert tuple(result["y"].shape) == (None, 2)


def test_create_keras_inputs_invalid_shape_raises() -> None:
    """A non-shape scalar raises ValueError."""
    with pytest.raises(ValueError, match="Invalid tensor shape"):
        create_keras_inputs("not_a_shape")


def test_initial_model_returns_existing_model_when_shapes_is_none() -> None:
    """An existing Keras model is returned unchanged when no shapes are provided."""
    inputs = keras.Input(shape=(3,))
    outputs = keras.layers.Lambda(lambda x: x)(inputs)
    model = keras.Model(inputs=inputs, outputs=outputs)
    assert initial_model(model, shapes=None) is model


def test_initial_model_raises_for_layer_without_shapes() -> None:
    """A non-model Keras callable requires shapes for initialization."""
    with pytest.raises(ValueError, match="Input shapes are required"):
        initial_model(keras.layers.Dense(2), shapes=None)


def test_initial_model_builds_model_from_symbolic_inputs() -> None:
    """A Keras layer is wrapped into a built Keras model using symbolic inputs."""

    # keras ships no py.typed, so `Layer` is `Any` here; the src-side config relaxes both checks
    # for `structcast_model.keras.*` only.
    class AddLayer(keras.layers.Layer):  # type: ignore[misc, no-any-unimported]
        def call(self, x: Any, y: Any) -> Any:
            """Add two inputs."""
            return x + y

    model = initial_model(AddLayer(), {"x": (3,), "y": (3,)})
    assert isinstance(model, keras.Model)

    outputs = model(
        {
            "x": np.ones((1, 3), dtype=np.float32),
            "y": np.full((1, 3), 2.0, dtype=np.float32),
        }
    )
    np.testing.assert_allclose(np.array(outputs), np.full((1, 3), 3.0, dtype=np.float32))


# ---------------------------------------------------------------------------
# create_numpy_inputs — additional branches
# ---------------------------------------------------------------------------


def test_create_numpy_inputs_from_list_returns_list() -> None:
    """A list of shapes produces a list of NumPy arrays."""
    result = create_numpy_inputs([(3,), (4, 5)])
    assert isinstance(result, list)
    assert len(result) == 2
    assert result[0].shape == (1, 3)
    assert result[1].shape == (1, 4, 5)


def test_create_numpy_inputs_custom_batch_size() -> None:
    """Custom batch_size is respected."""
    result = create_numpy_inputs((2,), batch_size=8)
    assert result.shape == (8, 2)


def test_create_numpy_inputs_int_dtype_falls_back_to_zeros_with_warning(caplog: pytest.LogCaptureFixture) -> None:
    """An integer dtype without an initializer falls back to zeros, because random floats cannot be integers.

    The fallback is a guess about the caller's intent, so it must be reported.
    """
    with caplog.at_level(logging.WARNING):
        result = create_numpy_inputs({"_SHAPE_": [5], "_DTYPE_": "int64"})
    assert result.dtype == np.int64
    assert np.array_equal(result, np.zeros((1, 5), dtype=np.int64))
    assert "Falling back to zeros" in caplog.text


def test_create_numpy_inputs_honours_explicit_initializer() -> None:
    """An explicit `_INIT_` address replaces the dtype-based default initializer."""
    result = create_numpy_inputs({"_SHAPE_": [4], "_INIT_": "numpy.ones"})
    assert np.array_equal(result, np.ones((1, 4), dtype=ml_dtypes.bfloat16))


def test_create_numpy_inputs_rejects_non_callable_initializer() -> None:
    """A `_INIT_` address resolving to a non-callable is rejected, instead of failing later at call time."""
    with pytest.raises(TypeError, match="not callable as a tensor initializer"):
        create_numpy_inputs({"_SHAPE_": [4], "_INIT_": "numpy.pi"})


# ---------------------------------------------------------------------------
# create_keras_inputs — additional branches
# ---------------------------------------------------------------------------


def test_create_keras_inputs_from_tuple_returns_tensor() -> None:
    """A tuple of ints produces a single Keras Input tensor."""
    result = create_keras_inputs((3, 4))
    assert tuple(result.shape) == (None, 3, 4)


def test_create_keras_inputs_from_list_returns_list() -> None:
    """A list of shapes produces a list of Keras Input tensors."""
    result = create_keras_inputs([(3,), (4,)])
    assert isinstance(result, list)
    assert len(result) == 2
    assert tuple(result[0].shape) == (None, 3)
    assert tuple(result[1].shape) == (None, 4)


def test_create_keras_inputs_with_batch_size() -> None:
    """Batch size is attached to symbolic input when specified."""
    result = create_keras_inputs((5,), batch_size=4)
    assert tuple(result.shape) == (4, 5)


@pytest.mark.parametrize(
    ("shape", "expected"),
    [((5,), "bfloat16"), ({"_SHAPE_": [5], "_DTYPE_": "int32"}, "int32")],
)
def test_create_keras_inputs_uses_spec_dtype(shape: Any, expected: str) -> None:
    """The symbolic input carries the element type of the specification, so the traced model is built for it."""
    assert create_keras_inputs(shape).dtype == expected


# ---------------------------------------------------------------------------
# initial_model — list inputs
# ---------------------------------------------------------------------------


def test_initial_model_with_list_inputs() -> None:
    """A layer accepting positional args is wrapped via list shaped inputs."""

    # keras ships no py.typed, so `Layer` is `Any` here (see `AddLayer` above).
    class ConcatLayer(keras.layers.Layer):  # type: ignore[misc, no-any-unimported]
        def call(self, a: Any, b: Any) -> Any:
            """Concatenate two inputs."""
            return keras.ops.concatenate([a, b], axis=-1)

    model = initial_model(ConcatLayer(), [(3,), (2,)])
    assert isinstance(model, keras.Model)
    out = model([np.ones((1, 3), dtype=np.float32), np.ones((1, 2), dtype=np.float32)])
    assert np.array(out).shape == (1, 5)


# ---------------------------------------------------------------------------
# get_keras_device
# ---------------------------------------------------------------------------


def test_get_keras_device_default() -> None:
    """get_keras_device with no arg returns an available device in the gpu:N / cpu:N spelling."""
    device = get_keras_device()
    assert isinstance(device, str)
    assert ":" in device


def test_get_keras_device_explicit_valid() -> None:
    """get_keras_device returns the specified device when it exists; cpu:0 exists on every backend."""
    assert get_keras_device("cpu:0") == "cpu:0"


def test_get_keras_device_invalid_raises() -> None:
    """get_keras_device raises ValueError for a non-existent device."""
    with pytest.raises(ValueError, match="not available"):
        get_keras_device("nonexistent_device:99")


# ---------------------------------------------------------------------------
# KerasTracker
# ---------------------------------------------------------------------------


class _StubLearner:
    """A minimal Learner reporting one criterion, so the loop can run without a generated learner."""

    def __init__(self, loss: float = 1.0) -> None:
        """Report *loss* from every step."""
        self.loss = loss
        self.learning_rates: dict[str, float] = {}
        self.steps = 0
        self.updates = 0
        self.has_updated = False

    @property
    def models(self) -> dict[str, Any]:
        """No models: the tracker and the event routing are what these tests drive."""
        return {}

    @property
    def optimizers(self) -> dict[str, Any]:
        """No optimizers."""
        return {}

    @property
    def optimizer_models(self) -> dict[str, list[str]]:
        """No pairing."""
        return {}

    @property
    def flow_functions(self) -> dict[str, Any]:
        """No separable flows: these tests drive the loop, never a strategy."""
        return {}

    def restore_counters(self, steps: int, updates: int) -> None:
        """Seed the counters, the way a resume path would."""
        self.steps = steps
        self.updates = updates

    def training_step(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """Count the Step as one Update and report the current loss as a backend tensor."""
        self.steps += 1
        self.updates += 1
        self.has_updated = True
        return {"loss": keras.ops.convert_to_tensor(self.loss)}

    def inference_step(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """Report twice the current loss, so a validation average cannot be confused with a training one."""
        return {"loss": keras.ops.convert_to_tensor(2 * self.loss)}


class _Recorder:
    """Records the events it receives, implementing exactly the protocols it is asked to."""

    def __init__(self, log: list[str]) -> None:
        """Attach a recording method for every lifecycle event."""
        self.log = log
        for event in EVENTS:
            setattr(self, event, partial(self._record, event))

    def _record(self, event: str, info: BaseInfo) -> None:
        """Append the event name to the shared log."""
        self.log.append(event)


def _trainer(tracker: Any, learner: _StubLearner | None = None, **kwargs: Any) -> KerasTrainer:
    """Build a trainer over one training and one validation batch."""
    return KerasTrainer(
        learner=learner or _StubLearner(),
        tracker=tracker,
        data=SimpleDataProvider(training_dataset=[{"x": 1}], validation_dataset=[{"x": 2}]),
        **kwargs,
    )


def test_keras_tracker_reports_the_running_mean_as_floats() -> None:
    """The trainer writes what the tracker returns into the history the loggers read.

    Floats, not backend tensors: `BaseTrainer.tracker` is typed to return them, `BestCriterion`
    compares them against a float infinity, and the loggers take `float` metric values.
    """
    tracker = KerasTracker.from_criteria(["loss"])

    means = [tracker(loss=keras.ops.convert_to_tensor(value)) for value in (1.0, 2.0, 6.0)]

    assert means == [{"loss": 1.0}, {"loss": 1.5}, {"loss": 3.0}]
    assert all(type(value) is float for mean in means for value in mean.values())


def test_keras_tracker_from_criteria_keeps_the_requested_order() -> None:
    """The criteria come from the generated learner's outputs, which are read once as any iterable."""
    tracker = KerasTracker.from_criteria(iter(["loss", "accuracy"]))

    assert tracker.criteria == ("loss", "accuracy")
    assert sorted(tracker.sums) == ["accuracy", "loss"]


def test_keras_tracker_logs_are_empty_before_the_first_step() -> None:
    """A split that ran no step has no average to report, and reporting 0.0 would be a lie."""
    assert KerasTracker.from_criteria(["loss"]).logs() == {}


def test_keras_tracker_resets_between_the_splits_of_one_epoch() -> None:
    """Without the reset, the validation average of an epoch would carry its training values.

    The tracker is routed by protocol, so this drives the real loop: the reset only happens if
    `on_training_begin` and `on_validation_begin` are the names the trainer dispatches.
    """
    tracker = KerasTracker.from_criteria(["loss"])
    trainer = _trainer(tracker)

    trainer.fit(epochs=2)

    # The stub reports 1.0 while training and 2.0 while validating, one step each: a tracker that
    # never reset would report 1.5 for the validation split and 1.0 again for the second epoch.
    for epoch in (1, 2):
        assert trainer.logs(epoch)["loss"] == pytest.approx(1.0)
        assert trainer.logs(epoch)["val_loss"] == pytest.approx(2.0)


def test_keras_tracker_is_routed_into_the_two_reset_events_only() -> None:
    """The tracker takes part in the loop by protocol alone, exactly as the torch tracker does."""
    assert _trainer(KerasTracker.from_criteria(["loss"])).describe() == {
        "on_training_begin": ["KerasTracker"],
        "on_validation_begin": ["KerasTracker"],
    }


# ---------------------------------------------------------------------------
# KerasTrainer
# ---------------------------------------------------------------------------


def test_keras_trainer_dispatches_the_documented_event_order() -> None:
    """The Keras trainer inherits the loop, so its lifecycle must be the base trainer's, unchanged."""
    log: list[str] = []

    _trainer(KerasTracker.from_criteria(["loss"]), callbacks=[_Recorder(log)]).fit(epochs=1)

    assert log == [
        "on_epoch_begin",
        "on_training_begin",
        "on_training_step_begin",
        "on_update",
        "on_training_step_end",
        "on_training_end",
        "on_validation_begin",
        "on_validation_step_begin",
        "on_validation_step_end",
        "on_validation_end",
        "on_epoch_end",
    ]


# ---------------------------------------------------------------------------
# KerasBestCriterion
# ---------------------------------------------------------------------------


class _LossSchedule:
    """Gives the stub learner a different loss per epoch, so a run has exactly one best epoch."""

    def __init__(self, learner: _StubLearner, losses: dict[int, float]) -> None:
        """Drive *learner* through the loss of each epoch of *losses*."""
        self.learner = learner
        self.losses = losses

    def on_epoch_begin(self, info: BaseInfo) -> None:
        """Set the loss the coming epoch reports."""
        self.learner.loss = self.losses[info.epoch]


class _BestRecorder:
    """Records what the monitor announced after each checked epoch."""

    def __init__(self) -> None:
        """Start with nothing recorded."""
        self.seen: list[tuple[int, float]] = []

    def on_best(self, info: BaseInfo, best: BestCriterion) -> None:
        """Record the best value and the step that reached it."""
        self.seen.append((best.step, best.value))


def test_keras_best_criterion_keeps_the_lowest_value_and_the_step_that_reached_it() -> None:
    """The monitor is what a run's best weights and best metrics are keyed on.

    A later, worse epoch must not overwrite either, and the announcement must reach the registered
    `OnBest` participants on every checked epoch -- that is how the state savers learn whether the
    epoch that just ended is the one to write.
    """
    learner = _StubLearner()
    monitor = KerasBestCriterion(target="loss", mode="min")
    recorder = _BestRecorder()
    monitor.callbacks.append(recorder)
    trainer = _trainer(
        KerasTracker.from_criteria(["loss"]),
        learner=learner,
        callbacks=[monitor, _LossSchedule(learner, {1: 1.0, 2: 0.25, 3: 0.5})],
    )

    trainer.fit(epochs=3)

    assert (monitor.value, monitor.step) == (0.25, 2)
    assert recorder.seen == [(1, 1.0), (2, 0.25), (2, 0.25)]
