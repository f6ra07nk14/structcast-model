"""Unit tests for structcast_model.flax.trainer."""

from __future__ import annotations

from collections import OrderedDict
from functools import partial
import logging
from typing import Any

import jax
import jax.numpy as jnp
import pytest

from structcast_model.base_trainer import EVENTS, BaseInfo, SimpleDataProvider
from structcast_model.flax.trainer import (
    FlaxTracker,
    FlaxTrainer,
    create_jax_inputs,
    get_jax_device,
    get_jax_devices,
)


class _StubLearner:
    """A minimal Learner reporting one criterion, so the loop can run without a generated learner."""

    def __init__(self, loss: float = 1.0) -> None:
        """Report *loss* from every step."""
        self.loss = loss
        self.learning_rates: dict[str, float] = {}

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

    def update(self, step: int) -> bool:
        """Always signal an update."""
        return True

    def training_step(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """Report the fixed loss as a device array, the way a jitted step does."""
        return {"loss": jnp.asarray(self.loss)}

    def inference_step(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """Report twice the fixed loss, so a validation average cannot be confused with a training one."""
        return {"loss": jnp.asarray(2 * self.loss)}


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


def _trainer(tracker: Any, **kwargs: Any) -> FlaxTrainer:
    """Build a trainer over one training and one validation batch."""
    return FlaxTrainer(
        learner=_StubLearner(),
        tracker=tracker,
        data=SimpleDataProvider(training_dataset=[{"x": 1}], validation_dataset=[{"x": 2}]),
        **kwargs,
    )


# ---------------------------------------------------------------------------
# FlaxTracker
# ---------------------------------------------------------------------------


def test_flax_tracker_reports_the_running_mean_as_floats() -> None:
    """The trainer writes what the tracker returns into the history the loggers read.

    Floats, not arrays: `BaseTrainer.tracker` is typed to return them, `BestCriterion` compares
    them against a float infinity, and the loggers take `float` metric values.
    """
    tracker = FlaxTracker.from_criteria(["loss"])

    means = [tracker(loss=jnp.asarray(value)) for value in (1.0, 2.0, 6.0)]

    assert means == [{"loss": 1.0}, {"loss": 1.5}, {"loss": 3.0}]
    assert all(type(value) is float for mean in means for value in mean.values())


def test_flax_tracker_logs_are_empty_before_the_first_step() -> None:
    """A split that ran no step has no average to report, and reporting 0.0 would be a lie."""
    assert FlaxTracker.from_criteria(["loss"]).logs() == {}


def test_flax_tracker_resets_between_the_splits_of_one_epoch() -> None:
    """Without the reset, the validation average of an epoch would carry its training values.

    The tracker is routed by protocol, so this drives the real loop: the reset only happens if
    `on_training_begin` and `on_validation_begin` are the names the trainer dispatches.
    """
    tracker = FlaxTracker.from_criteria(["loss"])
    trainer = _trainer(tracker)

    trainer.fit(epochs=2)

    # The stub reports 1.0 while training and 2.0 while validating, one step each: a tracker that
    # never reset would report 1.5 for the validation split and 1.0 again for the second epoch.
    for epoch in (1, 2):
        assert trainer.logs(epoch)["loss"] == pytest.approx(1.0)
        assert trainer.logs(epoch)["val_loss"] == pytest.approx(2.0)


def test_flax_tracker_is_routed_into_the_two_reset_events_only() -> None:
    """The tracker takes part in the loop by protocol alone, exactly as the torch tracker does."""
    assert _trainer(FlaxTracker.from_criteria(["loss"])).describe() == {
        "on_training_begin": ["FlaxTracker"],
        "on_validation_begin": ["FlaxTracker"],
    }


# ---------------------------------------------------------------------------
# FlaxTrainer
# ---------------------------------------------------------------------------


def test_flax_trainer_dispatches_the_documented_event_order() -> None:
    """The Flax trainer inherits the loop, so its lifecycle must be the base trainer's, unchanged."""
    log: list[str] = []

    _trainer(FlaxTracker.from_criteria(["loss"]), callbacks=[_Recorder(log)]).fit(epochs=1)

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


def test_create_jax_inputs_from_int_tuple_returns_array() -> None:
    """A tuple of ints produces a bfloat16 JAX array with batch dimension 1, bfloat16 being the default dtype."""
    result = create_jax_inputs((3, 4))
    assert result.shape == (1, 3, 4)
    assert result.dtype == jnp.bfloat16


def test_create_jax_inputs_from_list_returns_list() -> None:
    """A list of shapes returns a list of JAX arrays."""
    result = create_jax_inputs([(3,), (4,)])
    assert isinstance(result, list)
    assert len(result) == 2
    assert all(getattr(item, "shape", None) is not None for item in result)


def test_create_jax_inputs_from_dict_returns_dict() -> None:
    """A dict of shapes returns a dict of JAX arrays."""
    result = create_jax_inputs({"image": (3, 4), "mask": (1, 4)})
    assert isinstance(result, dict)
    assert set(result.keys()) == {"image", "mask"}
    assert all(getattr(item, "shape", None) is not None for item in result.values())


def test_create_jax_inputs_invalid_shape_raises() -> None:
    """A non-shape scalar raises ValueError."""
    with pytest.raises(ValueError, match="Invalid tensor shape"):
        create_jax_inputs("not_a_shape")


def test_create_jax_inputs_custom_batch_size() -> None:
    """Custom batch_size is prepended to the shape."""
    result = create_jax_inputs((5,), batch_size=4)
    assert result.shape == (4, 5)


def test_create_jax_inputs_int_dtype_falls_back_to_zeros_with_warning(caplog: pytest.LogCaptureFixture) -> None:
    """An integer dtype without an initializer falls back to zeros, because random floats cannot be integers.

    The fallback is a guess about the caller's intent, so it must be reported.
    JAX truncates `int64` to `int32` unless `jax_enable_x64` is set, so only the kind of the dtype is asserted.
    """
    with caplog.at_level(logging.WARNING):
        result = create_jax_inputs({"_SHAPE_": [5], "_DTYPE_": "int64"})
    assert jnp.issubdtype(result.dtype, jnp.integer)
    assert jnp.array_equal(result, jnp.zeros((1, 5), dtype=result.dtype))
    assert "Falling back to zeros" in caplog.text


def test_create_jax_inputs_honours_explicit_initializer() -> None:
    """An explicit `_INIT_` address replaces the dtype-based default initializer."""
    result = create_jax_inputs({"_SHAPE_": [4], "_INIT_": "jax.numpy.ones"})
    assert jnp.array_equal(result, jnp.ones((1, 4), dtype=jnp.bfloat16))


# ---------------------------------------------------------------------------
# get_jax_devices
# ---------------------------------------------------------------------------


def test_get_jax_devices_returns_ordered_dict() -> None:
    """get_jax_devices returns an OrderedDict of JAX devices."""
    devices = get_jax_devices()
    assert isinstance(devices, OrderedDict)
    assert len(devices) > 0
    for key, dev in devices.items():
        assert isinstance(key, str)
        assert isinstance(dev, jax.Device)


def test_get_jax_devices_keys_match_platform_id() -> None:
    """Each key has the form 'platform:id'."""
    devices = get_jax_devices()
    for key, dev in devices.items():
        assert key == f"{dev.platform}:{dev.id}"


# ---------------------------------------------------------------------------
# get_jax_device
# ---------------------------------------------------------------------------


def test_get_jax_device_default() -> None:
    """get_jax_device with no arg returns the first available device."""
    device = get_jax_device()
    assert isinstance(device, jax.Device)
    first_key = next(iter(get_jax_devices()))
    assert device is get_jax_devices()[first_key]


def test_get_jax_device_explicit_valid() -> None:
    """get_jax_device returns the requested device when it exists."""
    devices = get_jax_devices()
    key = next(iter(devices))
    device = get_jax_device(key)
    assert device is devices[key]


def test_get_jax_device_invalid_raises() -> None:
    """get_jax_device raises ValueError for a non-existent device string."""
    with pytest.raises(ValueError, match="not available"):
        get_jax_device("nonexistent:99")
