"""Unit tests for structcast_model.base_trainer."""

from __future__ import annotations

from collections.abc import Iterator
from math import inf
from typing import Any

import pytest

from structcast_model.base_trainer import (
    GLOBAL_CALLBACKS,
    BaseInfo,
    BaseTrainer,
    BestCriterion,
    Callbacks,
    callbacks_session,
    get_dataset,
    get_dataset_size,
    invoke_callback,
)

# ---------------------------------------------------------------------------
# get_dataset
# ---------------------------------------------------------------------------


def test_get_dataset_returns_iterable_unchanged() -> None:
    """A plain iterable is returned as-is."""
    data = [{"x": 1}]
    assert get_dataset(data) is data


def test_get_dataset_calls_callable_and_returns_result() -> None:
    """A callable is invoked and its return value is returned."""
    data = [{"x": 1}]
    result = get_dataset(lambda: data)
    assert result is data


# ---------------------------------------------------------------------------
# get_dataset_size
# ---------------------------------------------------------------------------


def test_get_dataset_size_uses_len_when_available() -> None:
    """Uses __len__ when the dataset supports it."""
    data = [{"x": 1}, {"x": 2}, {"x": 3}]
    assert get_dataset_size(data) == 3


def test_get_dataset_size_iterates_when_no_len() -> None:
    """Falls back to iteration when __len__ is missing."""

    def gen() -> Iterator[dict[str, int]]:
        yield {"x": 1}
        yield {"x": 2}

    assert get_dataset_size(gen) == 2


def test_get_dataset_size_with_callable_producing_generator() -> None:
    """Works with a callable that produces a generator."""

    def make() -> Iterator[dict[str, int]]:
        def _g() -> Iterator[dict[str, int]]:
            yield {"x": 1}

        return _g()

    assert get_dataset_size(make) == 1


def test_get_dataset_size_via_callable_with_len() -> None:
    """Works with a callable that produces a list."""
    data = [{"x": 1}, {"x": 2}]
    assert get_dataset_size(lambda: data) == 2


# ---------------------------------------------------------------------------
# BaseInfo.logs
# ---------------------------------------------------------------------------


def test_base_info_logs_returns_dict_for_current_epoch() -> None:
    """logs() without arguments returns a dict keyed to the current epoch."""
    info = BaseInfo()
    info.epoch = 2
    logs = info.logs()
    assert isinstance(logs, dict)
    assert info.history[2] is logs


def test_base_info_logs_with_valid_epoch() -> None:
    """logs(epoch) returns the log for a known epoch."""
    info = BaseInfo()
    info.epoch = 1
    info.history[1] = {"loss": 0.5}
    assert info.logs(1) == {"loss": 0.5}


def test_base_info_logs_raises_key_error_for_unknown_epoch() -> None:
    """logs(epoch) raises KeyError when the epoch is not in history."""
    info = BaseInfo()
    with pytest.raises(KeyError, match="No logs found for key: 99"):
        info.logs(99)


# ---------------------------------------------------------------------------
# invoke_callback
# ---------------------------------------------------------------------------


def test_invoke_callback_calls_all_in_order() -> None:
    """All callbacks are invoked in insertion order."""
    info = BaseInfo()
    calls: list[int] = []
    invoke_callback([lambda i, **kw: calls.append(1), lambda i, **kw: calls.append(2)], info)
    assert calls == [1, 2]


def test_invoke_callback_forwards_model_kwargs() -> None:
    """Extra keyword arguments are forwarded to each callback."""
    info = BaseInfo()
    received: list[dict] = []
    invoke_callback([lambda i, **kw: received.append(kw)], info, model="my_model")
    assert received[0] == {"model": "my_model"}


def test_invoke_callback_empty_list_is_noop() -> None:
    """Empty callback list does not raise."""
    invoke_callback([], BaseInfo())


# ---------------------------------------------------------------------------
# Callbacks.__post_init__ with add_global_callbacks=True
# ---------------------------------------------------------------------------


def test_callbacks_post_init_copies_from_global_callbacks() -> None:
    """When add_global_callbacks=True, callbacks from GLOBAL_CALLBACKS are copied."""
    marker: list[str] = []

    def cb(i: Any, **kw: Any) -> None:
        marker.append("called")

    GLOBAL_CALLBACKS.on_epoch_end.append(cb)
    try:
        cbs = Callbacks(add_global_callbacks=True)
        assert cb in cbs.on_epoch_end
    finally:
        GLOBAL_CALLBACKS.on_epoch_end.remove(cb)


def test_callbacks_post_init_skips_global_when_disabled() -> None:
    """When add_global_callbacks=False, global callbacks are not copied."""

    def cb(i: Any, **kw: Any) -> None:
        pass

    GLOBAL_CALLBACKS.on_update.append(cb)
    try:
        cbs = Callbacks(add_global_callbacks=False)
        assert cb not in cbs.on_update
    finally:
        GLOBAL_CALLBACKS.on_update.remove(cb)


# ---------------------------------------------------------------------------
# Callbacks.clear
# ---------------------------------------------------------------------------


def test_callbacks_clear_empties_all_lists() -> None:
    """clear() resets every callback list to empty."""

    def cb(i: Any, **kw: Any) -> None:
        pass

    cbs = Callbacks(add_global_callbacks=False)
    cbs.on_update.append(cb)
    cbs.on_epoch_end.append(cb)
    cbs.on_training_begin.append(cb)
    cbs.clear()
    assert cbs.on_update == []
    assert cbs.on_epoch_end == []
    assert cbs.on_training_begin == []


def test_callbacks_clear_on_global_callbacks() -> None:
    """clear() on GLOBAL_CALLBACKS removes all previously registered callbacks."""

    def cb(i: Any, **kw: Any) -> None:
        pass

    GLOBAL_CALLBACKS.on_epoch_end.append(cb)
    GLOBAL_CALLBACKS.on_update.append(cb)
    GLOBAL_CALLBACKS.clear()
    assert cb not in GLOBAL_CALLBACKS.on_epoch_end
    assert cb not in GLOBAL_CALLBACKS.on_update


# ---------------------------------------------------------------------------
# callbacks_session
# ---------------------------------------------------------------------------


def test_callbacks_session_clears_on_entry() -> None:
    """callbacks_session() clears GLOBAL_CALLBACKS before yielding."""

    def cb(i: Any, **kw: Any) -> None:
        pass

    GLOBAL_CALLBACKS.on_epoch_end.append(cb)
    with callbacks_session():
        assert cb not in GLOBAL_CALLBACKS.on_epoch_end


def test_callbacks_session_clears_on_exit() -> None:
    """callbacks_session() clears GLOBAL_CALLBACKS when the block exits normally."""

    def cb(i: Any, **kw: Any) -> None:
        pass

    with callbacks_session():
        GLOBAL_CALLBACKS.on_epoch_end.append(cb)
    assert cb not in GLOBAL_CALLBACKS.on_epoch_end


def test_callbacks_session_clears_on_exception() -> None:
    """callbacks_session() clears GLOBAL_CALLBACKS even when an exception is raised."""

    def cb(i: Any, **kw: Any) -> None:
        pass

    def _run_with_error() -> None:
        with callbacks_session():
            GLOBAL_CALLBACKS.on_update.append(cb)
            raise RuntimeError("test")

    with pytest.raises(RuntimeError):
        _run_with_error()
    assert cb not in GLOBAL_CALLBACKS.on_update


def test_callbacks_session_isolates_multiple_runs() -> None:
    """Callbacks registered in one session are not seen by the next session."""
    registered: list[Any] = []

    def cb(i: Any, **kw: Any) -> None:
        pass

    with callbacks_session():
        GLOBAL_CALLBACKS.on_training_begin.append(cb)
        registered.append(cb)

    # Second session starts clean
    with callbacks_session():
        for item in registered:
            assert item not in GLOBAL_CALLBACKS.on_training_begin


# ---------------------------------------------------------------------------
# BaseTrainer helpers
# ---------------------------------------------------------------------------


class _FakeBackward:
    """Minimal Backward protocol implementation for tests."""

    def __init__(self, should_update: bool = True) -> None:
        self._should_update = should_update

    def update(self, step: int) -> bool:
        return self._should_update

    def __call__(self, **criteria: Any) -> None:
        pass


def _make_trainer(
    *,
    prefix: str = "",
    validation_prefix: str = "val_",
    should_update: bool = True,
    validation_step: Any = None,
    inference_wrapper: Any = None,
) -> BaseTrainer:
    def _forward(inputs: dict, **models: Any) -> dict:
        return {"loss": 0.5}

    def _tracker(**criteria: Any) -> dict[str, float]:
        return {"loss": float(criteria.get("loss", 0.0))}

    return BaseTrainer(
        training_step=_forward,
        backward=_FakeBackward(should_update),
        tracker=_tracker,
        training_prefix=prefix,
        validation_prefix=validation_prefix,
        validation_step=validation_step,
        inference_wrapper=inference_wrapper,
        add_global_callbacks=False,
    )


# ---------------------------------------------------------------------------
# BaseTrainer.train
# ---------------------------------------------------------------------------


def test_base_trainer_train_returns_logs_with_loss() -> None:
    """train() returns a dict containing tracked criteria."""
    trainer = _make_trainer()
    logs = trainer.train([{"x": 1}, {"x": 2}])
    assert "loss" in logs


def test_base_trainer_train_increments_step_and_update() -> None:
    """Step and update counters are incremented correctly."""
    trainer = _make_trainer()
    trainer.train([{"x": 1}, {"x": 2}])
    assert trainer.step == 2
    assert trainer.update == 2


def test_base_trainer_train_with_prefix_renames_log_keys() -> None:
    """training_prefix is prepended to all log keys."""
    trainer = _make_trainer(prefix="train_")
    logs = trainer.train([{"x": 1}])
    assert "train_loss" in logs
    assert "loss" not in logs


def test_base_trainer_train_no_update_when_backward_returns_false() -> None:
    """Update counter stays 0 when backward always returns False."""
    trainer = _make_trainer(should_update=False)
    trainer.train([{"x": 1}, {"x": 2}])
    assert trainer.update == 0


def test_base_trainer_train_invokes_all_callbacks() -> None:
    """Training lifecycle callbacks are fired in the correct order."""
    trainer = _make_trainer()
    events: list[str] = []
    trainer.on_training_begin.append(lambda i, **kw: events.append("begin"))
    trainer.on_training_end.append(lambda i, **kw: events.append("end"))
    trainer.on_training_step_begin.append(lambda i, **kw: events.append("step_begin"))
    trainer.on_training_step_end.append(lambda i, **kw: events.append("step_end"))
    trainer.on_update.append(lambda i, **kw: events.append("update"))
    trainer.train([{"x": 1}])
    assert events == ["begin", "step_begin", "update", "step_end", "end"]


def test_base_trainer_train_with_callable_dataset() -> None:
    """Callable dataset factories are supported in train()."""
    trainer = _make_trainer()
    data = [{"x": 1}]
    logs = trainer.train(lambda: data)
    assert "loss" in logs


def test_base_trainer_train_stores_logs_in_history() -> None:
    """Logs are stored in trainer.history under the current epoch."""
    trainer = _make_trainer()
    trainer.epoch = 1
    trainer.train([{"x": 1}])
    assert "loss" in trainer.history[1]


# ---------------------------------------------------------------------------
# BaseTrainer.evaluate
# ---------------------------------------------------------------------------


def test_base_trainer_evaluate_returns_empty_without_validation_step() -> None:
    """evaluate() returns {} when no validation_step is configured."""
    trainer = _make_trainer()
    result = trainer.evaluate([{"x": 1}])
    assert result == {}


def test_base_trainer_evaluate_returns_prefixed_val_logs() -> None:
    """evaluate() prepends validation_prefix to log keys."""

    def _val_forward(inputs: dict, **models: Any) -> dict:
        return {"loss": 0.3}

    trainer = _make_trainer(validation_step=_val_forward)
    logs = trainer.evaluate([{"x": 1}])
    assert "val_loss" in logs


def test_base_trainer_evaluate_calls_inference_wrapper_before_loop() -> None:
    """inference_wrapper is called once before the validation loop."""

    def _val_forward(inputs: dict, **models: Any) -> dict:
        return {"loss": 0.3}

    wrapper_called: list[bool] = []

    def _wrapper(info: Any, **models: Any) -> dict:
        wrapper_called.append(True)
        return models

    trainer = _make_trainer(validation_step=_val_forward, inference_wrapper=_wrapper)
    trainer.evaluate([{"x": 1}])
    assert wrapper_called


def test_base_trainer_evaluate_invokes_callbacks() -> None:
    """Validation lifecycle callbacks are fired in the correct order."""

    def _val_forward(inputs: dict, **models: Any) -> dict:
        return {"loss": 0.3}

    trainer = _make_trainer(validation_step=_val_forward)
    events: list[str] = []
    trainer.on_validation_begin.append(lambda i, **kw: events.append("begin"))
    trainer.on_validation_end.append(lambda i, **kw: events.append("end"))
    trainer.on_validation_step_begin.append(lambda i, **kw: events.append("step_begin"))
    trainer.on_validation_step_end.append(lambda i, **kw: events.append("step_end"))
    trainer.evaluate([{"x": 1}])
    assert events == ["begin", "step_begin", "step_end", "end"]


def test_base_trainer_evaluate_with_callable_dataset() -> None:
    """Callable dataset factories are supported in evaluate()."""

    def _val_forward(inputs: dict, **models: Any) -> dict:
        return {"loss": 0.3}

    trainer = _make_trainer(validation_step=_val_forward)
    data = [{"x": 1}]
    logs = trainer.evaluate(lambda: data)
    assert "val_loss" in logs


# ---------------------------------------------------------------------------
# BaseTrainer.fit
# ---------------------------------------------------------------------------


def test_base_trainer_fit_runs_correct_number_of_epochs() -> None:
    """fit() runs exactly the requested number of epochs."""
    trainer = _make_trainer()
    history = trainer.fit(epochs=3, training_dataset=[{"x": 1}])
    assert list(history.keys()) == [1, 2, 3]


def test_base_trainer_fit_with_validation_dataset() -> None:
    """fit() calls evaluate each epoch when validation_dataset is given."""

    def _val_forward(inputs: dict, **models: Any) -> dict:
        return {"loss": 0.3}

    trainer = _make_trainer(validation_step=_val_forward)
    trainer.fit(epochs=2, training_dataset=[{"x": 1}], validation_dataset=[{"x": 2}])
    assert "val_loss" in trainer.history[2]


def test_base_trainer_fit_respects_validation_frequency() -> None:
    """Validation only runs at multiples of validation_frequency."""

    def _val_forward(inputs: dict, **models: Any) -> dict:
        return {"loss": 0.3}

    trainer = _make_trainer(validation_step=_val_forward)
    evaluated_epochs: list[int] = []
    trainer.on_validation_end.append(lambda i, **kw: evaluated_epochs.append(i.epoch))
    trainer.fit(
        epochs=4,
        training_dataset=[{"x": 1}],
        validation_dataset=[{"x": 2}],
        validation_frequency=2,
    )
    assert evaluated_epochs == [2, 4]


def test_base_trainer_fit_with_start_epoch() -> None:
    """fit() starts from start_epoch, not from 1."""
    trainer = _make_trainer()
    history = trainer.fit(epochs=3, training_dataset=[{"x": 1}], start_epoch=2)
    assert list(history.keys()) == [2, 3]


def test_base_trainer_fit_invokes_epoch_callbacks() -> None:
    """on_epoch_begin and on_epoch_end are fired for every epoch."""
    trainer = _make_trainer()
    begins: list[int] = []
    ends: list[int] = []
    trainer.on_epoch_begin.append(lambda i, **kw: begins.append(i.epoch))
    trainer.on_epoch_end.append(lambda i, **kw: ends.append(i.epoch))
    trainer.fit(epochs=2, training_dataset=[{"x": 1}])
    assert begins == [1, 2]
    assert ends == [1, 2]


def test_base_trainer_fit_raises_for_zero_validation_frequency() -> None:
    """fit() raises ValueError when validation_frequency < 1."""
    trainer = _make_trainer()
    with pytest.raises(ValueError, match="Validation frequency"):
        trainer.fit(epochs=2, training_dataset=[], validation_frequency=0)


def test_base_trainer_fit_raises_for_start_epoch_zero() -> None:
    """fit() raises ValueError when start_epoch < 1."""
    trainer = _make_trainer()
    with pytest.raises(ValueError, match="Start epoch must be at least 1"):
        trainer.fit(epochs=2, training_dataset=[], start_epoch=0)


def test_base_trainer_fit_raises_when_start_epoch_exceeds_epochs() -> None:
    """fit() raises ValueError when start_epoch > epochs."""
    trainer = _make_trainer()
    with pytest.raises(ValueError, match="Start epoch must be less than or equal"):
        trainer.fit(epochs=2, training_dataset=[], start_epoch=3)


# ---------------------------------------------------------------------------
# BestCriterion
# ---------------------------------------------------------------------------


def test_best_criterion_min_mode_initial_best_is_inf() -> None:
    """In 'min' mode the initial best value is +inf."""
    criterion = BestCriterion(target="loss", mode="min")
    assert criterion._best == inf


def test_best_criterion_max_mode_initial_best_is_neg_inf() -> None:
    """In 'max' mode the initial best value is -inf."""
    criterion = BestCriterion(target="acc", mode="max")
    assert criterion._best == -inf


def test_best_criterion_min_mode_updates_on_improvement() -> None:
    """In 'min' mode _best decreases when a lower value is found."""
    best_values: list[float] = []

    def cb(info: Any, best: BestCriterion, **kw: Any) -> None:
        best_values.append(best.value)

    criterion = BestCriterion(target="loss", mode="min", on_best=[cb])

    info = BaseInfo()
    info.epoch = 1
    info.history[1] = {"loss": 0.5}
    criterion(info)
    assert best_values[-1] == pytest.approx(0.5)

    info.epoch = 2
    info.history[2] = {"loss": 0.3}
    criterion(info)
    assert best_values[-1] == pytest.approx(0.3)


def test_best_criterion_min_mode_does_not_update_on_regression() -> None:
    """In 'min' mode _best does not change when a higher value is seen."""
    best_values: list[float] = []

    def cb(info: Any, best: BestCriterion, **kw: Any) -> None:
        best_values.append(best.value)

    criterion = BestCriterion(target="loss", mode="min", on_best=[cb])

    info = BaseInfo()
    info.epoch = 1
    info.history[1] = {"loss": 0.3}
    criterion(info)

    info.epoch = 2
    info.history[2] = {"loss": 0.5}
    criterion(info)
    assert best_values[-1] == pytest.approx(0.3)


def test_best_criterion_max_mode_updates_on_improvement() -> None:
    """In 'max' mode _best increases when a higher value is found."""
    best_values: list[float] = []

    def cb(info: Any, best: BestCriterion, **kw: Any) -> None:
        best_values.append(best.value)

    criterion = BestCriterion(target="acc", mode="max", on_best=[cb])

    info = BaseInfo()
    info.epoch = 1
    info.history[1] = {"acc": 0.8}
    criterion(info)
    assert best_values[-1] == pytest.approx(0.8)


def test_best_criterion_on_best_not_called_when_target_missing() -> None:
    """on_best callbacks are skipped when the target key is absent from logs."""
    called: list[bool] = []

    def cb(info: Any, best: BestCriterion, **kw: Any) -> None:
        called.append(True)

    criterion = BestCriterion(target="loss", mode="min", on_best=[cb])

    info = BaseInfo()
    info.epoch = 1
    info.history[1] = {}  # no "loss" key
    criterion(info)
    assert not called


def test_best_criterion_on_best_called_even_without_improvement() -> None:
    """on_best IS always called as long as the target is present in logs."""
    called_count = 0

    def cb(info: Any, best: BestCriterion, **kw: Any) -> None:
        nonlocal called_count
        called_count += 1

    criterion = BestCriterion(target="loss", mode="min", on_best=[cb])
    info = BaseInfo()
    info.epoch = 1
    info.history[1] = {"loss": 0.5}
    criterion(info)
    info.epoch = 2
    info.history[2] = {"loss": 0.9}  # worse, but on_best still fires
    criterion(info)
    assert called_count == 2
