"""Unit tests for structcast_model.base_trainer."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from functools import partial
import inspect
from math import inf
from typing import Any, Literal

import pytest

from structcast_model.base_trainer import (
    EVENTS,
    BaseInfo,
    BaseTrainer,
    BestCriterion,
    Printer,
    ProgressBar,
    SimpleDataProvider,
    get_dataset,
    get_dataset_size,
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


def test_get_dataset_size_prefers_len_over_calling_a_callable_dataset() -> None:
    """A loader wrapper is callable but sized; counting it must not materialize an epoch of data."""

    class _SizedFactory:
        called = False

        def __len__(self) -> int:
            return 7

        def __call__(self) -> Iterator[dict[str, int]]:
            self.called = True
            yield {"x": 1}

    factory = _SizedFactory()
    assert get_dataset_size(factory) == 7
    assert factory.called is False


# ---------------------------------------------------------------------------
# SimpleDataProvider – steps_per_epoch / validation_steps
# ---------------------------------------------------------------------------


def test_simple_data_provider_counts_steps_from_its_datasets() -> None:
    """The provider owns the step counts, so consumers need no dataset-size logic of their own."""
    provider = SimpleDataProvider(training_dataset=[{"x": 1}, {"x": 2}], validation_dataset=[{"x": 3}])
    assert provider.steps_per_epoch == 2
    assert provider.validation_steps == 1


def test_simple_data_provider_satisfies_the_data_provider_protocol() -> None:
    """Widening the protocol must not orphan the package's own provider.

    Checked with getattr_static: DataProvider is deliberately not runtime_checkable, because on
    Python 3.11 an isinstance check would execute the property getters.
    """
    provider = SimpleDataProvider(training_dataset=[])
    for member in ("training_dataset", "validation_dataset", "steps_per_epoch", "validation_steps"):
        assert inspect.getattr_static(provider, member, None) is not None


def test_simple_data_provider_reports_zero_validation_steps_without_a_validation_dataset() -> None:
    """No validation dataset means fit() skips validation, so the count must be 0, not an error."""
    provider = SimpleDataProvider(training_dataset=[{"x": 1}])
    assert provider.validation_steps == 0


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
# Test collaborators
# ---------------------------------------------------------------------------


class _FakeLearner:
    """Minimal Learner implementation producing deterministic criteria."""

    def __init__(
        self,
        *,
        should_update: bool = True,
        inference_loss: float = 0.3,
        optimizers: Mapping[str, Any] | None = None,
    ) -> None:
        self._should_update = should_update
        self._inference_loss = inference_loss
        self.optimizers = dict(optimizers) if optimizers is not None else {}
        self.learning_rates = {"lr": 0.1}

    @property
    def models(self) -> dict[str, Any]:
        """Return the named models handed to every callback."""
        return {"model": "the-model"}

    def update(self, step: int) -> bool:
        """Report whether this step ends an update."""
        return self._should_update

    def training_step(self, **inputs: Any) -> dict[str, Any]:
        """Return fixed training criteria."""
        return {"loss": 0.5}

    def inference_step(self, **inputs: Any) -> dict[str, Any]:
        """Return fixed inference criteria."""
        return {"loss": self._inference_loss}


class _Tracker:
    """Callable tracker object, so that event methods can be attached to it."""

    def __call__(self, **criteria: Any) -> dict[str, float]:
        """Average nothing: pass the loss criterion straight through."""
        return {"loss": float(criteria.get("loss", 0.0))}


def _tracker(**criteria: Any) -> dict[str, float]:
    """Pass the loss criterion straight through, as a plain function."""
    return {"loss": float(criteria.get("loss", 0.0))}


class _Recorder:
    """Records the events it receives, implementing exactly the protocols it is asked to.

    The event methods are set as instance attributes so one class can stand in for any subset
    of the eleven event protocols; ``runtime_checkable`` protocols only check attribute presence.
    """

    def __init__(self, log: list[str], label: str = "rec", events: Sequence[str] = EVENTS) -> None:
        self.log = log
        self.label = label
        for event in events:
            setattr(self, event, partial(self._record, event))

    def _record(self, event: str, info: BaseInfo, **models: Any) -> None:
        """Append ``label:event`` to the shared log."""
        self.log.append(f"{self.label}:{event}")


def _hook(obj: Any, log: list[str], label: str, *events: str) -> Any:
    """Attach recording methods for *events* to *obj* so protocol routing picks it up."""
    for event in events:
        setattr(obj, event, partial(_Recorder(log, label, events=())._record, event))
    return obj


class _RecordingProvider:
    """DataProvider that records ``on_epoch_end``; SimpleDataProvider is slotted, so hooks need a class."""

    def __init__(self, log: list[str]) -> None:
        self.log = log
        self.training_dataset = [{"x": 1}]
        self.validation_dataset = None
        self.steps_per_epoch = 1
        self.validation_steps = 0

    def on_epoch_end(self, info: BaseInfo, **models: Any) -> None:
        """Record that the data provider received the event."""
        self.log.append("data:on_epoch_end")


class _EpochEndOnly:
    """Callback implementing only ``on_epoch_end``."""

    def __init__(self, log: list[str]) -> None:
        self.log = log

    def on_epoch_end(self, info: BaseInfo, **models: Any) -> None:
        """Record the epoch that just finished."""
        self.log.append(f"epoch_end:{info.epoch}")


def _make_trainer(
    *,
    learner: Any = None,
    tracker: Any = None,
    data: Any = None,
    callbacks: Sequence[Any] = (),
    training_prefix: str = "",
    validation_prefix: str = "val_",
) -> BaseTrainer:
    """Build a trainer wired with fake but real (non-mock) collaborators."""
    return BaseTrainer(
        learner=learner if learner is not None else _FakeLearner(),
        tracker=tracker if tracker is not None else _tracker,
        data=data if data is not None else SimpleDataProvider(training_dataset=[]),
        callbacks=callbacks,
        training_prefix=training_prefix,
        validation_prefix=validation_prefix,
    )


# ---------------------------------------------------------------------------
# Event routing
# ---------------------------------------------------------------------------


def test_all_eleven_events_route_at_the_right_moment() -> None:
    """Every event protocol implemented by a callback is dispatched in lifecycle order.

    This pins the contract the whole design rests on: implementing a protocol method is the
    only thing an object needs to do to take part in the loop.
    """
    log: list[str] = []
    trainer = _make_trainer(
        data=SimpleDataProvider(training_dataset=[{"x": 1}], validation_dataset=[{"x": 2}]),
        callbacks=[_Recorder(log)],
    )
    trainer.fit(epochs=1)
    assert log == [
        "rec:on_epoch_begin",
        "rec:on_training_begin",
        "rec:on_training_step_begin",
        "rec:on_update",
        "rec:on_training_step_end",
        "rec:on_training_end",
        "rec:on_validation_begin",
        "rec:on_validation_step_begin",
        "rec:on_validation_step_end",
        "rec:on_validation_end",
        "rec:on_epoch_end",
    ]
    assert set(EVENTS) == {entry.split(":", 1)[1] for entry in log}


def test_callbacks_receive_info_and_models() -> None:
    """A routed callback is called with the trainer as info plus the learner's models."""
    received: list[tuple[Any, dict[str, Any]]] = []

    class _Capture:
        def on_epoch_end(self, info: BaseInfo, **models: Any) -> None:
            """Capture the arguments passed by the trainer."""
            received.append((info, models))

    trainer = _make_trainer(data=SimpleDataProvider(training_dataset=[{"x": 1}]), callbacks=[_Capture()])
    trainer.fit(epochs=1)
    assert received == [(trainer, {"model": "the-model"})]


def test_object_without_event_methods_is_ignored() -> None:
    """An object implementing no event protocol is never registered and never called."""
    trainer = _make_trainer(data=SimpleDataProvider(training_dataset=[{"x": 1}]), callbacks=[object()])
    trainer.fit(epochs=1)
    assert trainer.describe() == {}


def test_explicit_callback_matching_no_event_warns(caplog: pytest.LogCaptureFixture) -> None:
    """A typo'd hook name (on_epoch_ended) would die silently, so dead explicit callbacks must warn."""

    class Typoed:
        def on_epoch_ended(self, info: Any, **models: Any) -> None:
            raise AssertionError("never called")

    with caplog.at_level("WARNING", logger="structcast_model.base_trainer"):
        _make_trainer(data=SimpleDataProvider(training_dataset=[{"x": 1}]), callbacks=[Typoed()]).fit(epochs=1)
    assert any("Typoed" in record.message and "no event protocol" in record.message for record in caplog.records)


def test_scanned_participants_without_events_do_not_warn(caplog: pytest.LogCaptureFixture) -> None:
    """The learner/tracker/data legitimately may implement no event: only explicit callbacks warn."""
    with caplog.at_level("WARNING", logger="structcast_model.base_trainer"):
        _make_trainer(data=SimpleDataProvider(training_dataset=[{"x": 1}])).fit(epochs=1)
    assert not caplog.records


def test_callbacks_appended_after_construction_join_on_first_use() -> None:
    """Late-appended callbacks still join the loop: the scan runs on first use.

    The CLI relies on this to build the trainer first and read display prefixes off the instance.
    """
    log: list[str] = []
    callbacks: list[Any] = []
    trainer = _make_trainer(data=SimpleDataProvider(training_dataset=[{"x": 1}]), callbacks=callbacks)
    callbacks.append(_Recorder(log, "late", ("on_epoch_end",)))
    trainer.fit(epochs=1)
    assert log == ["late:on_epoch_end"]


def test_describe_before_fit_does_not_freeze_the_scan() -> None:
    """describe() is a preview: inspecting the wiring must not drop callbacks appended afterwards."""
    log: list[str] = []
    callbacks: list[Any] = []
    trainer = _make_trainer(data=SimpleDataProvider(training_dataset=[{"x": 1}]), callbacks=callbacks)
    assert trainer.describe() == {}
    callbacks.append(_Recorder(log, "late", ("on_epoch_end",)))
    trainer.fit(epochs=1)
    assert log == ["late:on_epoch_end"]
    assert "on_epoch_end" in trainer.describe()


def test_provider_datasets_are_scanned_for_events_before_the_callbacks() -> None:
    """Provider datasets join the events without registration, ahead of the explicit callbacks.

    A dataset hook (e.g. a distributed sampler's set_epoch) must fire on the scan alone, and
    reshuffling must precede reporters.
    """
    log: list[str] = []

    class _EventfulDataset(list[dict[str, Any]]):
        def on_epoch_begin(self, info: BaseInfo, **models: Any) -> None:
            """Record that the trainer reached this dataset."""
            log.append("dataset:on_epoch_begin")

    trainer = _make_trainer(
        data=SimpleDataProvider(
            training_dataset=_EventfulDataset([{"x": 1}]),
            validation_dataset=_EventfulDataset([{"x": 2}]),
        ),
        callbacks=[_Recorder(log, "rec", ("on_epoch_begin",))],
    )
    trainer.fit(epochs=1)
    assert log == ["dataset:on_epoch_begin", "dataset:on_epoch_begin", "rec:on_epoch_begin"]


def test_scan_order_is_learner_then_tracker_then_data_then_callbacks() -> None:
    """Registration order fixes call order, so scheduler-like participants run before reporters."""
    log: list[str] = []
    learner = _hook(_FakeLearner(), log, "learner", "on_epoch_end")
    tracker = _hook(_Tracker(), log, "tracker", "on_epoch_end")
    data = _RecordingProvider(log)
    trainer = _make_trainer(
        learner=learner,
        tracker=tracker,
        data=data,
        callbacks=[_Recorder(log, "first", ("on_epoch_end",)), _Recorder(log, "second", ("on_epoch_end",))],
    )
    trainer.fit(epochs=1)
    assert log == [
        "learner:on_epoch_end",
        "tracker:on_epoch_end",
        "data:on_epoch_end",
        "first:on_epoch_end",
        "second:on_epoch_end",
    ]


def test_learner_optimizers_are_scanned_for_events() -> None:
    """Optimizer objects owned by the learner are routed too: this is how schedulers step."""
    log: list[str] = []
    learner = _FakeLearner(
        optimizers={
            "opt_a": _Recorder(log, "opt_a", ("on_update", "on_epoch_end")),
            "opt_b": _Recorder(log, "opt_b", ("on_update",)),
        }
    )
    trainer = _make_trainer(learner=learner, data=SimpleDataProvider(training_dataset=[{"x": 1}]))
    trainer.fit(epochs=1)
    assert log == ["opt_a:on_update", "opt_b:on_update", "opt_a:on_epoch_end"]


def test_same_object_registered_once_per_event() -> None:
    """An object appearing twice in the scan (tracker and callback) fires only once per event."""
    log: list[str] = []
    tracker = _hook(_Tracker(), log, "shared", "on_epoch_end")
    trainer = _make_trainer(tracker=tracker, data=SimpleDataProvider(training_dataset=[{"x": 1}]), callbacks=[tracker])
    trainer.fit(epochs=1)
    assert log == ["shared:on_epoch_end"]


def test_describe_lists_type_names_and_omits_empty_events() -> None:
    """describe() reports who listens to what, for run-time introspection of the wiring."""
    log: list[str] = []
    trainer = _make_trainer(callbacks=[_EpochEndOnly(log), _Recorder(log, "rec", ("on_update",))])
    assert trainer.describe() == {"on_update": ["_Recorder"], "on_epoch_end": ["_EpochEndOnly"]}


# ---------------------------------------------------------------------------
# BaseTrainer.train / BaseTrainer.evaluate
# ---------------------------------------------------------------------------


def test_train_returns_logs_and_counts_steps_and_updates() -> None:
    """train() accepts an explicit dataset and advances the step/update counters."""
    trainer = _make_trainer()
    logs = trainer.train([{"x": 1}, {"x": 2}])
    assert "loss" in logs
    assert trainer.step == 2
    assert trainer.update == 2


def test_train_with_prefix_renames_log_keys() -> None:
    """training_prefix is prepended to all log keys so training and validation never collide."""
    trainer = _make_trainer(training_prefix="train_")
    logs = trainer.train([{"x": 1}])
    assert "train_loss" in logs
    assert "loss" not in logs


def test_train_reports_average_elapsed_time_per_step() -> None:
    """elapsed_time is averaged over the steps taken so far, not accumulated."""
    trainer = _make_trainer()
    logs = trainer.train([{"x": 1}, {"x": 2}, {"x": 3}])
    assert 0.0 <= logs["elapsed_time"] < 1.0


def test_train_stores_logs_in_history() -> None:
    """Logs are stored in trainer.history under the current epoch."""
    trainer = _make_trainer()
    trainer.epoch = 1
    trainer.train([{"x": 1}])
    assert "loss" in trainer.history[1]


def test_train_with_callable_dataset() -> None:
    """Callable dataset factories are supported in train()."""
    trainer = _make_trainer()
    data = [{"x": 1}]
    logs = trainer.train(lambda: data)
    assert "loss" in logs


def test_on_update_fires_only_when_the_learner_updates() -> None:
    """With gradient accumulation the learner decides when an update happened; the trainer follows."""
    log: list[str] = []
    trainer = _make_trainer(
        learner=_FakeLearner(should_update=False),
        callbacks=[_Recorder(log, "rec", ("on_update",))],
    )
    trainer.train([{"x": 1}, {"x": 2}])
    assert log == []
    assert trainer.update == 0
    assert trainer.step == 2


def test_evaluate_prefixes_logs_and_fires_validation_events() -> None:
    """evaluate() accepts an explicit dataset and prefixes its logs with validation_prefix."""
    log: list[str] = []
    trainer = _make_trainer(callbacks=[_Recorder(log)])
    logs = trainer.evaluate([{"x": 1}])
    assert "val_loss" in logs
    assert "val_elapsed_time" in logs
    assert log == [
        "rec:on_validation_begin",
        "rec:on_validation_step_begin",
        "rec:on_validation_step_end",
        "rec:on_validation_end",
    ]


def test_evaluate_with_callable_dataset() -> None:
    """Callable dataset factories are supported in evaluate()."""
    trainer = _make_trainer()
    data = [{"x": 1}]
    logs = trainer.evaluate(lambda: data)
    assert "val_loss" in logs


# ---------------------------------------------------------------------------
# BaseTrainer.fit
# ---------------------------------------------------------------------------


def test_fit_pulls_datasets_from_the_data_provider() -> None:
    """fit() takes no dataset arguments: a fully wired trainer already knows its data."""
    trainer = _make_trainer(data=SimpleDataProvider(training_dataset=[{"x": 1}], validation_dataset=[{"x": 2}]))
    history = trainer.fit(epochs=3)
    assert list(history.keys()) == [1, 2, 3]
    assert "loss" in history[3]
    assert "val_loss" in history[3]


def test_fit_without_validation_dataset_skips_evaluation() -> None:
    """A provider with no validation dataset means training only."""
    trainer = _make_trainer(data=SimpleDataProvider(training_dataset=[{"x": 1}]))
    history = trainer.fit(epochs=2)
    assert "val_loss" not in history[2]


def test_fit_respects_validation_frequency() -> None:
    """Validation only runs at multiples of validation_frequency."""
    log: list[str] = []
    trainer = _make_trainer(
        data=SimpleDataProvider(training_dataset=[{"x": 1}], validation_dataset=[{"x": 2}]),
        callbacks=[_EpochEndOnly(log), _Recorder(log, "val", ("on_validation_end",))],
    )
    trainer.fit(epochs=4, validation_frequency=2)
    assert log == [
        "epoch_end:1",
        "val:on_validation_end",
        "epoch_end:2",
        "epoch_end:3",
        "val:on_validation_end",
        "epoch_end:4",
    ]


def test_fit_starts_from_start_epoch() -> None:
    """fit() starts from start_epoch, not from 1, so a run can be resumed."""
    trainer = _make_trainer(data=SimpleDataProvider(training_dataset=[{"x": 1}]))
    history = trainer.fit(epochs=3, start_epoch=2)
    assert list(history.keys()) == [2, 3]


def test_trainer_requires_a_data_provider() -> None:
    """The data provider is a required constructor argument: a trainer without data cannot exist."""
    with pytest.raises(TypeError, match="data"):
        BaseTrainer(learner=_FakeLearner(), tracker=_tracker)  # type: ignore[call-arg]  # the missing
        # argument is the behavior under test; mypy correctly rejects the call at type-check time.


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"epochs": 2, "validation_frequency": 0}, "Validation frequency"),
        ({"epochs": 2, "start_epoch": 0}, "Start epoch must be at least 1"),
        ({"epochs": 2, "start_epoch": 3}, "Start epoch must be less than or equal"),
    ],
)
def test_fit_rejects_invalid_loop_parameters(kwargs: dict[str, int], message: str) -> None:
    """Loop parameters are validated before any epoch runs."""
    trainer = _make_trainer(data=SimpleDataProvider(training_dataset=[{"x": 1}]))
    with pytest.raises(ValueError, match=message):
        trainer.fit(**kwargs)


# ---------------------------------------------------------------------------
# BestCriterion
# ---------------------------------------------------------------------------


def test_best_criterion_min_mode_initial_best_is_inf() -> None:
    """In 'min' mode the initial best value is +inf, so any first value improves on it."""
    criterion = BestCriterion(target="loss", mode="min")
    assert criterion.value == inf


def test_best_criterion_max_mode_initial_best_is_neg_inf() -> None:
    """In 'max' mode the initial best value is -inf, so any first value improves on it."""
    criterion = BestCriterion(target="acc", mode="max")
    assert criterion.value == -inf


@pytest.mark.parametrize(
    ("mode", "values", "expected"),
    [
        ("min", [0.5, 0.3], 0.3),
        ("min", [0.3, 0.5], 0.3),
        ("max", [0.5, 0.8], 0.8),
        ("max", [0.8, 0.5], 0.8),
    ],
)
def test_best_criterion_tracks_the_best_value(
    mode: Literal["min", "max"], values: list[float], expected: float
) -> None:
    """The best value survives regressions: that is what makes 'best' meaningful."""
    criterion = BestCriterion(target="loss", mode=mode)
    info = BaseInfo()
    for epoch, value in enumerate(values, start=1):
        info.epoch = epoch
        info.step = epoch
        info.history[epoch] = {"loss": value}
        criterion.on_epoch_end(info)
    assert criterion.value == pytest.approx(expected)
    assert criterion.step == 1 + values.index(expected)


class _BestRecorder:
    """OnBest participant recording what the criterion reports, the way a logger would."""

    def __init__(self) -> None:
        self.seen: list[tuple[int, float, dict[str, Any]]] = []

    def on_best(self, info: BaseInfo, best: BestCriterion[Any], **models: Any) -> None:
        """Record the epoch, the best value, and the models."""
        self.seen.append((info.epoch, best.value, models))


def test_best_criterion_on_best_receives_info_best_and_models() -> None:
    """on_best participants get the criterion itself, so they can log or save by its value/step."""
    recorder = _BestRecorder()
    criterion = BestCriterion(target="loss", on_best=[recorder])
    info = BaseInfo()
    info.epoch = 1
    info.history[1] = {"loss": 0.5}
    criterion.on_epoch_end(info, model="the-model")
    assert recorder.seen == [(1, 0.5, {"model": "the-model"})]


def test_best_criterion_on_best_called_even_without_improvement() -> None:
    """on_best fires whenever the target is present, so consumers can log the best value each epoch."""
    recorder = _BestRecorder()
    criterion = BestCriterion(target="loss", on_best=[recorder])
    info = BaseInfo()
    info.epoch = 1
    info.history[1] = {"loss": 0.5}
    criterion.on_epoch_end(info)
    info.epoch = 2
    info.history[2] = {"loss": 0.9}
    criterion.on_epoch_end(info)
    assert [value for _, value, _ in recorder.seen] == [0.5, 0.5]


def test_best_criterion_on_best_skipped_when_target_missing() -> None:
    """A criterion that was not produced this epoch must not trigger best-value side effects."""
    recorder = _BestRecorder()
    criterion = BestCriterion(target="loss", on_best=[recorder])
    info = BaseInfo()
    info.epoch = 1
    info.history[1] = {}
    criterion.on_epoch_end(info)
    assert not recorder.seen


def test_best_criterion_routes_through_the_trainer() -> None:
    """Passed as a callback, BestCriterion is routed by its on_epoch_end method alone."""
    criterion = BestCriterion(target="val_loss", mode="min")
    trainer = _make_trainer(
        learner=_FakeLearner(inference_loss=0.2),
        data=SimpleDataProvider(training_dataset=[{"x": 1}], validation_dataset=[{"x": 2}]),
        callbacks=[criterion],
    )
    assert trainer.describe() == {"on_epoch_end": ["BestCriterion"]}
    trainer.fit(epochs=1)
    assert criterion.value == pytest.approx(0.2)


# ---------------------------------------------------------------------------
# Printer / ProgressBar
# ---------------------------------------------------------------------------


def test_printer_prints_epoch_criteria(capsys: pytest.CaptureFixture[str]) -> None:
    """Printer is the CI-mode reporter: one plain-text block per epoch, learning rates included."""
    trainer = _make_trainer(data=SimpleDataProvider(training_dataset=[{"x": 1}]), callbacks=[Printer()])
    trainer.fit(epochs=1)
    out = capsys.readouterr().out
    assert "epoch: 1" in out
    assert "  lr: 0.1" in out
    assert "  loss: 0.5" in out


def test_progress_bar_runs_a_full_epoch(capsys: pytest.CaptureFixture[str]) -> None:
    """ProgressBar drives a real tqdm bar through training, validation, and the epoch summary."""
    bar = ProgressBar(
        steps_per_epoch=2, validation_steps=1, training_criteria=["loss"], validation_criteria=["val_loss"]
    )
    trainer = _make_trainer(
        data=SimpleDataProvider(training_dataset=[{"x": 1}, {"x": 2}], validation_dataset=[{"x": 3}]), callbacks=[bar]
    )
    trainer.fit(epochs=1)
    assert "epoch: 1" in capsys.readouterr().out
    bar.bar.close()
