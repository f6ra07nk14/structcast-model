"""Unit tests for structcast_model.keras.adapters.

Every test here runs on whichever backend `KERAS_BACKEND` selects (the conftest defaults it to
tensorflow), because the whole point of the adapters is that one learner behaves the same on all
three. Each test encodes one silent failure: a training path whose dominant failure mode is a no-op
rather than an exception is what `docs/adr/0016` rejected the alternatives for.

The loss-decrease assertions all reuse one fixed batch. A moving batch hides a dead optimizer: the
loss wanders on its own and a step that updates nothing still produces a plausible-looking curve.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import numpy as np
import pytest

import keras
from structcast_model.keras.adapters import (
    AdapterSegment,
    BackendAdapter,
    Flow,
    InferenceFlow,
    select_backend_adapter,
)


@pytest.fixture
def adapter() -> BackendAdapter:
    """The adapter of the active backend."""
    return select_backend_adapter()


@pytest.fixture
def restore_policy() -> Iterator[None]:
    """Restore the global mixed precision policy, which is process-wide state."""
    original = keras.mixed_precision.global_policy()
    yield
    keras.mixed_precision.set_global_policy(original)


def _batch(size: int = 8) -> dict[str, Any]:
    """One fixed batch, as backend tensors so a compiled step traces once."""
    generator = np.random.default_rng(0)
    return {
        "x": keras.ops.convert_to_tensor(generator.standard_normal((size, 3)).astype("float32")),
        "y": keras.ops.convert_to_tensor(generator.standard_normal((size, 2)).astype("float32")),
    }


def _model(*layers: Any, seed: int = 0) -> Any:
    """A small dense model, seeded so every backend starts from the same weights."""
    keras.utils.set_random_seed(seed)
    return keras.Sequential([keras.Input((3,)), *layers, keras.layers.Dense(2)])


def _flow(model: Any, *, training: bool = True) -> Flow:
    """The mean squared error of `model` on the batch, as a training flow."""

    def flow(batch: Any) -> tuple[Any, dict[str, Any]]:
        loss = keras.ops.mean(keras.ops.square(keras.ops.subtract(model(batch["x"], training=training), batch["y"])))
        return loss, {"loss": loss}

    return flow


def _inference_flow(model: Any) -> InferenceFlow:
    """The same criterion without a loss to differentiate."""

    def flow(batch: Any) -> dict[str, Any]:
        prediction = model(batch["x"], training=False)
        return {"loss": keras.ops.mean(keras.ops.square(keras.ops.subtract(prediction, batch["y"])))}

    return flow


def _segment(model: Any, *, name: str = "sgd", learning_rate: float = 0.1, optimizer: Any = None) -> AdapterSegment:
    """One optimizer segment training every trainable variable of `model`."""
    return AdapterSegment(
        name=name,
        flow=_flow(model),
        optimizer=keras.optimizers.SGD(learning_rate=learning_rate) if optimizer is None else optimizer,
        variables=list(model.trainable_variables),
        models=[model],
    )


def _value(variable: Any) -> np.ndarray:
    """The host-side value of a variable or a tensor."""
    return np.asarray(keras.ops.convert_to_numpy(getattr(variable, "value", variable)))


def _run(step: InferenceFlow, batch: dict[str, Any], times: int) -> list[float]:
    """Run a step repeatedly on the same batch and collect its losses."""
    return [float(_value(step(batch)["loss"])) for _ in range(times)]


def test_training_step_decreases_the_loss_and_moves_the_weights(adapter: BackendAdapter) -> None:
    """Twenty steps on one batch must lower the loss every time and actually move a kernel.

    An adapter that computes gradients but drops the update -- the wrong variable list, a stateless
    apply whose result is discarded, an optimizer left unbuilt -- raises nothing and returns a
    perfectly stable loss. Requiring both a strictly decreasing loss and a moved kernel makes the
    no-op fail here rather than after a training run.
    """
    model = _model()
    segment = _segment(model)
    adapter.prepare([segment])
    before = _value(model.trainable_variables[0])
    losses = _run(adapter.build_train_step([segment]), _batch(), 20)
    assert all(later < earlier for earlier, later in zip(losses, losses[1:], strict=False))
    assert np.abs(_value(model.trainable_variables[0]) - before).max() > 0


def test_batch_normalization_statistics_move_in_training_and_freeze_in_inference(adapter: BackendAdapter) -> None:
    """Moving statistics are updated by the flow, not by the optimizer, so an adapter can drop them.

    On JAX they are only updated if the adapter threads the non-trainable state out of the
    stateless scope and writes it back; dropping it leaves the statistics at their initial values
    and normalizes inference with mean 0 and variance 1 forever.
    """
    model = _model(keras.layers.BatchNormalization())
    moving_mean = model.layers[0].moving_mean
    segment = _segment(model)
    adapter.prepare([segment])
    before = _value(moving_mean)
    _run(adapter.build_train_step([segment]), _batch(), 2)
    trained = _value(moving_mean)
    assert np.abs(trained - before).max() > 0

    _run(adapter.build_inference_step(_inference_flow(model)), _batch(), 2)
    assert np.array_equal(_value(moving_mean), trained)


def test_dropout_advances_its_seed_in_training_and_stays_deterministic_in_inference(
    adapter: BackendAdapter,
) -> None:
    """Two training steps on the identical batch must differ, because the dropout mask must differ.

    The learning rate is zero so the weights cannot move: the only thing that can change the loss
    is the `SeedGenerator` state. On JAX that state lives in the same stateless scope as the moving
    statistics, and an adapter that forgets it replays one frozen mask for the whole run -- with no
    error, and with a loss curve that still goes down.
    """
    model = _model(keras.layers.Dropout(0.5, seed=0))
    seed = model.layers[0].seed_generator.state
    segment = _segment(model, learning_rate=0.0)
    adapter.prepare([segment])
    before = _value(seed)
    losses = _run(adapter.build_train_step([segment]), _batch(), 2)
    assert losses[0] != losses[1]
    assert not np.array_equal(_value(seed), before)

    inference = _run(adapter.build_inference_step(_inference_flow(model)), _batch(), 2)
    assert inference[0] == inference[1]


def test_each_segment_updates_only_its_own_variables(adapter: BackendAdapter) -> None:
    """Two segments must train their own model with their own optimizer and leave the other alone.

    A gradient taken against the union of the variables, or an optimizer applied to the union,
    trains both models from one loss: the criteria still look sane, and only the second model's
    quality shows it.
    """
    first, second = _model(seed=0), _model(seed=1)
    segments = [_segment(first, name="first"), _segment(second, name="second")]
    adapter.prepare(segments)
    before = [_value(model.trainable_variables[0]) for model in (first, second)]
    adapter.build_train_step(segments)(_batch())
    assert all(
        np.abs(_value(model.trainable_variables[0]) - snapshot).max() > 0
        for model, snapshot in zip((first, second), before, strict=True)
    )

    untouched = _value(second.trainable_variables[0])
    adapter.build_train_step(segments[:1])(_batch())
    assert np.array_equal(_value(second.trainable_variables[0]), untouched)


def test_float16_wraps_the_optimizer_and_keeps_the_gradients_scaled(
    adapter: BackendAdapter, restore_policy: None
) -> None:
    """A float16 policy must wrap the optimizer and the scaling must actually reach the loss.

    `LossScaleOptimizer` always unscales the gradients by its dynamic scale, so an adapter that
    never calls `scale_loss` divides every update by 2**15 instead: the loss still decreases, just
    imperceptibly. Asserting the size of the update is what separates the two.
    """
    keras.mixed_precision.set_global_policy("mixed_float16")
    model = _model()
    segment = _segment(model)
    adapter.prepare([segment], mixed_precision=True, mixed_precision_type="float16")
    assert isinstance(segment.optimizer, keras.optimizers.LossScaleOptimizer)
    before = _value(model.trainable_variables[0])
    losses = _run(adapter.build_train_step([segment]), _batch(), 20)
    assert losses[-1] < losses[0]
    assert np.abs(_value(model.trainable_variables[0]) - before).max() > 1e-3


def test_float16_mapping_supplies_the_loss_scale_optimizer_keywords(adapter: BackendAdapter) -> None:
    """A dict-valued mixed precision configures the wrapper instead of only enabling it."""
    segment = _segment(_model())
    adapter.prepare(
        [segment],
        mixed_precision={"initial_scale": 128.0, "dynamic_growth_steps": 5},
        mixed_precision_type="float16",
    )
    assert isinstance(segment.optimizer, keras.optimizers.LossScaleOptimizer)
    assert segment.optimizer.initial_scale == 128.0
    assert segment.optimizer.dynamic_growth_steps == 5


def test_float16_empty_mapping_still_wraps_the_optimizer(adapter: BackendAdapter) -> None:
    """An empty dict means enabled with default keywords, as the schema and the torch builder read it.

    Treating it as disabled would train float16 with unscaled gradients: no error, just updates
    lost to underflow.
    """
    segment = _segment(_model())
    adapter.prepare([segment], mixed_precision={}, mixed_precision_type="float16")
    assert isinstance(segment.optimizer, keras.optimizers.LossScaleOptimizer)


def test_bfloat16_leaves_the_optimizer_unwrapped(adapter: BackendAdapter, restore_policy: None) -> None:
    """bfloat16 keeps float32's exponent range, so loss scaling is pure overhead and must not appear."""
    keras.mixed_precision.set_global_policy("mixed_bfloat16")
    model = _model()
    segment = _segment(model)
    adapter.prepare([segment], mixed_precision=True, mixed_precision_type="bfloat16")
    assert not isinstance(segment.optimizer, keras.optimizers.LossScaleOptimizer)
    assert model.layers[-1].compute_dtype == "bfloat16"
    assert model.trainable_variables[0].dtype == "float32"
    losses = _run(adapter.build_train_step([segment]), _batch(), 20)
    # Only the endpoints, not every adjacent pair: one bfloat16 step's rounding error is the size
    # of one loss decrement here, and monotonicity is float32's trap, tested above.
    assert losses[-1] < losses[0]


def test_prepare_rejects_a_segment_without_trainable_variables(adapter: BackendAdapter) -> None:
    """A segment whose layers are all frozen would train nothing without ever failing."""
    model = _model()
    model.trainable = False
    with pytest.raises(ValueError, match="no trainable variables"):
        adapter.prepare([_segment(model)])


def test_inference_step_changes_no_variable(adapter: BackendAdapter) -> None:
    """Validation must leave weights, moving statistics and seeds exactly as it found them."""
    model = _model(keras.layers.BatchNormalization(), keras.layers.Dropout(0.5, seed=0))
    before = [_value(variable) for variable in model.variables]
    _run(adapter.build_inference_step(_inference_flow(model), models=[model]), _batch(), 3)
    assert all(
        np.array_equal(_value(variable), snapshot) for variable, snapshot in zip(model.variables, before, strict=True)
    )


def test_inference_step_reads_the_current_weights(adapter: BackendAdapter) -> None:
    """A step built before training must answer with the weights training left, not the traced ones.

    On JAX the compiled step only does that if the variables are threaded through the jit as
    arguments; a jitted closure reading them directly constant-folds the initial weights and keeps
    reporting the same validation loss for the whole run.
    """
    model = _model()
    segment = _segment(model)
    adapter.prepare([segment])
    inference = adapter.build_inference_step(_inference_flow(model), models=[model])
    before = _run(inference, _batch(), 1)[0]
    _run(adapter.build_train_step([segment]), _batch(), 5)
    assert _run(inference, _batch(), 1)[0] < before


def test_select_backend_adapter_caches_the_adapter_of_the_active_backend() -> None:
    """The backend is resolved once: a second resolution could disagree with the first."""
    adapter = select_backend_adapter()
    assert adapter is select_backend_adapter()
    assert isinstance(adapter, BackendAdapter)
    assert adapter.name == keras.backend.backend()


def test_select_backend_adapter_rejects_an_unknown_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    """An unsupported backend must be named, not silently trained with the wrong mechanics."""
    monkeypatch.setattr(keras.backend, "backend", lambda: "openvino")
    select_backend_adapter.cache_clear()
    try:
        with pytest.raises(ValueError, match="has no training adapter"):
            select_backend_adapter()
    finally:
        # The cache holds the wrong answer either way, so clear it for the tests that follow.
        select_backend_adapter.cache_clear()
