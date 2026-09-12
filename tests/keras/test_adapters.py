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
    JaxAdapter,
    TensorFlowAdapter,
    TorchAdapter,
    select_backend_adapter,
    swap_ema_weights,
)

COMPILE_OPTIONS: list[dict[str, Any] | None] = [None] if keras.backend.backend() == "torch" else [None, {}]
"""The two compilation choices the two tests below are checked under.

Only those two: they are the mechanics whose failure mode needs a compiler to exist -- a jitted
closure folding its variables into constants, and a loss-scaled `apply` raising under a trace -- so
everywhere else the second case would double the run time without a way to fail. The torch backend
builds no compiled step, so it has the one case.
"""


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
    """The mean squared error of `model` on the batch, as a training flow.

    Keyword-only, as a generated flow is: an adapter that handed the batch over positionally, or as
    one mapping, would fail here instead of binding a learner's inputs by declaration order.
    """

    def flow(*, x: Any, y: Any) -> tuple[Any, dict[str, Any]]:
        loss = keras.ops.mean(keras.ops.square(keras.ops.subtract(model(x, training=training), y)))
        return loss, {"loss": loss}

    return flow


def _inference_flow(model: Any) -> InferenceFlow:
    """The same criterion without a loss to differentiate."""

    def flow(*, x: Any, y: Any) -> dict[str, Any]:
        prediction = model(x, training=False)
        return {"loss": keras.ops.mean(keras.ops.square(keras.ops.subtract(prediction, y)))}

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
    return [float(_value(step(**batch)["loss"])) for _ in range(times)]


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

    _run(adapter.build_inference_step(_inference_flow(model), models=[model]), _batch(), 2)
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

    inference = _run(adapter.build_inference_step(_inference_flow(model), models=[model]), _batch(), 2)
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
    adapter.build_train_step(segments)(**_batch())
    assert all(
        np.abs(_value(model.trainable_variables[0]) - snapshot).max() > 0
        for model, snapshot in zip((first, second), before, strict=True)
    )

    untouched = _value(second.trainable_variables[0])
    adapter.build_train_step(segments[:1])(**_batch())
    assert np.array_equal(_value(second.trainable_variables[0]), untouched)


@pytest.mark.parametrize("compile_kw", COMPILE_OPTIONS, ids=lambda kw: "eager" if kw is None else "compiled")
def test_float16_wraps_the_optimizer_and_keeps_the_gradients_scaled(
    compile_kw: dict[str, Any] | None, restore_policy: None
) -> None:
    """A float16 policy must wrap the optimizer and the scaling must actually reach the loss.

    `LossScaleOptimizer` always unscales the gradients by its dynamic scale, so an adapter that
    never calls `scale_loss` divides every update by 2**15 instead: the loss still decreases, just
    imperceptibly. Asserting the size of the update is what separates the two.

    Both choices, unlike everything else here but the freeze-at-first-trace test: the wrapper's
    stateful `apply` raises an `UnexpectedTracerError` under a JAX trace, which is the whole reason
    the adapter threads the loss scale through `stateless_apply`, and an eager run never reaches
    that failure. A fresh adapter rather than the cached one, whose choice belongs to whoever ran
    the command.
    """
    adapter = type(select_backend_adapter())()
    adapter.compile_kw = compile_kw
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


@pytest.mark.parametrize("compile_kw", COMPILE_OPTIONS, ids=lambda kw: "eager" if kw is None else "compiled")
def test_inference_step_reads_the_current_weights(compile_kw: dict[str, Any] | None) -> None:
    """A step built before training must answer with the weights training left, not the traced ones.

    Both choices, unlike everything else here: on JAX the compiled step only does that if the
    variables are threaded through the jit as arguments, and a jitted closure reading them directly
    constant-folds the initial weights and keeps reporting the same validation loss for the whole
    run. The eager case is what says that threading costs an uncompiled step nothing. A fresh
    adapter rather than the cached one, whose choice belongs to whoever ran the command.
    """
    adapter = type(select_backend_adapter())()
    adapter.compile_kw = compile_kw
    model = _model()
    segment = _segment(model)
    adapter.prepare([segment])
    inference = adapter.build_inference_step(_inference_flow(model), models=[model])
    before = _run(inference, _batch(), 1)[0]
    _run(adapter.build_train_step([segment]), _batch(), 5)
    assert _run(inference, _batch(), 1)[0] < before


def _averaging(model: Any, *, applies: int = 1, learning_rate: float = 0.1) -> Any:
    """An SGD keeping an average of every trainable variable of `model`, stepped `applies` times."""
    optimizer = keras.optimizers.SGD(learning_rate=learning_rate, use_ema=True, ema_momentum=0.5)
    variables = list(model.trainable_variables)
    optimizer.build(variables)
    for _ in range(applies):
        # Built from the shape rather than with `ones_like`, which hands the JAX backend a Keras
        # variable it refuses to read a dtype off.
        optimizer.apply([keras.ops.ones(v.shape, v.dtype) for v in variables], variables)
    return optimizer


def test_swap_ema_weights_leaves_an_unstarted_average_alone() -> None:
    """`Trainer.evaluate()` before the first update must measure the model, not a field of zeros.

    Keras zero-initializes `_model_variables_moving_average` and only writes it from an `apply`, so
    a swap that trusted the array to hold an average would evaluate an all-zero model and report a
    validation number belonging to nothing -- finite, plausible and wrong.
    """
    model = _model()
    optimizer = _averaging(model, applies=0)
    before = [_value(variable) for variable in model.trainable_variables]

    swap_ema_weights([optimizer])

    assert all(np.array_equal(_value(v), b) for v, b in zip(model.trainable_variables, before, strict=True))
    assert all(not np.any(_value(a)) for a in optimizer._model_variables_moving_average)


def test_swap_ema_weights_trades_a_shared_variable_once() -> None:
    """Two optimizers averaging one model must not trade the same variable twice.

    A second trade puts the second average into the model on the way in and leaves the *first* one
    there on the way out, so the weights come back as an average and training continues from them --
    corruption, not a wrong reading. The learner builder refuses the configuration that reaches this,
    so the rule here is what keeps a hand-wired one from being silently destroyed.
    """
    model = _model()
    first, second = _averaging(model), _averaging(model, learning_rate=0.2)
    weights = [_value(variable) for variable in model.trainable_variables]
    averages = [_value(average) for average in first._model_variables_moving_average]
    assert not np.array_equal(averages[0], _value(second._model_variables_moving_average[0]))

    swap_ema_weights([first, second])
    swapped = [_value(variable) for variable in model.trainable_variables]
    swap_ema_weights([first, second])

    # The first optimizer in the sequence wins, and the second one's average is left where it was.
    assert all(np.array_equal(value, average) for value, average in zip(swapped, averages, strict=True))
    assert all(np.array_equal(_value(v), w) for v, w in zip(model.trainable_variables, weights, strict=True))
    assert all(
        np.array_equal(_value(average), value)
        for average, value in zip(first._model_variables_moving_average, averages, strict=True)
    )


def test_swap_ema_weights_puts_back_what_it_traded_when_a_trade_fails() -> None:
    """A swap that dies partway must unwind: its caller's `finally` can only undo a whole one.

    An allocation failure inside the copy is the real shape of this, and a shape the assignment
    refuses is the deterministic stand-in. Without the unwind the variables traded before the
    failure keep the average, and the run trains on from there.
    """
    model = _model()
    # Three applies, so the average has actually moved away from the weights: after a single one it
    # is seeded to them, and a swap that never unwound would still look like it had.
    optimizer = _averaging(model, applies=3)
    weights = [_value(variable) for variable in model.trainable_variables]
    assert not np.array_equal(weights[0], _value(optimizer._model_variables_moving_average[0]))
    # The second pair alone is broken, so the first has already been traded when the swap fails.
    optimizer._model_variables_moving_average[1] = keras.Variable(np.zeros((3, 3), "float32"))

    with pytest.raises(ValueError, match="shape"):
        swap_ema_weights([optimizer])

    assert all(np.array_equal(_value(v), w) for v, w in zip(model.trainable_variables, weights, strict=True))


def _repeated_python_calls(adapter: BackendAdapter) -> int:
    """Run one training step until it is warm, then count the flow's Python runs over two more calls.

    The one difference a compiled step shows from outside: it runs the flow while it is traced and
    never again, where an eager step runs it on every call. The warm-up is two calls because a
    backend may trace more than once before it settles, and only what happens after that is counted.
    """
    calls: list[str] = []
    model = _model()
    segment = _segment(model)
    flow = segment.flow

    def counted(**batch: Any) -> tuple[Any, dict[str, Any]]:
        calls.append("run")
        return flow(**batch)

    segment.flow = counted
    adapter.prepare([segment])
    step = adapter.build_train_step([segment])
    batch = _batch()
    step(**batch)
    step(**batch)
    warm = len(calls)
    step(**batch)
    step(**batch)
    return len(calls) - warm


def test_the_steps_stay_eager_until_compilation_is_asked_for() -> None:
    """An adapter nobody asked to compile must not: `--compile` owns that choice on every backend.

    The TensorFlow and JAX adapters used to compile whatever their framework could, which made the
    same learner run one way here and another way on torch. Nothing else in a run tells the two
    apart -- the criteria are identical -- so the Python the flow stops running is what pins it.
    """
    adapter = type(select_backend_adapter())()

    assert adapter.compile_kw is None
    assert _repeated_python_calls(adapter) == 2


@pytest.mark.skipif(keras.backend.backend() == "torch", reason="The torch backend builds no compiled step.")
def test_asking_for_compilation_traces_the_step_instead_of_rerunning_it() -> None:
    """`--compile true` must reach `tf.function`/`jax.jit`, and a warm step going quiet is the proof."""
    adapter = type(select_backend_adapter())()
    adapter.compile_kw = {}

    assert _repeated_python_calls(adapter) == 0


def test_the_torch_backend_refuses_the_compilation_it_cannot_do() -> None:
    """Dropping the arguments would report a compiled run to a user who asked for one and got none.

    Both builders, because a learner builds both: an inference step that quietly stayed eager while
    the training step refused would make the same flag mean two things in one run. Checked on every
    backend, not only under torch: the refusal belongs to the adapter that has no compiler, and it is
    the reason `--compile` is not silently backend-dependent.
    """
    adapter = TorchAdapter()
    adapter.compile_kw = {"mode": "max-autotune"}

    with pytest.raises(ValueError, match="builds no compiled step"):
        adapter.build_train_step([])
    with pytest.raises(ValueError, match="builds no compiled step"):
        adapter.build_inference_step(lambda **batch: dict(batch))


def test_the_jax_adapter_refuses_to_jit_an_inference_step_it_cannot_thread() -> None:
    """Without a model to thread, jitting would freeze the weights, so the request is refused.

    The one place a Keras compile request could still be dropped silently: that early return exists
    because a jitted closure reading its variables directly answers with the weights of its first
    trace forever, and returning the eager flow while `--compile` asked for a compiled one would be
    the same lie the torch backend refuses to tell. No model reaches it from a generated learner,
    which is why nothing else would notice.
    """
    adapter = JaxAdapter()

    def flow(**batch: Any) -> dict[str, Any]:
        return dict(batch)

    # Eager, the case the early return exists for: the flow is handed back as it is.
    assert adapter.build_inference_step(flow) is flow

    adapter.compile_kw = {}
    with pytest.raises(ValueError, match="no model to thread"):
        adapter.build_inference_step(flow)


def _recorded_compiler_kw(monkeypatch: pytest.MonkeyPatch, framework: str, compiler: str) -> list[dict[str, Any]]:
    """Collect what the adapters hand `<framework>.<compiler>`, with the real compiler still running.

    The adapter module binds each framework to a `LazyModuleImporter` in its own globals, and that
    importer copies the framework's `__dict__` onto itself the first time it is read, so patching
    the real `jax` or `tensorflow` module would leave the copy an adapter reads untouched. The
    binding it does read is reached here through the globals of one of its own methods:
    `sys.modules` holds a `LazySelectedImporter` that exposes nothing but `__all__`.
    """
    importer = JaxAdapter._compile_step.__globals__[framework]
    compile_step = getattr(importer, compiler)
    recorded: list[dict[str, Any]] = []

    def recording(step: Any, **kwargs: Any) -> Any:
        recorded.append(kwargs)
        return compile_step(step, **kwargs)

    monkeypatch.setattr(importer, compiler, recording)
    return recorded


@pytest.mark.skipif(keras.backend.backend() != "jax", reason="Only the JAX adapter jits with these arguments.")
def test_the_jax_adapter_drops_the_contract_arguments_of_the_step_it_jits(monkeypatch: pytest.MonkeyPatch) -> None:
    """What is static and what is donated belongs to the step, not to whoever passes `--compile`.

    One mapping is splatted into a training step and an inference step whose positional signatures
    differ, so the same `donate_argnums` names the optimizer state in one and the batch in the
    other: both spellings of both keys are dropped, from both steps, and everything else --
    `inline` here -- is passed through as it came. What reaches `jax.jit` is what is asserted
    because donation is a no-op on CPU: a forwarded `donate_argnums` changes nothing a run could
    observe here, so only the call itself can say the key went.
    """
    recorded = _recorded_compiler_kw(monkeypatch, "jax", "jit")
    adapter = JaxAdapter()
    adapter.compile_kw = {
        "static_argnums": [0],
        "static_argnames": "batch",
        "donate_argnums": [1],
        "donate_argnames": "batch",
        "inline": True,
    }
    model = _model()
    segment = _segment(model)
    adapter.prepare([segment])

    train_step = adapter.build_train_step([segment])
    adapter.build_inference_step(_inference_flow(model), models=[model])

    assert recorded == [{"inline": True}, {"inline": True}]
    # What survives still has to trace and compute; the recorded call alone cannot say that it did.
    assert np.isfinite(_run(train_step, _batch(), 1)[0])


@pytest.mark.skipif(
    keras.backend.backend() != "tensorflow", reason="Only the TensorFlow adapter takes an input signature."
)
def test_the_tensorflow_adapter_drops_an_input_signature_it_cannot_trace_with(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The steps built here take their batch by name, which is exactly what a signature forbids.

    `tf.function` replaces the traced function's parameters with the signature it is given, so a
    step declared as `**batch` is left unable to bind the batch it is called with. That one key
    goes and no other: `reduce_retracing` below is the caller's to set and reaches the trace. The
    value is not even a `tf.TensorSpec`, so a forwarded one fails at construction -- the run says
    the drop happened, the recorded call says nothing else was dropped with it.
    """
    recorded = _recorded_compiler_kw(monkeypatch, "tf", "function")
    adapter = TensorFlowAdapter()
    adapter.compile_kw = {"input_signature": ["not a tf.TensorSpec"], "reduce_retracing": True}
    model = _model()

    step = adapter.build_inference_step(_inference_flow(model), models=[model])

    assert recorded == [{"reduce_retracing": True}]
    assert np.isfinite(_run(step, _batch(), 1)[0])


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
