"""Runtime tests for the learner modules `FlaxLearnerBuilder` generates.

The generated script is exec'd from a file, the way a run would import it, and driven with real
`flax.nnx` models: everything these tests assert -- who owns which parameters, when an update lands,
what the reported rate is -- is only decided when the emitted code actually runs.
"""

from functools import wraps
from importlib.util import module_from_spec, spec_from_file_location
from inspect import Parameter, signature
import logging
from pathlib import Path
from types import ModuleType
from typing import Any
from warnings import catch_warnings, simplefilter

import jax
import jax.numpy as jnp
import pytest

from flax import nnx
from structcast_model.base_trainer import Learner, SimpleDataProvider
from structcast_model.builders.flax import FlaxBuilder, FlaxLearnerBuilder
from structcast_model.flax.trainer import FlaxTracker, FlaxTrainer
from structcast_model.flax.utils import donate_argnames
from structcast_model.utils.base import load_any
from tests import FIXTURES_DIR

CFG_DIR = FIXTURES_DIR / "cfg" / "flax"
MODEL_YAML = CFG_DIR / "Linear.yaml"
LEARNER_YAML = CFG_DIR / "LinearLearner.yaml"
SEGMENTS_YAML = CFG_DIR / "TwoSegmentLearner.yaml"

X = jnp.asarray([[1.0, 0.5, -0.5, 2.0], [0.0, 1.0, 1.0, -1.0]])
"""One fixed batch, so a loss that moves can only be the optimizer's doing."""

Y = jnp.asarray([[1.0, -1.0], [0.5, 0.25]])

POISONED = X.at[0, 0].set(jnp.nan)
"""The fixed batch with one NaN in it, which is what makes a step's gradients come back non-finite.

Injected through the batch rather than into the gradients: the whole path a loss scale sits on --
the scaled flow, the unscaling, the finiteness check and the skip -- then runs the way a real step
runs it, and nothing inside the generated learner has to be reached into.
"""


def _load(path: Path, name: str) -> ModuleType:
    """Load a generated module by file path, the way a configuration does: not by import name."""
    spec = spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _model_type(tmp_path: Path) -> Any:
    """Generate the tiny linear model the learner fixtures are written against."""
    FlaxBuilder.from_path(MODEL_YAML)()(tmp_path / "model.py")
    return _load(tmp_path / "model.py", "generated_model").Model


def _learner_type(tmp_path: Path, path: Path = LEARNER_YAML, **kwargs: Any) -> Any:
    """Generate a learner from a fixture configuration and return its class."""
    FlaxLearnerBuilder.from_path(path)(**kwargs)(tmp_path / "learner.py")
    return _load(tmp_path / "learner.py", "generated_learner").Learner


def _parameters(model: Any) -> list[jax.Array]:
    """Read the parameter arrays of a model, in a stable order.

    Copied, because a compiled step is handed these buffers donated and deletes them: a snapshot
    taken to be compared after the step has to outlive the step that consumed it.
    """
    return [jnp.copy(leaf) for leaf in jax.tree.leaves(nnx.state(model, nnx.Param))]


def _optimizer_state(learner: Any, name: str = "optimizer") -> list[jax.Array]:
    """Read the optimizer's own state arrays -- its step count above all -- in a stable order.

    Through `nnx.to_pure_dict`: a tree walk over an `nnx.State` yields Variables, which do not
    compare as arrays.
    """
    return jax.tree.leaves(nnx.to_pure_dict(nnx.state(learner.optimizers[name])))


def _sgd_step(model: Any, learning_rate: float) -> tuple[jax.Array, jax.Array]:
    """Return the kernel and bias one plain SGD step on the fixed batch should produce.

    The fixture models are a single `flax.nnx.Linear`, so the expected update is analytic: it does
    not go through the generated code, the optimizer or the builder.
    """
    kernel, bias = model.fc.kernel[...], model.fc.bias[...]
    gradients = jax.grad(lambda p: jnp.mean((X @ p[0] + p[1] - Y) ** 2))((kernel, bias))
    return kernel - learning_rate * gradients[0], bias - learning_rate * gradients[1]


def test_generated_learner_satisfies_the_trainer_learner_protocol(tmp_path: Path) -> None:
    """The trainer drives learners through the protocol only, so a missing member breaks every run."""
    learner = _learner_type(tmp_path)(_model_type(tmp_path)(rngs=nnx.Rngs(0)))

    assert learner.outputs == ["loss"]
    assert learner.optimizer_models == {"optimizer": ["model"]}
    assert sorted(learner.flow_functions) == ["_inference_step", "_training_step"]
    assert isinstance(learner, Learner)


def test_training_step_lowers_the_loss_it_reports(tmp_path: Path) -> None:
    """Three steps on one batch must bring the reported loss down monotonically.

    A learner that differentiated the wrong argument, or applied the gradients to a copy of the
    model, would still return finite losses -- they just would not move.
    """
    learner = _learner_type(tmp_path)(_model_type(tmp_path)(rngs=nnx.Rngs(0)))

    losses = []
    for _ in range(3):
        losses.append(float(learner.training_step(x=X, y=Y)["loss"]))
        assert learner.has_updated is True

    assert losses == sorted(losses, reverse=True)
    assert losses[-1] < losses[0]


@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "compiled"])
@pytest.mark.parametrize(
    ("window", "gates"),
    [(3, [False, False, True]), (2, [False, True, False, True])],
    ids=["window-of-three", "window-of-two"],
)
def test_accumulated_gradients_apply_only_on_the_gated_step(
    tmp_path: Path, window: int, gates: list[bool], compiled: bool
) -> None:
    """With a `MultiSteps` window the parameters may move on every k-th step and on no other.

    The accumulation lives inside the optimizer state on the device, and the step compares the count
    it advanced across its own `update` call, so `has_updated` must agree, step by step, with which
    step the parameters actually moved on -- and `updates` must count exactly those steps.

    Compiled as well as eager, under the donation the command derives from the step's signature:
    that is the only shape a real run takes, and a window whose counter did not survive tracing --
    or whose state the donated buffers dropped -- would leave every step accumulating into a window
    that never closes, which reads as a model that trains without ever moving.
    """
    learner = _learner_type(tmp_path, parameters={"DEFAULT": {"accumulate_gradients": window}})(
        _model_type(tmp_path)(rngs=nnx.Rngs(0))
    )
    if compiled:
        step = learner._training_step
        learner._training_step = nnx.jit(step, donate_argnames=donate_argnames(step))
    previous = _parameters(learner.models["model"])

    reported, moved = [], []
    for _ in gates:
        learner.training_step(x=X, y=Y)
        reported.append(learner.has_updated)
        current = _parameters(learner.models["model"])
        moved.append(not all(jnp.array_equal(a, b) for a, b in zip(previous, current, strict=True)))
        previous = current

    assert reported == gates
    assert moved == gates
    assert (learner.steps, learner.updates) == (len(gates), sum(gates))


@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "compiled"])
def test_a_scaled_learner_backs_off_and_grows_over_the_steps_it_skipped(tmp_path: Path, compiled: bool) -> None:
    """The scale is the mechanism, so the assertion is its trajectory, not a flag.

    One non-finite step halves it, the finite ones after it hold it, and it doubles once the run of
    finite steps reaches the growth interval. The parameters say the same from the other side: they
    may not move on the step the scale rejected and must move on every other one. The counters keep
    reporting the apply that was attempted, as the torch gradient scaler's do -- a skipped apply is
    still an update the schedules it drives are indexed by.

    Compiled as well as eager, under the donation the command derives from the step's signature: the
    scale is one more donated argument and one more result there, and a carry that did not survive
    that round trip would leave every step scaling by the value the learner was built with.
    """
    learner = _learner_type(tmp_path, parameters={"DEFAULT": {"mixed_precision": {"growth_interval": 2}}})(
        _model_type(tmp_path)(rngs=nnx.Rngs(0))
    )
    assert donate_argnames(learner._training_step) == ("model", "optimizer", "optimizer_dynamic_scale")
    if compiled:
        step = learner._training_step
        learner._training_step = nnx.jit(step, donate_argnames=donate_argnames(step))
    previous = _parameters(learner.models["model"])

    scales, moved = [], []
    for batch in (POISONED, X, X, X):
        learner.training_step(x=batch, y=Y)
        scales.append(float(learner.grad_scalers["optimizer_dynamic_scale"].scale))
        current = _parameters(learner.models["model"])
        moved.append(not all(jnp.array_equal(a, b) for a, b in zip(previous, current, strict=True)))
        previous = current

    assert scales == [32768.0, 32768.0, 32768.0, 65536.0]
    assert moved == [False, True, True, True]
    assert (learner.steps, learner.updates, learner.has_updated) == (4, 4, True)


def test_an_overflowed_micro_step_pauses_the_window_it_landed_in(tmp_path: Path) -> None:
    """A window whose micro-step overflowed drops that micro-step, not the window.

    The skip rolls the optimizer state back, and an `optax.MultiSteps` keeps its accumulator and its
    window counter there, so the non-finite gradients never enter the accumulation and the window
    closes one step later than it would have. The torch twin drops the whole window instead: its
    accumulator is each parameter's own gradient buffer, which a scaler can only discard entire.
    """
    parameters = {"DEFAULT": {"accumulate_gradients": 2, "mixed_precision": True}}
    learner = _learner_type(tmp_path, parameters=parameters)(_model_type(tmp_path)(rngs=nnx.Rngs(0)))
    previous = _parameters(learner.models["model"])

    moved = []
    for batch in (POISONED, X, X):
        learner.training_step(x=batch, y=Y)
        current = _parameters(learner.models["model"])
        moved.append(not all(jnp.array_equal(a, b) for a, b in zip(previous, current, strict=True)))
        previous = current

    assert moved == [False, False, True]
    assert (learner.steps, learner.updates) == (3, 1)


def test_each_optimizer_moves_only_the_models_it_owns(tmp_path: Path) -> None:
    """Two segments: each optimizer applies its own rate to the parameters it owns, and to no other.

    The first optimizer owns two models, passed to it and to its update as a plain tuple, the second
    a single model; a segment reaching into the other would show up as the wrong rate on the wrong
    model.
    """
    model_type = _model_type(tmp_path)
    models = [model_type(rngs=nnx.Rngs(seed)) for seed in range(3)]
    expected = [_sgd_step(models[0], 0.1), _sgd_step(models[1], 0.1), _sgd_step(models[2], 0.01)]
    learner = _learner_type(tmp_path, SEGMENTS_YAML)(*models)

    learner.training_step(x=X, y=Y)

    for model, (kernel, bias) in zip(models, expected, strict=True):
        assert jnp.allclose(model.fc.kernel[...], kernel, atol=1e-6)
        assert jnp.allclose(model.fc.bias[...], bias, atol=1e-6)


class _Dropped(nnx.Module):
    """A model with dropout, to show that the inference views actually turn it off."""

    def __init__(self, *, rngs: nnx.Rngs) -> None:
        """Build the linear layer the fixture learner feeds and the dropout that follows it."""
        self.fc = nnx.Linear(4, 2, rngs=rngs)
        self.dropout = nnx.Dropout(0.9, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        """Run the layer, dropping activations unless the caller's view says otherwise."""
        return self.dropout(self.fc(x))


def test_inference_step_runs_deterministically_and_leaves_the_parameters_untouched(tmp_path: Path) -> None:
    """Evaluation must be repeatable and must not train.

    The views are built once with `deterministic=True`, so a model that drops 90% of its activations
    while training returns the same criteria twice in a row here, without an optimizer ever stepping.
    """
    model = _Dropped(rngs=nnx.Rngs(0))
    learner = _learner_type(tmp_path)(model)
    before = _parameters(model)
    optimizer_state = _optimizer_state(learner)

    first = learner.inference_step(x=X, y=Y)
    second = learner.inference_step(x=X, y=Y)

    assert float(first["loss"]) == float(second["loss"])
    assert all(jnp.array_equal(a, b) for a, b in zip(before, _parameters(model), strict=True))
    # The optimizer's own state has to be bitwise untouched too: an evaluation that stepped it would
    # leave the parameters alone on this fixture and only show up as a drifted schedule later.
    assert all(jnp.array_equal(a, b) for a, b in zip(optimizer_state, _optimizer_state(learner), strict=True))
    # The same batch through the trained models drops activations, so it cannot report the same loss.
    assert float(learner.training_step(x=X, y=Y)["loss"]) != float(first["loss"])
    # ... and that training step does move the optimizer state, so the comparison above can fail.
    assert not all(jnp.array_equal(a, b) for a, b in zip(optimizer_state, _optimizer_state(learner), strict=True))


def test_inference_views_see_the_trained_parameters(tmp_path: Path) -> None:
    """The views share arrays with the models, so validation reads what the last update wrote."""
    learner = _learner_type(tmp_path)(_model_type(tmp_path)(rngs=nnx.Rngs(0)))
    learner.training_step(x=X, y=Y)

    trained = _parameters(learner.models["model"])
    viewed = _parameters(learner._view_model)

    assert all(jnp.array_equal(a, b) for a, b in zip(trained, viewed, strict=True))


def test_learning_rate_is_nan_until_a_step_reports_it(tmp_path: Path) -> None:
    """The rate is read inside the step, so it is only known once one has run."""
    learner = _learner_type(tmp_path)(_model_type(tmp_path)(rngs=nnx.Rngs(0)))

    assert jnp.isnan(jnp.asarray(learner.learning_rates["optimizer"]))

    learner.training_step(x=X, y=Y)

    assert learner.learning_rates == {"optimizer": pytest.approx(0.1)}


def test_learning_rate_stays_nan_when_the_pattern_hides_it(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """A positionally passed rate cannot be injected, and NaN is how that is reported at run time."""
    raw = load_any(LEARNER_YAML)
    raw["LEARNERS"][0]["OPTIMIZER"][2] = {"_bind_": {"tx": ["_obj_", {"_addr_": "optax.sgd"}, ["_call_", 0.1]]}}
    with caplog.at_level(logging.WARNING):
        built = FlaxLearnerBuilder(raw=raw, current_path=str(LEARNER_YAML))()
    assert "reports no learning rate" in caplog.text
    built(tmp_path / "hidden.py")
    learner = _load(tmp_path / "hidden.py", "hidden_learner").Learner(_model_type(tmp_path)(rngs=nnx.Rngs(0)))

    losses = [float(learner.training_step(x=X, y=Y)["loss"]) for _ in range(2)]

    assert jnp.isnan(jnp.asarray(learner.learning_rates["optimizer"]))
    assert losses[1] < losses[0]


def test_a_value_no_later_code_reads_never_leaves_the_differentiated_closure(tmp_path: Path) -> None:
    """A flow may compute a value that is not an array, and only the closure has to see it.

    The auxiliary tuple is what the differentiated closure hands back to the enclosing step, so it
    has to carry the criteria and nothing else here: the python string this flow computes stays a
    local of the closure, and the step still trains.
    """
    raw = load_any(LEARNER_YAML)
    raw["LEARNERS"][0]["FLOW"].insert(0, ["eval: 'tag'", "tag", None])
    FlaxLearnerBuilder(raw=raw, current_path=str(LEARNER_YAML))()(tmp_path / "tagged.py")
    learner = _load(tmp_path / "tagged.py", "tagged_learner").Learner(_model_type(tmp_path)(rngs=nnx.Rngs(0)))

    assert "tag = 'tag'" in (tmp_path / "tagged.py").read_text()
    assert "return loss, (loss,)" in (tmp_path / "tagged.py").read_text()

    losses = [float(learner.training_step(x=X, y=Y)["loss"]) for _ in range(2)]

    assert losses[1] < losses[0]


@pytest.mark.parametrize(
    ("path", "models", "accumulate", "rates", "scaled"),
    [
        (LEARNER_YAML, 1, 3, {"optimizer": pytest.approx(0.1)}, False),
        (SEGMENTS_YAML, 3, 2, {"optimizer_ab": pytest.approx(0.1), "optimizer_c": pytest.approx(0.01)}, False),
        (LEARNER_YAML, 1, 3, {"optimizer": pytest.approx(0.1)}, True),
    ],
    ids=["one-segment", "two-segments", "one-segment-scaled"],
)
def test_compiled_training_step_never_retraces_across_the_window(
    tmp_path: Path, path: Path, models: int, accumulate: int, rates: dict[str, Any], scaled: bool
) -> None:
    """The step is compilable at the seam the learner exposes, and one trace covers the whole run.

    The accumulation gate lives in the `MultiSteps` state on the device, so no flag flips between
    the accumulating and the applying step: a second trace would mean the window leaked back into
    the step's signature. The seam is the one the trainers already use: every key of
    `flow_functions` is an attribute holding the current implementation, so compiling is
    `setattr(learner, name, compile(getattr(learner, name)))`. Donating every state parameter the
    step declares has to be safe there, on every segment count -- a buffer the caller still holds
    would make XLA warn and silently copy instead. A loss scale is one more such parameter and one
    more result, and a carry whose element types changed between two calls would trace twice.
    """
    model_type = _model_type(tmp_path)
    parameters = {"DEFAULT": {"accumulate_gradients": accumulate, "mixed_precision": scaled}}
    learner = _learner_type(tmp_path, path, parameters=parameters)(
        *[model_type(rngs=nnx.Rngs(seed)) for seed in range(models)]
    )
    step = learner._training_step
    traces: list[bool] = []

    @wraps(step)
    def counted(*state: Any, **kwargs: Any) -> Any:
        """Record one entry per trace: the body of a compiled function runs only when it is traced."""
        traces.append(True)
        return step(*state, **kwargs)

    def compiled(name: str, function: Any) -> Any:
        """Compile one step the way the CLI does: every positional-or-keyword parameter donated."""
        if name == "_inference_step":
            return nnx.jit(function)
        donated = [p.name for p in signature(function).parameters.values() if p.kind is Parameter.POSITIONAL_OR_KEYWORD]
        return nnx.jit(function, donate_argnames=tuple(donated))

    learner._training_step = counted
    for name, function in learner.flow_functions.items():
        setattr(learner, name, compiled(name, function))
    with catch_warnings(record=True) as caught:
        simplefilter("always")
        for _ in range(6):
            learner.training_step(x=X, y=Y)

    assert [str(w.message) for w in caught if "donated" in str(w.message)] == []
    assert traces == [True]
    assert learner.updates == 6 // accumulate
    assert learner.learning_rates == rates


def test_a_training_forward_draws_exactly_one_dropout_key(tmp_path: Path) -> None:
    """A stochastic layer must advance its stream once per logical forward, and never during evaluation.

    Advancing twice would make the mask depend on how many times the step internally calls the model
    -- the differentiated flow runs it once -- and advancing during evaluation would make validation
    depend on how often it ran. Both are invisible in the criteria and only surface as a run that
    cannot be reproduced from its seed.
    """
    model = _Dropped(rngs=nnx.Rngs(0))
    learner = _learner_type(tmp_path)(model)
    counts = []

    for _ in range(2):
        learner.training_step(x=X, y=Y)
        counts.append(int(model.dropout.rngs.count[...]))
    before_inference = counts[-1]
    learner.inference_step(x=X, y=Y)

    assert counts == [1, 2]
    assert int(model.dropout.rngs.count[...]) == before_inference


class _Normalized(nnx.Module):
    """A model with batch normalization, whose running statistics are state the run has to carry."""

    def __init__(self, *, rngs: nnx.Rngs) -> None:
        """Build the linear layer the fixture learner feeds and the norm that follows it."""
        self.fc = nnx.Linear(4, 2, rngs=rngs)
        self.bn = nnx.BatchNorm(2, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        """Normalize the layer's output, in whichever mode the caller's view selected."""
        return self.bn(self.fc(x))


def test_running_statistics_move_while_training_and_hold_while_evaluating(tmp_path: Path) -> None:
    """Normalization is stateful, and the two flows must treat that state differently.

    A training step has to fold the batch into the running mean and variance; an inference step has
    to read them and leave them alone. A view that missed `use_running_average=True` would keep
    updating them during validation, quietly training on the data the run is being measured on.
    """
    model = _Normalized(rngs=nnx.Rngs(0))
    learner = _learner_type(tmp_path)(model)
    initial = jax.tree.leaves(nnx.state(model, nnx.BatchStat))

    learner.training_step(x=X, y=Y)
    trained = jax.tree.leaves(nnx.state(model, nnx.BatchStat))
    learner.inference_step(x=X, y=Y)
    evaluated = jax.tree.leaves(nnx.state(model, nnx.BatchStat))

    assert not any(jnp.array_equal(a, b) for a, b in zip(initial, trained, strict=True))
    assert all(jnp.array_equal(a, b) for a, b in zip(trained, evaluated, strict=True))


class _UpdateRecorder:
    """A callback recording the trainer step of every update event it receives."""

    def __init__(self) -> None:
        """Start with no updates recorded."""
        self.steps: list[int] = []

    def on_update(self, info: Any) -> None:
        """Record the step the update landed on."""
        self.steps.append(info.step)


class _MovementRecorder:
    """A callback recording, per trainer step, whether the model's parameters moved."""

    def __init__(self, model: nnx.Module) -> None:
        """Snapshot the parameters the trainer is about to move."""
        self.model = model
        self.previous = jax.tree.leaves(jax.tree.map(jnp.copy, nnx.state(model, nnx.Param)))
        self.moved_steps: list[int] = []

    def on_training_step_end(self, info: Any) -> None:
        """Record the step when any parameter changed since the previous step."""
        current = jax.tree.leaves(jax.tree.map(jnp.copy, nnx.state(self.model, nnx.Param)))
        if any(not jnp.array_equal(a, b) for a, b in zip(self.previous, current, strict=True)):
            self.moved_steps.append(info.step)
        self.previous = current


def test_a_generated_accumulating_learner_updates_twice_over_six_trainer_steps(tmp_path: Path) -> None:
    """The update event must land on the step the parameters actually move.

    `on_update` consumers (a per-update LR scheduler, the update counter) act on the step the
    optimizer applies, and with `MultiSteps` the apply happens on the device, on the k-th
    `optimizer.update` call. The learner reads the applied count back from the optimizer state
    after each step (`docs/adr/0018`), so the event may only fire on the step the weights actually
    move -- which this test pins by asserting events and movement on the same steps. Six steps at
    three is the smallest run that closes two windows. The cadence follows the native mechanism,
    not the torch learners' historically short first window (`docs/adr/0017`).
    """
    model = _model_type(tmp_path)(rngs=nnx.Rngs(0))
    learner = _learner_type(tmp_path, parameters={"DEFAULT": {"accumulate_gradients": 3}})(model)
    recorder = _UpdateRecorder()
    movement = _MovementRecorder(model)
    trainer = FlaxTrainer(
        learner=learner,
        tracker=FlaxTracker.from_criteria(learner.outputs),
        data=SimpleDataProvider(training_dataset=[{"x": X, "y": Y}] * 6),
        callbacks=[recorder, movement],
    )

    trainer.fit(epochs=1)

    assert recorder.steps == [3, 6]
    assert movement.moved_steps == [3, 6]
    assert trainer.update == 2


SGD_TX = ["_obj_", {"_addr_": "optax.sgd"}, {"_call_": {"learning_rate": 1.0}}]
"""A transformation whose step is the raw gradient: whatever the clip lets through is what moves."""

CLIPPED_TX = [
    "_obj_",
    {"_addr_": "optax.chain"},
    {"_call_": [["_obj_", {"_addr_": "optax.clip_by_global_norm"}, {"_call_": {"max_norm": 1e-3}}], SGD_TX]},
]

LOUD_Y = 100.0 * Y
"""Targets far from anything the model predicts, so an unclipped step is orders of magnitude too big."""


def _tx_learner(tmp_path: Path, name: str, tx: list[Any]) -> Any:
    """Generate a learner from the linear fixture with *tx* as its optimizer's transformation."""
    raw = load_any(LEARNER_YAML)
    raw["LEARNERS"][0]["OPTIMIZER"][2] = {"_bind_": {"tx": tx}}
    FlaxLearnerBuilder(raw=raw, current_path=str(LEARNER_YAML))()(tmp_path / f"{name}.py")
    return _load(tmp_path / f"{name}.py", f"{name}_learner").Learner


def _step_distance(learner: Any) -> float:
    """Run one training step on the loud batch and return how far the parameters moved."""
    before = _parameters(learner.models["model"])
    learner.training_step(x=X, y=LOUD_Y)
    after = _parameters(learner.models["model"])
    return float(jnp.sqrt(sum(jnp.sum((a - b) ** 2) for a, b in zip(before, after, strict=True))))


def test_a_chained_clip_bounds_what_a_generated_update_actually_moves(tmp_path: Path) -> None:
    """A transformation the pattern chains before the optimizer has to reach the applied update.

    The builder emits the pattern and appends the owned container to it; nothing in the emitted code
    re-implements the transformation, so the only proof that the chain survived is the parameters
    moving no further than the clip allows. The twin without the clip moves on the same batch, so a
    step that simply did nothing could not pass.
    """
    model_type = _model_type(tmp_path)
    clipped = _tx_learner(tmp_path, "clipped", CLIPPED_TX)(model_type(rngs=nnx.Rngs(0)))
    plain = _tx_learner(tmp_path, "plain", SGD_TX)(model_type(rngs=nnx.Rngs(0)))

    # The bound is the clip's, up to the float32 error of summing the squared deltas back up.
    assert _step_distance(clipped) <= 1e-3 * (1 + 1e-4)
    assert _step_distance(plain) > 1.0


def _multi_steps_tx(**arguments: Any) -> list[Any]:
    """Build a `MultiSteps` tx pattern over the fixture's sgd, with *arguments* as extra keywords."""
    inner = ["_obj_", {"_addr_": "optax.sgd"}, {"_call_": {"learning_rate": 0.1}}]
    return ["_obj_", {"_addr_": "optax.MultiSteps"}, {"_call_": {"opt": inner, **arguments}}]


READING_SEGMENT = [
    ["eval: a.fc.kernel[...].mean()", "reg_c", None],
    ["x", "out_c", "c"],
    [{"predictions": "out_c", "targets": "y"}, "errors_c", "mse"],
    ["eval: errors_c.mean() + reg_c", "loss_c", None],
]
"""A second segment whose loss reads a parameter of the model the first segment trains."""


def test_a_model_a_segment_only_reads_is_not_frozen_into_the_compiled_step(tmp_path: Path) -> None:
    """A model read in an expression has to reach the flow as an argument, not from `__init__`.

    Read from the enclosing scope, it is a constant to the tracer: the compiled step would keep
    using the values that model had when the learner was built, while the eager one follows the
    updates the other segment applies. Nothing but running both says which one happened, and they
    can only part once the first segment has moved the model -- from the second step on.
    """
    raw = load_any(SEGMENTS_YAML)
    raw["LEARNERS"][1]["FLOW"] = READING_SEGMENT
    raw["LEARNERS"][1]["INFERENCE_FLOW"] = READING_SEGMENT
    FlaxLearnerBuilder(raw=raw, current_path=str(SEGMENTS_YAML))()(tmp_path / "reading.py")
    learner_type = _load(tmp_path / "reading.py", "reading_learner").Learner
    model_type = _model_type(tmp_path)

    def _run(compiled: bool) -> list[float]:
        """Run two steps of a fresh learner, its training step compiled or not."""
        learner = learner_type(*[model_type(rngs=nnx.Rngs(seed)) for seed in range(3)])
        if compiled:
            learner._training_step = nnx.jit(
                learner._training_step, donate_argnames=("a", "b", "c", "optimizer_ab", "optimizer_c")
            )
        return [float(learner.training_step(x=X, y=Y)["loss_c"]) for _ in range(2)]

    eager, jitted = _run(compiled=False), _run(compiled=True)

    assert jitted == pytest.approx(eager, rel=1e-6)
    assert eager[1] != eager[0]


def test_the_first_optimizer_is_the_clock_of_a_learner_whose_segments_are_out_of_phase(tmp_path: Path) -> None:
    """One learner, one update count: the first segment decides what an Update is for the whole run.

    The trainer's update counter answers for the learner, and the segments need not share a window:
    here the first optimizer accumulates over two steps while the second applies on every one, and
    the counted updates follow the first alone.
    """
    raw = load_any(SEGMENTS_YAML)
    raw["LEARNERS"][0]["OPTIMIZER"][2] = {"_bind_": {"tx": _multi_steps_tx(every_k_schedule=2)}}
    FlaxLearnerBuilder(raw=raw, current_path=str(SEGMENTS_YAML))()(tmp_path / "dephased.py")
    model_type = _model_type(tmp_path)
    learner_type = _load(tmp_path / "dephased.py", "dephased_learner").Learner
    learner = learner_type(*[model_type(rngs=nnx.Rngs(seed)) for seed in range(3)])

    gates = []
    for _ in range(4):
        learner.training_step(x=X, y=Y)
        gates.append(learner.has_updated)

    assert gates == [False, True, False, True]
    assert (learner.steps, learner.updates) == (4, 2)


def test_restore_counters_seeds_both_counts_from_the_checkpoint(tmp_path: Path) -> None:
    """A resumed run continues its own clocks, which only the saved meta knows.

    Neither count is recoverable from the optimizer state -- a window of one leaves no counter at
    all -- so both are seeded as given, and the next step continues from them.
    """
    learner = _learner_type(tmp_path)(_model_type(tmp_path)(rngs=nnx.Rngs(0)))

    learner.restore_counters(7, 3)

    assert (learner.steps, learner.updates) == (7, 3)

    learner.training_step(x=X, y=Y)

    assert (learner.steps, learner.updates) == (8, 4)


@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "compiled"])
def test_a_half_precision_model_trains_under_the_scale_the_learner_owns(tmp_path: Path, compiled: bool) -> None:
    """The two halves of a Flax float16 run are configured apart and have to meet here.

    Precision is the model's -- the `dtype` its generated `__init__` now takes, whatever its template
    threads -- and the loss scale is the learner's `MIXED_PRECISION`, so nothing pairs them at build
    time. What this asserts is that they compose: the activations are float16 over the fp32 master
    weights, the scaled step still brings the loss down, and the counters still report the applies.

    The constructor arguments are not step arguments, which is the other half. They are read once,
    in `__init__`, and what they leave behind is a static attribute of each layer, so the donation
    contract the command derives from the step's signature must be the float32 one, a compiled step
    must accept them without a new argument, and the inference views the learner runs validation
    through must see the same narrowed layers.

    The scale is started low on purpose: at the `DynamicScale` default of 65536 a gradient of order
    one overflows float16 on the first step, which is the mechanism working rather than the pairing
    failing, and it would leave three skipped steps to assert nothing about.
    """
    model = _model_type(tmp_path)(rngs=nnx.Rngs(0), dtype=jnp.float16)
    learner = _learner_type(tmp_path, parameters={"DEFAULT": {"mixed_precision": {"scale": 128.0}}})(model)

    assert model.fc.dtype is jnp.float16
    assert {str(leaf.dtype) for leaf in jax.tree.leaves(nnx.state(model, nnx.Param))} == {"float32"}
    assert donate_argnames(learner._training_step) == ("model", "optimizer", "optimizer_dynamic_scale")
    if compiled:
        step = learner._training_step
        learner._training_step = nnx.jit(step, donate_argnames=donate_argnames(step))
    before = _parameters(learner.models["model"])

    losses = [float(learner.training_step(x=X, y=Y)["loss"]) for _ in range(3)]
    after = _parameters(learner.models["model"])

    assert bool(jnp.all(jnp.isfinite(jnp.asarray(losses))))
    assert losses[-1] < losses[0]
    assert any(not jnp.array_equal(a, b) for a, b in zip(before, after, strict=True))
    assert (learner.steps, learner.updates) == (3, 3)
    # The inference path runs `nnx.view` copies of the models, which carry the layers themselves:
    # a view that had rebuilt them would evaluate a float32 model the run never trained.
    assert learner._view_model.fc.dtype is jnp.float16
    assert jnp.isfinite(learner.inference_step(x=X, y=Y)["loss"])
    assert all(jnp.array_equal(a, b) for a, b in zip(after, _parameters(learner.models["model"]), strict=True))
