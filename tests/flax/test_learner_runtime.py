"""Runtime tests for the learner modules `FlaxLearnerBuilder` generates.

The generated script is exec'd from a file, the way a run would import it, and driven with real
`flax.nnx` models: everything these tests assert -- who owns which parameters, when an update lands,
what the reported rate is -- is only decided when the emitted code actually runs.
"""

from importlib.util import module_from_spec, spec_from_file_location
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
from structcast_model.utils.base import load_any
from tests import FIXTURES_DIR

CFG_DIR = FIXTURES_DIR / "cfg" / "flax"
MODEL_YAML = CFG_DIR / "Linear.yaml"
LEARNER_YAML = CFG_DIR / "LinearLearner.yaml"
SEGMENTS_YAML = CFG_DIR / "TwoSegmentLearner.yaml"

X = jnp.asarray([[1.0, 0.5, -0.5, 2.0], [0.0, 1.0, 1.0, -1.0]])
"""One fixed batch, so a loss that moves can only be the optimizer's doing."""

Y = jnp.asarray([[1.0, -1.0], [0.5, 0.25]])


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
    """Read the parameter arrays of a model, in a stable order."""
    return jax.tree.leaves(nnx.state(model, nnx.Param))


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
    for step in range(3):
        learner.update(step)
        losses.append(float(learner.training_step(x=X, y=Y)["loss"]))

    assert losses == sorted(losses, reverse=True)
    assert losses[-1] < losses[0]


def test_accumulated_gradients_apply_only_on_the_gated_step(tmp_path: Path) -> None:
    """With a `MultiSteps` window of 3 the parameters may move on every third step and no other.

    The accumulation now lives inside the optimizer state on the device; the generated `update`
    bakes the same window as a host formula, so the two must agree on which step the apply lands.
    """
    learner = _learner_type(tmp_path, parameters={"DEFAULT": {"accumulate_gradients": 3}})(
        _model_type(tmp_path)(rngs=nnx.Rngs(0))
    )
    before = _parameters(learner.models["model"])

    gates = []
    for step in range(3):
        gates.append(learner.update(step))
        learner.training_step(x=X, y=Y)
        moved = not all(
            jnp.array_equal(a, b) for a, b in zip(before, _parameters(learner.models["model"]), strict=True)
        )
        assert moved == (step == 2)

    assert gates == [False, False, True]
    assert not any(jnp.array_equal(a, b) for a, b in zip(before, _parameters(learner.models["model"]), strict=True))


def test_each_optimizer_moves_only_the_models_it_owns(tmp_path: Path) -> None:
    """Two segments, two containers: each optimizer applies its own rate to its own parameters.

    The first optimizer owns a two-model `flax.nnx.List`, the second a single model; a container
    that leaked into the other segment would show up as the wrong rate on the wrong model.
    """
    model_type = _model_type(tmp_path)
    models = [model_type(rngs=nnx.Rngs(seed)) for seed in range(3)]
    expected = [_sgd_step(models[0], 0.1), _sgd_step(models[1], 0.1), _sgd_step(models[2], 0.01)]
    learner = _learner_type(tmp_path, SEGMENTS_YAML)(*models)

    learner.update(0)
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
    learner.update(0)
    learner.training_step(x=X, y=Y)

    trained = _parameters(learner.models["model"])
    viewed = _parameters(learner._views["model"])

    assert all(jnp.array_equal(a, b) for a, b in zip(trained, viewed, strict=True))


def test_learning_rate_is_nan_until_a_step_reports_it(tmp_path: Path) -> None:
    """The rate is read inside the step, so it is only known once one has run."""
    learner = _learner_type(tmp_path)(_model_type(tmp_path)(rngs=nnx.Rngs(0)))

    assert jnp.isnan(jnp.asarray(learner.learning_rates["optimizer"]))

    learner.update(0)
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

    learner.update(0)
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

    learner.update(0)
    losses = [float(learner.training_step(x=X, y=Y)["loss"]) for _ in range(2)]

    assert losses[1] < losses[0]


@pytest.mark.parametrize(
    ("path", "models", "accumulate", "rates"),
    [
        (LEARNER_YAML, 1, 3, {"optimizer": pytest.approx(0.1)}),
        (SEGMENTS_YAML, 3, 2, {"optimizer_ab": pytest.approx(0.1), "optimizer_c": pytest.approx(0.01)}),
    ],
    ids=["one-segment", "two-segments"],
)
def test_compiled_training_step_never_retraces_across_the_window(
    tmp_path: Path, path: Path, models: int, accumulate: int, rates: dict[str, Any]
) -> None:
    """The step is compilable at the seam the learner exposes, and one trace covers the whole run.

    The accumulation gate lives in the `MultiSteps` state on the device, so no flag flips between
    the accumulating and the applying step: a second trace would mean the window leaked back into
    the step's signature. The seam is the one the trainers already use: every key of
    `flow_functions` is an attribute holding the current implementation, so compiling is
    `setattr(learner, name, compile(getattr(learner, name)))`. Donating the two state arguments has
    to be safe there, on every segment count -- a buffer the caller still holds would make XLA warn
    and silently copy instead.
    """
    model_type = _model_type(tmp_path)
    learner = _learner_type(tmp_path, path, parameters={"DEFAULT": {"accumulate_gradients": accumulate}})(
        *[model_type(rngs=nnx.Rngs(seed)) for seed in range(models)]
    )
    step = learner._training_step
    traces: list[bool] = []

    def counted(models: Any, optimizers: Any, **kwargs: Any) -> Any:
        """Record one entry per trace: the body of a compiled function runs only when it is traced."""
        traces.append(True)
        return step(models, optimizers, **kwargs)

    def compiled(name: str, function: Any) -> Any:
        """Compile one step the way a trainer does: the two state arguments donated."""
        if name == "_inference_step":
            return nnx.jit(function)
        return nnx.jit(function, donate_argnames=("models", "optimizers"))

    learner._training_step = counted
    for name, function in learner.flow_functions.items():
        setattr(learner, name, compiled(name, function))
    with catch_warnings(record=True) as caught:
        simplefilter("always")
        for index in range(6):
            learner.update(index)
            learner.training_step(x=X, y=Y)

    assert [str(w.message) for w in caught if "donated" in str(w.message)] == []
    assert traces == [True]
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

    for step in range(2):
        learner.update(step)
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

    learner.update(0)
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


def test_a_generated_accumulating_learner_updates_twice_over_six_trainer_steps(tmp_path: Path) -> None:
    """The accumulation window is the learner's, and the trainer must see exactly its updates.

    `BaseTrainer` asks the learner whether this step updates and counts the answer, so a window the
    generated `update` got wrong -- one update too many, or a gate out of phase with the on-device
    `MultiSteps` counter -- shows up here as an update count that no longer matches the configured
    window. Six steps at three is the smallest run that closes two windows.

    The events land on steps 2 and 5, not 3 and 6: the emitted gate is `(step + 1) % 3 == 0`, which
    counts from zero, while the trainer's own step counter starts at one. The first window is
    therefore one step short and every later one is exactly three. This is what the PyTorch learners
    emit too (`builders/torch.py`), so it is asserted as parity rather than fixed here.
    """
    learner = _learner_type(tmp_path, parameters={"DEFAULT": {"accumulate_gradients": 3}})(
        _model_type(tmp_path)(rngs=nnx.Rngs(0))
    )
    recorder = _UpdateRecorder()
    trainer = FlaxTrainer(
        learner=learner,
        tracker=FlaxTracker.from_criteria(learner.outputs),
        data=SimpleDataProvider(training_dataset=[{"x": X, "y": Y}] * 6),
        callbacks=[recorder],
    )

    trainer.fit(epochs=1)

    assert recorder.steps == [2, 5]
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
    learner.update(0)
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
