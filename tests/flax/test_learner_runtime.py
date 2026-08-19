"""Runtime tests for the learner modules `FlaxLearnerBuilder` generates.

The generated script is exec'd from a file, the way a run would import it, and driven with real
`flax.nnx` models: everything these tests assert -- who owns which parameters, when an update lands,
what the reported rate is -- is only decided when the emitted code actually runs.
"""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType
from typing import Any
from warnings import catch_warnings, simplefilter

import jax
import jax.numpy as jnp
import pytest

from flax import nnx
from structcast_model.base_trainer import Learner
from structcast_model.builders.flax import FlaxBuilder, FlaxLearnerBuilder
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
    """With `ACCUMULATE_GRADIENTS: 3` the parameters may move on every third step and no other.

    The buffer has to hold the sum of the micro-step gradients meanwhile and be zeroed by the
    update, or the next window would apply the previous one's gradients a second time.
    """
    learner = _learner_type(tmp_path, parameters={"DEFAULT": {"accumulate_gradients": 3}})(
        _model_type(tmp_path)(rngs=nnx.Rngs(0))
    )
    before = _parameters(learner.models["model"])

    gates = []
    for step in range(3):
        gates.append(learner.update(step))
        learner.training_step(x=X, y=Y)
        buffered = float(sum(jnp.sum(jnp.abs(leaf)) for leaf in jax.tree.leaves(learner._acc_grads["optimizer"])))
        if step < 2:
            assert buffered > 0.0
            assert all(jnp.array_equal(a, b) for a, b in zip(before, _parameters(learner.models["model"]), strict=True))
        else:
            assert buffered == 0.0

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

    first = learner.inference_step(x=X, y=Y)
    second = learner.inference_step(x=X, y=Y)

    assert float(first["loss"]) == float(second["loss"])
    assert all(jnp.array_equal(a, b) for a, b in zip(before, _parameters(model), strict=True))
    # The same batch through the trained models drops activations, so it cannot report the same loss.
    assert float(learner.training_step(x=X, y=Y)["loss"]) != float(first["loss"])


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


def test_learning_rate_stays_nan_when_the_pattern_hides_it(tmp_path: Path) -> None:
    """A positionally passed rate cannot be injected, and NaN is how that is reported at run time."""
    raw = load_any(LEARNER_YAML)
    raw["LEARNERS"][0]["OPTIMIZER"][2]["_bind_"]["tx"] = ["_obj_", {"_addr_": "optax.sgd"}, ["_call_", 0.1]]
    with pytest.warns(UserWarning, match="reports no learning rate"):
        built = FlaxLearnerBuilder(raw=raw, current_path=str(LEARNER_YAML))()
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
def test_compiled_training_step_settles_at_one_variant_per_gate(
    tmp_path: Path, path: Path, models: int, accumulate: int, rates: dict[str, Any]
) -> None:
    """The step is compilable at the seam the learner exposes, and `need_update` is static.

    Two variants -- accumulate and apply -- is the whole cost: once both are compiled, further steps
    must add no trace, which is what a traced (rather than static) gate would break. The seam is the
    one the trainers already use: every key of `flow_functions` is an attribute holding the current
    implementation, so compiling is `setattr(learner, name, compile(getattr(learner, name)))`.
    Donating the three state arguments has to be safe there, on every segment count -- a buffer the
    caller still holds would make XLA warn and silently copy instead.
    """
    model_type = _model_type(tmp_path)
    learner = _learner_type(tmp_path, path, parameters={"DEFAULT": {"accumulate_gradients": accumulate}})(
        *[model_type(rngs=nnx.Rngs(seed)) for seed in range(models)]
    )
    step = learner._training_step
    traces: list[bool] = []

    def counted(models: Any, optimizers: Any, acc_grads: Any, need_update: bool, **kwargs: Any) -> Any:
        """Record one entry per trace: the body of a compiled function runs only when it is traced."""
        traces.append(need_update)
        return step(models, optimizers, acc_grads, need_update, **kwargs)

    def compiled(name: str, function: Any) -> Any:
        """Compile one step the way a trainer does: the gate static, the three state arguments donated."""
        if name == "_inference_step":
            return nnx.jit(function)
        return nnx.jit(function, static_argnames="need_update", donate_argnames=("models", "optimizers", "acc_grads"))

    learner._training_step = counted
    for name, function in learner.flow_functions.items():
        setattr(learner, name, compiled(name, function))
    with catch_warnings(record=True) as caught:
        simplefilter("always")
        for index in range(6):
            learner.update(index)
            learner.training_step(x=X, y=Y)

    assert [str(w.message) for w in caught if "donated" in str(w.message)] == []
    assert traces == [False, True]
    assert learner.learning_rates == rates
