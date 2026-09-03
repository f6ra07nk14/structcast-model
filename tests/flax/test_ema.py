"""Runtime tests for the `EMA` shadow models a Flax learner declares (`docs/adr/0021`).

The generated learner is exec'd from a file and driven with real `flax.nnx` modules, compiled the
way the CLI compiles it: the update is a deliberate host call outside the traced step, so whether it
lands, when, and by how much is only decided once the emitted code actually runs under `nnx.jit`.
"""

from importlib.util import module_from_spec, spec_from_file_location
from inspect import Parameter, signature
from pathlib import Path
from types import ModuleType
from typing import Any

import jax
import jax.numpy as jnp
import pytest
from structcast.core.exceptions import SpecError

from flax import nnx
from structcast_model.builders.flax import FlaxBuilder, FlaxLearnerBuilder
from structcast_model.flax.distributed import FlaxDistributedStrategy
from structcast_model.utils.base import load_any
from tests import FIXTURES_DIR

CFG_DIR = FIXTURES_DIR / "cfg" / "flax"
MODEL_YAML = CFG_DIR / "Linear.yaml"
LEARNER_YAML = CFG_DIR / "LinearLearner.yaml"

X = jnp.asarray([[1.0, 0.5, -0.5, 2.0], [0.0, 1.0, 1.0, -1.0]])
"""One fixed batch, so a parameter that moves can only be the optimizer's doing."""

Y = jnp.asarray([[1.0, -1.0], [0.5, 0.25]])

AVERAGED_INFERENCE = [
    ["x", "prediction", "ema_model"],
    [{"predictions": "prediction", "targets": "y"}, "errors", "mse"],
    ["eval: errors.mean()", "loss", None],
]
"""An inference flow validating over the average instead of over the trained parameters."""


@pytest.fixture(autouse=True)
def _clear_mesh() -> Any:
    """Unset the mesh the state-dict test's strategy activated: `jax.set_mesh` is process-wide."""
    yield
    jax.set_mesh(None)


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


def _ema_raw(**overrides: Any) -> dict[str, Any]:
    """Load the linear learner fixture with an EMA over its single model."""
    return {**load_any(LEARNER_YAML), "EMA": {"model": True}, **overrides}


def _learner(tmp_path: Path, raw: dict[str, Any], name: str = "learner", **kwargs: Any) -> Any:
    """Generate a learner from a mutated fixture configuration and build it on a fresh model."""
    FlaxLearnerBuilder(raw=raw, current_path=str(LEARNER_YAML))(**kwargs)(tmp_path / f"{name}.py")
    learner_type = _load(tmp_path / f"{name}.py", f"generated_{name}").Learner
    return learner_type(_model_type(tmp_path)(rngs=nnx.Rngs(0)))


def _kernel(learner: Any, name: str = "ema_model") -> jax.Array:
    """Read the kernel of one of the learner's models, copied out of the live variable."""
    return jnp.copy(learner.models[name].fc.kernel[...])


@pytest.mark.parametrize(
    ("window", "gates"),
    [(None, [True, True, True, True]), (2, [False, True, False, True])],
    ids=["every-step", "window-of-two"],
)
def test_the_average_moves_on_update_steps_and_on_no_other(
    tmp_path: Path, window: int | None, gates: list[bool]
) -> None:
    """The blend follows Updates, not steps: an accumulation micro-step must leave the average alone.

    The window lives in the optimizer state on the device, so the host only learns which step applied
    from what the step reports back. An average blended on every call would advance `1/k` as fast as
    its decay says, against gradients no optimizer has applied yet -- invisible in the criteria, and
    visible only as an evaluation curve that never catches up with the training one.
    """
    learner = _learner(tmp_path, _ema_raw(), "cadence", parameters={"DEFAULT": {"accumulate_gradients": window}})
    previous = _kernel(learner)

    moved, reported = [], []
    for _ in gates:
        learner.training_step(x=X, y=Y)
        current = _kernel(learner)
        moved.append(not jnp.array_equal(previous, current))
        reported.append(learner.has_updated)
        previous = current

    assert reported == gates
    assert moved == gates


def test_one_update_blends_the_average_by_its_decay(tmp_path: Path) -> None:
    """The rule is exact arithmetic, so it is asserted exactly rather than within a tolerance.

    The average starts as the parameters it was built from, and one Update takes it to
    `decay * average + (1 - decay) * parameters` -- the whole reason to keep one is that it lags, so
    a blend by the wrong factor is a different feature, not a rounding difference.
    """
    learner = _learner(tmp_path, _ema_raw(), "math")
    average = _kernel(learner)
    assert jnp.array_equal(average, _kernel(learner, "model"))

    learner.training_step(x=X, y=Y)

    assert jnp.array_equal(_kernel(learner), 0.999 * average + (1 - 0.999) * _kernel(learner, "model"))
    assert not jnp.array_equal(_kernel(learner), _kernel(learner, "model"))


def _compiled(learner: Any) -> Any:
    """Compile both steps the way `scm flax train` does, donating the training step's state."""
    for name, function in learner.flow_functions.items():
        donated = tuple(
            p.name for p in signature(function).parameters.values() if p.kind is Parameter.POSITIONAL_OR_KEYWORD
        )
        setattr(
            learner, name, nnx.jit(function, donate_argnames=donated) if name == "_training_step" else nnx.jit(function)
        )
    return learner


def test_the_average_advances_while_the_steps_are_compiled(tmp_path: Path) -> None:
    """The update is a host call on purpose, and a compiled run is where that has to hold.

    An `nnx.EMA` a compiled step closed over would be mutated from another trace level, which flax
    rejects outright; keeping the blend outside the step also keeps it out of the donation contract.
    Three steps of a compiled learner therefore have to advance the average exactly as the eager one
    does, and the criteria of an inference step compiled alongside must still come from it.
    """
    raw = _ema_raw()
    raw["LEARNERS"][0]["INFERENCE_FLOW"] = AVERAGED_INFERENCE
    eager = _learner(tmp_path, raw, "eager")
    compiled = _compiled(_learner(tmp_path, raw, "compiled"))

    for _ in range(3):
        eager.training_step(x=X, y=Y)
        compiled.training_step(x=X, y=Y)

    assert jnp.allclose(_kernel(compiled), _kernel(eager), atol=1e-6)
    assert not jnp.array_equal(_kernel(compiled), _kernel(compiled, "model"))
    assert jnp.isfinite(compiled.inference_step(x=X, y=Y)["loss"])


def test_the_inference_flow_runs_the_average_and_not_the_trained_parameters(tmp_path: Path) -> None:
    """Validating over the average is the point of keeping one, so the criteria have to come from it.

    The two are far apart here -- two Updates against a decay of 0.999 -- so a flow that quietly ran
    the trained model instead would report the training loss, which is what the second half pins.
    """
    raw = _ema_raw()
    raw["LEARNERS"][0]["INFERENCE_FLOW"] = AVERAGED_INFERENCE
    learner = _learner(tmp_path, raw, "inference")
    plain = _learner(tmp_path, load_any(LEARNER_YAML), "plain")
    for _ in range(2):
        learner.training_step(x=X, y=Y)
        plain.training_step(x=X, y=Y)

    averaged = learner.inference_step(x=X, y=Y)["loss"]

    assert jnp.isfinite(averaged)
    assert float(averaged) != pytest.approx(float(plain.inference_step(x=X, y=Y)["loss"]))


def test_a_training_flow_that_reads_the_average_is_rejected() -> None:
    """An average is a view no optimizer owns: differentiating it computes gradients for nothing."""
    raw = _ema_raw()
    raw["LEARNERS"][0]["FLOW"][0][2] = "ema_model"

    with pytest.raises(SpecError, match='The training FLOW reads "ema_model"'):
        # `scripts` is a cached property: binding it is what runs the check being tested here.
        _ = FlaxLearnerBuilder(raw=raw, current_path=str(LEARNER_YAML))().scripts


def test_the_models_property_carries_the_average_through_a_state_round_trip(tmp_path: Path) -> None:
    """The `models` property is the whole persistence path: a resume restores what it names.

    The view's variables are the average itself, so saving the view saves the average and loading
    into it writes the average back -- without which a resumed run would keep training against
    weights it can no longer validate.
    """
    learner = _learner(tmp_path, _ema_raw(), "state")
    for _ in range(3):
        learner.training_step(x=X, y=Y)
    strategy = FlaxDistributedStrategy(preset="single")
    saved = strategy.state_dict(dict(learner.models))["models"]
    assert set(saved) == {"model", "ema_model"}

    restored = _learner(tmp_path, _ema_raw(), "state")
    strategy.load_state_dict(dict(restored.models), {}, None, {"models": saved})

    assert jnp.array_equal(_kernel(restored), _kernel(learner))
    # The state landed in the average itself, not only in the view: the next blend continues from it.
    restored.training_step(x=X, y=Y)
    assert jnp.array_equal(_kernel(restored), 0.999 * _kernel(learner) + (1 - 0.999) * _kernel(restored, "model"))


def test_an_ema_key_that_names_no_model_is_rejected() -> None:
    """A typo in the key would otherwise average nothing at all, silently."""
    raw = {**load_any(LEARNER_YAML), "EMA": {"modle": True}}

    with pytest.raises(SpecError, match='EMA names "modle"'):
        FlaxLearnerBuilder(raw=raw, current_path=str(LEARNER_YAML))()


def test_a_mapping_carries_the_averaging_keywords_into_the_run(tmp_path: Path) -> None:
    """The keywords are DSL values, so a declared decay has to reach the constructor.

    A decay of one half is far from the default: the assertion fails if the mapping was dropped,
    resolved to something else, or merged the wrong way round with the defaults it completes.
    """
    learner = _learner(tmp_path, _ema_raw(EMA={"model": {"decay": 0.5}}), "keywords")
    average = _kernel(learner)

    learner.training_step(x=X, y=Y)

    assert jnp.array_equal(_kernel(learner), 0.5 * average + 0.5 * _kernel(learner, "model"))


class _Dropped(nnx.Module):
    """A model with dropout, whose RNG stream is state an average must not try to blend."""

    def __init__(self, *, rngs: nnx.Rngs) -> None:
        """Build the linear layer the fixture learner feeds and the dropout that follows it."""
        self.fc = nnx.Linear(4, 2, rngs=rngs)
        self.dropout = nnx.Dropout(0.5, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        """Run the layer, dropping activations unless the caller's view says otherwise."""
        return self.dropout(self.fc(x))


def test_the_average_is_taken_over_the_parameters_alone(tmp_path: Path) -> None:
    """A stochastic model must stay averageable: an RNG key cannot be blended at all.

    `flax.nnx.EMA` tracks every Variable by default, so a model with dropout would fail on its first
    Update -- the second half of this test is that failure -- and its counter would be cast through a
    float on the way back. The emitted default therefore filters to the parameters, which is what the
    optimizers are built `wrt` as well.
    """
    FlaxLearnerBuilder(raw=_ema_raw(), current_path=str(LEARNER_YAML))()(tmp_path / "dropped.py")
    model = _Dropped(rngs=nnx.Rngs(0))
    learner = _load(tmp_path / "dropped.py", "dropped_learner").Learner(model)

    learner.training_step(x=X, y=Y)

    assert learner.updates == 1
    assert not jnp.array_equal(_kernel(learner), _kernel(learner, "model"))
    with pytest.raises(TypeError):
        nnx.EMA(model, decay=0.999).update(model)


def _rename_input(raw: dict[str, Any], name: str) -> None:
    """Rename the `y` input of the linear fixture, which its flows read as the regression target."""
    raw["INPUTS"] = ["x", name]
    raw["LEARNERS"][0]["FLOW"][1]["INPUTS"]["targets"] = name
    raw["LEARNERS"][0]["INFERENCE_FLOW"][1][0]["targets"] = name


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda raw: raw["LEARNERS"][0].update(NAME="ema_model"), 'Duplicate variable name "ema_model"'),
        (lambda raw: _rename_input(raw, "ema_model"), 'is emitted as "ema_model"'),
        (
            lambda raw: raw["LEARNERS"][0]["FLOW"].insert(0, ["eval: 1.0", "ema_model", None]),
            "which the generated training step already binds",
        ),
    ],
    ids=["optimizer", "input", "flow-output"],
)
def test_a_name_the_average_needs_and_the_learner_already_uses_is_rejected(mutate: Any, message: str) -> None:
    """`ema_<model>` is an attribute, a step parameter and a flow name: nothing else may answer to it.

    Whichever of the two `__init__` bound last would be the one the inference step runs -- an
    optimizer called as a model, or an average nobody updates -- and neither failure points back at
    the name that caused it.
    """
    raw = _ema_raw()
    mutate(raw)

    with pytest.raises(SpecError, match=message):
        # `scripts` is a cached property: binding it is what runs the checks being tested here.
        _ = FlaxLearnerBuilder(raw=raw, current_path=str(LEARNER_YAML))().scripts


def test_resuming_a_checkpoint_written_before_the_average_was_declared_says_so(tmp_path: Path) -> None:
    """Adding an `EMA` to a learner outdates every checkpoint of it, and the resume has to say which.

    The bare `KeyError` this used to raise named the dictionary, not the migration; the message is
    the torch one word for word, because the two frameworks share the checkpoint contract.
    """
    plain = _learner(tmp_path, load_any(LEARNER_YAML), "plain")
    strategy = FlaxDistributedStrategy(preset="single")
    saved = strategy.state_dict(dict(plain.models))
    averaged = _learner(tmp_path, _ema_raw(), "averaged")

    with pytest.raises(ValueError, match='carries no state for model "ema_model"'):
        strategy.load_state_dict(dict(averaged.models), {}, None, saved)


def test_a_checkpoint_carrying_an_average_the_learner_dropped_still_resumes(tmp_path: Path) -> None:
    """The other direction is not a failure: state for a model the learner no longer has is ignored.

    Dropping an `EMA` declaration leaves the weights, the optimizer state and the counters resumable,
    and nothing in the run depends on the average that was.
    """
    averaged = _learner(tmp_path, _ema_raw(), "averaged")
    for _ in range(2):
        averaged.training_step(x=X, y=Y)
    strategy = FlaxDistributedStrategy(preset="single")
    saved = strategy.state_dict(dict(averaged.models))
    plain = _learner(tmp_path, load_any(LEARNER_YAML), "plain")

    strategy.load_state_dict(dict(plain.models), {}, None, saved)

    assert set(plain.models) == {"model"}
    assert jnp.array_equal(_kernel(plain, "model"), _kernel(averaged, "model"))
