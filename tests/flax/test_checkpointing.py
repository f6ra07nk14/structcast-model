"""Runtime tests for `GradientCheckpointingModule`, driven through the code the builder emits into it.

The generated module is exec'd from a file and trained by a generated learner, the way a run would.
Both the eager and the compiled path are exercised: `nnx.remat` sits inside the function a trainer
wraps in `nnx.jit`, and a shape that only works untraced would pass every eager assertion and fail
on the first real run.
"""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType
from typing import Any

import jax
import jax.numpy as jnp
import pytest

from flax import nnx
from structcast_model.builders.flax import FlaxBuilder, FlaxLearnerBuilder
from structcast_model.flax.layers import checkpointing
from structcast_model.utils.base import load_any
from tests import FIXTURES_DIR

CFG_DIR = FIXTURES_DIR / "cfg" / "flax"
MODEL_YAML = CFG_DIR / "Linear.yaml"
LEARNER_YAML = CFG_DIR / "LinearLearner.yaml"

X = jnp.asarray([[1.0, 0.5, -0.5, 2.0], [0.0, 1.0, 1.0, -1.0]])
"""One fixed batch, so two runs can only differ through the code under test."""

Y = jnp.asarray([[1.0, -1.0], [0.5, 0.25]])


def _load(path: Path, name: str) -> ModuleType:
    """Load a generated module by file path, the way a configuration does: not by import name."""
    spec = spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _model(tmp_path: Path, checkpointing_value: Any) -> Any:
    """Generate the fixture model with the given `GRADIENT_CHECKPOINTING` value and seed it."""
    raw = {**load_any(MODEL_YAML), "GRADIENT_CHECKPOINTING": checkpointing_value}
    directory = tmp_path / str(checkpointing_value)
    FlaxBuilder(raw=raw)()(directory / "model.py")
    return _load(directory / "model.py", "generated_model").Model(rngs=nnx.Rngs(0))


def _train(tmp_path: Path, checkpointing_value: Any, *, jit: bool = False) -> tuple[list[float], jax.Array]:
    """Run two training steps of the generated learner and report the losses and the kernel."""
    FlaxLearnerBuilder.from_path(LEARNER_YAML)()(tmp_path / "learner.py")
    learner = _load(tmp_path / "learner.py", "generated_learner").Learner(_model(tmp_path, checkpointing_value))
    if jit:
        # Compiled the way the CLI does it: every state parameter donated, the batch never.
        learner._training_step = nnx.jit(learner._training_step, donate_argnames=("model", "optimizer"))
    losses = [float(learner.training_step(x=X, y=Y)["loss"]) for _ in range(2)]
    return losses, learner.models["model"].fc.kernel[...]


@pytest.mark.parametrize("jit", [False, True], ids=["eager", "jit"])
@pytest.mark.parametrize(
    "checkpointing_value",
    [True, {"policy": "dots_saveable", "prevent_cse": False}],
    ids=["defaults", "policy"],
)
def test_checkpointing_changes_neither_the_losses_nor_the_parameters(
    tmp_path: Path, checkpointing_value: Any, jit: bool
) -> None:
    """Rematerialization is a memory trade, so exact equality is the contract, not a tolerance."""
    baseline_losses, baseline_kernel = _train(tmp_path / "off", False, jit=jit)
    losses, kernel = _train(tmp_path / "on", checkpointing_value, jit=jit)
    assert losses == baseline_losses
    assert jnp.array_equal(kernel, baseline_kernel)


def test_an_inference_call_never_rematerializes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Nothing is recomputed without a backward pass, so an inference view must take the plain path."""
    model = _model(tmp_path, True)
    calls: list[int] = []
    remat = nnx.remat

    def _counted(*args: Any, **kwargs: Any) -> Any:
        calls.append(1)
        return remat(*args, **kwargs)

    monkeypatch.setattr(checkpointing.nnx, "remat", _counted)

    assert jnp.array_equal(model(X, training=False), model(x=X, training=False))
    assert calls == []
    model(X, training=True)
    assert calls == [1]


@pytest.mark.parametrize("checkpointing_value", [False, True], ids=["off", "on"])
def test_the_traced_program_actually_carries_a_remat(tmp_path: Path, checkpointing_value: bool) -> None:
    """The whole point is the second forward pass, and equal gradients alone would not prove one."""
    model = _model(tmp_path, checkpointing_value)
    jaxpr = nnx.jit(nnx.grad(lambda m, x: jnp.sum(m(x)))).trace(model, X).traced.jaxpr
    assert ("remat2" in str(jaxpr)) is checkpointing_value


def test_a_batch_passed_by_name_reaches_the_rematerialized_body(tmp_path: Path) -> None:
    """The CLI initializes a model with its batch as keyword arguments, and a root model may be checkpointed.

    `nnx.remat` takes its arrays positionally, so the declared inputs are moved across; forgetting
    them would leave the body without its arguments on exactly that call.
    """
    plain, checkpointed = _model(tmp_path, False), _model(tmp_path, True)
    assert jnp.array_equal(checkpointed(x=X), plain(x=X))
