"""Unit tests for the hand-written training example in examples/flax/simple_training.py."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import pytest

from flax import nnx
from structcast_model.base_trainer import Learner
from structcast_model.flax.utils import donate_argnames


def _load_example_module() -> Any:
    """Load the example module the way a configuration does: by file path, not by import name."""
    path = Path(__file__).resolve().parents[2] / "examples" / "flax" / "simple_training.py"
    spec = importlib.util.spec_from_file_location("example_flax_simple_training", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_EXAMPLE = _load_example_module()


def _learner_and_batch() -> tuple[Any, dict[str, jax.Array]]:
    """Build the example's learner over freshly seeded weights, with one batch to feed it."""
    model = _EXAMPLE.build_model(rngs=nnx.Rngs(_EXAMPLE.SEED))
    return _EXAMPLE.SimpleLearner(model), _EXAMPLE.make_dataset(1, _EXAMPLE.SEED)[0]


def _spy(name: str, step: Any, seen: list[str]) -> Any:
    """Wrap *step*, recording its name on every call, as a compiled rebinding would."""

    def _wrapper(*state: Any, **batch: Any) -> Any:
        seen.append(name)
        return step(*state, **batch)

    return _wrapper


def test_the_example_trains_end_to_end(capsys: pytest.CaptureFixture[str]) -> None:
    """The tutorial is only worth reading if running it as documented still completes a run."""
    _EXAMPLE.main()
    assert "Best val_loss" in capsys.readouterr().out


def test_the_example_learner_satisfies_the_trainer_protocol() -> None:
    """The tutorial claims to implement `Learner` by structure alone, so the check is the claim.

    A missing property would only surface deep inside a run -- when a callback reads `info.models`,
    or when the checkpoint saver asks which models an optimizer owns -- so it is pinned here.
    """
    learner, _ = _learner_and_batch()
    concrete: Any = learner  # the protocol view below has no ``outputs`` member

    assert isinstance(learner, Learner)
    assert learner.optimizer_models == {"optimizer": ["model"]}
    assert concrete.outputs == ["loss", "accuracy"]


def test_the_training_step_declares_its_state_the_way_the_cli_donates_it() -> None:
    """The step signature is the donation contract a hand-written learner opts into (ADR-0019).

    `scm flax train` reads the positional-or-keyword parameters off the step and donates exactly
    those buffers. A batch entry taken positionally would be donated too, and a model taken
    keyword-only would be copied on every step instead of being written in place -- neither of
    which anything but this signature decides.
    """
    learner, _ = _learner_and_batch()

    assert donate_argnames(learner._training_step) == ("model", "optimizer")
    assert donate_argnames(learner._inference_step) == ("model",)


def test_the_steps_are_the_flow_functions_the_cli_rebinds() -> None:
    """`cmd_flax.train` compiles a learner by `setattr` over every `flow_functions` name.

    A learner declaring no flow, or one whose public steps called the closures directly instead of
    through the attribute, would leave that stage a silent no-op: the run would look compiled and
    train uncompiled. On the Flax side the compile unit is the whole step, not the flow inside it.
    """
    learner, batch = _learner_and_batch()
    assert sorted(learner.flow_functions) == ["_inference_step", "_training_step"]
    seen: list[str] = []

    for name in list(learner.flow_functions):
        setattr(learner, name, _spy(name, getattr(learner, name), seen))
    learner.training_step(**batch)
    learner.inference_step(**batch)

    assert seen == ["_training_step", "_inference_step"]


def test_the_steps_survive_the_compile_stage_and_train_the_same() -> None:
    """The steps are the compile units, so they must trace on their own and change nothing but speed.

    Donation is what makes a compiled Flax step cheap and is also what would corrupt a run that
    kept reading the donated buffers afterwards, so the compiled learner is compared against an
    eager twin on the same batch: same reported loss, and the same parameters after the update.
    """
    learner, batch = _learner_and_batch()
    for name in list(learner.flow_functions):
        step = getattr(learner, name)
        donated: Any = {"donate_argnames": donate_argnames(step)} if name == "_training_step" else {}
        setattr(learner, name, nnx.jit(step, **donated))

    compiled = learner.training_step(**batch)
    eager, _ = _learner_and_batch()
    reference = eager.training_step(**batch)

    assert learner.updates == 1
    assert float(compiled["loss"]) == pytest.approx(float(reference["loss"]), rel=1e-5)
    # The weights after the update, not just the loss before it: the gradients are computed inside
    # the traced region, so only the parameters prove the compiled step fed the same ones back.
    for a, b in zip(
        jax.tree.leaves(nnx.state(learner.model, nnx.Param)),
        jax.tree.leaves(nnx.state(eager.model, nnx.Param)),
        strict=True,
    ):
        assert jnp.allclose(a, b, atol=1e-6)


def test_the_schedule_lowers_the_rate_once_per_epoch_of_the_tutorial_dataset() -> None:
    """The tutorial's schedule replaces the torch example's `on_epoch_end`, so it must actually step.

    optax counts updates, not epochs, so the staircase is keyed to `BATCHES`; a schedule that
    counted anything else would leave `Printer` showing the same rate for the whole run.
    """
    learner, batch = _learner_and_batch()

    rates = []
    for _ in range(2 * _EXAMPLE.BATCHES):
        learner.training_step(**batch)
        rates.append(learner.learning_rates["optimizer"])

    assert rates[0] == pytest.approx(0.1)
    assert rates[_EXAMPLE.BATCHES - 1] == pytest.approx(0.1)
    assert rates[-1] == pytest.approx(0.05)
