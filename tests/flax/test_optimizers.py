"""Unit tests for structcast_model.flax.optimizers."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import optax
import pytest

from flax import nnx
from structcast_model.flax.optimizers import get_learning_rate, no_weight_decay_mask


def _optimizer(tx: Any) -> tuple[Any, Any]:
    """Bind a transformation to the smallest module an `nnx.Optimizer` can own."""
    model = nnx.Linear(2, 1, rngs=nnx.Rngs(0))
    return model, nnx.Optimizer(model, tx, wrt=nnx.Param)


def _update(model: Any, optimizer: Any) -> None:
    """Run one real update so the optimizer state advances the way training advances it."""
    optimizer.update(model, nnx.grad(lambda m: jnp.mean(m(jnp.ones((4, 2))) ** 2))(model))


def test_get_learning_rate_follows_an_injected_schedule() -> None:
    """A scheduled learning rate must be readable and must move, so logs track the schedule.

    The injected state reports the value used by the last update: it stays at the initial value
    through the first update and only then follows the schedule.
    """
    model, optimizer = _optimizer(
        optax.inject_hyperparams(optax.adamw)(learning_rate=optax.linear_schedule(0.1, 0.0, 10))
    )
    reported = [float(get_learning_rate(optimizer))]
    for _ in range(2):
        _update(model, optimizer)
        reported.append(float(get_learning_rate(optimizer)))
    assert reported == pytest.approx([0.1, 0.1, 0.09], abs=1e-6)


def test_get_learning_rate_reads_a_constant_injected_rate() -> None:
    """A constant learning rate is reported as the float32 scalar the optimizer was built with."""
    learning_rate = get_learning_rate(_optimizer(optax.inject_hyperparams(optax.adamw)(learning_rate=0.003))[1])
    assert learning_rate.dtype == jnp.float32
    assert float(learning_rate) == pytest.approx(0.003)


def test_get_learning_rate_reads_inside_a_trace() -> None:
    """The generated training step calls this inside `nnx.jit`, so the walk must survive tracing."""
    optimizer = _optimizer(optax.inject_hyperparams(optax.adamw)(learning_rate=0.003))[1]
    assert float(nnx.jit(get_learning_rate)(optimizer)) == pytest.approx(0.003)


def test_get_learning_rate_without_inject_hyperparams_is_nan() -> None:
    """Optax keeps a plain learning rate in a closure, so an un-injected chain reports NaN."""
    assert jnp.isnan(get_learning_rate(_optimizer(optax.adamw(0.003))[1]))


def test_get_learning_rate_with_two_injected_rates_is_nan() -> None:
    """Two injected rates in one chain are ambiguous, so no single rate is reported."""
    tx = optax.chain(
        optax.inject_hyperparams(optax.sgd)(learning_rate=0.01),
        optax.inject_hyperparams(optax.adamw)(learning_rate=0.001),
    )
    assert jnp.isnan(get_learning_rate(_optimizer(tx)[1]))


def test_no_weight_decay_mask_marks_matching_paths_false() -> None:
    """The mask mirrors the parameter tree, flagging every path matched by any regex."""
    tree = {"encoder": {"kernel": 1.0, "bias": 2.0}, "scale": 3.0, "head": {"kernel": 4.0}}
    mask = no_weight_decay_mask(r"\.bias$", r"^scale$")
    assert mask(tree) == {"encoder": {"kernel": True, "bias": False}, "scale": False, "head": {"kernel": True}}


def test_no_weight_decay_mask_exempts_parameters_from_adamw_decay() -> None:
    """Used as the `adamw` mask, the excluded parameters must not be decayed.

    With zero gradients the adam term vanishes, so the whole update is the weight decay: the
    masked-out bias keeps its value while the kernel shrinks by `learning_rate * weight_decay`.
    """
    params = {"layer": {"kernel": jnp.ones((2, 2)), "bias": jnp.ones((2,))}}
    tx = optax.adamw(learning_rate=0.1, weight_decay=1.0, mask=no_weight_decay_mask(r"\.bias$"))
    state = tx.init(params)
    updates, _ = tx.update(jax.tree.map(jnp.zeros_like, params), state, params)
    updated = optax.apply_updates(params, updates)
    assert jnp.array_equal(updated["layer"]["bias"], jnp.ones((2,)))
    assert jnp.allclose(updated["layer"]["kernel"], jnp.full((2, 2), 0.9))
