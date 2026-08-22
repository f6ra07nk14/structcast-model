"""Unit tests for the optax schedule example in examples/flax/optimizers.py."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import pytest


def _load_example_module() -> Any:
    """Load the example module the way a configuration does: by file path, not by import name."""
    path = Path(__file__).resolve().parents[2] / "examples" / "flax" / "optimizers.py"
    spec = importlib.util.spec_from_file_location("example_flax_optimizers", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_EXAMPLE = _load_example_module()


def _torch_lambda(epoch: int, *, epochs: int, decay_epoch: int, offset: int = 0) -> float:
    """The `LambdaLR` of `cfg/torch/learners/CycleGAN.yaml`, written out for comparison."""
    return 1.0 - max(0, epoch + offset - decay_epoch) / (epochs - decay_epoch)


def test_linear_decay_after_reproduces_the_torch_lambda_at_every_epoch() -> None:
    """The CycleGAN recipe is the schedule, so the optax twin has to hit the same rate per epoch.

    optax counts optimizer applies where torch counts epochs; `steps_per_epoch` is the whole
    conversion, and getting it wrong would decay the rate at a different pace than the reference
    implementation this configuration mirrors.
    """
    steps_per_epoch = 5
    schedule = _EXAMPLE.linear_decay_after(init_value=1.0, epochs=200, decay_epoch=100, steps_per_epoch=steps_per_epoch)

    for epoch in (0, 50, 100, 150, 199):
        expected = _torch_lambda(epoch, epochs=200, decay_epoch=100)
        assert float(schedule(epoch * steps_per_epoch)) == pytest.approx(expected, abs=1e-6)


def test_linear_decay_after_starts_further_down_the_ramp_when_epochs_are_already_trained() -> None:
    """`offset` is what makes a resumed run continue the ramp instead of restarting it."""
    schedule = _EXAMPLE.linear_decay_after(init_value=1.0, epochs=4, decay_epoch=2, steps_per_epoch=1, offset=1)

    assert [float(schedule(step)) for step in range(4)] == pytest.approx([1.0, 1.0, 0.5, 0.0])


def test_linear_decay_after_refuses_a_ramp_with_no_length() -> None:
    """A decay epoch at or past the end divides by zero, which optax would report far from here."""
    with pytest.raises(ValueError, match="must be greater than decay_epoch"):
        _EXAMPLE.linear_decay_after(init_value=1.0, epochs=100, decay_epoch=100, steps_per_epoch=1)


def test_warmup_cosine_peaks_at_the_end_of_the_warmup_and_anneals_to_the_floor() -> None:
    """A warmup that peaked anywhere else would be a different recipe than the timm one it mirrors."""
    schedule = _EXAMPLE.warmup_cosine(peak_value=1.0, epochs=10, steps_per_epoch=4, warmup_epochs=2, end_value=0.01)

    assert float(schedule(0)) == pytest.approx(0.0)
    assert float(schedule(8)) == pytest.approx(1.0)
    assert float(schedule(40)) == pytest.approx(0.01)
    # Strictly falling after the peak: an annealed rate that plateaued would not be a cosine.
    assert float(schedule(20)) > float(schedule(30)) > float(schedule(40))


def test_warmup_cosine_refuses_a_warmup_that_does_not_fit_in_the_run() -> None:
    """Warming up for the whole run leaves no annealing at all, which is never what was meant."""
    with pytest.raises(ValueError, match="must be smaller than epochs"):
        _EXAMPLE.warmup_cosine(peak_value=1.0, epochs=2, steps_per_epoch=1, warmup_epochs=2)
