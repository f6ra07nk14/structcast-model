"""Unit tests for the hand-written training example in examples/torch/simple_training.py."""

from __future__ import annotations

from collections.abc import Callable
import importlib.util
from pathlib import Path
from typing import Any

import pytest

from structcast_model.torch.distributed import SingleDeviceStrategy
import torch


def _load_example_module() -> Any:
    """Load the example module the way a configuration does: by file path, not by import name."""
    path = Path(__file__).resolve().parents[2] / "examples" / "torch" / "simple_training.py"
    spec = importlib.util.spec_from_file_location("example_simple_training", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_EXAMPLE = _load_example_module()

Flow = Callable[[bool, torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]


def _learner_and_batch() -> tuple[Any, dict[str, torch.Tensor]]:
    """Build the example's learner over freshly seeded weights, with one batch to feed it."""
    torch.manual_seed(_EXAMPLE.SEED)
    return _EXAMPLE.SimpleLearner(_EXAMPLE.build_model()), _EXAMPLE.make_dataset(1, _EXAMPLE.SEED)[0]


def _spy(flow: Flow, seen: list[bool]) -> Flow:
    """Wrap *flow*, recording the update flag of every call, as a compiled rebinding would."""

    def _wrapper(need_update: bool, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        seen.append(need_update)
        return flow(need_update, x, y)

    return _wrapper


def test_the_example_trains_end_to_end(capsys: pytest.CaptureFixture[str]) -> None:
    """The tutorial is only worth reading if running it as documented still completes a run."""
    _EXAMPLE.main()
    assert "Best val_loss" in capsys.readouterr().out


def test_the_steps_call_the_flow_through_the_attribute_the_cli_rebinds() -> None:
    """`cmd_torch._assemble_learner` compiles a learner by `setattr` over every `flow_functions` name.

    A learner declaring no flow, or one whose steps called the closure directly instead of through
    the attribute, would leave that stage a silent no-op: the run would look compiled and train
    uncompiled.
    """
    learner, batch = _learner_and_batch()
    assert list(learner.flow_functions) == ["_flow_optimizer"]
    seen: list[bool] = []
    for name in list(learner.flow_functions):
        setattr(learner, name, _spy(getattr(learner, name), seen))

    learner.training_step(**batch)
    learner.inference_step(**batch)

    # The gate is armed for the step that applies the optimizer, disarmed for the validation pass.
    assert seen == [True, False]


def test_the_flow_survives_the_compile_stage_and_trains_the_same() -> None:
    """The flow is the compile unit, so it must trace on its own and change nothing but speed.

    The `eager` dynamo backend runs the real trace without the C++ code generation an inductor
    build needs, which is what the CLI's `--compile true` would add on top.
    """
    learner, batch = _learner_and_batch()
    strategy = SingleDeviceStrategy(device="cpu")
    for name in list(learner.flow_functions):
        setattr(learner, name, strategy.compile(getattr(learner, name), {"backend": "eager"}))

    compiled = learner.training_step(**batch)
    eager, _ = _learner_and_batch()
    reference = eager.training_step(**batch)

    assert learner.updates == 1
    assert compiled["loss"] == reference["loss"]
    # The weights after the update, not just the loss before it: the backward runs on the traced
    # graph, so only the parameters prove the compiled flow fed the same gradients into the step.
    assert all(torch.equal(a, b) for a, b in zip(learner.model.parameters(), eager.model.parameters(), strict=True))
