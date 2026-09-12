"""Runtime tests for `GradientCheckpointingLayer`, driven through the code the builder emits into it.

The generated model is exec'd from a file and trained by a generated learner, the way a run would:
activation checkpointing is only worth anything if it changes memory and nothing else, so what is
asserted here is that the same seed produces the same losses and the same weights either way, that
the module tree the learner and the sharding globs walk is untouched, and that an inference call
never pays for a recomputation it will not use.
"""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from structcast_model.builders.torch import TorchBuilder, TorchLearnerBuilder
from structcast_model.torch.trainer import initial_model
from structcast_model.utils.base import load_any
from tests import FIXTURES_DIR
import torch

CFG_DIR = FIXTURES_DIR / "cfg" / "torch"
MODEL_YAML = CFG_DIR / "Linear.yaml"
LEARNER_YAML = CFG_DIR / "LinearLearner.yaml"

BATCH = {
    "x": torch.tensor([[1.0, 0.5, -0.5, 2.0], [0.0, 1.0, 1.0, -1.0]]),
    "y": torch.tensor([[1.0, -1.0], [0.5, 0.25]]),
}
"""One fixed batch, so two runs can only differ through the code under test."""

TWO_LAYERS: dict[str, Any] = {
    "INPUTS": ["x"],
    "OUTPUTS": ["y"],
    "FLOW": [
        ["x", "h", "fc", {"_obj_": [["_addr_", "torch.nn.LazyLinear"], {"_call_": {"out_features": 4}}]}],
        ["h", "y", "out", {"_obj_": [["_addr_", "torch.nn.LazyLinear"], {"_call_": {"out_features": 2}}]}],
    ],
}
"""Two layers, so the checkpointed region holds an activation the backward pass has to ask back for.

The single-layer fixture proves nothing here: its backward needs only the region's own input and the
weight, both kept alive, so a correct checkpoint recomputes nothing at all.
"""


def _load(path: Path, name: str) -> ModuleType:
    """Load a generated module by file path, the way a configuration does: not by import name."""
    spec = spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _model(tmp_path: Path, checkpointing: Any, raw: dict[str, Any] | None = None) -> Any:
    """Build a seeded model, materializing its lazy layer before anything reads its parameters."""
    raw = {**(raw if raw is not None else load_any(MODEL_YAML)), "GRADIENT_CHECKPOINTING": checkpointing}
    directory = tmp_path / str(checkpointing)
    TorchBuilder(raw=raw)()(directory / "model.py")
    torch.manual_seed(0)
    model = _load(directory / "model.py", "generated_model").Model()
    initial_model(model, {"x": (4,)})
    return model


def _train(tmp_path: Path, checkpointing: Any) -> tuple[list[float], torch.Tensor]:
    """Run two training steps of the generated learner and report the losses and the weight."""
    TorchLearnerBuilder.from_path(LEARNER_YAML)()(tmp_path / "learner.py")
    learner = _load(tmp_path / "learner.py", "generated_learner").Learner(_model(tmp_path, checkpointing))
    losses = [float(learner.training_step(**BATCH)["loss"]) for _ in range(2)]
    return losses, learner.models["model"].fc.weight.detach().clone()


@pytest.mark.parametrize(
    "checkpointing", [True, {"determinism_check": "constant:none", "debug": False}], ids=["defaults", "options"]
)
def test_checkpointing_changes_neither_the_losses_nor_the_weights(tmp_path: Path, checkpointing: Any) -> None:
    """Recomputation is a memory trade, so exact equality is the contract, not a tolerance."""
    baseline_losses, baseline_weight = _train(tmp_path / "off", False)
    losses, weight = _train(tmp_path / "on", checkpointing)
    assert losses == baseline_losses
    assert torch.equal(weight, baseline_weight)


def test_the_module_tree_is_the_one_the_layer_would_have_without_checkpointing(tmp_path: Path) -> None:
    """No wrapper module: `shard_modules` globs and state-dict keys are written against these paths."""
    plain, checkpointed = _model(tmp_path, False), _model(tmp_path, True)
    assert [name for name, _ in checkpointed.named_modules()] == [name for name, _ in plain.named_modules()]
    assert list(checkpointed.state_dict()) == list(plain.state_dict())
    assert [name for name, _ in checkpointed.named_parameters()] == [name for name, _ in plain.named_parameters()]


def test_an_inference_call_never_recomputes_even_while_gradients_are_recorded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Both halves of the predicate are exercised, because a learner leaves them disagreeing.

    A generated learner runs the models it does not train in `eval()` while gradients still flow
    through the ones it does, and `inference_step` runs everything under `torch.no_grad()`. Either
    half alone would therefore recompute in a pass whose activations nobody asks for.
    """
    model = _model(tmp_path, True)
    calls: list[int] = []
    checkpoint = torch.utils.checkpoint.checkpoint

    def _counted(*args: Any, **kwargs: Any) -> Any:
        calls.append(1)
        return checkpoint(*args, **kwargs)

    # Patched after the model is built: materializing the lazy layer forwards once, in training mode.
    monkeypatch.setattr(torch.utils.checkpoint, "checkpoint", _counted)

    model.eval()
    model(BATCH["x"]).sum().backward()
    assert calls == []

    model.train()
    model(BATCH["x"]).sum().backward()
    assert calls == [1]

    with torch.no_grad():
        model(BATCH["x"])
    assert calls == [1]


@pytest.mark.parametrize("checkpointing", [False, True], ids=["off", "on"])
def test_the_backward_pass_runs_the_inner_layer_a_second_time(tmp_path: Path, checkpointing: bool) -> None:
    """The whole point is the second forward pass, and nothing else here proves one happens."""
    model = _model(tmp_path, checkpointing, TWO_LAYERS)
    forwards: list[int] = []
    model.fc.register_forward_hook(lambda *_: forwards.append(1))

    loss = model(BATCH["x"]).sum()
    assert forwards == [1]
    loss.backward()
    assert forwards == ([1, 1] if checkpointing else [1])
