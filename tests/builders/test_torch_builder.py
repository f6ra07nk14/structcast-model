"""API-level tests for torch builder classes."""

from collections import defaultdict

from structcast_model.builders.torch_builder import (
    TorchBackwardBuilder,
    TorchBuilder,
    TorchLayerIntermediate,
)


def test_torch_layer_intermediate_script_contains_train_and_infer_paths() -> None:
    """Generate forward code branches when INFERENCE_FLOW is present."""
    script = TorchLayerIntermediate(
        classname="Unit",
        imports={},
        inputs=["x"],
        outputs=["y"],
        layers={},
        flow=[("x", "y", None)],
        inference_flow=[("x", "y", None)],
        structured_output=False,
    )._get_layer_script("Unit", ["proj = torch.nn.Identity()"])
    assert "class Unit(torch.nn.Module):" in script
    assert "if self.training:" in script
    assert "self.proj = torch.nn.Identity()" in script
    assert "return y" in script


def test_torch_backward_builder_get_mixed_precision_variants() -> None:
    """Build mixed precision string from bool/dict options."""
    raw = {"BACKWARDS": [["loss", [[{"_obj_": [["_addr_", "torch.optim.SGD"]]}, ["model"]]]]]}
    builder = TorchBackwardBuilder(raw=raw)
    imports: defaultdict[str, set[str | None]] = defaultdict(set)
    assert builder._get_mixed_precision(imports, False) is None
    imports = defaultdict(set)
    assert builder._get_mixed_precision(imports, True) == "torch.amp.GradScaler()"
    assert None in imports["torch.amp"]
    imports = defaultdict(set)
    mixed = builder._get_mixed_precision(imports, {"enabled": "eval: use_amp"})
    assert mixed == "torch.amp.GradScaler(enabled=use_amp)"


def test_torch_builder_builds_intermediate_and_scripts() -> None:
    """Build a minimal torch model and render Python script content."""
    raw = {
        "INPUTS": ["x"],
        "OUTPUTS": ["y"],
        "FLOW": [["x", "y", {"_obj_": [["_addr_", "torch.nn.Identity"]]}]],
    }
    built = TorchBuilder(raw=raw)(classname="TinyNet")
    assert built.classname == "TinyNet"
    assert "torch" in built.collected_imports
    assert len(built.scripts) == 1
    assert "class TinyNet(torch.nn.Module):" in built.scripts[0]


def test_torch_backward_builder_renders_accumulation_script() -> None:
    """Render backward code with gradient accumulation and clipping."""
    raw = {
        "MIXED_PRECISION": True,
        "ACCUMULATE_GRADIENTS": 2,
        "BACKWARDS": [
            [
                "ce_loss",
                [
                    [
                        "optimizer",
                        {"_obj_": [["_addr_", "torch.optim.AdamW"], ["_call_", {"lr": 1.0e-3}]]},
                        ["model"],
                        {"_obj_": [["_addr_", "timm.utils.clip_grad.dispatch_clip_grad"]]},
                    ]
                ],
            ]
        ],
    }
    backward = TorchBackwardBuilder(raw=raw)(classname="Backward")
    script = backward.scripts[0]
    assert "(ce_loss / 2).backward(" in script
    assert "should_update = (step + 1) % 2 == 0" in script
    assert "self.optimizer_scaler.unscale_(self.optimizer)" in script
    assert "self.optimizer_clip(" in script
    assert "return should_update" in script


def test_torch_backward_builder_renders_non_accumulation_without_mixed_precision() -> None:
    """Render direct backward/step branch when accumulation and AMP are disabled."""
    raw = {
        "MIXED_PRECISION": False,
        "BACKWARDS": [["ce_loss", [[{"_obj_": [["_addr_", "torch.optim.SGD"]]}, ["model"]]]]],
    }
    script = TorchBackwardBuilder(raw=raw)(classname="BackwardNoAmp").scripts[0]
    assert "ce_loss.backward(" in script
    assert "return True" in script
    assert "self.SGD.step()" in script
