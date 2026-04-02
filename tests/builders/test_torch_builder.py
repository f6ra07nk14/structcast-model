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
        "MIXED_PRECISION_TYPE": "bfloat16",
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
    assert "ce_loss = ce_loss / 2" in script
    assert "self.optimizer_scaler.scale(ce_loss).backward()" in script
    assert "self.need_update = (step + 1) % 2 == 0" in script
    assert "self.optimizer_scaler.unscale_(self.optimizer)" in script
    assert "self.optimizer_clip(" in script
    assert "return self.need_update" in script


def test_torch_backward_builder_renders_non_accumulation_without_mixed_precision() -> None:
    """Render direct backward/step branch when accumulation and AMP are disabled."""
    raw = {
        "MIXED_PRECISION": False,
        "BACKWARDS": [["ce_loss", [[{"_obj_": [["_addr_", "torch.optim.SGD"]]}, ["model"]]]]],
    }
    script = TorchBackwardBuilder(raw=raw)(classname="BackwardNoAmp").scripts[0]
    assert "ce_loss.backward(" in script
    assert "return self.need_update" in script
    assert "self.SGD.step()" in script


def test_torch_backward_builder_no_accumulation_zero_grad_after_step() -> None:
    """Without accumulation, zero_grad is emitted after step in each entry (backward → step → zero_grad)."""
    raw = {
        "MIXED_PRECISION": False,
        "BACKWARDS": [["ce_loss", [[{"_obj_": [["_addr_", "torch.optim.SGD"]]}, ["model"]]]]],
    }
    script = TorchBackwardBuilder(raw=raw)(classname="BackwardZeroGrad").scripts[0]
    backward_pos = script.index("ce_loss.backward(")
    step_pos = script.index("self.SGD.step()")
    zero_pos = script.index("self.SGD.zero_grad()")
    assert backward_pos < step_pos < zero_pos


def test_torch_backward_builder_gan_multi_entry_per_entry_flow() -> None:
    """GAN: multiple BACKWARDS entries each get their own backward → step → zero_grad cycle."""
    raw = {
        "MIXED_PRECISION": False,
        "BACKWARDS": [
            ["loss_G", [["optimizer_G", {"_obj_": [["_addr_", "torch.optim.Adam"]]}, ["G"]]]],
            ["loss_D", [["optimizer_D", {"_obj_": [["_addr_", "torch.optim.Adam"]]}, ["D"]]]],
        ],
    }
    script = TorchBackwardBuilder(raw=raw)(classname="GANBackward").scripts[0]

    # Both losses must be present
    assert "loss_G.backward(" in script
    assert "loss_D.backward(" in script

    # backward is called exactly once per loss (no duplicate backward calls)
    assert script.count("loss_G.backward(") == 1
    assert script.count("loss_D.backward(") == 1

    # Per-entry ordering: G cycle comes before D cycle
    g_back = script.index("loss_G.backward(")
    g_step = script.index("self.optimizer_G.step()")
    g_zero = script.index("self.optimizer_G.zero_grad()")
    d_back = script.index("loss_D.backward(")
    d_step = script.index("self.optimizer_D.step()")
    d_zero = script.index("self.optimizer_D.zero_grad()")

    assert g_back < g_step < g_zero, "G entry must follow backward → step → zero_grad order"
    assert d_back < d_step < d_zero, "D entry must follow backward → step → zero_grad order"
    # D entry comes after G entry's zero_grad (sequential per-entry processing)
    assert g_zero < d_back, "D entry must start after G entry completes"


def test_torch_backward_builder_gan_multi_entry_with_mixed_precision() -> None:
    """GAN: multiple BACKWARDS entries with AMP — each entry uses its own GradScaler."""
    raw = {
        "MIXED_PRECISION": True,
        "MIXED_PRECISION_TYPE": "float16",
        "BACKWARDS": [
            ["loss_G", [["optimizer_G", {"_obj_": [["_addr_", "torch.optim.Adam"]]}, ["G"]]]],
            ["loss_D", [["optimizer_D", {"_obj_": [["_addr_", "torch.optim.Adam"]]}, ["D"]]]],
        ],
    }
    backward = TorchBackwardBuilder(raw=raw)(classname="GANAmpBackward")
    script = backward.scripts[0]

    # Each entry has its own scaler
    assert "self.optimizer_G_scaler" in script
    assert "self.optimizer_D_scaler" in script

    # Scalers scale the correct loss
    assert "self.optimizer_G_scaler.scale(loss_G)" in script
    assert "self.optimizer_D_scaler.scale(loss_D)" in script

    # grad_scalers property exposes per-entry scalers
    assert '"optimizer_G": self.optimizer_G_scaler' in script
    assert '"optimizer_D": self.optimizer_D_scaler' in script

    # backward called once per entry (not per optimizer); with AMP the call is scaler.scale(loss).backward()
    assert script.count(".backward(") == 2


def test_torch_backward_builder_multi_optimizer_single_entry_shared_scaler() -> None:
    """Single backward entry with multiple optimizers shares one GradScaler for the whole entry."""
    raw = {
        "MIXED_PRECISION": True,
        "MIXED_PRECISION_TYPE": "float16",
        "BACKWARDS": [
            [
                "loss_D",
                [
                    ["optimizer_D_A", {"_obj_": [["_addr_", "torch.optim.Adam"]]}, ["D_A"]],
                    ["optimizer_D_B", {"_obj_": [["_addr_", "torch.optim.Adam"]]}, ["D_B"]],
                ],
            ]
        ],
    }
    backward = TorchBackwardBuilder(raw=raw)(classname="SharedScalerBackward")
    script = backward.scripts[0]

    # Only one scaler created (for the first optimizer of the entry)
    assert "self.optimizer_D_A_scaler = torch.amp.GradScaler()" in script
    assert "self.optimizer_D_B_scaler" not in script

    # backward called exactly once (with AMP: scaler.scale(loss).backward())
    assert script.count(".backward(") == 1
    assert "self.optimizer_D_A_scaler.scale(loss_D)" in script

    # Both optimizers are unscaled and stepped using the shared (first-opt) scaler
    assert "self.optimizer_D_A_scaler.unscale_(self.optimizer_D_A)" in script
    assert "self.optimizer_D_A_scaler.unscale_(self.optimizer_D_B)" in script
    assert "self.optimizer_D_A_scaler.step(self.optimizer_D_A)" in script
    assert "self.optimizer_D_A_scaler.step(self.optimizer_D_B)" in script

    # scaler.update() called once per entry (not once per optimizer)
    assert script.count("self.optimizer_D_A_scaler.update()") == 1


def test_torch_backward_builder_gan_accumulation_all_backwards_first() -> None:
    """GAN with gradient accumulation: all backward passes run first, steps are deferred."""
    raw = {
        "MIXED_PRECISION": False,
        "ACCUMULATE_GRADIENTS": 4,
        "BACKWARDS": [
            ["loss_G", [["optimizer_G", {"_obj_": [["_addr_", "torch.optim.Adam"]]}, ["G"]]]],
            ["loss_D", [["optimizer_D", {"_obj_": [["_addr_", "torch.optim.Adam"]]}, ["D"]]]],
        ],
    }
    script = TorchBackwardBuilder(raw=raw)(classname="GANAccumBackward").scripts[0]

    # Loss scaling applied
    assert "loss_G = loss_G / 4" in script
    assert "loss_D = loss_D / 4" in script

    # Both backward calls appear before 'if self.need_update:'
    need_update_pos = script.index("if self.need_update:")
    assert script.index("loss_G.backward(") < need_update_pos
    assert script.index("loss_D.backward(") < need_update_pos

    # Steps appear inside the conditional block
    g_step_pos = script.index("self.optimizer_G.step()")
    d_step_pos = script.index("self.optimizer_D.step()")
    assert g_step_pos > need_update_pos
    assert d_step_pos > need_update_pos
