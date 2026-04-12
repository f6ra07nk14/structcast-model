"""Advanced builder tests using real cfg templates."""

import pytest

from structcast_model.builders.torch_builder import TorchBackwardBuilder
from tests import ASSETS_DIR

BACKWARD_YAML = ASSETS_DIR / "cfg" / "torch" / "ConvNeXtV2Backward.yaml"


# ---------------------------------------------------------------------------
# TorchBackwardBuilder: basic build
# ---------------------------------------------------------------------------


def test_backward_default_params_classname_and_io() -> None:
    """Build backward with defaults and check classname, inputs, outputs."""
    built = TorchBackwardBuilder.from_path(BACKWARD_YAML)(classname="ConvNeXtBackward")
    assert built.classname == "ConvNeXtBackward"
    assert built.inputs == ["image", "label"]
    assert built.outputs == ["ce_loss", "acc1", "acc5"]


def test_backward_default_models_and_optimizers() -> None:
    """Default build should expose one model and one optimizer."""
    built = TorchBackwardBuilder.from_path(BACKWARD_YAML)()
    assert built.models == ["model"]
    assert built.optimizers == ["optimizer"]


def test_backward_default_mixed_precision_type() -> None:
    """Default config uses bfloat16 mixed precision."""
    built = TorchBackwardBuilder.from_path(BACKWARD_YAML)()
    assert built.mixed_precision_type == "bfloat16"
    assert built.mixed_precision_scales == ["optimizer_grad_scaler"]


def test_backward_default_no_accumulation() -> None:
    """Default config does not accumulate gradients."""
    built = TorchBackwardBuilder.from_path(BACKWARD_YAML)()
    assert built.accumulate_gradients is None


# ---------------------------------------------------------------------------
# TorchBackwardBuilder: script content
# ---------------------------------------------------------------------------


def test_backward_script_contains_autocast() -> None:
    """Default bfloat16 config wraps forward in autocast."""
    script = TorchBackwardBuilder.from_path(BACKWARD_YAML)().scripts[0]
    assert "torch.autocast(device_type, torch.bfloat16)" in script


def test_backward_script_contains_grad_scaler() -> None:
    """GradScaler instantiation appears in the generated script."""
    script = TorchBackwardBuilder.from_path(BACKWARD_YAML)().scripts[0]
    assert "torch.amp.GradScaler(" in script


def test_backward_script_defines_training_and_inference_steps() -> None:
    """Script defines both _training_step and _inference_step functions."""
    script = TorchBackwardBuilder.from_path(BACKWARD_YAML)().scripts[0]
    assert "def _training_step(" in script
    assert "def _inference_step(" in script


def test_backward_script_exposes_properties() -> None:
    """Script exposes models, optimizers, grad_scalers, learning_rates, param_group_names."""
    script = TorchBackwardBuilder.from_path(BACKWARD_YAML)().scripts[0]
    for prop in (
        "def models(self)",
        "def optimizers(self)",
        "def grad_scalers(self)",
        "def learning_rates(self)",
        "def param_group_names(self)",
    ):
        assert prop in script


def test_backward_script_is_compilable() -> None:
    """Generated script can be compiled without syntax errors."""
    scripts = TorchBackwardBuilder.from_path(BACKWARD_YAML)().scripts
    for script in scripts:
        compile(script, "<test>", "exec")


# ---------------------------------------------------------------------------
# TorchBackwardBuilder: gradient accumulation
# ---------------------------------------------------------------------------


def test_backward_accumulate_gradients_stored() -> None:
    """Accumulation count is stored on the intermediate."""
    params = {"DEFAULT": {"accumulate_gradients": 4}}
    built = TorchBackwardBuilder.from_path(BACKWARD_YAML)(parameters=params)
    assert built.accumulate_gradients == 4


def test_backward_accumulate_gradients_script_patterns() -> None:
    """Script contains loss division, need_update guard, and modular update."""
    params = {"DEFAULT": {"accumulate_gradients": 4}}
    script = TorchBackwardBuilder.from_path(BACKWARD_YAML)(parameters=params).scripts[0]
    assert "ce_loss = ce_loss / 4" in script
    assert "if __need_update__:" in script
    assert "self.need_update = (step + 1) % 4 == 0" in script
    assert "return self.need_update" in script


# ---------------------------------------------------------------------------
# TorchBackwardBuilder: gradient clipping
# ---------------------------------------------------------------------------


def test_backward_clip_grad_norm_in_script() -> None:
    """Gradient clipping function appears in the script when configured."""
    params = {"DEFAULT": {"clip_grad_norm": 2.0}}
    script = TorchBackwardBuilder.from_path(BACKWARD_YAML)(parameters=params).scripts[0]
    assert "dispatch_clip_grad" in script
    assert "optimizer_grad_scaler.unscale_(optimizer)" in script


def test_backward_no_clip_when_null() -> None:
    """No clipping code when clip_grad_norm is null."""
    params = {"DEFAULT": {"clip_grad_norm": None}}
    script = TorchBackwardBuilder.from_path(BACKWARD_YAML)(parameters=params).scripts[0]
    assert "dispatch_clip_grad" not in script
    assert "unscale_" not in script


# ---------------------------------------------------------------------------
# TorchBackwardBuilder: mixed precision backward
# ---------------------------------------------------------------------------


def test_backward_mp_scale_backward_without_accumulation() -> None:
    """Without accumulation the scaler.scale().backward() has no division."""
    params = {"DEFAULT": {"accumulate_gradients": None}}
    script = TorchBackwardBuilder.from_path(BACKWARD_YAML)(parameters=params).scripts[0]
    assert "optimizer_grad_scaler.scale(ce_loss).backward()" in script
    assert "ce_loss = ce_loss /" not in script


def test_backward_mp_scale_backward_with_accumulation() -> None:
    """With accumulation loss is divided and backward uses scaler."""
    params = {"DEFAULT": {"accumulate_gradients": 2}}
    script = TorchBackwardBuilder.from_path(BACKWARD_YAML)(parameters=params).scripts[0]
    assert "ce_loss = ce_loss / 2" in script
    assert "optimizer_grad_scaler.scale(ce_loss).backward()" in script
    assert "optimizer_grad_scaler.step(optimizer)" in script
    assert "optimizer_grad_scaler.update()" in script


# ---------------------------------------------------------------------------
# TorchBackwardBuilder: layer decay types
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("decay_type", ["single", "group"])
def test_backward_layer_decay_types_produce_regexes(decay_type: str) -> None:
    """Both single and group layer decay types produce layer_group_regexes."""
    params = {"DEFAULT": {"layer_decay_type": decay_type}}
    script = TorchBackwardBuilder.from_path(BACKWARD_YAML)(parameters=params).scripts[0]
    assert "layer_group_regexes" in script


def test_backward_no_layer_decay_produces_empty_regexes() -> None:
    """Null layer_decay_type produces empty layer_group_regexes."""
    params = {"DEFAULT": {"layer_decay_type": None}}
    script = TorchBackwardBuilder.from_path(BACKWARD_YAML)(parameters=params).scripts[0]
    assert "'layer_group_regexes': []" in script


def test_backward_invalid_layer_decay_type_raises() -> None:
    """Raise from Jinja filter when unsupported layer_decay_type is provided."""
    params = {"DEFAULT": {"layer_decay_type": "not-supported"}}
    with pytest.raises(ValueError, match="Invalid layer_decay_type"):
        TorchBackwardBuilder.from_path(BACKWARD_YAML)(parameters=params)


# ---------------------------------------------------------------------------
# TorchBackwardBuilder: backbone variants
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backbone", ["atto", "femto", "pico", "nano", "tiny"])
def test_backward_backbone_variants_compile(backbone: str) -> None:
    """Each backbone variant produces a compilable script."""
    params = {"DEFAULT": {"backbone": backbone}}
    scripts = TorchBackwardBuilder.from_path(BACKWARD_YAML)(parameters=params).scripts
    for script in scripts:
        compile(script, "<test>", "exec")


# ---------------------------------------------------------------------------
# TorchBackwardBuilder: collected imports
# ---------------------------------------------------------------------------


def test_backward_collected_imports_include_torch_and_amp() -> None:
    """Collected imports include torch and torch.amp for mixed precision."""
    built = TorchBackwardBuilder.from_path(BACKWARD_YAML)()
    imports = built.collected_imports
    assert "torch" in imports
    assert "torch.amp" in imports


def test_backward_collected_imports_include_optimizer_module() -> None:
    """Collected imports include the optimizer factory module."""
    built = TorchBackwardBuilder.from_path(BACKWARD_YAML)()
    imports = built.collected_imports
    assert "structcast_model.torch.optimizers" in imports


# ---------------------------------------------------------------------------
# TorchBackwardBuilder: full accumulation + clipping + mp combo
# ---------------------------------------------------------------------------


def test_backward_full_combo_accumulate_clip_mp() -> None:
    """Combine accumulation, clipping, and mixed precision in one build."""
    params = {"DEFAULT": {"accumulate_gradients": 4, "clip_grad_norm": 2.0, "layer_decay_type": "single"}}
    built = TorchBackwardBuilder.from_path(BACKWARD_YAML)(parameters=params, classname="FullCombo")
    script = built.scripts[0]
    assert built.classname == "FullCombo"
    assert built.accumulate_gradients == 4
    assert "optimizer" in built.optimizers
    assert "optimizer_grad_scaler" in built.mixed_precision_scales
    assert "ce_loss = ce_loss / 4" in script
    assert "optimizer_grad_scaler.scale(ce_loss).backward()" in script
    assert "optimizer_grad_scaler.unscale_(optimizer)" in script
    assert "dispatch_clip_grad" in script
    assert "if __need_update__:" in script
    assert "self.need_update = (step + 1) % 4 == 0" in script
