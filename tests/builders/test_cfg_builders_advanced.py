"""Advanced builder tests using real cfg templates."""

from pathlib import Path

import pytest

from structcast_model.builders.torch_builder import TorchLearnerBuilder
from tests import ASSETS_DIR

LEARNER_YAML = ASSETS_DIR / "cfg" / "torch" / "ConvNeXtV2Learner.yaml"


# ---------------------------------------------------------------------------
# TorchLearnerBuilder: basic build
# ---------------------------------------------------------------------------


def test_learner_default_params_classname_and_io() -> None:
    """Build the learner with defaults and check classname, inputs, outputs."""
    built = TorchLearnerBuilder.from_path(LEARNER_YAML)(classname="ConvNeXtLearner")
    assert built.classname == "ConvNeXtLearner"
    assert built.inputs == ["image", "label"]
    assert built.outputs == ["ce_loss", "acc1", "acc5"]


def test_learner_default_models_and_optimizers() -> None:
    """Default build should expose one model and one optimizer."""
    built = TorchLearnerBuilder.from_path(LEARNER_YAML)()
    assert built.models == ["model"]
    assert built.optimizers == ["optimizer"]


def test_learner_default_mixed_precision_type() -> None:
    """Default config uses bfloat16 mixed precision."""
    built = TorchLearnerBuilder.from_path(LEARNER_YAML)()
    assert built.mixed_precision_type == "bfloat16"
    assert built.mixed_precision_scales == ["optimizer_grad_scaler"]


def test_learner_default_no_accumulation() -> None:
    """Default config does not accumulate gradients."""
    built = TorchLearnerBuilder.from_path(LEARNER_YAML)()
    assert built.accumulate_gradients is None


# ---------------------------------------------------------------------------
# TorchLearnerBuilder: script content
# ---------------------------------------------------------------------------


def test_learner_script_contains_autocast() -> None:
    """Default bfloat16 config wraps forward in autocast."""
    script = TorchLearnerBuilder.from_path(LEARNER_YAML)().scripts[0]
    assert "torch.autocast(device_type, torch.bfloat16)" in script


def test_learner_script_contains_grad_scaler() -> None:
    """GradScaler instantiation appears in the generated script."""
    script = TorchLearnerBuilder.from_path(LEARNER_YAML)().scripts[0]
    assert "torch.amp.GradScaler(" in script


def test_learner_script_defines_training_and_inference_steps() -> None:
    """Script defines both _training_step and _inference_step functions."""
    script = TorchLearnerBuilder.from_path(LEARNER_YAML)().scripts[0]
    assert "def _training_step(" in script
    assert "def _inference_step(" in script


def test_learner_script_exposes_properties() -> None:
    """Script exposes models, optimizers, grad_scalers, learning_rates, param_group_names."""
    script = TorchLearnerBuilder.from_path(LEARNER_YAML)().scripts[0]
    for prop in (
        "def models(self)",
        "def optimizers(self)",
        "def grad_scalers(self)",
        "def learning_rates(self)",
        "def param_group_names(self)",
    ):
        assert prop in script


def test_learner_script_is_compilable() -> None:
    """Generated script can be compiled without syntax errors."""
    scripts = TorchLearnerBuilder.from_path(LEARNER_YAML)().scripts
    for script in scripts:
        compile(script, "<test>", "exec")


# ---------------------------------------------------------------------------
# TorchLearnerBuilder: gradient accumulation
# ---------------------------------------------------------------------------


def test_learner_accumulate_gradients_stored() -> None:
    """Accumulation count is stored on the intermediate."""
    params = {"DEFAULT": {"accumulate_gradients": 4}}
    built = TorchLearnerBuilder.from_path(LEARNER_YAML)(parameters=params)
    assert built.accumulate_gradients == 4


def test_learner_accumulate_gradients_script_patterns() -> None:
    """Script contains loss division, need_update guard, and modular update."""
    params = {"DEFAULT": {"accumulate_gradients": 4}}
    script = TorchLearnerBuilder.from_path(LEARNER_YAML)(parameters=params).scripts[0]
    assert "ce_loss = ce_loss / 4" in script
    assert "if __need_update__:" in script
    assert "self.need_update = (step + 1) % 4 == 0" in script
    assert "return self.need_update" in script


# ---------------------------------------------------------------------------
# TorchLearnerBuilder: gradient clipping
# ---------------------------------------------------------------------------


def test_learner_clip_grad_norm_in_script() -> None:
    """Gradient clipping function appears in the script when configured."""
    params = {"DEFAULT": {"clip_grad_norm": 2.0}}
    script = TorchLearnerBuilder.from_path(LEARNER_YAML)(parameters=params).scripts[0]
    assert "dispatch_clip_grad" in script
    assert "optimizer_grad_scaler.unscale_(optimizer)" in script


def test_learner_no_clip_when_null() -> None:
    """No clipping code when clip_grad_norm is null."""
    params = {"DEFAULT": {"clip_grad_norm": None}}
    script = TorchLearnerBuilder.from_path(LEARNER_YAML)(parameters=params).scripts[0]
    assert "dispatch_clip_grad" not in script
    assert "unscale_" not in script


# ---------------------------------------------------------------------------
# TorchLearnerBuilder: mixed precision backward pass
# ---------------------------------------------------------------------------


def test_learner_mp_scale_backward_without_accumulation() -> None:
    """Without accumulation the scaler.scale().backward() has no division."""
    params = {"DEFAULT": {"accumulate_gradients": None}}
    script = TorchLearnerBuilder.from_path(LEARNER_YAML)(parameters=params).scripts[0]
    assert "optimizer_grad_scaler.scale(ce_loss).backward()" in script
    assert "ce_loss = ce_loss /" not in script


def test_learner_mp_scale_backward_with_accumulation() -> None:
    """With accumulation loss is divided and backward uses scaler."""
    params = {"DEFAULT": {"accumulate_gradients": 2}}
    script = TorchLearnerBuilder.from_path(LEARNER_YAML)(parameters=params).scripts[0]
    assert "ce_loss = ce_loss / 2" in script
    assert "optimizer_grad_scaler.scale(ce_loss).backward()" in script
    assert "optimizer_grad_scaler.step(optimizer)" in script
    assert "optimizer_grad_scaler.update()" in script


# ---------------------------------------------------------------------------
# TorchLearnerBuilder: layer decay types
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("decay_type", ["single", "group"])
def test_learner_layer_decay_types_produce_regexes(decay_type: str) -> None:
    """Both single and group layer decay types produce layer_group_regexes."""
    params = {"DEFAULT": {"layer_decay_type": decay_type}}
    script = TorchLearnerBuilder.from_path(LEARNER_YAML)(parameters=params).scripts[0]
    assert "layer_group_regexes" in script


def test_learner_no_layer_decay_produces_empty_regexes() -> None:
    """Null layer_decay_type produces empty layer_group_regexes."""
    params = {"DEFAULT": {"layer_decay_type": None}}
    script = TorchLearnerBuilder.from_path(LEARNER_YAML)(parameters=params).scripts[0]
    assert "'layer_group_regexes': []" in script


def test_learner_invalid_layer_decay_type_raises() -> None:
    """Raise from Jinja filter when unsupported layer_decay_type is provided."""
    params = {"DEFAULT": {"layer_decay_type": "not-supported"}}
    with pytest.raises(ValueError, match="Invalid layer_decay_type"):
        TorchLearnerBuilder.from_path(LEARNER_YAML)(parameters=params)


# ---------------------------------------------------------------------------
# TorchLearnerBuilder: backbone variants
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backbone", ["atto", "femto", "pico", "nano", "tiny"])
def test_learner_backbone_variants_compile(backbone: str) -> None:
    """Each backbone variant produces a compilable script."""
    params = {"DEFAULT": {"backbone": backbone}}
    scripts = TorchLearnerBuilder.from_path(LEARNER_YAML)(parameters=params).scripts
    for script in scripts:
        compile(script, "<test>", "exec")


# ---------------------------------------------------------------------------
# TorchLearnerBuilder: collected imports
# ---------------------------------------------------------------------------


def test_learner_collected_imports_include_torch_and_amp() -> None:
    """Collected imports include torch and torch.amp for mixed precision."""
    built = TorchLearnerBuilder.from_path(LEARNER_YAML)()
    imports = built.collected_imports
    assert "torch" in imports
    assert "torch.amp" in imports


def test_learner_script_calls_the_optimizer_referenced_by_file_path(tmp_path: Path) -> None:
    """A `_file_` optimizer reference cannot be imported by module name.

    The rendered script must bind the class through import_from_address, or the generated Learner
    raises NameError at construction time.
    """
    built = TorchLearnerBuilder.from_path(LEARNER_YAML)()
    assert "AdamWWithCosine(" in built.scripts[0]
    assert "structcast_model.torch.optimizers" not in built.collected_imports
    script_path = tmp_path / "learner.py"
    built(script_path)
    code = script_path.read_text(encoding="utf-8")
    resolved = str(Path("examples/torch/optimizers.py").resolve())
    assert f"AdamWWithCosine = import_from_address('AdamWWithCosine', module_file={resolved!r})" in code
    assert "from structcast.utils.base import import_from_address" in code


# ---------------------------------------------------------------------------
# TorchLearnerBuilder: full accumulation + clipping + mp combo
# ---------------------------------------------------------------------------


def test_learner_full_combo_accumulate_clip_mp() -> None:
    """Combine accumulation, clipping, and mixed precision in one build."""
    params = {"DEFAULT": {"accumulate_gradients": 4, "clip_grad_norm": 2.0, "layer_decay_type": "single"}}
    built = TorchLearnerBuilder.from_path(LEARNER_YAML)(parameters=params, classname="FullCombo")
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
