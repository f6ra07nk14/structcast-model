"""Advanced builder tests using real cfg templates."""

from pathlib import Path

import pytest

from structcast_model.builders.torch import TorchLearnerBuilder
from structcast_model.utils.base import load_any
from tests import CFG_DIR

LEARNER_YAML = CFG_DIR / "torch" / "learners" / "ConvNeXtV2.yaml"


def fp16_builder() -> TorchLearnerBuilder:
    """Build the learner in the float16 + gradient-scaler configuration.

    The template fixes the mixed-precision keys at its root instead of exposing them as template
    variables, so fp16 is selected by overriding those keys on the loaded config.
    """
    raw = {**load_any(LEARNER_YAML), "MIXED_PRECISION": True, "MIXED_PRECISION_TYPE": "float16"}
    return TorchLearnerBuilder(raw=raw, current_path=str(LEARNER_YAML))


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
    """Default config uses bfloat16 autocast, which must not construct a gradient scaler."""
    built = TorchLearnerBuilder.from_path(LEARNER_YAML)()
    assert built.mixed_precision_type == "bfloat16"
    assert built.mixed_precision_scales == []


def test_learner_float16_builds_a_grad_scaler() -> None:
    """float16 gradients underflow without scaling, so fp16 configs must build a scaler."""
    built = fp16_builder()()
    assert built.mixed_precision_type == "float16"
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
    """fp16 scripts build the scaler through the injectable creator, on the training device."""
    script = fp16_builder()().scripts[0]
    assert "__grad_scaler_creator__=torch.amp.GradScaler" in script
    assert "__grad_scaler_creator__(device=device_type" in script


def test_learner_bfloat16_script_has_no_grad_scaler() -> None:
    """bfloat16 shares float32's exponent range: a scaler would be pure overhead."""
    script = TorchLearnerBuilder.from_path(LEARNER_YAML)().scripts[0]
    assert "GradScaler" not in script


def test_learner_script_gates_model_invocations() -> None:
    """Every model call is wrapped in a sync gate so distributed reducers arm exactly once."""
    script = TorchLearnerBuilder.from_path(LEARNER_YAML)().scripts[0]
    assert "sync_gate(model, __need_update__)" in script
    assert script.index("sync_gate(model, __need_update__)") < script.index("cls = model(image)")
    assert "def _sync_gate(module, armed):" not in script  # the package helper, never an inline copy
    assert 'restore_requires_grad(model, self._requires_grad_defaults["model"])' in script
    assert "def _restore" not in script  # the package helper, never an inline copy


def test_learner_script_defines_steps_as_methods() -> None:
    """The steps are class-level methods, not closures bound onto the instance in `__init__`.

    The bodies rebind the models and optimizers off `self`, so a compiled `_flow_*` function can be
    swapped on the instance and still be picked up; the training method reads `need_update` off
    `self` too, so no wrapper has to thread it through.
    """
    script = TorchLearnerBuilder.from_path(LEARNER_YAML)().scripts[0]
    assert "    def training_step(self, image, label, **kwargs):" in script
    assert "__need_update__ = self.need_update" in script
    assert "    @torch.no_grad()\n    def inference_step(self, image, label, **kwargs):" in script
    assert "self.training_step = training_step" not in script
    assert "self.inference_step = inference_step" not in script
    assert "forward_training_step" not in script
    assert "forward_inference_step" not in script


def test_learner_script_exposes_properties() -> None:
    """Exposes models, optimizers, optimizer_models, grad_scalers, learning_rates, weight_decays, param_group_names."""
    script = TorchLearnerBuilder.from_path(LEARNER_YAML)().scripts[0]
    for prop in (
        "def models(self)",
        "def optimizers(self)",
        "def optimizer_models(self)",
        "def grad_scalers(self)",
        "def learning_rates(self)",
        "def weight_decays(self)",
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
    """Script scales only the backward pass, keeps the need_update guard and modular update."""
    params = {"DEFAULT": {"accumulate_gradients": 4}}
    script = TorchLearnerBuilder.from_path(LEARNER_YAML)(parameters=params).scripts[0]
    assert "(ce_loss / 4).backward()" in script
    assert "if __need_update__:" in script
    assert "self.need_update = (step + 1) % 4 == 0" in script
    assert "return self.need_update" in script


def test_learner_accumulate_gradients_reports_unscaled_loss() -> None:
    """The tracked loss must not be rebound to the accumulation-scaled value.

    Reporting ``loss / accumulate_gradients`` makes training curves incomparable between an
    accumulating run and an equivalent multi-device run at the same global batch size.
    """
    params = {"DEFAULT": {"accumulate_gradients": 4}}
    script = TorchLearnerBuilder.from_path(LEARNER_YAML)(parameters=params).scripts[0]
    assert "ce_loss = ce_loss / 4" not in script
    assert "return {'ce_loss': ce_loss," in script


# ---------------------------------------------------------------------------
# TorchLearnerBuilder: gradient clipping
# ---------------------------------------------------------------------------


def test_learner_clip_grad_norm_in_script() -> None:
    """Gradient clipping appears in the script; with fp16 it unscales before clipping."""
    params = {"DEFAULT": {"clip_grad_norm": 2.0}}
    script = TorchLearnerBuilder.from_path(LEARNER_YAML)(parameters=params).scripts[0]
    assert "dispatch_clip_grad" in script
    assert "unscale_" not in script  # bf16 default has no scaler to unscale
    fp16_script = fp16_builder()(parameters={"DEFAULT": {"clip_grad_norm": 2.0}}).scripts[0]
    assert "optimizer_grad_scaler.unscale_(optimizer)" in fp16_script


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
    script = fp16_builder()(parameters=params).scripts[0]
    assert "optimizer_grad_scaler.scale(ce_loss).backward()" in script
    assert "ce_loss = ce_loss /" not in script


def test_learner_mp_scale_backward_with_accumulation() -> None:
    """With accumulation loss is divided and backward uses scaler."""
    params = {"DEFAULT": {"accumulate_gradients": 2}}
    script = fp16_builder()(parameters=params).scripts[0]
    assert "ce_loss = ce_loss / 2" not in script
    assert "optimizer_grad_scaler.scale((ce_loss / 2)).backward()" in script
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
    """Collected imports include torch always, the sync gate always, and torch.amp only for fp16."""
    imports = TorchLearnerBuilder.from_path(LEARNER_YAML)().collected_imports
    assert "torch" in imports
    assert "sync_gate" in imports["structcast_model.torch.distributed"]
    assert "get_decays" in imports["structcast_model.torch.optimizers"]
    assert "contextlib" not in imports
    assert "torch.amp" not in imports
    fp16_imports = fp16_builder()().collected_imports
    assert "torch.amp" in fp16_imports


def test_learner_script_calls_the_optimizer_referenced_by_file_path(tmp_path: Path) -> None:
    """A `_file_` optimizer reference cannot be imported by module name.

    The rendered script must bind the class through import_from_address, or the generated Learner
    raises NameError at construction time.
    """
    built = TorchLearnerBuilder.from_path(LEARNER_YAML)()
    assert "AdamWWithCosine(" in built.scripts[0]
    # The module is imported for `get_decays`, but the file-referenced class must not ride along.
    assert "AdamWWithCosine" not in built.collected_imports["structcast_model.torch.optimizers"]
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
    built = fp16_builder()(parameters=params, classname="FullCombo")
    script = built.scripts[0]
    assert built.classname == "FullCombo"
    assert built.accumulate_gradients == 4
    assert "optimizer" in built.optimizers
    assert "optimizer_grad_scaler" in built.mixed_precision_scales
    assert "optimizer_grad_scaler.scale((ce_loss / 4)).backward()" in script
    assert "optimizer_grad_scaler.unscale_(optimizer)" in script
    assert "dispatch_clip_grad" in script
    assert "if __need_update__:" in script
    assert "self.need_update = (step + 1) % 4 == 0" in script
