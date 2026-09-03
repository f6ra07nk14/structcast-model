"""Advanced builder tests using real cfg templates."""

from pathlib import Path
from re import findall
from typing import Any

import pytest
from timm.utils.clip_grad import dispatch_clip_grad

from structcast_model.builders.torch import TorchLearnerBuilder
from structcast_model.utils.base import load_any
from tests import CFG_DIR
import torch

LEARNER_YAML = CFG_DIR / "torch" / "learners" / "ConvNeXtV2.yaml"


def fp16_builder() -> TorchLearnerBuilder:
    """Build the learner in the float16 + gradient-scaler configuration.

    The template fixes the mixed-precision keys at its root instead of exposing them as template
    variables, so fp16 is selected by overriding those keys on the loaded config.
    """
    raw = {**load_any(LEARNER_YAML), "MIXED_PRECISION": True, "MIXED_PRECISION_TYPE": "float16"}
    return TorchLearnerBuilder(raw=raw, current_path=str(LEARNER_YAML))


def rendered_module(built: Any, tmp_path: Path) -> str:
    """Render one built learner to the whole module, not just the class script.

    What a binding of literal arguments renders to lives above the class, next to the imports: the
    builder hoists it there so every instance shares the one callable object.
    """
    built(path := tmp_path / "learner.py")
    return path.read_text(encoding="utf-8")


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


def test_learner_class_documents_the_compile_seam_and_update_gating() -> None:
    """The generated class carries the docstring; nothing else in the emitted file explains it.

    Whoever opens `learner.py` has to learn from the file itself that `flow_functions` is the seam a
    trainer rebinds compiled, and that the learner's own counters decide when the optimizers step.
    """
    script = TorchLearnerBuilder.from_path(LEARNER_YAML)().scripts[0]
    docstring = script.split("class Learner:\n", 1)[1].split('"""')[1]

    assert docstring.startswith("Learner generated from a PyTorch learner template.")
    assert "`flow_functions`" in docstring
    assert "`has_updated`" in docstring


def test_learner_script_explains_itself_without_citing_repository_documents() -> None:
    """A generated learner is read where this repository is not, so a citation there names nothing.

    The float16 caveat is the one that has to survive as prose: `has_updated` reports the intent to
    apply, which the gradient scaler may still skip.
    """
    script = TorchLearnerBuilder.from_path(LEARNER_YAML)().scripts[0]

    assert "docs/adr" not in script
    assert "# Intent, not detection: under float16" in script


def test_learner_script_leaves_no_commented_out_layer_assignments() -> None:
    """The layers are locals the flow functions close over; nothing reads them off the learner.

    A commented-out assignment for each of them is noise a reader has to rule out as a leftover,
    and the trainers scan a learner's attributes for the events it handles, never for its layers.
    """
    script = TorchLearnerBuilder.from_path(LEARNER_YAML)().scripts[0]

    assert "# self." not in script


def test_learner_script_contains_autocast() -> None:
    """Default bfloat16 config wraps forward in autocast."""
    script = TorchLearnerBuilder.from_path(LEARNER_YAML)().scripts[0]
    assert "torch.autocast(device_type, torch.bfloat16)" in script


def test_learner_script_contains_grad_scaler() -> None:
    """fp16 scripts construct the scaler directly, on the training device rather than the cuda default."""
    script = fp16_builder()().scripts[0]
    assert "torch.amp.GradScaler(device=device_type" in script
    assert "__grad_scaler_creator__" not in script


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


def test_learner_script_bind_arguments_are_deterministic(tmp_path: Path) -> None:
    """Bind-lambda argument names derive from the pattern position, never from id().

    An id()-derived suffix changes with every process, so the same template rendered twice would
    differ byte-for-byte -- phantom diffs for committed scripts and no way to hash-check "already
    generated". The template's binds must therefore always render as `_arg0`, and the name each one
    is hoisted under must come from the expression, which is stable for the same reason.
    """
    module = rendered_module(TorchLearnerBuilder.from_path(LEARNER_YAML)(), tmp_path)

    assert set(findall(r"_arg\d+", module)) == {"_arg0"}
    assert module == rendered_module(TorchLearnerBuilder.from_path(LEARNER_YAML)(), tmp_path)


def test_learner_script_defines_steps_as_methods() -> None:
    """The steps are class-level methods, not closures bound onto the instance in `__init__`.

    The bodies rebind the models and optimizers off `self`, so a compiled `_flow_*` function can be
    swapped on the instance and still be picked up; the training method counts `self._steps` and
    computes the gate itself, so no wrapper has to thread it through.
    """
    script = TorchLearnerBuilder.from_path(LEARNER_YAML)().scripts[0]
    assert "    def training_step(self, image, label, **kwargs):" in script
    assert "self._steps += 1" in script
    assert "__need_update__ = True" in script
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
    """Script scales only the backward pass, keeps the need_update guard, and counts on the learner.

    Incrementing `_steps` before the `(self._steps + 1) % 4` gate keeps the historical 1-based
    cadence: the first window is one step short, so the applies land at steps 3, 7, 11.
    """
    params = {"DEFAULT": {"accumulate_gradients": 4}}
    script = TorchLearnerBuilder.from_path(LEARNER_YAML)(parameters=params).scripts[0]
    assert "(ce_loss / 4).backward()" in script
    assert "if __need_update__:" in script
    assert "__need_update__ = (self._steps + 1) % 4 == 0" in script
    assert script.index("self._steps += 1") < script.index("__need_update__ = (self._steps + 1) % 4 == 0")
    assert "self._updates += 1" in script
    assert "self._has_updated = __need_update__" in script
    # The intent-vs-detection note belongs where a reader of the generated learner meets the flag.
    assert script.index("# Intent, not detection") < script.index("self._has_updated = __need_update__")


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


@pytest.mark.parametrize("template", ["ConvNeXtV2", "ImageClassifier", "ImageClassifierShowcase", "SmallLanguageModel"])
def test_learner_clip_grad_norm_is_the_threshold_not_the_p_norm(template: str, tmp_path: Path) -> None:
    """Every torch learner template must hand `clip_grad_norm` to timm as `value`, never `norm_type`.

    `value` is the norm the gradients are scaled down to and `norm_type` is the p of that norm, so
    binding the knob to `norm_type` leaves the threshold at the literal beside it and turns a
    request to clip at 2.0 into an L2-vs-p choice clipped at 1.0. It is a silent inversion -- the
    build succeeds and training runs -- and it would put the torch templates at odds with the Flax
    (`optax.clip_by_global_norm`) and Keras (`global_clipnorm`) twins, where the same parameter has
    always been the L2 bound.
    """
    yaml = CFG_DIR / "torch" / "learners" / f"{template}.yaml"
    built = TorchLearnerBuilder.from_path(yaml)(parameters={"DEFAULT": {"clip_grad_norm": 2.0}})
    assert "dispatch_clip_grad(*_arg0, value=2.0, mode='norm', norm_type=2.0, **_kw0)" in rendered_module(
        built, tmp_path
    )


def test_dispatch_clip_grad_value_is_the_l2_bound() -> None:
    """What the templates assume of timm: `value` bounds the global norm, and only when exceeded.

    Pinned against the real `dispatch_clip_grad` because the binding above is only correct for as
    long as this holds; a timm release that renamed or reordered these would otherwise land as a
    silent change in what every torch template clips at.
    """

    def gradients() -> list[torch.nn.Parameter]:
        """A single parameter whose gradient has a global L2 norm of exactly 5.0."""
        parameter = torch.nn.Parameter(torch.zeros(2))
        parameter.grad = torch.tensor([3.0, 4.0])
        return [parameter]

    (loose,) = gradients()
    dispatch_clip_grad(loose, value=100.0, mode="norm", norm_type=2.0)
    assert loose.grad is not None
    assert loose.grad.norm().item() == pytest.approx(5.0)

    (tight,) = gradients()
    dispatch_clip_grad(tight, value=1.0, mode="norm", norm_type=2.0)
    assert tight.grad is not None
    assert tight.grad.norm().item() == pytest.approx(1.0)


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
def test_learner_layer_decay_types_produce_regexes(decay_type: str, tmp_path: Path) -> None:
    """Both single and group layer decay types produce layer_group_regexes."""
    params = {"DEFAULT": {"layer_decay_type": decay_type}}
    module = rendered_module(TorchLearnerBuilder.from_path(LEARNER_YAML)(parameters=params), tmp_path)
    assert "layer_group_regexes" in module


def test_learner_no_layer_decay_produces_empty_regexes(tmp_path: Path) -> None:
    """Null layer_decay_type produces empty layer_group_regexes."""
    params = {"DEFAULT": {"layer_decay_type": None}}
    module = rendered_module(TorchLearnerBuilder.from_path(LEARNER_YAML)(parameters=params), tmp_path)
    assert "'layer_group_regexes': []" in module


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
    # The module is imported for `get_decays`, but the file-referenced class must not ride along.
    assert "AdamWWithCosine" not in built.collected_imports["structcast_model.torch.optimizers"]
    code = rendered_module(built, tmp_path)
    assert "AdamWWithCosine(" in code
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
    assert "__need_update__ = (self._steps + 1) % 4 == 0" in script


# ---------------------------------------------------------------------------
# TorchLearnerBuilder: EMA shadow models
# ---------------------------------------------------------------------------


UNWRAPPED = "(model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model)"
"""What the average is taken over: by wrapper type, so a model owning a `module` keeps its own."""


def _ema_script(**ema: Any) -> str:
    """Build the learner with an `EMA` over its model and return the script holding the class."""
    raw = {**load_any(LEARNER_YAML), "EMA": {"model": ema or True}}
    return TorchLearnerBuilder(raw=raw, current_path=str(LEARNER_YAML))().scripts[0]


def test_learner_emits_the_average_over_the_module_a_wrapper_holds() -> None:
    """An `AveragedModel` copies what it is given, and the models reach the learner already wrapped.

    Copying the DDP wrapper is not possible at all, and averaging is meant to happen over the weights
    either way, so the average is built over what the wrapper holds. `multi_avg_fn` is what makes it
    exponential: without it torch averages every Update equally, which is a different feature under
    the same key. The build stays sharding-aware where it cannot work -- a DTensor parameter list is
    one FSDP2 refuses to copy and one the averaging kernel refuses to blend -- and the average is
    never trained, so it is put in eval mode once (`docs/adr/0021`).
    """
    script = _ema_script()

    assert 'if any(type(p).__name__ == "DTensor" for p in model.parameters()):' in script
    assert "which an AveragedModel cannot average" in script
    assert "torch._foreach_lerp_" in script
    assert (
        f"ema_model = torch.optim.swa_utils.AveragedModel({UNWRAPPED}, "
        "multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(0.999))"
    ) in script
    assert "ema_model.eval()" in script
    assert script.index("DTensor") < script.index("ema_model = torch.optim.swa_utils.AveragedModel")
    assert (
        "torch.optim.swa_utils"
        in TorchLearnerBuilder(
            raw={**load_any(LEARNER_YAML), "EMA": {"model": True}}, current_path=str(LEARNER_YAML)
        )().collected_imports
    )


def test_learner_blends_the_average_once_per_update_and_persists_it_as_a_model() -> None:
    """The blend rides the Update gate, and the `models` property is what carries it to a checkpoint.

    Blending on every call would advance the average against gradients no optimizer has applied; the
    average also has to be in `models`, which is the only path a trainer saves and restores through.
    """
    script = _ema_script()

    assert "if self._has_updated:" in script
    assert f"self.ema_model.update_parameters({UNWRAPPED})" in script
    assert script.index("self._has_updated = __need_update__") < script.index("if self._has_updated:")
    assert '"model": self.model, "ema_model": self.ema_model' in script


def test_learner_ema_mapping_completes_the_defaults_it_leaves_out() -> None:
    """A mapping declares keywords, not a different mechanism: what it omits stays what `true` means."""
    script = _ema_script(use_buffers=True)

    assert "multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(0.999), use_buffers=True)" in script


def test_learner_without_ema_is_emitted_as_it_was_before_the_field_existed() -> None:
    """The field is opt-in: a learner that declares none may gain no line and no import from it."""
    built = TorchLearnerBuilder.from_path(LEARNER_YAML)()

    assert built.ema == ()
    assert "ema_" not in built.scripts[0]
    assert "torch.optim.swa_utils" not in built.collected_imports
