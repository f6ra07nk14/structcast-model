"""Behaviour tests for the model `cfg/torch/models/VisionTransformer.yaml` generates.

The configuration claims to be `timm`'s `vit_base_patch16_224` written in the DSL, so the test that
matters transplants timm's own weights into the generated model and compares logits: anything the
DSL had to approximate -- the class token as a one-row embedding, the position table, the head split
of the fused projection -- would show up there. The remaining tests pin the properties the shared
`cfg/torch/learners/ImageClassifier.yaml` depends on.
"""

from importlib.util import module_from_spec, spec_from_file_location
from types import ModuleType
from typing import Any, cast

import pytest
import timm
from torch.distributed.fsdp import fully_shard

from structcast_model.builders.torch import TorchBuilder, TorchLearnerBuilder
from structcast_model.commands.utils import path_or_any_parser
from structcast_model.torch.distributed import matched_shard_modules
from structcast_model.torch.layers import GradientCheckpointingLayer
from tests import TEST_DIR
import torch

CFG_DIR = TEST_DIR.parent / "cfg" / "torch"
MODEL_YAML = CFG_DIR / "models" / "VisionTransformer.yaml"
CONVNEXT_YAML = CFG_DIR / "models" / "ConvNeXtV2.yaml"
LEARNER_YAML = CFG_DIR / "learners" / "ImageClassifier.yaml"

DEPTH = 12
"""The number of blocks of the default `base` preset, i.e. how many FSDP2 shards it has to expose."""

TIMM_TO_GENERATED = {
    "cls_token": "cls_token_embedding.weight",
    "pos_embed": "position_embedding.weight",
    "patch_embed.proj.weight": "patchify.weight",
    "patch_embed.proj.bias": "patchify.bias",
    "norm.weight": "layer_norm.weight",
    "norm.bias": "layer_norm.bias",
    "head.weight": "head.weight",
    "head.bias": "head.bias",
    **{
        f"blocks.{i}.{source}.{suffix}": f"backbone.block{i}.{target}.{suffix}"
        for i in range(DEPTH)
        for source, target in (
            ("norm1", "layer_norm"),
            ("attn.qkv", "self_attention.qkv_proj"),
            ("attn.proj", "self_attention.out_proj"),
            ("norm2", "layer_norm_1"),
            ("mlp.fc1", "linear"),
            ("mlp.fc2", "linear_1"),
        )
        for suffix in ("weight", "bias")
    },
}
"""Every timm `vit_base_patch16_224` parameter and the generated parameter it becomes.

`cls_token` and `pos_embed` carry a leading singleton batch axis that the embedding tables do not,
so those two are reshaped; the mapping is otherwise one to one, which the parity test asserts.
"""


def _load(module_path: Any) -> ModuleType:
    """Import the generated module written to *module_path*."""
    spec = spec_from_file_location(module_path.stem, module_path)
    assert spec is not None
    assert spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def model(tmp_path_factory: pytest.TempPathFactory) -> torch.nn.Module:
    """Generate the `base` preset with drop path disabled, as `vit_base_patch16_224` has it."""
    module_path = tmp_path_factory.mktemp("generated") / "vision_transformer.py"
    # The override must name the size group: a SHARED value never reaches a group scope whose own
    # parameters define the same name, so a bare SHARED override silently keeps the 0.1 ramp.
    TorchBuilder.from_path(MODEL_YAML)(parameters={"SHARED": {"drop_path_rate": 0.0}, "base": {}})(module_path)
    torch.manual_seed(0)
    return _load(module_path).Model().eval()


def test_generated_model_reproduces_timm_vit_base_patch16_224(model: torch.nn.Module) -> None:
    """Loading timm's weights into the generated model must reproduce timm's logits.

    This is what the configuration is for: a strategy comparison is only worth reading if the model
    under it is the published architecture, and every DSL workaround here (class token as a one-row
    `torch.nn.Embedding`, `Split` plus `Unflatten` instead of timm's single `reshape`) is only valid
    while this holds exactly.
    """
    reference = timm.create_model("vit_base_patch16_224", pretrained=False).eval()
    reference_state = reference.state_dict()
    generated_state = model.state_dict()

    assert set(TIMM_TO_GENERATED) == set(reference_state)
    assert set(TIMM_TO_GENERATED.values()) == set(generated_state)

    model.load_state_dict(
        {
            target: reference_state[source].reshape(generated_state[target].shape)
            for source, target in TIMM_TO_GENERATED.items()
        }
    )
    torch.manual_seed(0)
    image = torch.randn(2, 3, 224, 224)

    with torch.no_grad():
        logits, expected = model(image), reference(image)

    assert torch.allclose(logits, expected, atol=1e-5), (logits - expected).abs().max().item()


def test_generated_model_reads_every_patch(model: torch.nn.Module) -> None:
    """Rewriting the last patch must change the logits.

    The head reads the class token, which sits at position 0. Under the causal mask of
    `SmallLanguageModel.yaml` that token would see nothing after it and this would be invariant, so
    the assertion is exactly the `is_causal=False` of the attention section.
    """
    torch.manual_seed(0)
    image = torch.randn(1, 3, 224, 224)
    rewritten = image.clone()
    rewritten[:, :, -16:, -16:] = 1.0

    with torch.no_grad():
        logits, changed = model(image), model(rewritten)

    assert not torch.allclose(logits, changed, atol=1e-6)


def test_generated_model_distinguishes_patch_order(model: torch.nn.Module) -> None:
    """Swapping two patches must change the logits.

    Attention alone is order-blind, so a permuted image would give the same class token without the
    position table; this failing would mean the table never reached the tokens.
    """
    torch.manual_seed(0)
    image = torch.randn(1, 3, 224, 224)
    swapped = image.clone()
    swapped[:, :, :16, :16], swapped[:, :, -16:, -16:] = image[:, :, -16:, -16:], image[:, :, :16, :16]

    with torch.no_grad():
        logits, reordered = model(image), model(swapped)

    assert not torch.allclose(logits, reordered, atol=1e-6)


def test_generated_blocks_are_addressable_for_per_block_sharding(model: torch.nn.Module) -> None:
    """`backbone.block*` must match the blocks and nothing else.

    `cfg/torch/strategies/fsdp2.yaml` shards on exactly that pattern, so a block renamed or inlined
    into the backbone would silently turn per-block FSDP2 into a single group.
    """
    paths = dict(model.named_modules())
    assert "backbone.block0" in paths

    matched = [path for path, _ in matched_shard_modules({"model": model}, ["backbone.block*"])["model"]]

    assert matched == [f"backbone.block{i}" for i in range(DEPTH)]
    # Every backbone parameter has to sit inside a matched block; one left outside would silently
    # stay unsharded. The blocks are not of one class -- the drop path ramp gives each its own, as in
    # ConvNeXt V2 -- so the pattern, not the type, is what has to cover them.
    assert {name for name, _ in paths["backbone"].named_parameters()} == {
        f"{path.removeprefix('backbone.')}.{name}" for path in matched for name, _ in paths[path].named_parameters()
    }


def _shipped_plan(name: str) -> list[tuple[str, str]]:
    """The `parallel_modules` plan one shipped strategy template binds, as (glob, style) pairs."""
    # The parser is typed for the mapping form; a strategy template is the list form of the same DSL.
    bound = cast(Any, path_or_any_parser(str(CFG_DIR / "strategies" / name)))[2]["_bind_"]
    return [(pattern, style) for pattern, style in bound["parallel_modules"]]


def test_the_shipped_tensor_parallel_plans_name_this_models_layers(model: torch.nn.Module) -> None:
    """`cfg/torch/strategies/tp.yaml` is written for this template, so its globs have to match it.

    A plan whose patterns match nothing is refused at wrap time -- loud, but only once a distributed
    run has started -- and a plan that matched only half of a column/row pair would train a wrong
    answer, so the four layers and their styles are pinned here. `fsdp_tp.yaml` carries the same
    plan on the model axis of its two-dimensional mesh; the two drifting apart is the other half.
    """
    plan = _shipped_plan("tp.yaml")
    assert _shipped_plan("fsdp_tp.yaml") == plan

    matched = [path for path, _ in matched_shard_modules({"model": model}, [p for p, _ in plan])["model"]]

    assert [style for _, style in plan] == ["column_heads", "row", "column", "row"]
    assert matched == [
        f"backbone.block{i}.{layer}"
        for i in range(DEPTH)
        for layer in ("self_attention.qkv_proj", "self_attention.out_proj", "linear", "linear_1")
    ]


@pytest.mark.parametrize(
    ("cfg_path", "parameters"),
    [
        (MODEL_YAML, {"base": {"dim": 64, "heads": 4, "depth": 2}, "SHARED": {"image_size": 32, "num_classes": 10}}),
        (CONVNEXT_YAML, {"SHARED": {"num_classes": 10}}),
    ],
    ids=["vision_transformer", "convnext_v2"],
)
def test_image_classifier_learner_trains_both_models(
    tmp_path_factory: pytest.TempPathFactory, cfg_path: Any, parameters: dict[str, Any]
) -> None:
    """One learner has to drive both classifiers, so the strategy is the only variable in a comparison.

    Both roots must therefore keep taking `image` and emitting a bare `cls` tensor: a structured
    output or a renamed head would leave the learner wiring dangling.
    """
    generated = tmp_path_factory.mktemp("generated")
    TorchBuilder.from_path(cfg_path)(parameters=parameters)(generated / "model.py")
    TorchLearnerBuilder.from_path(LEARNER_YAML)()(generated / "learner.py")
    torch.manual_seed(0)
    model = _load(generated / "model.py").Model()
    image, label = torch.randn(2, 3, 32, 32), torch.tensor([1, 7])
    model(image)  # materialize the lazy layers of ConvNeXt V2 before the optimizer sees parameters

    learner = _load(generated / "learner.py").Learner(model)
    trained = learner.training_step(image=image, label=label)
    inferred = learner.inference_step(image=image, label=label)

    assert torch.isfinite(trained["ce_loss"])
    assert sorted(trained) == ["acc1", "acc5", "ce_loss"]
    assert sorted(inferred) == ["acc1", "acc5", "ce_loss"]


def test_image_classifier_learner_lowers_the_loss_on_a_fixed_batch(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """Twenty steps on one batch must bring the loss down.

    The optimizer is a native `torch.optim.AdamW` built by `create_opt` over regex-grouped
    parameters; a mis-grouped or empty parameter group would still return finite losses, but they
    would not move.
    """
    generated = tmp_path_factory.mktemp("generated")
    TorchBuilder.from_path(MODEL_YAML)(
        parameters={"base": {"dim": 64, "heads": 4, "depth": 2}, "SHARED": {"image_size": 32, "num_classes": 10}}
    )(generated / "model.py")
    TorchLearnerBuilder.from_path(LEARNER_YAML)()(generated / "learner.py")
    torch.manual_seed(0)
    model = _load(generated / "model.py").Model()
    learner = _load(generated / "learner.py").Learner(model)
    image, label = torch.randn(4, 3, 32, 32), torch.tensor([1, 7, 3, 9])

    losses = [float(learner.training_step(image=image, label=label)["ce_loss"]) for _ in range(20)]

    assert losses[-1] < losses[0] / 2


SHOWCASE_YAML = CFG_DIR / "learners" / "ImageClassifierShowcase.yaml"
"""The learner that turns accumulation, autocast and the EMA on at once."""

SHOWCASE_MODEL_PARAMETERS: dict[str, dict[str, Any]] = {
    "base": {"dim": 32, "heads": 2, "depth": 2},
    "SHARED": {"image_size": 16, "patch_size": 8, "num_classes": 5, "drop_path_rate": 0.0},
}
"""A two-block, four-patch Vision Transformer, with the stochastic depth of the recipe switched off.

`GRADIENT_CHECKPOINTING` is the one parameter left out: the two builds below set it either way, and
the whole point of the equality assertions is that nothing else differs between them.
"""


def _showcase_model(directory: Any, checkpointing: bool) -> Any:
    """Generate the showcase model with checkpointing on or off, from the same seed either way."""
    module_path = directory / f"model_{checkpointing}.py"
    shared = {**SHOWCASE_MODEL_PARAMETERS["SHARED"], "gradient_checkpointing": checkpointing}
    TorchBuilder.from_path(MODEL_YAML)(parameters={**SHOWCASE_MODEL_PARAMETERS, "SHARED": shared})(module_path)
    torch.manual_seed(0)
    return _load(module_path).Model()


def test_the_showcase_pair_runs_all_four_features_in_one_training_step(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """Checkpointing, accumulation, autocast and the EMA have to hold together, not just one by one.

    Each is pinned on its own elsewhere; what only a combined run can answer is whether they compose.
    They interact: the average must follow the accumulation gate rather than the step count, the
    checkpointed blocks are recomputed inside the autocast region, and the checkpointed model is the
    one the `AveragedModel` copies. A checkpointed layer must also stay the same layer -- `blockN` is
    what an FSDP2 `shard_modules` glob and a state dict address -- so the parameter names and the
    forward pass are asserted against the same model built without it.
    """
    generated = tmp_path_factory.mktemp("showcase")
    model = _showcase_model(generated, True)
    plain = _showcase_model(generated, False)
    TorchLearnerBuilder.from_path(SHOWCASE_YAML)(parameters={"DEFAULT": {"accumulate_gradients": 2}})(
        generated / "learner.py"
    )
    learner = _load(generated / "learner.py").Learner(model)
    image, label = torch.randn(4, 3, 16, 16), torch.tensor([0, 1, 2, 3])

    assert isinstance(model.backbone.block0, GradientCheckpointingLayer)
    assert model.backbone.block0.gradient_checkpointing is True
    assert not isinstance(plain.backbone.block0, GradientCheckpointingLayer)
    assert [name for name, _ in model.named_parameters()] == [name for name, _ in plain.named_parameters()]
    assert torch.allclose(model(image), plain(image))
    assert "with torch.autocast(device_type, torch.bfloat16):" in (generated / "learner.py").read_text()

    averages, gates = [], []
    for _ in range(4):
        learner.training_step(image=image, label=label)
        gates.append(learner.has_updated)
        averages.append(learner.ema_model.module.head.weight.detach().clone())

    assert gates == [True, False, True, False]  # the short first window torch has always had
    assert [not torch.equal(a, b) for a, b in zip(averages, averages[1:], strict=False)] == [False, True, False]
    assert list(learner.models) == ["model", "ema_model"]
    assert torch.isfinite(learner.inference_step(image=image, label=label)["ce_loss"])


def test_the_showcase_window_can_be_taken_off_by_parameter(tmp_path_factory: pytest.TempPathFactory) -> None:
    """Every knob on this template is one a run turns off, and accumulation had no off switch.

    `ACCUMULATE_GRADIENTS` rendered unguarded puts Python's `None` into the YAML as the *string*
    `None`, which the learner schema rejects outright -- so the showcase could not express "train
    without a window" at all, and the three backends disagreed about a parameter they are read side
    by side for. Off has to reach both halves of the mechanism, since they are emitted separately:
    the loss keeps its full scale and every step applies, which is what the divisor and the gate
    below say. Set, the divisor is the window, which is the half a guard is easiest to drop.
    """
    generated = tmp_path_factory.mktemp("window")
    sources = {}
    for name, window in (("off", None), ("on", 4)):
        TorchLearnerBuilder.from_path(SHOWCASE_YAML)(parameters={"DEFAULT": {"accumulate_gradients": window}})(
            generated / f"{name}.py"
        )
        sources[name] = (generated / f"{name}.py").read_text()
    learner = _load(generated / "off.py").Learner(_showcase_model(generated, False))
    image, label = torch.randn(4, 3, 16, 16), torch.tensor([0, 1, 2, 3])

    gates = [(learner.training_step(image=image, label=label), learner.has_updated)[1] for _ in range(2)]

    assert "ce_loss.backward()" in sources["off"]  # no window, so no divisor and no gate to miss
    assert "(ce_loss / 4).backward()" in sources["on"]  # the integer the window was set to
    assert gates == [True, True]  # without a window every step applies
    assert learner.updates == 2


def test_the_showcase_learner_adds_the_scaler_only_for_the_float16_precision(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """`mixed_precision_type` picks the autocast dtype and, with it, whether a scaler is built.

    Gradient scaling counteracts float16 underflow and nothing else, so the template derives
    `MIXED_PRECISION` from the type rather than exposing a second knob that can contradict it -- a
    `bfloat16` run asking for a scaler is a build-time `SpecError`, which is the failure this gate
    exists to make unreachable.
    """
    generated = tmp_path_factory.mktemp("precision")
    scripts = {}
    for precision in ("bfloat16", "float16"):
        path = generated / f"{precision}.py"
        TorchLearnerBuilder.from_path(SHOWCASE_YAML)(parameters={"DEFAULT": {"mixed_precision_type": precision}})(path)
        scripts[precision] = path.read_text()

    assert "torch.autocast(device_type, torch.bfloat16)" in scripts["bfloat16"]
    assert "GradScaler" not in scripts["bfloat16"]
    assert "torch.autocast(device_type, torch.float16)" in scripts["float16"]
    assert "torch.amp.GradScaler(device=device_type)" in scripts["float16"]


def test_the_showcase_drops_its_average_for_a_run_that_shards(
    tmp_path_factory: pytest.TempPathFactory, single_process_gloo: None
) -> None:
    """A sharding run refuses an EMA by design, so the showcase has to be able to leave it out.

    A DTensor parameter list is one FSDP2 refuses to copy and one the averaging kernel refuses to
    blend, which the generated learner turns into its own `__init__` failure -- so a template that
    declared `EMA` unconditionally could not run its other three features under FSDP2 or tensor
    parallelism at all. `ema: false` is the switch, and what it has to produce is a learner a sharded
    model can enter, validating over `model` rather than over an average that no longer exists.
    """
    generated = tmp_path_factory.mktemp("no_ema")
    model = fully_shard(_showcase_model(generated, True))
    image, label = torch.randn(4, 3, 16, 16), torch.tensor([0, 1, 2, 3])
    TorchLearnerBuilder.from_path(SHOWCASE_YAML)()(generated / "averaged.py")
    TorchLearnerBuilder.from_path(SHOWCASE_YAML)(parameters={"SHARED": {"ema": False}})(generated / "plain.py")

    with pytest.raises(ValueError, match="which an AveragedModel cannot average"):
        _load(generated / "averaged.py").Learner(model)
    learner = _load(generated / "plain.py").Learner(model)

    assert "ema_model" in (generated / "averaged.py").read_text()
    assert "AveragedModel" not in (script := (generated / "plain.py").read_text())
    assert list(learner.models) == ["model"]
    # The three features the sharded run is for are the ones that must survive the switch.
    assert "with torch.autocast(device_type, torch.bfloat16):" in script
    assert sorted(learner.training_step(image=image, label=label)) == ["acc1", "acc5", "ce_loss"]
    assert learner.has_updated is False  # the accumulation window the run is for, still gating
    assert torch.isfinite(learner.inference_step(image=image, label=label)["ce_loss"])


def test_the_image_classifier_learner_derives_both_precision_fields_from_one_parameter(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """A precision comparison varies an arm of the run, not the shipped file.

    `MIXED_PRECISION_TYPE` is the autocast dtype and `MIXED_PRECISION` is the `GradScaler`, which only
    float16 needs -- the two contradicting each other is a build-time `SpecError`, so one parameter
    derives both rather than exposing a pair a `-p` could break. The default arm is the one this
    template has always emitted.
    """
    generated = tmp_path_factory.mktemp("precision")
    scripts = {}
    for arm in (None, "bfloat16", "float16"):
        path = generated / f"{arm}.py"
        TorchLearnerBuilder.from_path(LEARNER_YAML)(parameters={"SHARED": {"mixed_precision_type": arm}})(path)
        scripts[arm] = path.read_text()
    TorchLearnerBuilder.from_path(LEARNER_YAML)()(generated / "default.py")

    assert (generated / "default.py").read_text() == scripts["bfloat16"]
    assert "torch.autocast" not in scripts[None]
    assert "GradScaler" not in scripts[None]
    assert "torch.autocast(device_type, torch.bfloat16)" in scripts["bfloat16"]
    assert "GradScaler" not in scripts["bfloat16"]
    assert "torch.autocast(device_type, torch.float16)" in scripts["float16"]
    assert "torch.amp.GradScaler(device=device_type)" in scripts["float16"]
