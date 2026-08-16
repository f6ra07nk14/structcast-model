"""Behaviour tests for the model `cfg/torch/models/VisionTransformer.yaml` generates.

The configuration claims to be `timm`'s `vit_base_patch16_224` written in the DSL, so the test that
matters transplants timm's own weights into the generated model and compares logits: anything the
DSL had to approximate -- the class token as a one-row embedding, the position table, the head split
of the fused projection -- would show up there. The remaining tests pin the properties the shared
`cfg/torch/learners/ImageClassifier.yaml` depends on.
"""

from importlib.util import module_from_spec, spec_from_file_location
from types import ModuleType
from typing import Any

import pytest
import timm

from structcast_model.builders.torch_builder import TorchBuilder, TorchLearnerBuilder
from structcast_model.torch.distributed import matched_shard_modules
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
    TorchBuilder.from_path(MODEL_YAML)(parameters={"SHARED": {"drop_path_rate": 0.0}})(module_path)
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
