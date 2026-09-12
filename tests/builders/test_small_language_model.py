"""Behaviour tests for the model `cfg/torch/models/SmallLanguageModel.yaml` generates.

The causal attention of that example is no longer a package layer but a section of the configuration
itself, so what has to hold — a position never sees the future, and positions are distinguishable
without a learned table — can only be checked on the generated model.
"""

from importlib.util import module_from_spec, spec_from_file_location
from types import ModuleType

import pytest

from structcast_model.builders.torch import TorchBuilder
from structcast_model.torch.distributed import matched_shard_modules
from tests import TEST_DIR
import torch

MODEL_YAML = TEST_DIR.parent / "cfg" / "torch" / "models" / "SmallLanguageModel.yaml"

VOCAB_SIZE = 65
"""The default `vocab_size` of the configuration, i.e. the token range the tests may sample from."""


@pytest.fixture(scope="module")
def model(tmp_path_factory: pytest.TempPathFactory) -> torch.nn.Module:
    """Generate the tiny preset from the configuration file and instantiate it in evaluation mode."""
    module_path = tmp_path_factory.mktemp("generated") / "small_language_model.py"
    TorchBuilder.from_path(MODEL_YAML)()(module_path)
    spec = spec_from_file_location(module_path.stem, module_path)
    assert spec is not None
    assert spec.loader is not None
    module: ModuleType = module_from_spec(spec)
    spec.loader.exec_module(module)
    torch.manual_seed(0)
    return module.Model().eval()


def test_generated_model_never_attends_to_later_tokens(model: torch.nn.Module) -> None:
    """Rewrite the future of a sequence and every earlier logit must stay identical.

    Next-token training is only honest if a position cannot read the answer it is asked to predict,
    so this is the property the whole `is_causal=True` attention section exists for.
    """
    tokens = torch.arange(16).remainder(VOCAB_SIZE).unsqueeze(0)
    rewritten = tokens.clone()
    rewritten[:, 8:] = torch.arange(8, 16).remainder(VOCAB_SIZE).flip(0) + 20

    with torch.no_grad():
        logits, changed = model(tokens), model(rewritten)

    assert torch.equal(logits[:, :8], changed[:, :8])
    assert not torch.allclose(logits[:, 8:], changed[:, 8:], atol=1e-6)


def test_generated_model_distinguishes_token_order(model: torch.nn.Module) -> None:
    """Swapping two earlier tokens must change a later position's logits.

    Attention alone is order-blind: without positions, a later position sums over the same set of
    tokens either way. The rotary embedding is the only thing carrying order here — the learned
    position table is gone — so this failing would mean position never reached the attention scores.
    """
    tokens = torch.arange(16).remainder(VOCAB_SIZE).unsqueeze(0)
    swapped = tokens.clone()
    swapped[:, [2, 5]] = swapped[:, [5, 2]]

    with torch.no_grad():
        logits, reordered = model(tokens), model(swapped)

    assert not torch.allclose(logits[:, 9], reordered[:, 9], atol=1e-6)
    # Position 1 attends only to tokens 0 and 1, which the swap left alone.
    assert torch.equal(logits[:, 1], reordered[:, 1])


def test_generated_blocks_are_addressable_for_per_block_sharding(model: torch.nn.Module) -> None:
    """`backbone.block*` must match the blocks and nothing else.

    `cfg/torch/strategies/fsdp2.yaml` shards on exactly that pattern, so a block renamed or inlined
    into the backbone would silently turn per-block FSDP2 into a single group.
    """
    paths = dict(model.named_modules())
    assert "backbone.block0" in paths

    matched = [path for path, _ in matched_shard_modules({"model": model}, ["backbone.block*"])["model"]]

    assert matched == [f"backbone.block{i}" for i in range(4)]
    assert all(type(paths[path]) is type(paths["backbone.block0"]) for path in matched)
