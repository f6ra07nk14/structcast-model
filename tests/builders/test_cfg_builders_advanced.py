"""Advanced builder tests using real cfg templates."""

import pytest

from structcast_model.builders.torch_builder import TorchBackwardBuilder, TorchBuilder
from tests import ASSETS_DIR


def test_cfg_convnext_model_builds_expected_topology() -> None:
    """Build ConvNeXt model from cfg and check key topology outputs."""
    parameters = {"DEFAULT": {"backbone": "atto", "num_classes": 8}}
    builder = TorchBuilder.from_path(ASSETS_DIR / "cfg/ConvNeXtV2.yaml")
    built = builder(parameters=parameters, classname="ConvNeXtAtto")
    assert built.classname == "ConvNeXtAtto"
    assert built.structured_output is True
    assert built.outputs == ["cls"]
    assert "backbone" in built.layers
    assert "head" in built.layers
    assert any(i.startswith("backbone_output[") and o == "feature" and L is None for i, o, L in built.flow)
    assert len(built.scripts) > 0


def test_cfg_convnext_backward_supports_accumulation_and_mp() -> None:
    """Build ConvNeXt backward from cfg and verify generated control flow."""
    parameters = {
        "DEFAULT": {
            "backbone": "atto",
            "accumulate_gradients": 4,
            "layer_decay_type": "single",
            "clip_grad_norm": 2.0,
        }
    }
    builder = TorchBackwardBuilder.from_path(ASSETS_DIR / "cfg/ConvNeXtV2Backward.yaml")
    built = builder(parameters=parameters, classname="ConvNeXtBackward")
    script = built.scripts[0]
    assert built.classname == "ConvNeXtBackward"
    assert built.accumulate_gradients == 4
    assert "optimizer" in built.optimizers
    assert "torch.amp.GradScaler(" in script
    assert "ce_loss = ce_loss / 4" in script
    assert "self.optimizer_scaler.scale(ce_loss).backward()" in script
    assert "return should_update" in script


def test_cfg_convnext_backward_invalid_layer_decay_type_raises() -> None:
    """Raise from Jinja filter when unsupported layer_decay_type is provided."""
    parameters = {"DEFAULT": {"layer_decay_type": "not-supported"}}
    with pytest.raises(ValueError, match="Invalid layer_decay_type"):
        TorchBackwardBuilder.from_path(ASSETS_DIR / "cfg/ConvNeXtV2Backward.yaml")(parameters=parameters)
