"""API-level tests for builder schema models."""

import pytest
from structcast.core.exceptions import SpecError

from structcast_model.builders.schema import (
    BackwardBehavior,
    LayerBehavior,
    OptimizerBehavior,
    TemplateBackward,
    UserDefinedBackward,
    UserDefinedLayer,
    resolve_flow,
)


def test_resolve_flow_returns_unique_inputs_and_outputs() -> None:
    """Resolve flow and keep deterministic unique order."""
    flow = [
        LayerBehavior.model_validate(["x", "h1", "l1"]),
        LayerBehavior.model_validate([["h1", "aux"], "h2", "l2"]),
        LayerBehavior.model_validate(["h2", "y", "l3"]),
    ]
    inputs, outputs = resolve_flow(flow)
    assert inputs == ["x", "aux"]
    assert outputs == ["h1", "h2", "y"]


def test_user_defined_layer_normalizes_imports() -> None:
    """Normalize IMPORTS for both module-level and from-import styles."""
    module_level = UserDefinedLayer.model_validate({"IMPORTS": ["torch"], "FLOW": []})
    layer = UserDefinedLayer.model_validate({"IMPORTS": {"torch.nn": ["Linear", "ReLU"]}, "FLOW": []})
    assert module_level.IMPORTS["torch"] == {None}
    assert layer.IMPORTS["torch.nn"] == {"Linear", "ReLU"}


def test_optimizer_behavior_tuple_variants_and_models() -> None:
    """Support 2/3/4 tuple forms for optimizer behavior."""
    opt = {"_obj_": [["_addr_", "torch.optim.AdamW"], ["_call_", {"lr": 1e-3}]]}
    clip = {"_obj_": [["_addr_", "timm.utils.clip_grad.dispatch_clip_grad"]]}
    two = OptimizerBehavior.model_validate([opt, ["model"]])
    three = OptimizerBehavior.model_validate(["adamw", opt, ["model.backbone"]])
    four = OptimizerBehavior.model_validate(["adamw_clip", opt, ["model.head"], clip])
    assert two.NAME is None
    assert three.NAME == "adamw"
    assert four.CLIP is not None
    assert two.models == {"model"}
    assert three.models == {"model"}


def test_backward_behavior_parses_extra_kwargs() -> None:
    """Parse tuple form and keep extra backward kwargs."""
    opt = [{"_obj_": [["_addr_", "torch.optim.AdamW"]]}, ["model"]]
    backward = BackwardBehavior.model_validate(["main", "ce_loss", [opt], {"retain_graph": True}])
    assert backward.NAME == "main"
    assert backward.LOSS == "ce_loss"
    assert backward.model_extra["retain_graph"] is True


def test_user_defined_backward_infers_losses_and_models() -> None:
    """Infer LOSSES and MODELS from BACKWARDS when omitted."""
    raw = {
        "BACKWARDS": [
            [
                "loss_a",
                [
                    [{"_obj_": [["_addr_", "torch.optim.AdamW"]]}, ["model"]],
                    [{"_obj_": [["_addr_", "torch.optim.SGD"]]}, ["aux_model.block"]],
                ],
            ]
        ]
    }
    cfg = UserDefinedBackward.model_validate(raw)
    assert cfg.LOSSES == ["loss_a"]
    assert set(cfg.MODELS) == {"model", "aux_model"}


def test_user_defined_backward_validates_unknown_losses() -> None:
    """Raise when LOSSES includes names not present in BACKWARDS."""
    raw = {
        "BACKWARDS": [["loss_a", [[{"_obj_": [["_addr_", "torch.optim.AdamW"]]}, ["model"]]]]],
        "LOSSES": ["loss_a", "loss_b"],
    }
    with pytest.raises(SpecError, match="Unknown losses found in LOSSES"):
        UserDefinedBackward.model_validate(raw)


def test_template_backward_separates_raw_and_others() -> None:
    """Expose target raw fields and non-target extras separately."""
    raw = {
        "BACKWARDS": [["loss_a", [[{"_obj_": [["_addr_", "torch.optim.AdamW"]]}, ["model"]]]]],
        "custom_option": {"enabled": True},
    }
    template = TemplateBackward.model_validate(raw)
    assert "BACKWARDS" in template.raw
    assert "custom_option" in template.others
    assert isinstance(template(), UserDefinedBackward)
