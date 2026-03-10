"""API-level tests for builder schema models."""

from typing import Any

from pydantic import ValidationError
import pytest
from structcast.core.exceptions import SpecError
from structcast.utils.security import register_dir, unregister_dir

from structcast_model.builders.schema import (
    BackwardBehavior,
    LayerBehavior,
    OptimizerBehavior,
    Template,
    TemplateBackward,
    TemplateLayer,
    UserDefinedBackward,
    UserDefinedLayer,
    resolve_flow,
    resolve_inputs,
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


def test_layer_behavior_serialization_and_instance_passthrough() -> None:
    """Serialize LayerBehavior with NAME/LAYER and accept instance input."""
    behavior = LayerBehavior.model_validate(["x", "y", "unit", {"_obj_": [["_addr_", "torch.nn.Identity"]]}])
    dumped = behavior.model_dump()
    assert dumped[2] == "unit"
    assert len(dumped) == 4
    assert LayerBehavior.model_validate(behavior) is behavior


def test_resolve_inputs_supports_constant_specs() -> None:
    """Constant specs in INPUTS do not contribute dependency names."""
    unit = LayerBehavior.model_validate([{"a": "x", "b": "constant:10", "c": ["eval: 1 + 2", "y"]}, "out"])
    assert unit.INPUTS is not None
    assert resolve_inputs(unit.INPUTS) == ["x", "y"]


def test_validate_imports_returns_raw_for_invalid_non_iterable() -> None:
    """Invalid IMPORTS payload falls through and then fails type validation."""
    with pytest.raises(ValidationError, match="Input should be a valid dictionary"):
        UserDefinedLayer.model_validate({"IMPORTS": 123, "FLOW": []})


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


def test_optimizer_behavior_instance_passthrough_and_serialize() -> None:
    """Support instance passthrough and serializer branches for NAME/CLIP."""
    raw_opt = {"_obj_": [["_addr_", "torch.optim.AdamW"]]}
    raw_clip = {"_obj_": [["_addr_", "timm.utils.clip_grad.dispatch_clip_grad"]]}
    opt = OptimizerBehavior.model_validate(["adamw", raw_opt, ["model.block"], raw_clip])
    assert OptimizerBehavior.model_validate(opt) is opt
    dumped = opt.model_dump()
    assert dumped[0] == "adamw"
    assert dumped[2] == ["model.block"]
    assert len(dumped) == 4


def test_optimizer_behavior_three_tuple_without_name_and_invalid_length() -> None:
    """Handle 3-tuple (opt, layers, clip) and reject invalid tuple length."""
    raw_opt = {"_obj_": [["_addr_", "torch.optim.AdamW"]]}
    raw_clip = {"_obj_": [["_addr_", "timm.utils.clip_grad.dispatch_clip_grad"]]}
    parsed = OptimizerBehavior.model_validate([raw_opt, ["model"], raw_clip])
    assert parsed.NAME is None
    assert parsed.CLIP is not None
    with pytest.raises(SpecError, match="OptimizerBehavior must have 2, 3, or 4"):
        OptimizerBehavior.model_validate([raw_opt])


def test_backward_behavior_parses_extra_kwargs() -> None:
    """Parse tuple form and keep extra backward kwargs."""
    opt = [{"_obj_": [["_addr_", "torch.optim.AdamW"]]}, ["model"]]
    backward = BackwardBehavior.model_validate(["main", "ce_loss", [opt], {"retain_graph": True}])
    assert backward.NAME == "main"
    assert backward.LOSS == "ce_loss"
    assert backward.model_extra["retain_graph"] is True


def test_backward_behavior_instance_variants_and_serializer() -> None:
    """Cover tuple parsing variants and serializer for BackwardBehavior."""
    raw_opt = [{"_obj_": [["_addr_", "torch.optim.AdamW"]]}, ["model"]]
    named = BackwardBehavior.model_validate(["main", "ce_loss", [raw_opt], {"create_graph": False}])
    assert BackwardBehavior.model_validate(named) is named
    dumped = named.model_dump()
    assert dumped[0] == "main"
    assert dumped[1] == "ce_loss"
    assert "create_graph" in dumped
    with pytest.raises(ValidationError, match="Input should be a valid string"):
        BackwardBehavior.model_validate([123, [raw_opt], {"retain_graph": True}])
    with pytest.raises(SpecError, match="BackwardBehavior must have 2, 3, or 4"):
        BackwardBehavior.model_validate(["only_loss"])


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


def test_user_defined_layer_validates_inference_flow_mismatch_cases() -> None:
    """Raise for unknown/missing inputs and outputs in INFERENCE_FLOW."""
    base = {"INPUTS": ["x"], "OUTPUTS": ["y"], "FLOW": [["x", "y"]]}
    with pytest.raises(SpecError, match="Unknown inputs found in INFERENCE_FLOW"):
        UserDefinedLayer.model_validate({**base, "INFERENCE_FLOW": [["z", "y"]]})
    with pytest.raises(SpecError, match="Missing inputs found in INFERENCE_FLOW"):
        UserDefinedLayer.model_validate({**base, "INFERENCE_FLOW": [["x", "y"], ["aux", "y2"]]})
    with pytest.raises(SpecError, match="Unknown outputs found in INFERENCE_FLOW"):
        UserDefinedLayer.model_validate({**base, "INFERENCE_FLOW": [["x", "y2"]]})


def test_user_defined_backward_validates_missing_and_model_errors() -> None:
    """Raise for missing losses and unknown/missing models."""
    base = {
        "BACKWARDS": [
            ["loss_a", [[{"_obj_": [["_addr_", "torch.optim.AdamW"]]}, ["model"]]]],
            ["loss_b", [[{"_obj_": [["_addr_", "torch.optim.SGD"]]}, ["aux"]]]],
        ]
    }
    with pytest.raises(SpecError, match="Missing losses found in LOSSES"):
        UserDefinedBackward.model_validate({**base, "LOSSES": ["loss_a"]})
    with pytest.raises(SpecError, match="Unknown models found in MODELS"):
        UserDefinedBackward.model_validate({**base, "MODELS": ["model", "extra"]})
    with pytest.raises(SpecError, match="Missing models found in MODELS"):
        UserDefinedBackward.model_validate({**base, "MODELS": ["model"]})


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


def test_template_layer_call_with_merged_false_and_none_parameters() -> None:
    """Use template parameters directly when merged=False and parameters omitted."""
    raw = {
        "PARAMETERS": {"SHARED": {"width": 32}},
        "INPUTS": ["x"],
        "OUTPUTS": ["y"],
        "FLOW": [
            [
                "x",
                "y",
                {
                    "_obj_": [
                        ["_addr_", "torch.nn.Linear"],
                        ["_call_", {"in_features": 8, "out_features": "constant:{{ SHARED.width }}"}],
                    ]
                },
            ]
        ],
    }
    built = TemplateLayer.model_validate(raw)(None, merged=False)
    assert isinstance(built, UserDefinedLayer)
    assert built.FLOW[0].LAYER is not None


# ---------------------------------------------------------------------------
# _validate_name – invalid identifier
# ---------------------------------------------------------------------------


def test_validate_name_raises_for_invalid_identifier() -> None:
    """NAME with spaces or non-identifier chars raises SpecError."""
    with pytest.raises((SpecError, ValidationError)):
        LayerBehavior.model_validate({"INPUTS": "x", "OUTPUTS": "y", "NAME": "not valid!"})


def test_validate_name_raises_via_optimizer_behavior() -> None:
    """OptimizerBehavior NAME with invalid identifier raises SpecError."""
    with pytest.raises((SpecError, ValidationError)):
        OptimizerBehavior.model_validate(
            {"NAME": "123invalid", "OPTIMIZER": {"_obj_": [["_addr_", "torch.optim.SGD"]]}, "LAYERS": ["model"]}
        )


# ---------------------------------------------------------------------------
# Instance passthrough – OptimizerBehavior and BackwardBehavior
# ---------------------------------------------------------------------------


def test_optimizer_behavior_instance_passthrough() -> None:
    """Passing an existing OptimizerBehavior to model_validate returns it unchanged."""
    raw = {"OPTIMIZER": {"_obj_": [["_addr_", "torch.optim.SGD"]]}, "LAYERS": ["model"]}
    ob = OptimizerBehavior.model_validate(raw)
    assert OptimizerBehavior.model_validate(ob) is ob


def test_backward_behavior_instance_passthrough() -> None:
    """Passing an existing BackwardBehavior to model_validate returns it unchanged."""
    raw = {
        "LOSS": "ce_loss",
        "OPTIMIZERS": [{"OPTIMIZER": {"_obj_": [["_addr_", "torch.optim.SGD"]]}, "LAYERS": ["model"]}],
    }
    bb = BackwardBehavior.model_validate(raw)
    assert BackwardBehavior.model_validate(bb) is bb


# ---------------------------------------------------------------------------
# UserDefinedBackward – MIXED_PRECISION without type raises
# ---------------------------------------------------------------------------


def test_user_defined_backward_mixed_precision_without_type_raises() -> None:
    """MIXED_PRECISION=True without MIXED_PRECISION_TYPE raises SpecError."""
    raw = {
        "BACKWARDS": [
            {
                "LOSS": "ce_loss",
                "OPTIMIZERS": [{"OPTIMIZER": {"_obj_": [["_addr_", "torch.optim.SGD"]]}, "LAYERS": ["model"]}],
            }
        ],
        "MIXED_PRECISION": True,
    }
    with pytest.raises((SpecError, ValidationError)):
        UserDefinedBackward.model_validate(raw)


def test_user_defined_backward_mixed_precision_dict_without_type_raises() -> None:
    """MIXED_PRECISION dict without MIXED_PRECISION_TYPE raises SpecError."""
    raw = {
        "BACKWARDS": [
            {
                "LOSS": "ce_loss",
                "OPTIMIZERS": [{"OPTIMIZER": {"_obj_": [["_addr_", "torch.optim.SGD"]]}, "LAYERS": ["model"]}],
            }
        ],
        "MIXED_PRECISION": {"enabled": True},
    }
    with pytest.raises((SpecError, ValidationError)):
        UserDefinedBackward.model_validate(raw)


# ---------------------------------------------------------------------------
# Template.from_path
# ---------------------------------------------------------------------------


def test_template_from_path_loads_yaml_file(tmp_path: Any) -> None:
    """Template.from_path can load a simple YAML file."""
    register_dir(tmp_path)
    try:
        cfg = tmp_path / "simple.yaml"
        cfg.write_text("key: value\ncount: 42\n")
        tmpl = Template.from_path(cfg)
        assert isinstance(tmpl, Template)
    finally:
        unregister_dir(tmp_path)


def test_template_raw_and_others_for_with_extra_target(tmp_path: Any) -> None:
    """When target_type is WithExtra, _raw_and_others returns all fields in raw."""
    register_dir(tmp_path)
    try:
        cfg = tmp_path / "extra.yaml"
        cfg.write_text("foo: 1\nbar: 2\n")
        tmpl = Template.from_path(cfg)
        # Template.target_type defaults to WithExtra, so all extra fields land in raw
        assert "foo" in tmpl.raw
        assert tmpl.others == {}
    finally:
        unregister_dir(tmp_path)
