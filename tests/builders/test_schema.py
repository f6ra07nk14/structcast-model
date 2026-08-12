"""API-level tests for builder schema models."""

from typing import Any

from pydantic import TypeAdapter, ValidationError
import pytest
from structcast.core.exceptions import SpecError
from structcast.utils.base import register_dir, unregister_dir

from structcast_model.builders.schema import (
    LayerBehavior,
    LearnerBehavior,
    Template,
    TemplateLayer,
    TemplateLearner,
    TensorSpec,
    TensorSpecTree,
    UserDefinedLayer,
    UserDefinedLearner,
    resolve_flow,
    resolve_inputs,
)

TREE = TypeAdapter(TensorSpecTree)
"""Adapter validating a single INPUT_SHAPES entry, the way consumers of the tree do."""


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


def test_learner_behavior_extra_kwargs() -> None:
    """LearnerBehavior EXTRA dict stores additional learner configuration."""
    learner = LearnerBehavior.model_validate(
        {
            "NAME": "main",
            "LOSS": "ce_loss",
            "TRAINABLE_LAYERS": ["model"],
            "OPTIMIZER": {"_obj_": [["_addr_", "torch.optim.AdamW"]]},
            "EXTRA": {"retain_graph": True},
        }
    )
    assert learner.NAME == "main"
    assert learner.LOSS == "ce_loss"
    assert learner.EXTRA["retain_graph"] is True


def test_learner_behavior_instance_passthrough_and_clip() -> None:
    """Cover instance passthrough and CLIP field for LearnerBehavior."""
    raw = {
        "NAME": "main",
        "LOSS": "ce_loss",
        "TRAINABLE_LAYERS": ["model"],
        "OPTIMIZER": {"_obj_": [["_addr_", "torch.optim.AdamW"]]},
        "CLIP": {"_obj_": [["_addr_", "timm.utils.clip_grad.dispatch_clip_grad"]]},
    }
    named = LearnerBehavior.model_validate(raw)
    assert LearnerBehavior.model_validate(named) is named
    assert named.CLIP is not None
    with pytest.raises(ValidationError):
        LearnerBehavior.model_validate(
            {
                "LOSS": 123,
                "TRAINABLE_LAYERS": ["model"],
                "OPTIMIZER": {"_obj_": [["_addr_", "torch.optim.AdamW"]]},
            }
        )


def test_user_defined_learner_infers_losses_and_trainable_layers() -> None:
    """Infer LOSSES and TRAINABLE_LAYERS from LEARNERS when omitted."""
    raw = {
        "LEARNERS": [
            {
                "LOSS": "loss_a",
                "TRAINABLE_LAYERS": ["model", "aux_model"],
                "OPTIMIZER": {"_obj_": [["_addr_", "torch.optim.AdamW"]]},
                "FLOW": [["x", "loss_a"]],
            },
        ],
    }
    cfg = UserDefinedLearner.model_validate(raw)
    assert cfg.LOSSES == ["loss_a"]
    assert set(cfg.TRAINABLE_LAYERS) == {"model", "aux_model"}


def test_user_defined_learner_validates_unknown_losses() -> None:
    """Raise when LOSSES includes names not present in LEARNERS."""
    raw = {
        "LEARNERS": [
            {
                "LOSS": "loss_a",
                "TRAINABLE_LAYERS": ["model"],
                "OPTIMIZER": {"_obj_": [["_addr_", "torch.optim.AdamW"]]},
                "FLOW": [["x", "loss_a"]],
            },
        ],
        "LOSSES": ["loss_a", "loss_b"],
    }
    with pytest.raises(SpecError, match="Unknown losses found"):
        UserDefinedLearner.model_validate(raw)


def test_user_defined_layer_validates_inference_flow_mismatch_cases() -> None:
    """Raise for unknown/missing inputs and outputs in INFERENCE_FLOW."""
    base = {"INPUTS": ["x"], "OUTPUTS": ["y"], "FLOW": [["x", "y"]]}
    with pytest.raises(SpecError, match="Unknown inputs found in INFERENCE_FLOW"):
        UserDefinedLayer.model_validate({**base, "INFERENCE_FLOW": [["z", "y"]]})
    with pytest.raises(SpecError, match="Missing inputs found in INFERENCE_FLOW"):
        UserDefinedLayer.model_validate({**base, "INFERENCE_FLOW": [["x", "y"], ["aux", "y2"]]})
    with pytest.raises(SpecError, match="Unknown outputs found in INFERENCE_FLOW"):
        UserDefinedLayer.model_validate({**base, "INFERENCE_FLOW": [["x", "y2"]]})


def test_user_defined_learner_validates_missing_and_trainable_layer_errors() -> None:
    """Raise for missing losses and unknown/missing trainable layers."""
    base = {
        "LEARNERS": [
            {
                "LOSS": "loss_a",
                "TRAINABLE_LAYERS": ["model"],
                "OPTIMIZER": {"_obj_": [["_addr_", "torch.optim.AdamW"]]},
                "FLOW": [["x", "loss_a"]],
            },
            {
                "LOSS": "loss_b",
                "TRAINABLE_LAYERS": ["aux"],
                "OPTIMIZER": {"_obj_": [["_addr_", "torch.optim.SGD"]]},
                "FLOW": [["x", "loss_b"]],
            },
        ]
    }
    with pytest.raises(SpecError, match="Missing losses"):
        UserDefinedLearner.model_validate({**base, "LOSSES": ["loss_a"]})
    with pytest.raises(SpecError, match="Unknown trainable layers found"):
        UserDefinedLearner.model_validate({**base, "TRAINABLE_LAYERS": ["model", "extra"]})
    with pytest.raises(SpecError, match="Missing trainable layers found"):
        UserDefinedLearner.model_validate({**base, "TRAINABLE_LAYERS": ["model"]})


def test_template_learner_separates_raw_and_others() -> None:
    """Expose target raw fields and non-target extras separately."""
    raw = {
        "LEARNERS": [
            {
                "LOSS": "loss_a",
                "TRAINABLE_LAYERS": ["model"],
                "OPTIMIZER": {"_obj_": [["_addr_", "torch.optim.AdamW"]]},
                "FLOW": [["x", "loss_a"]],
            },
        ],
        "custom_option": {"enabled": True},
    }
    template = TemplateLearner.model_validate(raw)
    assert "LEARNERS" in template.raw
    assert "custom_option" in template.others
    assert isinstance(template(), UserDefinedLearner)


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


def test_validate_name_raises_via_learner_behavior() -> None:
    """LearnerBehavior NAME with invalid identifier raises SpecError."""
    with pytest.raises((SpecError, ValidationError)):
        LearnerBehavior.model_validate(
            {
                "NAME": "123invalid",
                "LOSS": "ce_loss",
                "TRAINABLE_LAYERS": ["model"],
                "OPTIMIZER": {"_obj_": [["_addr_", "torch.optim.SGD"]]},
            }
        )


# ---------------------------------------------------------------------------
# Instance passthrough – LearnerBehavior
# ---------------------------------------------------------------------------


def test_learner_behavior_instance_passthrough() -> None:
    """Passing an existing LearnerBehavior to model_validate returns it unchanged."""
    raw = {
        "LOSS": "ce_loss",
        "TRAINABLE_LAYERS": ["model"],
        "OPTIMIZER": {"_obj_": [["_addr_", "torch.optim.SGD"]]},
    }
    bb = LearnerBehavior.model_validate(raw)
    assert LearnerBehavior.model_validate(bb) is bb


# ---------------------------------------------------------------------------
# UserDefinedLearner – MIXED_PRECISION without type raises
# ---------------------------------------------------------------------------


def test_user_defined_learner_mixed_precision_type_without_mixed_precision_raises() -> None:
    """MIXED_PRECISION_TYPE set when MIXED_PRECISION is False raises SpecError."""
    raw = {
        "LEARNERS": [
            {
                "LOSS": "ce_loss",
                "TRAINABLE_LAYERS": ["model"],
                "OPTIMIZER": {"_obj_": [["_addr_", "torch.optim.SGD"]]},
                "FLOW": [["x", "ce_loss"]],
            }
        ],
        "MIXED_PRECISION": False,
        "MIXED_PRECISION_TYPE": "bfloat16",
    }
    with pytest.raises((SpecError, ValidationError)):
        UserDefinedLearner.model_validate(raw)


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


# ---------------------------------------------------------------------------
# LayerBehavior._validate_raw — tuple/list branches
# ---------------------------------------------------------------------------


def test_layer_behavior_from_tuple_with_layer_dict() -> None:
    """3-element tuple with dict third element treats it as LAYER."""
    layer_obj = {"_obj_": [["_addr_", "torch.nn.Identity"]]}
    behavior = LayerBehavior.model_validate(["x", "y", layer_obj])
    assert behavior.INPUTS is not None
    assert behavior.OUTPUTS is not None
    assert behavior.NAME is None
    assert behavior.LAYER is not None


def test_layer_behavior_from_tuple_with_4_elements() -> None:
    """4-element tuple sets INPUTS, OUTPUTS, NAME, and LAYER."""
    layer_obj = {"_obj_": [["_addr_", "torch.nn.Identity"]]}
    behavior = LayerBehavior.model_validate(["x", "y", "my_layer", layer_obj])
    assert behavior.NAME == "my_layer"
    assert behavior.LAYER is not None


def test_layer_behavior_from_tuple_with_wrong_length_raises() -> None:
    """Tuple with 1 or 5+ elements raises SpecError."""
    with pytest.raises((SpecError, ValidationError)):
        LayerBehavior.model_validate(["x"])
    with pytest.raises((SpecError, ValidationError)):
        LayerBehavior.model_validate(["a", "b", "c", "d", "e"])


# ---------------------------------------------------------------------------
# UserDefinedLearner — mixed precision None branch
# ---------------------------------------------------------------------------


def test_user_defined_learner_mixed_precision_none_raises() -> None:
    """MIXED_PRECISION=None with MIXED_PRECISION_TYPE set raises SpecError."""
    raw = {
        "LEARNERS": [
            {
                "LOSS": "ce_loss",
                "TRAINABLE_LAYERS": ["model"],
                "OPTIMIZER": {"_obj_": [["_addr_", "torch.optim.SGD"]]},
                "FLOW": [["x", "ce_loss"]],
            }
        ],
        "MIXED_PRECISION": None,
        "MIXED_PRECISION_TYPE": "float16",
    }
    with pytest.raises((SpecError, ValidationError)):
        UserDefinedLearner.model_validate(raw)


# ---------------------------------------------------------------------------
# UserDefinedLearner — loss not in outputs branch
# ---------------------------------------------------------------------------


def test_user_defined_learner_loss_not_in_flow_outputs_raises() -> None:
    """LOSS that does not appear as a flow output raises SpecError."""
    raw = {
        "LEARNERS": [
            {
                "LOSS": "missing_loss",
                "TRAINABLE_LAYERS": ["model"],
                "OPTIMIZER": {"_obj_": [["_addr_", "torch.optim.SGD"]]},
                "FLOW": [["x", "y"]],
            }
        ],
    }
    with pytest.raises(SpecError, match='Loss "missing_loss" must be in the outputs'):
        UserDefinedLearner.model_validate(raw)


# ---------------------------------------------------------------------------
# UserDefinedLearner — unknown/missing inputs and outputs
# ---------------------------------------------------------------------------


def test_user_defined_learner_unknown_inputs_raises() -> None:
    """INPUTS containing names not in the flow raises SpecError."""
    raw = {
        "LEARNERS": [
            {
                "LOSS": "loss",
                "TRAINABLE_LAYERS": ["model"],
                "OPTIMIZER": {"_obj_": [["_addr_", "torch.optim.SGD"]]},
                "FLOW": [["x", "loss"]],
            }
        ],
        "INPUTS": ["x", "extra_input"],
    }
    with pytest.raises(SpecError, match="Unknown inputs found"):
        UserDefinedLearner.model_validate(raw)


def test_user_defined_learner_missing_inputs_raises() -> None:
    """Omitting an input that the flow needs raises SpecError."""
    raw = {
        "LEARNERS": [
            {
                "LOSS": "loss1",
                "TRAINABLE_LAYERS": ["model"],
                "OPTIMIZER": {"_obj_": [["_addr_", "torch.optim.SGD"]]},
                "FLOW": [["x", "loss1"]],
            },
            {
                "LOSS": "loss2",
                "TRAINABLE_LAYERS": ["aux"],
                "OPTIMIZER": {"_obj_": [["_addr_", "torch.optim.SGD"]]},
                "FLOW": [["y", "loss2"]],
            },
        ],
        "INPUTS": ["x"],
    }
    with pytest.raises(SpecError, match="Missing inputs found"):
        UserDefinedLearner.model_validate(raw)


def test_user_defined_learner_unknown_outputs_raises() -> None:
    """OUTPUTS with extra names not generated by flow raises SpecError."""
    raw = {
        "LEARNERS": [
            {
                "LOSS": "loss",
                "TRAINABLE_LAYERS": ["model"],
                "OPTIMIZER": {"_obj_": [["_addr_", "torch.optim.SGD"]]},
                "FLOW": [["x", "loss"]],
            }
        ],
        "OUTPUTS": ["loss", "ghost"],
    }
    with pytest.raises(SpecError, match="Unknown outputs found"):
        UserDefinedLearner.model_validate(raw)


# ---------------------------------------------------------------------------
# UserDefinedLearner — inference flow mismatches
# ---------------------------------------------------------------------------


def test_user_defined_learner_unknown_inputs_in_inference_flow_raises() -> None:
    """Unknown inputs in inference flow raises SpecError."""
    raw = {
        "LEARNERS": [
            {
                "LOSS": "loss",
                "TRAINABLE_LAYERS": ["model"],
                "OPTIMIZER": {"_obj_": [["_addr_", "torch.optim.SGD"]]},
                "FLOW": [["x", "loss"]],
                "INFERENCE_FLOW": [["z", "loss"]],
            }
        ],
    }
    with pytest.raises(SpecError, match="Unknown inputs found in inference flow"):
        UserDefinedLearner.model_validate(raw)


def test_user_defined_learner_missing_inputs_in_inference_flow_raises() -> None:
    """Missing inputs in inference flow raises SpecError."""
    raw = {
        "LEARNERS": [
            {
                "LOSS": "loss",
                "TRAINABLE_LAYERS": ["model"],
                "OPTIMIZER": {"_obj_": [["_addr_", "torch.optim.SGD"]]},
                "FLOW": [["x", "loss"]],
                "INFERENCE_FLOW": [["x", "loss"], ["extra", "out2"]],
            }
        ],
    }
    with pytest.raises(SpecError, match="Missing inputs found in inference flow"):
        UserDefinedLearner.model_validate(raw)


def test_user_defined_learner_unknown_outputs_in_inference_flow_raises() -> None:
    """Unknown outputs in inference flow raises SpecError."""
    raw = {
        "LEARNERS": [
            {
                "LOSS": "loss",
                "TRAINABLE_LAYERS": ["model"],
                "OPTIMIZER": {"_obj_": [["_addr_", "torch.optim.SGD"]]},
                "FLOW": [["x", "loss"]],
                "INFERENCE_FLOW": [["x", "other_output"]],
            }
        ],
        "OUTPUTS": ["loss"],
    }
    with pytest.raises(SpecError, match="Unknown outputs found in inference flow"):
        UserDefinedLearner.model_validate(raw)


# ---------------------------------------------------------------------------
# TensorSpec / TensorSpecTree
# ---------------------------------------------------------------------------


def test_tensor_spec_compact_form_fills_defaults() -> None:
    """A bare shape is the compact form of a TensorSpec and gets the default dtype."""
    spec = TREE.validate_python([3, 224, 224])
    assert isinstance(spec, TensorSpec)
    assert spec.SHAPE == (3, 224, 224)
    assert spec.DTYPE == "bfloat16"
    assert spec.INIT is None


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ([3, 224, 224], (3, 224, 224)),
        ([], ()),
        ({"_SHAPE_": [512], "_DTYPE_": "int64"}, {"_SHAPE_": (512,), "_DTYPE_": "int64"}),
        ({"_SHAPE_": [8], "_INIT_": "torch.zeros"}, {"_SHAPE_": (8,), "_INIT_": "torch.zeros"}),
        ({"_SHAPE_": [8], "_DTYPE_": "bfloat16"}, (8,)),
        ({"a": [10], "b": [5]}, {"a": (10,), "b": (5,)}),
        ([[3, 224], [5]], [(3, 224), (5,)]),
    ],
)
def test_tensor_spec_tree_round_trips(raw: Any, expected: Any) -> None:
    """Validation is reversible: dumping restores the compact form whenever nothing but the shape is set."""
    assert TREE.dump_python(TREE.validate_python(raw)) == expected


def test_tensor_spec_tree_keeps_nested_containers() -> None:
    """Nesting is preserved, dictionaries and lists stay themselves and only the leaves become TensorSpec."""
    nested = TREE.validate_python({"a": [10], "b": [[3, 224], [5]]})
    assert isinstance(nested, dict)
    assert isinstance(nested["a"], TensorSpec)
    branch = nested["b"]
    assert isinstance(branch, list)
    assert [leaf.SHAPE for leaf in branch if isinstance(leaf, TensorSpec)] == [(3, 224), (5,)]


@pytest.mark.parametrize(
    ("raw", "match"),
    [
        (10, "Input should be a valid dictionary or instance of TensorSpec"),
        ({"_SHAPE_": [1], "_DTYPE_": "float64"}, "Input should be 'bfloat16', 'float16', 'float32'"),
    ],
)
def test_tensor_spec_tree_rejects_invalid_input(raw: Any, match: str) -> None:
    """Scalars and unsupported dtypes fail as ValidationError, so the union can report every branch."""
    with pytest.raises(ValidationError, match=match):
        TREE.validate_python(raw)


@pytest.mark.parametrize(
    "raw",
    [
        {"_SHAPE_": [4], "extra": [2]},
        {"_DTYPE_": [4]},
        {"nested": {"_INIT_": [2]}},
    ],
)
def test_tensor_spec_tree_rejects_marker_keys_in_nested_dict(raw: Any) -> None:
    """A malformed explicit form must error instead of being reinterpreted as nested inputs named by markers."""
    with pytest.raises(ValidationError, match="marker keys"):
        TREE.validate_python(raw)


def test_user_defined_layer_input_shapes_defaults_to_empty() -> None:
    """INPUT_SHAPES is optional."""
    assert UserDefinedLayer.model_validate({"FLOW": [["x", "y", "layer"]]}).INPUT_SHAPES == {}


def test_user_defined_layer_input_shapes_round_trip() -> None:
    """INPUT_SHAPES validates its entries and dumps them back to plain nested data."""
    layer = UserDefinedLayer.model_validate(
        {
            "FLOW": [["x", "y", "layer"]],
            "INPUT_SHAPES": {"x": [3, 224, 224], "aux": {"m": {"_SHAPE_": [4], "_DTYPE_": "int32"}}},
        }
    )
    assert isinstance(layer.INPUT_SHAPES["x"], TensorSpec)
    assert layer.model_dump()["INPUT_SHAPES"] == {
        "x": (3, 224, 224),
        "aux": {"m": {"_SHAPE_": (4,), "_DTYPE_": "int32"}},
    }
