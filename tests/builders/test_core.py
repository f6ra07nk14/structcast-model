"""Test core builders functionality."""

from typing import Any

from pydantic import ValidationError
import pytest
from structcast.core.exceptions import SpecError

from structcast_model.builders.base_builder import BaseModelBuilder, LayerIntermediate
from structcast_model.builders.schema import (
    FlexSpec,
    LayerBehavior,
    Parameters,
    Serializable,
    TemplateLayer,
    UserDefinedLayer,
    UserLayer,
    WithExtra,
    resolve_inputs,
    resolve_outputs,
)
from structcast_model.utils.base import load_any


def test_serializable_frozen() -> None:
    """Test that Serializable instances are frozen."""

    class TestConfig(Serializable):
        value: int

    with pytest.raises(ValidationError, match="Instance is frozen"):
        TestConfig.model_validate({"value": 42}).value = 100


def test_serializable_extra_forbid() -> None:
    """Test that extra fields are forbidden in Serializable."""

    class TestConfig(Serializable):
        value: int

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        TestConfig.model_validate({"value": 42, "extra": 100})


# Test WithExtra
def test_with_extra_allows_extra_fields() -> None:
    """Test that WithExtra allows extra fields."""

    class TestConfig(WithExtra):
        value: int

    config = TestConfig.model_validate({"value": 42, "extra": 100})
    assert config.value == 42
    assert config.model_extra["extra"] == 100


def test_with_extra_model_extra_property() -> None:
    """Test model_extra property returns extra fields."""

    class TestConfig(WithExtra):
        value: int

    config = TestConfig.model_validate({"value": 42, "custom_field": "test"})
    assert config.model_extra["custom_field"] == "test"


# Test UserLayer
def test_user_layer_basic() -> None:
    """Test basic UserLayer functionality."""
    layer = UserLayer()
    assert layer.CFG is None
    assert layer.TYPE is None
    assert isinstance(layer.PARAM, Parameters)


def test_user_layer_with_values() -> None:
    """Test UserLayer with values."""
    layer = UserLayer.model_validate({"TYPE": "conv", "PARAM": {"DEFAULT": {"key": "value"}}})
    assert layer.TYPE == "conv"
    assert layer.PARAM.default == {"key": "value"}


# Test LayerBehavior
def test_layer_behavior_from_dict() -> None:
    """Test LayerBehavior from dictionary."""
    behavior = LayerBehavior.model_validate({"INPUTS": ["input1"], "OUTPUTS": ["output1"], "NAME": "layer_name"})
    assert behavior.NAME == "layer_name"
    assert behavior.INPUTS is not None
    assert behavior.OUTPUTS is not None


def test_layer_behavior_from_tuple_3_elements() -> None:
    """Test LayerBehavior from 3-element tuple."""
    behavior = LayerBehavior.model_validate((["input1"], ["output1"], "layer_name"))
    assert behavior.INPUTS is not None
    assert behavior.OUTPUTS is not None
    assert behavior.NAME == "layer_name"
    assert behavior.LAYER is None


def test_layer_behavior_from_tuple_4_elements() -> None:
    """Test LayerBehavior from 4-element tuple with ObjectPattern."""
    raw = (["input1"], ["output1"], "layer_name", {"_obj_": [["_addr_", "module.Layer"]]})
    behavior = LayerBehavior.model_validate(raw)
    assert behavior.INPUTS is not None
    assert behavior.OUTPUTS is not None
    assert behavior.NAME == "layer_name"
    assert behavior.LAYER is not None


def test_layer_behavior_from_tuple_wrong_length() -> None:
    """Test LayerBehavior from tuple with wrong length."""
    with pytest.raises(SpecError, match="must have 2, 3, or 4 elements"):
        LayerBehavior.model_validate(("a",))


def test_layer_behavior_from_list() -> None:
    """Test LayerBehavior from list."""
    behavior = LayerBehavior.model_validate([["input1"], ["output1"], "layer_name"])
    assert behavior.NAME == "layer_name"
    assert behavior.INPUTS is not None
    assert behavior.OUTPUTS is not None


# Test _resolve_inputs unit tests
def test_resolve_inputs_with_simple_list() -> None:
    """Test _resolve_inputs with a simple list input."""
    behavior = LayerBehavior.model_validate({"INPUTS": ["input1"], "OUTPUTS": ["output1"]})
    assert behavior.INPUTS is not None
    assert resolve_inputs(behavior.INPUTS) == ["input1"]


def test_resolve_inputs_with_multiple_inputs() -> None:
    """Test _resolve_inputs with multiple inputs in a list."""
    behavior = LayerBehavior.model_validate({"INPUTS": ["input1", "input2", "input3"], "OUTPUTS": ["output1"]})
    assert behavior.INPUTS is not None
    assert resolve_inputs(behavior.INPUTS) == ["input1", "input2", "input3"]


def test_resolve_inputs_with_dict_of_inputs() -> None:
    """Test _resolve_inputs with a dictionary containing inputs."""
    raw = {"INPUTS": {"key1": ["input1"], "key2": ["input2"]}, "OUTPUTS": ["output1"]}
    behavior = LayerBehavior.model_validate(raw)
    assert behavior.INPUTS is not None
    assert set(resolve_inputs(behavior.INPUTS)) == {"input1", "input2"}


def test_resolve_inputs_with_nested_list() -> None:
    """Test _resolve_inputs with nested lists of inputs."""
    behavior = LayerBehavior.model_validate({"INPUTS": [["input1", "input2"], ["input3"]], "OUTPUTS": ["output1"]})
    assert behavior.INPUTS is not None
    assert resolve_inputs(behavior.INPUTS) == ["input1", "input2", "input3"]


def test_resolve_inputs_with_complex_nested_structure() -> None:
    """Test _resolve_inputs with complex nested structures."""
    raw = {"INPUTS": {"layer1": ["a", "b"], "layer2": [["c"], ["d"]]}, "OUTPUTS": ["output1"]}
    behavior = LayerBehavior.model_validate(raw)
    assert behavior.INPUTS is not None
    assert set(resolve_inputs(behavior.INPUTS)) == {"a", "b", "c", "d"}


# Test _resolve_outputs unit tests
def test_resolve_outputs_with_simple_list() -> None:
    """Test _resolve_outputs with a simple list output."""
    behavior = LayerBehavior.model_validate({"INPUTS": ["input1"], "OUTPUTS": ["output1"]})
    assert behavior.OUTPUTS is not None
    assert resolve_outputs(behavior.OUTPUTS) == ["output1"]


def test_resolve_outputs_with_dict_at_top_level() -> None:
    """Test _resolve_outputs with dictionary at top level returns keys."""
    behavior = LayerBehavior.model_validate({"INPUTS": ["input1"], "OUTPUTS": {"output1": ["a"], "output2": ["b"]}})
    assert behavior.OUTPUTS is not None
    assert resolve_outputs(behavior.OUTPUTS) == ["output1", "output2"]


def test_resolve_outputs_with_multiple_outputs() -> None:
    """Test _resolve_outputs with multiple outputs in a list."""
    behavior = LayerBehavior.model_validate({"INPUTS": ["input1"], "OUTPUTS": ["output1", "output2", "output3"]})
    assert behavior.OUTPUTS is not None
    assert resolve_outputs(behavior.OUTPUTS) == ["output1", "output2", "output3"]


def test_resolve_outputs_with_nested_list() -> None:
    """Test _resolve_outputs with nested lists of outputs."""
    behavior = LayerBehavior.model_validate({"INPUTS": ["input1"], "OUTPUTS": [["output1"], ["output2", "output3"]]})
    assert behavior.OUTPUTS is not None
    assert resolve_outputs(behavior.OUTPUTS) == ["output1", "output2", "output3"]


def test_resolve_outputs_with_dict_in_list_raises_error() -> None:
    """Test _resolve_outputs with dictionary in list form raises SpecError."""
    behavior = LayerBehavior.model_validate({"INPUTS": ["input1"], "OUTPUTS": [{"key": ["output1"]}]})
    assert behavior.OUTPUTS is not None
    with pytest.raises(SpecError, match="Outputs cannot be a dictionary in list form"):
        resolve_outputs(behavior.OUTPUTS)


# Test _resolve_inputs error cases (raise SpecError)
def test_resolve_inputs_with_non_string_source_index_raises_error() -> None:
    """Test _resolve_inputs with non-string index in source identifier raises SpecError."""
    with pytest.raises(SpecError, match="First element of source identifier must be a string index"):
        resolve_inputs(FlexSpec.model_validate("123"))


def test_resolve_inputs_with_unsupported_identifier_raises_error() -> None:
    """Test _resolve_inputs with unsupported identifier raises SpecError."""
    with pytest.raises(SpecError, match="Unsupported spec identifier"):
        resolve_inputs(FlexSpec.model_validate("skip:"))


def test_resolve_inputs_with_unsupported_spec_type_raises_error() -> None:
    """Test _resolve_inputs with unsupported spec type raises SpecError."""

    class FakeFlexSpec:
        def __init__(self, spec: Any) -> None:
            self.spec = spec

        def model_dump(self) -> dict[str, Any]:
            return {"spec": self.spec}

    with pytest.raises(SpecError, match="Unsupported spec type"):
        resolve_inputs(FakeFlexSpec("invalid_string"))


# Test _resolve_outputs error cases (raise SpecError)
def test_resolve_outputs_with_non_source_identifier_raises_error() -> None:
    """Test _resolve_outputs with non-source identifier raises SpecError."""
    with pytest.raises(SpecError, match="Outputs must be consist of source identifier"):
        resolve_outputs(FlexSpec.model_validate([0]))


def test_resolve_outputs_with_multiple_indices_raises_error() -> None:
    """Test _resolve_outputs with source having multiple indices raises SpecError."""
    with pytest.raises(SpecError, match="Outputs must be a source identifier with a single string index"):
        resolve_outputs(FlexSpec.model_validate(["output1.output2"]))


def test_resolve_outputs_with_unsupported_type_raises_error() -> None:
    """Test _resolve_outputs with unsupported type raises SpecError."""

    class FakeFlexSpec:
        def __init__(self, spec: Any) -> None:
            self.spec = spec

        def model_dump(self) -> dict[str, Any]:
            return {"spec": self.spec}

    fake_spec = FakeFlexSpec("invalid_string")
    with pytest.raises(SpecError, match="Outputs must be a dictionary or consist of a source identifier"):
        resolve_outputs(fake_spec)


# Test _resolve_inputs and _resolve_outputs through integration tests
def test_user_defined_layer_resolves_inputs_from_flow() -> None:
    """Test that inputs are resolved from FLOW correctly."""
    behavior = LayerBehavior.model_validate({"INPUTS": ["input1"], "OUTPUTS": ["output1"], "NAME": "layer1"})
    layer = UserDefinedLayer.model_validate({"FLOW": [behavior]})
    assert "input1" in layer.INPUTS
    assert "output1" in layer.OUTPUTS


def test_user_defined_layer_resolves_multiple_inputs() -> None:
    """Test resolving multiple inputs from FLOW."""
    behavior_kw = {
        "INPUTS": [{"__src__": ["input1"]}, {"__src__": ["input2"]}],
        "OUTPUTS": {"__src__": ["output1"]},
        "NAME": "layer1",
    }
    layer = UserDefinedLayer.model_validate({"FLOW": [behavior_kw]})
    assert "input1" in layer.INPUTS
    assert "input2" in layer.INPUTS
    assert "__src__" in layer.OUTPUTS


# Test UserDefinedLayer
def test_user_defined_layer_basic() -> None:
    """Test basic UserDefinedLayer functionality."""
    layer = UserDefinedLayer()
    assert layer.INPUTS == []
    assert layer.OUTPUTS == []
    assert layer.FLOW == []
    assert layer.STRUCTURED_OUTPUT is False


def test_user_defined_layer_with_inputs_outputs() -> None:
    """Test UserDefinedLayer with explicit inputs and outputs matching FLOW."""
    raw = {
        "INPUTS": ["input1", "input2"],
        "OUTPUTS": ["output1"],
        "FLOW": [
            {"INPUTS": ["input1"], "OUTPUTS": ["intermediate"], "NAME": "layer1"},
            {"INPUTS": ["input2", "intermediate"], "OUTPUTS": ["output1"], "NAME": "layer2"},
        ],
    }
    layer = UserDefinedLayer.model_validate(raw)
    assert layer.INPUTS == ["input1", "input2"]
    assert layer.OUTPUTS == ["output1"]


def test_user_defined_layer_validates_inout_format() -> None:
    """Test that INPUTS and OUTPUTS are validated to be lists."""
    # Check that string inputs are converted to lists via check_elements
    raw = {
        "INPUTS": "input1",
        "OUTPUTS": "output1",
        "FLOW": [{"INPUTS": ["input1"], "OUTPUTS": ["output1"], "NAME": "layer1"}],
    }
    layer = UserDefinedLayer.model_validate(raw)
    assert layer.INPUTS == ["input1"]
    assert layer.OUTPUTS == ["output1"]


def test_user_defined_layer_flow_validation_auto_inputs() -> None:
    """Test that INPUTS are automatically derived from FLOW."""
    raw = {"FLOW": [{"INPUTS": ["input1"], "OUTPUTS": ["output1"], "NAME": "layer1"}]}
    layer = UserDefinedLayer.model_validate(raw)
    assert "input1" in layer.INPUTS
    assert "output1" in layer.OUTPUTS


def test_user_defined_layer_unknown_inputs() -> None:
    """Test that unknown inputs raise error."""
    raw = {
        "INPUTS": ["wrong_input"],
        "FLOW": [{"INPUTS": ["input1"], "OUTPUTS": ["output1"], "NAME": "layer1"}],
    }
    with pytest.raises(SpecError, match="Unknown inputs found"):
        UserDefinedLayer.model_validate(raw)


def test_user_defined_layer_missing_inputs() -> None:
    """Test that missing inputs raise error."""
    raw = {
        "INPUTS": ["input1"],
        "FLOW": [
            {"INPUTS": ["input1"], "OUTPUTS": ["output1"], "NAME": "layer1"},
            {"INPUTS": ["input2"], "OUTPUTS": ["output2"], "NAME": "layer2"},
        ],
    }
    with pytest.raises(SpecError, match="Missing inputs found"):
        UserDefinedLayer.model_validate(raw)


def test_user_defined_layer_unknown_outputs() -> None:
    """Test that unknown outputs raise error."""
    raw = {
        "INPUTS": ["input1"],
        "OUTPUTS": ["wrong_output"],
        "FLOW": [{"INPUTS": ["input1"], "OUTPUTS": ["output1"], "NAME": "layer1"}],
    }
    with pytest.raises(SpecError, match="Unknown outputs found"):
        UserDefinedLayer.model_validate(raw)


# Test TemplateLayer
def test_template_layer_basic() -> None:
    """Test basic TemplateLayer functionality."""
    template = TemplateLayer()
    assert isinstance(template.PARAMETERS, Parameters)
    assert template.raw == {}
    assert template.others == {}


def test_template_layer_with_parameters() -> None:
    """Test TemplateLayer with parameters."""
    template = TemplateLayer.model_validate({"PARAMETERS": {"DEFAULT": {"key": "value"}}})
    assert template.PARAMETERS.default == {"key": "value"}


def test_template_layer_raw_and_user_defined_layers() -> None:
    """Test TemplateLayer separates raw and user-defined layers."""
    raw = {"INPUTS": ["input1"], "OUTPUTS": ["output1"], "custom_layer": {"INPUTS": ["a"], "OUTPUTS": ["b"]}}
    template = TemplateLayer.model_validate(raw)
    assert "INPUTS" in template.raw
    assert "OUTPUTS" in template.raw
    assert "custom_layer" in template.others


def test_template_layer_format() -> None:
    """Test TemplateLayer format method."""
    raw = {"INPUTS": ["input1"], "OUTPUTS": ["output1"], "FLOW": [[["input1"], ["output1"], "layer1"]]}
    result = TemplateLayer.model_validate(raw)({})
    assert isinstance(result, UserDefinedLayer)
    assert result.INPUTS == ["input1"]
    assert result.OUTPUTS == ["output1"]


def test_base_builder_user_defined_layers() -> None:
    """Test BaseBuilder user_defined_layers property."""
    raw = {"INPUTS": "input1", "OUTPUTS": "output1", "FLOW": [], "custom_layer": {"INPUTS": "a", "OUTPUTS": "b"}}
    assert "custom_layer" in BaseModelBuilder(raw=raw).user_defined_layers


def test_base_builder_with_predefined_layers() -> None:
    """Test BaseBuilder with predefined user-defined layers."""
    raw = {"INPUTS": "input1", "OUTPUTS": "output1", "FLOW": []}
    predefined = {"predefined_layer": {"INPUTS": "x", "OUTPUTS": "y"}}
    builder = BaseModelBuilder(raw=raw, predefined_user_defined_layers=predefined)
    assert "predefined_layer" in builder.user_defined_layers


def test_base_builder_call_simple() -> None:
    """Test BaseBuilder __call__ method with simple layer."""
    result = BaseModelBuilder(raw={"FLOW": []})({}, "TestLayer")
    assert isinstance(result, LayerIntermediate)
    assert result.classname == "TestLayer"
    assert result.inputs == []
    assert result.outputs == []


def test_base_builder_circular_reference_detection() -> None:
    """Test that circular references cause errors (RecursionError or SpecError)."""
    raw = {
        "FLOW": [],
        "layer_a": {"FLOW": [[["input1"], ["output1"], {"TYPE": "layer_b"}]]},
        "layer_b": {"FLOW": [[["input1"], ["output1"], {"TYPE": "layer_b"}]]},
    }
    with pytest.raises(SpecError, match="Circular reference detected"):
        BaseModelBuilder(raw=raw).get_user_defined_layer(["layer_a"], {}, "LayerA")


def test_base_builder_layer_not_found() -> None:
    """Test that missing user-defined layer raises error."""
    raw = {"INPUTS": "input1", "OUTPUTS": "output1", "FLOW": []}
    with pytest.raises(SpecError, match='User-defined layer with key "nonexistent" not found'):
        BaseModelBuilder(raw=raw).get_user_defined_layer(["nonexistent"], {}, "TestLayer")


def test_base_builder_flow_with_name_reference() -> None:
    """Test BaseBuilder with FLOW referencing layer by NAME."""
    raw = {
        "INPUTS": "input1",
        "OUTPUTS": "output2",
        "FLOW": [
            [["input1"], ["output1"], "my_layer", {"_obj_": [["_addr_", "module.TestLayer"]]}],
            [["output1"], ["output2"], "my_layer"],  # Reference to the same layer by NAME
        ],
    }
    result = BaseModelBuilder(raw=raw)({}, "TestLayer")
    assert len(result.layers) == 1
    assert "my_layer" in result.layers


def test_base_builder_flow_with_undefined_name() -> None:
    """Test that referencing undefined NAME raises error."""
    raw = {"INPUTS": "input1", "OUTPUTS": "output1", "FLOW": [[["input1"], ["output1"], "undefined_layer"]]}
    with pytest.raises(SpecError, match='Layer with name "undefined_layer" not defined'):
        BaseModelBuilder(raw=raw)({}, "TestLayer")


def test_base_builder_flow_duplicate_name() -> None:
    """Test that duplicate layer names raise error."""
    raw = {
        "INPUTS": "input1",
        "OUTPUTS": "output2",
        "FLOW": [
            [["input1"], ["output1"], "my_layer", {"_obj_": [["_addr_", "module.TestLayer"]]}],
            [["output1"], ["output2"], "my_layer", {"_obj_": [["_addr_", "module.TestLayer"]]}],
        ],
    }
    with pytest.raises(SpecError, match='Duplicate layer name "my_layer"'):
        BaseModelBuilder(raw=raw)({}, "TestLayer")


def test_base_builder_flow_with_object_pattern() -> None:
    """Test BaseBuilder FLOW with ObjectPattern."""
    raw = {
        "INPUTS": "input1",
        "OUTPUTS": "output1",
        "FLOW": [[["input1"], ["output1"], {"_obj_": [["_addr_", "torch.nn.Linear"]]}]],
    }
    result = BaseModelBuilder(raw=raw)({}, "TestLayer")
    assert len(result.layers) == 1
    assert "linear" in result.layers  # Auto-named based on class name


def test_base_builder_flow_without_address_pattern() -> None:
    """Test that LAYER without proper AddressPattern raises error."""
    raw = {
        "INPUTS": "input1",
        "OUTPUTS": "output1",
        "FLOW": [[["input1"], ["output1"], {"_obj_": [["_call_", "invalid"]]}]],
    }
    with pytest.raises(SpecError, match="First pattern of an ObjectPattern must be an AddressPattern or ObjectPattern"):
        BaseModelBuilder(raw=raw)({}, "TestLayer")


def test_base_builder_flow_with_user_layer_type() -> None:
    """Test BaseBuilder FLOW with UserLayer TYPE."""
    raw = {
        "INPUTS": "input1",
        "OUTPUTS": "output1",
        "FLOW": [[["input1"], ["output1"], {"TYPE": "custom_layer"}]],
        "custom_layer": {"FLOW": [[["input1"], ["output1"], {"_obj_": [["_addr_", "torch.nn.Identity"]]}]]},
    }
    result = BaseModelBuilder(raw=raw)({}, "TestLayer")
    assert len(result.layers) == 1
    assert "custom_layer" in result.layers


def test_base_builder_flow_layer_without_cfg_or_type() -> None:
    """Test that LAYER without CFG or TYPE raises error."""
    raw = {"INPUTS": "input1", "OUTPUTS": "output1", "FLOW": [[["input1"], ["output1"], {}]]}
    with pytest.raises(SpecError, match="LAYER must have either CFG or TYPE specified"):
        BaseModelBuilder(raw=raw)({}, "TestLayer")


def test_base_builder_circular_reference_with_cfg() -> None:
    """Test that circular references via CFG cause errors."""
    with pytest.raises(SpecError, match="Circular reference detected"):
        BaseModelBuilder(raw=load_any("tests/fixtures/circular.yaml"))({}, "TestLayer")


def test_template_layer_jinja_yaml_basic() -> None:
    """Test TemplateLayer with _jinja_yaml_ template."""
    raw = {
        "PARAMETERS": {"DEFAULT": {"count": 2}},
        "_jinja_yaml_": """
INPUTS: [input1]
FLOW:
{% for i in range(count) %}
  - - {% if i == 0 %}[input1]{% else %}[out{{ i - 1 }}]{% endif +%}
    - [out{{ i }}]
    - {_obj_: [[_addr_, torch.nn.Identity]]}
{% endfor %}
""",
    }
    result = TemplateLayer.model_validate(raw)({})
    assert isinstance(result, UserDefinedLayer)
    assert result.INPUTS == ["input1"]
    assert result.OUTPUTS == ["out0", "out1"]
    assert len(result.FLOW) == 2


def test_template_layer_jinja_yaml_with_parameters() -> None:
    """Test TemplateLayer _jinja_yaml_ with custom parameters."""
    raw = {
        "PARAMETERS": {"DEFAULT": {"layer_count": 1}},
        "_jinja_yaml_": """
INPUTS: [x]
FLOW:
{% for i in range(layer_count) %}
  - - [x]
    - [layer_{{ i }}_out]
    - {_obj_: [[_addr_, torch.nn.ReLU]]}
{% endfor %}
""",
    }
    result = TemplateLayer.model_validate(raw)({"default": {"layer_count": 3}})
    assert isinstance(result, UserDefinedLayer)
    assert len(result.FLOW) == 3


def test_template_layer_jinja_yaml_with_groups() -> None:
    """Test TemplateLayer _jinja_yaml_ with parameter groups."""
    raw = {
        "PARAMETERS": {"DEFAULT": {"depth": 2}, "group1": {"depth": 4}},
        "_jinja_yaml_": """
INPUTS: [input]
OUTPUTS: [output]
FLOW:
{% for i in range(depth) %}
  - - {% if i == 0 %}[input]{% else %}[hidden{{ i - 1 }}]{% endif +%}
    - {% if i == depth - 1 %}[output]{% else %}[hidden{{ i }}]{% endif +%}
    - {_obj_: [[_addr_, torch.nn.Linear]]}
{% endfor %}
""",
    }
    template = TemplateLayer.model_validate(raw)
    result_default = template(Parameters(DEFAULT={"depth": 2}))
    assert len(result_default.FLOW) == 2
    result_custom = template(Parameters(DEFAULT={"depth": 5}))
    assert len(result_custom.FLOW) == 5


def test_template_layer_jinja_yaml_complex_structure() -> None:
    """Test TemplateLayer _jinja_yaml_ with complex nested structures."""
    raw = {
        "PARAMETERS": {"DEFAULT": {"units": [64, 32], "activation": "ReLU"}},
        "_jinja_yaml_": """
INPUTS: [input]
OUTPUTS: [output]
FLOW:
{% for i in range(units|length) %}
  - - {% if i == 0 %}[input]{% else %}[layer{{ i - 1 }}_out]{% endif +%}
    - [layer{{ i }}_out]
    - {_obj_: [[_addr_, torch.nn.Linear]]}
  {% if i < units|length - 1 %}
  - - [layer{{ i }}_out]
    - [act{{ i }}_out]
    - {_obj_: [[_addr_, torch.nn.{{ activation }}]]}
  {% endif %}
{% endfor %}
  - - [act{{ units|length - 2 }}_out]
    - [output]
    - {_obj_: [[_addr_, torch.nn.Identity]]}
""",
    }
    result = TemplateLayer.model_validate(raw)({})
    assert isinstance(result, UserDefinedLayer)
    # Should have 2 linear layers + 1 activation (between them) + 1 identity to connect to output
    assert len(result.FLOW) == 4
