"""Test core builders functionality."""

import pytest
from structcast.core.exceptions import SpecError

from structcast_model.builders.core import (
    BaseBuilder,
    LayerBehavior,
    LayerIntermediate,
    Parameters,
    Serializable,
    TemplateLayer,
    UserDefinedLayer,
    UserLayer,
    WithExtra,
)


# Test Serializable
def test_serializable_basic() -> None:
    """Test basic Serializable functionality."""

    class TestConfig(Serializable):
        value: int

    config = TestConfig(value=42)
    assert config.value == 42


def test_serializable_frozen() -> None:
    """Test that Serializable instances are frozen."""

    class TestConfig(Serializable):
        value: int

    config = TestConfig(value=42)
    with pytest.raises(Exception):  # ValidationError from pydantic
        config.value = 100  # type: ignore[misc]


def test_serializable_extra_forbid() -> None:
    """Test that extra fields are forbidden in Serializable."""

    class TestConfig(Serializable):
        value: int

    with pytest.raises(Exception):  # ValidationError
        TestConfig(value=42, extra=100)  # type: ignore[call-arg]


# Test WithExtra
def test_with_extra_allows_extra_fields() -> None:
    """Test that WithExtra allows extra fields."""

    class TestConfig(WithExtra):
        value: int

    config = TestConfig(value=42, extra=100)  # type: ignore[call-arg]
    assert config.value == 42
    assert config.model_extra["extra"] == 100


def test_with_extra_model_extra_property() -> None:
    """Test model_extra property returns extra fields."""

    class TestConfig(WithExtra):
        value: int

    config = TestConfig(value=42, custom_field="test")  # type: ignore[call-arg]
    extra = config.model_extra
    assert extra["custom_field"] == "test"


# Test Parameters
def test_parameters_basic() -> None:
    """Test basic Parameters functionality."""
    params = Parameters()
    assert params.SHARED == {}
    assert params.DEFAULT == {}
    assert params.model_extra == {}


def test_parameters_with_shared_and_default() -> None:
    """Test Parameters with SHARED and DEFAULT fields."""
    params = Parameters(SHARED={"key1": "value1"}, DEFAULT={"key2": "value2"})
    assert params.SHARED == {"key1": "value1"}
    assert params.DEFAULT == {"key2": "value2"}


def test_parameters_with_groups() -> None:
    """Test Parameters with custom groups."""
    params = Parameters(
        SHARED={"shared_key": "shared_value"},
        DEFAULT={"default_key": "default_value"},
        group1={"g1_key": "g1_value"},  # type: ignore[call-arg]
        group2={"g2_key": "g2_value"},  # type: ignore[call-arg]
    )
    assert params.model_extra["group1"] == {"g1_key": "g1_value"}
    assert params.model_extra["group2"] == {"g2_key": "g2_value"}


def test_parameters_default_lowercase_merges_to_uppercase() -> None:
    """Test that lowercase 'default' merges into 'DEFAULT'."""
    params = Parameters(DEFAULT={"key1": "value1"}, default={"key2": "value2"})  # type: ignore[call-arg]
    assert params.DEFAULT == {"key1": "value1", "key2": "value2"}
    assert "default" not in params.model_extra


def test_parameters_template_aliases_forbidden() -> None:
    """Test that template aliases are forbidden in parameters."""
    with pytest.raises(SpecError, match="reserved for template aliases"):
        Parameters(_jinja_={"key": "value"})  # type: ignore[call-arg]


def test_parameters_group_must_be_dict() -> None:
    """Test that parameter groups must be dictionaries."""
    with pytest.raises(SpecError, match="must be a dictionary"):
        Parameters(group1="not a dict")  # type: ignore[call-arg]


def test_parameters_duplicate_keys_between_default_and_shared() -> None:
    """Test that duplicate keys between DEFAULT and SHARED raise error."""
    with pytest.raises(ValueError, match="Duplicate keys found"):
        Parameters(SHARED={"key": "shared"}, DEFAULT={"key": "default"})


def test_parameters_template_kwargs() -> None:
    """Test template_kwargs property."""
    params = Parameters(
        SHARED={"shared_key": "shared_value"},
        DEFAULT={"default_key": "default_value"},
        group1={"g1_key": "g1_value"},  # type: ignore[call-arg]
    )
    kwargs = params.template_kwargs
    assert kwargs["default"] == {"default_key": "default_value", "shared_key": "shared_value"}
    assert kwargs["group1"] == {"g1_key": "g1_value", "shared_key": "shared_value"}


def test_parameters_merge_with_dict() -> None:
    """Test merging parameters with a dictionary."""
    params1 = Parameters(DEFAULT={"key1": "value1"}, group1={"g1": "v1"})  # type: ignore[call-arg]
    merged = params1.merge({"default": {"key2": "value2"}, "group2": {"g2": "v2"}})

    result = merged.template_kwargs
    assert result["default"]["key1"] == "value1"
    assert result["default"]["key2"] == "value2"
    assert "group1" in result
    assert "group2" in result


def test_parameters_merge_with_parameters() -> None:
    """Test merging parameters with another Parameters instance."""
    params1 = Parameters(DEFAULT={"key1": "value1"})
    params2 = Parameters(DEFAULT={"key2": "value2"})
    merged = params1.merge(params2)

    result = merged.template_kwargs
    assert result["default"]["key1"] == "value1"
    assert result["default"]["key2"] == "value2"


# Test UserLayer
def test_user_layer_basic() -> None:
    """Test basic UserLayer functionality."""
    layer = UserLayer()
    assert layer.CFG is None
    assert layer.TYPE is None
    assert isinstance(layer.PARAM, Parameters)


def test_user_layer_with_values() -> None:
    """Test UserLayer with values."""
    layer = UserLayer(TYPE="conv", PARAM=Parameters(DEFAULT={"key": "value"}))
    assert layer.TYPE == "conv"
    assert layer.PARAM.DEFAULT == {"key": "value"}


# Test LayerBehavior
def test_layer_behavior_from_dict() -> None:
    """Test LayerBehavior from dictionary."""
    behavior = LayerBehavior(
        INPUTS=["input1"],
        OUTPUTS=["output1"],
        NAME="layer_name",
    )
    assert behavior.NAME == "layer_name"
    assert behavior.INPUTS is not None
    assert behavior.OUTPUTS is not None


def test_layer_behavior_from_tuple_3_elements() -> None:
    """Test LayerBehavior from 3-element tuple."""
    raw = (
        ["input1"],
        ["output1"],
        "layer_name",
    )
    behavior = LayerBehavior.model_validate(raw)
    assert behavior.INPUTS is not None
    assert behavior.OUTPUTS is not None
    assert behavior.NAME == "layer_name"
    assert behavior.LAYER is None


def test_layer_behavior_from_tuple_4_elements() -> None:
    """Test LayerBehavior from 4-element tuple with ObjectPattern."""
    raw = (
        ["input1"],  # Will be converted to FlexSpec
        ["output1"],  # Will be converted to FlexSpec
        "layer_name",
        {"_obj_": [["_addr_", "module.Layer"]]},  # ObjectPattern with AddressPattern
    )
    behavior = LayerBehavior.model_validate(raw)
    assert behavior.INPUTS is not None
    assert behavior.OUTPUTS is not None
    assert behavior.NAME == "layer_name"
    assert behavior.LAYER is not None


def test_layer_behavior_from_tuple_wrong_length() -> None:
    """Test LayerBehavior from tuple with wrong length."""
    with pytest.raises(SpecError, match="must have 3 or 4 elements"):
        LayerBehavior.model_validate((1, 2))


def test_layer_behavior_from_list() -> None:
    """Test LayerBehavior from list."""
    raw = [
        ["input1"],
        ["output1"],
        "layer_name",
    ]
    behavior = LayerBehavior.model_validate(raw)
    assert behavior.NAME == "layer_name"
    assert behavior.INPUTS is not None
    assert behavior.OUTPUTS is not None


# Test _resolve_inputs and _resolve_outputs through integration tests
def test_user_defined_layer_resolves_inputs_from_flow() -> None:
    """Test that inputs are resolved from FLOW correctly."""
    layer = UserDefinedLayer(
        FLOW=[
            LayerBehavior.model_validate(
                {
                    "INPUTS": ["input1"],  # List format - will resolve to input1
                    "OUTPUTS": ["output1"],  # List format - will resolve to output1
                    "NAME": "layer1",
                }
            )
        ]
    )
    assert "input1" in layer.INPUTS
    assert "output1" in layer.OUTPUTS


def test_user_defined_layer_resolves_multiple_inputs() -> None:
    """Test resolving multiple inputs from FLOW."""
    layer = UserDefinedLayer(
        FLOW=[
            LayerBehavior.model_validate(
                {
                    "INPUTS": [{"__src__": ["input1"]}, {"__src__": ["input2"]}],
                    "OUTPUTS": {"__src__": ["output1"]},
                    "NAME": "layer1",
                }
            )
        ]
    )
    assert "input1" in layer.INPUTS
    assert "input2" in layer.INPUTS


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
    layer = UserDefinedLayer(
        INPUTS=["input1", "input2"],
        OUTPUTS=["output1"],
        FLOW=[
            LayerBehavior.model_validate(
                {
                    "INPUTS": ["input1"],
                    "OUTPUTS": ["intermediate"],
                    "NAME": "layer1",
                }
            ),
            LayerBehavior.model_validate(
                {
                    "INPUTS": ["input2", "intermediate"],
                    "OUTPUTS": ["output1"],
                    "NAME": "layer2",
                }
            ),
        ],
    )
    assert layer.INPUTS == ["input1", "input2"]
    assert layer.OUTPUTS == ["output1"]


def test_user_defined_layer_validates_inout_format() -> None:
    """Test that INPUTS and OUTPUTS are validated to be lists."""
    # Check that string inputs are converted to lists via check_elements
    layer = UserDefinedLayer(
        INPUTS="input1",  # type: ignore[arg-type]
        OUTPUTS="output1",  # type: ignore[arg-type]
        FLOW=[
            LayerBehavior.model_validate(
                {
                    "INPUTS": ["input1"],
                    "OUTPUTS": ["output1"],
                    "NAME": "layer1",
                }
            )
        ],
    )
    assert layer.INPUTS == ["input1"]
    assert layer.OUTPUTS == ["output1"]


def test_user_defined_layer_flow_validation_auto_inputs() -> None:
    """Test that INPUTS are automatically derived from FLOW."""
    layer = UserDefinedLayer(
        FLOW=[
            LayerBehavior.model_validate(
                {
                    "INPUTS": ["input1"],
                    "OUTPUTS": ["output1"],
                    "NAME": "layer1",
                }
            )
        ]
    )
    assert "input1" in layer.INPUTS
    assert "output1" in layer.OUTPUTS


def test_user_defined_layer_unknown_inputs() -> None:
    """Test that unknown inputs raise error."""
    with pytest.raises(SpecError, match="Unknown inputs found"):
        UserDefinedLayer(
            INPUTS=["wrong_input"],
            FLOW=[
                LayerBehavior.model_validate(
                    {
                        "INPUTS": ["input1"],
                        "OUTPUTS": ["output1"],
                        "NAME": "layer1",
                    }
                )
            ],
        )


def test_user_defined_layer_missing_inputs() -> None:
    """Test that missing inputs raise error."""
    with pytest.raises(SpecError, match="Missing inputs found"):
        UserDefinedLayer(
            INPUTS=["input1"],
            FLOW=[
                LayerBehavior.model_validate(
                    {
                        "INPUTS": ["input1"],
                        "OUTPUTS": ["output1"],
                        "NAME": "layer1",
                    }
                ),
                LayerBehavior.model_validate(
                    {
                        "INPUTS": ["input2"],
                        "OUTPUTS": ["output2"],
                        "NAME": "layer2",
                    }
                ),
            ],
        )


def test_user_defined_layer_unknown_outputs() -> None:
    """Test that unknown outputs raise error."""
    with pytest.raises(SpecError, match="Unknown outputs found"):
        UserDefinedLayer(
            INPUTS=["input1"],
            OUTPUTS=["wrong_output"],
            FLOW=[
                LayerBehavior.model_validate(
                    {
                        "INPUTS": ["input1"],
                        "OUTPUTS": ["output1"],
                        "NAME": "layer1",
                    }
                )
            ],
        )


# Test TemplateLayer
def test_template_layer_basic() -> None:
    """Test basic TemplateLayer functionality."""
    template = TemplateLayer()
    assert isinstance(template.PARAMETERS, Parameters)
    assert template.raw == {}
    assert template.user_defined_layers == {}


def test_template_layer_with_parameters() -> None:
    """Test TemplateLayer with parameters."""
    template = TemplateLayer(PARAMETERS=Parameters(DEFAULT={"key": "value"}))
    assert template.PARAMETERS.DEFAULT == {"key": "value"}


def test_template_layer_raw_and_user_defined_layers() -> None:
    """Test TemplateLayer separates raw and user-defined layers."""
    template = TemplateLayer(
        INPUTS=["input1"],  # type: ignore[call-arg]
        OUTPUTS=["output1"],  # type: ignore[call-arg]
        custom_layer={"INPUTS": ["a"], "OUTPUTS": ["b"]},  # type: ignore[call-arg]
    )
    assert "INPUTS" in template.raw
    assert "OUTPUTS" in template.raw
    assert "custom_layer" in template.user_defined_layers


def test_template_layer_format() -> None:
    """Test TemplateLayer format method."""
    template = TemplateLayer(
        INPUTS=["input1"],  # type: ignore[call-arg]
        OUTPUTS=["output1"],  # type: ignore[call-arg]
        FLOW=[  # type: ignore[call-arg]
            LayerBehavior.model_validate(
                {
                    "INPUTS": ["input1"],
                    "OUTPUTS": ["output1"],
                    "NAME": "layer1",
                }
            )
        ],
    )
    result = template.format(Parameters())
    assert isinstance(result, UserDefinedLayer)
    assert result.INPUTS == ["input1"]
    assert result.OUTPUTS == ["output1"]


# Test LayerIntermediate
def test_layer_intermediate_basic() -> None:
    """Test basic LayerIntermediate functionality."""
    intermediate = LayerIntermediate(
        classname="TestLayer",
        inputs=["input1"],
        outputs=["output1"],
        layers={},
        flow=[],
        structured_output=False,
    )
    assert intermediate.classname == "TestLayer"
    assert intermediate.inputs == ["input1"]
    assert intermediate.outputs == ["output1"]
    assert intermediate.layers == {}
    assert intermediate.flow == []
    assert intermediate.structured_output is False


def test_layer_intermediate_with_layers() -> None:
    """Test LayerIntermediate with sub-layers."""
    sublayer = LayerIntermediate(
        classname="SubLayer",
        inputs=["a"],
        outputs=["b"],
        layers={},
        flow=[],
        structured_output=False,
    )
    intermediate = LayerIntermediate(
        classname="MainLayer",
        inputs=["input1"],
        outputs=["output1"],
        layers={"sub": sublayer},
        flow=[],
        structured_output=False,
    )
    assert "sub" in intermediate.layers
    assert intermediate.layers["sub"] == sublayer


# Test BaseBuilder
def test_base_builder_basic() -> None:
    """Test basic BaseBuilder functionality."""
    raw = {
        "INPUTS": "input1",
        "OUTPUTS": "output1",
        "FLOW": [],
    }
    builder = BaseBuilder(raw=raw)
    assert isinstance(builder.template, TemplateLayer)
    assert builder.training is False
    assert builder.current_path == ""
    assert builder.current_parts == []


def test_base_builder_with_training_flag() -> None:
    """Test BaseBuilder with training flag."""
    raw = {
        "INPUTS": ["input1"],
        "OUTPUTS": ["output1"],
        "FLOW": [],
    }
    builder = BaseBuilder(raw=raw, training=True)
    assert builder.training is True


def test_base_builder_user_defined_layers() -> None:
    """Test BaseBuilder user_defined_layers property."""
    raw = {
        "INPUTS": "input1",
        "OUTPUTS": "output1",
        "FLOW": [],
        "custom_layer": {"INPUTS": "a", "OUTPUTS": "b"},
    }
    builder = BaseBuilder(raw=raw)
    assert "custom_layer" in builder.user_defined_layers


def test_base_builder_with_predefined_layers() -> None:
    """Test BaseBuilder with predefined user-defined layers."""
    raw = {
        "INPUTS": "input1",
        "OUTPUTS": "output1",
        "FLOW": [],
    }
    predefined = {"predefined_layer": {"INPUTS": "x", "OUTPUTS": "y"}}
    builder = BaseBuilder(raw=raw, predefined_user_defined_layers=predefined)
    assert "predefined_layer" in builder.user_defined_layers


def test_base_builder_reference_property() -> None:
    """Test BaseBuilder reference property."""
    raw = {"INPUTS": [], "OUTPUTS": [], "FLOW": []}
    builder = BaseBuilder(raw=raw, current_path="path/to/config", current_parts=["part1", "part2"])
    assert builder.reference == "path/to/config:part1:part2"


def test_base_builder_call_simple() -> None:
    """Test BaseBuilder __call__ method with simple layer."""
    raw = {
        "FLOW": [],
    }
    builder = BaseBuilder(raw=raw)
    result = builder(Parameters(), "TestLayer")

    assert isinstance(result, LayerIntermediate)
    assert result.classname == "TestLayer"
    assert result.inputs == []
    assert result.outputs == []


def test_base_builder_circular_reference_detection() -> None:
    """Test that circular references cause errors (RecursionError or SpecError)."""
    raw = {
        "FLOW": [],
        "layer_a": {
            "FLOW": [
                {
                    "INPUTS": ["input1"],
                    "OUTPUTS": ["output1"],
                    "LAYER": {"TYPE": "layer_b"},
                }
            ]
        },
        "layer_b": {
            "FLOW": [
                {
                    "INPUTS": ["input1"],
                    "OUTPUTS": ["output1"],
                    "LAYER": {"TYPE": "layer_a"},
                }
            ]
        },
    }
    builder = BaseBuilder(raw=raw)
    # This should raise an error when trying to build layer_a
    # which references layer_b, which references layer_a
    # Either RecursionError (from pydantic) or SpecError (from circular detection)
    with pytest.raises((RecursionError, SpecError)):
        builder.get_user_defined_layer(["layer_a"], Parameters(), "LayerA")


def test_base_builder_layer_not_found() -> None:
    """Test that missing user-defined layer raises error."""
    raw = {
        "INPUTS": "input1",
        "OUTPUTS": "output1",
        "FLOW": [],
    }
    builder = BaseBuilder(raw=raw)
    with pytest.raises(SpecError, match='User-defined layer with key "nonexistent" not found'):
        builder.get_user_defined_layer(["nonexistent"], Parameters(), "TestLayer")


def test_base_builder_flow_with_name_reference() -> None:
    """Test BaseBuilder with FLOW referencing layer by NAME."""
    raw = {
        "INPUTS": "input1",
        "OUTPUTS": "output2",
        "FLOW": [
            {
                "INPUTS": ["input1"],
                "OUTPUTS": ["output1"],
                "LAYER": {"_obj_": [["_addr_", "module.TestLayer"]]},
                "NAME": "my_layer",
            },
            {
                "INPUTS": ["output1"],
                "OUTPUTS": ["output2"],
                "NAME": "my_layer",  # Reusing existing layer
            },
        ],
    }
    builder = BaseBuilder(raw=raw)
    result = builder(Parameters(), "TestLayer")

    assert len(result.layers) == 1
    assert "my_layer" in result.layers


def test_base_builder_flow_with_undefined_name() -> None:
    """Test that referencing undefined NAME raises error."""
    raw = {
        "INPUTS": "input1",
        "OUTPUTS": "output1",
        "FLOW": [
            {
                "INPUTS": ["input1"],
                "OUTPUTS": ["output1"],
                "NAME": "undefined_layer",
            },
        ],
    }
    builder = BaseBuilder(raw=raw)
    with pytest.raises(SpecError, match='Layer with name "undefined_layer" not defined'):
        builder(Parameters(), "TestLayer")


def test_base_builder_flow_duplicate_name() -> None:
    """Test that duplicate layer names raise error."""
    raw = {
        "INPUTS": "input1",
        "OUTPUTS": "output2",
        "FLOW": [
            {
                "INPUTS": ["input1"],
                "OUTPUTS": ["output1"],
                "LAYER": {"_obj_": [["_addr_", "module.TestLayer"]]},
                "NAME": "my_layer",
            },
            {
                "INPUTS": ["output1"],
                "OUTPUTS": ["output2"],
                "LAYER": {"_obj_": [["_addr_", "module.TestLayer"]]},
                "NAME": "my_layer",  # Duplicate name
            },
        ],
    }
    builder = BaseBuilder(raw=raw)
    with pytest.raises(SpecError, match='Duplicate layer name "my_layer"'):
        builder(Parameters(), "TestLayer")


def test_base_builder_flow_with_object_pattern() -> None:
    """Test BaseBuilder FLOW with ObjectPattern."""
    raw = {
        "INPUTS": "input1",
        "OUTPUTS": "output1",
        "FLOW": [
            {
                "INPUTS": ["input1"],
                "OUTPUTS": ["output1"],
                "LAYER": {"_obj_": [["_addr_", "torch.nn.Linear"]]},
            },
        ],
    }
    builder = BaseBuilder(raw=raw)
    result = builder(Parameters(), "TestLayer")

    assert len(result.layers) == 1
    assert "linear" in result.layers  # Auto-named based on class name


def test_base_builder_flow_without_address_pattern() -> None:
    """Test that LAYER without proper AddressPattern raises error."""
    raw = {
        "INPUTS": "input1",
        "OUTPUTS": "output1",
        "FLOW": [
            {
                "INPUTS": ["input1"],
                "OUTPUTS": ["output1"],
                "LAYER": {"_obj_": [["_call_", "invalid"]]},  # CallPattern instead of AddressPattern
            },
        ],
    }
    builder = BaseBuilder(raw=raw)
    with pytest.raises(SpecError, match="first pattern must be an AddressPattern"):
        builder(Parameters(), "TestLayer")


def test_base_builder_flow_with_user_layer_type() -> None:
    """Test BaseBuilder FLOW with UserLayer TYPE."""
    raw = {
        "INPUTS": "input1",
        "OUTPUTS": "output1",
        "FLOW": [
            {
                "INPUTS": ["input1"],
                "OUTPUTS": ["output1"],
                "LAYER": {"TYPE": "custom_layer"},
            },
        ],
        "custom_layer": {
            "FLOW": [
                {
                    "INPUTS": ["input1"],
                    "OUTPUTS": ["output1"],
                    "LAYER": {"_obj_": [["_addr_", "torch.nn.Identity"]]},
                }
            ],
        },
    }
    builder = BaseBuilder(raw=raw)
    result = builder(Parameters(), "TestLayer")

    assert len(result.layers) == 1
    assert "custom_layer" in result.layers


def test_base_builder_flow_layer_without_cfg_or_type() -> None:
    """Test that LAYER without CFG or TYPE raises error."""
    raw = {
        "INPUTS": "input1",
        "OUTPUTS": "output1",
        "FLOW": [
            {
                "INPUTS": ["input1"],
                "OUTPUTS": ["output1"],
                "LAYER": {},  # No CFG or TYPE
            },
        ],
    }
    builder = BaseBuilder(raw=raw)
    with pytest.raises(SpecError, match="LAYER must have either CFG or TYPE defined"):
        builder(Parameters(), "TestLayer")


def test_base_builder_injects_training_parameter() -> None:
    """Test that BaseBuilder injects training parameter."""
    raw = {
        "FLOW": [],
    }
    builder = BaseBuilder(raw=raw, training=True)
    # The training flag should be injected via Parameters
    result = builder(Parameters(), "TestLayer")
    assert isinstance(result, LayerIntermediate)
