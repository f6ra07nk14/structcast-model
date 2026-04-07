"""API-level tests for keras builder classes."""

import pytest

from structcast_model.builders.keras_builder import (
    KerasBuilder,
    KerasLayerIntermediate,
)
from tests import ASSETS_DIR


def test_keras_layer_intermediate_generates_call_method_without_inference_flow() -> None:
    """Generate call method without if/else when INFERENCE_FLOW is absent."""
    script = KerasLayerIntermediate(
        classname="Unit",
        imports={},
        inputs=["x"],
        outputs=["y"],
        layers={},
        flow=[("x", "y", None)],
        inference_flow=[],
        structured_output=False,
    )._get_layer_script("Unit", [])
    assert "class Unit(keras.layers.Layer):" in script
    assert "def call(self, inputs, training=None, mask=None):" in script
    assert "if training:" not in script
    assert "return y" in script


def test_keras_layer_intermediate_generates_training_inference_branches() -> None:
    """Generate if/else branches when INFERENCE_FLOW is present."""
    script = KerasLayerIntermediate(
        classname="Unit",
        imports={},
        inputs=["x"],
        outputs=["y"],
        layers={},
        flow=[("x", "y", None)],
        inference_flow=[("x", "y", None)],
        structured_output=False,
    )._get_layer_script("Unit", ["proj = keras.layers.Dense(units=4)"])
    assert "class Unit(keras.layers.Layer):" in script
    assert "if training:" in script
    assert "else:" in script
    assert "self.proj = keras.layers.Dense(units=4)" in script
    assert "return y" in script


def test_keras_layer_intermediate_propagates_training_to_sublayer_calls() -> None:
    """Sublayer calls include training=training keyword argument."""
    intermediate = KerasLayerIntermediate(
        classname="Unit",
        imports={},
        inputs=["x"],
        outputs=["y"],
        layers={"dense": "keras.layers.Dense(units=4)"},
        flow=[("x", "y", "dense")],
        inference_flow=[],
        structured_output=False,
    )
    script = intermediate._get_layer_script("Unit", ["dense = keras.layers.Dense(units=4)"])
    assert "self.dense(x, training=training)" in script


def test_keras_layer_intermediate_uses_input_output_names_attributes() -> None:
    """Use input_names/output_names to avoid clashing with Keras built-ins."""
    script = KerasLayerIntermediate(
        classname="Unit",
        imports={},
        inputs=["image"],
        outputs=["cls"],
        layers={},
        flow=[("image", "cls", None)],
        inference_flow=[],
        structured_output=False,
    )._get_layer_script("Unit", [])
    assert "self.input_names = ['image']" in script
    assert "self.output_names = ['cls']" in script


def test_keras_builder_builds_intermediate_and_scripts() -> None:
    """Build a minimal Keras model and render Python script content."""
    raw = {
        "INPUTS": ["x"],
        "OUTPUTS": ["y"],
        "FLOW": [["x", "y", {"_obj_": [["_addr_", "keras.layers.Dense"], ["_call_", {"units": 4}]]}]],
    }
    built = KerasBuilder(raw=raw)(classname="TinyNet")
    assert built.classname == "TinyNet"
    assert "keras" in built.collected_imports
    assert len(built.scripts) == 1
    assert "class TinyNet(keras.layers.Layer):" in built.scripts[0]


def test_keras_builder_structured_output_returns_dict() -> None:
    """Build a model with structured output and verify dict return."""
    raw = {
        "INPUTS": ["x"],
        "OUTPUTS": ["y"],
        "STRUCTURED_OUTPUT": True,
        "FLOW": [["x", "y", {"_obj_": [["_addr_", "keras.layers.Dense"], ["_call_", {"units": 4}]]}]],
    }
    built = KerasBuilder(raw=raw)(classname="TinyNet", forced_structured_output=True)
    script = built.scripts[0]
    assert "return {'y': y}" in script


def test_keras_builder_cfg_convnext_builds_expected_topology() -> None:
    """Build Keras ConvNeXt model from cfg and check key topology outputs."""
    parameters = {"DEFAULT": {"backbone": "tiny", "num_classes": 10}}
    builder = KerasBuilder.from_path(ASSETS_DIR / "cfg/ConvNeXtV2Keras.yaml")
    built = builder(parameters=parameters, classname="ConvNeXtKerasTiny")
    assert built.classname == "ConvNeXtKerasTiny"
    assert built.structured_output is True
    assert built.outputs == ["cls"]
    assert "backbone" in built.layers
    assert "head" in built.layers
    assert len(built.scripts) > 0
    assert "class ConvNeXtKerasTiny(keras.layers.Layer):" in built.scripts[-1]


@pytest.mark.parametrize("backbone", ["tiny"])
def test_keras_builder_cfg_convnext_sublayer_builds_backbone(backbone: str) -> None:
    """Build Backbone sublayer from Keras ConvNeXt cfg."""
    parameters = {"DEFAULT": {"backbone": backbone}}
    builder = KerasBuilder.from_path(ASSETS_DIR / "cfg/ConvNeXtV2Keras.yaml")
    built = builder(parameters=parameters, classname="Backbone", user_defined_layer="Backbone")
    assert built.classname == "Backbone"
    assert built.structured_output is True
    assert "stem" in built.layers
    assert "downsample" in built.layers
    assert len(built.scripts) > 0
