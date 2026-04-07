"""API-level tests for flax builder classes."""

import pytest

from structcast_model.builders.flax_builder import (
    FlaxBuilder,
    FlaxLayerIntermediate,
)
from tests import ASSETS_DIR


def test_flax_layer_intermediate_generates_call_method_without_inference_flow() -> None:
    """Generate __call__ without if/else when INFERENCE_FLOW is absent."""
    script = FlaxLayerIntermediate(
        classname="Unit",
        imports={},
        inputs=["x"],
        outputs=["y"],
        layers={},
        flow=[("x", "y", None)],
        inference_flow=[],
        structured_output=False,
    )._get_layer_script("Unit", [])
    assert "class Unit(flax.nnx.Module):" in script
    assert "def __call__(self, x, *, training: bool | None = None, **kwargs):" in script
    assert "if training:" not in script
    assert "return y" in script


def test_flax_layer_intermediate_generates_training_inference_branches() -> None:
    """Generate if/else branches when INFERENCE_FLOW is present."""
    script = FlaxLayerIntermediate(
        classname="Unit",
        imports={},
        inputs=["x"],
        outputs=["y"],
        layers={},
        flow=[("x", "y", None)],
        inference_flow=[("x", "y", None)],
        structured_output=False,
    )._get_layer_script("Unit", ["linear = flax.nnx.Linear(8, 4, rngs=rngs)"])
    assert "class Unit(flax.nnx.Module):" in script
    assert "if training:" in script
    assert "else:" in script
    assert "self.linear = flax.nnx.Linear(8, 4, rngs=rngs)" in script
    assert "return y" in script


def test_flax_layer_intermediate_init_accepts_rngs() -> None:
    """Generated __init__ signature must include rngs: flax.nnx.Rngs."""
    script = FlaxLayerIntermediate(
        classname="Unit",
        imports={},
        inputs=["x"],
        outputs=["y"],
        layers={},
        flow=[("x", "y", None)],
        inference_flow=[],
        structured_output=False,
    )._get_layer_script("Unit", [])
    assert "def __init__(self, *, rngs: flax.nnx.Rngs, training: bool = True):" in script


def test_flax_layer_intermediate_uses_inputs_outputs_attributes() -> None:
    """Generated __init__ stores inputs/outputs as plain lists."""
    script = FlaxLayerIntermediate(
        classname="Unit",
        imports={},
        inputs=["image"],
        outputs=["cls"],
        layers={},
        flow=[("image", "cls", None)],
        inference_flow=[],
        structured_output=False,
    )._get_layer_script("Unit", [])
    assert "self.inputs = ['image']" in script
    assert "self.outputs = ['cls']" in script


def test_flax_builder_builds_intermediate_and_scripts() -> None:
    """Build a minimal Flax nnx model and render Python script content."""
    raw = {
        "INPUTS": ["x"],
        "OUTPUTS": ["y"],
        "FLOW": [
            [
                "x",
                "y",
                {
                    "_obj_": [
                        ["_addr_", "flax.nnx.Linear"],
                        {"_call_": {"in_features": 8, "out_features": 4, "rngs": "eval: rngs"}},
                    ]
                },
            ]
        ],
    }
    built = FlaxBuilder(raw=raw)(classname="TinyNet")
    assert built.classname == "TinyNet"
    assert "flax" in built.collected_imports
    assert len(built.scripts) == 1
    assert "class TinyNet(flax.nnx.Module):" in built.scripts[0]


def test_flax_builder_structured_output_returns_dict() -> None:
    """Build a model with structured output and verify dict return."""
    raw = {
        "INPUTS": ["x"],
        "OUTPUTS": ["y"],
        "STRUCTURED_OUTPUT": True,
        "FLOW": [
            [
                "x",
                "y",
                {
                    "_obj_": [
                        ["_addr_", "flax.nnx.Linear"],
                        {"_call_": {"in_features": 8, "out_features": 4, "rngs": "eval: rngs"}},
                    ]
                },
            ]
        ],
    }
    built = FlaxBuilder(raw=raw)(classname="TinyNet", forced_structured_output=True)
    script = built.scripts[0]
    assert "return {'y': y}" in script


def test_flax_builder_eval_rngs_renders_in_init() -> None:
    """The eval: rngs keyword generates rngs=rngs in the __init__ body."""
    raw = {
        "INPUTS": ["x"],
        "OUTPUTS": ["y"],
        "FLOW": [
            [
                "x",
                "y",
                {
                    "_obj_": [
                        ["_addr_", "flax.nnx.Linear"],
                        {"_call_": {"in_features": 8, "out_features": 4, "rngs": "eval: rngs"}},
                    ]
                },
            ]
        ],
    }
    script = FlaxBuilder(raw=raw)(classname="TinyNet").scripts[0]
    assert "rngs=rngs" in script


def test_flax_builder_cfg_convnext_builds_expected_topology() -> None:
    """Build Flax ConvNeXt model from cfg and check key topology outputs."""
    parameters = {"DEFAULT": {"backbone": "tiny", "num_classes": 10}}
    builder = FlaxBuilder.from_path(ASSETS_DIR / "cfg/ConvNeXtV2Flax.yaml")
    built = builder(parameters=parameters, classname="ConvNeXtFlaxTiny")
    assert built.classname == "ConvNeXtFlaxTiny"
    assert built.structured_output is True
    assert built.outputs == ["cls"]
    assert "backbone" in built.layers
    assert "head" in built.layers
    assert len(built.scripts) > 0
    assert "class ConvNeXtFlaxTiny(flax.nnx.Module):" in built.scripts[-1]


@pytest.mark.parametrize("backbone", ["tiny"])
def test_flax_builder_cfg_convnext_sublayer_builds_backbone(backbone: str) -> None:
    """Build Backbone sublayer from Flax ConvNeXt cfg."""
    parameters = {"DEFAULT": {"backbone": backbone}}
    builder = FlaxBuilder.from_path(ASSETS_DIR / "cfg/ConvNeXtV2Flax.yaml")
    built = builder(parameters=parameters, classname="Backbone", user_defined_layer="Backbone")
    assert built.classname == "Backbone"
    assert built.structured_output is True
    assert "stem" in built.layers
    assert "downsample" in built.layers
    assert len(built.scripts) > 0
