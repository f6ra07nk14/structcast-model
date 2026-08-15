"""API-level tests for base builder utilities."""

from collections import defaultdict
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType
from typing import TypeAlias

import pytest
from structcast.core.exceptions import SpecError
from structcast.core.instantiator import ObjectPattern
from structcast.core.specifier import SpecIntermediate

from structcast_model.builders.base_builder import (
    BaseLearnerBuilder,
    BaseModelBuilder,
    LayerIntermediate,
)
from structcast_model.builders.schema import Parameters, UserLayer
from structcast_model.builders.torch_builder import TorchBuilder, TorchLayerIntermediate, TorchLearnerBuilder
from structcast_model.builders.utils import resolve_getter, resolve_object
from tests import ASSETS_DIR


def test_resolve_object_collects_import_and_class_name() -> None:
    """Resolve an object pattern and collect imports."""
    imports: defaultdict[str, set[str | None]] = defaultdict(set)
    raw = {"_obj_": [["_addr_", "torch.nn.Linear"], ["_call_", {"in_features": 8, "out_features": 4}]]}
    resolved, class_name = resolve_object(imports, ObjectPattern.model_validate(raw))
    assert resolved.startswith("Linear(")
    assert "'in_features': 8" in resolved
    assert "'out_features': 4" in resolved
    assert class_name == "Linear"
    assert imports["torch.nn"] == {"Linear"}


def test_resolve_object_with_bind_pattern() -> None:
    """Resolve an object pattern containing a bind operation."""
    imports: defaultdict[str, set[str | None]] = defaultdict(set)
    raw = {"_obj_": [["_addr_", "timm.utils.clip_grad.dispatch_clip_grad"], ["_bind_", {"value": 1.0, "mode": "norm"}]]}
    resolved, class_name = resolve_object(imports, ObjectPattern.model_validate(raw))
    assert "lambda" in resolved
    assert "'value': 1.0" in resolved
    assert "'mode': 'norm'" in resolved
    assert class_name == "dispatch_clip_grad"


def test_resolve_object_rejects_secondary_address_pattern() -> None:
    """Reject object patterns where non-first entries are address patterns."""
    imports: defaultdict[str, set[str | None]] = defaultdict(set)
    with pytest.raises(SpecError, match="Only the first pattern"):
        resolve_object(imports, ObjectPattern.model_validate({"_obj_": [["_addr_", "a.b"], ["_addr_", "c.d"]]}))


def test_resolve_object_supports_nested_object_and_attribute() -> None:
    """Resolve nested object first-pattern and attribute chaining."""
    imports: defaultdict[str, set[str | None]] = defaultdict(set)
    nested = {"_obj_": [{"_obj_": [["_addr_", "torch.nn.Identity"]]}]}
    resolved, class_name = resolve_object(imports, ObjectPattern.model_validate(nested))
    assert resolved == "Identity"
    assert class_name == "Identity"
    with_attr = {"_obj_": [["_addr_", "torch.nn"], {"_attr_": "Identity"}]}
    attr_resolved, _ = resolve_object(imports, ObjectPattern.model_validate(with_attr))
    assert attr_resolved == "nn.Identity"


def test_resolve_object_rejects_unsupported_literal_type() -> None:
    """Raise when bind arguments include unsupported Python literals."""
    imports: defaultdict[str, set[str | None]] = defaultdict(set)
    raw = {
        "_obj_": [
            ["_addr_", "torch.nn.Identity"],
            ["_call_", {"unexpected": {"not", "serializable"}}],
        ]
    }
    with pytest.raises(SpecError, match="Unsupported type for validation"):
        resolve_object(imports, ObjectPattern.model_validate(raw))


def test_resolve_object_supports_scalar_call_args() -> None:
    """Resolve call pattern with scalar argument payload."""
    imports: defaultdict[str, set[str | None]] = defaultdict(set)
    raw = {"_obj_": [["_addr_", "torch.manual_seed"], ["_call_", 42]]}
    resolved, class_name = resolve_object(imports, ObjectPattern.model_validate(raw))
    assert resolved == "manual_seed(42)"
    assert class_name == "manual_seed"


def test_resolve_getter_supports_source_eval_and_object() -> None:
    """Resolve source/eval/object specs into code strings."""
    imports: defaultdict[str, set[str | None]] = defaultdict(set)
    assert resolve_getter(imports, "model.layer") == "model['layer']"
    assert resolve_getter(imports, "eval: x + 1") == "x + 1"
    assert resolve_getter(imports, 7) == "7"
    assert resolve_getter(imports, {"_obj_": [["_addr_", "torch.nn.Identity"]]}) == "Identity"


def test_resolve_getter_supports_dict_tuple_and_constant() -> None:
    """Resolve nested dict/tuple values and constant specs."""
    imports: defaultdict[str, set[str | None]] = defaultdict(set)
    resolved = resolve_getter(imports, {"left": "x", "right": ("y", "constant:5")})
    assert resolved == "{'left': x, 'right': (y, '5')}"


def test_resolve_getter_rejects_unknown_identifier(monkeypatch: pytest.MonkeyPatch) -> None:
    """Raise when spec parser returns an unsupported identifier."""

    class _FakeSpec:
        identifier = "unsupported_identifier"
        value = "x"

    monkeypatch.setattr(SpecIntermediate, "convert_spec", lambda _raw: _FakeSpec())
    imports: defaultdict[str, set[str | None]] = defaultdict(set)
    with pytest.raises(SpecError, match="Unsupported spec identifier"):
        resolve_getter(imports, "x")


def test_base_model_builder_from_path_and_user_defined_entry() -> None:
    """Build from path and resolve a named user-defined layer."""
    builder = BaseModelBuilder.from_path(ASSETS_DIR / "cfg" / "torch" / "ConvNeXtV2.yaml")
    assert builder.current_path.endswith("cfg/torch/ConvNeXtV2.yaml")
    assert builder.from_references[builder.current_path] == ["__root__"]
    sublayer = builder(classname="BackboneOnly", user_defined_layer="Backbone")
    assert sublayer.classname == "BackboneOnly"
    assert sublayer.outputs == ["feat1", "feat2", "feat3", "feat4"]


def _import_module(module_path: Path) -> ModuleType:
    """Import the generated module from its file path."""
    spec = spec_from_file_location(module_path.stem, module_path)
    assert spec is not None
    assert spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_builder_emits_input_shapes_onto_the_generated_model(tmp_path: Path) -> None:
    """The declared INPUT_SHAPES survive as a literal, so a built model knows its own inputs without a --shape flag."""
    raw = {
        "INPUTS": ["x"],
        "OUTPUTS": ["y"],
        "INPUT_SHAPES": {"x": [3, 224, 224], "tokens": {"_SHAPE_": [512], "_DTYPE_": "int64"}},
        "FLOW": [["x", "y", {"_obj_": [["_addr_", "torch.nn.LazyLinear"], ["_call_", {"out_features": 2}]]}]],
    }
    expected = {"x": (3, 224, 224), "tokens": {"_SHAPE_": (512,), "_DTYPE_": "int64"}}
    built = TorchBuilder(raw=raw)(classname="TinyNet")
    assert built.input_shapes == expected
    module_path = tmp_path / "tiny_net.py"
    built(module_path)
    assert _import_module(module_path).TinyNet().input_shapes == expected


def test_builder_deduplicates_sublayers_sharing_input_shapes(tmp_path: Path) -> None:
    """Identical sub-layers still collapse into a single class now that input_shapes takes part in their hash."""
    raw = {
        "INPUTS": ["x"],
        "OUTPUTS": ["z"],
        "FLOW": [["x", "y", "first", {"TYPE": "Unit"}], ["y", "z", "second", {"TYPE": "Unit"}]],
        "Unit": {
            "INPUTS": ["a"],
            "OUTPUTS": ["b"],
            "FLOW": [["a", "b", {"_obj_": [["_addr_", "torch.nn.LazyLinear"], ["_call_", {"out_features": 2}]]}]],
        },
    }
    built = TorchBuilder(raw=raw)(classname="TinyNet")
    module_path = tmp_path / "tiny_net.py"
    built(module_path)
    assert module_path.read_text(encoding="utf-8").count("class Unit(") == 1
    model = _import_module(module_path).TinyNet()
    assert model.first is not model.second


def test_layer_intermediate_default_methods_raise_not_implemented() -> None:
    """Default abstract-style methods raise in base intermediate types."""
    inter = LayerIntermediate(
        classname="Unit",
        imports={},
        inputs=["x"],
        outputs=["y"],
        layers={},
        flow=[],
        inference_flow=[],
        structured_output=False,
    )
    assert inter._get_layer("proj") == "proj"
    with pytest.raises(NotImplementedError, match="_get_layer_script"):
        inter._get_layer_script("Unit", [])


def test_intermediate_call_writes_default_and_explicit_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Write generated scripts for default and string module paths."""
    unit = TorchLayerIntermediate(
        classname="TinyModel",
        imports={"json": {None}, "math": {"sqrt"}},
        inputs=["x"],
        outputs=["y"],
        layers={},
        flow=[("x", "y", None)],
        inference_flow=[],
        structured_output=False,
    )
    monkeypatch.chdir(tmp_path)
    unit()
    default_path = tmp_path / "tiny_model.py"
    assert default_path.exists()
    assert "import json" in default_path.read_text(encoding="utf-8")
    assert "from math import sqrt" in default_path.read_text(encoding="utf-8")
    explicit_path = tmp_path / "nested" / "module.py"
    unit(str(explicit_path))
    assert explicit_path.exists()


def test_base_model_builder_get_sublayer_cfg_with_type(tmp_path: Path) -> None:
    """Resolve sublayer from CFG+TYPE branch and derive subclass name."""
    cfg_path = tmp_path / "tmp_sub_cfg.yaml"
    cfg_path.write_text(
        "\n".join(
            [
                "INPUTS: [x]",
                "OUTPUTS: [y]",
                "FLOW:",
                "  - [x, y, {_obj_: [[_addr_, torch.nn.Identity]]}]",
                "Backbone:",
                "  INPUTS: [x]",
                "  OUTPUTS: [y]",
                "  FLOW:",
                "    - [x, y, {_obj_: [[_addr_, torch.nn.Identity]]}]",
            ]
        ),
        encoding="utf-8",
    )
    builder = BaseModelBuilder(raw={"FLOW": []})
    cfg_unit = UserLayer.model_validate({"CFG": cfg_path, "TYPE": "Backbone"})
    subclassname, _sub = builder._get_layer(Parameters(), cfg_unit)
    assert subclassname.endswith("Backbone")


def test_base_model_builder_flow_inputs_dict_and_partial_inout_error() -> None:
    """Cover dict INPUTS formatting and strict INPUTS/OUTPUTS pair validation."""
    raw_ok = {
        "INPUTS": ["x", "y"],
        "OUTPUTS": ["out"],
        "FLOW": [[{"left": "x", "right": "y"}, "out", {"_obj_": [["_addr_", "torch.add"]]}]],
    }
    built = BaseModelBuilder(raw=raw_ok)(classname="DictInput")
    assert built.flow[0][0] == "left=x, right=y"
    raw_bad = {"FLOW": [{"INPUTS": "x"}]}
    with pytest.raises(SpecError, match="Both INPUTS and OUTPUTS"):
        BaseModelBuilder(raw=raw_bad)()


def test_base_learner_builder_duplicate_name_and_optimizer_raise() -> None:
    """Reject duplicate learner names and optimizer names."""
    opt = {"_obj_": [["_addr_", "torch.optim.SGD"]]}
    duplicate_learner = {
        "LEARNERS": [
            {
                "NAME": "main",
                "LOSS": "loss_a",
                "TRAINABLE_LAYERS": ["model"],
                "OPTIMIZER": opt,
                "FLOW": [["x", "loss_a"]],
            },
            {
                "NAME": "main",
                "LOSS": "loss_b",
                "TRAINABLE_LAYERS": ["model"],
                "OPTIMIZER": opt,
                "FLOW": [["x", "loss_b"]],
            },
        ],
    }
    with pytest.raises(SpecError, match='Duplicate variable name "main" for optimizer'):
        TorchLearnerBuilder(raw=duplicate_learner)()


def test_base_learner_builder_mixed_precision_default_warns(caplog: pytest.LogCaptureFixture) -> None:
    """Base learner builder logs a warning for mixed precision and returns None."""

    class _NoMixedPrecisionBuilder(BaseLearnerBuilder):
        pass

    raw = {
        "LEARNERS": [
            {
                "LOSS": "loss",
                "TRAINABLE_LAYERS": ["model"],
                "OPTIMIZER": {"_obj_": [["_addr_", "torch.optim.SGD"]]},
                "FLOW": [["x", "loss"]],
            },
        ],
    }
    _NoMixedPrecisionBuilder(raw=raw)()
    assert "Mixed precision is not implemented" in caplog.text


# ---------------------------------------------------------------------------
# resolve_object — bind with list args (non-dict bind)
# ---------------------------------------------------------------------------


def test_resolve_object_with_list_bind_pattern() -> None:
    """Resolve an object pattern where bind arguments are a list (not a dict)."""
    imports: defaultdict[str, set[str | None]] = defaultdict(set)
    raw = {"_obj_": [["_addr_", "torch.nn.Identity"], ["_bind_", [1, 2, 3]]]}
    resolved, class_name = resolve_object(imports, ObjectPattern.model_validate(raw))
    assert "lambda" in resolved
    assert class_name == "Identity"
    # list bind places positional args before *args
    assert "1, 2, 3" in resolved


# ---------------------------------------------------------------------------
# resolve_object — _repr with dict and list values
# ---------------------------------------------------------------------------


def test_resolve_object_with_dict_literal_in_call() -> None:
    """Dict literal in call arguments is formatted correctly."""
    imports: defaultdict[str, set[str | None]] = defaultdict(set)
    raw = {
        "_obj_": [
            ["_addr_", "torch.nn.Identity"],
            ["_call_", {"config": {"nested_key": 42}}],
        ]
    }
    resolved, _ = resolve_object(imports, ObjectPattern.model_validate(raw))
    assert "'nested_key': 42" in resolved


def test_resolve_object_with_list_literal_in_call() -> None:
    """List literal in call arguments is formatted correctly."""
    imports: defaultdict[str, set[str | None]] = defaultdict(set)
    raw = {
        "_obj_": [
            ["_addr_", "torch.nn.Identity"],
            ["_call_", {"sizes": [1, 2, 3]}],
        ]
    }
    resolved, _ = resolve_object(imports, ObjectPattern.model_validate(raw))
    assert "[1, 2, 3]" in resolved


# ---------------------------------------------------------------------------
# TorchLearnerBuilder — full learner build with flow
# ---------------------------------------------------------------------------


def test_torch_learner_builder_simple_learner_generates_scripts() -> None:
    """Building a simple learner configuration produces scripts with training/inference steps."""
    raw = {
        "LEARNERS": [
            {
                "LOSS": "loss",
                "TRAINABLE_LAYERS": ["model"],
                "OPTIMIZER": {"_obj_": [["_addr_", "torch.optim.SGD"], ["_call_", {"lr": 0.01}]]},
                "FLOW": [["x", "loss", {"_obj_": [["_addr_", "torch.nn.Identity"]]}]],
            },
        ],
    }
    intermediate = TorchLearnerBuilder(raw=raw)()
    scripts = intermediate._get_scripts()
    combined = "\n".join(scripts)
    assert "class Learner" in combined
    assert "def training_step(self, x, **kwargs):" in combined
    assert "def inference_step(self, x, **kwargs):" in combined
    assert "optimizer" in combined.lower() or "sgd" in combined.lower()


def test_torch_learner_builder_with_mixed_precision() -> None:
    """Building with mixed precision generates GradScaler code."""
    raw = {
        "LEARNERS": [
            {
                "LOSS": "loss",
                "TRAINABLE_LAYERS": ["model"],
                "OPTIMIZER": {"_obj_": [["_addr_", "torch.optim.SGD"], ["_call_", {"lr": 0.01}]]},
                "FLOW": [["x", "loss", {"_obj_": [["_addr_", "torch.nn.Identity"]]}]],
            },
        ],
        "MIXED_PRECISION": True,
        "MIXED_PRECISION_TYPE": "float16",
    }
    intermediate = TorchLearnerBuilder(raw=raw)()
    scripts = intermediate._get_scripts()
    combined = "\n".join(scripts)
    assert "GradScaler" in combined
    assert "autocast" in combined


def test_torch_learner_builder_with_clip_gradient() -> None:
    """Building with CLIP generates a gradient clipping call."""
    raw = {
        "LEARNERS": [
            {
                "LOSS": "loss",
                "TRAINABLE_LAYERS": ["model"],
                "OPTIMIZER": {"_obj_": [["_addr_", "torch.optim.SGD"], ["_call_", {"lr": 0.01}]]},
                "CLIP": {
                    "_obj_": [
                        ["_addr_", "timm.utils.clip_grad.dispatch_clip_grad"],
                        ["_bind_", {"value": 1.0, "mode": "'norm'"}],
                    ]
                },
                "FLOW": [["x", "loss", {"_obj_": [["_addr_", "torch.nn.Identity"]]}]],
            },
        ],
    }
    intermediate = TorchLearnerBuilder(raw=raw)()
    scripts = intermediate._get_scripts()
    combined = "\n".join(scripts)
    assert "dispatch_clip_grad" in combined


def test_torch_learner_builder_with_accumulate_gradients() -> None:
    """Building with ACCUMULATE_GRADIENTS generates conditional update logic."""
    raw = {
        "LEARNERS": [
            {
                "LOSS": "loss",
                "TRAINABLE_LAYERS": ["model"],
                "OPTIMIZER": {"_obj_": [["_addr_", "torch.optim.SGD"], ["_call_", {"lr": 0.01}]]},
                "FLOW": [["x", "loss", {"_obj_": [["_addr_", "torch.nn.Identity"]]}]],
            },
        ],
        "ACCUMULATE_GRADIENTS": 4,
    }
    intermediate = TorchLearnerBuilder(raw=raw)()
    scripts = intermediate._get_scripts()
    combined = "\n".join(scripts)
    assert "need_update" in combined.lower() or "__need_update__" in combined


def test_torch_learner_builder_with_extra_kwargs() -> None:
    """EXTRA dict in a learner generates kwargs in the backward call."""
    raw = {
        "LEARNERS": [
            {
                "LOSS": "loss",
                "TRAINABLE_LAYERS": ["model"],
                "OPTIMIZER": {"_obj_": [["_addr_", "torch.optim.SGD"], ["_call_", {"lr": 0.01}]]},
                "EXTRA": {"retain_graph": True},
                "FLOW": [["x", "loss", {"_obj_": [["_addr_", "torch.nn.Identity"]]}]],
            },
        ],
    }
    intermediate = TorchLearnerBuilder(raw=raw)()
    scripts = intermediate._get_scripts()
    combined = "\n".join(scripts)
    assert "retain_graph" in combined


# ---------------------------------------------------------------------------
# _Intermediate._get_scripts raises NotImplementedError (line 190)
# ---------------------------------------------------------------------------


def test_intermediate_get_scripts_raises_not_implemented() -> None:
    """_Intermediate._get_scripts must be overridden; calling it bare raises."""
    # LazySelectedImporter only exposes __all__; get _Intermediate via its public subclass.
    _Intermediate: TypeAlias = LayerIntermediate.__bases__[0]

    class _BareIntermediate(_Intermediate):
        """Subclass that does NOT override _get_scripts."""

    inter = _BareIntermediate(classname="Test", imports={})
    with pytest.raises(NotImplementedError, match="_get_scripts"):
        inter._get_scripts()
