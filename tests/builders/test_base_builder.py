"""API-level tests for base builder utilities."""

from collections import defaultdict
from pathlib import Path

import pytest
from structcast.core.exceptions import SpecError
from structcast.core.instantiator import ObjectPattern
from structcast.core.specifier import SpecIntermediate
from structcast.utils.security import configure_security

from structcast_model.builders.base_builder import (
    BaseBackwardBuilder,
    BaseModelBuilder,
    LayerIntermediate,
    resolve_getter,
    resolve_object,
)
from structcast_model.builders.schema import Parameters, UserLayer
from structcast_model.builders.torch_builder import TorchBackwardBuilder, TorchLayerIntermediate
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
    builder = BaseModelBuilder.from_path(ASSETS_DIR / "cfg/ConvNeXtV2.yaml")
    assert builder.current_path.endswith("cfg/ConvNeXtV2.yaml")
    assert builder.from_references[builder.current_path] == ["__root__"]
    sublayer = builder(classname="BackboneOnly", user_defined_layer="Backbone")
    assert sublayer.classname == "BackboneOnly"
    assert sublayer.outputs == ["feat1", "feat2", "feat3", "feat4"]


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
    try:
        configure_security(working_dir_check=False)
        subclassname, _sub = builder._get_sublayer(Parameters(), cfg_unit)
        assert subclassname.endswith("Backbone")
    finally:
        configure_security()


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


def test_base_backward_builder_duplicate_name_and_optimizer_raise() -> None:
    """Reject duplicate backward names and optimizer names."""
    opt = {"_obj_": [["_addr_", "torch.optim.SGD"]]}
    duplicate_backward = {"BACKWARDS": [["main", "loss_a", [[opt, ["model"]]]], ["main", "loss_b", [[opt, ["model"]]]]]}
    with pytest.raises(SpecError, match='Duplicate backward name "main"'):
        TorchBackwardBuilder(raw=duplicate_backward)()
    duplicate_optimizer = {
        "BACKWARDS": [
            [
                "loss",
                [["same_opt", opt, ["model"]], ["same_opt", {"_obj_": [["_addr_", "torch.optim.AdamW"]]}, ["model"]]],
            ]
        ]
    }
    with pytest.raises(SpecError, match='Duplicate optimizer name "same_opt"'):
        TorchBackwardBuilder(raw=duplicate_optimizer)()


def test_base_backward_builder_mixed_precision_default_raises() -> None:
    """Base backward builder requires subclass mixed precision implementation."""

    class _NoMixedPrecisionBuilder(BaseBackwardBuilder):
        pass

    raw = {"BACKWARDS": [["loss", [[{"_obj_": [["_addr_", "torch.optim.SGD"]]}, ["model"]]]]]}
    with pytest.raises(NotImplementedError, match="_get_mixed_precision"):
        _NoMixedPrecisionBuilder(raw=raw)()
