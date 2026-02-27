"""API-level tests for base builder utilities."""

from collections import defaultdict

import pytest
from structcast.core.exceptions import SpecError
from structcast.core.instantiator import ObjectPattern

from structcast_model.builders.base_builder import BaseModelBuilder, resolve_getter, resolve_object
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


def test_resolve_getter_supports_source_eval_and_object() -> None:
    """Resolve source/eval/object specs into code strings."""
    imports: defaultdict[str, set[str | None]] = defaultdict(set)
    assert resolve_getter(imports, "model.layer") == "model['layer']"
    assert resolve_getter(imports, "eval: x + 1") == "x + 1"
    assert resolve_getter(imports, 7) == "7"
    assert resolve_getter(imports, {"_obj_": [["_addr_", "torch.nn.Identity"]]}) == "Identity"


def test_base_model_builder_from_path_and_user_defined_entry() -> None:
    """Build from path and resolve a named user-defined layer."""
    builder = BaseModelBuilder.from_path(ASSETS_DIR / "cfg/ConvNeXtV2.yaml")
    assert builder.current_path.endswith("cfg/ConvNeXtV2.yaml")
    assert builder.from_references[builder.current_path] == ["__root__"]
    sublayer = builder(classname="BackboneOnly", user_defined_layer="Backbone")
    assert sublayer.classname == "BackboneOnly"
    assert sublayer.outputs == ["feat1", "feat2", "feat3", "feat4"]
