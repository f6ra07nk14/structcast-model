"""Unit tests for structcast_model.commands.utils."""

import pydantic
import pytest

from structcast_model.commands.utils import dict_parser, reduce_dict, tensor_shape_parser

# ---------------------------------------------------------------------------
# reduce_dict
# ---------------------------------------------------------------------------


def test_reduce_dict_none() -> None:
    """None input returns an empty dict."""
    assert reduce_dict(None) == {}


def test_reduce_dict_empty_list() -> None:
    """Empty list returns an empty dict."""
    assert reduce_dict([]) == {}


def test_reduce_dict_single() -> None:
    """Single-dict list returns that dict."""
    assert reduce_dict([{"a": 1, "b": 2}]) == {"a": 1, "b": 2}


def test_reduce_dict_multiple() -> None:
    """Multiple dicts are merged left-to-right."""
    assert reduce_dict([{"a": 1}, {"b": 2}, {"c": 3}]) == {"a": 1, "b": 2, "c": 3}


def test_reduce_dict_later_value_wins() -> None:
    """When keys overlap, the later dict's value takes precedence."""
    assert reduce_dict([{"a": 1}, {"a": 99}]) == {"a": 99}


# ---------------------------------------------------------------------------
# dict_parser
# ---------------------------------------------------------------------------


def test_dict_parser_empty_string() -> None:
    """Empty string returns an empty dict."""
    assert dict_parser("") == {}


def test_dict_parser_valid_yaml() -> None:
    """A valid YAML mapping string is parsed into a dict."""
    assert dict_parser("key: value\ncount: 42") == {"key": "value", "count": 42}


def test_dict_parser_inline_yaml() -> None:
    """Inline YAML dict notation is parsed correctly."""
    assert dict_parser("{x: 1, y: 2}") == {"x": 1, "y": 2}


def test_dict_parser_nested() -> None:
    """Nested YAML mappings are preserved."""
    assert dict_parser("outer:\n  inner: 123") == {"outer": {"inner": 123}}


def test_dict_parser_non_mapping_raises() -> None:
    """A YAML scalar (non-mapping) value raises a ValidationError."""
    with pytest.raises(pydantic.ValidationError):
        dict_parser("- item1\n- item2")


# ---------------------------------------------------------------------------
# tensor_shape_parser
# ---------------------------------------------------------------------------


def test_tensor_shape_parser_empty_string() -> None:
    """Empty string returns an empty dict."""
    assert tensor_shape_parser("") == {}


def test_tensor_shape_parser_single_shape() -> None:
    """A single key mapping to a list of ints is parsed as a tuple."""
    assert tensor_shape_parser("image: [3, 224, 224]") == {"image": (3, 224, 224)}


def test_tensor_shape_parser_multiple_shapes() -> None:
    """Multiple keys are each parsed as integer tuples."""
    assert tensor_shape_parser("x: [1, 2]\ny: [3, 4, 5]") == {"x": (1, 2), "y": (3, 4, 5)}


def test_tensor_shape_parser_nested_dict() -> None:
    """A nested dict of shapes is recursively validated."""
    assert tensor_shape_parser("group:\n  a: [2, 2]\n  b: [4]") == {"group": {"a": (2, 2), "b": (4,)}}


def test_tensor_shape_parser_list_of_shapes() -> None:
    """A list of shapes is recursively validated element-wise."""
    assert tensor_shape_parser("inputs: [[1, 2], [3, 4]]") == {"inputs": [(1, 2), (3, 4)]}


def test_tensor_shape_parser_invalid_shape_raises() -> None:
    """A non-integer in a shape raises a ValueError."""
    with pytest.raises((ValueError, pydantic.ValidationError)):
        tensor_shape_parser('image: ["a", "b"]')


def test_tensor_shape_parser_scalar_raises() -> None:
    """A scalar value (not a sequence or mapping) raises a ValueError."""
    with pytest.raises((ValueError, pydantic.ValidationError)):
        tensor_shape_parser("image: not_a_shape")
