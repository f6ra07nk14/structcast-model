"""Test tools."""

from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest
from structcast.utils.security import register_dir, unregister_dir

from structcast_model.utils.base import load_any, load_json, to_camel, to_pascal, to_snake, unique


@contextmanager
def temp_allowed_dir(path: Path) -> Generator[None, Any, None]:
    """Context manager for temporarily registering an allowed directory."""
    register_dir(path)
    try:
        yield
    finally:
        unregister_dir(path)


def test_load_json(tmp_path: Path) -> None:
    """Test load_json function."""
    with temp_allowed_dir(tmp_path):
        # Create a test JSON file
        json_file = tmp_path / "test.json"
        json_file.write_text('{"name": "test", "value": 42}')
        assert load_json(json_file) == {"name": "test", "value": 42}


def test_load_any_json(tmp_path: Path) -> None:
    """Test load_any function with JSON file."""
    with temp_allowed_dir(tmp_path):
        json_file = tmp_path / "test.json"
        json_file.write_text('{"name": "test", "value": 42}')
        assert load_any(json_file) == {"name": "test", "value": 42}


def test_load_any_yaml(tmp_path: Path) -> None:
    """Test load_any function with YAML file."""
    with temp_allowed_dir(tmp_path):
        yaml_file = tmp_path / "test.yaml"
        yaml_file.write_text("name: test\nvalue: 42\n")
        assert load_any(yaml_file) == {"name": "test", "value": 42}


def test_load_any_yml(tmp_path: Path) -> None:
    """Test load_any function with YML file."""
    with temp_allowed_dir(tmp_path):
        yml_file = tmp_path / "test.yml"
        yml_file.write_text("name: test\nvalue: 42\n")
        assert load_any(yml_file) == {"name": "test", "value": 42}


def test_load_any_jsonl(tmp_path: Path) -> None:
    """Test load_any function with JSONL file."""
    with temp_allowed_dir(tmp_path):
        jsonl_file = tmp_path / "test.jsonl"
        jsonl_file.write_text('{"key": "value1"}\n{"key": "value2"}\n{"key": "value3"}\n')
        assert load_any(jsonl_file) == [{"key": "value1"}, {"key": "value2"}, {"key": "value3"}]


def test_load_any_unsupported(tmp_path: Path) -> None:
    """Test load_any function with unsupported file type."""
    with temp_allowed_dir(tmp_path):
        txt_file = tmp_path / "test.txt"
        txt_file.write_text("some text")
        with pytest.raises(ValueError, match="Unsupported file type: .txt"):
            load_any(txt_file)


def test_unique() -> None:
    """Test unique function."""
    assert unique(["a", "b", "a", "c"]) == ["a", "b", "c"]
    assert unique([1, 2, 1, 3]) == [1, 2, 3]
    assert unique([1, 2, 3]) == [1, 2, 3]
    assert unique([]) == []
    assert unique(["a", "a", "a"]) == ["a"]


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("PascalCase", "pascal_case"),
        ("camelCase", "camel_case"),
        ("kebab-case", "kebab_case"),
        ("snake_case", "snake_case"),
        ("HTTPSRequest", "https_request"),
        ("ConvNeXtV2", "conv_ne_xt_v2"),
        ("already_snake", "already_snake"),
        ("ABC", "abc"),
        ("simpleword", "simpleword"),
    ],
)
def test_to_snake(value: str, expected: str) -> None:
    """Test to_snake function."""
    assert to_snake(value) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("snake_case", "SnakeCase"),
        ("hello_world", "HelloWorld"),
        ("camelCase", "CamelCase"),
        ("PascalCase", "PascalCase"),
        ("single", "Single"),
        ("already_pascal", "AlreadyPascal"),
    ],
)
def test_to_pascal(value: str, expected: str) -> None:
    """Test to_pascal function."""
    assert to_pascal(value) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("snake_case", "snakeCase"),
        ("hello_world", "helloWorld"),
        ("camelCase", "camelCase"),
        ("PascalCase", "pascalCase"),
        ("single", "single"),
        ("already_camel", "alreadyCamel"),
    ],
)
def test_to_camel(value: str, expected: str) -> None:
    """Test to_camel function."""
    assert to_camel(value) == expected
