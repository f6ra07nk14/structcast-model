"""Test tools."""

from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
import time
from typing import Any

import pytest
from structcast.utils.security import register_dir, unregister_dir

from structcast_model.utils.base import Cache, load_any, load_json, unique


@contextmanager
def temp_allowed_dir(path: Path) -> Generator[None, Any, None]:
    """Context manager for temporarily registering an allowed directory."""
    register_dir(path)
    try:
        yield
    finally:
        unregister_dir(path)


def test_cache_func_maxsize() -> None:
    """Test the Cache decorator with maxsize limit."""
    call_count = 0

    @Cache(maxsize=2)
    def cached_function(x: int) -> int:
        nonlocal call_count
        call_count += 1
        return x

    assert cached_function(1) == 1
    assert call_count == 1
    assert cached_function(1) == 1
    assert call_count == 1  # Cached result
    assert cached_function(2) == 2
    assert call_count == 2
    assert cached_function(3) == 3
    assert call_count == 3
    assert cached_function(1) == 1
    assert call_count == 4  # Cache miss, as (1) was evicted
    assert cached_function(2) == 2
    assert call_count == 5  # Cache miss, as (2) was evicted


def test_cache_func_dict() -> None:
    """Test the Cache decorator with dictionary arguments."""
    call_count = 0

    @Cache()
    def cached_function(data: dict) -> dict:
        nonlocal call_count
        call_count += 1
        return data

    assert cached_function({"a": 1}) == {"a": 1}
    assert call_count == 1
    assert cached_function({"a": 1}) == {"a": 1}
    assert call_count == 1  # Cached result
    assert cached_function({"a": 2}) == {"a": 2}
    assert call_count == 2
    assert cached_function({"a": 1}) == {"a": 1}
    assert call_count == 2  # Cached result
    assert cached_function({"b": 1}) == {"b": 1}
    assert call_count == 3
    assert cached_function({"a": 2}) == {"a": 2}
    assert call_count == 3  # Cached result
    assert cached_function({"b": 1}) == {"b": 1}
    assert call_count == 3  # Cached result
    assert cached_function({"c": 3, "d": 4}) == {"c": 3, "d": 4}
    assert call_count == 4
    assert cached_function({"a": 1}) == {"a": 1}
    assert call_count == 4  # Cached result


def test_cache_func_timeout() -> None:
    """Test the Cache decorator with timeout."""
    call_count = 0

    @Cache(timeout=0.1)
    def cached_function(x: int) -> int:
        nonlocal call_count
        call_count += 1
        return x

    assert cached_function(1) == 1
    assert call_count == 1
    assert cached_function(1) == 1
    assert call_count == 1  # Cached result
    time.sleep(0.2)
    assert cached_function(1) == 1
    assert call_count == 2  # Cache expired
    assert cached_function(2) == 2
    assert call_count == 3


def test_cache_class_maxsize() -> None:
    """Test the Cache decorator with maxsize limit for class method."""

    class _CachedClass:
        def __init__(self) -> None:
            """Initialize call count."""
            self.call_count = 0

        @Cache(maxsize=2)
        def method(self, x: int) -> int:
            """Increment call count and return x."""
            self.call_count += 1
            return x

    obj = _CachedClass()
    assert obj.method(1) == 1
    assert obj.call_count == 1
    assert obj.method(1) == 1
    assert obj.call_count == 1  # Cached result
    assert obj.method(2) == 2
    assert obj.call_count == 2
    assert obj.method(3) == 3
    assert obj.call_count == 3
    assert obj.method(1) == 1
    assert obj.call_count == 4  # Cache miss, as (1) was evicted
    assert obj.method(2) == 2
    assert obj.call_count == 5  # Cache miss, as (2) was evicted


def test_cache_class_timeout() -> None:
    """Test the Cache decorator with timeout for class method."""

    class _CachedClass:
        def __init__(self) -> None:
            """Initialize call count."""
            self.call_count = 0

        @Cache(timeout=0.1)
        def method(self, x: int) -> int:
            """Increment call count and return x."""
            self.call_count += 1
            return x

    obj = _CachedClass()
    assert obj.method(1) == 1
    assert obj.call_count == 1
    assert obj.method(1) == 1
    assert obj.call_count == 1  # Cached result
    time.sleep(0.2)
    assert obj.method(1) == 1
    assert obj.call_count == 2  # Cache expired
    assert obj.method(2) == 2
    assert obj.call_count == 3


def test_cache_class_dict() -> None:
    """Test the Cache decorator with dictionary arguments for class method."""

    class _CachedClass:
        def __init__(self) -> None:
            """Initialize call count."""
            self.call_count = 0

        @Cache()
        def method(self, data: dict) -> dict:
            """Increment call count and return x."""
            self.call_count += 1
            return data

    obj = _CachedClass()
    assert obj.method({"a": 1}) == {"a": 1}
    assert obj.call_count == 1
    assert obj.method({"a": 1}) == {"a": 1}
    assert obj.call_count == 1  # Cached result
    assert obj.method({"a": 2}) == {"a": 2}
    assert obj.call_count == 2
    assert obj.method({"a": 1}) == {"a": 1}
    assert obj.call_count == 2  # Cached result
    assert obj.method({"b": 1}) == {"b": 1}
    assert obj.call_count == 3
    assert obj.method({"a": 2}) == {"a": 2}
    assert obj.call_count == 3  # Cached result
    assert obj.method({"b": 1}) == {"b": 1}
    assert obj.call_count == 3  # Cached result
    assert obj.method({"c": 3, "d": 4}) == {"c": 3, "d": 4}
    assert obj.call_count == 4
    assert obj.method({"a": 1}) == {"a": 1}
    assert obj.call_count == 4  # Cached result


@pytest.mark.asyncio
async def test_cache_async_func_maxsize() -> None:
    """Test the Cache decorator with maxsize limit for async function."""
    call_count = 0

    @Cache(maxsize=2)
    async def cached_function(x: int) -> int:
        nonlocal call_count
        call_count += 1
        return x

    assert await cached_function(1) == 1
    assert call_count == 1
    assert await cached_function(1) == 1
    assert call_count == 1  # Cached result
    assert await cached_function(2) == 2
    assert call_count == 2
    assert await cached_function(3) == 3
    assert call_count == 3
    assert await cached_function(1) == 1
    assert call_count == 4  # Cache miss, as (1) was evicted
    assert await cached_function(2) == 2
    assert call_count == 5  # Cache miss, as (2) was evicted


@pytest.mark.asyncio
async def test_cache_async_func_timeout() -> None:
    """Test the Cache decorator with timeout for async function."""
    call_count = 0

    @Cache(timeout=0.1)
    async def cached_function(x: int) -> int:
        nonlocal call_count
        call_count += 1
        return x

    assert await cached_function(1) == 1
    assert call_count == 1
    assert await cached_function(1) == 1
    assert call_count == 1  # Cached result
    time.sleep(0.2)
    assert await cached_function(1) == 1
    assert call_count == 2  # Cache expired
    assert await cached_function(2) == 2
    assert call_count == 3


@pytest.mark.asyncio
async def test_cache_async_func_dict() -> None:
    """Test the Cache decorator with dictionary arguments for async function."""
    call_count = 0

    @Cache()
    async def cached_function(data: dict) -> dict:
        nonlocal call_count
        call_count += 1
        return data

    assert await cached_function({"a": 1}) == {"a": 1}
    assert call_count == 1
    assert await cached_function({"a": 1}) == {"a": 1}
    assert call_count == 1  # Cached result
    assert await cached_function({"a": 2}) == {"a": 2}
    assert call_count == 2
    assert await cached_function({"a": 1}) == {"a": 1}
    assert call_count == 2  # Cached result
    assert await cached_function({"b": 1}) == {"b": 1}
    assert call_count == 3
    assert await cached_function({"a": 2}) == {"a": 2}
    assert call_count == 3  # Cached result
    assert await cached_function({"b": 1}) == {"b": 1}
    assert call_count == 3  # Cached result
    assert await cached_function({"c": 3, "d": 4}) == {"c": 3, "d": 4}
    assert call_count == 4
    assert await cached_function({"a": 1}) == {"a": 1}
    assert call_count == 4  # Cached result


@pytest.mark.asyncio
async def test_cache_async_class_maxsize() -> None:
    """Test the Cache decorator with maxsize limit for async class method."""

    class _CachedClass:
        def __init__(self) -> None:
            """Initialize call count."""
            self.call_count = 0

        @Cache(maxsize=2)
        async def method(self, x: int) -> int:
            """Increment call count and return x."""
            self.call_count += 1
            return x

    obj = _CachedClass()
    assert await obj.method(1) == 1
    assert obj.call_count == 1
    assert await obj.method(1) == 1
    assert obj.call_count == 1  # Cached result
    assert await obj.method(2) == 2
    assert obj.call_count == 2
    assert await obj.method(3) == 3
    assert obj.call_count == 3
    assert await obj.method(1) == 1
    assert obj.call_count == 4  # Cache miss, as (1) was evicted
    assert await obj.method(2) == 2
    assert obj.call_count == 5  # Cache miss, as (2) was evicted


@pytest.mark.asyncio
async def test_cache_async_class_timeout() -> None:
    """Test the Cache decorator with timeout for async class method."""

    class _CachedClass:
        def __init__(self) -> None:
            """Initialize call count."""
            self.call_count = 0

        @Cache(timeout=0.1)
        async def method(self, x: int) -> int:
            """Increment call count and return x."""
            self.call_count += 1
            return x

    obj = _CachedClass()
    assert await obj.method(1) == 1
    assert obj.call_count == 1
    assert await obj.method(1) == 1
    assert obj.call_count == 1  # Cached result
    time.sleep(0.2)
    assert await obj.method(1) == 1
    assert obj.call_count == 2  # Cache expired
    assert await obj.method(2) == 2
    assert obj.call_count == 3


@pytest.mark.asyncio
async def test_cache_async_class_dict() -> None:
    """Test the Cache decorator with dictionary arguments for async class method."""

    class _CachedClass:
        def __init__(self) -> None:
            """Initialize call count."""
            self.call_count = 0

        @Cache()
        async def method(self, data: dict) -> dict:
            """Increment call count and return x."""
            self.call_count += 1
            return data

    obj = _CachedClass()
    assert await obj.method({"a": 1}) == {"a": 1}
    assert obj.call_count == 1
    assert await obj.method({"a": 1}) == {"a": 1}
    assert obj.call_count == 1  # Cached result
    assert await obj.method({"a": 2}) == {"a": 2}
    assert obj.call_count == 2
    assert await obj.method({"a": 1}) == {"a": 1}
    assert obj.call_count == 2  # Cached result
    assert await obj.method({"b": 1}) == {"b": 1}
    assert obj.call_count == 3
    assert await obj.method({"a": 2}) == {"a": 2}
    assert obj.call_count == 3  # Cached result
    assert await obj.method({"b": 1}) == {"b": 1}
    assert obj.call_count == 3  # Cached result
    assert await obj.method({"c": 3, "d": 4}) == {"c": 3, "d": 4}
    assert obj.call_count == 4
    assert await obj.method({"a": 1}) == {"a": 1}
    assert obj.call_count == 4  # Cached result


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


def test_cache_with_skips() -> None:
    """Test Cache decorator with skips parameter.

    When a parameter is skipped, it's not included in the cache key.
    This means different values for skipped parameters return the same cached result.
    """
    call_count = 0

    @Cache(skips=[0])
    def cached_function(x: int, y: int) -> int:
        nonlocal call_count
        call_count += 1
        return x + y

    assert cached_function(1, 2) == 3
    assert call_count == 1
    assert cached_function(1, 2) == 3
    assert call_count == 1  # Cached result
    assert cached_function(5, 2) == 3  # Returns cached value (3) because x is skipped, y=2 matches
    assert call_count == 1  # Still cached because x (index 0) is skipped
    assert cached_function(5, 3) == 8
    assert call_count == 2  # Cache miss because y changed


def test_cache_with_skips_kwarg() -> None:
    """Test Cache decorator with skips parameter for keyword arguments.

    When a parameter is skipped, it's not included in the cache key.
    Different values for skipped parameters return the same cached result.
    """
    call_count = 0

    @Cache(skips="x")
    def cached_function(x: int, y: int) -> int:
        nonlocal call_count
        call_count += 1
        return x + y

    assert cached_function(x=1, y=2) == 3
    assert call_count == 1
    assert cached_function(x=1, y=2) == 3
    assert call_count == 1  # Cached result
    assert cached_function(x=5, y=2) == 3  # Returns cached value (3) because x is skipped
    assert call_count == 1  # Still cached because x is skipped
    assert cached_function(x=5, y=3) == 8
    assert call_count == 2  # Cache miss because y changed


def test_cache_with_multiple_skips() -> None:
    """Test Cache decorator with multiple skips.

    When multiple parameters are skipped, only the non-skipped parameters
    affect the cache key.
    """
    call_count = 0

    @Cache(skips=[0, "z"])
    def cached_function(x: int, y: int, z: int = 0) -> int:
        nonlocal call_count
        call_count += 1
        return x + y + z

    assert cached_function(1, 2, z=10) == 13
    assert call_count == 1
    assert cached_function(5, 2, z=20) == 13  # Returns cached value (13) because x and z are skipped
    assert call_count == 1  # Still cached because x (index 0) and z are skipped
    assert cached_function(5, 3, z=30) == 38
    assert call_count == 2  # Cache miss because y changed
