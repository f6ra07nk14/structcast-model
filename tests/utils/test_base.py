"""Test tools."""

import time

import pytest

from structcast_model.utils.base import Cache


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
