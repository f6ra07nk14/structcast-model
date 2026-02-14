"""Base utility functions for StructCast-Model."""

from collections import OrderedDict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import wraps
from hashlib import sha256
from inspect import getfullargspec, iscoroutinefunction
from json import dumps as json_dumps
from time import time
from typing import Any, TypeVar, cast

from pydantic_core import from_json, to_jsonable_python
from structcast.utils.base import check_elements, load_yaml
from structcast.utils.security import check_path
from structcast.utils.types import PathLike

T = TypeVar("T")


def load_json(path: PathLike) -> Any:
    """Load a JSON file.

    Args:
        path (PathLike): The path to the JSON file.

    Returns:
        The loaded data.
    """
    with check_path(path).open("r", encoding="utf-8") as f:
        return from_json(f.read())


def load_any(path: PathLike) -> Any:
    """Load any file.

    Args:
        path (PathLike): The path to the file.

    Returns:
        The loaded data.
    """
    path = check_path(path)
    suffix = path.suffix.lower()
    if suffix in (".yaml", ".yml"):
        return load_yaml(path)
    if suffix == ".json":
        with path.open("r", encoding="utf-8") as f:
            return from_json(f.read())
    if suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            return [from_json(line) for line in f]
    raise ValueError(f"Unsupported file type: {suffix}")


def unique(values: Sequence[T]) -> list[T]:
    """Get the unique values from the list.

    Examples:

    .. code-block:: python

    >>> unique(["a", "b", "a", "c"])
    ['a', 'b', 'c']
    >>> unique([1, 2, 1, 3])
    [1, 2, 3]

    Args:
        values (Sequence[T]): The values to check.

    Returns:
        The unique values.
    """
    return list(OrderedDict.fromkeys(values))


def hash_params(args: Any, kwargs: Any, skips: list[int | str]) -> str:
    """Hash the parameters.

    Args:
        args (Any): The positional arguments.
        kwargs (Any): The keyword arguments.
        skips (List[int | str]): The indices or names of the arguments to skip.

    Returns:
        The hash of the parameters.
    """
    if skips:
        raw = {f"__{i}__": v for i, v in enumerate(args) if i not in skips}
        raw.update({k: v for k, v in kwargs.items() if k not in skips})
    else:
        raw = {f"__{i}__": v for i, v in enumerate(args)} | kwargs
    return sha256(json_dumps(to_jsonable_python(raw), sort_keys=True).encode()).hexdigest()


@dataclass(kw_only=True, slots=True)
class _CacheItem:
    value: Any
    timestamp: float


class Cache:
    """Cache the result of a function."""

    def __init__(
        self,
        maxsize: int = 128,
        timeout: float | None = None,
        skips: list[int | str] | int | str | None = None,
    ) -> None:
        """Initialize the cache.

        Args:
            maxsize (int): Maximum size of the cache. If the cache exceeds this size, the oldest item will be removed.
                If maxsize is less than or equal to 0, the cache will grow indefinitely.
            timeout (float | None): Timeout for the cache in seconds. If None, the cache will not expire.
            skips (list[int | str] | int | str | None): The indices or names of the arguments to skip when hashing.
        """
        self.maxsize = maxsize
        self.timeout = timeout
        self.skips: list[int | str] = check_elements(skips)
        self._cache = OrderedDict[str, _CacheItem]()

    def _get_key(self, __has_self__: bool, *args: Any, **kwargs: Any) -> str:
        if __has_self__:
            return f"{id(args[0])}:{hash_params(args[1:], kwargs, self.skips)}"
        return hash_params(args, kwargs, self.skips)

    def _get_value(self, key: str) -> Any:
        if len(self._cache) > self.maxsize > 0:
            self._cache.popitem(last=False)
        return self._cache[key].value

    def _update_time(self, key: str, current: float) -> bool:
        if self.timeout:
            return (current - self._cache[key].timestamp > self.timeout) if key in self._cache else True
        return key not in self._cache

    def __call__(self, func: Callable[..., T]) -> Callable[..., T]:
        """Decorator to cache the result of a function."""
        has_self = "self" in getfullargspec(func).args

        if iscoroutinefunction(func):

            @wraps(func)
            async def awrapper(*args: Any, **kwargs: Any) -> T:
                current = time()
                key = self._get_key(has_self, *args, **kwargs)
                if self._update_time(key, current):
                    self._cache[key] = _CacheItem(value=await func(*args, **kwargs), timestamp=current)
                return self._get_value(key)

            return cast("Callable[..., T]", awrapper)

        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            current = time()
            key = self._get_key(has_self, *args, **kwargs)
            if self._update_time(key, current):
                self._cache[key] = _CacheItem(value=func(*args, **kwargs), timestamp=current)
            return self._get_value(key)

        return wrapper
