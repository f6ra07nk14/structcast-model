"""Base utility functions for StructCast-Model."""

from collections import OrderedDict
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, TypeVar

from pydantic_core import from_json
from structcast.utils.base import load_yaml
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


__all__ = ["load_any", "load_json", "unique"]


if not TYPE_CHECKING:
    import sys

    from structcast.utils.lazy_import import LazySelectedImporter

    sys.modules[__name__] = LazySelectedImporter(__name__, globals())
