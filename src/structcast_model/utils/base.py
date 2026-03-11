"""Base utility functions for StructCast-Model."""

from collections import OrderedDict
from collections.abc import Sequence
import re
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


def to_snake(value: str) -> str:
    """Convert a PascalCase, camelCase, or kebab-case string to snake_case.

    Args:
        value: The string to convert.

    Returns:
        The converted string in snake_case.
    """
    # Handle the sequence of uppercase letters followed by a lowercase letter
    value = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", value)
    # Insert an underscore between a lowercase letter and an uppercase letter
    value = re.sub(r"([a-z])([A-Z])", r"\1_\2", value)
    # Insert an underscore between a digit and an uppercase letter
    value = re.sub(r"([0-9])([A-Z])", r"\1_\2", value)
    # Insert an underscore between a lowercase letter and a digit
    value = re.sub(r"([a-z])([0-9])", r"\1_\2", value)
    value = re.sub(r"(\W+)", "_", value)
    value = re.sub("__([A-Z])", r"_\1", value)
    return value.lower()


def to_pascal(value: str) -> str:
    """Convert a snake_case string to PascalCase.

    Args:
        value: The string to convert.

    Returns:
        The PascalCase string.
    """
    return "".join(word.title() for word in to_snake(value).split("_"))


def to_camel(value: str) -> str:
    """Convert a snake_case string to camelCase.

    Args:
        value: The string to convert.

    Returns:
        The converted camelCase string.
    """
    camel = to_pascal(value)
    return camel[0].lower() + camel[1:] if camel else ""


__all__ = ["load_any", "load_json", "to_camel", "to_pascal", "to_snake", "unique"]


if not TYPE_CHECKING:
    import sys

    from structcast.utils.lazy_import import LazySelectedImporter

    sys.modules[__name__] = LazySelectedImporter(__name__, globals())
