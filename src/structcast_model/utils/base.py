"""Base utility functions for StructCast-Model."""

from collections import OrderedDict
from collections.abc import Mapping, Sequence
from logging import getLogger
import re
from typing import TYPE_CHECKING, Any, TypeVar

from pydantic_core import from_json
from structcast.utils.base import find_path, import_from_address, load_yaml
from structcast.utils.types import PathLike

logger = getLogger(__name__)

T = TypeVar("T")


def load_json(path: PathLike) -> Any:
    """Load a JSON file.

    Args:
        path (PathLike): The path to the JSON file.

    Returns:
        The loaded data.
    """
    with find_path(path).open("r", encoding="utf-8") as f:
        return from_json(f.read())


def load_any(path: PathLike) -> Any:
    """Load any file.

    Args:
        path (PathLike): The path to the file.

    Returns:
        The loaded data.
    """
    path = find_path(path)
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


def resolve_tensor_initializer(
    init: str | None,
    dtype: str,
    *,
    float_default: Any,
    int_default: Any,
    protocol: Any,
) -> Any:
    """Resolve the callable creating a dummy tensor for a tensor specification.

    Args:
        init (str | None): The address of the initializer to use,
            or `None` to select a default based on `dtype`.
        dtype (str): The name of the element type of the tensor, e.g. `"bfloat16"` or `"int64"`.
        float_default (Any): The initializer to use for floating point element types.
        int_default (Any): The initializer to use for integer element types,
            since the floating point default cannot produce integer values.
        protocol (Any): The runtime-checkable protocol the resolved initializer must satisfy.

    Returns:
        Any: The initializer, to be called as `initializer(size, dtype=...)`.

    Raises:
        TypeError: If the initializer resolved from `init` does not satisfy `protocol`.

    Note:
        A runtime-checkable protocol only verifies that `__call__` exists, which makes this check
        equivalent to `callable(...)`. A mismatched signature is only detected when the initializer is called.
    """
    if init is not None:
        initializer = import_from_address(init)
        if not isinstance(initializer, protocol):
            raise TypeError(f"Initializer is not callable as a tensor initializer: {init!r}")
        return initializer
    if dtype.startswith("int"):
        logger.warning('No initializer specified for dtype "%s". Falling back to zeros.', dtype)
        return int_default
    return float_default


def resolve_input_shapes(model: Any, shapes: Any = None) -> Any:
    """Resolve the input shapes to create dummy inputs from, preferring the explicitly requested ones.

    Args:
        model (Any): The built model, or a mapping or sequence of models. The `input_shapes` attribute
            emitted by the builders is used when no shapes are requested; for a collection of models,
            the attributes of its members are merged.
        shapes (Any): The explicitly requested shapes, which take precedence when they are not empty.

    Returns:
        Any: The requested shapes, the shapes declared by the model, or `None` when neither is available.
    """
    if shapes:
        return shapes
    if declared := getattr(model, "input_shapes", None):
        return declared
    values = model.values() if isinstance(model, Mapping) else model if isinstance(model, (list, tuple)) else ()
    merged: dict[str, Any] = {}
    for value in values:
        merged.update(resolve_input_shapes(value) or {})
    return merged or None


__all__ = [
    "load_any",
    "load_json",
    "resolve_input_shapes",
    "resolve_tensor_initializer",
    "to_camel",
    "to_pascal",
    "to_snake",
    "unique",
]


if not TYPE_CHECKING:
    import sys

    from structcast.utils.lazy_import import LazySelectedImporter

    sys.modules[__name__] = LazySelectedImporter(__name__, globals())
