"""Jinja filters for structcast model builders."""

from collections.abc import Sequence
from functools import reduce
from itertools import accumulate
from operator import or_
from re import RegexFlag, compile as re_compile, escape as re_escape
from typing import TYPE_CHECKING, TypeVar

from pydantic import TypeAdapter

from structcast_model.builders.schema import check_elements

T = TypeVar("T")


def cumsum(value: list[T]) -> list[T]:
    """Calculate the cumulative sum of a list."""
    return list(accumulate(value))


def _reduce_regex_flags(flags: str | set[str] | Sequence[str] | None) -> int:
    flags = check_elements(TypeAdapter(str | set[str] | Sequence[str] | None).validate_python(flags))
    return reduce(or_, [RegexFlag[f.upper()] for f in flags], 0)


def regex_escape(value: str) -> str:
    """Escape special characters in a string for use in a regular expression."""
    return re_escape(value)


def regex_replace(
    value: str, pattern: str, replacement: str = "", flags: str | set[str] | Sequence[str] | None = None
) -> str:
    """Replace a pattern in a string with a replacement using regular expressions."""
    return re_compile(pattern, _reduce_regex_flags(flags)).sub(replacement, value)


def regex_findall(value: str, pattern: str, flags: str | set[str] | Sequence[str] | None = None) -> list[str]:
    """Find all occurrences of a pattern in a string using regular expressions."""
    return re_compile(pattern, _reduce_regex_flags(flags)).findall(value)


def regex_search(value: str, pattern: str, flags: str | set[str] | Sequence[str] | None = None) -> str | None:
    """Search for a pattern in a string using regular expressions and return the first match."""
    match = re_compile(pattern, _reduce_regex_flags(flags)).search(value)
    return match.group(0) if match else None


def raise_error(message: str) -> None:
    """Raise an error with the given message."""
    raise ValueError(message)


def print_value(value: T) -> T:
    """Print the value and return it."""
    print(value)
    return value


__all__: list[str] = [
    "cumsum",
    "print_value",
    "raise_error",
    "regex_escape",
    "regex_findall",
    "regex_replace",
    "regex_search",
]

if not TYPE_CHECKING:
    import sys

    from structcast.utils.lazy_import import LazySelectedImporter

    sys.modules[__name__] = LazySelectedImporter(__name__, globals())
