"""Tests for Jinja filter helpers used by builders."""

import pytest

from structcast_model.builders import jinja_filters


def test_cumsum_returns_cumulative_values() -> None:
    """Return cumulative sum for a numeric list."""
    assert jinja_filters.cumsum([1, 3, 2, 4]) == [1, 4, 6, 10]


def test_regex_escape_escapes_meta_characters() -> None:
    """Escape regex metacharacters in plain text."""
    assert jinja_filters.regex_escape("a+b?") == "a\\+b\\?"


def test_regex_replace_supports_multiple_flags() -> None:
    """Apply regex replacement with configurable flags."""
    assert jinja_filters.regex_replace("Abc\nabc", "^abc", "X", flags=["ignorecase", "multiline"]) == "X\nX"


def test_regex_findall_and_search() -> None:
    """Return all matches and first match for regex operations."""
    value = "id=10; id=20; name=foo"
    assert jinja_filters.regex_findall(value, r"id=(\d+)") == ["10", "20"]
    assert jinja_filters.regex_search(value, r"name=\w+") == "name=foo"
    assert jinja_filters.regex_search(value, r"missing=\w+") is None


def test_raise_error_and_print_value(capsys: pytest.CaptureFixture[str]) -> None:
    """Raise explicit ValueError and print passthrough values."""
    with pytest.raises(ValueError, match="boom"):
        jinja_filters.raise_error("boom")
    assert jinja_filters.print_value("hello") == "hello"
    assert "hello" in capsys.readouterr().out
