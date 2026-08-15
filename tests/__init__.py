"""Tests package for StructCast-Model."""

from pathlib import Path

TEST_DIR = Path(__file__).parent
"""Root directory of the tests."""

FIXTURES_DIR = TEST_DIR / "fixtures"
"""Test-only configuration files."""

CFG_DIR = TEST_DIR.parent / "cfg"
"""Configuration templates shipped at the repository root."""
