"""Pytest configuration."""

import pytest
from typer.testing import CliRunner


@pytest.fixture
def cli_runner() -> CliRunner:
    """Fixture that provides a Typer CliRunner for testing."""
    return CliRunner()
