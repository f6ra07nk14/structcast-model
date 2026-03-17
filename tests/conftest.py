"""Pytest configuration."""

from collections.abc import Generator
import pathlib

import pytest
import torch.distributed as dist
from typer.testing import CliRunner


@pytest.fixture
def cli_runner() -> CliRunner:
    """Fixture that provides a Typer CliRunner for testing."""
    return CliRunner()


@pytest.fixture
def single_process_gloo(tmp_path: pathlib.Path) -> Generator[None, None, None]:
    """Initialize a single-process gloo distributed group for testing."""
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{tmp_path / 'dist_init'}",
        rank=0,
        world_size=1,
    )
    yield
    dist.destroy_process_group()
