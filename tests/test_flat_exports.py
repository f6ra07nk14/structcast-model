"""Unit tests for the flat symbol exports of the lazy package routers."""

from __future__ import annotations

from importlib import import_module
from types import ModuleType

import pytest

import structcast_model
import structcast_model.flax
import structcast_model.keras
import structcast_model.loggers
import structcast_model.torch


@pytest.mark.parametrize(
    "package",
    [
        structcast_model,
        structcast_model.flax,
        structcast_model.keras,
        structcast_model.loggers,
        structcast_model.torch,
    ],
    ids=["structcast_model", "flax", "keras", "loggers", "torch"],
)
def test_flat_attributes_resolve_to_their_routed_module(package: ModuleType) -> None:
    """Every flat symbol must reach the very object its routed module holds.

    The routing table in each `__init__.py` is hand-maintained, and `LazySelectedImporter` builds
    `_class_to_module` from it with no validation: a symbol pointed at the wrong module -- or listed
    twice, where only the last entry survives -- resolves silently on first access and only surfaces
    as a confusing `AttributeError` or a stale duplicate far from the mistake. Sweeping the table
    itself keeps the guarantee tied to every entry rather than to a hand-picked sample.
    """
    for module_name, symbols in package._imported_structure.items():
        if not symbols:
            continue
        module = import_module(f"{package.__name__}.{module_name}")
        for symbol in symbols:
            assert getattr(package, symbol) is getattr(module, symbol)
