"""Unit tests for the laziness of the framework imports in structcast_model.loggers."""

from __future__ import annotations

import subprocess
import sys

FRAMEWORKS = ("torch", "jax", "flax", "orbax", "keras", "tensorflow")
"""The frameworks a logger may serialize with, none of which importing one may pull in."""

REPORT = (
    "import sys; imported = [name for name in "
    f"{FRAMEWORKS!r}"
    " if name in sys.modules]; "
    "raise SystemExit(f'imported: {imported}' if imported else 0)"
)
"""Tail of every script below: fail naming whichever framework the import dragged in."""


def _run(script: str) -> subprocess.CompletedProcess[str]:
    """Run *script* in a fresh interpreter, so the session's own imports cannot mask a regression."""
    return subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=False)


def test_importing_the_loggers_does_not_import_a_framework() -> None:
    """The loggers must stay framework-free until a checkpoint call needs one.

    The property is invisible in ordinary use, so a single stray top-level `import torch` -- here or
    in a dependency such as `mlflow.pytorch` -- would silently revert it. The state backends are
    covered by the same rule: a torch run must not import jax to construct its logger, nor the other
    way round. A subprocess is the only honest check: the test session itself has torch imported
    long before this runs.
    """
    result = _run(
        "import structcast_model.loggers.base, structcast_model.loggers.mlflow, "
        "structcast_model.loggers.state_backends, structcast_model.loggers.wandb; "
        "import structcast_model.loggers as pkg; pkg.Logger; pkg.NullLogger; "
        "pkg.StateBackend; pkg.TorchStateBackend; pkg.FlaxStateBackend; pkg.KerasStateBackend; " + REPORT
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_building_a_logger_with_a_flax_backend_does_not_import_torch() -> None:
    """Choosing a backend must not cost the framework of the other one.

    The torch backend is the default and is constructed for every logger, so a default that imported
    its framework would make a Flax run pay for torch before it ever chose a backend.
    """
    result = _run(
        "from structcast_model.loggers.state_backends import FlaxStateBackend; "
        "from structcast_model.loggers.wandb import WandbLogger; "
        "WandbLogger(experiment='phase-two', state_backend=FlaxStateBackend()); " + REPORT
    )
    assert result.returncode == 0, result.stdout + result.stderr
