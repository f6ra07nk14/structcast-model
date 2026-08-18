"""Unit tests for the laziness of the framework imports in structcast_model.loggers."""

from __future__ import annotations

import subprocess
import sys


def test_importing_the_loggers_does_not_import_torch() -> None:
    """The loggers must stay framework-free until a checkpoint call needs torch.

    The property is invisible in ordinary use, so a single stray top-level `import torch` -- here or
    in a dependency such as `mlflow.pytorch` -- would silently revert it. A subprocess is the only
    honest check: the test session itself has torch imported long before this runs.
    """
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import structcast_model.loggers.base, structcast_model.loggers.mlflow, "
            "structcast_model.loggers.wandb; import sys; "
            "raise SystemExit('torch was imported' if 'torch' in sys.modules else 0)",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
