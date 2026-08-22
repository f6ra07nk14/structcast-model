"""Canary guarding that mypy still checks Protocol conformance in this repository.

Issue #25: `no_site_packages = true` in the mypy config made the installed `structcast_model`
package unresolvable while running `mypy tests`, so `ignore_missing_imports = true` silently
degraded `Learner` -- and every other Protocol -- to `Any`. Fakes missing protocol members then
type-checked clean, and only runtime `AttributeError`s in pytest revealed them. Nothing in the
suite could tell "the types are fine" apart from "the types are not being checked".

This test type-checks a generated snippet with the repository's real mypy config and fails when a
deliberately non-conforming `Learner` stops being an `[assignment]` error, which is exactly what a
silent `Any` degradation looks like. The conforming control in the same snippet keeps the canary
honest: if the snippet ever stops type-checking for an unrelated reason, the extra error breaks the
"exactly one error" assertion instead of letting the canary rot into a green no-op.

See docs/adr/0007-mypy-tests-resolve-src-with-tripwires.md.
"""

from pathlib import Path

from mypy import api
import pytest

from tests import TEST_DIR

REPO_ROOT = TEST_DIR.parent

CANARY_ASSIGNMENT = "canary: Learner[Any] = NonConforming()"

SNIPPET = f'''"""Snippet type-checked by the mypy Protocol canary."""

from typing import Any

from structcast_model.base_trainer import Learner


class Conforming:
    """Implements every `Learner` member."""

    @property
    def models(self) -> dict[str, Any]:
        return {{}}

    @property
    def optimizers(self) -> dict[str, Any]:
        return {{}}

    @property
    def optimizer_models(self) -> dict[str, list[str]]:
        return {{}}

    @property
    def learning_rates(self) -> dict[str, float]:
        return {{}}

    @property
    def steps(self) -> int:
        return 0

    @property
    def updates(self) -> int:
        return 0

    @property
    def has_updated(self) -> bool:
        return True

    def restore_counters(self, steps: int, updates: int) -> None:
        return None

    def training_step(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return {{}}

    def inference_step(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return {{}}


class NonConforming:
    """Missing every `Learner` member except `models`, like the fakes issue #25 let through."""

    @property
    def models(self) -> dict[str, Any]:
        return {{}}


control: Learner[Any] = Conforming()
{CANARY_ASSIGNMENT}
'''


def test_mypy_rejects_a_non_conforming_learner(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A class missing `Learner` members must be an `[assignment]` error under the repo config."""
    snippet = tmp_path / "learner_conformance.py"
    snippet.write_text(SNIPPET)
    canary_line = SNIPPET.splitlines().index(CANARY_ASSIGNMENT) + 1
    # `mypy_path = ["src"]` in the config is resolved against the working directory, not the config
    # file, so the canary would see an unresolvable package from anywhere else.
    monkeypatch.chdir(REPO_ROOT)

    stdout, stderr, _ = api.run(
        [
            "--config-file",
            str(REPO_ROOT / "pyproject.toml"),
            "--cache-dir",
            str(tmp_path / "mypy_cache"),
            str(snippet),
        ]
    )

    assert stderr == "", stderr
    errors = [line for line in stdout.splitlines() if line.startswith(f"{snippet}:") and ": error: " in line]
    assert len(errors) == 1, f"expected only the non-conforming assignment to fail:\n{stdout}"
    assert errors[0].startswith(f"{snippet}:{canary_line}: error: Incompatible types in assignment"), stdout
    # The reported type proves `Learner` resolved to the protocol instead of degrading to `Any`.
    assert 'variable has type "Learner[Any]"' in errors[0], stdout
    assert "[assignment]" in errors[0], stdout
