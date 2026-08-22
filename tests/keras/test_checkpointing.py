"""Runtime tests for the `keras.remat` wrapper the builder emits around a checkpointed layer's call.

The generated layer is exec'd from a file and trained by a generated learner on whichever backend
`KERAS_BACKEND` selects (the conftest defaults it to tensorflow): the wrapper is written in `keras`
alone, so the same emission has to train identically on all three.
"""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest

import keras
from structcast_model.builders.keras import KerasBuilder, KerasLearnerBuilder
from structcast_model.keras.trainer import initial_model
from structcast_model.utils.base import load_any
from tests import FIXTURES_DIR

CFG_DIR = FIXTURES_DIR / "cfg" / "keras"
MODEL_YAML = CFG_DIR / "Linear.yaml"
LEARNER_YAML = CFG_DIR / "LinearLearner.yaml"

X = np.asarray([[1.0, 0.5, -0.5, 2.0], [0.0, 1.0, 1.0, -1.0]], dtype="float32")
"""One fixed batch, so two runs can only differ through the code under test."""

Y = np.asarray([[1.0, -1.0], [0.5, 0.25]], dtype="float32")


def _load(path: Path, name: str) -> ModuleType:
    """Load a generated module by file path, the way a configuration does: not by import name."""
    spec = spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _layer(tmp_path: Path, checkpointing: Any) -> Any:
    """Generate the fixture layer with the given `GRADIENT_CHECKPOINTING` value and seed it."""
    raw = {**load_any(MODEL_YAML), "GRADIENT_CHECKPOINTING": checkpointing}
    directory = tmp_path / str(checkpointing)
    KerasBuilder(raw=raw)()(directory / "model.py")
    keras.utils.set_random_seed(0)
    layer = _load(directory / "model.py", "generated_model").Model()
    layer.build({"x": (None, 4)})
    return layer


def _values(variables: Any) -> list[np.ndarray]:
    """Read the host-side values of variables, in the order they were given."""
    return [np.asarray(keras.ops.convert_to_numpy(getattr(v, "value", v))) for v in variables]


def _train(tmp_path: Path, checkpointing: Any) -> tuple[list[float], list[np.ndarray]]:
    """Run two training steps of the generated learner and report the losses and the variables."""
    KerasLearnerBuilder.from_path(LEARNER_YAML)()(tmp_path / "learner.py")
    model = initial_model(_layer(tmp_path, checkpointing), {"x": (4,)})
    learner = _load(tmp_path / "learner.py", "generated_learner").Learner(model)
    losses = [float(keras.ops.convert_to_numpy(learner.training_step(x=X, y=Y)["loss"])) for _ in range(2)]
    return losses, _values(model.trainable_variables)


def test_checkpointing_changes_neither_the_losses_nor_the_variables(tmp_path: Path) -> None:
    """Rematerialization is a memory trade, so exact equality is the contract, not a tolerance."""
    baseline_losses, baseline_variables = _train(tmp_path / "off", False)
    losses, variables = _train(tmp_path / "on", True)
    assert losses == baseline_losses
    for expected, actual in zip(baseline_variables, variables, strict=True):
        assert np.array_equal(actual, expected)


def test_an_inference_call_never_rematerializes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Nothing is recomputed without a backward pass, so the inference call must take the plain path."""
    layer = _layer(tmp_path, True)
    calls: list[int] = []
    remat = keras.remat

    def _counted(function: Any) -> Any:
        calls.append(1)
        return remat(function)

    monkeypatch.setattr(keras, "remat", _counted)

    inference = layer(X, training=False)
    assert calls == []
    training = layer(X, training=True)
    assert calls == [1]
    assert np.array_equal(_values([inference])[0], _values([training])[0])


def test_the_emission_leaves_the_layer_a_plain_keras_layer(tmp_path: Path) -> None:
    """No base class stands between the layer and Keras, which reads `call` to route a batch by name."""
    layer = _layer(tmp_path, True)
    assert type(layer).__mro__[1] is keras.layers.Layer
    assert np.array_equal(_values([layer(x=X, training=False)])[0], _values([layer(X, training=False)])[0])
