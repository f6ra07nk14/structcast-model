"""Runtime tests for the Keras model and learner templates shipped under `cfg/keras`.

The templates are what a validation run trains, so building the pair and running a step is the only
check that answers whether they work together: a model whose output key the learner does not read, a
loss that cannot consume the labels a dataset produces, or a segment whose variables belong to
nobody all render into perfectly valid Python and only fail on the first batch.

Everything here is deliberately tiny -- one block per stage, sixteen pixels, four items -- and runs
on the CPU on whichever backend `KERAS_BACKEND` selects (the conftest defaults it to tensorflow).
The batches are fixed: a moving batch hides a dead optimizer, since the loss wanders on its own and
a step that updates nothing still produces a plausible-looking curve.
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
from tests import CFG_DIR

MODELS = CFG_DIR / "keras" / "models"
LEARNERS = CFG_DIR / "keras" / "learners"

RNG = np.random.default_rng(0)
"""One generator for every fixed batch below, so the arrays are the same in every process."""


def _load(path: Path, name: str) -> ModuleType:
    """Load a generated module by file path, the way a configuration does: not by import name."""
    spec = spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _model(tmp_path: Path, name: str, parameters: dict[str, Any], shapes: dict[str, Any] | None) -> Any:
    """Render a model template and trace it into a built `keras.Model`, as the training CLI does."""
    KerasBuilder.from_path(MODELS / f"{name}.yaml")(parameters=parameters)(tmp_path / f"{name}.py")
    keras.utils.set_random_seed(0)
    return initial_model(_load(tmp_path / f"{name}.py", f"model_{name}").Model(), shapes)


def _learner(tmp_path: Path, name: str, parameters: dict[str, Any], **models: Any) -> Any:
    """Render a learner template and construct it over *models*, as the training CLI does."""
    KerasLearnerBuilder.from_path(LEARNERS / f"{name}.yaml")(parameters=parameters)(tmp_path / f"{name}_learner.py")
    return _load(tmp_path / f"{name}_learner.py", f"learner_{name}").Learner(**models)


def _values(variables: Any) -> list[np.ndarray]:
    """Read the host-side values of variables, in the order they were given."""
    return [np.asarray(keras.ops.convert_to_numpy(variable.value)) for variable in variables]


def _moved(before: list[np.ndarray], after: list[np.ndarray]) -> float:
    """Return the largest absolute change between two reads of the same variables."""
    return max(float(np.abs(a - b).max()) for a, b in zip(before, after, strict=True))


def _floats(criteria: dict[str, Any]) -> dict[str, float]:
    """Bring one step's criteria back to the host as plain floats, as the tracker does."""
    return {name: float(keras.ops.convert_to_numpy(value)) for name, value in criteria.items()}


IMAGE_BATCH = {
    "image": RNG.random((4, 16, 16, 3), dtype="float32"),
    "label": RNG.integers(0, 10, 4).astype("int64"),
}
"""One fixed classification batch for the two image learners."""

TOKEN_BATCH = {
    "tokens": RNG.integers(0, 12, (4, 8)).astype("int64"),
    "targets": RNG.integers(0, 12, (4, 8)).astype("int64"),
}
"""One fixed next-token batch, targets independent of tokens: only the loss falling matters here."""

Pair = tuple[str, dict[str, Any], dict[str, Any] | None, str, dict[str, Any], dict[str, Any]]
"""One case: model template, model parameters, model shapes, learner template, its parameters, batch."""

PAIRS: dict[str, Pair] = {
    "ConvNeXtV2": (
        "ConvNeXtV2",
        {"SHARED": {"num_classes": 10}, "atto": {"dims": [4, 8, 8, 16], "depths": [1, 1, 1, 1]}},
        {"image": (16, 16, 3)},
        "ConvNeXtV2",
        {"DEFAULT": {"epochs": 2, "warmup_epochs": 1, "steps_per_epoch": 2, "label_smoothing": 0.1}},
        IMAGE_BATCH,
    ),
    "VisionTransformer": (
        "VisionTransformer",
        {"base": {"dim": 16, "heads": 2, "depth": 1, "image_size": 16, "patch_size": 8, "num_classes": 10}},
        None,
        "ImageClassifier",
        {},
        IMAGE_BATCH,
    ),
    "SmallLanguageModel": (
        "SmallLanguageModel",
        {"tiny": {"dim": 16, "heads": 2, "depth": 1, "max_seq_len": 8, "vocab_size": 12}},
        None,
        "SmallLanguageModel",
        {},
        TOKEN_BATCH,
    ),
}


@pytest.mark.parametrize("case", list(PAIRS), ids=list(PAIRS))
def test_a_shipped_model_and_learner_pair_trains_on_the_cpu(case: str, tmp_path: Path) -> None:
    """Two steps on one batch must report finite criteria and actually move the weights.

    A learner whose segment holds the variables of another copy of the model, or one differentiating
    a criterion that happens not to depend on them, still returns finite losses -- they just would
    not move. `None` shapes are the point of the two templates that declare `INPUT_SHAPES`: the CLI
    traces them without a `--shape`, so a declaration that did not survive rendering fails here.
    """
    model_name, model_parameters, shapes, learner_name, learner_parameters, batch = PAIRS[case]
    model = _model(tmp_path, model_name, model_parameters, shapes)
    learner = _learner(tmp_path, learner_name, learner_parameters, model=model)
    before = _values(model.trainable_variables)

    first = _floats(learner.training_step(**batch))
    second = _floats(learner.training_step(**batch))

    assert sorted(first) == sorted(learner.outputs)
    assert all(np.isfinite(value) for value in second.values())
    assert second["ce_loss"] < first["ce_loss"]
    assert learner.has_updated is True
    assert (learner.steps, learner.updates) == (2, 2)
    assert _moved(before, _values(model.trainable_variables)) > 0.0


@pytest.mark.parametrize("case", list(PAIRS), ids=list(PAIRS))
def test_a_shipped_pair_evaluates_without_training(case: str, tmp_path: Path) -> None:
    """Validation must report the same criteria and move nothing, or every run's curve is a lie."""
    model_name, model_parameters, shapes, learner_name, learner_parameters, batch = PAIRS[case]
    model = _model(tmp_path, model_name, model_parameters, shapes)
    learner = _learner(tmp_path, learner_name, learner_parameters, model=model)
    before = _values(model.trainable_variables)

    criteria = _floats(learner.inference_step(**batch))

    assert sorted(criteria) == sorted(learner.outputs)
    assert all(np.isfinite(value) for value in criteria.values())
    assert _moved(before, _values(model.trainable_variables)) == 0.0


def test_a_shipped_learner_accumulates_over_the_window_its_optimizer_was_given(tmp_path: Path) -> None:
    """`accumulate_gradients` is a template parameter, and it has to reach the run's update cadence.

    It travels through `gradient_accumulation_steps` in the OPTIMIZER pattern, where Keras owns both
    the buffering and the gate, so the only proof it arrived is a run: the variables must sit still
    for two steps and move on the third, and `has_updated` must agree step by step -- a template that
    dropped the keyword would train at three times the intended step count and report nothing.
    """
    model_name, model_parameters, shapes, learner_name, _, batch = PAIRS["VisionTransformer"]
    model = _model(tmp_path, model_name, model_parameters, shapes)
    learner = _learner(tmp_path, learner_name, {"DEFAULT": {"accumulate_gradients": 3}}, model=model)

    flags, moves = [], []
    for _ in range(6):
        before = _values(model.trainable_variables)
        learner.training_step(**batch)
        flags.append(learner.has_updated)
        moves.append(_moved(before, _values(model.trainable_variables)) > 0.0)

    assert learner.optimizers["optimizer"].gradient_accumulation_steps == 3
    assert flags == [False, False, True, False, False, True]
    assert flags == moves
    assert (learner.steps, learner.updates) == (6, 2)


CYCLEGAN_BATCH = {
    "real_A": RNG.random((2, 16, 16, 3), dtype="float32"),
    "real_B": RNG.random((2, 16, 16, 3), dtype="float32"),
}
"""One fixed unpaired batch: the CycleGAN learner reads the two real images and nothing else."""


def _cyclegan(tmp_path: Path) -> tuple[Any, dict[str, Any]]:
    """Build the four tiny CycleGAN models -- two generators, two discriminators -- and their learner."""
    shapes = {"image": (16, 16, 3)}
    KerasBuilder.from_path(MODELS / "CycleGAN_generator.yaml")(
        parameters={"DEFAULT": {"n_residual_blocks": 1, "init_features": 4}}
    )(tmp_path / "generator.py")
    KerasBuilder.from_path(MODELS / "CycleGAN_discriminator.yaml")()(tmp_path / "discriminator.py")
    types = {
        "G": _load(tmp_path / "generator.py", "cyclegan_generator").Model,
        "D": _load(tmp_path / "discriminator.py", "cyclegan_discriminator").Model,
    }
    keras.utils.set_random_seed(0)
    models = {name: initial_model(types[name[0]](), shapes) for name in ("G_AB", "G_BA", "D_A", "D_B")}
    parameters = {"DEFAULT": {"epochs": 4, "decay_epoch": 2, "steps_per_epoch": 2}}
    learner = _learner(tmp_path, "CycleGAN", parameters, **models)
    return learner, models


def test_the_cyclegan_pair_trains_every_one_of_its_four_models(tmp_path: Path) -> None:
    """Three segments, four models: one step has to move all of them and report finite criteria.

    The generators are trained by the first segment and the two discriminators by their own, so a
    variable list that missed a model -- the second generator above all, which the first segment
    owns alongside the first -- shows up here as weights that never move.
    """
    learner, models = _cyclegan(tmp_path)
    before = {name: _values(model.trainable_variables) for name, model in models.items()}

    criteria = _floats(learner.training_step(**CYCLEGAN_BATCH))

    assert sorted(criteria) == sorted(learner.outputs)
    assert all(np.isfinite(value) for value in criteria.values())
    assert learner.optimizer_models == {
        "optimizer_G": ["G_AB", "G_BA"],
        "optimizer_D_A": ["D_A"],
        "optimizer_D_B": ["D_B"],
    }
    for name, model in models.items():
        assert _moved(before[name], _values(model.trainable_variables)) > 0.0, name


def test_each_cyclegan_segment_computes_everything_it_reads(tmp_path: Path) -> None:
    """A Keras segment is one function the adapter calls with the batch alone, and nothing else.

    That is why the discriminator segments generate their own fake image and build their own "valid"
    target instead of reading the generator segment's, as the torch twin does. Calling each flow on
    the batch by itself is what proves the template respects it: a segment reaching for another's
    value raises `NameError` here rather than inside a traced graph on the first real batch.
    """
    learner, _ = _cyclegan(tmp_path)

    for name in ("optimizer_G", "optimizer_D_A", "optimizer_D_B"):
        loss, criteria = getattr(learner, f"_flow_{name}")(**CYCLEGAN_BATCH)

        assert np.isfinite(float(keras.ops.convert_to_numpy(loss)))
        assert all(np.isfinite(value) for value in _floats(criteria).values())


def test_the_cyclegan_pair_evaluates_every_criterion_without_training(tmp_path: Path) -> None:
    """One inference flow reports what three training segments compute between them, and trains none.

    The learner's inference flow is the concatenation of the three segments' -- the only place the
    generator and both discriminator criteria are produced together -- so a name a segment stores
    but the inference flow never reaches is missing exactly here, and nowhere else.
    """
    learner, models = _cyclegan(tmp_path)
    before = {name: _values(model.trainable_variables) for name, model in models.items()}

    criteria = _floats(learner.inference_step(**CYCLEGAN_BATCH))

    assert sorted(criteria) == sorted(learner.outputs)
    assert all(np.isfinite(value) for value in criteria.values())
    for name, model in models.items():
        assert _moved(before[name], _values(model.trainable_variables)) == 0.0, name
