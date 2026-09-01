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
from re import search as re_search
from types import ModuleType
from typing import Any

import numpy as np
import pytest

import keras
from structcast_model.builders.keras import KerasBuilder, KerasLearnerBuilder
from structcast_model.keras.trainer import initial_model
from structcast_model.utils.base import load_any
from tests import CFG_DIR

MODELS = CFG_DIR / "keras" / "models"
LEARNERS = CFG_DIR / "keras" / "learners"
STRATEGIES = CFG_DIR / "keras" / "strategies"

BACKEND = keras.backend.backend()

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


def test_the_language_model_rotates_its_attention_rather_than_looking_a_position_up(tmp_path: Path) -> None:
    """The rotary position embedding is what makes this template the twin of the torch and flax ones.

    A learned position table -- what this template carried before -- reads as plausibly as a rotation
    on the batch it was traced with, so two properties are pinned instead: the attention section
    against the reference RoPE formula in numpy, which no other position scheme reproduces, and a
    forward pass three times longer than the traced `max_seq_len`, which a table of `max_seq_len`
    rows cannot serve at all. The raw layer is called rather than a traced `keras.Model`, whose input
    spec would pin the length of the trace.
    """
    model_name, model_parameters, *_ = PAIRS["SmallLanguageModel"]
    dim, heads = model_parameters["tiny"]["dim"], model_parameters["tiny"]["heads"]
    head_dim, half, batch, seq = dim // heads, dim // heads // 2, 2, 6
    rng = np.random.default_rng(1)
    KerasBuilder.from_path(MODELS / f"{model_name}.yaml")(parameters=model_parameters)(tmp_path / "rope.py")
    module = _load(tmp_path / "rope.py", "model_rope")
    keras.utils.set_random_seed(0)
    attention = module.CausalSelfAttention()
    hidden = rng.standard_normal((batch, seq, dim)).astype("float32")

    attended = keras.ops.convert_to_numpy(attention(hidden))

    def rotate_half(tensor: np.ndarray) -> np.ndarray:
        """The second half of every head's features, negated, brought in front of the first."""
        return np.concatenate((-tensor[..., half:], tensor[..., :half]), axis=-1)

    qkv_kernel, qkv_bias, out_kernel, out_bias = _values(attention.weights)
    fused = (hidden @ qkv_kernel + qkv_bias).reshape(batch, seq, 3, heads, head_dim)
    query, key, value = fused[:, :, 0], fused[:, :, 1], fused[:, :, 2]
    freqs = np.arange(seq, dtype="float32")[:, None] * 10000.0 ** (-np.arange(0, head_dim, 2, "float32") / head_dim)
    angles = np.concatenate((freqs, freqs), axis=-1)
    cos, sin = np.cos(angles)[None, :, None], np.sin(angles)[None, :, None]
    query, key = query * cos + rotate_half(query) * sin, key * cos + rotate_half(key) * sin
    scores = np.einsum("bqhd,bkhd->bhqk", query, key) / head_dim**0.5
    scores = np.where(np.tril(np.ones((seq, seq), bool)), scores, -np.inf)
    weights = np.exp(scores - scores.max(-1, keepdims=True))
    heads_merged = np.einsum("bhqk,bkhd->bqhd", weights / weights.sum(-1, keepdims=True), value)
    assert np.allclose(attended, heads_merged.reshape(batch, seq, dim) @ out_kernel + out_bias, atol=1e-5)

    traced = model_parameters["tiny"]["max_seq_len"]
    tokens = rng.integers(0, 12, (batch, 3 * traced)).astype("int64")
    logits = keras.ops.convert_to_numpy(module.Model()(tokens)["logits"])

    assert logits.shape == (batch, 3 * traced, model_parameters["tiny"]["vocab_size"])
    assert np.isfinite(logits).all()


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


@pytest.mark.parametrize("case", ["VisionTransformer", "SmallLanguageModel"], ids=str.lower)
def test_the_shipped_tensor_parallel_rules_name_this_models_variables(case: str, tmp_path: Path) -> None:
    """`cfg/keras/strategies/tp.yaml` is written for these templates, so its rules have to match them.

    A rule matching no variable is refused at wrap time -- loud, but only once the run started -- and
    a table that matched only half of a column/row pair would train a wrong answer, so both halves
    are pinned: against the `MultiHeadAttention` sublayer names Keras builds for the transformer, and
    against the two projections the language model's own attention section names. The MLP is
    deliberately outside the plan: Keras numbers its `Dense` layers from a global counter, so no
    stable regex tells the first of a block from the second.
    """
    model_name, model_parameters, shapes, *_ = PAIRS[case]
    model = _model(tmp_path, model_name, model_parameters, shapes)
    strategy = STRATEGIES / "tp.yaml"
    if case == "SmallLanguageModel":
        # The two tables cannot ship active together, so this model's pair is commented under the
        # transformer's. Uncommenting it back into a file that loads is what keeps it honest: an
        # alternate nothing parses is exactly what went stale when this attention was rewritten.
        lines = [line for line in strategy.read_text().splitlines() if '- ["multi_head' not in line]
        strategy = tmp_path / "tp.yaml"
        strategy.write_text("\n".join(line.replace("# - [", "- [") for line in lines))
    rules = [(pattern, tactic) for pattern, tactic in load_any(str(strategy))[2]["_bind_"]["rules"]]
    paths = [variable.path for variable in model.variables]

    matched = {tactic: [path for path in paths if re_search(pattern, path)] for pattern, tactic in rules}

    depth = next(iter(model_parameters.values()))["depth"]
    columns = ("query", "key", "value") if case == "VisionTransformer" else ("qkv_proj",)
    output = "attention_output" if case == "VisionTransformer" else "out_proj"
    assert [tactic for _, tactic in rules] == ["column", "row"]
    # A kernel and a bias for each projection the column rule names, and for the output one, per
    # block: the counts are what make this a pinned pair rather than "something matched".
    assert len(matched["column"]) == 2 * len(columns) * depth
    assert len(matched["row"]) == 2 * depth
    assert all(path.rsplit("/", 2)[-2] in columns for path in matched["column"])
    assert all(path.rsplit("/", 2)[-2] == output for path in matched["row"])


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


SHOWCASE_PARAMETERS: dict[str, dict[str, Any]] = {
    "base": {"dim": 16, "heads": 2, "depth": 2, "image_size": 16, "patch_size": 8, "num_classes": 10},
    "SHARED": {"drop_path_rate": 0.0},
}
"""The showcase Vision Transformer: two blocks, and the stochastic depth of the recipe switched off.

`drop_path_rate` is not decoration here. A block's `DropPath` is a `keras.layers.Dropout` one TYPE
sublayer down, and `keras.remat` re-draws seed state on the recomputation, so the checkpointed build
is refused at any other rate. At rate 0 the layer draws nothing and the build goes through; the
showcase command sets it for that reason, and `gradient_checkpointing` is left out so the two builds
below can set it either way.
"""


@pytest.fixture
def restore_policy() -> Any:
    """Restore the global mixed precision policy, which is process-wide state."""
    original = keras.mixed_precision.global_policy()
    yield
    keras.mixed_precision.set_global_policy(original)


def test_the_showcase_pair_runs_every_feature_this_backend_has_in_one_step(
    tmp_path: Path, restore_policy: None
) -> None:
    """Checkpointing, accumulation, the policy and the optimizer's EMA have to hold together.

    Each is pinned on its own elsewhere; what only a combined run can answer is whether they
    compose, and on this backend three of them meet inside one optimizer. They do interact, not
    always kindly: the average advances on the accumulation no-ops too (a recorded limitation, and
    the reason the average is read here rather than its cadence), and under the policy the head
    emits bfloat16, which `keras.metrics.sparse_top_k_categorical_accuracy` refuses outright -- the
    float32 cast in the showcase flow is what keeps the two composable.
    """
    keras.mixed_precision.set_global_policy("mixed_bfloat16")
    models = {}
    for remat in (True, False):
        name = f"VisionTransformer_{remat}"
        parameters = {
            **SHOWCASE_PARAMETERS,
            "SHARED": {**SHOWCASE_PARAMETERS["SHARED"], "gradient_checkpointing": remat},
        }
        KerasBuilder.from_path(MODELS / "VisionTransformer.yaml")(parameters=parameters)(tmp_path / f"{name}.py")
        keras.utils.set_random_seed(0)
        models[remat] = initial_model(_load(tmp_path / f"{name}.py", name).Model(), None)
    learner = _learner(
        tmp_path, "ImageClassifierShowcase", {"DEFAULT": {"accumulate_gradients": 2}}, model=models[True]
    )
    optimizer = learner.optimizers["optimizer"]

    assert "keras.remat(lambda *arrays: self._call_body(" in (tmp_path / "VisionTransformer_True.py").read_text()
    assert "keras.remat" not in (tmp_path / "VisionTransformer_False.py").read_text()
    # Keras numbers its layers from a process-global counter, so two builds cannot share a path;
    # what checkpointing must not do is add a variable or reshape one, which the layout does show.
    assert [tuple(v.shape) for v in models[True].variables] == [tuple(v.shape) for v in models[False].variables]
    logits = [keras.ops.cast(m(IMAGE_BATCH["image"], training=True)["cls"], "float32") for m in models.values()]
    assert np.array_equal(*(keras.ops.convert_to_numpy(value) for value in logits))

    assert (learner.MIXED_PRECISION, learner.MIXED_PRECISION_TYPE) == (True, "bfloat16")
    assert keras.backend.standardize_dtype(models[True](IMAGE_BATCH["image"])["cls"].dtype) == "bfloat16"
    assert all(v.dtype == "float32" for v in models[True].trainable_variables)
    assert (optimizer.gradient_accumulation_steps, optimizer.use_ema, optimizer.ema_momentum) == (2, True, 0.99)

    flags = [(learner.training_step(**IMAGE_BATCH), learner.has_updated)[1] for _ in range(4)]

    assert flags == [False, True, False, True]
    assert (learner.steps, learner.updates) == (4, 2)
    # Keras keeps the average in the optimizer rather than in a second model, so what a combined run
    # can assert is that the averages exist and shadow every trained variable; the evaluation the
    # swap runs on them is asserted on its own below.
    assert len(optimizer._model_variables_moving_average) == len(models[True].trainable_variables)
    assert all(np.isfinite(value) for value in _floats(learner.inference_step(**IMAGE_BATCH)).values())


def test_the_showcase_validates_on_the_average_it_trains(tmp_path: Path) -> None:
    """The shipped template is what a validation run trains, so its EMA is asserted on the real thing.

    Everything the swap has to reach is only true of a generated learner: the average lives in the
    optimizer the `OPTIMIZER` pattern built, the window is the template's, and the models are the
    ones the CLI hands over. Two updates are enough to pull the average away from the weights, which
    is what makes the two evaluations below distinguishable at all.
    """
    model = _model(tmp_path, "VisionTransformer", SHOWCASE_PARAMETERS, None)
    learner = _learner(tmp_path, "ImageClassifierShowcase", {"DEFAULT": {"accumulate_gradients": 2}}, model=model)
    optimizer = learner.optimizers["optimizer"]
    for _ in range(4):
        learner.training_step(**IMAGE_BATCH)
    weights = _values(model.trainable_variables)
    averages = _values(optimizer._model_variables_moving_average)

    averaged = _floats(learner.inference_step(**IMAGE_BATCH))

    assert _moved(weights, averages) > 0.0
    assert _moved(weights, _values(model.trainable_variables)) == 0.0
    assert _moved(averages, _values(optimizer._model_variables_moving_average)) == 0.0
    # The raw-weight reading of the same batch, taken by dropping the average the swap reaches for.
    learner._ema_optimizers = []
    raw = _floats(learner.inference_step(**IMAGE_BATCH))

    assert averaged["ce_loss"] != pytest.approx(raw["ce_loss"], rel=1e-6)


def test_the_showcase_average_can_be_left_out_by_parameter(tmp_path: Path) -> None:
    """The `ema` parameter exists for the torch twin, whose FSDP2 runs refuse an average outright.

    Nothing on this backend refuses one, but the three templates are read side by side and a knob
    spelled differently on each is a knob nobody trusts. Here it has to reach both EMA keywords and
    only those: `ema_momentum` left behind alone would be Keras' own default and invisible, while an
    accumulation window silently dropped with it would change what the run measures.
    """
    model = _model(tmp_path, "VisionTransformer", SHOWCASE_PARAMETERS, None)
    learner = _learner(tmp_path, "ImageClassifierShowcase", {"SHARED": {"ema": False}}, model=model)

    optimizer = learner.optimizers["optimizer"]

    assert optimizer.use_ema is False
    assert "ema_momentum" not in (tmp_path / "ImageClassifierShowcase_learner.py").read_text()
    assert optimizer.gradient_accumulation_steps == 4
    assert all(np.isfinite(value) for value in _floats(learner.inference_step(**IMAGE_BATCH)).values())


def test_the_showcase_window_can_be_taken_off_by_parameter(tmp_path: Path) -> None:
    """Every knob on this template is one a run turns off, and accumulation had no off switch.

    `gradient_accumulation_steps` rendered unguarded puts Python's `None` into the YAML as the
    *string* `None`, which Keras compares against 2 and rejects -- so the showcase could not express
    "train without a window" at all. Off, the keyword has to be absent rather than passed as
    anything: Keras reads a missing one as no accumulation, and the learner's own window read
    (`... or 1`) would take any non-empty string for a real window.
    """
    model = _model(tmp_path, "VisionTransformer", SHOWCASE_PARAMETERS, None)
    learner = _learner(tmp_path, "ImageClassifierShowcase", {"DEFAULT": {"accumulate_gradients": None}}, model=model)
    source = (tmp_path / "ImageClassifierShowcase_learner.py").read_text()

    flags = [(learner.training_step(**IMAGE_BATCH), learner.has_updated)[1] for _ in range(2)]

    assert "gradient_accumulation_steps=" not in source  # the keyword is absent, not passed as anything
    assert learner.optimizers["optimizer"].gradient_accumulation_steps is None
    assert flags == [True, True]  # without a window every step applies
    assert learner.updates == 2


def test_the_image_classifier_learner_derives_both_precision_fields_from_one_parameter(tmp_path: Path) -> None:
    """A precision comparison varies an arm of the run, not the shipped file.

    Both fields are required together on this backend -- either one alone is refused at build time --
    so one parameter carries the pair rather than exposing two a `-p` could set into a refusal. The
    default arm is the float32 one this template has always emitted.
    """
    policies = {}
    for name, parameters in (("default", {}), ("mixed", {"SHARED": {"mixed_precision_type": "bfloat16"}})):
        path = tmp_path / f"{name}.py"
        KerasLearnerBuilder.from_path(LEARNERS / "ImageClassifier.yaml")(parameters=parameters)(path)
        learner_type = _load(path, f"precision_{name}").Learner
        policies[name] = (learner_type.MIXED_PRECISION, learner_type.MIXED_PRECISION_TYPE)

    assert policies["default"] == (False, None)
    assert policies["mixed"] == (True, "bfloat16")


IMAGE_CLASSIFIERS: dict[str, dict[str, Any]] = {
    "ImageClassifier": {"SHARED": {"mixed_precision_type": "bfloat16"}},
    "ImageClassifierShowcase": {},
}
"""The two templates of one image-classification recipe, each in the parametrization of its mixed arm.

The showcase declares the policy in the file; the base learner reaches the same one through the
parameter that carries both precision fields.
"""


@pytest.mark.skipif(BACKEND != "tensorflow", reason="Only the tensorflow backend's in_top_k refuses bfloat16.")
@pytest.mark.parametrize("case", list(IMAGE_CLASSIFIERS), ids=list(IMAGE_CLASSIFIERS))
def test_both_image_classifiers_evaluate_their_top_k_under_a_bfloat16_policy(
    case: str, tmp_path: Path, restore_policy: None
) -> None:
    """One recipe in two templates: the mixed arm has to run on both, or the pair has drifted apart.

    Under the policy the head emits bfloat16, and `tf.math.in_top_k` -- what
    `keras.metrics.sparse_top_k_categorical_accuracy` reaches on this backend -- refuses a bfloat16
    prediction outright, so a flow handing it the head's output dies on the run's first accuracy.
    The showcase carried the float32 cast that avoids this and the base learner did not, because
    nothing read the two side by side; this does, over the same model and the same batch. Both flows
    are exercised: each spells the cast separately, and a training run reaches the training one first.
    """
    keras.mixed_precision.set_global_policy("mixed_bfloat16")
    model = _model(tmp_path, "VisionTransformer", SHOWCASE_PARAMETERS, None)
    learner = _learner(tmp_path, case, IMAGE_CLASSIFIERS[case], model=model)

    assert (learner.MIXED_PRECISION, learner.MIXED_PRECISION_TYPE) == (True, "bfloat16")
    assert keras.backend.standardize_dtype(model(IMAGE_BATCH["image"])["cls"].dtype) == "bfloat16"

    trained = _floats(learner.training_step(**IMAGE_BATCH))
    evaluated = _floats(learner.inference_step(**IMAGE_BATCH))

    assert sorted(trained) == sorted(evaluated) == sorted(learner.outputs)
    assert all(np.isfinite(value) for value in (*trained.values(), *evaluated.values()))


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
