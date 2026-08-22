"""Behaviour tests for the Flax models and learners the templates under `cfg/flax` generate.

The templates are what a validation run trains, so the properties that matter -- a generator that
gives its input resolution back, a decoder that cannot read its own answer, a learner whose three
segments all move their own parameters -- are only decided once the emitted code runs. Every case
here shrinks the template to a few channels and a couple of blocks, and the generated classes are
module-scoped, so the whole file stays a CPU-seconds affair.
"""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType
from typing import Any

import jax
import jax.numpy as jnp
import pytest
from structcast.core import instantiator

from flax import nnx
from structcast_model.builders import schema
from structcast_model.builders.flax import FlaxBuilder, FlaxLearnerBuilder
from structcast_model.commands.utils import instantiate_object
from structcast_model.utils.base import load_any
from tests import CFG_DIR

MODELS = CFG_DIR / "flax" / "models"
LEARNERS = CFG_DIR / "flax" / "learners"
OTHERS = CFG_DIR / "flax" / "others"

CYCLE_GAN_MODELS = ("G_AB", "G_BA", "D_A", "D_B")
"""The models a CycleGAN learner is constructed with, in the order the template declares them."""

CYCLE_GAN_CRITERIA = ["loss_D_A", "loss_D_B", "loss_G", "loss_GAN", "loss_cycle", "loss_identity"]
"""The criteria both CycleGAN steps report, sorted."""

VIT_PARAMETERS = {"base": {"dim": 16, "heads": 2, "depth": 2, "image_size": 16, "patch_size": 8, "num_classes": 5}}
"""A four-patch, two-block Vision Transformer: the smallest one that still has a patch order."""


def _load(path: Path, name: str) -> ModuleType:
    """Load a generated module by file path, the way a configuration does: not by import name."""
    spec = spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _model_type(directory: Path, name: str, parameters: dict[str, Any]) -> Any:
    """Generate one shipped model template into *directory* and return its class."""
    path = directory / f"{name.lower()}.py"
    FlaxBuilder.from_path(MODELS / f"{name}.yaml")(parameters=parameters)(path)
    return _load(path, path.stem).Model


def _learner_type(directory: Path, name: str, parameters: dict[str, Any] | None = None) -> Any:
    """Generate one shipped learner template into *directory* and return its class."""
    path = directory / f"{name.lower()}_learner.py"
    FlaxLearnerBuilder.from_path(LEARNERS / f"{name}.yaml")(parameters=parameters)(path)
    return _load(path, path.stem).Learner


def _parameters(model: nnx.Module) -> list[jax.Array]:
    """Snapshot every parameter of *model*, so a later read can prove the optimizer moved it."""
    return [jnp.copy(leaf) for leaf in jax.tree.leaves(nnx.state(model, nnx.Param))]


def _moved(before: list[jax.Array], model: nnx.Module) -> bool:
    """Report whether any parameter of *model* differs from the snapshot in *before*."""
    return any(not jnp.array_equal(a, b) for a, b in zip(before, _parameters(model), strict=True))


def _named_parameters(model: nnx.Module) -> list[tuple[str, jax.Array]]:
    """Return every parameter as (dotted path, array), the path stripped of the nnx `.value` leaf.

    That trailing key is why an optimizer mask cannot be checked by reading the tree at
    construction time: the paths optax matches during an update carry no `.value`.
    """
    leaves, _ = jax.tree_util.tree_flatten_with_path(nnx.state(model, nnx.Param))
    return [
        (jax.tree_util.keystr(path, simple=True, separator=".").removesuffix(".value"), leaf) for path, leaf in leaves
    ]


def _evaluating(model: nnx.Module) -> nnx.Module:
    """Return the inference view of *model*, which is what a learner runs its validation against."""
    return nnx.view(model, raise_if_not_found=False, training=False, deterministic=True, use_running_average=True)


# ---------------------------------------------------------------------------
# CycleGAN
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def cycle_gan(tmp_path_factory: pytest.TempPathFactory) -> tuple[Any, Any, Any]:
    """Generate the two CycleGAN model templates and their learner once, shrunk to a few channels."""
    directory = tmp_path_factory.mktemp("generated")
    return (
        _model_type(directory, "CycleGAN_generator", {"DEFAULT": {"n_residual_blocks": 1, "init_features": 4}}),
        _model_type(directory, "CycleGAN_discriminator", {}),
        _learner_type(directory, "CycleGAN", {"DEFAULT": {"epochs": 4, "decay_epoch": 2, "steps_per_epoch": 2}}),
    )


def _cycle_gan_models(cycle_gan: tuple[Any, Any, Any]) -> dict[str, nnx.Module]:
    """Build one freshly seeded instance of each of the four models a CycleGAN learner takes."""
    generator, discriminator, _ = cycle_gan
    rngs = nnx.Rngs(0)
    return {name: (generator if name.startswith("G") else discriminator)(rngs=rngs) for name in CYCLE_GAN_MODELS}


def test_cycle_gan_generator_returns_the_resolution_it_was_given(cycle_gan: tuple[Any, Any, Any]) -> None:
    """The generator maps an image onto an image of the same shape, which the cycle loss assumes.

    The torch template keeps the resolution with an explicit `ReflectionPad2d` before every wide
    convolution and an `Upsample` that undoes each stride-2 block; this template folds both into
    the Flax layers -- `padding: REFLECT` and a repeat -- so a single off-by-one in that folding
    would show up here as a shape the cycle loss could not subtract.
    """
    out = cycle_gan[0](rngs=nnx.Rngs(0))(jnp.zeros((2, 32, 32, 3)))["out"]

    assert out.shape == (2, 32, 32, 3)
    # "flax.nnx.tanh" closes the generator, so its range is what bounds the identity and cycle losses.
    assert bool(jnp.all(jnp.abs(out) <= 1.0))


def test_cycle_gan_discriminator_scores_a_patch_grid_not_an_image(cycle_gan: tuple[Any, Any, Any]) -> None:
    """The 70x70 PatchGAN emits one score per overlapping patch, four strides down from the input.

    A 64-pixel input has to come back as a 4x4 grid of single-channel scores -- the same grid
    `cfg/torch/models/CycleGAN_discriminator.yaml` produces -- because that is what makes the
    least-squares GAN loss compare patches rather than whole images. The zero padding torch applies
    before the head convolution is folded into that convolution here, and getting the fold wrong
    would move the grid.
    """
    assert cycle_gan[1](rngs=nnx.Rngs(0))(jnp.zeros((2, 64, 64, 3)))["out"].shape == (2, 4, 4, 1)


def test_cycle_gan_learner_moves_all_four_models_over_its_three_segments(cycle_gan: tuple[Any, Any, Any]) -> None:
    """Every segment must move the models it owns, and every criterion must stay finite.

    Three optimizers over four models is the whole point of this template: a segment whose
    gradients never reached its own modules -- because the discriminators are read-only inside the
    generator flow, or because the generated images did not survive the closure -- would still run
    and still report a loss, so only the parameters prove it.
    """
    models = _cycle_gan_models(cycle_gan)
    learner = cycle_gan[2](**models)
    before = {name: _parameters(model) for name, model in models.items()}
    batch = {
        "real_A": jax.random.normal(jax.random.key(0), (2, 32, 32, 3)),
        "real_B": jax.random.normal(jax.random.key(1), (2, 32, 32, 3)),
    }

    criteria = learner.training_step(**batch)

    assert sorted(criteria) == CYCLE_GAN_CRITERIA
    assert all(bool(jnp.isfinite(value)) for value in criteria.values())
    assert all(_moved(before[name], model) for name, model in models.items())
    assert (learner.steps, learner.updates, learner.has_updated) == (1, 1, True)
    # Each rate is the first value of the "linear_decay_after" schedule the example file builds.
    assert all(rate == pytest.approx(2e-4) for rate in learner.learning_rates.values())


def test_cycle_gan_learner_reports_the_same_criteria_without_training(cycle_gan: tuple[Any, Any, Any]) -> None:
    """The inference flow has to compute every criterion the training flow does, and move nothing.

    Only the generator segment spells its `INFERENCE_FLOW` out -- the two discriminator segments
    fall back to their `FLOW` -- so a criterion dropped from that spelled-out copy would surface
    here as a missing validation number rather than as an error.
    """
    models = _cycle_gan_models(cycle_gan)
    learner = cycle_gan[2](**models)
    before = {name: _parameters(model) for name, model in models.items()}
    batch = {"real_A": jnp.zeros((2, 32, 32, 3)), "real_B": jnp.ones((2, 32, 32, 3))}

    criteria = learner.inference_step(**batch)

    assert sorted(criteria) == CYCLE_GAN_CRITERIA
    assert all(bool(jnp.isfinite(value)) for value in criteria.values())
    assert not any(_moved(before[name], model) for name, model in models.items())


# ---------------------------------------------------------------------------
# Small language model
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def small_language_model(tmp_path_factory: pytest.TempPathFactory) -> tuple[Any, Any]:
    """Generate the language model at two blocks of sixteen features, with its learner."""
    directory = tmp_path_factory.mktemp("generated")
    parameters = {"tiny": {"dim": 16, "heads": 2, "depth": 2, "vocab_size": 11}}
    return _model_type(directory, "SmallLanguageModel", parameters), _learner_type(directory, "SmallLanguageModel")


def test_small_language_model_never_attends_to_later_tokens(small_language_model: tuple[Any, Any]) -> None:
    """Rewrite the future of a sequence and every earlier logit must stay identical.

    Next-token training is only honest if a position cannot read the answer it is asked to predict,
    so this is the property the whole `is_causal=True` attention section exists for.
    """
    model = _evaluating(small_language_model[0](rngs=nnx.Rngs(0)))
    tokens = jnp.arange(16, dtype=jnp.int32)[None] % 11
    rewritten = tokens.at[:, 8:].set(jnp.arange(8, 16, dtype=jnp.int32)[::-1][None] % 11)

    logits, changed = model(tokens)["logits"], model(rewritten)["logits"]

    assert jnp.array_equal(logits[:, :8], changed[:, :8])
    assert not jnp.allclose(logits[:, 8:], changed[:, 8:], atol=1e-6)


def test_small_language_model_distinguishes_token_order(small_language_model: tuple[Any, Any]) -> None:
    """Swapping two earlier tokens must change a later position's logits.

    Attention alone is order-blind: without positions, a later position sums over the same set of
    tokens either way. The rotary embedding computed inside the attention section is the only thing
    carrying order here -- there is no learned position table -- so this failing would mean the
    rotation never reached the attention scores.
    """
    model = _evaluating(small_language_model[0](rngs=nnx.Rngs(0)))
    tokens = jnp.arange(16, dtype=jnp.int32)[None] % 11
    swapped = tokens.at[:, [2, 5]].set(tokens[:, [5, 2]])

    logits, reordered = model(tokens)["logits"], model(swapped)["logits"]

    assert not jnp.allclose(logits[:, 9], reordered[:, 9], atol=1e-6)
    # Position 1 attends only to tokens 0 and 1, which the swap left alone.
    assert jnp.allclose(logits[:, 1], reordered[:, 1], atol=1e-6)


def test_small_language_model_learner_lowers_the_loss_it_reports(small_language_model: tuple[Any, Any]) -> None:
    """Next-token cross entropy over the unflattened logits has to be the loss that is minimized.

    The torch template flattens the batch and time axes before its `CrossEntropyLoss`; this one
    hands `optax.softmax_cross_entropy_with_integer_labels` the (batch, time, vocabulary) logits
    directly. Both must average one prediction per position, and a misaligned reduction would give
    a loss that no longer falls when the parameters move.
    """
    model_type, learner_type = small_language_model
    learner = learner_type(model_type(rngs=nnx.Rngs(0)))
    before = _parameters(learner.models["model"])
    tokens = jnp.arange(8, dtype=jnp.int32).reshape(2, 4)
    batch = {"tokens": tokens, "targets": tokens + 1}

    first = learner.training_step(**batch)
    for _ in range(4):
        last = learner.training_step(**batch)

    assert list(first) == ["ce_loss"]
    assert bool(jnp.isfinite(last["ce_loss"]))
    assert float(last["ce_loss"]) < float(first["ce_loss"])
    assert _moved(before, learner.models["model"])


# ---------------------------------------------------------------------------
# Vision Transformer and the image classifier learner
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def vision_transformer(tmp_path_factory: pytest.TempPathFactory) -> Any:
    """Generate the Vision Transformer template once, at four patches and two blocks."""
    return _model_type(tmp_path_factory.mktemp("generated"), "VisionTransformer", VIT_PARAMETERS)


@pytest.fixture(scope="module")
def image_classifier(tmp_path_factory: pytest.TempPathFactory) -> Any:
    """Return a factory generating the image classifier learner for one set of parameters."""
    directory = tmp_path_factory.mktemp("generated")

    def _build(name: str, parameters: dict[str, Any]) -> Any:
        path = directory / f"{name}.py"
        FlaxLearnerBuilder.from_path(LEARNERS / "ImageClassifier.yaml")(parameters=parameters)(path)
        return _load(path, path.stem).Learner

    return _build


def _classification_batch() -> dict[str, jax.Array]:
    """One fixed batch of four images, so a loss that moves can only be the optimizer's doing."""
    return {"image": jax.random.normal(jax.random.key(0), (4, 16, 16, 3)), "label": jnp.asarray([0, 1, 2, 3])}


def test_vision_transformer_reads_a_shuffled_image_differently(vision_transformer: Any) -> None:
    """Swapping two patches must change the classification.

    Attention is order-blind, so only the learned position table can make a shuffled image look
    different; a template that added the table before prepending the class token, or none at all,
    would classify the two images identically.
    """
    model = _evaluating(vision_transformer(rngs=nnx.Rngs(0)))
    image = jax.random.normal(jax.random.key(0), (1, 16, 16, 3))
    # The four 8x8 patches are the quadrants; swapping the two top ones swaps patch 0 and patch 1.
    swapped = image.at[:, :8, :8].set(image[:, :8, 8:]).at[:, :8, 8:].set(image[:, :8, :8])

    assert model(image)["cls"].shape == (1, 5)
    assert not jnp.allclose(model(image)["cls"], model(swapped)["cls"], atol=1e-6)


def test_vision_transformer_drop_path_only_fires_while_training(vision_transformer: Any) -> None:
    """Stochastic depth has to stop at the inference view, or validation would measure noise.

    `flax.nnx.Dropout` reads the module's own `deterministic` flag, which is why the DropPath
    section of the template needs no `INFERENCE_FLOW` of its own -- and why only running both
    views proves the view a learner takes actually reaches that layer.
    """
    model = vision_transformer(rngs=nnx.Rngs(0))
    evaluating = _evaluating(model)
    # A batch wide enough that two independent drop masks over it cannot coincide by accident.
    image = jax.random.normal(jax.random.key(0), (64, 16, 16, 3))

    assert not jnp.allclose(model(image)["cls"], model(image)["cls"])
    assert jnp.array_equal(evaluating(image)["cls"], evaluating(image)["cls"])


def test_image_classifier_weight_decay_reaches_the_kernels_and_nothing_else(
    vision_transformer: Any, image_classifier: Any
) -> None:
    """The exemption regexes have to match the paths optax hands the mask during an update.

    A mask that matched nothing would still train, and would quietly decay the class token and the
    position table this recipe means to leave alone; the paths are also not the ones a tree read at
    construction shows, which carry a trailing `.value`. So the decay is measured one parameter at
    a time: two learners over identically seeded models, one identical batch, differing only in
    `weight_decay` -- the parameters that end up apart are exactly the ones that were decayed.
    """
    decayed = image_classifier("decayed", {"DEFAULT": {"weight_decay": 0.5}})(vision_transformer(rngs=nnx.Rngs(0)))
    undecayed = image_classifier("undecayed", {"DEFAULT": {"weight_decay": 0.0}})(vision_transformer(rngs=nnx.Rngs(0)))
    batch = _classification_batch()

    decayed.training_step(**batch)
    undecayed.training_step(**batch)

    after = _named_parameters(decayed.models["model"])
    reference = _named_parameters(undecayed.models["model"])
    apart = {path for (path, a), (_, b) in zip(after, reference, strict=True) if not jnp.allclose(a, b)}

    assert apart == {path for path, _ in after if path.endswith("kernel")}
    # The class token and the position table are the two-dimensional embeddings the torch side's
    # "never decay a parameter with at most one dimension" rule would not have spared either.
    assert {"cls_token_embedding.embedding", "position_embedding.embedding"} <= {p for p, _ in after} - apart


def test_image_classifier_learner_lowers_the_loss_of_the_vision_transformer(
    vision_transformer: Any, image_classifier: Any
) -> None:
    """One learner has to fit both shipped classifiers, so the Vision Transformer proves the harder half.

    Its class token and its position table are two-dimensional `flax.nnx.Embed` parameters, which
    the weight-decay mask exempts by name; a mask that matched nothing would still train, so what
    is asserted is the training itself -- a loss that falls on a fixed batch over a few steps.
    """
    model = vision_transformer(rngs=nnx.Rngs(0))
    learner = image_classifier("clipped", {"DEFAULT": {"clip_grad_norm": 1.0}})(model)
    before = _parameters(model)
    batch = _classification_batch()

    first = learner.training_step(**batch)
    for _ in range(4):
        last = learner.training_step(**batch)

    assert sorted(first) == ["acc1", "acc5", "ce_loss"]
    assert all(bool(jnp.isfinite(value)) for value in last.values())
    assert float(last["ce_loss"]) < float(first["ce_loss"])
    assert _moved(before, model)
    assert learner.learning_rates == {"optimizer": pytest.approx(1e-3)}


def test_image_classifier_learner_applies_only_on_the_accumulated_step(
    vision_transformer: Any, image_classifier: Any
) -> None:
    """`accumulate_gradients` has to reach the update gate, not just the transformation.

    The window is an `optax.MultiSteps` wrapping the whole chain, so it gates on the device; the
    generated step reads the applied count back across its own update. A window buried inside the
    chain would accumulate just the same but count every step as an update, which is what would
    make a per-update schedule or an update-counting callback fire twice too often.
    """
    learner = image_classifier("accumulating", {"DEFAULT": {"accumulate_gradients": 2}})(
        vision_transformer(rngs=nnx.Rngs(0))
    )
    batch = _classification_batch()

    gates = [(learner.training_step(**batch), learner.has_updated)[1] for _ in range(4)]

    assert gates == [False, True, False, True]
    assert learner.updates == 2


# ---------------------------------------------------------------------------
# The "others" templates: what --training-dataset and --compile are pointed at
# ---------------------------------------------------------------------------


def test_default_tfdata_template_renders_into_the_example_loader() -> None:
    """`scm format` plus an object pattern is the whole path from that template to a dataset.

    The template is the only thing tying `examples/flax/data.py` to a run, by file path and by
    field name, so a renamed field or a `name:` that renders as YAML null would only fail here --
    at the start of a training run that already built its models. The split follows `training`, so
    that no single forgotten parameter can point a validation loader at the training data.
    """
    # Instantiating the pattern imports the example, which imports TensorFlow: not installed in the
    # floor environment, where this integration is not what is being pinned.
    pytest.importorskip("tensorflow")
    parameters = {"DEFAULT": {"training": True, "batch_size": 4, "image_size": 16}}
    rendered: Any = schema.Template.from_path(OTHERS / "default_tfdata.yaml")(parameters).model_dump(mode="json")

    loader = instantiate_object(rendered)

    assert type(loader).__name__ == "TFDataLoader"
    assert (loader.name, loader.is_training, loader.split, loader.batch_size) == ("", True, "train", 4)
    assert next(iter(loader()))["image"].shape == (4, 16, 16, 3)
    validation: Any = schema.Template.from_path(OTHERS / "default_tfdata.yaml")({}).model_dump(mode="json")
    assert instantiate_object(validation).split == "validation"


def test_compile_default_template_only_carries_arguments_nnx_jit_accepts() -> None:
    """`--compile <file>` merges this template into the jit call, minus the contract arguments.

    The CLI drops `static_argnames`, `static_argnums`, `donate_argnames` and `donate_argnums`
    because they are the generated step's contract, so naming one here would be silently ignored;
    everything else is passed straight through and has to be a real `flax.nnx.jit` keyword.
    """
    compile_kw = instantiator.instantiate(load_any(OTHERS / "compile_default.yaml"))

    assert not {"static_argnames", "static_argnums", "donate_argnames", "donate_argnums"} & set(compile_kw)
    assert float(nnx.jit(lambda value: value + 1, **compile_kw)(jnp.asarray(1.0))) == 2.0
