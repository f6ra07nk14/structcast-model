"""Behaviour tests for the Flax models and learners the templates under `cfg/flax` generate.

The templates are what a validation run trains, so the properties that matter -- a generator that
gives its input resolution back, a decoder that cannot read its own answer, a learner whose three
segments all move their own parameters -- are only decided once the emitted code runs. Every case
here shrinks the template to a few channels and a couple of blocks, and the generated classes are
module-scoped, so the whole file stays a CPU-seconds affair.
"""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from re import MULTILINE, findall as re_findall, search as re_search
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
from structcast_model.flax.layers import GradientCheckpointingModule
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

CONVNEXT_PARAMETERS: dict[str, Any] = {
    "SHARED": {"num_classes": 5},
    "atto": {"dims": [4, 8, 8, 16], "depths": [1, 1, 1, 1]},
}
"""A four-stage ConvNeXt V2 one block deep: every layer type the template builds, built fast."""


def _load(path: Path, name: str) -> ModuleType:
    """Load a generated module by file path, the way a configuration does: not by import name."""
    spec = spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _render(path: Path, parameters: dict[str, Any]) -> Any:
    """Render a template the way `scm format` does, returning the raw mapping it produces."""
    return schema.Template.from_path(path)(parameters).model_dump(mode="json")


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
    # Each rate is the first value of the "optax.linear_schedule" the template builds.
    assert all(rate == pytest.approx(2e-4) for rate in learner.learning_rates.values())


def _torch_lambda(epoch: int, *, epochs: int, decay_epoch: int, offset: int = 0) -> float:
    """The `LambdaLR` of `cfg/torch/learners/CycleGAN.yaml`, written out for comparison."""
    return 1.0 - max(0, epoch + offset - decay_epoch) / (epochs - decay_epoch)


def test_cycle_gan_schedule_matches_the_torch_lambda_at_every_epoch_boundary() -> None:
    """The recipe is the schedule, so the rate has to land where the torch template puts it.

    optax counts optimizer applies where torch counts epochs, and `steps_per_epoch` is the whole
    conversion -- getting it wrong decays at a different pace than the reference implementation
    this template mirrors. The two curves are not identical: torch holds one rate for a whole epoch
    and this one falls continuously, so they are compared where they must agree, at the boundaries.
    """
    steps_per_epoch = 5
    rendered = _render(LEARNERS / "CycleGAN.yaml", {"DEFAULT": {"steps_per_epoch": steps_per_epoch}})
    # The schedule is a nested pattern under the first segment's optimizer; instantiating it alone
    # keeps this a schedule test rather than a training one.
    schedule = instantiate_object(rendered["LEARNERS"][0]["OPTIMIZER"][2]["_bind_"]["tx"][2]["_call_"]["learning_rate"])

    for epoch in (0, 50, 100, 150, 199):
        expected = 2e-4 * _torch_lambda(epoch, epochs=200, decay_epoch=100)
        assert float(schedule(epoch * steps_per_epoch)) == pytest.approx(expected, rel=1e-5)


def test_cycle_gan_schedule_starts_further_down_the_ramp_for_a_resumed_run() -> None:
    """`offset` is what makes a resumed run continue the ramp instead of restarting it."""
    parameters = {"DEFAULT": {"epochs": 4, "decay_epoch": 2, "steps_per_epoch": 1, "offset": 1, "lr": 1.0}}
    rendered = _render(LEARNERS / "CycleGAN.yaml", parameters)
    schedule = instantiate_object(rendered["LEARNERS"][0]["OPTIMIZER"][2]["_bind_"]["tx"][2]["_call_"]["learning_rate"])

    assert [float(schedule(step)) for step in range(4)] == pytest.approx([1.0, 1.0, 0.5, 0.0])


def test_cycle_gan_refuses_a_decay_that_starts_at_or_after_the_end_of_the_run() -> None:
    """A ramp with no length would silently render a constant schedule instead of a decaying one.

    optax answers a non-positive `transition_steps` with a constant schedule and a log line, which
    is the kind of misconfiguration that only shows up as a run that never converges; the template
    rejects it while it is still a configuration.
    """
    with pytest.raises(ValueError, match="must be greater than decay_epoch"):
        _render(LEARNERS / "CycleGAN.yaml", {"DEFAULT": {"decay_epoch": 200}})


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


def _shipped_rules(name: str) -> list[tuple[str, str]]:
    """The (parameter-path regex, tactic) table one shipped strategy template binds."""
    bound = load_any(str(CFG_DIR / "flax" / "strategies" / name))[2]["_bind_"]
    return [(pattern, tactic) for pattern, tactic in bound["rules"]]


def test_the_shipped_tensor_parallel_rules_name_this_models_parameters(vision_transformer: Any) -> None:
    """`cfg/flax/strategies/tp.yaml` is written for this template, so its rules have to match it.

    A rule matching no parameter is refused at wrap time -- loud, but only once the run started --
    and a table that matched only half of a column/row pair would train a wrong answer, so the four
    layers and their tactics are pinned here. `fsdp_tp.yaml` carries the same four in front of its
    catch-all `fsdp` rule; the two drifting apart is the other half of the same defect.
    """
    rules = _shipped_rules("tp.yaml")
    assert _shipped_rules("fsdp_tp.yaml") == [*rules, (".*", "fsdp")]

    model = vision_transformer(rngs=nnx.Rngs(0))
    names = [
        jax.tree_util.keystr(path, simple=True, separator=".")
        for path, _ in jax.tree_util.tree_flatten_with_path(nnx.to_pure_dict(nnx.state(model, nnx.Param)))[0]
    ]
    matched: dict[str, set[str]] = {"column": set(), "row": set()}
    for pattern, tactic in rules:
        matched[tactic] |= {name for name in names if re_search(pattern, name)}

    assert [tactic for _, tactic in rules] == ["column", "row", "column", "row"]
    assert sorted(matched["column"]) == sorted(
        f"backbone.block{i}.{layer}.{leaf}"
        for i in range(2)
        for layer in ("self_attention.qkv_proj", "linear")
        for leaf in ("bias", "kernel")
    )
    assert sorted(matched["row"]) == sorted(
        f"backbone.block{i}.{layer}.{leaf}"
        for i in range(2)
        for layer in ("self_attention.out_proj", "linear_1")
        for leaf in ("bias", "kernel")
    )


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


SHOWCASE_PARAMETERS: dict[str, dict[str, Any]] = {
    "base": VIT_PARAMETERS["base"],
    "SHARED": {"drop_path_rate": 0.0},
}
"""The showcase Vision Transformer, with the stochastic depth of the recipe switched off.

`gradient_checkpointing` is the one parameter left out: the two builds below set it either way, and
the point of the equality assertions is that nothing else differs between them.
"""


def test_the_showcase_pair_runs_every_feature_this_backend_has_in_one_step(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """Checkpointing, accumulation and the EMA have to hold together, not just one by one.

    Each is pinned on its own elsewhere; what only a combined run can answer is whether they
    compose. They interact: the average follows the `MultiSteps` gate rather than the step count,
    and the module the average shadows is the rematerialized one. A rematerialized module must also
    stay the same module -- its variable paths are what a sharding rule and a checkpoint address --
    so the paths and the forward pass are asserted against the same model built without it.

    Mixed precision is left out of this pair on purpose: it is a `dtype` on the model's layers on
    this backend rather than anything the learner carries, so it is pinned on the template next to
    the other model-side knob instead of here.
    """

    def _build(remat: bool) -> Any:
        shared = {**SHOWCASE_PARAMETERS["SHARED"], "gradient_checkpointing": remat}
        directory = tmp_path_factory.mktemp("remat" if remat else "plain")
        model_type = _model_type(directory, "VisionTransformer", {**SHOWCASE_PARAMETERS, "SHARED": shared})
        return model_type(rngs=nnx.Rngs(0))

    checkpointed, plain = _build(True), _build(False)
    path = tmp_path_factory.mktemp("learner") / "showcase.py"
    FlaxLearnerBuilder.from_path(LEARNERS / "ImageClassifierShowcase.yaml")(
        parameters={"DEFAULT": {"accumulate_gradients": 2}}
    )(path)
    learner = _load(path, path.stem).Learner(checkpointed)
    batch = _classification_batch()

    assert isinstance(checkpointed.backbone.block0, GradientCheckpointingModule)
    assert checkpointed.backbone.block0.gradient_checkpointing is True
    assert not isinstance(plain.backbone.block0, GradientCheckpointingModule)
    assert [name for name, _ in _named_parameters(checkpointed)] == [name for name, _ in _named_parameters(plain)]
    assert jnp.allclose(checkpointed(batch["image"], training=True)["cls"], plain(batch["image"], training=True)["cls"])

    averages, gates = [], []
    for _ in range(4):
        learner.training_step(**batch)
        gates.append(learner.has_updated)
        averages.append(_parameters(learner.ema_model))

    blended = [
        any(not jnp.array_equal(a, b) for a, b in zip(x, y, strict=True))
        for x, y in zip(averages, averages[1:], strict=False)
    ]

    assert gates == [False, True, False, True]
    assert blended == [True, False, True]  # one blend per Update, none on the micro-steps between
    assert sorted(learner.models) == ["ema_model", "model"]
    assert bool(jnp.isfinite(learner.inference_step(**batch)["ce_loss"]))


def test_the_showcase_average_can_be_left_out_by_parameter(vision_transformer: Any, tmp_path: Path) -> None:
    """The `ema` parameter exists for the torch twin, whose FSDP2 runs refuse an average outright.

    Nothing on this backend refuses one, but the three templates are read side by side and a knob
    spelled differently on each is a knob nobody trusts. What it has to do here is drop the field and
    the shadow it emits, and leave the inference flow validating over `model` -- a flow still naming
    `ema_model` would fail on a name the learner no longer defines.
    """
    paths = {}
    for name, parameters in (("averaged", {}), ("plain", {"SHARED": {"ema": False}})):
        paths[name] = tmp_path / f"{name}.py"
        FlaxLearnerBuilder.from_path(LEARNERS / "ImageClassifierShowcase.yaml")(parameters=parameters)(paths[name])
    learner = _load(paths["plain"], "plain").Learner(vision_transformer(rngs=nnx.Rngs(0)))

    assert "flax.nnx.EMA" in paths["averaged"].read_text()
    assert "flax.nnx.EMA" not in paths["plain"].read_text()
    assert sorted(learner.models) == ["model"]
    assert bool(jnp.isfinite(learner.inference_step(**_classification_batch())["ce_loss"]))


# ---------------------------------------------------------------------------
# The activation the three model templates share
# ---------------------------------------------------------------------------


GELU_MODELS: dict[str, dict[str, Any]] = {
    "ConvNeXtV2": CONVNEXT_PARAMETERS,
    "SmallLanguageModel": {"tiny": {"dim": 16, "heads": 2, "depth": 1, "vocab_size": 11}},
    "VisionTransformer": VIT_PARAMETERS,
}
"""Every shipped model template whose MLP activates through `flax.nnx.gelu`, shrunk to build fast."""


@pytest.mark.parametrize("name", list(GELU_MODELS), ids=list(GELU_MODELS))
def test_the_model_templates_activate_through_the_exact_gelu(name: str, tmp_path: Path) -> None:
    """`flax.nnx.gelu` defaults to the tanh approximation; `torch.nn.GELU` and Keras' do not.

    Left bare, these templates would compute a different MLP from their torch and Keras twins on
    the same weights -- a divergence neither a shape nor a falling loss can see. The probe is what
    makes the keyword worth reading back: the two forms agree to about three decimals, so a test
    that only found `approximate=False` in the emitted module would not say it changed any number.
    It is read off the constant the binding is hoisted to, which is the one object every block of
    every instance activates through.
    """
    probe = jnp.linspace(-3.0, 3.0, 7)
    assert not bool(jnp.allclose(jax.nn.gelu(probe, approximate=False), jax.nn.gelu(probe), atol=1e-5))

    path = tmp_path / f"{name.lower()}.py"
    FlaxBuilder.from_path(MODELS / f"{name}.yaml")(parameters=GELU_MODELS[name])(path)

    code = path.read_text()
    constants = re_findall(r"^(_bound_\w+) = .*approximate=False.*$", code, flags=MULTILINE)
    assert len(constants) == 1, "the exact GELU must be one module-level constant, shared by every block"
    assert code.count("lambda") == 1, "a bound callable survived as a per-instance closure"
    assert re_findall(rf"^\s+self\.\w+ = {constants[0]}$", code, flags=MULTILINE), "no block activates through it"
    activated = getattr(_load(path, path.stem), constants[0])(probe)
    assert bool(jnp.allclose(activated, jax.nn.gelu(probe, approximate=False), atol=1e-6))


def test_two_instances_of_a_bound_activation_share_one_traced_step(tmp_path: Path) -> None:
    """Two instances of one generated class must share a graphdef, and a jit trace with it.

    A callable attribute is part of the static half nnx compares, and neither a lambda nor a
    `functools.partial` defines `__eq__`: built per instance, the bound activation would give every
    instance its own graphdef, so an EMA shadow or a second model would pay a full recompile of a
    step it should have shared. Counting the traces is what says it -- each one is a compile.
    """
    model_type = _model_type(tmp_path, "SmallLanguageModel", GELU_MODELS["SmallLanguageModel"])
    first, second = model_type(rngs=nnx.Rngs(0)), model_type(rngs=nnx.Rngs(1))
    traces: list[int] = []

    @nnx.jit
    def _logits(model: Any, tokens: jax.Array) -> jax.Array:
        traces.append(1)
        return model(tokens)["logits"]

    tokens = jnp.arange(8, dtype=jnp.int32)[None] % 11
    _logits(first, tokens)
    _logits(second, tokens)

    assert nnx.graphdef(first) == nnx.graphdef(second)
    assert len(traces) == 1


# ---------------------------------------------------------------------------
# The "others" templates: what --training-dataset and --compile are pointed at
# ---------------------------------------------------------------------------


def test_default_tfdata_template_renders_into_the_example_loader(tmp_path: Path) -> None:
    """`scm format` plus an object pattern is the whole path from that template to a dataset.

    The template is the only thing tying `examples/flax/data.py` to a run, by file path and by
    field name, so a renamed field would only fail here -- at the start of a training run that
    already built its models. The split follows `training`, so that no single forgotten parameter
    can point a validation loader at the training data; it also names the directory under
    `data_dir`, which is what lets one host tree serve a run at ImageNet's size; and a source is
    required either way, so that nothing silently trains on a placeholder.
    """
    # Instantiating the pattern imports the example, which imports TensorFlow: not installed in the
    # floor environment, where this integration is not what is being pinned.
    pytest.importorskip("tensorflow")
    parameters = {"DEFAULT": {"dataset": "cifar10", "training": True, "batch_size": 4, "image_size": 16}}
    rendered: Any = _render(OTHERS / "default_tfdata.yaml", parameters)

    loader = instantiate_object(rendered)

    assert type(loader).__name__ == "TFDataLoader"
    assert (loader.name, loader.is_training, loader.split, loader.batch_size) == ("cifar10", True, "train", 4)
    assert loader.image_size == 16
    assert instantiate_object(_render(OTHERS / "default_tfdata.yaml", {"DEFAULT": {"dataset": "cifar10"}})).split == (
        "validation"
    )
    # The directory form: the split is appended to the root, so one render per split covers a tree.
    (tmp_path / "train").mkdir()
    directory = instantiate_object(
        _render(OTHERS / "default_tfdata.yaml", {"DEFAULT": {"data_dir": str(tmp_path), "training": True}})
    )
    assert directory.name == tmp_path / "train"
    with pytest.raises(ValueError, match="A source is required"):
        _render(OTHERS / "default_tfdata.yaml", {})


def test_compile_default_template_only_carries_arguments_nnx_jit_accepts() -> None:
    """`--compile <file>` merges this template into the jit call, minus the contract arguments.

    The CLI drops `static_argnames`, `static_argnums`, `donate_argnames` and `donate_argnums`
    because they are the generated step's contract, so naming one here would be silently ignored;
    everything else is passed straight through and has to be a real `flax.nnx.jit` keyword.
    """
    compile_kw = instantiator.instantiate(load_any(OTHERS / "compile_default.yaml"))

    assert not {"static_argnames", "static_argnums", "donate_argnames", "donate_argnums"} & set(compile_kw)
    assert float(nnx.jit(lambda value: value + 1, **compile_kw)(jnp.asarray(1.0))) == 2.0


def _vit_script(parameters: dict[str, Any], directory: Path) -> str:
    """Emit the Vision Transformer template for *parameters* and return the generated module text."""
    path = directory / "vit.py"
    FlaxBuilder.from_path(MODELS / "VisionTransformer.yaml")(parameters=parameters)(path)
    return path.read_text()


def test_vision_transformer_emits_the_same_module_until_a_dtype_is_asked_for(tmp_path: Path) -> None:
    """The precision knob must be invisible to every run that does not set it.

    A `-p` knob threaded into eleven layer patterns is exactly the kind of change that shifts a
    keyword or a default nobody asked to move, and the generated module is what a run trains and
    what a checkpoint is keyed against. Byte equality against the same template with the knob
    absent is the only assertion that catches all of that at once -- the same discipline the
    `gradient_checkpointing` pair follows, which is set both ways here so the two knobs are proven
    independent.
    """
    absent = _vit_script(VIT_PARAMETERS, tmp_path / "absent")
    explicit_none = _vit_script({**VIT_PARAMETERS, "SHARED": {"dtype": None}}, tmp_path / "none")
    remat: dict[str, Any] = {**VIT_PARAMETERS, "SHARED": {"gradient_checkpointing": True}}
    checkpointed = _vit_script(remat, tmp_path / "remat")
    checkpointed_none = _vit_script({**remat, "SHARED": {**remat["SHARED"], "dtype": None}}, tmp_path / "remat-none")

    assert absent == explicit_none
    assert checkpointed == checkpointed_none
    # Layer constructions only: the class-token index is a "jax.numpy.zeros(..., dtype=...)"
    # expression that has always been there, so a bare "dtype=" would not mean what it looks like.
    assert not [line for line in absent.splitlines() if line.lstrip().startswith("self.") and "dtype=" in line]
    assert "gradient_checkpointing = True" in checkpointed


def test_vision_transformer_computes_in_bfloat16_over_float32_weights(tmp_path: Path) -> None:
    """`-p "base: {dtype: bfloat16}"` is the flax-native counterpart of torch bf16 autocast.

    Mixed, not pure: only `dtype` is threaded, so `param_dtype` stays float32 and the weights -- and
    with them the gradients and the optax moments -- keep an fp32 master copy while the matmuls and
    the normalizations run in bf16. A template that also narrowed `param_dtype` would train a
    different, and over a long run a worse, model; one that narrowed neither would report bf16
    nowhere. Both halves are therefore asserted, and on every parameterized layer type the template
    builds, since a keyword threaded onto four of five is the failure that looks like it worked.
    """
    parameters = {"base": {**VIT_PARAMETERS["base"], "dtype": "bfloat16"}}
    script = _vit_script(parameters, tmp_path / "bf16")
    model = _model_type(tmp_path, "VisionTransformer", parameters)(rngs=nnx.Rngs(0))

    for layer in ("Conv", "Embed", "LayerNorm", "Linear"):
        assert f"{layer}(" in script
        assert all("dtype='bfloat16'" in line for line in script.splitlines() if f"= {layer}(" in line)
    assert {str(leaf.dtype) for leaf in jax.tree.leaves(nnx.state(model, nnx.Param))} == {"float32"}
    assert model(jnp.zeros((2, 16, 16, 3)))["cls"].dtype == jnp.bfloat16
    # Why the template names the size group rather than "SHARED": the root flow renders in that
    # group's scope, and a command-line "SHARED" is merged onto the default group alone. Threaded
    # down to each section as a parameter, that near miss is a no-op; read from the shared scope it
    # used to leave the patch embedding, the two tables, the final norm and the head at float32
    # while the blocks narrowed -- a half-precision model that trains, reports nothing wrong and is
    # not the one that was asked for. The shared form still has to work alongside the group one,
    # because that is the incantation the template used to document. Both are rendered at the
    # shipped size on purpose: naming a size group for anything at all is what pulls the shared half
    # into it, so a shrunk render would hide the trap.
    assert _vit_script({"SHARED": {"dtype": "bfloat16"}}, tmp_path / "shared-only") == _vit_script(
        {}, tmp_path / "no-knob"
    )
    assert _vit_script({"base": {"dtype": "bfloat16"}}, tmp_path / "group") == _vit_script(
        {"SHARED": {"dtype": "bfloat16"}, "base": {"dtype": "bfloat16"}}, tmp_path / "group-and-shared"
    )


def _conv_next_script(parameters: dict[str, Any], directory: Path) -> str:
    """Emit the ConvNeXt V2 template for *parameters* and return the generated module text."""
    path = directory / "convnext.py"
    FlaxBuilder.from_path(MODELS / "ConvNeXtV2.yaml")(parameters=parameters)(path)
    return path.read_text()


def _typed_layers(script: str) -> list[tuple[str, bool]]:
    """Every parameterized layer the module constructs, paired with whether it was given a `dtype`."""
    return [
        (match.group(1), "dtype=" in line)
        for line in script.splitlines()
        if (match := re_search(r"= (Conv|LayerNorm|Linear|GlobalResponseNorm)\(", line))
    ]


def test_conv_next_v2_narrows_every_layer_from_the_one_size_group_override(tmp_path: Path) -> None:
    """`-p "atto: {dtype: bfloat16}"` is the whole incantation, and it has to reach every layer.

    Mixed, not pure: only `dtype` is threaded, so `param_dtype` stays float32 and the weights -- and
    with them the gradients and the optax moments -- keep an fp32 master copy while the convolutions,
    the matmuls and the normalizations run in bf16, which is what makes this comparable to the torch
    twin's `MIXED_PRECISION_TYPE: bfloat16` autocast. The count is what makes it worth asserting:
    the template reaches its layers through four sections, and a knob threaded into three of them
    would narrow most of the model, train, converge and be a different model from the one asked for.
    So every constructed layer is read back, not a sample of them, and the two renders are compared
    site for site so that a `dtype` bought by dropping a layer would not pass either.
    """
    typed = {**CONVNEXT_PARAMETERS, "atto": {**CONVNEXT_PARAMETERS["atto"], "dtype": "bfloat16"}}
    narrowed = _typed_layers(_conv_next_script(typed, tmp_path / "bf16"))
    absent = _typed_layers(_conv_next_script(CONVNEXT_PARAMETERS, tmp_path / "fp32"))

    assert [name for name, _ in narrowed] == [name for name, _ in absent]
    assert {"Conv", "LayerNorm", "Linear", "GlobalResponseNorm"} == {name for name, _ in narrowed}
    assert [name for name, typed_here in narrowed if not typed_here] == []
    assert [name for name, typed_here in absent if typed_here] == []

    model = _model_type(tmp_path, "ConvNeXtV2", typed)(rngs=nnx.Rngs(0))

    assert {str(leaf.dtype) for leaf in jax.tree.leaves(nnx.state(model, nnx.Param))} == {"float32"}
    assert model(jnp.zeros((2, 32, 32, 3)))["cls"].dtype == jnp.bfloat16


def test_conv_next_v2_leaves_the_module_untouched_for_a_shared_only_override(tmp_path: Path) -> None:
    """The knob is all-or-nothing on purpose: a size group owns it, so "SHARED" must not half apply.

    A command-line "SHARED" is merged onto the default group alone, so on its own it never reaches
    the size group the flows select. The template therefore hands the group's value down to each
    section as a parameter rather than reading a shared one, which turns that near miss into a
    no-op: byte equality with the knob absent is what says the module a run trains is never left
    partly narrowed. Rendered at the shipped size, because naming a size group for anything at all
    is what pulls the shared half into it.
    """
    assert _conv_next_script({"SHARED": {"dtype": "bfloat16"}}, tmp_path / "shared") == _conv_next_script(
        {}, tmp_path / "absent"
    )


def test_conv_next_v2_drops_whole_samples_when_stochastic_depth_is_on(tmp_path: Path) -> None:
    """Stochastic depth drops a sample's residual branch whole, so its mask spans every axis but the batch.

    The activations here are NHWC, so that is dims 1, 2 and 3; naming an axis they do not have
    makes `flax.nnx.Dropout` raise on the first block whose rate is above zero, which is every
    block of the recipe this template ships for. The rate is therefore what is under test rather
    than the emitted keyword: nothing below runs at all under a mask that indexes past the last
    axis.

    Repeated rows are what say the drop is per sample. Three of the four blocks draw a mask, so one
    image repeated across the batch can only come back as the eight outcomes those masks spell out,
    where an element-wise mask would give all sixty-four rows a value of their own.
    """
    parameters = {**CONVNEXT_PARAMETERS, "atto": {**CONVNEXT_PARAMETERS["atto"], "drop_path_rate": 0.9}}
    model = _model_type(tmp_path, "ConvNeXtV2", parameters)(rngs=nnx.Rngs(0))
    image = jnp.broadcast_to(jax.random.normal(jax.random.key(0), (1, 16, 16, 3)), (64, 16, 16, 3))

    dropped = model(image)["cls"]

    assert not jnp.allclose(dropped, model(image)["cls"])
    assert jnp.array_equal(_evaluating(model)(image)["cls"], _evaluating(model)(image)["cls"])
    assert len({tuple(row.tolist()) for row in dropped}) <= 8
