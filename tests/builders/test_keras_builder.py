"""API-level tests for keras builder classes."""

import ast
from pathlib import Path
from typing import Any

import pytest
from structcast.core.exceptions import SpecError

from structcast_model.builders.keras import (
    KerasBuilder,
    KerasLayerIntermediate,
    KerasLearnerBuilder,
)
from structcast_model.utils.base import load_any
from tests import CFG_DIR, FIXTURES_DIR


def test_keras_layer_intermediate_generates_call_method_without_inference_flow() -> None:
    """Generate call method without if/else when INFERENCE_FLOW is absent."""
    script = KerasLayerIntermediate(
        classname="Unit",
        imports={},
        inputs=["x"],
        outputs=["y"],
        layers={},
        flow=[("x", "y", None)],
        inference_flow=[],
        structured_output=False,
    )._get_layer_script("Unit", [])
    assert "class Unit(keras.layers.Layer):" in script
    assert "def call(self, x, *, training = None, **kwargs):" in script
    assert "if training:" not in script
    assert "return y" in script


def test_keras_layer_intermediate_generates_training_inference_branches() -> None:
    """Generate if/else branches when INFERENCE_FLOW is present."""
    script = KerasLayerIntermediate(
        classname="Unit",
        imports={},
        inputs=["x"],
        outputs=["y"],
        layers={},
        flow=[("x", "y", None)],
        inference_flow=[("x", "y", None)],
        structured_output=False,
    )._get_layer_script("Unit", ["proj = keras.layers.Dense(units=4)"])
    assert "class Unit(keras.layers.Layer):" in script
    assert "if training:" in script
    assert "else:" in script
    assert "self.proj = keras.layers.Dense(units=4)" in script
    assert "return y" in script


def test_keras_layer_intermediate_propagates_training_to_sublayer_calls() -> None:
    """Sublayer calls include training=training keyword argument."""
    intermediate = KerasLayerIntermediate(
        classname="Unit",
        imports={},
        inputs=["x"],
        outputs=["y"],
        layers={"dense": "keras.layers.Dense(units=4)"},
        flow=[("x", "y", "dense")],
        inference_flow=[],
        structured_output=False,
    )
    script = intermediate._get_layer_script("Unit", ["dense = keras.layers.Dense(units=4)"])
    assert "self.dense(x, training=training)" in script


def test_keras_layer_intermediate_uses_input_output_names_attributes() -> None:
    """Use input_names/output_names to avoid clashing with Keras built-ins."""
    script = KerasLayerIntermediate(
        classname="Unit",
        imports={},
        inputs=["image"],
        outputs=["cls"],
        layers={},
        flow=[("image", "cls", None)],
        inference_flow=[],
        structured_output=False,
    )._get_layer_script("Unit", [])
    assert "self.input_names = ['image']" in script
    assert "self.output_names = ['cls']" in script


def test_keras_layer_intermediate_emits_input_shapes_literal() -> None:
    """Emit the declared input shapes as a literal so the built model can create its own dummy inputs."""
    script = KerasLayerIntermediate(
        classname="Unit",
        imports={},
        inputs=["image"],
        input_shapes={"image": (3, 224, 224)},
        outputs=["cls"],
        layers={},
        flow=[("image", "cls", None)],
        inference_flow=[],
        structured_output=False,
    )._get_layer_script("Unit", [])
    assert "self.input_shapes = {'image': (3, 224, 224)}" in script


def test_keras_builder_builds_intermediate_and_scripts() -> None:
    """Build a minimal Keras model and render Python script content."""
    raw = {
        "INPUTS": ["x"],
        "OUTPUTS": ["y"],
        "FLOW": [["x", "y", {"_obj_": [["_addr_", "keras.layers.Dense"], ["_call_", {"units": 4}]]}]],
    }
    built = KerasBuilder(raw=raw)(classname="TinyNet")
    assert built.classname == "TinyNet"
    assert "keras" in built.collected_imports
    assert len(built.scripts) == 1
    assert "class TinyNet(keras.layers.Layer):" in built.scripts[0]


def test_keras_builder_structured_output_returns_dict() -> None:
    """Build a model with structured output and verify dict return."""
    raw = {
        "INPUTS": ["x"],
        "OUTPUTS": ["y"],
        "STRUCTURED_OUTPUT": True,
        "FLOW": [["x", "y", {"_obj_": [["_addr_", "keras.layers.Dense"], ["_call_", {"units": 4}]]}]],
    }
    built = KerasBuilder(raw=raw)(classname="TinyNet", forced_structured_output=True)
    script = built.scripts[0]
    assert "return {'y': y}" in script


def test_keras_builder_cfg_convnext_builds_expected_topology() -> None:
    """Build Keras ConvNeXt model from cfg and check key topology outputs."""
    parameters = {"DEFAULT": {"backbone": "tiny", "num_classes": 10}}
    builder = KerasBuilder.from_path(CFG_DIR / "keras" / "models" / "ConvNeXtV2.yaml")
    built = builder(parameters=parameters, classname="ConvNeXtKerasTiny")
    assert built.classname == "ConvNeXtKerasTiny"
    assert built.structured_output is True
    assert built.outputs == ["cls"]
    assert "backbone" in built.layers
    assert "head" in built.layers
    assert len(built.scripts) > 0
    assert "class ConvNeXtKerasTiny(keras.layers.Layer):" in built.scripts[-1]


@pytest.mark.parametrize("backbone", ["tiny"])
def test_keras_builder_cfg_convnext_sublayer_builds_backbone(backbone: str) -> None:
    """Build Backbone sublayer from Keras ConvNeXt cfg."""
    parameters = {"DEFAULT": {"backbone": backbone}}
    builder = KerasBuilder.from_path(CFG_DIR / "keras" / "models" / "ConvNeXtV2.yaml")
    built = builder(parameters=parameters, classname="Backbone", user_defined_layer="Backbone")
    assert built.classname == "Backbone"
    assert built.structured_output is True
    assert "stem" in built.layers
    assert any("downsample" in k for k in built.layers)
    assert len(built.scripts) > 0


MODEL_TEMPLATES = {
    "ConvNeXtV2": ({"SHARED": {"num_classes": 10}, "atto": {"dims": [4, 8, 8, 16], "depths": [1, 1, 1, 1]}}, ["cls"]),
    "CycleGAN_generator": ({"DEFAULT": {"n_residual_blocks": 1, "init_features": 4}}, ["out"]),
    "CycleGAN_discriminator": ({}, ["out"]),
    "SmallLanguageModel": ({"tiny": {"dim": 16, "heads": 2, "depth": 1}}, ["logits"]),
    "VisionTransformer": ({"base": {"dim": 16, "heads": 2, "depth": 1}}, ["cls"]),
}
"""Every shipped Keras model template, with parameters small enough to render quickly."""


@pytest.mark.parametrize(("name", "parameters", "outputs"), [(n, p, o) for n, (p, o) in MODEL_TEMPLATES.items()])
def test_every_shipped_keras_model_template_renders_a_layer_class(
    name: str, parameters: dict[str, Any], outputs: list[str]
) -> None:
    """A template nobody can build is not a template: parsing it proves nothing about its flow.

    Rendering is where a layer address that does not exist, a jinja expression over a missing
    parameter or a flow line reading an unbound name is caught, so every shipped template is
    rendered here rather than only the one the older tests happen to use.
    """
    built = KerasBuilder.from_path(CFG_DIR / "keras" / "models" / f"{name}.yaml")(parameters=parameters)

    assert built.outputs == outputs
    # Structured, as "scm keras create model" forces by default: the learner templates below read
    # the model output as a mapping, so a template emitting a bare tensor would break them.
    assert built.structured_output is True
    assert "class Model(keras.layers.Layer):" in built.scripts[-1]


LEARNER_YAML = FIXTURES_DIR / "cfg" / "keras" / "LinearLearner.yaml"
SEGMENTS_YAML = FIXTURES_DIR / "cfg" / "keras" / "TwoSegmentLearner.yaml"


def _learner_script(path: Path, parameters: dict[str, dict[str, Any]] | None = None) -> str:
    """Build a learner from a configuration file and return the script holding its class."""
    return KerasLearnerBuilder.from_path(path)(parameters=parameters).scripts[-1]


def _built(raw: dict[str, Any], path: Path = LEARNER_YAML) -> str:
    """Build a learner from raw configuration data, as a modified fixture."""
    return KerasLearnerBuilder(raw=raw, current_path=str(path))().scripts[-1]


def test_keras_learner_emits_one_flow_method_per_segment() -> None:
    """Each segment is one `keras.ops` function the adapter calls with the batch entries by name.

    The batch parameters are keyword-only, so a caller that handed the entries over positionally --
    the strategy replicating a step, a hand-written trainer -- fails instead of binding them in
    declaration order; nothing unpacks a mapping any more. Nothing in a flow may name a backend
    either: the same method has to differentiate under `tf.GradientTape`, under
    `jax.value_and_grad` and under torch autograd.
    """
    script = _learner_script(SEGMENTS_YAML)

    assert "def _flow_optimizer_ab(self, *, x, y):" in script
    assert "        out_a = self.a(x, training=True)" in script
    assert "        return loss_ab, {'loss_ab': loss_ab}" in script
    assert "def _flow_optimizer_c(self, *, x, y):" in script
    assert "        return loss_c, {'loss_c': loss_c}" in script
    assert "def _flow_inference(self, *, x, y):" in script
    assert "        out_c = self.c(x, training=False)" in script
    assert "batch[" not in script
    for backend in ("import tensorflow", "import jax", "import torch", "keras.backend.backend()"):
        assert backend not in script


def test_keras_learner_prepares_the_segments_before_it_builds_the_steps() -> None:
    """The order is the adapter's contract, and only JAX fails loudly when it is wrong.

    `prepare` builds every optimizer against its variables; TensorFlow and torch would happily build
    one inside the compiled step instead, so a learner that compiled first would only break on JAX.
    The `optimizers` property reads through the segment for a related reason: `prepare` replaces the
    optimizer of a segment it wraps for loss scaling, and a dictionary built before that would
    report the optimizer that never applied anything.
    """
    script = _learner_script(LEARNER_YAML)

    prepare = script.index("adapter.prepare([self._segment_optimizer]")
    assert script.index("adapter = select_backend_adapter()") < prepare
    assert prepare < script.index("adapter.build_train_step([self._segment_optimizer])")
    assert "self._inference_step = adapter.build_inference_step(self._flow_inference, models=[self.model])" in script
    assert "return {'optimizer': self._segment_optimizer.optimizer}" in script


def test_keras_learner_hands_each_segment_the_variables_of_the_models_it_owns() -> None:
    """The segment's variable list is what the optimizer updates, so it must be exactly the owned ones.

    A list gathered from every model would let one optimizer train what another owns, silently and
    at the wrong learning rate.
    """
    script = _learner_script(SEGMENTS_YAML)

    assert "variables=[v for m in (self.a, self.b,) for v in m.trainable_variables]," in script
    assert "variables=list(self.c.trainable_variables)," in script
    assert "models=[self.a, self.b]," in script
    assert "return {'optimizer_ab': ['a', 'b'], 'optimizer_c': ['c']}" in script


def test_keras_learner_runs_a_model_another_segment_owns_in_inference_mode() -> None:
    """A model a segment only reads must not update its normalization statistics there.

    That is what the torch learner's `eval()` does for the same case; in Keras the mode is an
    argument of the call, so it has to be emitted per segment.
    """
    raw = load_any(SEGMENTS_YAML)
    raw["LEARNERS"][1]["FLOW"].insert(0, ["x", "read_a", "a"])

    script = _built(raw, SEGMENTS_YAML)

    assert "        read_a = self.a(x, training=False)" in script
    assert "        out_c = self.c(x, training=True)" in script


def test_keras_learner_counts_updates_from_the_optimizers_own_counter() -> None:
    """The window is the OPTIMIZER pattern's `gradient_accumulation_steps`, and the learner reads it back.

    The generated `__init__` refuses optimizers that disagree on the window, and the first segment
    is the learner's clock: `training_step` counts itself on the host and reads that optimizer's
    public `iterations` -- the count of completed windows, whatever the window is -- back after the
    step, so `has_updated` is detection, not prediction. Under a float16 loss-scale skip the frozen
    counter truthfully reports no update, and the read goes through the segment because `prepare`
    may have wrapped the optimizer since.
    """
    script = _learner_script(LEARNER_YAML, {"DEFAULT": {"accumulate_gradients": 3}})

    assert "optimizer = SGD(learning_rate=0.1, gradient_accumulation_steps=3)" in script
    assert (
        'inners = [getattr(s.optimizer, "inner_optimizer", s.optimizer) for s in (self._segment_optimizer,)]' in script
    )
    assert "windows = sorted({inner.gradient_accumulation_steps or 1 for inner in inners})" in script
    assert "if len(windows) > 1:" in script
    assert "raise ValueError" in script
    assert "self._steps += 1" in script
    # One expression and no local of the template's own: every other name in a step's namespace is
    # a batch input the user named, and `convert_to_numpy` rather than a bare `int()` because under
    # `tf.distribute` the counter's value is a `MirroredVariable`, which `int()` refuses to read.
    assert "current = int(keras.ops.convert_to_numpy(" in script
    assert (
        'getattr(self._segment_optimizer.optimizer, "inner_optimizer", self._segment_optimizer.optimizer).iterations'
        in script
    )
    assert "self._has_updated = current > self._last_updates" in script
    assert "def restore_counters(self, steps: int, updates: int) -> None:" in script
    assert "def update(" not in script
    # The public property already divides the raw call count by the window, so neither the private
    # counter nor a stored window is read any more.
    assert "_iterations" not in script
    assert "self._window" not in script


def test_keras_learner_leaves_a_window_of_one_to_the_plain_optimizer() -> None:
    """An unset window emits no optimizer keyword, and the counter read stays the truth source.

    Keras refuses `gradient_accumulation_steps=1` outright, so nothing may be added to the pattern;
    `iterations` is then the raw call count, and there is deliberately no short-circuit around the
    post-step read: under a float16 loss-scale skip the counter freezes, and only the read reports
    the skipped step truthfully.
    """
    script = _learner_script(LEARNER_YAML)

    assert "optimizer = SGD(learning_rate=0.1)" in script
    assert ".gradient_accumulation_steps = " not in script
    assert "current = int(keras.ops.convert_to_numpy(" in script
    assert ".iterations" in script


@pytest.mark.parametrize(
    ("mixed_precision", "expected"),
    [
        (True, "mixed_precision=True, mixed_precision_type='float16'"),
        ({"initial_scale": 128.0}, "mixed_precision={'initial_scale': 128.0}, mixed_precision_type='float16'"),
    ],
    ids=["enabled", "keyword-arguments"],
)
def test_keras_learner_forwards_the_mixed_precision_fields_to_the_adapter(mixed_precision: Any, expected: str) -> None:
    """Loss scaling is the adapter's to apply, and a dict carries the wrapper's keyword arguments."""
    raw = {**load_any(LEARNER_YAML), "MIXED_PRECISION": mixed_precision, "MIXED_PRECISION_TYPE": "float16"}

    script = _built(raw)

    assert f"adapter.prepare([self._segment_optimizer], {expected})" in script


def test_keras_learner_never_sets_the_global_mixed_precision_policy() -> None:
    """The policy has to be in place before the models are built, which is before this learner exists.

    A learner that set it would apply it to nothing -- the models it receives are already built --
    while looking like it had (`docs/adr/0016`).
    """
    raw = {**load_any(LEARNER_YAML), "MIXED_PRECISION": True, "MIXED_PRECISION_TYPE": "bfloat16"}

    script = _built(raw)

    assert "mixed_precision_type='bfloat16'" in script
    assert "set_global_policy" not in script
    assert "keras.mixed_precision.Policy" not in script


@pytest.mark.parametrize(
    ("raw_fields", "expected"),
    [
        ({}, "    MIXED_PRECISION = False\n    MIXED_PRECISION_TYPE = None"),
        (
            {"MIXED_PRECISION": True, "MIXED_PRECISION_TYPE": "bfloat16"},
            "    MIXED_PRECISION = True\n    MIXED_PRECISION_TYPE = 'bfloat16'",
        ),
        (
            {"MIXED_PRECISION": {"initial_scale": 128.0}, "MIXED_PRECISION_TYPE": "float16"},
            "    MIXED_PRECISION = {'initial_scale': 128.0}\n    MIXED_PRECISION_TYPE = 'float16'",
        ),
        (
            {"MIXED_PRECISION": {}, "MIXED_PRECISION_TYPE": "float16"},
            "    MIXED_PRECISION = {}\n    MIXED_PRECISION_TYPE = 'float16'",
        ),
    ],
    ids=["disabled", "enabled", "keyword-arguments", "empty-keyword-arguments"],
)
def test_keras_learner_declares_its_mixed_precision_on_the_class(raw_fields: dict[str, Any], expected: str) -> None:
    """The training CLI reads the policy off the class before it instantiates it.

    It cannot ask the learner: the policy has to be set before the models the learner is built over
    exist, so it has to be readable from the class the CLI already holds. An empty mapping is
    emitted as it was written, because it enables the policy here as it does in the adapter: only
    `False` disables it.
    """
    script = _built({**load_any(LEARNER_YAML), **raw_fields})

    assert script.startswith("class Learner:")
    assert expected in script


def test_keras_learner_schema_rejects_the_torch_only_clip_field() -> None:
    """Clipping is a keyword of the Keras optimizer, so a CLIP key must be sent to the OPTIMIZER.

    Rejecting it is not enough: the inherited `extra="forbid"` already does that, with a generic
    "Extra inputs are not permitted" that leaves a reader porting a torch learner with no idea where
    clipping went on this backend. The message therefore has to name the substitute keywords.
    """
    raw = load_any(LEARNER_YAML)
    raw["LEARNERS"][0]["CLIP"] = {"_obj_": [["_addr_", "keras.utils.clip_by_norm"]]}

    with pytest.raises(SpecError, match="CLIP has no Keras equivalent") as error:
        KerasLearnerBuilder(raw=raw, current_path=str(LEARNER_YAML))()

    message = str(error.value)
    assert "OPTIMIZER" in message
    for keyword in ("clipnorm", "clipvalue", "global_clipnorm"):
        assert keyword in message


def test_keras_learner_schema_rejects_extra_backward_arguments() -> None:
    """`optimizer.apply(gradients, variables)` takes nothing else, so EXTRA has nowhere to go.

    Accepting and ignoring it is exactly the silent fallback issue #21 bans: the run would train
    without whatever the user asked for.
    """
    raw = load_any(LEARNER_YAML)
    raw["LEARNERS"][0]["EXTRA"] = {"value": "eval: loss"}

    with pytest.raises(SpecError, match="EXTRA has no Keras equivalent"):
        KerasLearnerBuilder(raw=raw, current_path=str(LEARNER_YAML))()


@pytest.mark.parametrize(
    ("mixed_precision", "mixed_precision_type", "message"),
    [
        (True, None, "no default element type"),
        (False, "float16", "alone it would be dropped"),
    ],
    ids=["without-a-type", "type-alone"],
)
def test_keras_learner_schema_rejects_half_a_mixed_precision_configuration(
    mixed_precision: Any, mixed_precision_type: str | None, message: str
) -> None:
    """Either field alone is a setting that would silently do nothing.

    `MIXED_PRECISION` picks the policy the CLI sets and cannot guess its element type; the type
    without it names a policy nobody would ever set.
    """
    raw = {
        **load_any(LEARNER_YAML),
        "MIXED_PRECISION": mixed_precision,
        "MIXED_PRECISION_TYPE": mixed_precision_type,
    }

    with pytest.raises(SpecError, match=message):
        KerasLearnerBuilder(raw=raw, current_path=str(LEARNER_YAML))()


def test_keras_learner_schema_rejects_loss_scale_kwargs_under_a_bfloat16_policy() -> None:
    """A dict `MIXED_PRECISION` carries `LossScaleOptimizer` kwargs, which only float16 ever wraps.

    Under bfloat16 the adapter never builds a `LossScaleOptimizer`, so the kwargs would be read by
    nobody and the run would train unscaled without a word about the discarded configuration.
    """
    raw = {
        **load_any(LEARNER_YAML),
        "MIXED_PRECISION": {"initial_scale": 1024.0},
        "MIXED_PRECISION_TYPE": "bfloat16",
    }

    with pytest.raises(SpecError, match="only a float16 policy wraps the optimizers in"):
        KerasLearnerBuilder(raw=raw, current_path=str(LEARNER_YAML))()


def test_keras_learner_rejects_an_optimizer_named_inference() -> None:
    """Every segment emits `_flow_<optimizer>`, and the learner already defines `_flow_inference`.

    Python keeps the last definition of a duplicated method name, so the generated class would run
    one of the two flows in place of the other with nothing in the script hinting at the swap.
    """
    raw = load_any(LEARNER_YAML)
    raw["LEARNERS"][0]["NAME"] = "inference"

    with pytest.raises(SpecError, match='An optimizer named "inference"'):
        # `scripts` is a cached property: binding it is what runs the emission being rejected here.
        _ = KerasLearnerBuilder(raw=raw, current_path=str(LEARNER_YAML))().scripts


def test_keras_learner_rejects_two_segments_reporting_the_same_criterion() -> None:
    """The training step merges its segments' criteria last-wins, so a clash loses one of them.

    The adapter only sees the names at run time and cannot tell a clash from an intended overwrite,
    which makes distinct names the generator's job (`docs/adr/0016`).
    """
    raw = load_any(SEGMENTS_YAML)
    raw["LEARNERS"][1]["FLOW"].append(["eval: keras.ops.mean(errors_c)", "loss_ab", None])

    with pytest.raises(SpecError, match='Criterion "loss_ab" is computed by both'):
        # `scripts` is a cached property: binding it is what runs the emission being rejected here.
        _ = KerasLearnerBuilder(raw=raw, current_path=str(SEGMENTS_YAML))().scripts


def test_keras_learner_rejects_a_segment_that_reads_what_another_segment_computes() -> None:
    """A segment is one function the adapter calls with the batch alone: nothing else is in scope.

    The generated code would be valid Python raising `NameError` on the first batch, long after the
    script was written, so the flow is refused while it is still being generated.
    """
    raw = load_any(SEGMENTS_YAML)
    raw["LEARNERS"][1]["FLOW"].insert(0, ["eval: out_a * 2.0", "doubled", None])

    with pytest.raises(SpecError, match='reads "out_a" before its own FLOW stores it'):
        # `scripts` is a cached property: binding it is what runs the emission being rejected here.
        _ = KerasLearnerBuilder(raw=raw, current_path=str(SEGMENTS_YAML))().scripts


def test_keras_learner_module_holds_only_imports_and_the_class(tmp_path: Path) -> None:
    """Every module-level name a template emits is a collision with whatever the configuration names.

    A learner's layers, losses and metrics are the user's, and the builder imports them into this
    same module scope, so a constant beside the class is one configuration away from being
    overwritten -- silently, since the CLI would then read whatever landed there. The three
    constants it reads live on the class instead, which is also what it holds before it instantiates
    anything.
    """
    KerasLearnerBuilder.from_path(SEGMENTS_YAML)()(tmp_path / "learner.py")
    module = ast.parse((tmp_path / "learner.py").read_text(encoding="utf-8"))

    assert [type(node).__name__ for node in module.body if not isinstance(node, ast.Import | ast.ImportFrom)] == [
        "ClassDef"
    ]
    learner = module.body[-1]
    assert isinstance(learner, ast.ClassDef)
    declared = {t.id for n in learner.body if isinstance(n, ast.Assign) for t in n.targets if isinstance(t, ast.Name)}
    assert declared == {"MIXED_PRECISION", "MIXED_PRECISION_TYPE", "OPTIMIZER_HASHES"}


def test_keras_learner_cites_no_repository_document(tmp_path: Path) -> None:
    """A generated file is read where it was written, without the repository that produced it.

    Its comments therefore have to carry the constraint itself: a reader who cannot open the
    document a citation names is left with a rule and no reason for it. The whole written file, not
    just the learner class, since the layer scripts travel in it.
    """
    KerasLearnerBuilder.from_path(SEGMENTS_YAML)()(tmp_path / "learner.py")

    assert "docs/adr" not in (tmp_path / "learner.py").read_text(encoding="utf-8")


def test_keras_learner_imports_only_keras_and_the_adapter_helpers() -> None:
    """The generated learner calls the two adapter entry points and `keras.ops`, and nothing else."""
    imports = KerasLearnerBuilder.from_path(LEARNER_YAML)().collected_imports

    assert imports["keras"] == {None}
    assert imports["structcast_model.keras.adapters"] == {"AdapterSegment", "select_backend_adapter"}


def test_keras_learner_swaps_an_average_into_inference_only_when_one_is_declared() -> None:
    """The swap wrapper is emitted off the OPTIMIZER pattern, so a learner without an EMA is untouched.

    Emitting it unconditionally would put an import, a `try`/`finally` and two swap calls into every
    generated learner, iterating a list that is always empty. The generated file is what a reader
    checks a run against, and a step announcing that it evaluates an average where none exists costs
    more than the branch it saves.
    """
    raw = load_any(LEARNER_YAML)
    plain = _built(raw)
    raw["LEARNERS"][0]["OPTIMIZER"] = [
        "_obj_",
        {"_addr_": "keras.optimizers.SGD"},
        {"_call_": {"learning_rate": 0.1, "use_ema": True}},
    ]
    built = KerasLearnerBuilder(raw=raw, current_path=str(LEARNER_YAML))()

    assert "swap_ema_weights" not in plain
    assert "_ema_optimizers" not in plain
    # Both directions of the loan, and the import that carries them, only here.
    assert built.scripts[-1].count("swap_ema_weights(self._ema_optimizers)") == 2
    assert built.collected_imports["structcast_model.keras.adapters"] == {
        "AdapterSegment",
        "select_backend_adapter",
        "swap_ema_weights",
    }


def test_keras_learner_rejects_two_optimizers_averaging_the_same_model() -> None:
    """One model, one average: the second one would be blended every step and evaluated by nobody.

    The inference step can put only one average into a model's variables, and it resolves the clash
    by letting the first optimizer win -- the same silent drop `_criteria` refuses for a criterion
    two segments both compute, refused here for the same reason and named the same way.
    """
    raw = load_any(SEGMENTS_YAML)
    ema = [{"_addr_": "keras.optimizers.SGD"}, {"_call_": {"learning_rate": 0.1, "use_ema": True}}]
    # The second segment keeps the model its own flow runs and takes the first segment's "a" too.
    raw["LEARNERS"][1]["TRAINABLE_LAYERS"] = ["c", "a"]
    for learner in raw["LEARNERS"]:
        learner["OPTIMIZER"] = ["_obj_", *ema]

    with pytest.raises(SpecError, match='Model "a" is trained by optimizer "optimizer_ab"'):
        _built(raw, SEGMENTS_YAML)


def test_keras_learner_scripts_are_byte_identical_across_builds() -> None:
    """The same template must render the same script every time, in every process.

    A name derived from `id()` or from set iteration order would produce phantom diffs for committed
    scripts and defeat any "already generated" check.
    """
    assert _learner_script(SEGMENTS_YAML) == _learner_script(SEGMENTS_YAML)


LEARNER_TEMPLATES = {
    "ConvNeXtV2": (["image", "label"], ["optimizer"]),
    "CycleGAN": (["real_A", "real_B"], ["optimizer_G", "optimizer_D_A", "optimizer_D_B"]),
    "ImageClassifier": (["image", "label"], ["optimizer"]),
    "SmallLanguageModel": (["tokens", "targets"], ["optimizer"]),
}
"""Every shipped Keras learner template, with the batch it reads and the segments it emits."""


@pytest.mark.parametrize(("name", "inputs", "segments"), [(n, i, s) for n, (i, s) in LEARNER_TEMPLATES.items()])
def test_every_shipped_keras_learner_template_renders_one_flow_per_segment(
    name: str, inputs: list[str], segments: list[str]
) -> None:
    """The batch a template declares and the segments it emits are its contract with a run.

    The training CLI calls the generated steps with the dataset's keys, and the checkpoints are
    keyed by segment name, so both are pinned here: a template that quietly renamed an input would
    only fail on the first batch of a real run.
    """
    script = _learner_script(CFG_DIR / "keras" / "learners" / f"{name}.yaml")

    assert script.startswith("class Learner:")
    assert f"self.inputs = {inputs!r}" in script
    for segment in segments:
        assert f"def _flow_{segment}(self, *, {', '.join(inputs)}):" in script
    assert f"def _flow_inference(self, *, {', '.join(inputs)}):" in script


@pytest.mark.parametrize("name", ["ConvNeXtV2", "ImageClassifier", "SmallLanguageModel"])
def test_the_keras_learner_templates_put_clipping_and_accumulation_in_the_optimizer(name: str) -> None:
    """Both knobs are Keras optimizer keywords, and the templates are what has to know that.

    `CLIP` and `ACCUMULATE_GRADIENTS` are refused by the Keras learner schema (`docs/adr/0016`), so
    a template offering the parameters but dropping them anywhere else would silently train
    unclipped and unaccumulated -- which is what these two parameters exist to prevent.
    """
    parameters = {"DEFAULT": {"clip_grad_norm": 1.5, "accumulate_gradients": 4}}

    script = _learner_script(CFG_DIR / "keras" / "learners" / f"{name}.yaml", parameters)

    assert "global_clipnorm=1.5" in script
    assert "gradient_accumulation_steps=4" in script
    assert "CLIP" not in script
