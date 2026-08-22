"""API-level tests for flax builder classes."""

import ast
from collections import defaultdict
from collections.abc import Callable
import logging
from pathlib import Path
from re import findall, search
from typing import Any

import pytest
from structcast.core.exceptions import SpecError
from structcast.core.instantiator import ObjectPattern

from structcast_model.builders.flax import (
    FlaxBuilder,
    FlaxLayerIntermediate,
    FlaxLearnerBuilder,
    inject_learning_rate,
    optimizer_hash,
)
from structcast_model.builders.utils import resolve_object
from structcast_model.utils.base import load_any
from tests import CFG_DIR, FIXTURES_DIR


def test_flax_layer_intermediate_generates_call_method_without_inference_flow() -> None:
    """Generate __call__ without if/else when INFERENCE_FLOW is absent."""
    script = FlaxLayerIntermediate(
        classname="Unit",
        imports={},
        inputs=["x"],
        outputs=["y"],
        layers={},
        flow=[("x", "y", None)],
        inference_flow=[],
        structured_output=False,
    )._get_layer_script("Unit", [])
    assert "class Unit(flax.nnx.Module):" in script
    assert "def __call__(self, x, *, training = None, **kwargs):" in script
    assert "if training:" not in script
    assert "return y" in script


def test_flax_layer_intermediate_generates_training_inference_branches() -> None:
    """Generate if/else branches when INFERENCE_FLOW is present."""
    script = FlaxLayerIntermediate(
        classname="Unit",
        imports={},
        inputs=["x"],
        outputs=["y"],
        layers={},
        flow=[("x", "y", None)],
        inference_flow=[("x", "y", None)],
        structured_output=False,
    )._get_layer_script("Unit", ["linear = flax.nnx.Linear(8, 4, rngs=rngs)"])
    assert "class Unit(flax.nnx.Module):" in script
    assert "if training:" in script
    assert "else:" in script
    assert "self.linear = flax.nnx.Linear(8, 4, rngs=rngs)" in script
    assert "return y" in script


def test_flax_layer_intermediate_init_accepts_rngs() -> None:
    """Generated __init__ signature must include rngs: flax.nnx.Rngs."""
    script = FlaxLayerIntermediate(
        classname="Unit",
        imports={},
        inputs=["x"],
        outputs=["y"],
        layers={},
        flow=[("x", "y", None)],
        inference_flow=[],
        structured_output=False,
    )._get_layer_script("Unit", [])
    assert "def __init__(self, *, rngs: flax.nnx.Rngs, training: bool = True):" in script


def test_flax_layer_intermediate_uses_inputs_outputs_attributes() -> None:
    """Generated __init__ stores inputs/outputs as plain lists."""
    script = FlaxLayerIntermediate(
        classname="Unit",
        imports={},
        inputs=["image"],
        outputs=["cls"],
        layers={},
        flow=[("image", "cls", None)],
        inference_flow=[],
        structured_output=False,
    )._get_layer_script("Unit", [])
    assert "self.inputs = ['image']" in script
    assert "self.outputs = ['cls']" in script


def test_flax_layer_intermediate_emits_input_shapes_literal() -> None:
    """Emit the declared input shapes as a literal so the built model can create its own dummy inputs."""
    script = FlaxLayerIntermediate(
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


def test_flax_builder_builds_intermediate_and_scripts() -> None:
    """Build a minimal Flax nnx model and render Python script content."""
    raw = {
        "INPUTS": ["x"],
        "OUTPUTS": ["y"],
        "FLOW": [
            [
                "x",
                "y",
                {
                    "_obj_": [
                        ["_addr_", "flax.nnx.Linear"],
                        {"_call_": {"in_features": 8, "out_features": 4, "rngs": "eval: rngs"}},
                    ]
                },
            ]
        ],
    }
    built = FlaxBuilder(raw=raw)(classname="TinyNet")
    assert built.classname == "TinyNet"
    assert "flax.nnx" in built.collected_imports
    assert len(built.scripts) == 1
    assert "class TinyNet(flax.nnx.Module):" in built.scripts[0]


def test_flax_builder_structured_output_returns_dict() -> None:
    """Build a model with structured output and verify dict return."""
    raw = {
        "INPUTS": ["x"],
        "OUTPUTS": ["y"],
        "STRUCTURED_OUTPUT": True,
        "FLOW": [
            [
                "x",
                "y",
                {
                    "_obj_": [
                        ["_addr_", "flax.nnx.Linear"],
                        {"_call_": {"in_features": 8, "out_features": 4, "rngs": "eval: rngs"}},
                    ]
                },
            ]
        ],
    }
    built = FlaxBuilder(raw=raw)(classname="TinyNet", forced_structured_output=True)
    script = built.scripts[0]
    assert "return {'y': y}" in script


def test_flax_builder_eval_rngs_renders_in_init() -> None:
    """The eval: rngs keyword generates rngs=rngs in the __init__ body."""
    raw = {
        "INPUTS": ["x"],
        "OUTPUTS": ["y"],
        "FLOW": [
            [
                "x",
                "y",
                {
                    "_obj_": [
                        ["_addr_", "flax.nnx.Linear"],
                        {"_call_": {"in_features": 8, "out_features": 4, "rngs": "eval: rngs"}},
                    ]
                },
            ]
        ],
    }
    script = FlaxBuilder(raw=raw)(classname="TinyNet").scripts[0]
    assert "rngs=rngs" in script


def test_flax_builder_cfg_convnext_builds_expected_topology() -> None:
    """Build Flax ConvNeXt model from cfg and check key topology outputs."""
    parameters = {"DEFAULT": {"backbone": "tiny", "num_classes": 10}}
    builder = FlaxBuilder.from_path(CFG_DIR / "flax" / "models" / "ConvNeXtV2.yaml")
    built = builder(parameters=parameters, classname="ConvNeXtFlaxTiny")
    assert built.classname == "ConvNeXtFlaxTiny"
    assert built.structured_output is True
    assert built.outputs == ["cls"]
    assert "backbone" in built.layers
    assert "head" in built.layers
    assert len(built.scripts) > 0
    assert "class ConvNeXtFlaxTiny(flax.nnx.Module):" in built.scripts[-1]


@pytest.mark.parametrize("backbone", ["tiny"])
def test_flax_builder_cfg_convnext_sublayer_builds_backbone(backbone: str) -> None:
    """Build Backbone sublayer from Flax ConvNeXt cfg."""
    parameters = {"DEFAULT": {"backbone": backbone}}
    builder = FlaxBuilder.from_path(CFG_DIR / "flax" / "models" / "ConvNeXtV2.yaml")
    built = builder(parameters=parameters, classname="Backbone", user_defined_layer="Backbone")
    assert built.classname == "Backbone"
    assert built.structured_output is True
    assert "stem" in built.layers
    assert any("downsample" in k for k in built.layers)
    assert len(built.scripts) > 0


@pytest.mark.parametrize(
    ("name", "parameters", "outputs", "layer"),
    [
        ("CycleGAN_generator", {"DEFAULT": {"n_residual_blocks": 1, "init_features": 4}}, ["out"], "res_block0"),
        ("CycleGAN_discriminator", {}, ["out"], "head"),
        ("SmallLanguageModel", {"tiny": {"dim": 16, "heads": 2, "depth": 2}}, ["logits"], "backbone"),
        ("VisionTransformer", {"base": {"dim": 16, "heads": 2, "depth": 2}}, ["cls"], "backbone"),
    ],
)
def test_flax_builder_builds_every_shipped_model_template(
    name: str, parameters: dict[str, dict[str, Any]], outputs: list[str], layer: str
) -> None:
    """Every model template under cfg/flax has to emit a module, whatever else it is asserted to do.

    A template is only reachable through this builder, so a section that renders to invalid YAML or
    names a layer the DSL cannot resolve fails nowhere else -- and the shipped templates are what a
    validation run trains. The named layer of each is what a later flow line or a sharding rule
    addresses it by, so it is pinned here rather than left to the emitted text.
    """
    built = FlaxBuilder.from_path(CFG_DIR / "flax" / "models" / f"{name}.yaml")(parameters=parameters)

    assert built.outputs == outputs
    assert built.structured_output is True
    assert layer in built.layers
    assert "class Model(flax.nnx.Module):" in built.scripts[-1]


def _render(pattern: ObjectPattern) -> tuple[str, dict[str, set[str | None]]]:
    """Render a pattern the way the learner builder does, returning the code and collected imports."""
    imports: defaultdict[str, set[str | None]] = defaultdict(set)
    return resolve_object(imports, pattern)[0], dict(imports)


def test_inject_learning_rate_wraps_the_factory_carrying_the_learning_rate() -> None:
    """The rate-carrying factory is wrapped so its rate lands in a readable `hyperparams` slot.

    Every other keyword becomes a `static_args` entry: `inject_hyperparams` arrayifies numeric
    keywords otherwise, and `bool` is an `int`, so `nesterov=True` would reach `adamw` as `Array(1)`.
    """
    pattern = ObjectPattern.model_validate(
        {
            "_obj_": [
                ["_addr_", "flax.nnx.Optimizer"],
                {
                    "_bind_": {
                        "tx": {
                            "_obj_": [
                                ["_addr_", "optax.chain"],
                                {
                                    "_call_": [
                                        {
                                            "_obj_": [
                                                ["_addr_", "optax.clip_by_global_norm"],
                                                {"_call_": {"max_norm": 2.0}},
                                            ]
                                        },
                                        {
                                            "_obj_": [
                                                ["_addr_", "optax.adamw"],
                                                {"_call_": {"learning_rate": 0.0002, "nesterov": True}},
                                            ]
                                        },
                                    ]
                                },
                            ]
                        },
                        "wrt": "eval: Param",
                    }
                },
            ]
        }
    )
    rewritten, injected = inject_learning_rate(pattern)
    assert injected is True
    code, imports = _render(rewritten)
    assert code == (
        "(lambda *_arg0, **_kw0: Optimizer(*_arg0, tx=chain(clip_by_global_norm(max_norm=2.0), "
        "inject_hyperparams(inner_factory=adamw, static_args=['nesterov'])"
        "(learning_rate=0.0002, nesterov=True)), wrt=Param, **_kw0))"
    )
    assert imports == {
        "flax.nnx": {"Optimizer"},
        "optax": {"adamw", "chain", "clip_by_global_norm", "inject_hyperparams"},
    }


def test_inject_learning_rate_keeps_a_scheduled_rate_and_omits_empty_static_args() -> None:
    """A schedule stays the value of the wrapped call, and a lone learning rate needs no `static_args`."""
    pattern = ObjectPattern.model_validate(
        {
            "_obj_": [
                ["_addr_", "optax.sgd"],
                {
                    "_call_": {
                        "learning_rate": {
                            "_obj_": [
                                ["_addr_", "optax.linear_schedule"],
                                {"_call_": {"init_value": 0.1, "end_value": 0.0, "transition_steps": 10}},
                            ]
                        }
                    }
                },
            ]
        }
    )
    rewritten, injected = inject_learning_rate(pattern)
    assert injected is True
    assert _render(rewritten)[0] == (
        "inject_hyperparams(inner_factory=sgd)"
        "(learning_rate=linear_schedule(init_value=0.1, end_value=0.0, transition_steps=10))"
    )


def test_inject_learning_rate_leaves_an_already_injected_pattern_alone() -> None:
    """A hand-written `inject_hyperparams` already reports its rate and must not be wrapped twice."""
    pattern = ObjectPattern.model_validate(
        {
            "_obj_": [
                ["_addr_", "optax.inject_hyperparams"],
                {"_call_": [{"_obj_": [["_addr_", "optax.adamw"]]}]},
                {"_call_": {"learning_rate": 0.001}},
            ]
        }
    )
    rewritten, injected = inject_learning_rate(pattern)
    assert rewritten is pattern
    assert injected is True


def test_inject_learning_rate_reports_a_pattern_without_a_learning_rate_keyword() -> None:
    """A positional rate cannot be identified, so the pattern is left alone and flagged as unreported."""
    pattern = ObjectPattern.model_validate({"_obj_": [["_addr_", "optax.sgd"], ["_call_", 0.1]]})
    rewritten, injected = inject_learning_rate(pattern)
    assert rewritten is pattern
    assert injected is False


def test_inject_learning_rate_reports_an_ambiguous_pattern() -> None:
    """Two rate-carrying factories give no single rate to report, so neither is wrapped."""
    pattern = ObjectPattern.model_validate(
        {
            "_obj_": [
                ["_addr_", "optax.chain"],
                {
                    "_call_": [
                        {"_obj_": [["_addr_", "optax.sgd"], {"_call_": {"learning_rate": 0.01}}]},
                        {"_obj_": [["_addr_", "optax.adamw"], {"_call_": {"learning_rate": 0.001}}]},
                    ]
                },
            ]
        }
    )
    rewritten, injected = inject_learning_rate(pattern)
    assert rewritten is pattern
    assert injected is False


LEARNER_YAML = FIXTURES_DIR / "cfg" / "flax" / "LinearLearner.yaml"
SEGMENTS_YAML = FIXTURES_DIR / "cfg" / "flax" / "TwoSegmentLearner.yaml"


def _learner_script(path: Path, parameters: dict[str, dict[str, Any]] | None = None) -> str:
    """Build a learner from a configuration file and return the script holding its steps and class."""
    return FlaxLearnerBuilder.from_path(path)(parameters=parameters).scripts[-1]


def test_flax_learner_emits_the_steps_as_functions_over_named_state() -> None:
    """The steps must be plain functions taking every model and optimizer as its own parameter.

    That is the whole compile seam: `flax.nnx.jit` may not close over models or optimizers, and a
    bound method cannot be wrapped, so a step that read `self` would be uncompilable. The batch is
    keyword-only, which is what tells the caller donating the state apart from the batch.
    """
    script = _learner_script(LEARNER_YAML)

    assert "def _training_step(model, optimizer, *, x, y, **kwargs):" in script
    assert "def _inference_step(model, *, x, y, **kwargs):" in script
    assert "(_, (loss,)), _grads = flax.nnx.value_and_grad(_flow_optimizer, has_aux=True)(model, x=x, y=y)" in script
    assert "lrs = {'optimizer': get_learning_rate(optimizer)}" in script
    assert "return {'loss': loss}, lrs, _has_updated\n" in script
    # The keys are attribute names: a trainer compiles a step by rebinding the attribute it names.
    assert 'return {"_training_step": self._training_step, "_inference_step": self._inference_step}' in script
    assert "self._training_step = _training_step" in script
    assert "criteria, learning_rates, has_updated = self._training_step(self.model, self.optimizer, x=x, y=y" in script


def test_flax_learner_module_scope_holds_the_imports_and_the_class_alone() -> None:
    """Every generated module-level name is a collision waiting for the right configuration.

    The learner imports whatever the user's patterns reference into this module, so a flow layer or
    a step left at module scope could be shadowed by -- or shadow -- one of those imports. Keeping
    module scope to imports and the class itself is what makes that impossible.
    """
    script = FlaxLearnerBuilder.from_path(SEGMENTS_YAML)()
    module = ast.parse("\n".join(script.scripts))

    assert [type(node).__name__ for node in module.body] == ["ClassDef"]
    # The flow layers, the flows and the steps are all built where only the learner can see them.
    assert "mse = squared_error" in script.scripts[-1]
    assert "def _flow_optimizer_ab(a, b, x, y):" in script.scripts[-1]


def test_flax_learner_comments_describe_behavior_and_cite_no_repository_documents() -> None:
    """A generated learner is read where this repository is not, so a citation there names nothing.

    Its comments and docstring have to carry the caveat itself -- the flow layers being captured,
    the rates being read at trace time -- rather than point at a document the reader cannot open.
    """
    script = _learner_script(SEGMENTS_YAML)

    assert "docs/adr" not in script
    assert "they must be stateless" in script
    assert "Read at trace time" in script


def test_flax_learner_emits_the_optimizer_digests_as_a_class_attribute() -> None:
    """The digests are read off the class by the CLI, so they may not sit next to it at module scope."""
    (learner,) = ast.parse(_learner_script(LEARNER_YAML)).body
    assert isinstance(learner, ast.ClassDef)

    annotated = [node for node in learner.body if isinstance(node, ast.AnnAssign)]

    assert [ast.unparse(node.target) for node in annotated] == ["OPTIMIZER_HASHES"]


def test_flax_learner_applies_the_optimizer_pattern_to_the_models_it_owns() -> None:
    """The pattern is a callable returning an optimizer, so the owned module and `wrt` complete it."""
    script = _learner_script(LEARNER_YAML)

    assert ")(model, wrt=Param)" in script
    assert "optimizer.update(model, _grads)" in script


def test_flax_learner_never_reads_a_variable_value_or_an_update_result() -> None:
    """Two cross-version rules of the supported flax range (0.12.6..0.12.9) are pinned here.

    `Variable.value` is deprecated in favour of `variable[...]`, and `nnx.Optimizer.update` returns
    `None` on the floor version, so generated code that consumed either would break on one end of
    the range.
    """
    script = _learner_script(SEGMENTS_YAML)

    # `.value_and_grad` is the transform, not a variable read, hence the word boundary.
    assert search(r"\.value(?!\w)", script) is None
    assert search(r"=\s*\w+\.update\(", script) is None


def test_flax_learner_imports_the_helpers_its_steps_call() -> None:
    """The generated steps call `get_learning_rate` and `gradient_steps` directly."""
    imports = FlaxLearnerBuilder.from_path(LEARNER_YAML)().collected_imports

    assert imports["structcast_model.flax.optimizers"] == {"get_learning_rate", "gradient_steps"}
    assert imports["flax.nnx"] == {None, "Param", "Optimizer"}
    assert imports["jax"] == {None}
    assert imports["jax.numpy"] == {None}


def test_flax_learner_detects_its_updates_inside_the_step() -> None:
    """Accumulation is the pattern's `optax.MultiSteps`: the device gates, and the step reads it back.

    The pattern is never parsed for a window, and no window is stored: the step compares the count
    the first optimizer advanced across its own update and hands the answer back with the criteria,
    so the learner counts updates without a host read of its own -- detection, not prediction.
    """
    script = _learner_script(LEARNER_YAML, {"DEFAULT": {"accumulate_gradients": 3}})

    assert "MultiSteps(" in script
    assert "_before = gradient_steps(optimizer)" in script
    assert "_has_updated = True if _before is None else gradient_steps(optimizer) > _before" in script
    assert "self._steps += 1" in script
    assert "self._has_updated = bool(has_updated)" in script
    assert "self._updates += int(self._has_updated)" in script
    assert "def restore_counters(self, steps: int, updates: int) -> None:" in script
    assert "accumulation_window" not in script
    assert "self._window" not in script
    assert "def update(" not in script
    assert "acc_grads" not in script
    assert "need_update" not in script


def test_flax_learner_detects_on_the_first_optimizer_alone() -> None:
    """One learner, one update count, and the first segment is the clock it runs on.

    Segments need not share a window, so a second comparison would answer a question no counter of
    the learner asks; the later segments emit their update and nothing else.
    """
    script = _learner_script(SEGMENTS_YAML)

    assert "_before = gradient_steps(optimizer_ab)" in script
    assert "gradient_steps(optimizer_c)" not in script


def test_flax_learner_passes_several_owned_models_as_a_plain_tuple() -> None:
    """One optimizer over several models owns them as a tuple, wherever they are named.

    The optimizer state, the differentiated arguments and the update have to key off the same module
    paths, and the tuple is what gives all three the same structure without a container node the
    learner would have to keep among its models.
    """
    script = _learner_script(SEGMENTS_YAML)

    assert "def _flow_optimizer_ab(a, b, x, y):" in script
    assert "flax.nnx.value_and_grad(_flow_optimizer_ab, argnums=(0, 1), has_aux=True)(a, b, x=x, y=y)" in script
    assert "optimizer_ab.update((a, b), _grads)" in script
    assert ")((a, b), wrt=Param)" in script
    assert ")(c, wrt=Param)" in script
    # No container: the models a trainer sees are the ones the learner was built over.
    assert "return {'a': self.a, 'b': self.b, 'c': self.c}" in script
    assert "return {'optimizer_ab': ['a', 'b'], 'optimizer_c': ['c']}" in script
    assert "flax.nnx.List" not in script


def test_flax_learner_passes_a_model_a_segment_only_reads_as_a_parameter() -> None:
    """A model is state wherever a flow names it, not only where the flow calls it as its layer.

    The flows are built in `__init__`, where every model is also a local: a model read in an
    expression and not passed in would be captured from there and frozen into the compiled step,
    so the segment would keep computing with the values that model had when the learner was built.
    """
    raw = load_any(SEGMENTS_YAML)
    raw["LEARNERS"][1]["FLOW"].insert(0, ["eval: a.fc.kernel[...].mean()", "reg_c", None])

    script = FlaxLearnerBuilder(raw=raw, current_path=str(SEGMENTS_YAML))().scripts[-1]

    assert "def _flow_optimizer_c(c, a, x, y):" in script
    assert "flax.nnx.value_and_grad(_flow_optimizer_c, has_aux=True)(c, a, x=x, y=y)" in script


def test_flax_learner_carries_out_the_values_a_later_update_reads() -> None:
    """`EXTRA` is evaluated in the step, so a later one reads an earlier flow's values there.

    Those values only exist in the step if the flow that computed them returned them, and a keyword
    naming one that did not emits a step that fails on the first batch.
    """
    raw = load_any(SEGMENTS_YAML)
    raw["LEARNERS"][1]["EXTRA"] = {"value": "eval: errors_a"}

    script = FlaxLearnerBuilder(raw=raw, current_path=str(SEGMENTS_YAML))().scripts[-1]

    assert "return loss_ab, (errors_a, loss_ab,)" in script
    assert "optimizer_c.update(c, _grads, value=errors_a)" in script


NO_INPUT_LEARNER: dict[str, Any] = {
    "INPUTS": [],
    "OUTPUTS": ["loss"],
    "LEARNERS": [
        {
            "NAME": "optimizer",
            "LOSS": "loss",
            "TRAINABLE_LAYERS": ["model"],
            "OPTIMIZER": [
                "_obj_",
                {"_addr_": "flax.nnx.Optimizer"},
                {"_bind_": {"tx": ["_obj_", {"_addr_": "optax.sgd"}, {"_call_": {"learning_rate": 0.1}}]}},
            ],
            "FLOW": [
                ["eval: jax.numpy.ones((2, 4))", "prediction", "model"],
                ["eval: jax.numpy.mean(prediction ** 2)", "loss", None],
            ],
        }
    ],
}
"""A learner whose flow needs no batch at all, as a generative or a replay-buffer one does."""


def test_flax_learner_emits_steps_without_a_batch() -> None:
    """A learner declaring no inputs must still emit valid signatures and call sites.

    The batch is a keyword-only section of every step, and an empty one may not leave a dangling
    `*,` behind or a stray comma in the calls the learner makes.
    """
    script = FlaxLearnerBuilder(raw=NO_INPUT_LEARNER)().scripts[-1]

    assert "def _training_step(model, optimizer, **kwargs):" in script
    assert "def _inference_step(model, **kwargs):" in script
    assert "def training_step(self, **kwargs):" in script
    assert "self._training_step(self.model, self.optimizer, **kwargs)" in script
    assert "return self._inference_step(self._view_model, **kwargs)" in script
    compile(script, "<learner>", "exec")


def test_flax_learner_builds_inference_views_of_every_model() -> None:
    """Inference runs against views: arrays shared with the trained models, inference flags forced.

    `raise_if_not_found=False` keeps a model without dropout or normalization from failing the build.
    """
    script = _learner_script(LEARNER_YAML)

    assert (
        "self._view_model = flax.nnx.view(model, raise_if_not_found=False, training=False, "
        "deterministic=True, use_running_average=True)"
    ) in script
    assert "return self._inference_step(self._view_model, x=x, y=y, **kwargs)" in script


def test_flax_learner_keeps_a_hand_bound_wrt() -> None:
    """A pattern that binds `wrt` itself already says what to optimize, so nothing is appended."""
    raw = load_any(LEARNER_YAML)
    raw["LEARNERS"][0]["OPTIMIZER"][2]["_bind_"]["wrt"] = "eval: flax.nnx.Param"

    script = FlaxLearnerBuilder(raw=raw, current_path=str(LEARNER_YAML))().scripts[-1]

    # The hand-bound `wrt` survives inside the pattern, and none is appended to the applied call.
    assert "wrt=flax.nnx.Param" in script
    assert "(model, wrt=Param)" not in script
    assert "))(model)" in script


def test_flax_learner_warns_when_the_learning_rate_cannot_be_reported(caplog: pytest.LogCaptureFixture) -> None:
    """A rate the rewrite cannot find is reported as NaN for the whole run, which has to be said out loud."""
    raw = load_any(LEARNER_YAML)
    raw["LEARNERS"][0]["OPTIMIZER"][2] = {"_bind_": {"tx": ["_obj_", {"_addr_": "optax.sgd"}, ["_call_", 0.1]]}}

    with caplog.at_level(logging.WARNING):
        script = FlaxLearnerBuilder(raw=raw, current_path=str(LEARNER_YAML))().scripts[-1]

    assert "reports no learning rate" in caplog.text
    assert "inject_hyperparams" not in script


def test_flax_learner_rejects_a_segment_that_does_not_compute_its_own_loss() -> None:
    """A segment differentiates its own closure, so a loss computed by another segment is unreachable."""
    raw = load_any(SEGMENTS_YAML)
    raw["LEARNERS"][1]["LOSS"] = "loss_ab"

    with pytest.raises(SpecError, match='its FLOW does not compute its LOSS "loss_ab"'):
        # `scripts` is a cached property: binding it is what runs the emission being rejected here.
        _ = FlaxLearnerBuilder(raw=raw, current_path=str(SEGMENTS_YAML))().scripts


def test_flax_convnext_learner_cfg_keeps_its_rate_readable_and_its_norms_undecayed() -> None:
    """The shipped ConvNeXt V2 learner is the reference for the whole Flax optimizer story.

    Its rate has to survive the rewrite as an injected hyperparameter while every other keyword --
    including the mask callable, which `inject_hyperparams` would otherwise try to arrayify --
    stays static, and the structured model output has to be unpacked before the criteria read it.
    """
    script = _learner_script(CFG_DIR / "flax" / "learners" / "ConvNeXtV2.yaml")

    assert "cls = model_output['cls']" in script
    assert (
        "tx=chain(clip_by_global_norm(max_norm=1.0), "
        "inject_hyperparams(inner_factory=adamw, static_args=['weight_decay', 'b1', 'b2', 'mask'])"
        "(learning_rate=0.001, weight_decay=0.05, b1=0.9, b2=0.999, "
        "mask=no_weight_decay_mask('^(?:\\\\w+\\\\.)*bias$', '^(?:\\\\w+\\\\.)*scale$'))"
    ) in script


@pytest.mark.parametrize("name", ["ImageClassifier", "SmallLanguageModel"])
def test_flax_single_segment_learner_cfgs_wrap_the_whole_chain_in_multi_steps(name: str) -> None:
    """Accumulation only counts when `optax.MultiSteps` is the outermost transformation.

    The generated step reads its update gate off the optimizer's outermost `opt_state`, so a window
    nested inside the `optax.chain` would accumulate exactly the same and still report an update on
    every step (`docs/adr/0019`). Both templates offer clipping and accumulation as independent
    parameters, so the combination is where a misplaced wrapper would hide.
    """
    parameters = {"DEFAULT": {"clip_grad_norm": 1.0, "accumulate_gradients": 4}}
    script = _learner_script(CFG_DIR / "flax" / "learners" / f"{name}.yaml", parameters)

    assert "tx=MultiSteps(opt=chain(clip_by_global_norm(max_norm=1.0), inject_hyperparams(" in script
    assert "every_k_schedule=4" in script
    # Without the clip the chain is the adamw alone, and the window still wraps the whole chain.
    plain = _learner_script(CFG_DIR / "flax" / "learners" / f"{name}.yaml", {"DEFAULT": {"accumulate_gradients": 4}})
    assert "tx=MultiSteps(opt=chain(inject_hyperparams(" in plain
    assert "clip_by_global_norm" not in plain


def test_flax_cycle_gan_learner_cfg_hands_the_generated_images_to_the_discriminators() -> None:
    """The three segments are one program: what the generator flow computes is what the critics see.

    The torch template feeds its discriminators a replay-buffer sample it takes as an input; a Flax
    segment differentiates only what its own flow computes, so this template carries "fake_A" and
    "fake_B" out of the generator closure and passes them in as plain values instead -- which is
    also why no gradient can leak back into the generators from a discriminator step.
    """
    script = _learner_script(CFG_DIR / "flax" / "learners" / "CycleGAN.yaml")

    assert "def __init__(self, G_AB, G_BA, D_A, D_B, **kwargs):" in script
    # The generators are the leading parameters of their flow, so their positions are the argnums;
    # the discriminators follow as read-only models.
    assert "flax.nnx.value_and_grad(_flow_optimizer_G, argnums=(0, 1), has_aux=True)(G_AB, G_BA, D_B, D_A," in script
    assert "_flow_optimizer_D_A, has_aux=True)(D_A, real_A=real_A, fake_A=fake_A)" in script
    assert "_flow_optimizer_D_B, has_aux=True)(D_B, real_B=real_B, fake_B=fake_B)" in script


def test_flax_learner_keeps_only_the_values_that_leave_the_closure_in_the_aux_tuple() -> None:
    """A value a later segment reads has to ride out of the closure; one nothing reads must not.

    The auxiliary tuple is the closure's contract with the enclosing step: everything in it is a
    traced output the step then unpacks. Returning intermediates nothing reads makes every flow
    value part of that contract, so a flow could no longer compute anything a traced output cannot
    carry -- and the PyTorch flow functions already return only what is needed.
    """
    raw = load_any(SEGMENTS_YAML)
    raw["LEARNERS"][1]["FLOW"].insert(0, ["eval: out_a * 2.0", "scaled_a", None])

    script = FlaxLearnerBuilder(raw=raw, current_path=str(SEGMENTS_YAML))().scripts[-1]

    assert "return loss_ab, (out_a, loss_ab,)" in script
    assert "return loss_c, (loss_c,)" in script


def test_flax_learner_rejects_a_segment_that_reads_a_name_it_stores_later() -> None:
    """A segment is one nested function, so a name it stores is local to all of it.

    Reading it first is valid Python that raises `UnboundLocalError` on the first batch, long after
    the script was written, so the order has to be refused while it is still being generated.
    """
    raw = load_any(SEGMENTS_YAML)
    raw["LEARNERS"][0]["FLOW"].insert(0, ["eval: out_b * 2.0", "doubled", None])

    with pytest.raises(SpecError, match='reads "out_b" before its own FLOW stores it'):
        # `scripts` is a cached property: binding it is what runs the emission being rejected here.
        _ = FlaxLearnerBuilder(raw=raw, current_path=str(SEGMENTS_YAML))().scripts


def _rename_model(raw: dict[str, Any], name: str) -> None:
    """Rename the single model of the linear fixture everywhere its learner names it."""
    raw["LEARNERS"][0]["TRAINABLE_LAYERS"] = [name]
    for key in ("FLOW", "INFERENCE_FLOW"):
        raw["LEARNERS"][0][key][0][2] = name


def _rename_input(raw: dict[str, Any], name: str) -> None:
    """Rename the `y` input of the linear fixture, which its flows read as the regression target."""
    raw["INPUTS"] = ["x", name]
    raw["LEARNERS"][0]["FLOW"][1]["INPUTS"]["targets"] = name
    raw["LEARNERS"][0]["INFERENCE_FLOW"][1][0]["targets"] = name


def _build(raw: dict[str, Any]) -> None:
    """Emit the learner of a mutated linear fixture, which is what runs the checks under test."""
    # `scripts` is a cached property: binding it is what runs the emission being rejected here.
    _ = FlaxLearnerBuilder(raw=raw, current_path=str(LEARNER_YAML))().scripts


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda raw: raw["LEARNERS"][0].update(NAME="x"), 'Name "x" is both an input of the learner'),
        (lambda raw: _rename_model(raw, "inputs"), 'Name "inputs" is reserved'),
        (lambda raw: _rename_model(raw, "training_step"), 'Name "training_step" is reserved'),
        (lambda raw: _rename_input(raw, "kwargs"), 'Name "kwargs" is reserved'),
        (lambda raw: raw["LEARNERS"][0].update(NAME="self"), 'Name "self" is reserved'),
    ],
    ids=["optimizer-named-like-an-input", "model-named-inputs", "model-named-training-step", "input-kwargs", "self"],
)
def test_flax_learner_rejects_a_name_the_generated_class_cannot_carry(
    mutate: Callable[[dict[str, Any]], Any], message: str
) -> None:
    """The steps name every model, every optimizer and every batch entry in one signature.

    A name serving as two of those is emitted twice there -- a script that fails to import -- and one
    equal to a member of the class is worse: `self.inputs = model` in `__init__` overwrites what the
    trainer reads off the learner afterwards, or shadows a property with an attribute, and neither
    failure points back at the name that caused it.
    """
    raw = load_any(LEARNER_YAML)
    mutate(raw)

    with pytest.raises(SpecError, match=message):
        _build(raw)


def test_flax_learner_rejects_a_model_named_like_the_view_of_another() -> None:
    """Each model gets a `_view_<name>` attribute, which another model's name must not already be.

    The two would be one attribute, and whichever `__init__` wrote last would be the one the
    inference step runs against -- a model trained in place, or a view nobody can train.
    """
    raw = load_any(SEGMENTS_YAML)
    raw["LEARNERS"][0]["TRAINABLE_LAYERS"] = ["a", "_view_a"]
    for key in ("FLOW", "INFERENCE_FLOW"):
        raw["LEARNERS"][0][key][1][2] = "_view_a"

    with pytest.raises(SpecError, match='Name "_view_a" is reserved'):
        # `scripts` is a cached property: binding it is what runs the emission being rejected here.
        _ = FlaxLearnerBuilder(raw=raw, current_path=str(SEGMENTS_YAML))().scripts


@pytest.mark.parametrize(
    "mutate",
    [
        lambda raw: raw["LEARNERS"][0]["FLOW"].insert(0, ["eval: 1.0", "lrs", None]),
        lambda raw: raw["LEARNERS"][0]["FLOW"].insert(0, ["eval: 1.0", "_grads", None]),
        lambda raw: raw["LEARNERS"][0]["FLOW"].insert(0, ["eval: 1.0", "model", None]),
    ],
    ids=["learning-rates", "gradients", "model"],
)
def test_flax_learner_rejects_a_flow_output_the_step_already_binds(mutate: Callable[[dict[str, Any]], Any]) -> None:
    """The step binds the flow results next to the names it computes itself, so the two may not meet.

    An output named `lrs` would be returned in place of the learning rates it overwrote, and one
    named like a model would rebind the module before the optimizer is handed it -- both silent,
    both only visible in criteria nobody can explain.
    """
    raw = load_any(LEARNER_YAML)
    mutate(raw)

    with pytest.raises(SpecError, match="which the generated training step already binds"):
        _build(raw)


def test_inject_learning_rate_ignores_a_plain_string_that_looks_like_the_wrapper() -> None:
    """Only an address suppresses the rewrite: a label that reads the same makes no rate readable.

    Suppressing on any string would cost such a pattern both its injected rate and the warning that
    is supposed to announce the loss, leaving the run reporting NaN with nothing said.
    """
    pattern = ObjectPattern.model_validate(
        {
            "_obj_": [
                ["_addr_", "optax.named_chain"],
                {
                    "_call_": [
                        [
                            "sgd_inject_hyperparams",
                            {"_obj_": [["_addr_", "optax.sgd"], {"_call_": {"learning_rate": 0.1}}]},
                        ]
                    ]
                },
            ]
        }
    )

    rewritten, injected = inject_learning_rate(pattern)

    assert injected is True
    assert "inject_hyperparams(inner_factory=sgd)(learning_rate=0.1)" in _render(rewritten)[0]


def test_flax_learner_forwards_extra_keywords_to_the_update() -> None:
    """`EXTRA` is where a transformation that needs more than gradients -- a line search, say -- is fed."""
    raw = load_any(LEARNER_YAML)
    raw["LEARNERS"][0]["EXTRA"] = {"value": "eval: loss"}

    script = FlaxLearnerBuilder(raw=raw, current_path=str(LEARNER_YAML))().scripts[-1]

    assert "optimizer.update(model, _grads, value=loss)" in script


def test_the_learner_builder_never_uses_the_zero_argument_super() -> None:
    """The builder is a `slots=True` dataclass, which rebuilds the class after its methods are compiled.

    Below Python 3.12.4 -- the project floor is 3.11 -- the `__class__` cell those methods close over
    still points at the discarded class, so a zero-argument `super()` raises
    `TypeError: super(type, obj): obj must be an instance or subtype of type` and every generated
    learner fails. The CI interpreters that carry the fix cannot see it, so the shape is asserted
    instead of the behavior.
    """
    for name, member in vars(FlaxLearnerBuilder).items():
        code = getattr(member, "__code__", None)
        assert code is None or "__class__" not in code.co_freevars, name


def _emitted_hashes(script: str) -> dict[str, str]:
    """Read the `OPTIMIZER_HASHES` constant back out of a generated learner script."""
    line = search(r"OPTIMIZER_HASHES: dict\[str, str\] = \{(.*)\}", script)
    assert line is not None, script
    return dict(findall(r"'(\w+)': '(\w+)'", line.group(1)))


def test_flax_learner_emits_the_digest_of_the_optimizer_pattern_as_written() -> None:
    """The emitted digest identifies the OPTIMIZER pattern, so a resume can report a rebuilt optimizer.

    It is taken before `inject_learning_rate` rewrites the pattern, so the digest a run records is
    the one the configuration itself hashes to -- turning the injection on or off must not read as a
    changed optimizer. "As written" is after the jinja accumulation switch resolves: under the
    default parameters the fixture renders the plain sgd tx spelled out here.
    """
    pattern = ObjectPattern.model_validate(
        [
            "_obj_",
            {"_addr_": "flax.nnx.Optimizer"},
            {"_bind_": {"tx": ["_obj_", {"_addr_": "optax.sgd"}, {"_call_": {"learning_rate": 0.1}}]}},
        ]
    )

    assert _emitted_hashes(_learner_script(LEARNER_YAML)) == {"optimizer": optimizer_hash(pattern)}


def test_flax_learner_optimizer_hashes_are_stable_but_move_with_the_schedule() -> None:
    """Regenerating the same configuration must repeat the digest, and a new rate must change it.

    A digest that drifted would make every resume warn; one that did not move with the rate would
    never warn, and optax rebuilds `tx` from configuration without the restored state noticing.
    """
    raw = load_any(LEARNER_YAML)
    raw["LEARNERS"][0]["OPTIMIZER"][2] = {
        "_bind_": {"tx": ["_obj_", {"_addr_": "optax.sgd"}, {"_call_": {"learning_rate": 0.2}}]}
    }
    rebuilt = FlaxLearnerBuilder(raw=raw, current_path=str(LEARNER_YAML))().scripts[-1]

    assert _emitted_hashes(_learner_script(LEARNER_YAML)) == _emitted_hashes(_learner_script(LEARNER_YAML))
    assert _emitted_hashes(rebuilt) != _emitted_hashes(_learner_script(LEARNER_YAML))


def test_flax_learner_emits_one_optimizer_hash_per_segment() -> None:
    """Two segments are two independently rebuildable optimizers, so each carries its own digest."""
    hashes = _emitted_hashes(_learner_script(SEGMENTS_YAML))

    assert sorted(hashes) == ["optimizer_ab", "optimizer_c"]
    assert len(set(hashes.values())) == 2
