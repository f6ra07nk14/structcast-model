"""API-level tests for flax builder classes."""

from collections import defaultdict
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


def test_flax_learner_emits_the_steps_as_module_level_functions() -> None:
    """The steps must be plain module functions over explicit state arguments.

    That is the whole compile seam: `flax.nnx.jit` may not close over models or optimizers, and a
    bound method cannot be wrapped, so a step that read `self` would be uncompilable.
    """
    script = _learner_script(LEARNER_YAML)

    assert "def _training_step(models, optimizers, *, x, y, **kwargs):" in script
    assert "def _inference_step(models, *, x, y, **kwargs):" in script
    assert "(_, (loss,)), _grads = flax.nnx.value_and_grad(_flow_optimizer, has_aux=True)(model)" in script
    assert "lrs = {'optimizer': get_learning_rate(optimizers['optimizer'])}" in script
    assert "return {'loss': loss}, lrs\n" in script
    # The keys are attribute names: a trainer compiles a step by rebinding the attribute it names.
    assert 'return {"_training_step": self._training_step, "_inference_step": self._inference_step}' in script
    assert "self._training_step = _training_step" in script


def test_flax_learner_applies_the_optimizer_pattern_to_the_models_it_owns() -> None:
    """The pattern is a callable returning an optimizer, so the owned module and `wrt` complete it."""
    script = _learner_script(LEARNER_YAML)

    assert ")(model, wrt=Param)" in script
    assert "optimizers['optimizer'].update(model, _grads)" in script


def test_flax_learner_never_reads_a_variable_value_or_an_update_result() -> None:
    """Two cross-version rules of the supported flax range (0.12.6..0.12.9) are pinned here.

    `Variable.value` is deprecated in favour of `variable[...]`, and `nnx.Optimizer.update` returns
    `None` on the floor version, so generated code that consumed either would break on one end of
    the range.
    """
    script = _learner_script(SEGMENTS_YAML)

    # `.value_and_grad` is the transform, not a variable read, hence the word boundary.
    assert search(r"\.value(?!\w)", script) is None
    assert "= optimizers[" not in script


def test_flax_learner_imports_the_helpers_its_steps_call() -> None:
    """The generated steps call `get_learning_rate` and the `flax.nnx`/`jax` APIs directly."""
    imports = FlaxLearnerBuilder.from_path(LEARNER_YAML)().collected_imports

    assert imports["structcast_model.flax.optimizers"] == {"get_learning_rate"}
    assert imports["flax.nnx"] == {None, "Param", "Optimizer"}
    assert imports["jax"] == {None}
    assert imports["jax.numpy"] == {None}


def test_flax_learner_bakes_the_multi_steps_window_into_the_update_gate() -> None:
    """Accumulation is the pattern's `optax.MultiSteps`: the device gates, and `update` predicts it.

    The builder statically parses the wrapper's int-literal window and bakes `step % k == 0`
    as a pure host formula, so the generated step carries no accumulator buffer and no static flag
    (`docs/adr/0017`).
    """
    script = _learner_script(LEARNER_YAML, {"DEFAULT": {"accumulate_gradients": 3}})

    assert "MultiSteps(" in script
    assert "def update(self, step: int) -> bool:\n        return step % 3 == 0" in script
    assert "acc_grads" not in script
    assert "need_update" not in script


def test_flax_learner_without_multi_steps_updates_every_step() -> None:
    """A learner without the wrapper updates every step and pays for no buffer or gate arithmetic."""
    script = _learner_script(LEARNER_YAML)

    assert "MultiSteps(" not in script
    assert "def update(self, step: int) -> bool:\n        return True" in script
    assert "acc_grads" not in script
    assert "need_update" not in script


def _multi_steps_tx(**arguments: Any) -> list[Any]:
    """Build a `MultiSteps` tx pattern over the fixture's sgd, with *arguments* as extra keywords."""
    inner = ["_obj_", {"_addr_": "optax.sgd"}, {"_call_": {"learning_rate": 0.1}}]
    return ["_obj_", {"_addr_": "optax.MultiSteps"}, {"_call_": {"opt": inner, **arguments}}]


def test_flax_learner_rejects_a_multi_steps_window_that_is_not_a_literal() -> None:
    """Only an int literal window can be baked into the generated `update` at build time.

    A schedule (or any pattern) computes the window on the device, where the host formula cannot
    follow it, so the mismatch is refused while the script is still being generated.
    """
    raw = load_any(LEARNER_YAML)
    schedule = [
        "_obj_",
        {"_addr_": "optax.linear_schedule"},
        {"_call_": {"init_value": 2, "end_value": 4, "transition_steps": 10}},
    ]
    raw["LEARNERS"][0]["OPTIMIZER"][2] = {"_bind_": {"tx": _multi_steps_tx(every_k_schedule=schedule)}}

    with pytest.raises(SpecError, match="int literal"):
        FlaxLearnerBuilder(raw=raw, current_path=str(LEARNER_YAML))()


def test_flax_learner_rejects_a_multi_steps_skip_predicate() -> None:
    """`should_skip_update_fn` breaks the call-count identity the baked `update` gate relies on."""
    raw = load_any(LEARNER_YAML)
    predicate = ["_obj_", {"_addr_": "optax.skip_not_finite"}]
    raw["LEARNERS"][0]["OPTIMIZER"][2] = {
        "_bind_": {"tx": _multi_steps_tx(every_k_schedule=2, should_skip_update_fn=predicate)}
    }

    with pytest.raises(SpecError, match="should_skip_update_fn"):
        FlaxLearnerBuilder(raw=raw, current_path=str(LEARNER_YAML))()


def test_flax_learner_ignores_an_address_that_merely_ends_in_multi_steps() -> None:
    """Only an address whose last segment is `MultiSteps` declares a window, not any suffix match."""
    raw = load_any(LEARNER_YAML)
    inner = ["_obj_", {"_addr_": "optax.sgd"}, {"_call_": {"learning_rate": 0.1}}]
    lookalike = ["_obj_", {"_addr_": "mylib.NotMultiSteps"}, {"_call_": {"opt": inner, "every_k_schedule": 2}}]
    raw["LEARNERS"][0]["OPTIMIZER"][2] = {"_bind_": {"tx": lookalike}}

    scripts = FlaxLearnerBuilder(raw=raw, current_path=str(LEARNER_YAML))().scripts

    assert "def update(self, step: int) -> bool:\n        return True" in "".join(scripts)


def test_flax_learner_rejects_nested_multi_steps_wrappers() -> None:
    """Two wrappers declare two windows, so the one the gate should bake is ambiguous."""
    raw = load_any(LEARNER_YAML)
    doubled = [
        "_obj_",
        {"_addr_": "optax.MultiSteps"},
        {"_call_": {"opt": _multi_steps_tx(every_k_schedule=2), "every_k_schedule": 2}},
    ]
    raw["LEARNERS"][0]["OPTIMIZER"][2] = {"_bind_": {"tx": doubled}}

    with pytest.raises(SpecError, match="ambiguous"):
        FlaxLearnerBuilder(raw=raw, current_path=str(LEARNER_YAML))()


def test_flax_learner_rejects_windows_that_disagree_across_segments() -> None:
    """One learner, one window: the trainer's update counter answers for the whole learner.

    A segment without `MultiSteps` counts as a window of one, so wrapping only the first optimizer
    has to be refused with the two values named.
    """
    raw = load_any(SEGMENTS_YAML)
    raw["LEARNERS"][0]["OPTIMIZER"][2] = {"_bind_": {"tx": _multi_steps_tx(every_k_schedule=2)}}

    with pytest.raises(SpecError, match=r"disagree.*\[1, 2\]"):
        # `scripts` is a cached property: binding it is what runs the emission being rejected here.
        _ = FlaxLearnerBuilder(raw=raw, current_path=str(SEGMENTS_YAML))().scripts


def test_flax_learner_wraps_several_owned_models_in_one_persistent_list() -> None:
    """One optimizer over several models needs a single container built once.

    The same `flax.nnx.List` instance has to be the optimizer's module and the differentiated
    argument; a fresh list per step would be a fresh graph node.
    """
    script = _learner_script(SEGMENTS_YAML)

    assert "_seg_a_b = flax.nnx.List([a, b])" in script
    assert "self._models = {'a': a, 'b': b, 'c': c, '_seg_a_b': _seg_a_b}" in script
    assert "def _flow_optimizer_ab(_seg_a_b):\n        a, b = _seg_a_b" in script
    assert ")(_seg_a_b, wrt=Param)" in script
    assert ")(c, wrt=Param)" in script
    # The container is state, not a model: the trainer must not see it among the models.
    assert "return {'a': self._models['a'], 'b': self._models['b'], 'c': self._models['c']}" in script
    assert "return {'optimizer_ab': ['a', 'b'], 'optimizer_c': ['c']}" in script


def test_flax_learner_builds_inference_views_of_every_model() -> None:
    """Inference runs against views: arrays shared with the trained models, inference flags forced.

    `raise_if_not_found=False` keeps a model without dropout or normalization from failing the build.
    """
    script = _learner_script(LEARNER_YAML)

    assert (
        "self._views = {k: flax.nnx.view(v, raise_if_not_found=False, training=False, "
        "deterministic=True, use_running_average=True) for k, v in self.models.items()}"
    ) in script
    assert "return self._inference_step(self._views, x=x, y=y, **kwargs)" in script


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


def test_flax_learner_rejects_a_layer_named_like_a_module_container() -> None:
    """The container of a multi-module segment is a generated name a user layer may already hold.

    Both would key the same entry of the generated model dictionary, and the loser -- the user's
    model -- would silently never be trained.
    """
    raw = load_any(SEGMENTS_YAML)
    raw["LEARNERS"][1]["TRAINABLE_LAYERS"] = ["_seg_a_b"]
    raw["LEARNERS"][1]["FLOW"] = [
        [{"predictions": "x", "targets": "y"}, "errors_c", "mse"],
        ["eval: errors_c.mean()", "loss_c", None],
    ]
    raw["LEARNERS"][1]["INFERENCE_FLOW"] = raw["LEARNERS"][1]["FLOW"]

    with pytest.raises(SpecError, match='Duplicate variable name "_seg_a_b" for the module container'):
        FlaxLearnerBuilder(raw=raw, current_path=str(SEGMENTS_YAML))()


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

    assert "optimizers['optimizer'].update(model, _grads, value=loss)" in script


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
