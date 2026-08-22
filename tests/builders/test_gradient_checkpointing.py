"""Tests for the shared `GRADIENT_CHECKPOINTING` field and the three emissions it drives.

The field is one `UserDefinedLayer` member across the three frameworks, validated per framework
against the keywords of that framework's mechanism (`docs/adr/0020`). What is asserted here is what
a user of a build can see: which class a generated layer inherits, which imports the module gains,
which configurations stay separate classes -- and, above all, that a layer that does not ask for
checkpointing is emitted exactly as it was before the field existed.
"""

from typing import Any

import pytest
from structcast.core.exceptions import SpecError

from structcast_model.builders.base import BaseModelBuilder, LayerIntermediate
from structcast_model.builders.flax import FlaxBuilder
from structcast_model.builders.keras import KerasBuilder
from structcast_model.builders.torch import TorchBuilder

# The `{_call_: {...}}` mapping form, so the pinned text below is code that runs: the list form
# `["_call_", {...}]` emits the keywords as one positional dict, which no framework layer accepts.
TORCH_RAW: dict[str, Any] = {
    "INPUTS": ["x"],
    "OUTPUTS": ["y"],
    "FLOW": [["x", "y", "fc", {"_obj_": [["_addr_", "torch.nn.LazyLinear"], {"_call_": {"out_features": 2}}]}]],
}

FLAX_LINEAR = {
    "_obj_": [
        ["_addr_", "flax.nnx.Linear"],
        {"_call_": {"in_features": 4, "out_features": 2, "rngs": "eval: rngs"}},
    ]
}

FLAX_RAW: dict[str, Any] = {"INPUTS": ["x"], "OUTPUTS": ["y"], "FLOW": [["x", "y", "fc", FLAX_LINEAR]]}

KERAS_RAW: dict[str, Any] = {
    "INPUTS": ["x"],
    "OUTPUTS": ["y"],
    "FLOW": [["x", "y", "fc", {"_obj_": [["_addr_", "keras.layers.Dense"], {"_call_": {"units": 2}}]}]],
}

TORCH_DISABLED = (
    "class Model(torch.nn.Module):\n"
    "\n"
    "    def __init__(self):\n"
    "        super().__init__()\n"
    "        self.inputs = ['x']\n"
    "        self.input_shapes = {}\n"
    "        self.outputs = ['y']\n"
    "        self.fc = LazyLinear(out_features=2)\n"
    "\n"
    "    def forward(self, x, **kwargs):\n"
    "        y = self.fc(x)\n"
    "        return y\n"
)

FLAX_DISABLED = (
    "class Model(flax.nnx.Module):\n"
    "\n"
    "    def __init__(self, *, rngs: flax.nnx.Rngs, training: bool = True):\n"
    "        self.inputs = ['x']\n"
    "        self.input_shapes = {}\n"
    "        self.outputs = ['y']\n"
    "        self.training = training\n"
    "        self.fc = Linear(in_features=4, out_features=2, rngs=rngs)\n"
    "\n"
    "    def __call__(self, x, *, training = None, **kwargs):\n"
    "        training = self.training if training is None else training\n"
    "        y = self.fc(x)\n"
    "        return y\n"
    "\n"
    "    def set_view(self, training = None):\n"
    "        if training is not None:\n"
    "            self.training = training\n"
)

KERAS_DISABLED = (
    "class Model(keras.layers.Layer):\n"
    "\n"
    "    def __init__(self, **kwargs):\n"
    "        super().__init__(**kwargs)\n"
    "        self.input_names = ['x']\n"
    "        self.input_shapes = {}\n"
    "        self.output_names = ['y']\n"
    "        self.fc = Dense(units=2)\n"
    "\n"
    "    def call(self, x, *, training = None, **kwargs):\n"
    "        y = self.fc(x, training=training)\n"
    "        return y\n"
)


def _script(builder: Any, raw: dict[str, Any], checkpointing: Any = None) -> str:
    """Build one layer and return the class script the builder emits for it."""
    if checkpointing is not None:
        raw = {**raw, "GRADIENT_CHECKPOINTING": checkpointing}
    return builder(raw=raw)(classname="Model").scripts[0]


@pytest.mark.parametrize(
    ("builder", "raw", "expected"),
    [
        (TorchBuilder, TORCH_RAW, TORCH_DISABLED),
        (FlaxBuilder, FLAX_RAW, FLAX_DISABLED),
        (KerasBuilder, KERAS_RAW, KERAS_DISABLED),
    ],
    ids=["torch", "flax", "keras"],
)
def test_a_layer_that_does_not_ask_for_checkpointing_is_emitted_unchanged(
    builder: Any, raw: dict[str, Any], expected: str
) -> None:
    """The default must cost existing users nothing: not a base class, not an import, not a line.

    Pinned as literal text rather than as "false equals absent", which a change breaking both forms
    the same way would still pass.
    """
    assert _script(builder, raw) == expected
    assert _script(builder, raw, False) == expected


def test_a_checkpointed_torch_layer_inherits_the_runtime_base() -> None:
    """The options land as class attributes of the layer itself, so no wrapper module is inserted."""
    script = _script(TorchBuilder, TORCH_RAW, {"use_reentrant": False, "determinism_check": "constant:none"})
    assert script.startswith("class Model(GradientCheckpointingLayer):")
    assert "    gradient_checkpointing = True\n" in script
    assert "    _checkpoint_kwargs = {'use_reentrant': False, 'determinism_check': 'none'}\n" in script
    # The forward body is untouched: the base intercepts `__call__`, which reaches `forward` anyway.
    assert "    def forward(self, x, **kwargs):\n        y = self.fc(x)\n" in script


def test_the_non_reentrant_checkpoint_is_filled_in_for_torch() -> None:
    """Torch warns on every call when `use_reentrant` is left out, and promises to raise later on.

    The reentrant variant differentiates only what it was handed positionally, so it is the wrong
    default for a layer whose batch may arrive by name.
    """
    assert "    _checkpoint_kwargs = {'use_reentrant': False}\n" in _script(TorchBuilder, TORCH_RAW, True)
    explicit = _script(TorchBuilder, TORCH_RAW, {"use_reentrant": True})
    assert "    _checkpoint_kwargs = {'use_reentrant': True}\n" in explicit


def test_a_checkpointed_flax_module_emits_its_body_as_forward() -> None:
    """`nnx.remat` needs the body under a name of its own, and a policy named as a string is JAX's."""
    script = _script(FlaxBuilder, FLAX_RAW, {"policy": "dots_saveable", "prevent_cse": False})
    assert script.startswith("class Model(GradientCheckpointingModule):")
    assert "    _remat_kwargs = {'policy': jax.checkpoint_policies.dots_saveable, 'prevent_cse': False}\n" in script
    assert "    def _forward(self, x, *, training = None, **kwargs):\n" in script
    assert "def __call__" not in script


def test_a_checkpointed_keras_layer_wraps_its_call_body() -> None:
    """Only the training call is rematerialized; inference runs the body directly.

    The body is `_call_body`, never `_call_impl`: a Keras layer inherits `torch.nn.Module` on the
    torch backend, which owns that name for its call dispatcher.
    """
    script = _script(KerasBuilder, KERAS_RAW, True)
    # The arrays go positionally and the flags through the closure: the TensorFlow custom gradient
    # behind `keras.remat` takes no keyword arguments outside eager execution.
    assert (
        "        if training:\n"
        "            return keras.remat(lambda *arrays: self._call_body(*arrays, training=training, **kwargs))(x)\n"
    ) in script
    assert "        return self._call_body(x, training=training, **kwargs)\n" in script
    assert "    def _call_body(self, x, *, training = None, **kwargs):\n" in script


@pytest.mark.parametrize(
    ("builder", "raw", "module", "name"),
    [
        (TorchBuilder, TORCH_RAW, "structcast_model.torch.layers", "GradientCheckpointingLayer"),
        (FlaxBuilder, FLAX_RAW, "structcast_model.flax.layers", "GradientCheckpointingModule"),
    ],
    ids=["torch", "flax"],
)
def test_the_runtime_base_is_imported_only_by_the_modules_that_use_it(
    builder: Any, raw: dict[str, Any], module: str, name: str
) -> None:
    """A per-instance import, not a default one: an unchecked model must import nothing new."""
    assert module not in builder(raw=raw)(classname="Model").collected_imports
    assert builder(raw={**raw, "GRADIENT_CHECKPOINTING": True})(classname="Model").collected_imports[module] == {name}


def test_a_flax_policy_name_pulls_jax_into_the_generated_module() -> None:
    """The emitted attribute reference is only valid if `jax` itself is imported alongside it."""
    built = FlaxBuilder(raw={**FLAX_RAW, "GRADIENT_CHECKPOINTING": {"policy": "nothing_saveable"}})(classname="Model")
    assert built.collected_imports["jax"] == {None}


def test_an_unknown_torch_option_is_rejected_by_name() -> None:
    """A misspelled keyword would otherwise surface as a TypeError on the first backward pass."""
    with pytest.raises(SpecError, match='"use_reentrance" is not a keyword argument'):
        TorchBuilder(raw={**TORCH_RAW, "GRADIENT_CHECKPOINTING": {"use_reentrance": False}})()


def test_an_unknown_flax_option_is_rejected_by_name() -> None:
    """`nnx.remat` takes three keywords, and the message names the one that is not among them."""
    with pytest.raises(SpecError, match='"policies" is not a keyword argument'):
        FlaxBuilder(raw={**FLAX_RAW, "GRADIENT_CHECKPOINTING": {"policies": "nothing_saveable"}})()


def test_keras_rejects_a_non_empty_mapping() -> None:
    """`keras.remat` takes the function alone, so options given for it would be silently dropped."""
    with pytest.raises(SpecError, match="have no Keras equivalent"):
        KerasBuilder(raw={**KERAS_RAW, "GRADIENT_CHECKPOINTING": {"policy": "nothing_saveable"}})()


@pytest.mark.parametrize(
    "address",
    ["keras.layers.Dropout", "keras.layers.BatchNormalization", "keras.layers.RandomFlip"],
    ids=["dropout", "batch_norm", "random_preprocessing"],
)
def test_keras_refuses_to_checkpoint_a_layer_holding_recomputable_state(address: str) -> None:
    """`keras.remat` runs the body twice, and these sub-layers do their state change twice with it.

    Silently different gradients on the TensorFlow and PyTorch backends, a tracer error on JAX --
    none of which the user asked for by writing one boolean, so the build stops instead.
    """
    raw = {
        **KERAS_RAW,
        "GRADIENT_CHECKPOINTING": True,
        "FLOW": [*KERAS_RAW["FLOW"], ["y", "y", "extra", {"_obj_": [["_addr_", address], {"_call_": {}}]}]],
    }
    with pytest.raises(SpecError, match=f'builds "{address.rsplit(".", 1)[1]}"'):
        KerasBuilder(raw=raw)()


def test_keras_checkpoints_a_layer_whose_flow_holds_no_such_state() -> None:
    """The blocklist is by name, so a stateless flow must not be caught by it."""
    assert "keras.remat" in _script(KerasBuilder, KERAS_RAW, True)


def _with_drop_path(rate: float) -> dict[str, Any]:
    """A checkpointed layer whose stochastic-depth sublayer is a `Dropout` one TYPE level down."""
    return {
        **KERAS_RAW,
        "GRADIENT_CHECKPOINTING": True,
        "FLOW": [*KERAS_RAW["FLOW"], ["y", "y", "drop_path", {"TYPE": "DropPath"}]],
        "DropPath": {
            "INPUTS": ["a"],
            "OUTPUTS": ["b"],
            "FLOW": [["a", "b", {"_obj_": [["_addr_", "keras.layers.Dropout"], {"_call_": {"rate": rate}}]}]],
        },
    }


def test_keras_refuses_a_stateful_layer_reached_through_a_nested_sublayer() -> None:
    """Recomputation reaches as far as the forward pass does, so the refusal has to reach as far too.

    The shipped Vision Transformer is exactly this shape -- a checkpointed block whose stochastic
    depth is a `Dropout` in a TYPE section -- and a scan of the block's own flow sees none of it.
    """
    with pytest.raises(SpecError, match='builds "Dropout", here or in one of its sublayers'):
        KerasBuilder(raw=_with_drop_path(0.1))()


def test_keras_allows_a_nested_dropout_parametrized_down_to_zero() -> None:
    """`Dropout.call` is guarded by `rate > 0`, so at zero it draws nothing and recomputes identically.

    Refusing it anyway would take activation checkpointing away from every template that switches its
    stochastic depth off by parameter, which is how the shipped recipes turn the two on together.
    """
    assert "keras.remat" in "\n".join(KerasBuilder(raw=_with_drop_path(0.0))().scripts)


def test_the_torch_selective_checkpoint_context_resolves_like_any_other_value() -> None:
    """`context_fn` is torch's counterpart of the flax policy, so a pattern has to survive to the script."""
    context = {
        "_obj_": [
            ["_addr_", "torch.utils.checkpoint.create_selective_checkpoint_contexts"],
            {"_bind_": {"policy_fn": "eval: []"}},
        ]
    }
    script = _script(TorchBuilder, TORCH_RAW, {"context_fn": context})
    assert "'context_fn': (lambda" in script
    assert "create_selective_checkpoint_contexts" in script


def test_the_base_builder_refuses_a_field_it_cannot_implement() -> None:
    """The base builder emits no framework module, so it has no mechanism to check the options against."""
    builder: BaseModelBuilder[LayerIntermediate] = BaseModelBuilder(raw={**TORCH_RAW, "GRADIENT_CHECKPOINTING": True})
    with pytest.raises(SpecError, match="GRADIENT_CHECKPOINTING names a framework mechanism"):
        builder()


def _unit(checkpointing: str) -> dict[str, Any]:
    """Reference the shared `Unit` sublayer with one checkpoint configuration."""
    return {"TYPE": "Unit", "PARAM": {"DEFAULT": {"checkpointing": checkpointing}}}


def test_layers_differing_only_in_checkpointing_stay_separate_classes() -> None:
    """The configuration is a field of the intermediate, so the content hash has to tell them apart.

    Four references to one sublayer, three configurations: collapsing them would give two of the
    three the checkpoint behavior of whichever was emitted first.
    """
    raw: dict[str, Any] = {
        "INPUTS": ["x"],
        "OUTPUTS": ["w"],
        "FLOW": [
            ["x", "y", "off", _unit("false")],
            ["y", "z", "on", _unit("true")],
            ["z", "v", "options", _unit("{debug: true}")],
            ["v", "w", "again", _unit("true")],
        ],
        "Unit": {
            "PARAMETERS": {"DEFAULT": {"checkpointing": "false"}},
            "_jinja_yaml_": "GRADIENT_CHECKPOINTING: {{checkpointing}}",
            "INPUTS": ["a"],
            "OUTPUTS": ["b"],
            "FLOW": [["a", "b", {"_obj_": [["_addr_", "torch.nn.LazyLinear"], {"_call_": {"out_features": 2}}]}]],
        },
    }
    scripts = "\n".join(TorchBuilder(raw=raw)(classname="Model").scripts)
    assert scripts.count("(torch.nn.Module):") == 2  # the unchecked Unit and the root model
    assert scripts.count("(GradientCheckpointingLayer):") == 2  # `true` reused by "again", plus the mapping


def test_the_field_configures_the_layer_and_never_becomes_a_sublayer() -> None:
    """A reserved key of the layer schema, so the template partition must not read it as a sublayer.

    It is a `model_fields` member, which is what tells `Template` a key belongs to the layer rather
    than to the user-named layers beside it -- and it takes template parameters like any other key.
    """
    raw: dict[str, Any] = {
        "PARAMETERS": {"DEFAULT": {"checkpointing": "true"}},
        "_jinja_yaml_": "GRADIENT_CHECKPOINTING: {{checkpointing}}",
        **TORCH_RAW,
    }
    builder = TorchBuilder(raw=raw)
    assert "GRADIENT_CHECKPOINTING" not in builder.user_defined_layers
    assert builder(classname="Model").gradient_checkpointing == {"use_reentrant": "False"}
    assert builder({"DEFAULT": {"checkpointing": "false"}}, classname="Model").gradient_checkpointing is None
