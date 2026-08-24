"""Builder for Keras models."""

import ast
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, Any, ClassVar, Literal, Self

from pydantic import field_validator, model_validator
from structcast.core.exceptions import SpecError

from structcast_model.builders.auto_name import AutoName
from structcast_model.builders.base import (
    BaseLearnerBuilder,
    BaseModelBuilder,
    LayerIntermediate,
    LearnerIntermediate,
    OptimizerSegment,
)
from structcast_model.builders.schema import LearnerBehavior, Template, UserDefinedLearner
from structcast_model.builders.utils import optimizer_hash, statement_names, stored_names
from structcast_model.utils.base import unique

_STATEFUL_KERAS_LAYERS = frozenset(
    {
        "AlphaDropout",
        "BatchNormalization",
        "Dropout",
        "GaussianDropout",
        "GaussianNoise",
        "SpatialDropout1D",
        "SpatialDropout2D",
        "SpatialDropout3D",
        "SyncBatchNormalization",
    }
)
"""Keras layers that draw from a seed or update their own variables, plus every `Random*` preprocessing layer.

Matched by the name a layer is constructed under, so a user-defined class doing the same thing --
including one reached through `_file_` -- is invisible here and documented in `REFERENCE.md` instead.
"""


def _stateful_sublayer(expression: str) -> str | None:
    """Return the blocklisted Keras layer one emitted constructor expression builds, if it builds one.

    A `rate=0` member of the Dropout family is let through: `Dropout.call` is guarded by `self.rate > 0`,
    so it returns its input untouched and draws nothing, and the recomputation matches the first pass.
    That is the shape a stochastic-depth section takes when its rate is parametrized down to zero.
    """
    try:
        node = ast.parse(expression, mode="eval").body
    except SyntaxError:
        # A lambda or another expression form: not a constructor call, so not a layer to judge.
        return None
    if not isinstance(node, ast.Call):
        return None
    called = node.func
    name = called.attr if isinstance(called, ast.Attribute) else getattr(called, "id", "")
    if not (name in _STATEFUL_KERAS_LAYERS or name.startswith("Random")):
        return None
    rates = [k.value for k in node.keywords if k.arg == "rate"]
    if rates and all(isinstance(rate, ast.Constant) and rate.value == 0 for rate in rates):
        return None
    return name


def _stateful_sublayers(layer: LayerIntermediate) -> list[str]:
    """Collect the blocklisted layers anywhere under one layer, its nested TYPE/CFG sections included.

    Recursion is the point: a stochastic-depth section is a sublayer of the block that gets
    checkpointed, one level down, and recomputation reaches just as far as the forward pass does.
    """
    found: list[str] = []
    for value in layer.layers.values():
        if isinstance(value, LayerIntermediate):
            found += _stateful_sublayers(value)
        elif name := _stateful_sublayer(value):
            found.append(name)
    return found


class KerasLayerIntermediate(LayerIntermediate):
    """Intermediate representation of a Keras layer.

    Generates a ``keras.Layer`` subclass with a ``call`` method that accepts a ``training`` keyword argument,
    propagating it to every sub-layer call to support Keras's standard training/inference mode.

    Example:
        >>> from structcast_model.builders.keras import KerasLayerIntermediate
        >>> script = KerasLayerIntermediate(
        ...     classname="Unit",
        ...     imports={},
        ...     inputs=["x"],
        ...     outputs=["y"],
        ...     layers={},
        ...     flow=[("x", "y", None)],
        ...     inference_flow=[],
        ...     structured_output=False,
        ... )._get_layer_script("Unit", [])
        >>> "class Unit(keras.layers.Layer):" in script
        True
    """

    default_imports: ClassVar[dict[str, set[str | None]]] = {"keras": {None}}
    """Default imports for Keras layers."""

    @model_validator(mode="after")
    def _reject_stateful_sublayers(self) -> Self:
        """Refuse to checkpoint a layer whose subtree carries state the recomputation would advance.

        Here rather than in the builder hook because the whole subtree is only assembled by the time
        the intermediate exists: the sublayers of the sublayers are what the earlier import-name scan
        could not see.
        """
        if self.gradient_checkpointing is None or not (stateful := _stateful_sublayers(self)):
            return self
        raise SpecError(
            f'GRADIENT_CHECKPOINTING cannot be applied to a layer whose FLOW builds "{stateful[0]}", here or in '
            "one of its sublayers: keras.remat runs the wrapped body a second time in the backward pass, and a "
            "layer that draws from a seed or updates its own variables does it twice -- different gradients on "
            "the TensorFlow and PyTorch backends, a tracer error on JAX. Checkpoint a layer that holds no such "
            f"state, or parametrize {stateful[0]} down to a rate of 0, which draws nothing."
        )

    def _get_layer(self, layername: str) -> str:
        """Get the sub-layer with the given name."""
        return f"self.{layername}"

    def _forward_flow(self, flow: list[tuple[str, str, str | None]]) -> list[str]:
        """Generate call expressions that forward the ``training`` flag to sub-layers."""
        return [f"{o} = {self._get_layer(L)}({i}, training=training)" if L else f"{o} = {i}" for i, o, L in flow]

    def _get_layer_script(self, class_name: str, initialized_layers: list[str]) -> str:
        """Return the Python class script for a Keras layer."""
        indent = " " * 4
        sep = "\n" + indent * 2
        if self._forward_inference_flow:
            codes = [
                "if training:",
                *[indent + c for c in self._forward_training_flow],
                "else:",
                *[indent + c for c in self._forward_inference_flow],
            ]
        else:
            codes = self._forward_training_flow
        inputs = self._forward_inputs
        inputs += ", " if inputs else ""
        body = "call"
        wrapper = ""
        prologue = "super().__init__(**kwargs)"
        if self.gradient_checkpointing is not None:
            # Before the sub-layers are built, not at training time: `keras.layers.MultiHeadAttention`
            # caches the flash attention decision in its own `__init__`, so a later flip misses it.
            prologue += f"{sep}disable_flash_attention_for_remat()"
            # No base class: Keras reads the `call` signature to decide whether it forwards
            # `training` and how it maps a batch passed by name, and a `*args` base would erase both.
            # Not `_call_impl`: on the torch backend a Keras layer inherits `torch.nn.Module`, which
            # owns that name for its call dispatcher, so it is not this emission's to take.
            body = "_call_body"
            # The rematerialized callable takes the arrays positionally and reads the flags off the
            # closure: on the TensorFlow backend the custom gradient behind `keras.remat` refuses
            # keyword arguments outside eager execution, which is every compiled training step.
            remat = "keras.remat(lambda *arrays: self._call_body(*arrays, training=training, **kwargs))"
            wrapper = (
                f"    def call(self, {inputs}*, training = None, **kwargs):\n"
                f"        if training:\n"
                f"            return {remat}({self._forward_inputs})\n"
                f"        return self._call_body({inputs}training=training, **kwargs)\n\n"
            )
        return f"""\
class {class_name}(keras.layers.Layer):

    def __init__(self, **kwargs):
        {prologue}
        self.input_names = {self.inputs}
        self.input_shapes = {self.input_shapes}
        self.output_names = {self.outputs}
        {sep.join([f"{self._get_layer(v)}" for v in initialized_layers])}

{wrapper}    def {body}(self, {inputs}*, training = None, **kwargs):
        {sep.join(codes)}
        return {self._forward_outputs}
"""


@dataclass(kw_only=True, slots=True)
class KerasBuilder(BaseModelBuilder[KerasLayerIntermediate]):
    """Builder for Keras models.

    Generates Python scripts containing ``keras.Layer`` subclasses from a YAML template,
    following the same template-to-code pipeline as :class:`~structcast_model.builders.torch.TorchBuilder`.

    Example:
        >>> from structcast_model.builders.keras import KerasBuilder
        >>> raw = {
        ...     "INPUTS": ["x"],
        ...     "OUTPUTS": ["y"],
        ...     "FLOW": [["x", "y", {"_obj_": [["_addr_", "keras.layers.Dense"], ["_call_", {"units": 4}]]}]],
        ... }
        >>> built = KerasBuilder(raw=raw)(classname="TinyNet")
        >>> built.classname
        'TinyNet'
    """

    user_defined_layer_type: ClassVar[type[KerasLayerIntermediate]] = KerasLayerIntermediate

    def _resolve_gradient_checkpointing(
        self,
        imports: defaultdict[str, set[str | None]],
        config: bool | dict[str, Any],
    ) -> dict[str, str] | None:
        """Reject the mapping form: `keras.remat` takes the function alone.

        The sub-layers recomputation would run twice are rejected by
        `KerasLayerIntermediate._reject_stateful_sublayers`, which sees the whole subtree.

        The one import a checkpointed layer gains is the JAX flash attention guard its `__init__`
        calls, which no other layer imports (`REFERENCE.md`, "GRADIENT_CHECKPOINTING").
        """
        if config is False:
            return None
        if isinstance(config, dict) and config:
            raise SpecError(
                f"GRADIENT_CHECKPOINTING keyword arguments {sorted(config)} have no Keras equivalent: "
                "keras.remat takes the function alone, so set GRADIENT_CHECKPOINTING to true and drop them."
            )
        imports["structcast_model.keras.layers"].add("disable_flash_attention_for_remat")
        return {}


class KerasLearnerBehavior(LearnerBehavior):
    """Learner behavior configuration for Keras.

    The shared behavior minus `EXTRA`: a Keras optimizer applies gradients through
    `optimizer.apply(gradients, variables)`, which takes nothing else, so the field is rejected
    rather than silently dropped. There is no `CLIP` field either -- clipping is a keyword of the
    Keras optimizer (`clipnorm` / `clipvalue` / `global_clipnorm`), so it belongs in `OPTIMIZER`
    and `_reject_clip` says so, rather than leaving the inherited `extra="forbid"`
    to report a `CLIP` key as one more unpermitted extra input.
    """

    @model_validator(mode="before")
    @classmethod
    def _reject_clip(cls, data: Any) -> Any:
        """Point a `CLIP` key at the Keras optimizer keyword that replaces it.

        Before, not after: `extra="forbid"` rejects the key first, and its generic "Extra inputs are
        not permitted" says nothing about where clipping went on this backend.
        """
        if isinstance(data, Mapping) and "CLIP" in data:
            raise SpecError(
                "CLIP has no Keras equivalent: a Keras optimizer clips its own gradients, so configure clipnorm, "
                "clipvalue or global_clipnorm in the OPTIMIZER pattern instead of adding a CLIP step."
            )
        return data

    @field_validator("EXTRA", mode="after")
    @classmethod
    def _reject_extra(cls, data: dict[str, Any]) -> dict[str, Any]:
        """Reject `EXTRA`, which nothing in the Keras update path could consume."""
        if data:
            raise SpecError(
                f"EXTRA has no Keras equivalent but got: {sorted(data)}. A Keras optimizer applies gradients "
                "through optimizer.apply(gradients, variables), which takes no further arguments: configure the "
                "optimizer itself in OPTIMIZER instead."
            )
        return data


class KerasUserDefinedLearner(UserDefinedLearner[KerasLearnerBehavior]):
    """User defined learner configuration for Keras."""

    MIXED_PRECISION: bool | dict[str, Any] = False
    """Whether to train under a `keras.mixed_precision` policy of `MIXED_PRECISION_TYPE`.

    A dictionary enables it too and carries the keyword arguments of the `keras.optimizers.LossScaleOptimizer`
    that a `float16` policy wraps every optimizer in; a `bfloat16` policy needs no scaling and stays unwrapped.
    The policy itself is set by the training CLI before the models are built, never by the generated learner.
    """

    MIXED_PRECISION_TYPE: Literal["bfloat16", "float16"] | None = None
    """The element type of the mixed precision policy, required exactly when `MIXED_PRECISION` is enabled."""

    @model_validator(mode="after")
    def _validate_mixed_precision(self) -> Self:
        """Require the two mixed precision fields to agree, since either one alone does nothing.

        `MIXED_PRECISION` selects the global policy and cannot pick an element type
        by itself; a type without it names a policy nobody would set.
        """
        enabled = bool(self.MIXED_PRECISION) if isinstance(self.MIXED_PRECISION, bool) else True
        if enabled and self.MIXED_PRECISION_TYPE is None:
            raise SpecError(
                "MIXED_PRECISION selects a keras.mixed_precision policy, which has no default element type: "
                "set MIXED_PRECISION_TYPE to float16 (loss scaled) or bfloat16 (unscaled)."
            )
        if not enabled and self.MIXED_PRECISION_TYPE is not None:
            raise SpecError(
                "MIXED_PRECISION_TYPE names the policy MIXED_PRECISION turns on, so alone it would be dropped: "
                "set MIXED_PRECISION to true (or to the LossScaleOptimizer keyword arguments), or remove the type."
            )
        if isinstance(self.MIXED_PRECISION, dict) and self.MIXED_PRECISION and self.MIXED_PRECISION_TYPE != "float16":
            raise SpecError(
                f"MIXED_PRECISION keyword arguments {sorted(self.MIXED_PRECISION)} are "
                f"keras.optimizers.LossScaleOptimizer arguments, which only a float16 policy wraps the optimizers "
                f"in, but MIXED_PRECISION_TYPE is {self.MIXED_PRECISION_TYPE!r}: set it to float16, or drop the "
                "keyword arguments and set MIXED_PRECISION to true."
            )
        return self


class KerasTemplateLearner(Template[KerasUserDefinedLearner]):
    """Template for Keras user-defined learners."""

    target_type: ClassVar[type[KerasUserDefinedLearner]] = KerasUserDefinedLearner


@dataclass(kw_only=True, slots=True)
class KerasOptimizerSegment(OptimizerSegment):
    """One optimizer step of a Keras learner flow, carrying the digest of the pattern that built it."""

    optimizer_hash: str
    """The digest of the segment's `OPTIMIZER` pattern, emitted as `OPTIMIZER_HASHES`."""


class KerasLearnerIntermediate(LearnerIntermediate[KerasOptimizerSegment]):
    """Intermediate representation of a Keras learner.

    Every backend-specific mechanic -- how the loss is differentiated, how the optimizer is applied,
    how the step is compiled -- lives in the backend adapter the generated learner selects once,
    so the emitted script imports no framework beyond `keras` and branches on no
    backend. Each optimizer segment becomes one `_flow_<optimizer>` method written in `keras.ops`,
    handed to the adapter as the `flow` of a `_segment_<optimizer>` attribute; the adapter turns
    them into the compiled training step.

    The emitted module holds imports and the class alone, and the class keeps no anonymous
    collection: the constants are class attributes, every segment is a named attribute, the batch
    travels as named parameters and the views are properties assembling literal dictionaries.
    """

    mixed_precision: bool | dict[str, Any]
    """Whether the optimizers are wrapped for loss scaling, and with which keyword arguments.

    Emitted as a literal into the `prepare` call, so a `dict` value has to be plain data.
    """

    mixed_precision_type: str | None
    """The element type of the mixed precision policy, or `None` if mixed precision is not used."""

    default_imports: ClassVar[dict[str, set[str | None]]] = {
        "keras": {None},
        "structcast_model.keras.adapters": {"AdapterSegment", "select_backend_adapter"},
    }
    """Default imports for Keras learners; the generated learner calls these directly."""

    @cached_property
    def _segments(self) -> list[tuple[list[tuple[str, str, str | None]], KerasOptimizerSegment]]:
        """Split the training flow into the (flow steps, optimizer segment) pairs to emit in order."""
        segments: list[tuple[list[tuple[str, str, str | None]], KerasOptimizerSegment]] = []
        units: list[tuple[str, str, str | None]] = []
        for unit in self.flow:
            if isinstance(unit, OptimizerSegment):
                segments.append((units, unit))
                units = []
            else:
                units.append(unit)
        return segments

    @cached_property
    def _criteria(self) -> dict[str, list[str]]:
        """Get the criteria each segment reports, keyed by optimizer name.

        The adapter merges the segments' criteria last-wins and cannot see a clash, so two segments
        reporting the same name -- one of the two values silently lost -- is rejected here.
        """
        owners: dict[str, str] = {}
        criteria: dict[str, list[str]] = {}
        for units, segment in self._segments:
            if segment.optimizer == "inference":
                raise SpecError(
                    'An optimizer named "inference" emits a _flow_inference method that collides with the '
                    "learner's own inference flow, so one of the two definitions would silently replace the "
                    "other: rename the optimizer."
                )
            names = [n for n in unique([n for _, o, _ in units for n in stored_names(o)]) if n in self.outputs]
            for name in names:
                if name in owners:
                    raise SpecError(
                        f'Criterion "{name}" is computed by both optimizer "{owners[name]}" and optimizer '
                        f'"{segment.optimizer}". The training step merges the criteria of its segments, so one of '
                        "the two values would be dropped without a word: rename one of them."
                    )
                owners[name] = segment.optimizer
            criteria[segment.optimizer] = names
        return criteria

    @property
    def _flow_parameters(self) -> str:
        """The batch parameters of a flow method: one keyword-only parameter per input name.

        Keyword-only because every caller of a flow -- the adapters, the distributed strategy --
        passes the batch by name, and a positional batch would silently take the entries in
        declaration order.
        """
        return f", *, {self._forward_inputs}" if self.inputs else ""

    def _flow_step(self, inputs: str, output: str, layer: str | None, *, training: bool) -> str:
        """Emit one flow step, running a model in the mode this flow needs it in.

        Only the models take a `training` flag: the other flow layers are losses and metrics, whose
        `__call__` has no such argument.
        """
        if layer is None:
            return f"{output} = {inputs}"
        arguments = ", ".join(p for p in (inputs, f"training={training}" if layer in self.models else "") if p)
        return f"{output} = self.{layer}({arguments})"

    def _get_forward_training_flow(self) -> list[str]:
        """Get the `_flow_<optimizer>` method of every segment, indented one level into the class body."""
        indent = " " * 4
        # A segment is one function the adapter calls with the batch alone, so a value another
        # segment computed is simply not in scope there.
        elsewhere = {name for units, _ in self._segments for _, output, _ in units for name in stored_names(output)}
        lines: list[str] = []
        for units, segment in self._segments:
            owned = segment.trainable_layers
            body = [self._flow_step(i, o, L, training=L in owned) for i, o, L in units]
            bound = set(self.inputs)
            for line in body:
                loads, stores = statement_names(line)
                if unbound := sorted(loads & (elsewhere - bound)):
                    raise SpecError(
                        f'Optimizer "{segment.optimizer}" reads "{unbound[0]}" before its own FLOW stores it. '
                        "A Keras segment is one function the backend adapter calls on the batch alone, so each "
                        "segment has to compute every value it reads: move that step into this segment's FLOW."
                    )
                bound |= stores
            criteria = ", ".join(f"{name!r}: {name}" for name in self._criteria[segment.optimizer])
            body.append(f"return {segment.loss}, {{{criteria}}}")
            lines.append(f"def _flow_{segment.optimizer}(self{self._flow_parameters}):")
            lines += [f"{indent}{line}" for line in body]
            lines.append("")
        return lines

    def _get_forward_inference_flow(self) -> list[str]:
        """Get the body of the `_flow_inference` method, which runs every model in inference mode."""
        lines = [self._flow_step(i, o, L, training=False) for i, o, L in self.inference_flow]
        return [*lines, f"return {self._forward_outputs}"]

    def _get_learner_script(self, initialized_layers: dict[str, str]) -> str:
        """Get the script for the learner: one class whose backend half is the adapter's."""
        indent = " " * 4
        sep2 = "\n" + indent * 2
        sep3 = "\n" + indent * 3
        inputs = self._forward_inputs
        inputs += ", " if inputs else ""
        named = ", ".join(f"{name}={name}" for name in self.inputs)
        every_model = ", ".join(f"self.{name}" for name in self.models)
        flows = "\n".join(indent + line if line else "" for line in self._forward_training_flow)
        attributes = [f"self._segment_{segment.optimizer}" for _, segment in self._segments]
        listed = f"[{', '.join(attributes)}]"
        tupled = f"({', '.join(attributes)},)"
        body = [f"self.{name} = {name}" for name in self.models]
        body += [f"self.{k} = {v}" for k, v in initialized_layers.items()]
        body += [f"{k} = {v}" for k, v in self.others.items() if k != v]
        for _, segment in self._segments:
            owned = ", ".join(f"self.{name}" for name in segment.trainable_layers)
            variables = (
                f"list({owned}.trainable_variables)"
                if len(segment.trainable_layers) == 1
                else f"[v for m in ({owned},) for v in m.trainable_variables]"
            )
            fields = [
                f"name={segment.optimizer!r},",
                f"flow=self._flow_{segment.optimizer},",
                f"optimizer={segment.optimizer},",
                f"variables={variables},",
                f"models=[{owned}],",
            ]
            body.append(f"self._segment_{segment.optimizer} = AdapterSegment({sep3}{sep3.join(fields)}{sep2})")
        body.append("adapter = select_backend_adapter()")
        body.append(
            f"adapter.prepare({listed}, mixed_precision={self.mixed_precision!r}, "
            f"mixed_precision_type={self.mixed_precision_type!r})"
        )
        # After `prepare`: under a float16 policy the accumulation window is the wrapped inner
        # optimizer's, and the wrapping is final by now.
        body.append(f'inners = [getattr(s.optimizer, "inner_optimizer", s.optimizer) for s in {tupled}]')
        # The counters answer for the whole learner, so the optimizers must agree on one window
        # (`docs/adr/0017`) -- a ValueError, since generated scripts import no builder errors.
        body.append("windows = sorted({inner.gradient_accumulation_steps or 1 for inner in inners})")
        body.append(
            f'if len(windows) > 1:{sep3}raise ValueError(f"One learner, one update window: the optimizers '
            'disagree on gradient_accumulation_steps {windows}.")'
        )
        body.append("self._steps = 0")
        body.append("self._last_updates = 0")
        body.append("self._has_updated = False")
        body.append(f"self._training_step = adapter.build_train_step({listed})")
        body.append(
            f"self._inference_step = adapter.build_inference_step(self._flow_inference, models=[{every_model}])"
        )
        body.append(f"self.inputs = {self.inputs}")
        body.append(f"self.outputs = {self.outputs}")
        # The first segment is the learner's clock, read through the segment because `prepare`
        # replaced the optimizer of a segment it wrapped for loss scaling (`docs/adr/0019`). One
        # expression, no local: every name in a step's namespace is a batch input the user named.
        clock = (
            f"int(keras.ops.convert_to_numpy({sep3}getattr({attributes[0]}.optimizer, "
            f'"inner_optimizer", {attributes[0]}.optimizer).iterations{sep2}))'
        )
        models = ", ".join(f"{name!r}: self.{name}" for name in self.models)
        optimizers = ", ".join(f"{s.optimizer!r}: self._segment_{s.optimizer}.optimizer" for _, s in self._segments)
        optimizer_models = ", ".join(f"{s.optimizer!r}: {s.trainable_layers!r}" for _, s in self._segments)
        hashes = ", ".join(f"{s.optimizer!r}: {s.optimizer_hash!r}" for _, s in self._segments)
        return f"""\
class {self.classname}:
    \"\"\"Learner generated from a Keras learner template.

    One learner drives every Keras backend: the flows below are written in `keras.ops` alone, and
    the backend adapter selected in `__init__` owns the gradients, the optimizer application and the
    step compilation. The steps are built once here, so `prepare` -- which builds every optimizer
    against its variables and wraps it in a `keras.optimizers.LossScaleOptimizer` under a float16
    policy -- has run before the training step is compiled, as it must.

    The batch travels by name: the flows and the steps below take one keyword argument per input, so
    whoever rebinds a step -- the distributed strategy replicating it across devices, above all --
    hands the batch on as keyword arguments too.

    The `keras.mixed_precision` global policy is deliberately not set here: it has to be in place
    before the models are built, which happens before this learner exists, so the training CLI sets
    it and this learner only tells the adapter what it is.

    A model runs with `training=True` only inside the segment that owns it, and with
    `training=False` everywhere else, so a frozen model updates no normalization statistics.
    Learner-level flow layers -- the losses and metrics of the FLOW -- must be stateless: only the
    models' variables are threaded through a compiled step, so a variable held by such a layer would
    freeze at its first value on the JAX backend.

    Gradient accumulation is the optimizer's (`gradient_accumulation_steps` in the OPTIMIZER
    pattern). The learner owns the training counters: `steps` counts every `training_step` call on
    the host, while `updates` and `has_updated` come from a post-step read of the first segment's
    optimizer counter -- detection, not prediction, so a float16 loss-scale skip, which freezes the
    counter, truthfully reports no update. `restore_counters` re-seeds `steps` after a checkpoint
    restore and re-baselines the counter read.
    \"\"\"

    # Read by the training CLI off the class, after this module is imported and before the class is
    # instantiated: the `keras.mixed_precision` global policy has to be in place before the models
    # the learner receives are built.
    MIXED_PRECISION = {self.mixed_precision!r}
    MIXED_PRECISION_TYPE = {self.mixed_precision_type!r}
    # Read by the training CLI when a run resumes: the digest of each segment's OPTIMIZER pattern,
    # so a state saved under another optimizer is reported instead of silently continued.
    OPTIMIZER_HASHES = {{{hashes}}}

    def __init__(self, {self._learner_models}, **kwargs):
        {sep2.join(body)}

{flows}
    def _flow_inference(self{self._flow_parameters}):
        {sep2.join(self._forward_inference_flow)}

    def training_step(self, {inputs}**kwargs):
        self._steps += 1
        res = self._training_step({named})
        # A genuine post-step read: the adapter has assigned the optimizer variables back by now, so
        # under a float16 loss-scale skip the frozen counter truthfully reports no update. The
        # public `iterations` already counts completed windows, accumulated or not. It is a host
        # read on every step and cannot be deferred -- `has_updated` answers for the step that just
        # ran -- but it is not an extra wait: the tracker reads this step's criteria back right
        # after, on the same computation. Under an accumulation window it costs one device division.
        current = {clock}
        self._has_updated = current > self._last_updates
        self._last_updates = current
        return res

    def inference_step(self, {inputs}**kwargs):
        return self._inference_step({named})

    def restore_counters(self, steps: int, updates: int) -> None:
        # `updates` is ignored on purpose: the restored optimizer variables already carry the
        # count, and re-reading it here keeps the two sources from ever disagreeing.
        self._steps = steps
        self._last_updates = {clock}

    @property
    def steps(self):
        return self._steps

    @property
    def updates(self):
        return self._last_updates

    @property
    def has_updated(self):
        return self._has_updated

    @property
    def models(self):
        return {{{models}}}

    @property
    def optimizers(self):
        return {{{optimizers}}}

    @property
    def optimizer_models(self):
        return {{{optimizer_models}}}

    @property
    def flow_functions(self):
        return {{"_training_step": self._training_step, "_inference_step": self._inference_step}}

    @property
    def learning_rates(self):
        return {{s.name: float(keras.ops.convert_to_numpy(s.optimizer.learning_rate)) for s in {tupled}}}
"""


@dataclass(kw_only=True, slots=True)
class KerasLearnerBuilder(BaseLearnerBuilder[KerasLearnerIntermediate]):
    """Builder for Keras learners.

    The `OPTIMIZER` pattern is emitted as written -- a Keras optimizer already carries its clipping,
    its weight decay and its schedule, and reports its own learning rate -- and the backend adapter
    builds it against the variables of the segment it belongs to.
    """

    user_defined_learner_layer_type: ClassVar[type[KerasLearnerIntermediate]] = KerasLearnerIntermediate
    layer_builder_type: ClassVar[type[KerasBuilder]] = KerasBuilder
    template_type: ClassVar[type[KerasTemplateLearner]] = KerasTemplateLearner

    def _intermediate_fields(self, module: KerasUserDefinedLearner) -> dict[str, Any]:
        """Get the framework-specific fields of the built learner intermediate."""
        return {"mixed_precision": module.MIXED_PRECISION, "mixed_precision_type": module.MIXED_PRECISION_TYPE}

    def _build_segment(  # noqa: PLR0913, PLR0917  # The base signature, narrowed to the Keras schema.
        self,
        imports: defaultdict[str, set[str | None]],
        module: Any,
        learner: LearnerBehavior,
        opt_name: str,
        naming: AutoName,
        layers: dict[str, LayerIntermediate | str],
        others: dict[str, str],
    ) -> KerasOptimizerSegment:
        """Build the segment, recording the digest of the optimizer pattern it was built from."""
        # Named base rather than a zero-argument `super()`: `slots=True` rebuilds the class, and on
        # Python below 3.12.4 -- inside the project floor -- the `__class__` cell still points at the
        # discarded one, so `super()` raises here, exactly as in the Flax builder.
        base = BaseLearnerBuilder._build_segment(self, imports, module, learner, opt_name, naming, layers, others)
        return KerasOptimizerSegment(
            loss=base.loss,
            optimizer=base.optimizer,
            trainable_layers=base.trainable_layers,
            optimizer_hash=optimizer_hash(learner.OPTIMIZER),
        )


__all__ = [
    "KerasBuilder",
    "KerasLayerIntermediate",
    "KerasLearnerBehavior",
    "KerasLearnerBuilder",
    "KerasLearnerIntermediate",
    "KerasOptimizerSegment",
    "KerasTemplateLearner",
    "KerasUserDefinedLearner",
]


if not TYPE_CHECKING:
    import sys

    from structcast.utils.lazy_import import LazySelectedImporter

    sys.modules[__name__] = LazySelectedImporter(__name__, globals())
