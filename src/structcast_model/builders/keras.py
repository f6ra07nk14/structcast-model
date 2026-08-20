"""Builder for Keras models."""

from collections import defaultdict
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
    optimizer_hash,
)
from structcast_model.builders.schema import LearnerBehavior, Template, UserDefinedLearner
from structcast_model.builders.utils import statement_names, stored_names
from structcast_model.utils.base import unique


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
        return f"""\
class {class_name}(keras.layers.Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.input_names = {self.inputs}
        self.input_shapes = {self.input_shapes}
        self.output_names = {self.outputs}
        {sep.join([f"{self._get_layer(v)}" for v in initialized_layers])}

    def call(self, {inputs}*, training = None, **kwargs):
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


class KerasLearnerBehavior(LearnerBehavior):
    """Learner behavior configuration for Keras.

    The shared behavior minus `EXTRA`: a Keras optimizer applies gradients through
    `optimizer.apply(gradients, variables)`, which takes nothing else, so the field is rejected
    rather than silently dropped. There is no `CLIP` field either -- clipping is a keyword of the
    Keras optimizer (`clipnorm` / `global_clipnorm`), so it belongs in `OPTIMIZER` and the inherited
    `extra="forbid"` rejects a `CLIP` key by construction (`docs/adr/0016`).
    """

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

        `MIXED_PRECISION` selects the global policy (`docs/adr/0016`) and cannot pick an element type
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
    """The digest of the segment's `OPTIMIZER` pattern, emitted as `__optimizer_hashes__`."""


class KerasLearnerIntermediate(LearnerIntermediate[KerasOptimizerSegment]):
    """Intermediate representation of a Keras learner.

    Every backend-specific mechanic -- how the loss is differentiated, how the optimizer is applied,
    how the step is compiled -- lives in the backend adapter the generated learner selects once
    (`docs/adr/0016`), so the emitted script imports no framework beyond `keras` and branches on no
    backend. Each optimizer segment becomes one `_flow_<optimizer>` method written in `keras.ops`,
    handed to the adapter as the `flow` of an `AdapterSegment`; the adapter turns them into the
    compiled training step.
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
            body = [f"{name} = batch[{name!r}]" for name in self.inputs]
            body += [self._flow_step(i, o, L, training=L in owned) for i, o, L in units]
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
            lines.append(f"def _flow_{segment.optimizer}(self, batch):")
            lines += [f"{indent}{line}" for line in body]
            lines.append("")
        return lines

    def _get_forward_inference_flow(self) -> list[str]:
        """Get the body of the `_flow_inference` method, which runs every model in inference mode."""
        lines = [f"{name} = batch[{name!r}]" for name in self.inputs]
        lines += [self._flow_step(i, o, L, training=False) for i, o, L in self.inference_flow]
        return [*lines, f"return {self._forward_outputs}"]

    def _get_learner_script(self, initialized_layers: dict[str, str]) -> str:
        """Get the script for the learner: one class whose backend half is the adapter's."""
        indent = " " * 4
        sep2 = "\n" + indent * 2
        sep3 = "\n" + indent * 3
        sep4 = "\n" + indent * 4
        inputs = self._forward_inputs
        inputs += ", " if inputs else ""
        batch = f"{{{', '.join(f'{name!r}: {name}' for name in self.inputs)}}}"
        every_model = ", ".join(f"self.{name}" for name in self.models)
        flows = "\n".join(indent + line if line else "" for line in self._forward_training_flow)
        segments: list[str] = []
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
            segments.append(f"AdapterSegment({sep4}{sep4.join(fields)}{sep3}),")
        body = [f"self.{name} = {name}" for name in self.models]
        body += [f"self.{k} = {v}" for k, v in initialized_layers.items()]
        body += [f"{k} = {v}" for k, v in self.others.items() if k != v]
        # Keras accumulates and gates the update inside the optimizer, and refuses a window of one.
        if (steps := self.accumulate_gradients or 1) > 1:
            body += [f"{name}.gradient_accumulation_steps = {steps}" for name in self.optimizers]
        body.append(f"self._models = {{{', '.join(f'{n!r}: self.{n}' for n in self.models)}}}")
        body.append(f"self._segments = [{sep3}{sep3.join(segments)}{sep2}]")
        body.append("adapter = select_backend_adapter()")
        body.append(
            f"adapter.prepare(self._segments, mixed_precision={self.mixed_precision!r}, "
            f"mixed_precision_type={self.mixed_precision_type!r})"
        )
        # After `prepare`: it replaces the optimizer of a segment it wrapped for loss scaling.
        body.append("self._optimizers = {segment.name: segment.optimizer for segment in self._segments}")
        body.append("self._training_step = adapter.build_train_step(self._segments)")
        body.append(
            f"self._inference_step = adapter.build_inference_step(self._flow_inference, models=[{every_model}])"
        )
        body.append(f"self.inputs = {self.inputs}")
        body.append(f"self.outputs = {self.outputs}")
        optimizer_models = ", ".join(f"{s.optimizer!r}: {s.trainable_layers!r}" for _, s in self._segments)
        hashes = ", ".join(f"{s.optimizer!r}: {s.optimizer_hash!r}" for _, s in self._segments)
        return f"""\
# Read by the training CLI after this module is imported and before the class below is
# instantiated: the `keras.mixed_precision` global policy has to be in place before the models the
# learner receives are built (`docs/adr/0016`).
__mixed_precision__ = {self.mixed_precision!r}
__mixed_precision_type__ = {self.mixed_precision_type!r}
# Read by the training CLI when a run resumes: the digest of each segment's OPTIMIZER pattern, so a
# state saved under another optimizer is reported instead of silently continued (`docs/adr/0015`).
__optimizer_hashes__ = {{{hashes}}}


class {self.classname}:
    \"\"\"Learner generated from a Keras learner template.

    One learner drives every Keras backend: the flows below are written in `keras.ops` alone, and
    the backend adapter selected in `__init__` owns the gradients, the optimizer application and the
    step compilation (`docs/adr/0016`). The steps are built once here, so `prepare` -- which builds
    every optimizer against its variables and wraps it in a `keras.optimizers.LossScaleOptimizer`
    under a float16 policy -- has run before the training step is compiled, as it must.

    The `keras.mixed_precision` global policy is deliberately not set here: it has to be in place
    before the models are built, which happens before this learner exists, so the training CLI sets
    it and this learner only tells the adapter what it is.

    A model runs with `training=True` only inside the segment that owns it, and with
    `training=False` everywhere else, so a frozen model updates no normalization statistics.
    Learner-level flow layers -- the losses and metrics of the FLOW -- must be stateless: only the
    models' variables are threaded through a compiled step, so a variable held by such a layer would
    freeze at its first value on the JAX backend.

    Gradient accumulation is the optimizer's (`gradient_accumulation_steps`), so `update` is
    trivially true: every step feeds the optimizer, which decides when the update lands.
    \"\"\"

    def __init__(self, {self._learner_models}, **kwargs):
        {sep2.join(body)}

{flows}
    def _flow_inference(self, batch):
        {sep2.join(self._forward_inference_flow)}

    def training_step(self, {inputs}**kwargs):
        return self._training_step({batch})

    def inference_step(self, {inputs}**kwargs):
        return self._inference_step({batch})

    def update(self, step: int) -> bool:
        return True

    @property
    def models(self):
        return self._models

    @property
    def optimizers(self):
        return self._optimizers

    @property
    def optimizer_models(self):
        return {{{optimizer_models}}}

    @property
    def flow_functions(self):
        return {{"_training_step": self._training_step, "_inference_step": self._inference_step}}

    @property
    def learning_rates(self):
        return {{k: float(keras.ops.convert_to_numpy(v.learning_rate)) for k, v in self._optimizers.items()}}
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
