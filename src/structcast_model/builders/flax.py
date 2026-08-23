"""Builder for Flax (nnx) models."""

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from functools import cached_property
from logging import getLogger
from typing import TYPE_CHECKING, Any, ClassVar, cast

from pydantic import Field, ValidationError
from structcast.core.exceptions import SpecError
from structcast.core.instantiator import ObjectPattern

from structcast_model.builders.auto_name import AutoName
from structcast_model.builders.base import (
    BaseLearnerBuilder,
    BaseModelBuilder,
    LayerIntermediate,
    LearnerIntermediate,
    OptimizerSegment,
)
from structcast_model.builders.schema import LearnerBehavior, Template, UserDefinedLearner
from structcast_model.builders.utils import (
    # Framework-neutral and shared with the Keras builder, re-exported here because a caller reading
    # a learner's `OPTIMIZER_HASHES` reaches for it next to the builder that emitted them.
    optimizer_hash,
    resolve_getter,
    resolve_object,
    statement_names,
    stored_names,
)
from structcast_model.utils.base import unique

logger = getLogger(__name__)


class FlaxLayerIntermediate(LayerIntermediate):
    """Intermediate representation of a Flax nnx module.

    Generates a ``flax.nnx.Module`` subclass whose ``__init__`` accepts a ``rngs: flax.nnx.Rngs`` argument
    (passed down to sub-module constructors via ``eval: rngs`` in the YAML template) and
    whose ``__call__`` accepts a ``training: bool`` keyword argument for toggling training vs. inference behaviour.

    Example:
        >>> from structcast_model.builders.flax import FlaxLayerIntermediate
        >>> script = FlaxLayerIntermediate(
        ...     classname="Unit",
        ...     imports={},
        ...     inputs=["x"],
        ...     outputs=["y"],
        ...     layers={},
        ...     flow=[("x", "y", None)],
        ...     inference_flow=[],
        ...     structured_output=False,
        ... )._get_layer_script("Unit", [])
        >>> "class Unit(flax.nnx.Module):" in script
        True
    """

    default_imports: ClassVar[dict[str, set[str | None]]] = {"flax.nnx": {None}}
    """Default imports for Flax nnx modules."""

    def _get_layer(self, layername: str) -> str:
        """Get the sub-module with the given name."""
        return f"self.{layername}"

    @classmethod
    def _get_class_instance(cls, classname: str) -> str:
        return f"{classname}(rngs=rngs, training=training)"

    def _get_layer_script(self, class_name: str, initialized_layers: list[str]) -> str:
        """Return the Python class script for a Flax nnx module."""
        indent = " " * 4
        sep = "\n" + indent * 2
        base, attributes, forward = "flax.nnx.Module", "", "__call__"
        if self.gradient_checkpointing is not None:
            # The base owns `__call__` and rematerializes the body it finds under `_forward`.
            base, forward = "GradientCheckpointingModule", "_forward"
            lines = ["gradient_checkpointing = True"]
            if self.gradient_checkpointing:
                keywords = ", ".join(f"{k!r}: {v}" for k, v in self.gradient_checkpointing.items())
                lines.append(f"_remat_kwargs = {{{keywords}}}")
            attributes = "".join(f"{indent}{line}\n" for line in lines) + "\n"
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
class {class_name}({base}):

{attributes}    def __init__(self, *, rngs: flax.nnx.Rngs, training: bool = True):
        self.inputs = {self.inputs}
        self.input_shapes = {self.input_shapes}
        self.outputs = {self.outputs}
        self.training = training
        {sep.join([f"{self._get_layer(v)}" for v in initialized_layers])}

    def {forward}(self, {inputs}*, training = None, **kwargs):
        training = self.training if training is None else training
        {sep.join(codes)}
        return {self._forward_outputs}

    def set_view(self, training = None):
        if training is not None:
            self.training = training
"""


_REMAT_OPTIONS = frozenset({"graph", "graph_updates", "policy", "prevent_cse", "static_argnums"})
"""The keyword arguments `flax.nnx.remat` accepts, which `GRADIENT_CHECKPOINTING` carries."""


@dataclass(kw_only=True, slots=True)
class FlaxBuilder(BaseModelBuilder[FlaxLayerIntermediate]):
    """Builder for Flax nnx models.

    Generates Python scripts containing ``flax.nnx.Module`` subclasses from a YAML template,
    following the same template-to-code pipeline as :class:`~structcast_model.builders.torch.TorchBuilder`.

    Sub-modules that require a random-number generator should receive ``rngs: "eval: rngs"`` in
    their ``_call_`` arguments so that the builder emits ``rngs=rngs`` in the generated ``__init__`` body.

    Example:
        >>> from structcast_model.builders.flax import FlaxBuilder
        >>> layer_spec = {"_obj_": [["_addr_", "flax.nnx.Linear"], {"_call_": {"in_features": 8, "out_features": 4}}]}
        >>> raw = {"INPUTS": ["x"], "OUTPUTS": ["y"], "FLOW": [["x", "y", layer_spec]]}
        >>> built = FlaxBuilder(raw=raw)(classname="TinyNet")
        >>> built.classname
        'TinyNet'
    """

    user_defined_layer_type: ClassVar[type[FlaxLayerIntermediate]] = FlaxLayerIntermediate

    def _resolve_gradient_checkpointing(
        self,
        imports: defaultdict[str, set[str | None]],
        config: bool | dict[str, Any],
    ) -> dict[str, str] | None:
        """Validate the mapping against the keywords of `flax.nnx.remat`."""
        if config is False:
            return None
        options = {} if isinstance(config, bool) else config
        if unknown := sorted(options.keys() - _REMAT_OPTIONS):
            raise SpecError(
                f'GRADIENT_CHECKPOINTING option "{unknown[0]}" is not a keyword argument of '
                f"flax.nnx.remat, which accepts {sorted(_REMAT_OPTIONS)}."
            )
        imports["structcast_model.flax.layers"].add("GradientCheckpointingModule")
        resolved: dict[str, str] = {}
        for key, value in options.items():
            if key == "policy" and isinstance(value, str):
                # A bare name is one of the policies JAX ships; anything else -- a pattern building a
                # parameterized policy, say -- resolves like any other DSL value.
                imports["jax"].add(None)
                resolved[key] = f"jax.checkpoint_policies.{value}"
            else:
                resolved[key] = resolve_getter(imports, value)
        return resolved


def _keywords(part: Any) -> dict[str, Any] | None:
    """Return the keyword arguments of a serialized `_call_`/`_bind_` pattern part, if it has any."""
    if isinstance(part, dict):
        for key in ("_call_", "_bind_"):
            if isinstance(part.get(key), dict):
                return part[key]
    return None


def _is_inject(key: Any, value: Any) -> bool:
    """Report whether one serialized entry is an address naming `inject_hyperparams`."""
    return key in ("_addr_", "_file_") and isinstance(value, str) and value.endswith("inject_hyperparams")


def _references_inject(node: Any) -> bool:
    """Report whether any address in the serialized pattern already names `inject_hyperparams`.

    Only addresses count. A plain string that happens to end in the name -- the label of a
    `optax.named_chain` entry, say -- names no transformation, so suppressing the rewrite on it
    would cost the run both its readable learning rate and the warning that says so.
    """
    if isinstance(node, dict):
        if any(_is_inject(key, value) for key, value in node.items()):
            return True
        return any(_references_inject(value) for value in node.values())
    if isinstance(node, (list, tuple)):
        # Addresses serialize either as `{"_addr_": ...}` or as the `["_addr_", ...]` list form.
        if node and any(_is_inject(node[0], value) for value in node[1:]):
            return True
        return any(_references_inject(value) for value in node)
    return False


def _wrap_children(values: Iterable[Any]) -> tuple[list[Any], int]:
    """Rewrite every child, returning the new children and how many factory calls were wrapped."""
    walked = [_wrap(value) for value in values]
    return [value for value, _ in walked], sum(count for _, count in walked)


def _wrap(node: Any) -> tuple[Any, int]:
    """Rewrite the rate-carrying factory calls nested anywhere under a serialized pattern node."""
    try:
        # `ObjectPattern` serializes to the `["_obj_", <part>, ...]` list its validator accepts back,
        # which the `model_dump` signature cannot express.
        dumped = cast(list[Any], ObjectPattern.model_validate(node).model_dump(by_alias=True))
    except ValidationError:
        if isinstance(node, dict):
            children, count = _wrap_children(node.values())
            return dict(zip(node, children, strict=True)), count
        if isinstance(node, (list, tuple)):
            children, count = _wrap_children(node)
            return type(node)(children), count
        return node, 0
    parts, count = _wrap_children(dumped[1:])
    index = next((i for i, part in enumerate(parts) if "learning_rate" in (_keywords(part) or {})), None)
    if index is None:
        return ["_obj_", *parts], count
    # `static_args` is the safety valve: without it inject arrayifies every numeric keyword, and
    # `bool` is an `int` subclass, so a flag like `nesterov=True` would reach the factory as `Array(1)`.
    static_args = [key for key in _keywords(parts[index]) or {} if key != "learning_rate"]
    arguments: dict[str, Any] = {"inner_factory": ["_obj_", *parts[:index]]}
    if static_args:
        arguments["static_args"] = static_args
    return ["_obj_", ["_addr_", "optax.inject_hyperparams"], {"_call_": arguments}, *parts[index:]], count + 1


def inject_learning_rate(optimizer: ObjectPattern) -> tuple[ObjectPattern, bool]:
    """Wrap the learning-rate-carrying factory of an optimizer pattern in `optax.inject_hyperparams`.

    Optax exposes a learning rate only through the `hyperparams` dict that `inject_hyperparams`
    materializes, so a pattern emitted verbatim would report NaN for the rest of the run.
    The unique factory call carrying a `learning_rate` keyword -- at any depth,
    including inside a `chain` -- is therefore rewritten from `adamw(learning_rate=..., b1=...)` to
    `inject_hyperparams(inner_factory=adamw, static_args=['b1'])(learning_rate=..., b1=...)`. Every
    other part of the pattern, such as the `nnx.Optimizer` binding and the surrounding chain, is
    preserved.

    Args:
        optimizer (ObjectPattern): The validated `OPTIMIZER` pattern of one learner behavior.

    Returns:
        tuple[ObjectPattern, bool]: The pattern to emit, and whether its learning rate will be
            readable at run time. The pattern is returned unchanged when it already injects
            (readable), and when no factory call or more than one carries a `learning_rate` keyword
            -- those two are not readable, and the caller is expected to warn about it.
    """
    dumped = optimizer.model_dump(by_alias=True)
    if _references_inject(dumped):
        return optimizer, True
    rewritten, count = _wrap(dumped)
    if count != 1:
        return optimizer, False
    return ObjectPattern.model_validate(rewritten), True


def _stores(unit: Any) -> list[str]:
    """Return the names one flow unit assigns, nothing for the optimizer segment separating two."""
    return [] if isinstance(unit, OptimizerSegment) else stored_names(unit[1])


def _owned(trainable_layers: list[str]) -> str:
    """Return the expression naming the modules one optimizer owns.

    A single owned module is passed to `flax.nnx.Optimizer` and to `update` verbatim, several as a
    plain tuple, so the optimizer state and the differentiated arguments key off the same module
    paths without a container node the learner would have to keep.
    """
    if len(trainable_layers) == 1:
        return trainable_layers[0]
    return f"({', '.join(trainable_layers)})"


_EMA_DEFAULTS: dict[str, Any] = {"decay": 0.999, "only": "eval: Param"}
"""What `flax.nnx.EMA` is given when nothing else is declared.

Its own default tracks every Variable, which blends the RNG counters and the batch statistics along
with the weights -- a key stream cannot even be multiplied by a float -- so the average is taken over
the parameters, exactly as the optimizers are built `wrt=Param`."""


class FlaxUserDefinedLearner(UserDefinedLearner[LearnerBehavior]):
    """User defined learner configuration for Flax."""

    EMA: dict[str, bool | dict[str, Any]] = Field(default_factory=dict)
    """The models an exponential moving average shadows, keyed by the model name.

    `true` takes the defaults; a mapping carries the keyword arguments of `flax.nnx.EMA`, each value
    resolved like any other DSL value. The average is emitted as the learner attribute `ema_<model>`
    -- the `apply_to` view, so it is callable -- updated once per Update and runnable from
    `INFERENCE_FLOW` under that name.
    """


class FlaxTemplateLearner(Template[FlaxUserDefinedLearner]):
    """Template for Flax user-defined learners."""

    target_type: ClassVar[type[FlaxUserDefinedLearner]] = FlaxUserDefinedLearner


@dataclass(kw_only=True, slots=True)
class FlaxOptimizerSegment(OptimizerSegment):
    """One optimizer step of a Flax learner flow, carrying the digest of the pattern that built it."""

    optimizer_hash: str
    """The digest of the segment's `OPTIMIZER` pattern, emitted as `OPTIMIZER_HASHES` in the learner."""


class FlaxLearnerIntermediate(LearnerIntermediate[FlaxOptimizerSegment]):
    """Intermediate representation of a Flax (nnx) learner.

    The generated module holds the imports and the learner class alone: the flow layers, the
    differentiated flow of every segment and the two steps are all built inside `__init__`, where no
    user-chosen name can collide with an import the model builder pulled in. The
    steps are plain functions over named parameters -- never closures over state or bound methods --
    so a trainer can wrap them in `flax.nnx.jit` and rebind each attribute `flow_functions` names,
    exactly as the PyTorch learners are compiled. Every model and optimizer is one
    positional-or-keyword parameter and the batch is keyword-only, which is the donation contract
    the CLI reads back off the signature.

    Each segment differentiates its own flow function with `flax.nnx.value_and_grad`: the modules it
    owns come first -- they are the `argnums`, and a segment owning several passes them as a plain
    tuple -- followed by the models it only reads and by the batch entries and earlier stores it
    needs. All of those are positional-or-keyword, `flax.nnx` resolving keyword arguments to
    positions itself. The flow returns as auxiliary output only what the enclosing step reads back --
    the criteria, the `EXTRA` keywords and whatever a later segment reads -- and keeps the rest local.
    Gradient accumulation is the optimizer pattern's own `optax.MultiSteps`, gating on the device;
    the step compares the first optimizer's `gradient_step` across its update and hands the result
    back, so the learner counts updates without predicting them from a window.

    Learner-level flow layers (losses and metrics) are locals of `__init__` the flows close over, so
    they must be stateless: a variable-carrying layer would be captured as a constant by a compiled
    step. The `EXTRA` keywords of a segment are forwarded to `flax.nnx.Optimizer.update`, which hands
    them to the transformation, as `optax.GradientTransformationExtraArgs` expects.
    """

    default_imports: ClassVar[dict[str, set[str | None]]] = {
        "jax": {None},
        "jax.numpy": {None},
        "flax.nnx": {None, "Param"},
        "structcast_model.flax.optimizers": {"get_learning_rate", "gradient_steps"},
    }
    """Default imports for Flax learners; the generated steps and properties call these directly."""

    _learner_members: ClassVar[frozenset[str]] = frozenset(
        {
            "flow_functions",
            "has_updated",
            "inference_step",
            "inputs",
            "learning_rates",
            "models",
            "optimizer_models",
            "optimizers",
            "outputs",
            "restore_counters",
            "steps",
            "training_step",
            "updates",
            "_has_updated",
            "_inference_step",
            "_learning_rates",
            "_steps",
            "_training_step",
            "_updates",
        }
    )
    """Every attribute and property the generated class defines, besides the per-model view attributes."""

    _step_locals: ClassVar[frozenset[str]] = frozenset({"_", "_before", "_grads", "_has_updated", "lrs"})
    """The names the generated training step binds for itself, between the flow calls it makes."""

    ema: tuple[str, ...] = ()
    """The models carrying an exponential moving average, in `EMA` declaration order.

    Each one becomes the `flax.nnx.EMA` state `_ema_state_<model>` and the callable view `ema_<model>`
    it applies to the model, built from the expression the builder registered under that name in
    `others`."""

    @cached_property
    def _inference_shadows(self) -> list[str]:
        """The models whose average the inference flow runs, in `EMA` declaration order.

        Only those reach `_inference_step`, as one more parameter each: an average the flow never
        names would be one more donated-looking argument for a trainer to compile around.
        """
        lines = [self._get_regular_step(i, o, L) for i, o, L in self.inference_flow]
        read = {name for line in lines for name in statement_names(line)[0]}
        return [name for name in self.ema if f"ema_{name}" in read]

    @cached_property
    def _segments(self) -> list[tuple[list[tuple[str, str, str | None]], FlaxOptimizerSegment]]:
        """Split the training flow into the (flow steps, optimizer segment) pairs to emit in order."""
        segments: list[tuple[list[tuple[str, str, str | None]], FlaxOptimizerSegment]] = []
        units: list[tuple[str, str, str | None]] = []
        for unit in self.flow:
            if isinstance(unit, FlaxOptimizerSegment):
                segments.append((units, unit))
                units = []
            else:
                units.append(unit)
        return segments

    @cached_property
    def _segment_bodies(self) -> list[tuple[list[str], list[str], list[str], list[str]]]:
        """Analyze each segment into its (model parameters, passed values, body lines, stored names).

        Everything a flow reads and does not compute itself is a parameter: the models it owns first,
        then the models it only reads, then the batch entries and the values earlier segments stored.
        Anything else it reads -- the flow layers, the imported helpers -- it reads from the enclosing
        scope, which is what keeps the parameter list to what the step actually has to hand over.

        A model is read wherever its name is, not only where the flow calls it as its layer: a model
        the enclosing scope also binds would otherwise be captured from `__init__` and frozen into
        the compiled step, so a flow reading a parameter of a model another segment trains would
        silently keep computing with the values that model had when the learner was built.

        A segment is emitted as one function, so every name it stores is local to its whole body: a
        step reading such a name before the step that stores it emits valid Python that raises
        `UnboundLocalError` on the first batch. That order is rejected here instead, unless the
        enclosing step can pass the value in -- a batch entry or an earlier store becomes a parameter,
        and the later store shadows it rather than being read before it exists.
        """
        bodies: list[tuple[list[str], list[str], list[str], list[str]]] = []
        available = set(self.inputs)
        for units, segment in self._segments:
            owned = segment.trainable_layers
            body = [self._get_regular_step(i, o, L) for i, o, L in units]
            read = unique([name for line in body for name in sorted(statement_names(line)[0])])
            models = [*owned, *[name for name in read if name in self.models and name not in owned]]
            stores = unique([name for _, output, _ in units for name in stored_names(output)])
            passed: list[str] = []
            bound = set(models)
            deferred = set(stores) - bound
            for line in body:
                loads, stored = statement_names(line)
                fresh = sorted((loads & available) - bound)
                passed += fresh
                bound |= set(fresh)
                if shadowed := sorted(loads & (deferred - bound)):
                    raise SpecError(
                        f'Optimizer "{segment.optimizer}" reads "{shadowed[0]}" before its own FLOW stores it. '
                        "A Flax segment is one function, so a name it stores is local to the whole "
                        "segment: compute the value before it is read, or give one of the two another name."
                    )
                bound |= stored
            bodies.append((models, passed, body, stores))
            available |= set(stores)
        return bodies

    @cached_property
    def _training_flow_parts(self) -> tuple[list[str], list[str]]:
        """Split training into the per-segment flow definitions and the body of `_training_step`.

        Both halves are emitted into `__init__`, the flow functions as siblings of the step rather
        than as nested definitions: a flow reads everything it differentiates through its own
        parameters, so nesting it inside the step would only hide which values that is.
        """
        bodies = self._segment_bodies
        extras = [f", {segment.backward_kwargs}" if segment.backward_kwargs else "" for _, segment in self._segments]
        updates = [set(statement_names(f"_update(_grads{extra})")[0]) for extra in extras]
        definitions: list[str] = []
        step: list[str] = []
        for index, (_, segment) in enumerate(self._segments):
            models, passed, body, stores = bodies[index]
            owned = segment.trainable_layers
            extra = extras[index]
            # Only what the enclosing step reads leaves the flow: the criteria, what any update
            # expression from here on reads -- the `EXTRA` keywords are evaluated in the step, so a
            # later one reads this flow's values there -- and what a later segment takes as a
            # parameter. Every other intermediate stays local, so a flow may compute values a traced
            # auxiliary output could not carry.
            needed = {segment.loss, *self.outputs, *[name for reads in updates[index:] for name in reads]}
            needed |= {name for later in bodies[index + 1 :] for name in later[1]}
            aux = [name for name in stores if name in needed]
            if segment.loss not in aux:
                raise SpecError(
                    f'Optimizer "{segment.optimizer}" cannot be differentiated: its FLOW does not compute its '
                    f'LOSS "{segment.loss}". A Flax segment only differentiates what its own flow computes.'
                )
            returns = f"({', '.join(aux)},)"
            definitions.append(f"def _flow_{segment.optimizer}({', '.join([*models, *passed])}):")
            definitions += [f"    {line}" for line in [*body, f"return {segment.loss}, {returns}"]]
            # The owned models are the leading parameters, so their positions are the `argnums`; the
            # default of 0 already names the single owned model of a one-module segment.
            argnums = f", argnums={tuple(range(len(owned)))}" if len(owned) > 1 else ""
            arguments = ", ".join([*models, *(f"{name}={name}" for name in passed)])
            grad = f"flax.nnx.value_and_grad(_flow_{segment.optimizer}{argnums}, has_aux=True)({arguments})"
            step.append(f"(_, {returns}), _grads = {grad}")
            if index == 0:
                step += [
                    "# The first optimizer is the learner's clock: with an optax.MultiSteps window the",
                    "# device decides which step an update lands on, so the count it advanced is read",
                    "# across this update. Without a window there is no counter and every step applies.",
                    f"_before = gradient_steps({segment.optimizer})",
                ]
            step.append(f"{segment.optimizer}.update({_owned(owned)}, _grads{extra})")
            if index == 0:
                step.append(
                    f"_has_updated = True if _before is None else gradient_steps({segment.optimizer}) > _before"
                )
        rates = ", ".join(f"{name!r}: get_learning_rate({name})" for name in self.optimizers)
        step += [
            "# Read at trace time: the walk compiles to a reference to the injected rate rather than to",
            "# a host read, which after the step would touch the state buffers the caller donated.",
            f"lrs = {{{rates}}}",
            f"return {self._forward_outputs}, lrs, _has_updated",
        ]
        return definitions, step

    def _get_forward_training_flow(self) -> list[str]:
        """Get the flow function definitions the `__init__` body opens with."""
        return self._training_flow_parts[0]

    def _get_forward_inference_flow(self) -> list[str]:
        """Get the body of the `_inference_step` function."""
        lines = [self._get_regular_step(i, o, L) for i, o, L in self.inference_flow]
        return [*lines, f"return {self._forward_outputs}"]

    def _reject_reserved_names(self) -> None:
        """Reject the names the generated learner cannot carry, before a broken script is written.

        Every model and optimizer is an attribute of the learner and a parameter of both steps, and
        the batch entries are their keyword-only parameters. A name shared with `self`, with
        `kwargs`, with a member of the class or with a view attribute emits a step that fails to
        import -- or, worse, an `__init__` that silently overwrites what the trainer reads off the
        learner afterwards.

        What a flow stores is bound in the enclosing step too, next to the names the step binds for
        itself: an output named like one of them would return the learning rates as a criterion, and
        one named like a model would hand the optimizer something other than the module it owns.
        """
        shadows = [f"ema_{name}" for name in self.ema]
        for unit in self.flow:
            if isinstance(unit, OptimizerSegment):
                continue
            if shared := sorted(statement_names(self._get_regular_step(*unit))[0] & set(shadows)):
                raise SpecError(
                    f'The training FLOW reads "{shared[0]}", the exponential moving average of a model: the '
                    "average is a copy the optimizers never touch, and differentiating it trains nothing. "
                    "Read it from INFERENCE_FLOW instead."
                )
        state = [*self.models, *self.optimizers]
        reserved = {
            "self",
            "kwargs",
            *self._learner_members,
            *[f"_view_{name}" for name in self.models],
            *shadows,
            *[f"_ema_state_{name}" for name in self.ema],
            *[f"_view_ema_{name}" for name in self.ema],
        }
        if shared := sorted(set(state) & set(self.inputs)):
            raise SpecError(
                f'Name "{shared[0]}" is both an input of the learner and one of its models or optimizers: '
                "the generated Flax steps take every model and optimizer as a parameter of their own and the "
                "batch as keyword-only arguments, so one name cannot be both. Rename one of them."
            )
        for name in [*state, *self.inputs]:
            if name in reserved:
                raise SpecError(
                    f'Name "{name}" is reserved by the generated Flax learner, so it cannot name a model, an '
                    "optimizer or an input: the learner takes each of them as a parameter of its steps and keeps "
                    "its models and optimizers under attributes of its own. Rename it."
                )
        stored = unique([n for flow in (self.flow, self.inference_flow) for u in flow for n in _stores(u)])
        for name in [*stored, *self.outputs]:
            if name in {*state, *shadows, *self._step_locals}:
                raise SpecError(
                    f'A FLOW of the learner stores "{name}", which the generated training step already binds for '
                    "a model, an average of one, an optimizer or a value of its own: the store would overwrite it "
                    "before the step reads it. Rename the output."
                )

    def _get_learner_script(self, initialized_layers: dict[str, str]) -> str:
        """Get the script for the learner class, whose `__init__` builds everything the steps read."""
        self._reject_reserved_names()
        indent = " " * 4
        sep2 = "\n" + indent * 2
        named = f"*, {', '.join(self.inputs)}, " if self.inputs else ""
        passed = "".join(f"{name}={name}, " for name in self.inputs)
        inputs = f"{', '.join(self.inputs)}, " if self.inputs else ""
        view = (
            "flax.nnx.view({0}, raise_if_not_found=False, training=False, deterministic=True, use_running_average=True)"
        )
        state = ", ".join(f"self.{name}" for name in [*self.models, *self.optimizers])
        shadows = [f"ema_{name}" for name in self._inference_shadows]
        views = ", ".join(f"self._view_{name}" for name in [*self.models, *shadows])
        models_repr = ", ".join(f"{name!r}: self.{name}" for name in [*self.models, *[f"ema_{n}" for n in self.ema]])
        optimizers_repr = ", ".join(f"{name!r}: self.{name}" for name in self.optimizers)
        optimizer_models = ", ".join(f"{s.optimizer!r}: {s.trainable_layers!r}" for _, s in self._segments)
        hashes = ", ".join(f"{s.optimizer!r}: {s.optimizer_hash!r}" for _, s in self._segments)
        rates = ", ".join(f"{name!r}: float('nan')" for name in self.optimizers)
        body = [f"{k} = {v}" for k, v in initialized_layers.items() if k != v]
        body += [f"self.{name} = {name}" for name in self.models]
        body += [f"self.{name} = {self.others[name]}" for name in self.optimizers]
        body += [
            "self._steps = 0",
            "self._updates = 0",
            "self._has_updated = False",
            f"self._learning_rates = {{{rates}}}",
        ]
        body += [f"self._view_{name} = {view.format(name)}" for name in self.models]
        for name in self.ema:
            # The view shares its variables with the average, so running it is running the average.
            body.append(f"self._ema_state_{name} = {self.others[f'ema_{name}']}")
            body.append(f"self.ema_{name} = self._ema_state_{name}.apply_to({name})")
        body += [f"self._view_{name} = {view.format(f'self.{name}')}" for name in shadows]
        body += self._forward_training_flow
        body.append(f"def _training_step({', '.join([*self.models, *self.optimizers])}, {named}**kwargs):")
        body += [f"{indent}{line}" for line in self._training_flow_parts[1]]
        body.append(f"def _inference_step({', '.join([*self.models, *shadows])}, {named}**kwargs):")
        body += [f"{indent}{line}" for line in self._forward_inference_flow]
        body += [
            "self._training_step = _training_step",
            "self._inference_step = _inference_step",
            f"self.inputs = {self.inputs}",
            f"self.outputs = {self.outputs}",
        ]
        averages = (
            [
                "# One blend per Update, on the host: an EMA the compiled step captured would be",
                "# mutated from another trace level, which flax rejects.",
                "if self._has_updated:",
                *[f"{indent}self._ema_state_{name}.update(self.{name})" for name in self.ema],
            ]
            if self.ema
            else []
        )
        updates = "".join(f"\n{indent * 2}{line}" for line in averages)
        return f"""\
class {self.classname}:
    \"\"\"Learner generated from a Flax (nnx) learner template.

    The two steps are the plain functions `__init__` builds over named model and optimizer
    parameters, bound to the attributes `flow_functions` names. A trainer that compiles them wraps
    those functions -- the state parameters donated, the keyword-only batch never -- and rebinds
    each attribute to its wrapper; the learner itself is never traced. The flow layers the steps
    call are locals of `__init__` they close over, so they must be stateless: a variable-carrying
    layer would be captured as a constant when a step is compiled. The learning rates are read
    inside the step, at trace time; reading them from the optimizers afterwards would touch the
    buffers the step was handed.

    `steps` counts every `training_step` call on the host. Gradient accumulation, when configured,
    is the optimizer's own `optax.MultiSteps`, gating on the device: the step decides `has_updated`
    by comparing the first optimizer's applied count across its update -- without a window every
    step applies -- and `updates` accumulates it. `restore_counters` seeds both from a checkpoint.
    `outputs` names the criteria the steps return, and `inference_step` runs against inference views
    of the models, which share their arrays with the trained ones.
    \"\"\"

    OPTIMIZER_HASHES: dict[str, str] = {{{hashes}}}

    def __init__(self, {self._learner_models}, **kwargs):
        {sep2.join(body)}

    def training_step(self, {inputs}**kwargs):
        self._steps += 1
        criteria, learning_rates, has_updated = self._training_step({state}, {passed}**kwargs)
        self._learning_rates = learning_rates
        self._has_updated = bool(has_updated)
        self._updates += int(self._has_updated){updates}
        return criteria

    def inference_step(self, {inputs}**kwargs):
        return self._inference_step({views}, {passed}**kwargs)

    def restore_counters(self, steps: int, updates: int) -> None:
        self._steps = steps
        self._updates = updates

    @property
    def steps(self):
        return self._steps

    @property
    def updates(self):
        return self._updates

    @property
    def has_updated(self):
        return self._has_updated

    @property
    def models(self):
        return {{{models_repr}}}

    @property
    def optimizers(self):
        return {{{optimizers_repr}}}

    @property
    def optimizer_models(self):
        return {{{optimizer_models}}}

    @property
    def flow_functions(self):
        return {{"_training_step": self._training_step, "_inference_step": self._inference_step}}

    @property
    def learning_rates(self):
        return {{k: float(v) for k, v in self._learning_rates.items()}}
"""


@dataclass(kw_only=True, slots=True)
class FlaxLearnerBuilder(BaseLearnerBuilder[FlaxLearnerIntermediate]):
    """Builder for Flax (nnx) learners.

    The `OPTIMIZER` pattern is a callable returning a `flax.nnx.Optimizer` when applied to the modules
    one segment owns, so the builder appends them to the pattern: the module
    itself when the segment owns one, a plain tuple of them when it owns several.
    """

    user_defined_learner_layer_type: ClassVar[type[FlaxLearnerIntermediate]] = FlaxLearnerIntermediate
    layer_builder_type: ClassVar[type[FlaxBuilder]] = FlaxBuilder
    template_type: ClassVar[type[FlaxTemplateLearner]] = FlaxTemplateLearner

    def _build_segment(
        self,
        imports: defaultdict[str, set[str | None]],
        module: Any,
        learner: LearnerBehavior,
        opt_name: str,
        naming: AutoName,
        layers: dict[str, LayerIntermediate | str],
        others: dict[str, str],
    ) -> FlaxOptimizerSegment:
        """Build the optimizer segment, carrying the digest of the pattern that built it.

        The names a segment contributes are checked where every name of the learner is known, when
        the script is emitted: `FlaxLearnerIntermediate._reject_reserved_names`.
        """
        # Named base rather than a zero-argument `super()`: `slots=True` rebuilds the class, and on
        # Python below 3.12.4 -- inside the project floor -- the `__class__` cell still points at the
        # discarded one, so `super()` raises "obj must be an instance or subtype of type" here.
        base = BaseLearnerBuilder._build_segment(self, imports, module, learner, opt_name, naming, layers, others)
        return FlaxOptimizerSegment(
            loss=base.loss,
            optimizer=base.optimizer,
            trainable_layers=base.trainable_layers,
            optimizer_hash=optimizer_hash(learner.OPTIMIZER),
        )

    def _register_shadow_models(
        self,
        imports: defaultdict[str, set[str | None]],
        module: FlaxUserDefinedLearner,
        naming: AutoName,
        others: dict[str, str],
    ) -> None:
        """Register one `flax.nnx.EMA` per `EMA` entry, under the name of the view it is applied to.

        The registered expression builds the average itself; the learner keeps it as
        `_ema_state_<model>` and binds `apply_to(<model>)` -- the callable view sharing its variables
        -- to the registered name, which is what a flow runs.
        """
        for model, config in module.EMA.items():
            if model not in module.TRAINABLE_LAYERS:
                raise SpecError(
                    f'EMA names "{model}", which is not a model of the learner: an EMA key names a model the '
                    f"learner trains, which are {module.TRAINABLE_LAYERS}."
                )
            if (name := f"ema_{model}") in others or name in module.INPUTS or name in module.OUTPUTS:
                raise SpecError(
                    f'The EMA of "{model}" is emitted as "{name}", which the learner already uses for a model, '
                    "an input or an output of its own. Rename that one."
                )
            # Reserved with the rest, so an auto-named flow layer cannot claim the name afterwards.
            naming(name)
            options = {} if isinstance(config, bool) else config
            keywords = ", ".join(f"{k}={resolve_getter(imports, v)}" for k, v in {**_EMA_DEFAULTS, **options}.items())
            others[name] = f"flax.nnx.EMA({model}, {keywords})"

    def _intermediate_fields(self, module: FlaxUserDefinedLearner) -> dict[str, Any]:
        """Get the framework-specific fields of the built learner intermediate."""
        return {"ema": list(module.EMA)}

    def _get_optimizer(
        self,
        imports: defaultdict[str, set[str | None]],
        optimizer: ObjectPattern,
        trainable_layers: list[str],
    ) -> tuple[str, str]:
        """Emit the optimizer expression, applying the pattern to the modules the segment owns."""
        pattern, injected = inject_learning_rate(optimizer)
        if not injected:
            logger.warning(
                "The optimizer of %s reports no learning rate: no single factory call carries a "
                "learning_rate keyword. Pass the rate as a keyword argument, or wrap the factory in "
                "optax.inject_hyperparams yourself; until then the learner reports NaN.",
                trainable_layers,
            )
        opt_inst, opt_cls = resolve_object(imports, pattern)
        # `nnx.Optimizer` requires `wrt`, and the parameters are the only sensible default.
        # `Param` and `flax.nnx` itself are default imports of the learner, so nothing is added here.
        parts = cast(list[Any], pattern.model_dump(by_alias=True))[1:]
        wrt = "" if any("wrt" in (_keywords(part) or {}) for part in parts) else ", wrt=Param"
        return f"{opt_inst}({_owned(trainable_layers)}{wrt})", opt_cls


__all__ = [
    "FlaxBuilder",
    "FlaxLayerIntermediate",
    "FlaxLearnerBuilder",
    "FlaxLearnerIntermediate",
    "FlaxOptimizerSegment",
    "FlaxTemplateLearner",
    "FlaxUserDefinedLearner",
    "inject_learning_rate",
    "optimizer_hash",
]


if not TYPE_CHECKING:
    import sys

    from structcast.utils.lazy_import import LazySelectedImporter

    sys.modules[__name__] = LazySelectedImporter(__name__, globals())
