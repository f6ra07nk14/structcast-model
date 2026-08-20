"""Builder for Flax (nnx) models."""

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, Any, ClassVar, cast
from warnings import warn

from pydantic import ValidationError
from structcast.core.exceptions import SpecError
from structcast.core.instantiator import ObjectPattern

from structcast_model.builders.auto_name import AutoName
from structcast_model.builders.base import (
    BaseLearnerBuilder,
    BaseModelBuilder,
    LayerIntermediate,
    LearnerIntermediate,
    OptimizerSegment,
    # Framework-neutral and shared with the Keras builder, re-exported here because a caller reading
    # a learner's `OPTIMIZER_HASHES` reaches for it next to the builder that emitted them.
    optimizer_hash,
)
from structcast_model.builders.schema import LearnerBehavior, TemplateLearner
from structcast_model.builders.utils import resolve_object, statement_names, stored_names
from structcast_model.utils.base import unique


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
class {class_name}(flax.nnx.Module):

    def __init__(self, *, rngs: flax.nnx.Rngs, training: bool = True):
        self.inputs = {self.inputs}
        self.input_shapes = {self.input_shapes}
        self.outputs = {self.outputs}
        self.training = training
        {sep.join([f"{self._get_layer(v)}" for v in initialized_layers])}

    def __call__(self, {inputs}*, training = None, **kwargs):
        training = self.training if training is None else training
        {sep.join(codes)}
        return {self._forward_outputs}

    def set_view(self, training = None):
        if training is not None:
            self.training = training
"""


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
    materializes, so a pattern emitted verbatim would report NaN for the rest of the run
    (see `docs/adr/0013`). The unique factory call carrying a `learning_rate` keyword -- at any depth,
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


def _container(trainable_layers: list[str]) -> str:
    """Return the name of the variable holding the modules one optimizer owns.

    A single owned module is passed to `flax.nnx.Optimizer` verbatim; several are wrapped in one
    persistent `flax.nnx.List` so the optimizer state, the differentiated argument and the
    accumulated gradients all key off the same module paths.
    """
    if len(trainable_layers) == 1:
        return trainable_layers[0]
    return f"_seg_{'_'.join(trainable_layers)}"


@dataclass(kw_only=True, slots=True)
class FlaxOptimizerSegment(OptimizerSegment):
    """One optimizer step of a Flax learner flow, carrying the digest of the pattern that built it."""

    optimizer_hash: str
    """The digest of the segment's `OPTIMIZER` pattern, emitted as `OPTIMIZER_HASHES` in the learner."""


class FlaxLearnerIntermediate(LearnerIntermediate[FlaxOptimizerSegment]):
    """Intermediate representation of a Flax (nnx) learner.

    The two steps are emitted as module-level functions taking the models, the optimizers and the
    gradient accumulator as arguments -- never as closures or bound methods -- so that a trainer can
    wrap them in `flax.nnx.jit` (`need_update` static, the three state arguments donated) and rebind
    each attribute `flow_functions` names, exactly as the PyTorch learners are compiled. Each segment differentiates
    its own closure with `flax.nnx.value_and_grad`, whose first argument is the container of the
    modules that segment owns; the models it only reads are passed as further, non-differentiated
    arguments. The closure returns as auxiliary output only what the enclosing step reads back --
    the criteria, the `EXTRA` keywords and whatever a later segment reads -- and keeps the rest local.

    Learner-level flow layers (losses and metrics) are emitted at module scope, so they must be
    stateless: a variable-carrying layer would be captured as a constant by a compiled step. The
    `EXTRA` keywords of a segment are forwarded to `flax.nnx.Optimizer.update`, which hands them to
    the transformation, as `optax.GradientTransformationExtraArgs` expects.
    """

    default_imports: ClassVar[dict[str, set[str | None]]] = {
        "jax": {None},
        "jax.numpy": {None},
        "flax.nnx": {None, "Param"},
        "structcast_model.flax.optimizers": {"get_learning_rate"},
    }
    """Default imports for Flax learners; the generated steps and properties call these directly."""

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
    def _containers(self) -> dict[str, str]:
        """Get the variable holding the modules each optimizer owns, keyed by optimizer name."""
        return {segment.optimizer: _container(segment.trainable_layers) for _, segment in self._segments}

    @cached_property
    def _module_lists(self) -> dict[str, list[str]]:
        """Get the owned modules of every container that has to be built as a `flax.nnx.List`."""
        return {
            self._containers[s.optimizer]: s.trainable_layers for _, s in self._segments if len(s.trainable_layers) > 1
        }

    @cached_property
    def _segment_bodies(self) -> list[tuple[list[str], list[str], list[str], set[str]]]:
        """Analyze each segment into its (parameters, body lines, stored names, enclosing-scope reads).

        A segment is emitted as one nested function, so every name it stores is local to its whole
        body: a step reading such a name before the step that stores it emits valid Python that
        raises `UnboundLocalError` on the first batch. That order is rejected here instead.
        """
        bodies: list[tuple[list[str], list[str], list[str], set[str]]] = []
        for units, segment in self._segments:
            container, owned = self._containers[segment.optimizer], segment.trainable_layers
            parameters = [container, *unique([L for _, _, L in units if L in self.models and L not in owned])]
            body = [f"{', '.join(owned)} = {container}"] if len(owned) > 1 else []
            body += [self._get_regular_step(i, o, L) for i, o, L in units]
            stores = unique([name for _, output, _ in units for name in stored_names(output)])
            bound, external = set(parameters), set[str]()
            deferred = set(stores) - bound
            for line in body:
                loads, stored = statement_names(line)
                if shadowed := sorted(loads & (deferred - bound)):
                    raise SpecError(
                        f'Optimizer "{segment.optimizer}" reads "{shadowed[0]}" before its own FLOW stores it. '
                        "A Flax segment is one nested function, so a name it stores is local to the whole "
                        "segment: compute the value before it is read, or give one of the two another name."
                    )
                external |= loads - bound
                bound |= stored
            bodies.append((parameters, body, stores, external))
        return bodies

    def _get_forward_training_flow(self) -> list[str]:
        """Get the body of the module-level `_training_step` function."""
        lines = [f"{name} = models[{name!r}]" for name in [*self.models, *self._module_lists]]
        bodies = self._segment_bodies
        for index, (_, segment) in enumerate(self._segments):
            container = self._containers[segment.optimizer]
            parameters, body, stores, _ = bodies[index]
            arguments = ", ".join(parameters)
            extra = f", {segment.backward_kwargs}" if segment.backward_kwargs else ""
            # Only what the enclosing step reads leaves the closure: the criteria, whatever the
            # update expression reads, and whatever a later segment reads. Every other intermediate
            # stays local, so a flow may compute values a traced auxiliary output could not carry.
            needed = {segment.loss, *self.outputs, *statement_names(f"_update(_grads{extra})")[0]}
            needed |= {name for _, _, _, reads in bodies[index + 1 :] for name in reads}
            aux = [name for name in stores if name in needed]
            if segment.loss not in aux:
                raise SpecError(
                    f'Optimizer "{segment.optimizer}" cannot be differentiated: its FLOW does not compute its '
                    f'LOSS "{segment.loss}". A Flax segment only differentiates what its own flow computes.'
                )
            # Scale inside the differentiated closure so the reported loss keeps its unscaled value.
            scaled = f"({segment.loss} / {self.accumulate_gradients})" if self.accumulate_gradients else segment.loss
            returns = f"({', '.join(aux)},)"
            lines.append(f"def _flow_{segment.optimizer}({arguments}):")
            lines += [f"    {line}" for line in [*body, f"return {scaled}, {returns}"]]
            grad = f"flax.nnx.value_and_grad(_flow_{segment.optimizer}, has_aux=True)({arguments})"
            lines.append(f"(_, {returns}), _grads = {grad}")
            accumulated = f"acc_grads[{segment.optimizer!r}]"
            if self.accumulate_gradients:
                lines.append(f"{accumulated} = jax.tree.map(jax.numpy.add, {accumulated}, _grads)")
                lines.append("if need_update:")
                lines.append(f"    optimizers[{segment.optimizer!r}].update({container}, {accumulated}{extra})")
                lines.append(f"    {accumulated} = jax.tree.map(jax.numpy.zeros_like, {accumulated})")
            else:
                lines.append(f"optimizers[{segment.optimizer!r}].update({container}, _grads{extra})")
        # Read at trace time: the walk compiles to a reference to the injected rate, not to a host read.
        rates = ", ".join(f"{name!r}: get_learning_rate(optimizers[{name!r}])" for name in self.optimizers)
        lines.append(f"lrs = {{{rates}}}")
        lines.append(f"return {self._forward_outputs}, lrs, acc_grads")
        return lines

    def _get_forward_inference_flow(self) -> list[str]:
        """Get the body of the module-level `_inference_step` function."""
        lines = [f"{name} = models[{name!r}]" for name in self.models]
        lines += [self._get_regular_step(i, o, L) for i, o, L in self.inference_flow]
        return [*lines, f"return {self._forward_outputs}"]

    def _get_learner_script(self, initialized_layers: dict[str, str]) -> str:
        """Get the script for the learner: its flow layers, the two steps and the learner class."""
        indent = " " * 4
        sep = "\n" + indent
        sep2 = "\n" + indent * 2
        named = f"*, {', '.join(self.inputs)}, " if self.inputs else ""
        passed = "".join(f"{name}={name}, " for name in self.inputs)
        inputs = f"{', '.join(self.inputs)}, " if self.inputs else ""
        views = (
            "{k: flax.nnx.view(v, raise_if_not_found=False, training=False, deterministic=True, "
            "use_running_average=True) for k, v in self.models.items()}"
        )
        models_repr = ", ".join(f"{m!r}: self._models[{m!r}]" for m in self.models)
        acc_grads = ", ".join(
            f"{o!r}: jax.tree.map(jax.numpy.zeros_like, flax.nnx.state({c}, Param))"
            for o, c in (self._containers.items() if self.accumulate_gradients else {}.items())
        )
        optimizer_models = ", ".join(f"{s.optimizer!r}: {s.trainable_layers!r}" for _, s in self._segments)
        need_update = ["return self.need_update"]
        if self.accumulate_gradients:
            need_update = [f"self.need_update = (step + 1) % {self.accumulate_gradients} == 0", *need_update]
        body = [f"{name} = flax.nnx.List([{', '.join(owned)}])" for name, owned in self._module_lists.items()]
        body += [f"{k} = {v}" for k, v in self.others.items() if k != v]
        layers = [f"{k} = {v}" for k, v in initialized_layers.items() if k != v]
        hashes = ", ".join(f"{s.optimizer!r}: {s.optimizer_hash!r}" for _, s in self._segments)
        parts = ["\n".join(layers)] if layers else []
        parts.append(f"OPTIMIZER_HASHES: dict[str, str] = {{{hashes}}}")
        parts.append(f"""\
def _training_step(models, optimizers, acc_grads, need_update, {named}**kwargs):
    {sep.join(self._forward_training_flow)}""")
        parts.append(f"""\
def _inference_step(models, {named}**kwargs):
    {sep.join(self._forward_inference_flow)}""")
        parts.append(f"""\
class {self.classname}:
    \"\"\"Learner generated from a Flax (nnx) learner template.

    The steps are the module-level `_training_step` and `_inference_step` functions, bound as the
    attributes `flow_functions` names. A trainer that compiles them wraps those functions --
    `need_update` static, the models, optimizers and gradient accumulator donated -- and rebinds
    each attribute to its wrapper; the learner itself is never traced. `outputs` names the criteria
    the steps return, and `inference_step` runs against inference views of the models.
    \"\"\"

    def __init__(self, {self._learner_models}, **kwargs):
        {sep2.join(body)}
        self._models = {{{", ".join(f"{n!r}: {n}" for n in [*self.models, *self._module_lists])}}}
        self._optimizers = {{{", ".join(f"{n!r}: {n}" for n in self.optimizers)}}}
        self._acc_grads = {{{acc_grads}}}
        self._learning_rates = {{{", ".join(f"{n!r}: float('nan')" for n in self.optimizers)}}}
        self._views = {views}
        self._training_step = _training_step
        self._inference_step = _inference_step
        self.need_update = True
        self.inputs = {self.inputs}
        self.outputs = {self.outputs}

    def training_step(self, {inputs}**kwargs):
        criteria, learning_rates, self._acc_grads = self._training_step(
            self._models, self._optimizers, self._acc_grads, self.need_update, {passed}**kwargs
        )
        self._learning_rates = learning_rates
        return criteria

    def inference_step(self, {inputs}**kwargs):
        return self._inference_step(self._views, {passed}**kwargs)

    def update(self, step: int) -> bool:
        {sep2.join(need_update)}

    @property
    def models(self):
        return {{{models_repr}}}

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
        return {{k: float(v) for k, v in self._learning_rates.items()}}
""")
        return "\n\n\n".join(parts)


@dataclass(kw_only=True, slots=True)
class FlaxLearnerBuilder(BaseLearnerBuilder[FlaxLearnerIntermediate]):
    """Builder for Flax (nnx) learners.

    The `OPTIMIZER` pattern is a callable returning a `flax.nnx.Optimizer` when applied to the modules
    one segment owns, so the builder appends that container to the pattern (see `docs/adr/0013`).
    """

    user_defined_learner_layer_type: ClassVar[type[FlaxLearnerIntermediate]] = FlaxLearnerIntermediate
    layer_builder_type: ClassVar[type[FlaxBuilder]] = FlaxBuilder
    template_type: ClassVar[type[TemplateLearner]] = TemplateLearner

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
        """Build the optimizer segment, rejecting a container name something else already holds.

        The container of a multi-module segment is a generated name, but nothing stops a user model
        from carrying it: the two would then share one key in the generated learner's model
        dictionary, and whichever lost would never be trained.
        """
        container = _container(learner.TRAINABLE_LAYERS)
        if len(learner.TRAINABLE_LAYERS) > 1 and (container in layers or container in others):
            raise SpecError(
                f'Duplicate variable name "{container}" for the module container of optimizer "{opt_name}" '
                "found in the learner flow: rename the layer that already uses it."
            )
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

    def _get_optimizer(
        self,
        imports: defaultdict[str, set[str | None]],
        optimizer: ObjectPattern,
        trainable_layers: list[str],
    ) -> tuple[str, str]:
        """Emit the optimizer expression, applying the pattern to the modules the segment owns."""
        pattern, injected = inject_learning_rate(optimizer)
        if not injected:
            warn(
                f"The optimizer of {trainable_layers} reports no learning rate: no single factory call carries a "
                "learning_rate keyword. Pass the rate as a keyword argument, or wrap the factory in "
                "optax.inject_hyperparams yourself; until then the learner reports NaN.",
                UserWarning,
                stacklevel=2,
            )
        opt_inst, opt_cls = resolve_object(imports, pattern)
        # `nnx.Optimizer` requires `wrt`, and the parameters are the only sensible default.
        # `Param` and `flax.nnx` itself are default imports of the learner, so nothing is added here.
        parts = cast(list[Any], pattern.model_dump(by_alias=True))[1:]
        wrt = "" if any("wrt" in (_keywords(part) or {}) for part in parts) else ", wrt=Param"
        return f"{opt_inst}({_container(trainable_layers)}{wrt})", opt_cls


__all__ = [
    "FlaxBuilder",
    "FlaxLayerIntermediate",
    "FlaxLearnerBuilder",
    "FlaxLearnerIntermediate",
    "FlaxOptimizerSegment",
    "inject_learning_rate",
    "optimizer_hash",
]


if not TYPE_CHECKING:
    import sys

    from structcast.utils.lazy_import import LazySelectedImporter

    sys.modules[__name__] = LazySelectedImporter(__name__, globals())
