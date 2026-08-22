"""Builder for PyTorch models."""

from collections import defaultdict
from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, Any, ClassVar, Literal, Self, cast

from pydantic import Field, PositiveInt, model_validator
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
from structcast_model.builders.utils import resolve_getter, resolve_object, statement_names
from structcast_model.utils.base import to_snake, unique

_CHECKPOINT_OPTIONS = frozenset(
    {"context_fn", "debug", "determinism_check", "early_stop", "preserve_rng_state", "use_reentrant"}
)
"""The keyword arguments `torch.utils.checkpoint.checkpoint` accepts, which `GRADIENT_CHECKPOINTING` carries."""


def _unwrapped(model: str) -> str:
    """The emitted expression naming the module a DDP wrapper holds, or the model when it holds none.

    By type, never by attribute name: a model owning a submodule of its own called `module` would
    otherwise be averaged in fragments, silently.
    """
    return f"({model}.module if isinstance({model}, torch.nn.parallel.DistributedDataParallel) else {model})"


_EMA_DEFAULTS = {"multi_avg_fn": "eval: torch.optim.swa_utils.get_ema_multi_avg_fn(0.999)"}
"""What makes an `AveragedModel` an exponential moving average: without it torch averages equally (SWA).

Filled in under a mapping too, the way `use_reentrant` is, so a mapping that only sets `device` still
declares what its `EMA` key says it declares. Pass `multi_avg_fn: null` for torch's own SWA."""


class TorchLayerIntermediate(LayerIntermediate):
    """Intermediate representation of a PyTorch layer."""

    default_imports: ClassVar[dict[str, set[str | None]]] = {"torch": {None}}
    """Default imports for PyTorch layers."""

    def _get_layer(self, layername: str) -> str:
        """Get the sub-layer with the given name."""
        return f"self.{layername}"

    def _get_layer_script(self, class_name: str, initialized_layers: list[str]) -> str:
        """Implement the method to get the script for the layer."""
        indent = " " * 4
        sep = "\n" + indent * 2
        base, attributes = "torch.nn.Module", ""
        if self.gradient_checkpointing is not None:
            base = "GradientCheckpointingLayer"
            lines = ["gradient_checkpointing = True"]
            if self.gradient_checkpointing:
                keywords = ", ".join(f"{k!r}: {v}" for k, v in self.gradient_checkpointing.items())
                lines.append(f"_checkpoint_kwargs = {{{keywords}}}")
            attributes = "".join(f"{indent}{line}\n" for line in lines) + "\n"
        if self._forward_inference_flow:
            codes = [
                "if self.training:",
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

{attributes}    def __init__(self):
        super().__init__()
        self.inputs = {self.inputs}
        self.input_shapes = {self.input_shapes}
        self.outputs = {self.outputs}
        {sep.join([f"{self._get_layer(v)}" for v in initialized_layers])}

    def forward(self, {inputs}**kwargs):
        {sep.join(codes)}
        return {self._forward_outputs}
"""


@dataclass(kw_only=True, slots=True)
class TorchBuilder(BaseModelBuilder[TorchLayerIntermediate]):
    """Builder for PyTorch models."""

    user_defined_layer_type: ClassVar[type[TorchLayerIntermediate]] = TorchLayerIntermediate

    def _resolve_gradient_checkpointing(
        self,
        imports: defaultdict[str, set[str | None]],
        config: bool | dict[str, Any],
    ) -> dict[str, str] | None:
        """Validate the mapping against the keywords of `torch.utils.checkpoint.checkpoint`.

        `use_reentrant` is filled in as `False` when the mapping leaves it out: torch has no usable
        default for it -- it warns on every call and announces an exception in a later release -- and
        the reentrant variant differentiates nothing it was not handed positionally, which a
        generated layer called with its batch by name would be.
        """
        if config is False:
            return None
        options = {} if isinstance(config, bool) else config
        if unknown := sorted(options.keys() - _CHECKPOINT_OPTIONS):
            raise SpecError(
                f'GRADIENT_CHECKPOINTING option "{unknown[0]}" is not a keyword argument of '
                f"torch.utils.checkpoint.checkpoint, which accepts {sorted(_CHECKPOINT_OPTIONS)}."
            )
        imports["structcast_model.torch.layers"].add("GradientCheckpointingLayer")
        return {key: resolve_getter(imports, value) for key, value in {"use_reentrant": False, **options}.items()}


class TorchLearnerBehavior(LearnerBehavior):
    """Learner behavior configuration for PyTorch."""

    CLIP: ObjectPattern | None = None
    """Gradient clipping configuration, which can be an instance of the gradient clipping configuration
    or a pattern to instantiate the gradient clipping configuration."""


class TorchUserDefinedLearner(UserDefinedLearner[TorchLearnerBehavior]):
    """User defined learner configuration for PyTorch."""

    ACCUMULATE_GRADIENTS: PositiveInt | None = None
    """Whether to accumulate gradients for multiple steps before updating the parameters,
    and the number of steps to accumulate for.

    Torch-only: PyTorch has no native accumulation window, so the generated learner gates the
    optimizer step itself; the other backends declare the window through their optimizer
    (`docs/adr/0017`).
    """

    MIXED_PRECISION: bool | dict[str, Any] = False
    """Whether to use mixed precision during backward pass.

    If the value is a dictionary, it will be used as the keyword arguments for configuring mixed precision context.
    """

    MIXED_PRECISION_TYPE: Literal["bfloat16", "float16"] | None = None
    """The mixed precision type to use during backward pass when mixed precision is enabled."""

    EMA: dict[str, bool | dict[str, Any]] = Field(default_factory=dict)
    """The models an exponential moving average shadows, keyed by the model name.

    `true` takes the defaults; a mapping carries the keyword arguments of
    `torch.optim.swa_utils.AveragedModel`, each value resolved like any other DSL value. The average
    is emitted as the learner attribute `ema_<model>`, updated once per Update and runnable from
    `INFERENCE_FLOW` under that name (`docs/adr/0021`).
    """

    @model_validator(mode="after")
    def _validate_mixed_precision(self) -> Self:
        """Validate the mixed precision configuration.

        MIXED_PRECISION enables gradient scaling, which only counteracts float16 underflow;
        MIXED_PRECISION_TYPE alone configures autocast and is valid without a scaler.
        """
        enabled = bool(self.MIXED_PRECISION) if isinstance(self.MIXED_PRECISION, bool) else True
        if enabled and self.MIXED_PRECISION_TYPE != "float16":
            raise SpecError(
                "MIXED_PRECISION enables gradient scaling, which only applies to float16: set "
                "MIXED_PRECISION_TYPE: float16, or disable MIXED_PRECISION (bfloat16 autocast needs no scaler)."
            )
        return self


class TorchTemplateLearner(Template[TorchUserDefinedLearner]):
    """Template for PyTorch user-defined learners."""

    target_type: ClassVar[type[TorchUserDefinedLearner]] = TorchUserDefinedLearner


@dataclass(kw_only=True, slots=True)
class TorchOptimizerSegment(OptimizerSegment):
    """One optimizer step of a PyTorch learner flow."""

    clip: str | None
    """The variable name of the gradient clipping callable, or `None` if gradients are not clipped."""

    scaler: str | None
    """The variable name of the gradient scaler, or `None` if gradients are not scaled."""


class TorchLearnerIntermediate(LearnerIntermediate[TorchOptimizerSegment]):
    """Intermediate representation of a PyTorch learner."""

    mixed_precision_type: str | None
    """The mixed precision type for the learner, or `None` if mixed precision is not used."""

    ema: tuple[str, ...] = ()
    """The models carrying an exponential moving average, in `EMA` declaration order.

    Each one is emitted as the attribute `ema_<model>`, built from the expression the builder
    registered under that name in `others` (`docs/adr/0021`)."""

    default_imports: ClassVar[dict[str, set[str | None]]] = {
        "torch": {None},
        "structcast_model.torch.optimizers": {
            "get_decays",
            "get_learning_rate",
            "get_named_parameters",
            "get_param_groups",
            "restore_requires_grad",
        },
        "structcast_model.torch.distributed": {"sync_gate"},
    }
    """Default imports for PyTorch learners; the generated steps and properties call these directly."""

    @cached_property
    def mixed_precision_scales(self) -> list[str]:
        """Get the mixed precision scales used in the layer."""
        return unique([u.scaler for u in self.flow if isinstance(u, TorchOptimizerSegment) and u.scaler])

    def _with_autocast(self, flow: list[str]) -> list[str]:
        if not (self.mixed_precision_type and flow):
            return flow
        autocast = f"with torch.autocast(device_type, torch.{self.mixed_precision_type}):"
        return [autocast] + [f"{' ' * 4}{L}" for L in flow]

    def _flow_function(self, name: str, params: list[str], body: list[str], returns: list[str]) -> list[str]:
        """Emit one pure flow function plus the self-assignment that makes it rebindable (compilable)."""
        if not returns:
            raise ValueError(f"Flow function {name} produces no value any later code needs; check the learner FLOW.")
        indent = " " * 4
        header = f"def {name}(__need_update__{''.join(f', {p}' for p in params)}):"
        tail = [f"{indent}return {', '.join(returns)}", f"self.{name} = {name}"]
        return [header, *[f"{indent}{line}" for line in body], *tail]

    def _analyze_segment(self, units: list[tuple[str, str, str | None]]) -> dict[str, Any]:
        """Collect a segment's generated lines, external loads, stores, and per-model call counts."""
        local: set[str] = set()
        external: list[str] = []
        stores: list[str] = []
        counts: dict[str, int] = defaultdict(int)
        lines: list[tuple[str, str | None]] = []
        for inputs, output, layer in units:
            line = self._get_regular_step(inputs, output, layer)
            loads, stored = statement_names(line)
            external += [n for n in loads - local if n not in external]
            stores += [n for n in stored if n not in local]
            local |= stored
            if layer in self.models:
                counts[layer] += 1
            lines.append((line, layer))
        return {"lines": lines, "external": external, "stores": stores, "counts": counts}

    def _gated_body(self, info: dict[str, Any], trainable_layers: list[str]) -> list[str]:
        """Precede each model invocation with a sync gate, arming only a model's last owned call."""
        body: list[str] = []
        seen: dict[str, int] = defaultdict(int)
        for line, layer in info["lines"]:
            if layer in self.models:
                seen[layer] += 1
                last_owned = layer in trainable_layers and seen[layer] == info["counts"][layer]
                body.append(f"sync_gate({layer}, {'__need_update__' if last_owned else 'False'})")
            body.append(line)
        return body

    def _get_forward_inference_flow(self) -> list[str]:
        """Get the `__init__` half of the inference flow."""
        return self._inference_flow_parts[0]

    @cached_property
    def _inference_flow_parts(self) -> tuple[list[str], list[str]]:
        """Split inference into the compilable `_flow_inference` definition and the `inference_step` body."""
        info = self._analyze_segment(self.inference_flow)
        params = [n for n in info["external"] if n in self.inputs]
        body = self._with_autocast([line for line, _ in info["lines"]])
        defs = self._flow_function("_flow_inference", params, body, self.outputs)
        call = f"{', '.join(self.outputs)} = self._flow_inference(False{''.join(f', {p}' for p in params)})"
        return defs, [call]

    def _get_forward_training_flow(self) -> list[str]:
        """Get the `__init__` half of the training flow."""
        return self._training_flow_parts[0]

    @cached_property
    def _training_flow_parts(self) -> tuple[list[str], list[str]]:
        """Split training into the per-optimizer flow definitions and the eager `training_step` body.

        Each optimizer segment's pure computation becomes a `_flow_<optimizer>` function (the
        `torch.compile` unit); `train()/eval()`, `requires_grad_` freezing, backward, clipping,
        optimizer stepping, and `zero_grad()` stay in the eager step, where they neither break the
        compiled graph nor pollute it with guards.
        """

        def _param(layers: list[str]) -> str:
            if len(layers) == 1:
                return f"{layers[0]}.parameters()"
            return f"(p for m in ({', '.join(f'{L}' for L in layers)}) for p in m.parameters())"

        segments: list[tuple[list[tuple[str, str, str | None]], TorchOptimizerSegment]] = []
        units: list[tuple[str, str, str | None]] = []
        for unit in self.flow:
            if isinstance(unit, TorchOptimizerSegment):
                segments.append((units, unit))
                units = []
            else:
                units.append(unit)
        infos = [self._analyze_segment(seg_units) for seg_units, _ in segments]

        available = set(self.inputs)
        # Freezing restores each owned model's construction-time requires_grad states instead of a
        # blanket True, so submodules the user froze stay frozen across optimizer segments.
        defs: list[str] = []
        step: list[str] = []
        # `others` the step body reads off `self`, so the body lines stay plain local-variable code.
        used: list[str] = list(self.models)
        for i, ((_, opt_unit), info) in enumerate(zip(segments, infos, strict=True)):
            loss, backward_kwargs = opt_unit.loss, opt_unit.backward_kwargs
            optimizer_name, clip_name, mixed_precision_name = opt_unit.optimizer, opt_unit.clip, opt_unit.scaler
            trainable_layers = opt_unit.trainable_layers
            used += [n for n in (optimizer_name, clip_name, mixed_precision_name) if n and n not in used]
            # Scale inside the backward expression so the reported loss keeps its unscaled value.
            scaled = f"({loss} / {self.accumulate_gradients})" if self.accumulate_gradients else loss
            backward_line = (
                f"{scaled}.backward({backward_kwargs})"
                if mixed_precision_name is None
                else f"{mixed_precision_name}.scale({scaled}).backward({backward_kwargs})"
            )
            params = [n for n in info["external"] if n in available]
            needed = {loss} | set(self.outputs) | statement_names(backward_line)[0]
            needed |= {n for later in infos[i + 1 :] for n in later["external"]}
            returns = [n for n in info["stores"] if n in needed]
            function_name = f"_flow_{optimizer_name}"
            defs += self._flow_function(
                function_name, params, self._with_autocast(self._gated_body(info, trainable_layers)), returns
            )
            step += [f"{m}.{'train' if m in trainable_layers else 'eval'}()" for m in self.models]
            step += [
                f'restore_requires_grad({m}, self._requires_grad_defaults["{m}"])'
                if m in trainable_layers
                else f"{m}.requires_grad_(False)"
                for m in self.models
            ]
            arguments = "__need_update__" + "".join(f", {p}" for p in params)
            step.append(f"{', '.join(returns)} = self.{function_name}({arguments})")
            step.append(backward_line)
            if self.accumulate_gradients:
                step.append("if __need_update__:")
                indent = " " * 4
            else:
                indent = ""
            if mixed_precision_name is None:
                if clip_name is not None:
                    step.append(f"{indent}{clip_name}({_param(trainable_layers)})")
                step.append(f"{indent}{optimizer_name}.step()")
            else:
                if clip_name is not None:
                    step.append(f"{indent}{mixed_precision_name}.unscale_({optimizer_name})")
                    step.append(f"{indent}{clip_name}({_param(trainable_layers)})")
                step.append(f"{indent}{mixed_precision_name}.step({optimizer_name})")
                step.append(f"{indent}{mixed_precision_name}.update()")
            step.append(f"{indent}{optimizer_name}.zero_grad()")
            available |= set(info["stores"])
        # Incrementing `_steps` first keeps `_steps` on the trainer's old 1-based clock, so the
        # `(+ 1) % k` gate preserves the historically short first accumulation window.
        binds = [
            "self._steps += 1",
            f"__need_update__ = (self._steps + 1) % {self.accumulate_gradients} == 0"
            if self.accumulate_gradients
            else "__need_update__ = True",
        ] + [f"{n} = self.{n}" for n in used]
        return defs, binds + step + self._step_tail

    @cached_property
    def _step_tail(self) -> list[str]:
        """The lines closing the training step: the counters, and the averages that follow them."""
        tail = [
            "# Intent, not detection: under float16 the gradient scaler may skip the apply this flag reports,",
            "# and a torch optimizer exposes no counter to check it against.",
            "if __need_update__:",
            f"{' ' * 4}self._updates += 1",
            "self._has_updated = __need_update__",
        ]
        if self.ema:
            # One blend per Update, never per accumulation micro-step, and after every segment of the
            # step has applied: what an average follows is the weights a whole step produced.
            tail.append("if self._has_updated:")
            tail += [f"{' ' * 4}self.ema_{m}.update_parameters({_unwrapped(m)})" for m in self.ema]
        return tail

    @cached_property
    def _ema_lines(self) -> list[str]:
        """Emit the `__init__` lines building each averaged model, refusing an already sharded one."""
        lines: list[str] = []
        for model in self.ema:
            message = (
                f'The exponential moving average of "{model}" cannot be built: an AveragedModel copies the '
                "module it averages, and a module whose parameters are sharded (DTensor) has no copy to take. "
                "EMA works with neither FSDP2 nor tensor parallel; drop the EMA entry, or train this model "
                "under a strategy that keeps whole parameters."
            )
            lines += [
                f'if any(type(p).__name__ == "DTensor" for p in {model}.parameters()):',
                f"{' ' * 4}raise ValueError({message!r})",
                "# Averaged over the module a DDP wrapper holds: the wrapper is not copyable, and the",
                "# weights are the ones inside it. The average is only ever evaluated, never trained,",
                "# so it stays in eval mode for the whole run.",
                f"ema_{model} = {self.others[f'ema_{model}']}",
                f"ema_{model}.eval()",
            ]
        return lines

    def _reject_ema_in_training_flow(self) -> None:
        """Reject a training FLOW reading an EMA shadow: an average follows training, it is not trained."""
        shadows = {f"ema_{name}" for name in self.ema}
        for unit in self.flow:
            if isinstance(unit, OptimizerSegment):
                continue
            if shared := sorted(statement_names(self._get_regular_step(*unit))[0] & shadows):
                raise SpecError(
                    f'The training FLOW reads "{shared[0]}", the exponential moving average of a model: the '
                    "average is a copy the optimizers never touch, and differentiating it trains nothing. "
                    "Read it from INFERENCE_FLOW instead."
                )

    def _get_learner_script(self, initialized_layers: dict[str, str]) -> str:
        """Get the script for the learner."""
        self._reject_ema_in_training_flow()
        indent = " " * 4
        sep = "\n" + indent * 2
        shadows = [f"ema_{m}" for m in self.ema]
        models_repr = ", ".join([f'"{m}": self.{m}' for m in self.models] + [f'"{n}": self.{n}' for n in shadows])
        opts_repr = ", ".join([f'"{n}": self.{n}' for n in self.optimizers])
        grad_scalers_repr = ", ".join([f'"{n}": self.{n}' for n in self.mixed_precision_scales])
        optimizer_models_repr = ", ".join(
            f'"{u.optimizer}": {u.trainable_layers!r}' for u in self.flow if isinstance(u, TorchOptimizerSegment)
        )
        flow_names = [f"_flow_{n}" for n in self.optimizers] + ["_flow_inference"]
        flow_functions_repr = ", ".join(f'"{n}": self.{n}' for n in flow_names)
        inputs = self._forward_inputs
        inputs += ", " if inputs else ""
        defaults = ", ".join(f'"{m}": [p.requires_grad for p in {m}.parameters()]' for m in self.models)
        instances = [f"{k} = {v}" for k, v in self.others.items() if k != v and k not in shadows] + self._ema_lines
        return f"""\
class {self.classname}:
    \"\"\"Learner generated from a PyTorch learner template.

    The models arrive as constructor arguments, and each optimizer segment's pure computation becomes a
    `_flow_<optimizer>` closure bound as the attributes `flow_functions` names; a trainer that compiles them
    rebinds each attribute to its compiled wrapper, so backward, clipping and stepping stay eager here.
    `training_step` runs backward every call -- dividing the loss by the accumulation divisor inside the
    backward expression -- and gates clipping, the optimizer step, `zero_grad()` and, under mixed precision,
    the gradient scaler's unscale and update behind the accumulation gate it computes from its own `_steps`
    counter, while `inference_step` runs under `torch.no_grad()`. The learner owns the training counters:
    `steps`, `updates` and `has_updated` report completed counts after each step, and
    `restore_counters` seeds them after a checkpoint restore. Under float16 the gate keeps intent semantics:
    `has_updated` reports that an apply was attempted, not that the gradient scaler let it land. `outputs`
    names the criteria the steps return, and `models`, `optimizers`, `optimizer_models`, `grad_scalers`,
    `learning_rates`, `weight_decays` and `param_group_names` expose what a trainer reads off the learner.
    \"\"\"

    def __init__(self, {self._learner_models}, **kwargs):
        device_type = next({self.models[0]}.parameters()).device.type
        {sep.join([f"{m}.zero_grad()" for m in self.models])}
        {sep.join([f"{k} = {v}" for k, v in initialized_layers.items()])}
        {sep.join(instances)}
        {sep.join(self._forward_training_flow)}
        {sep.join(self._forward_inference_flow)}
        {sep.join([f"self.{k} = {k}" for k in self.others])}
        self._requires_grad_defaults = {{{defaults}}}
        self._steps = 0
        self._updates = 0
        self._has_updated = False
        self.inputs = {self.inputs}
        self.outputs = {self.outputs}

    def training_step(self, {inputs}**kwargs):
        {sep.join(self._training_flow_parts[1])}
        return {self._forward_outputs}

    @torch.no_grad()
    def inference_step(self, {inputs}**kwargs):
        {sep.join(self._inference_flow_parts[1])}
        return {self._forward_outputs}

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
        return {{{opts_repr}}}

    @property
    def grad_scalers(self):
        return {{{grad_scalers_repr}}}

    @property
    def optimizer_models(self):
        return {{{optimizer_models_repr}}}

    @property
    def flow_functions(self):
        return {{{flow_functions_repr}}}

    @property
    def learning_rates(self):
        return {{k: get_learning_rate(v) for k, v in self.optimizers.items()}}

    @property
    def weight_decays(self):
        return get_decays(self.optimizers)

    @property
    def param_group_names(self):
        return {{k: get_param_groups(v) for k, v in self.optimizers.items()}}
"""


@dataclass(kw_only=True, slots=True)
class TorchLearnerBuilder(BaseLearnerBuilder[TorchLearnerIntermediate]):
    """Builder for PyTorch learners."""

    user_defined_learner_layer_type: ClassVar[type[TorchLearnerIntermediate]] = TorchLearnerIntermediate
    layer_builder_type: ClassVar[type[TorchBuilder]] = TorchBuilder
    template_type: ClassVar[type[TorchTemplateLearner]] = TorchTemplateLearner

    def _build_segment(
        self,
        imports: defaultdict[str, set[str | None]],
        module: TorchUserDefinedLearner,
        learner: LearnerBehavior,
        opt_name: str,
        naming: AutoName,
        layers: dict[str, LayerIntermediate | str],
        others: dict[str, str],
    ) -> TorchOptimizerSegment:
        """Build the optimizer segment, registering the gradient clipper and scaler it needs."""
        # `learner` arrives through the base hook signature; `template_type` guarantees the torch schema.
        clip = cast(TorchLearnerBehavior, learner).CLIP
        amp_inst, amp_cls = self._get_mixed_precision(imports, module.MIXED_PRECISION, module.MIXED_PRECISION_TYPE)
        clip_name: str | None = None
        if clip:
            clip_inst, clip_cls = resolve_object(imports, clip)
            if (clip_name := naming(f"{opt_name}_{to_snake(clip_cls)}")) in layers or clip_name in others:
                raise SpecError(f'Duplicate variable name "{clip_name}" for clip found in the learner flow.')
            others[clip_name] = clip_inst
        amp_name: str | None = None
        if amp_cls is not None:
            if (amp_name := naming(f"{opt_name}_{to_snake(amp_cls)}")) in layers or amp_name in others:
                raise SpecError(
                    f'Duplicate variable name "{amp_name}" for mixed precision instance found in the learner flow.'
                )
            others[amp_name] = amp_inst
        return TorchOptimizerSegment(
            loss=learner.LOSS,
            optimizer=opt_name,
            trainable_layers=learner.TRAINABLE_LAYERS,
            clip=clip_name,
            scaler=amp_name,
        )

    def _register_shadow_models(
        self,
        imports: defaultdict[str, set[str | None]],
        module: TorchUserDefinedLearner,
        naming: AutoName,
        others: dict[str, str],
    ) -> None:
        """Register one `torch.optim.swa_utils.AveragedModel` per `EMA` entry, named `ema_<model>`."""
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
            imports["torch.optim.swa_utils"].add(None)
            options = {} if isinstance(config, bool) else config
            keywords = "".join(f", {k}={resolve_getter(imports, v)}" for k, v in {**_EMA_DEFAULTS, **options}.items())
            others[name] = f"torch.optim.swa_utils.AveragedModel({_unwrapped(model)}{keywords})"

    def _intermediate_fields(self, module: TorchUserDefinedLearner) -> dict[str, Any]:
        """Get the framework-specific fields of the built learner intermediate."""
        return {
            "accumulate_gradients": module.ACCUMULATE_GRADIENTS,
            "mixed_precision_type": module.MIXED_PRECISION_TYPE,
            "ema": list(module.EMA),
        }

    def _get_mixed_precision(
        self,
        imports: defaultdict[str, set[str | None]],
        mixed_precision: bool | dict[str, Any],
        mixed_precision_type: str | None,
    ) -> tuple[str, str | None]:
        if isinstance(mixed_precision, bool) and not mixed_precision:
            return "", None
        if mixed_precision_type != "float16":
            # bfloat16 shares float32's exponent range: gradients cannot underflow, so a scaler
            # is pure overhead. The schema rejects this pairing; returning nothing keeps the
            # builder safe regardless.
            return "", None
        if isinstance(mixed_precision, bool):
            mixed_precision = {}
        imports["torch.amp"].add(None)
        repr_mp_kw = "".join(f", {k}={resolve_getter(imports, v)}" for k, v in mixed_precision.items())
        return f"torch.amp.GradScaler(device=device_type{repr_mp_kw})", "GradScaler"

    def _get_optimizer(
        self,
        imports: defaultdict[str, set[str | None]],
        optimizer: ObjectPattern,
        trainable_layers: list[str],
    ) -> tuple[str, str]:
        opt_inst, opt_cls = resolve_object(imports, optimizer)
        return f"{opt_inst}(get_named_parameters([{', '.join(trainable_layers)}]))", opt_cls


__all__ = [
    "TorchBuilder",
    "TorchLayerIntermediate",
    "TorchLearnerBehavior",
    "TorchLearnerBuilder",
    "TorchLearnerIntermediate",
    "TorchOptimizerSegment",
    "TorchTemplateLearner",
    "TorchUserDefinedLearner",
]


if not TYPE_CHECKING:
    import sys

    from structcast.utils.lazy_import import LazySelectedImporter

    sys.modules[__name__] = LazySelectedImporter(__name__, globals())
