"""Builder for PyTorch models."""

import ast
from collections import defaultdict
from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, Any, ClassVar

from structcast.core.instantiator import ObjectPattern

from structcast_model.builders.base_builder import (
    BaseLearnerBuilder,
    BaseModelBuilder,
    LayerIntermediate,
    LearnerIntermediate,
    resolve_getter,
    resolve_object,
)


def _statement_names(line: str) -> tuple[set[str], set[str]]:
    """Return the (loaded, stored) variable names of one generated statement."""
    loads: set[str] = set()
    stores: set[str] = set()
    for node in ast.walk(ast.parse(line.strip())):
        if isinstance(node, ast.Name):
            (stores if isinstance(node.ctx, ast.Store) else loads).add(node.id)
    return loads, stores


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
class {class_name}(torch.nn.Module):

    def __init__(self):
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


class TorchLearnerIntermediate(LearnerIntermediate):
    """Intermediate representation of a PyTorch learner."""

    default_imports: ClassVar[dict[str, set[str | None]]] = {
        "torch": {None},
        "structcast_model.torch.optimizers": {"get_decays"},
        "structcast_model.torch.distributed": {"sync_gate"},
    }
    """Default imports for PyTorch learners; the generated steps and properties call these directly."""

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
            loads, stored = _statement_names(line)
            external += [n for n in loads - local if n not in external]
            stores += [n for n in stored if n not in local]
            local |= stored
            if layer in self.models:
                counts[layer] += 1
            lines.append((line, layer))
        return {"lines": lines, "external": external, "stores": stores, "counts": counts}

    def _gated_body(self, info: dict[str, Any], trainable_layers: list[str]) -> list[str]:
        """Wrap each model invocation in a sync gate, arming only a model's last owned call."""
        body: list[str] = []
        seen: dict[str, int] = defaultdict(int)
        for line, layer in info["lines"]:
            if layer in self.models:
                seen[layer] += 1
                last_owned = layer in trainable_layers and seen[layer] == info["counts"][layer]
                line = f"with sync_gate({layer}, {'__need_update__' if last_owned else 'False'}): {line}"
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

        segments: list[tuple[list[tuple[str, str, str | None]], tuple]] = []
        units: list[tuple[str, str, str | None]] = []
        for unit in self.flow:
            if len(unit) == 3:
                units.append(unit)
            else:
                segments.append((units, unit))
                units = []
        infos = [self._analyze_segment(seg_units) for seg_units, _ in segments]

        available = set(self.inputs)
        # Freezing restores each owned model's construction-time requires_grad states instead of a
        # blanket True, so submodules the user froze stay frozen across optimizer segments.
        defs: list[str] = []
        step: list[str] = []
        # `others` the step body reads off `self`, so the body lines stay plain local-variable code.
        used: list[str] = list(self.models)
        for i, ((_, opt_unit), info) in enumerate(zip(segments, infos, strict=True)):
            loss, backward_kwargs, optimizer_name, clip_name, mixed_precision_name, trainable_layers = opt_unit
            used += [n for n in (optimizer_name, clip_name, mixed_precision_name) if n and n not in used]
            backward_line = (
                f"{loss}.backward({backward_kwargs})"
                if mixed_precision_name is None
                else f"{mixed_precision_name}.scale({loss}).backward({backward_kwargs})"
            )
            params = [n for n in info["external"] if n in available]
            needed = {loss} | set(self.outputs) | _statement_names(backward_line)[0]
            needed |= {n for later in infos[i + 1 :] for n in later["external"]}
            returns = [n for n in info["stores"] if n in needed]
            function_name = f"_flow_{optimizer_name}"
            defs += self._flow_function(
                function_name, params, self._with_autocast(self._gated_body(info, trainable_layers)), returns
            )
            step += [f"{m}.{'train' if m in trainable_layers else 'eval'}()" for m in self.models]
            step += [
                f'_restore_requires_grad({m}, self._requires_grad_defaults["{m}"])'
                if m in trainable_layers
                else f"{m}.requires_grad_(False)"
                for m in self.models
            ]
            arguments = "__need_update__" + "".join(f", {p}" for p in params)
            step.append(f"{', '.join(returns)} = self.{function_name}({arguments})")
            if self.accumulate_gradients:
                step.append(f"{loss} = {loss} / {self.accumulate_gradients}")
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
        binds = ["__need_update__ = self.need_update"] + [f"{n} = self.{n}" for n in used]
        return defs, binds + step

    def _get_learner_script(self, initialized_layers: dict[str, str]) -> str:
        """Get the script for the learner."""
        indent = " " * 4
        sep = "\n" + indent * 2
        models_repr = ", ".join([f'"{m}": self.{m}' for m in self.models])
        opts_repr = ", ".join([f'"{n}": self.{n}' for n in self.optimizers])
        grad_scalers_repr = ", ".join([f'"{n}": self.{n}' for n in self.mixed_precision_scales])
        optimizer_models_repr = ", ".join(f'"{u[2]}": {u[5]!r}' for u in self.flow if len(u) == 6)
        flow_names = [f"_flow_{n}" for n in self.optimizers] + ["_flow_inference"]
        flow_functions_repr = ", ".join(f'"{n}": self.{n}' for n in flow_names)
        scaler_param = "__grad_scaler_creator__=torch.amp.GradScaler, " if self.mixed_precision_scales else ""
        need_update = ["return self.need_update"]
        if self.accumulate_gradients:
            need_update = [f"self.need_update = (step + 1) % {self.accumulate_gradients} == 0"] + need_update
        inputs = self._forward_inputs
        inputs += ", " if inputs else ""
        defaults = ", ".join(f'"{m}": [p.requires_grad for p in {m}.parameters()]' for m in self.models)
        return f"""\
def _restore_requires_grad(module, defaults):
    for p, d in zip(module.parameters(), defaults):
        p.requires_grad_(d)


class {self.classname}:

    def __init__(self, {self._learner_models}, {scaler_param}**kwargs):
        device_type = next({self.models[0]}.parameters()).device.type
        def _get_param(models):
            return [p for m in models for p in (m.named_parameters() if hasattr(m, "named_parameters") else m)]

        {sep.join([f"{m}.zero_grad()" for m in self.models])}
        {sep.join([f"{k} = {v}" for k, v in initialized_layers.items()])}
        {sep.join([f"{k} = {v}" for k, v in self.others.items() if k != v])}
        {sep.join(self._forward_training_flow)}
        {sep.join(self._forward_inference_flow)}
        {sep.join([f"# self.{k} = {k}" for k in initialized_layers])}
        {sep.join([f"self.{k} = {k}" for k in self.others])}
        self._requires_grad_defaults = {{{defaults}}}
        self.need_update = True
        self.inputs = {self.inputs}
        self.outputs = {self.outputs}

    def training_step(self, {inputs}**kwargs):
        {sep.join(self._training_flow_parts[1])}
        return {self._forward_outputs}

    @torch.no_grad()
    def inference_step(self, {inputs}**kwargs):
        {sep.join(self._inference_flow_parts[1])}
        return {self._forward_outputs}

    def update(self, step: int) -> bool:
        {sep.join(need_update)}

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
        def _get_lr(opt):
            return opt.param_groups[0]["lr"]

        return {{k: _get_lr(v) for k, v in self.optimizers.items()}}

    @property
    def weight_decays(self):
        return get_decays(self.optimizers)

    @property
    def param_group_names(self):
        def _get_param_groups(opt):
            return [{{k: v for k, v in pg.items() if k != "params"}} for pg in opt.param_groups]

        return {{k: _get_param_groups(v) for k, v in self.optimizers.items()}}
"""


@dataclass(kw_only=True, slots=True)
class TorchLearnerBuilder(BaseLearnerBuilder[TorchLearnerIntermediate]):
    """Builder for PyTorch learners."""

    user_defined_learner_layer_type: ClassVar[type[TorchLearnerIntermediate]] = TorchLearnerIntermediate
    layer_builder_type: ClassVar[type[TorchBuilder]] = TorchBuilder

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
        return f"__grad_scaler_creator__(device=device_type{repr_mp_kw})", "GradScaler"

    def _get_optimizer(
        self,
        imports: defaultdict[str, set[str | None]],
        optimizer: ObjectPattern,
        trainable_layers: list[str],
    ) -> tuple[str, str]:
        opt_inst, opt_cls = resolve_object(imports, optimizer)
        return f"{opt_inst}(_get_param([{', '.join(trainable_layers)}]))", opt_cls


__all__ = ["TorchBuilder", "TorchLayerIntermediate", "TorchLearnerBuilder", "TorchLearnerIntermediate"]


if not TYPE_CHECKING:
    import sys

    from structcast.utils.lazy_import import LazySelectedImporter

    sys.modules[__name__] = LazySelectedImporter(__name__, globals())
