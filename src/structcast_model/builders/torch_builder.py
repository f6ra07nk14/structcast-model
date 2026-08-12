"""Builder for PyTorch models."""

from collections import defaultdict
from dataclasses import dataclass
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

    default_imports: ClassVar[dict[str, set[str | None]]] = {"torch": {None}}
    """Default imports for PyTorch layers."""

    def _with_autocast(self, flow: list[str]) -> list[str]:
        if not (self.mixed_precision_type and flow):
            return flow
        autocast = f"with torch.autocast(device_type, torch.{self.mixed_precision_type}):"
        return [autocast] + [f"{' ' * 4}{L}" for L in flow]

    def _wrap_step_function(self, name: str, flow: list[str], extra_params: str = "") -> list[str]:
        """Wrap the given flow in a step function definition."""
        inputs = self._forward_inputs
        inputs += ", " if inputs else ""
        prefix = [f"def {name}({extra_params}{inputs}**kwargs):"]
        body = [f"{' ' * 4}{L}" for L in flow]
        suffix = [f"{' ' * 4}return {self._forward_outputs}"]
        return prefix + body + suffix

    def _get_forward_inference_flow(self) -> list[str]:
        """Get the code for the inference flow in the forward method."""
        return self._wrap_step_function("_inference_step", self._with_autocast(super()._get_forward_inference_flow()))

    def _get_forward_training_flow(self) -> list[str]:
        flow, start = [], 0

        def _param(layers: list[str]) -> str:
            if len(layers) == 1:
                return f"{layers[0]}.parameters()"
            return f"(p for m in ({', '.join(f'{L}' for L in layers)}) for p in m.parameters())"

        for unit in self.flow:
            if len(unit) == 3:
                flow.append(self._get_regular_step(*unit))
                continue
            loss, backward_kwargs, optimizer_name, clip_name, mixed_precision_name, trainable_layers = unit
            preset = [f"{m}.{'train' if m in trainable_layers else 'eval'}()" for m in self.models]
            flow = flow[:start] + preset + self._with_autocast(flow[start:])
            if self.accumulate_gradients:
                flow.append(f"{loss} = {loss} / {self.accumulate_gradients}")
            if mixed_precision_name is None:
                flow.append(f"{loss}.backward({backward_kwargs})")
            else:
                flow.append(f"{mixed_precision_name}.scale({loss}).backward({backward_kwargs})")
            if self.accumulate_gradients:
                flow.append("if __need_update__:")
                indent = " " * 4
            else:
                indent = ""
            if mixed_precision_name is None:
                if clip_name is not None:
                    flow.append(f"{indent}{clip_name}({_param(trainable_layers)})")
                flow.append(f"{indent}{optimizer_name}.step()")
            else:
                if clip_name is not None:
                    flow.append(f"{indent}{mixed_precision_name}.unscale_({optimizer_name})")
                    flow.append(f"{indent}{clip_name}({_param(trainable_layers)})")
                flow.append(f"{indent}{mixed_precision_name}.step({optimizer_name})")
                flow.append(f"{indent}{mixed_precision_name}.update()")
            flow.append(f"{indent}{optimizer_name}.zero_grad()")
            start = len(flow)
        return self._wrap_step_function("_training_step", flow, extra_params="__need_update__, ")

    def _get_learner_script(self, initialized_layers: dict[str, str]) -> str:
        """Get the script for the learner."""
        indent = " " * 4
        sep = "\n" + indent * 2
        models_repr = ", ".join([f'"{m}": self.{m}' for m in self.models])
        opts_repr = ", ".join([f'"{n}": self.{n}' for n in self.optimizers])
        grad_scalers_repr = ", ".join([f'"{n}": self.{n}' for n in self.mixed_precision_scales])
        need_update = ["return self.need_update"]
        if self.accumulate_gradients:
            need_update = [f"self.need_update = (step + 1) % {self.accumulate_gradients} == 0"] + need_update
        inputs = self._forward_inputs
        inputs += ", " if inputs else ""
        return f"""\
class {self.classname}:

    def __init__(self, {self._learner_models}, **kwargs):
        device_type = next({self.models[0]}.parameters()).device.type
        def _get_param(models):
            return [p for m in models for p in (m.named_parameters() if hasattr(m, "named_parameters") else m)]

        {sep.join([f"{m}.zero_grad()" for m in self.models])}
        {sep.join([f"{k} = {v}" for k, v in initialized_layers.items()])}
        {sep.join([f"{k} = {v}" for k, v in self.others.items() if k != v])}
        {sep.join(self._forward_training_flow)}
        {sep.join(self._forward_inference_flow)}
        self.forward_training_step = _training_step
        self.forward_inference_step = _inference_step
        {sep.join([f"# self.{k} = {k}" for k in initialized_layers])}
        {sep.join([f"self.{k} = {k}" for k in self.others])}
        self.need_update = True
        self.inputs = {self.inputs}
        self.outputs = {self.outputs}

    def update(self, step: int) -> bool:
        {sep.join(need_update)}

    def training_step(self, {inputs}**kwargs):
        return self.forward_training_step(self.need_update, {inputs}**kwargs)

    @torch.no_grad()
    def inference_step(self, {inputs}**kwargs):
        return self.forward_inference_step({inputs}**kwargs)

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
    def learning_rates(self):
        def _get_lr(opt):
            return opt.param_groups[0]["lr"]

        return {{k: _get_lr(v) for k, v in self.optimizers.items()}}

    @property
    def weight_decays(self):
        from structcast_model.torch.optimizers import get_decays

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
    ) -> tuple[str, str | None]:
        if isinstance(mixed_precision, bool):
            if not mixed_precision:
                return "", None
            mixed_precision = {}
        imports["torch.amp"].add(None)
        repr_mp_kw = ", ".join(f"{k}={resolve_getter(imports, v)}" for k, v in mixed_precision.items())
        return f"torch.amp.GradScaler({repr_mp_kw})", "GradScaler"

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
