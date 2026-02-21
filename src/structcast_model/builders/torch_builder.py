"""Builder for PyTorch models."""

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, ClassVar

from structcast_model.builders.base_builder import (
    BackwardIntermediate,
    BaseBackwardBuilder,
    BaseModelBuilder,
    LayerIntermediate,
    resolve_getter,
)


class TorchLayerIntermediate(LayerIntermediate):
    """Intermediate representation of a PyTorch layer."""

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
        return f"""\
class {class_name}(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.outputs = {self.outputs}
        {sep.join([f"self.{v}" for v in initialized_layers])}

    def forward(self, {self._forward_inputs}, **kwargs):
        {sep.join(codes)}
        return {self._forward_outputs}
"""


@dataclass(kw_only=True, slots=True)
class TorchBuilder(BaseModelBuilder[TorchLayerIntermediate]):
    """Builder for PyTorch models."""

    user_defined_layer_type: ClassVar[type[TorchLayerIntermediate]] = TorchLayerIntermediate


class TorchBackwardIntermediate(BackwardIntermediate):
    """Intermediate representation of a PyTorch backward layer."""

    def _get_scripts(self) -> list[str]:
        indent = " " * 4
        sep = "\n" + indent * 2
        init_opts = [f"self.{n} = {o}(_get_param([{', '.join(L)}]))" for n, (o, L, _) in self.optimizers.items()]
        flow = []
        if self.accumulate_gradients:
            flow.extend([f"({L} / {self.accumulate_gradients}).backward({kw})" for L, kw, _ in self.backwards])
            flow.append(f"if (step + 1) % {self.accumulate_gradients} == 0:")
            flow_indent = indent
        else:
            flow += [f"{L}.backward({kw})" for L, kw, _ in self.backwards]
            flow_indent = ""
        for name in [n for _, _, opts in self.backwards for n in opts]:
            if has_mp := self.mixed_precision is not None:
                init_opts.append(f"self.{name}_scaler = {self.mixed_precision}")
            flow.append(f"{flow_indent}self.{name}_scaler.unscale_(self.{name})")
            if clip := self.optimizers[name][2]:  # if clip_grad is not None
                init_opts.append(f"self.{name}_clip = {clip}")
                param = f"[p for pg in self.{name}.param_groups for p in pg['params']]"
                flow.append(f"{flow_indent}self.{name}_clip({param})")
            if has_mp:
                flow.append(f"{flow_indent}self.{name}_scaler.step(self.{name})")
                flow.append(f"{flow_indent}self.{name}_scaler.update()")
            else:
                flow.append(f"{flow_indent}self.{name}.step()")
            flow.append(f"{flow_indent}self.{name}.zero_grad()")
        res = f"""\
class {self.classname}:

    def __init__(self, {self._backward_models}, **kwargs):
        def _get_param(models):
            return [p for m in models for p in (m.named_parameters() if hasattr(m, "named_parameters") else m)]

        {sep.join(init_opts)}

    def __call__(self, step, {self._backward_losses}, **kwargs):
        {sep.join(flow)}
        return should_update

    @property
    def learning_rates(self) -> dict[str, float]:
        def _get_lr(opt):
            return opt.param_groups[0]["lr"]

        return {{{", ".join([f'"{n}_lr": _get_lr(self.{n})' for n in self.optimizers])}}}

    @property
    def param_group_names(self) -> dict[str, list[dict[str, Any]]]:
        def _get_param_groups(opt):
            return [{{k: v for k, v in pg.items() if k != "params"}} for pg in opt.param_groups]

        return {{{", ".join([f'"{n}": _get_param_groups(self.{n})' for n in self.optimizers])}}}
"""
        return [res]


@dataclass(kw_only=True, slots=True)
class TorchBackwardBuilder(BaseBackwardBuilder[TorchBackwardIntermediate]):
    """Builder for PyTorch backward layers."""

    user_defined_backward_layer_type: ClassVar[type[TorchBackwardIntermediate]] = TorchBackwardIntermediate

    def _get_mixed_precision(
        self,
        imports: defaultdict[str, set[str | None]],
        mixed_precision: bool | dict[str, Any],
    ) -> str | None:
        if isinstance(mixed_precision, bool):
            if not mixed_precision:
                return None
            mixed_precision = {}
        imports["torch.amp"].add(None)
        repr_mp_kw = ", ".join(f"{k}={resolve_getter(imports, v)}" for k, v in mixed_precision.items())
        return f"torch.amp.GradScaler({repr_mp_kw})"
