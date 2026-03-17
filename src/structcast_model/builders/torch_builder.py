"""Builder for PyTorch models."""

from collections import defaultdict
from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, Any, ClassVar

from structcast_model.builders.base_builder import (
    BackwardIntermediate,
    BaseBackwardBuilder,
    BaseModelBuilder,
    LayerIntermediate,
    resolve_getter,
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
        return f"""\
class {class_name}(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.inputs = {self.inputs}
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

    @cached_property
    def _backward_flow(self) -> list[str]:
        return [
            f"{L if self.mixed_precision is None else f'self.{n}_scaler.scale({L})'}.backward({kw})"
            for L, kw, opts in self.backwards
            for n in opts
        ]

    def _get_scripts(self) -> list[str]:
        indent = " " * 4
        sep = "\n" + indent * 2
        init_opts = [f"self.{n} = {o}(_get_param([{', '.join(L)}]))" for n, (o, L, _) in self.optimizers.items()]
        flow = []
        if self.accumulate_gradients:
            flow += [f"{L} = {L} / {self.accumulate_gradients}" for L, _, _ in self.backwards]
            flow += self._backward_flow
            flow += ["if self.need_update:"]
            flow_indent = indent
        else:
            flow += self._backward_flow
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
        opts = ", ".join([f'"{n}": self.{n}' for n in self.optimizers])
        if self.mixed_precision is None:
            grad_scalers = ""
        else:
            grad_scalers = ", ".join([f'"{n}": self.{n}_scaler' for n in self.optimizers])
        res = f"""\
class {self.classname}:

    def __init__(self, {self._backward_models}, **kwargs):
        def _get_param(models):
            return [p for m in models for p in (m.named_parameters() if hasattr(m, "named_parameters") else m)]

        {sep.join(init_opts)}
        self.mixed_precision_type = "{self.mixed_precision_type}"
        self.need_update = False

    def update(self, step: int) -> bool:
        self.need_update = (step + 1) % {self.accumulate_gradients} == 0
        return self.need_update

    def __call__(self, {self._backward_losses}, **kwargs):
        {sep.join(flow)}

    @property
    def optimizers(self):
        return {{{opts}}}

    @property
    def grad_scalers(self):
        return {{{grad_scalers}}}

    @property
    def learning_rates(self):
        def _get_lr(opt):
            return opt.param_groups[0]["lr"]

        return {{k: _get_lr(v) for k, v in self.optimizers.items()}}

    @property
    def param_group_names(self):
        def _get_param_groups(opt):
            return [{{k: v for k, v in pg.items() if k != "params"}} for pg in opt.param_groups]

        return {{k: _get_param_groups(v) for k, v in self.optimizers.items()}}
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


__all__ = ["TorchBackwardBuilder", "TorchBackwardIntermediate", "TorchBuilder", "TorchLayerIntermediate"]


if not TYPE_CHECKING:
    import sys

    from structcast.utils.lazy_import import LazySelectedImporter

    sys.modules[__name__] = LazySelectedImporter(__name__, globals())
