"""Builder for PyTorch models."""

from collections import defaultdict
from dataclasses import dataclass
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
        inputs = self._forward_inputs
        inputs += ", " if inputs else ""
        return f"""\
class {class_name}(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.inputs = {self.inputs}
        self.outputs = {self.outputs}
        {sep.join([f"self.{v}" for v in initialized_layers])}

    def forward(self, {inputs}**kwargs):
        {sep.join(codes)}
        return {self._forward_outputs}
"""


@dataclass(kw_only=True, slots=True)
class TorchBuilder(BaseModelBuilder[TorchLayerIntermediate]):
    """Builder for PyTorch models."""

    user_defined_layer_type: ClassVar[type[TorchLayerIntermediate]] = TorchLayerIntermediate


class TorchBackwardIntermediate(BackwardIntermediate):
    """Intermediate representation of a PyTorch backward layer."""

    def _entry_optimizer_steps(self, opts: list[str], *, indent: str) -> list[str]:
        """Generate unscale / clip / step / scaler-update lines for one backward entry."""
        has_mp = self.mixed_precision is not None
        scaler_opt = opts[0]
        lines: list[str] = []
        for n in opts:
            if has_mp:
                lines.append(f"{indent}self.{scaler_opt}_scaler.unscale_(self.{n})")
            if self.optimizers[n][2]:
                param = f"[p for pg in self.{n}.param_groups for p in pg['params']]"
                lines.append(f"{indent}self.{n}_clip({param})")
        for n in opts:
            if has_mp:
                lines.append(f"{indent}self.{scaler_opt}_scaler.step(self.{n})")
            else:
                lines.append(f"{indent}self.{n}.step()")
        if has_mp:
            lines.append(f"{indent}self.{scaler_opt}_scaler.update()")
        return lines

    def _build_flow_no_accumulation(self) -> list[str]:
        """Build per-entry flow (backward → step → zero_grad) for non-accumulation mode."""
        has_mp = self.mixed_precision is not None
        flow: list[str] = []
        for L, kw, opts in self.backwards:
            scaler_opt = opts[0]
            backward_target = f"self.{scaler_opt}_scaler.scale({L})" if has_mp else L
            flow.append(f"{backward_target}.backward({kw})")
            flow.extend(self._entry_optimizer_steps(opts, indent=""))
            for n in opts:
                flow.append(f"self.{n}.zero_grad()")
        return flow

    def _build_flow_accumulation(self, indent: str) -> list[str]:
        """Build all-backward-first flow for gradient-accumulation mode."""
        has_mp = self.mixed_precision is not None
        flow: list[str] = []
        for L, _, _ in self.backwards:
            flow.append(f"{L} = {L} / {self.accumulate_gradients}")
        for L, kw, opts in self.backwards:
            scaler_opt = opts[0]
            backward_target = f"self.{scaler_opt}_scaler.scale({L})" if has_mp else L
            flow.append(f"{backward_target}.backward({kw})")
        flow.append("if self.need_update:")
        for _, _, opts in self.backwards:
            flow.extend(self._entry_optimizer_steps(opts, indent=indent))
            for n in opts:
                flow.append(f"{indent}self.{n}.zero_grad()")
        return flow

    def _build_init_opts(self) -> list[str]:
        """Build __init__ statements for optimizers, clips and grad scalers."""
        has_mp = self.mixed_precision is not None
        init_opts = [f"self.{n} = {o}(_get_param([{', '.join(L)}]))" for n, (o, L, _) in self.optimizers.items()]
        for n, (_, _, clip) in self.optimizers.items():
            if clip:
                init_opts.append(f"self.{n}_clip = {clip}")
        if has_mp:
            for scaler_opt in {opts[0] for _, _, opts in self.backwards}:
                init_opts.append(f"self.{scaler_opt}_scaler = {self.mixed_precision}")
        return init_opts

    def _get_scripts(self) -> list[str]:
        indent = " " * 4
        sep = "\n" + indent * 2
        has_mp = self.mixed_precision is not None
        init_opts = self._build_init_opts()
        if self.accumulate_gradients:
            flow = self._build_flow_accumulation(indent)
        else:
            flow = self._build_flow_no_accumulation()
        opts_repr = ", ".join([f'"{n}": self.{n}' for n in self.optimizers])
        grad_scalers = (
            ", ".join([f'"{opts[0]}": self.{opts[0]}_scaler' for _, _, opts in self.backwards]) if has_mp else ""
        )
        need_update = ["return self.need_update"]
        if self.accumulate_gradients:
            need_update = [f"self.need_update = (step + 1) % {self.accumulate_gradients} == 0"] + need_update
        res = f"""\
class {self.classname}:

    def __init__(self, {self._backward_models}, **kwargs):
        def _get_param(models):
            return [p for m in models for p in (m.named_parameters() if hasattr(m, "named_parameters") else m)]

        {sep.join(init_opts)}
        self.mixed_precision_type = "{self.mixed_precision_type}"
        self.need_update = True

    def update(self, step: int) -> bool:
        {sep.join(need_update)}

    def __call__(self, {self._backward_losses}, **kwargs):
        {sep.join(flow)}

    @property
    def optimizers(self):
        return {{{opts_repr}}}

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
