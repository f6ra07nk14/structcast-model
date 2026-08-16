"""Build optimizers."""

from collections.abc import Callable, Mapping
from logging import getLogger
from re import Pattern as RePattern, compile as re_compile
from typing import TYPE_CHECKING, Any

from timm.optim import create_optimizer_v2

import torch

if TYPE_CHECKING:
    from torch.nn import Parameter
    from torch.optim import Optimizer
else:
    Parameter = Any
    Optimizer = Any


logger = getLogger(__name__)


def _match_no_weight_decay(
    name: str,
    parameter: Parameter,
    weight_decay_regexes: list[RePattern],
    no_weight_decay_regexes: list[RePattern],
) -> bool:
    """Check if the parameter should not be decayed."""
    if any(bool(p.match(name)) for p in weight_decay_regexes):
        return False
    return parameter.ndim <= 1 or any(bool(p.match(name)) for p in no_weight_decay_regexes)


def _get_layer_group_id(name: str, layer_groups: list[RePattern]) -> int:
    """Get the layer group id."""
    for i, pattern in enumerate(layer_groups):
        if bool(pattern.match(name)):
            return i
    return -1


def _param_groups_layer_decay(
    params: list[tuple[str, Parameter]],
    layer_decay: float,
    layer_group_regexes: list[RePattern],
    weight_decay: float,
    weight_decay_regexes: list[RePattern],
    no_weight_decay_regexes: list[RePattern],
) -> list[dict[str, Any]]:
    """Get the parameter groups with layer decay."""
    num_layer_groups = len(layer_group_regexes)
    layer_scales = [layer_decay ** (num_layer_groups - i) for i in range(num_layer_groups + 1)]
    pgs: dict[str, Any] = {}

    for name, param in params:
        if not param.requires_grad:
            continue
        if _match_no_weight_decay(
            name=name,
            parameter=param,
            weight_decay_regexes=weight_decay_regexes,
            no_weight_decay_regexes=no_weight_decay_regexes,
        ):
            g_decay = "no_decay"
            this_decay = 0.0
        else:
            g_decay = "decay"
            this_decay = weight_decay
        layer_id = _get_layer_group_id(name, layer_group_regexes)
        group_name = f"layer_{layer_id}_{g_decay}"
        if group_name not in pgs:
            this_scale = layer_scales[layer_id]
            # "lr_scale" only works with timm schedulers
            pgs[group_name] = {"lr_scale": this_scale, "weight_decay": this_decay, "params": [], "param_names": []}
        pgs[group_name]["params"].append(param)
        pgs[group_name]["param_names"].append(name)
    return list(pgs.values())


def _param_groups_weight_decay(
    params: list[tuple[str, Parameter]],
    weight_decay: float,
    weight_decay_regexes: list[RePattern],
    no_weight_decay_regexes: list[RePattern],
) -> list[dict[str, Any]]:
    """Get the parameter groups with weight decay."""
    decay = []
    no_decay = []
    decay_names = []
    no_decay_names = []
    for name, param in params:
        if param.requires_grad:
            if _match_no_weight_decay(
                name=name,
                parameter=param,
                weight_decay_regexes=weight_decay_regexes,
                no_weight_decay_regexes=no_weight_decay_regexes,
            ):
                no_decay.append(param)
                no_decay_names.append(name)
            else:
                decay.append(param)
                decay_names.append(name)
    return [
        {"params": no_decay, "weight_decay": 0.0, "param_names": no_decay_names},
        {"params": decay, "weight_decay": weight_decay, "param_names": decay_names},
    ]


def _native_optimizer(name: str) -> "type[Optimizer] | None":
    """Return the `torch.optim` class for an explicit ``torch.optim.X`` (or ``torch.X``) name, else None.

    Bare names ("sgd", "adamw") deliberately do NOT resolve here: timm's registry configures
    different defaults for several shared names (e.g. sgd with nesterov momentum), so bare names keep
    going to timm and the native engine must be requested explicitly.
    """
    attribute = name.removeprefix("torch.optim.") if name.startswith("torch.optim.") else name.removeprefix("torch.")
    if attribute == name:
        return None
    candidate = getattr(torch.optim, attribute, None)
    if isinstance(candidate, type) and issubclass(candidate, torch.optim.Optimizer):
        return candidate
    raise ValueError(f'"{name}" does not name a torch.optim optimizer class.')


def get_decays(optimizers: "Mapping[str, Optimizer]") -> dict[str, float]:
    """Flatten the per-group weight decay and layer-decay scale of every optimizer into metrics.

    Keys follow ``{optimizer}_group{index}_weight_decay`` / ``{optimizer}_group{index}_lr_scale``,
    so a logger can track how a schedule moves the values `create_opt` grouped, epoch by epoch.
    ``lr_scale`` only appears on the timm engine, which keeps it for its schedulers.
    """
    metrics: dict[str, float] = {}
    for name, optimizer in optimizers.items():
        for index, group in enumerate(optimizer.param_groups):
            for key in ("weight_decay", "lr_scale"):
                if key in group:
                    metrics[f"{name}_group{index}_{key}"] = group[key]
    return metrics


def set_lr_scale(optimizer: Optimizer, delete_lr_scale: bool = False) -> None:
    """Bake the `lr_scale` of every parameter group into its learning rate.

    Args:
        optimizer (Optimizer): The optimizer whose parameter groups to scale. Groups without an
            `lr_scale` key are left untouched.
        delete_lr_scale (bool): Whether to drop the `lr_scale` key afterwards, so a later call
            cannot apply the same scale twice.
    """
    for group in optimizer.param_groups:
        if "lr_scale" in group:
            if isinstance(group["lr"], torch.Tensor):
                group["lr"].mul_(group["lr_scale"])
            else:
                group["lr"] = group["lr"] * group["lr_scale"]
            if delete_lr_scale:
                del group["lr_scale"]


def create_opt(
    params: list[tuple[str, Parameter]],
    *,
    opt: "str | Callable[..., Optimizer]",
    layer_decay: float | None = None,
    layer_group_regexes: list[str] | None = None,
    weight_decay: float = 0.0,
    weight_decay_regexes: list[str] | None = None,
    no_weight_decay_regexes: list[str] | None = None,
    **kwargs: Any,
) -> Optimizer:
    """Create an optimizer over regex-grouped parameters.

    Grouping runs first and is engine-agnostic: layer decay emits one group per layer group and
    weight decay class, otherwise weight-decay regexes emit a decay and a no-decay group. When a
    group carries the weight decay, `weight_decay` is passed on as 0.0 so the engine default cannot
    override it.

    The engine is then chosen from *opt*: a callable is instantiated directly, an explicit
    `torch.optim.X` (or `torch.X`) name instantiates that class natively, and every bare name goes to
    `timm.optim.create_optimizer_v2` -- bare names keep timm's defaults (e.g. `sgd` with nesterov
    momentum), so configurations migrated from `create_with_scheduler` behave identically.

    Layer decay emits an `lr_scale` per group, which only timm schedulers consume. For the callable
    and native engines the scale is therefore baked into the learning rate right away; on the timm
    fallback the `lr_scale` keys are kept for the scheduler, and callers running such an optimizer
    without a timm scheduler have to call `set_lr_scale` themselves.

    Args:
        params (list[tuple[str, Parameter]]): The named model parameters to optimize.
        opt (str | Callable[..., Optimizer]): The optimizer class, factory, or name.
        layer_decay (float | None): The per-layer-group learning rate decay, disabled when None or 0.
        layer_group_regexes (list[str] | None): Regexes matching the parameters of each layer group,
            in order; parameters matching none of them form the first group.
        weight_decay (float): The weight decay applied to the decaying parameters.
        weight_decay_regexes (list[str] | None): Regexes matching parameters that must decay.
        no_weight_decay_regexes (list[str] | None): Regexes matching parameters that must not decay.
        **kwargs: Further keyword arguments for the optimizer engine, e.g. `lr`.

    Returns:
        Optimizer: The created optimizer.
    """
    wd_regexes = [re_compile(r) for r in weight_decay_regexes or []]
    nwd_regexes = [re_compile(r) for r in no_weight_decay_regexes or []]
    has_lr_scale = False
    if layer_decay is not None:
        logger.info(f"Using layer decay: {layer_decay}")
    if weight_decay > 0.0:
        logger.info(f"Using layer decay with weight decay: {weight_decay}")
    if layer_decay is not None and layer_decay > 0.0:
        parameters: Any = _param_groups_layer_decay(
            params,
            layer_decay=layer_decay,
            layer_group_regexes=[re_compile(r) for r in layer_group_regexes or []],
            weight_decay=weight_decay,
            weight_decay_regexes=wd_regexes,
            no_weight_decay_regexes=nwd_regexes,
        )
        has_lr_scale = True
        weight_decay = 0.0
    elif weight_decay > 0.0 and (wd_regexes or nwd_regexes):
        parameters = _param_groups_weight_decay(
            params, weight_decay=weight_decay, weight_decay_regexes=wd_regexes, no_weight_decay_regexes=nwd_regexes
        )
        weight_decay = 0.0
    else:
        parameters = params
    if callable(opt):
        engine: Callable[..., Optimizer] = opt
    else:
        native = _native_optimizer(opt)
        if native is None:
            return create_optimizer_v2(parameters, opt=opt, weight_decay=weight_decay, **kwargs)
        engine = native
    optimizer = engine(parameters, weight_decay=weight_decay, **kwargs)
    if has_lr_scale:
        set_lr_scale(optimizer, True)
    return optimizer


__all__ = ["create_opt", "get_decays", "set_lr_scale"]


if not TYPE_CHECKING:
    import sys

    from structcast.utils.lazy_import import LazySelectedImporter

    sys.modules[__name__] = LazySelectedImporter(__name__, globals())
