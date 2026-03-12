"""Build optimizers."""

from logging import getLogger
from re import Pattern as RePattern, compile as re_compile
from typing import TYPE_CHECKING, Any

from timm.optim import create_optimizer_v2
from timm.scheduler.scheduler_factory import create_scheduler_v2
from torch.optim import lr_scheduler

from structcast_model.base_trainer import GLOBAL_CALLBACKS
import torch

if TYPE_CHECKING:
    from torch.nn import Parameter
    from torch.optim import Optimizer
    from torch.optim.lr_scheduler import LRScheduler
else:
    Parameter = Any
    Optimizer = Any
    LRScheduler = Any


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


def _create_opt(
    params: list[tuple[str, Parameter]],
    layer_decay: float | None = None,
    layer_group_regexes: list[str] | None = None,
    weight_decay: float = 0.0,
    weight_decay_regexes: list[str] | None = None,
    no_weight_decay_regexes: list[str] | None = None,
    **kwargs: Any,
) -> tuple[bool, Optimizer]:
    """Create an optimizer with optional layer-wise learning rate decay and weight decay handling."""
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
    return has_lr_scale, create_optimizer_v2(parameters, weight_decay=weight_decay, **kwargs)


def _get_native_scheduler(optimizer: Optimizer, name: str, **kwargs: Any) -> LRScheduler:
    """Get the native scheduler."""
    # process "schedulers" key for SequentialLR, ChainedScheduler
    if "schedulers" in kwargs:
        kwargs["schedulers"] = [_get_native_scheduler(optimizer=optimizer, **s) for s in kwargs["schedulers"]]
    return getattr(lr_scheduler, name)(optimizer=optimizer, **kwargs)


def _create_native_scheduler(
    optimizer: Optimizer,
    name: str,
    has_lr_scale: bool,
    criterion: str | None = None,
    updates_per_epoch: int | None = None,
    **kwargs: Any,
) -> None:
    """Create the native scheduler."""
    scheduler = _get_native_scheduler(optimizer, name, **kwargs)
    if isinstance(scheduler, lr_scheduler.ReduceLROnPlateau):
        # ReduceLROnPlateau: step(self, metrics: SupportsFloat, epoch=None)
        if criterion is None:
            raise ValueError("criterion must be specified for ReduceLROnPlateau scheduler")
        GLOBAL_CALLBACKS.on_epoch_end.register(
            "lr_scheduler_step",
            lambda i, **kw: scheduler.step(i.logs()[criterion], i.epoch),  # type: ignore[arg-type]
        )
        if has_lr_scale:
            GLOBAL_CALLBACKS.on_epoch_end.register("lr_scale_set", lambda i, **kw: _set_lr_scale(optimizer))  # type: ignore[arg-type]
    elif isinstance(scheduler, lr_scheduler.CosineAnnealingWarmRestarts):
        # CosineAnnealingWarmRestarts: step(epoch + i / iters)
        if updates_per_epoch is None or updates_per_epoch <= 0:
            raise ValueError("updates_per_epoch must be a positive integer for CosineAnnealingWarmRestarts scheduler")
        GLOBAL_CALLBACKS.on_update.register(
            "lr_scheduler_step",
            lambda i, **kw: scheduler.step((i.update - 1) / updates_per_epoch),  # type: ignore[arg-type]
        )
        if has_lr_scale:
            GLOBAL_CALLBACKS.on_update.register("lr_scale_set", lambda i, **kw: _set_lr_scale(optimizer))  # type: ignore[arg-type]
    else:
        GLOBAL_CALLBACKS.on_epoch_end.register("lr_scheduler_step", lambda i, **kw: scheduler.step())  # type: ignore[arg-type]
        if has_lr_scale:
            GLOBAL_CALLBACKS.on_epoch_end.register("lr_scale_set", lambda i, **kw: _set_lr_scale(optimizer))  # type: ignore[arg-type]


def _create_timm_scheduler(optimizer: Optimizer, criterion: str, name: str, **kwargs: Any) -> None:
    """Create the timm scheduler."""
    scheduler, epochs = create_scheduler_v2(optimizer, sched=name, **kwargs)
    logger.info(f"Scheduled epochs: {epochs}. LR stepped per {'epoch' if scheduler.t_in_epochs else 'update'}.")
    GLOBAL_CALLBACKS.on_update.register(
        "lr_scheduler_step_update",
        lambda i, **kw: scheduler.step_update(i.update, i.logs()[criterion]),  # type: ignore[arg-type]
    )
    GLOBAL_CALLBACKS.on_epoch_end.register(
        "lr_scheduler_step",
        lambda i, **kw: scheduler.step(i.epoch, i.logs()[criterion]),  # type: ignore[arg-type]
    )


def _create_scheduler(optimizer: Optimizer, name: str, has_lr_scale: bool, **kwargs: Any) -> None:
    """Create the scheduler."""
    if hasattr(lr_scheduler, name):
        return _create_native_scheduler(optimizer=optimizer, name=name, has_lr_scale=has_lr_scale, **kwargs)
    return _create_timm_scheduler(optimizer=optimizer, name=name, **kwargs)


def _set_lr_scale(optimizer: Optimizer, delete_lr_scale: bool = False) -> None:
    for group in optimizer.param_groups:
        if "lr_scale" in group:
            if isinstance(group["lr"], torch.Tensor):
                group["lr"].mul_(group["lr_scale"])
            else:
                group["lr"] = group["lr"] * group["lr_scale"]
            if delete_lr_scale:
                del group["lr_scale"]


def create(params: list[tuple[str, Parameter]], **kwargs: Any) -> Optimizer:
    """Create an optimizer.

    Args:
        params (List[Tuple[str, Parameter]]): The model parameters.
        **kwargs: The optimizer arguments.

    Returns:
        Optimizer: The optimizer.
    """
    has_lr_scale, opt = _create_opt(params, **kwargs)
    if has_lr_scale:
        _set_lr_scale(opt, True)
    return opt


def create_with_scheduler(
    params: list[tuple[str, Parameter]],
    optimizer_kwargs: dict[str, Any],
    scheduler_kwargs: dict[str, Any],
) -> Optimizer:
    """Create an optimizer with scheduler.

    Args:
        params (List[Tuple[str, torch.nn.Parameter]]): The model parameters.
        optimizer_kwargs (Dict[str, Any]): The optimizer arguments.
        scheduler_kwargs (Dict[str, Any]): The scheduler arguments.

    Returns:
        Optimizer: The optimizer with scheduler.
    """
    has_lr_scale, opt = _create_opt(params, **optimizer_kwargs)
    _create_scheduler(opt, has_lr_scale=has_lr_scale, **scheduler_kwargs)
    return opt


__all__ = ["create", "create_with_scheduler"]


if not TYPE_CHECKING:
    import sys

    from structcast.utils.lazy_import import LazySelectedImporter

    sys.modules[__name__] = LazySelectedImporter(__name__, globals())
