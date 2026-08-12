"""Distributed strategies: how models are wrapped, synchronized, and turned into checkpointable state.

A distributed strategy is the replaceable unit deciding how models are wrapped for a training run,
how their initial weights are made identical across ranks, and how model/optimizer state becomes a
loadable checkpoint. Exactly one strategy is active per run; single-device training uses
:class:`SingleDeviceStrategy` rather than a special case.

Every ``state_dict``/``load_state_dict`` implementation routes through
``torch.distributed.checkpoint.state_dict`` so checkpoint keys are identical for raw,
``torch.compile``'d, DDP-wrapped, and ``fully_shard``'d models.
"""

from collections import OrderedDict
from collections.abc import Callable, Mapping
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass, field
from itertools import chain
from logging import getLogger
from typing import TYPE_CHECKING, Any

from typing_extensions import Protocol, runtime_checkable

import torch

logger = getLogger(__name__)

try:  # torch >= 2.6 ships the stable per-parameter sharding (FSDP2) API.
    from torch.distributed.fsdp import (
        FSDPModule as _FSDP_MODULE,
        MixedPrecisionPolicy as _MIXED_PRECISION_POLICY,
        fully_shard as _fully_shard,
    )
except ImportError:  # pragma: no cover - exercised only on old torch builds
    _FSDP_MODULE = None
    _MIXED_PRECISION_POLICY = None
    _fully_shard = None

try:  # torch >= 2.2; the older builds admitted by the torch-cpu extra floor lack both modules.
    from torch.distributed.checkpoint import state_dict as _dcp_state_dict
    from torch.distributed.device_mesh import init_device_mesh as _init_device_mesh
except ImportError:  # pragma: no cover - exercised only on old torch builds
    _dcp_state_dict = None
    _init_device_mesh = None

_DTYPES = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}

_WRAPPER_PREFIXES = ("module.", "_orig_mod.")


def _state_dict_api() -> Any:
    """Return the ``torch.distributed.checkpoint.state_dict`` module, or ``None`` on old torch."""
    return _dcp_state_dict


def _strip_wrapper_prefixes(state: dict[str, Any]) -> dict[str, Any]:
    """Strip ``module.`` / ``_orig_mod.`` wrapper prefixes from state-dict keys (old-torch fallback)."""

    def _clean(key: str) -> str:
        changed = True
        while changed:
            changed = False
            for prefix in _WRAPPER_PREFIXES:
                if key.startswith(prefix):
                    key = key[len(prefix) :]
                    changed = True
        return key

    return {_clean(k): v for k, v in state.items()}


def sync_gate(module: Any, armed: bool) -> AbstractContextManager[None]:
    """Return a context gating whether *module*'s next backward participates in gradient sync.

    Generated training steps wrap every model invocation in this gate. ``armed`` is computed at
    code-generation time as "the model is owned by the current optimizer segment AND this is its
    last invocation in the segment", multiplied at runtime by the update-step flag, so gradient
    all-reduce (DDP) or reduce-scatter (FSDP2) fires exactly once per update, on the final backward.
    Plain modules get a null context, keeping single-device flow functions fully traceable.
    """
    if isinstance(module, torch.nn.parallel.DistributedDataParallel):
        return nullcontext() if armed else module.no_sync()
    if _FSDP_MODULE is not None and isinstance(module, _FSDP_MODULE):
        return nullcontext() if armed else _NoGradientSync(module)
    return nullcontext()


class _NoGradientSync:
    """Context manager disabling FSDP2 gradient reduce-scatter while it is active."""

    def __init__(self, module: Any) -> None:
        self._module = module

    def __enter__(self) -> None:
        self._module.set_requires_gradient_sync(False)

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self._module.set_requires_gradient_sync(True)


@runtime_checkable
class DistributedStrategy(Protocol):
    """Protocol for the strategy owning the distributed lifecycle of a training run.

    ``wrap`` must run after shape resolution and initializers but before the learner is
    constructed, because generated learner step closures and optimizers capture the exact module
    objects handed to ``__init__``. ``sync_initial_weights`` must run before ``wrap`` so all
    implementations broadcast plain tensors.
    """

    grad_scaler_creator: Callable[..., Any]
    """Callable creating gradient scalers for fp16 learners built under this strategy."""

    def wrap(self, models: "OrderedDict[str, torch.nn.Module]") -> "OrderedDict[str, torch.nn.Module]":
        """Wrap the models for this strategy and return the wrapped mapping."""

    def sync_initial_weights(self, models: Mapping[str, torch.nn.Module]) -> None:
        """Make every rank's initial weights identical. Must be called on every rank, before wrap."""

    def state_dict(
        self,
        models: Mapping[str, torch.nn.Module],
        optimizers: Mapping[str, Any] | None = None,
        optimizer_models: Mapping[str, list[str]] | None = None,
    ) -> dict[str, Any]:
        """Produce ``{"models": ..., "optimizers": ...}`` with wrapper-free keys.

        Collective for distributed strategies: every rank must call it; rank 0 receives the
        gathered tensors and the other ranks receive empty mappings.
        """

    def load_state_dict(
        self,
        models: Mapping[str, torch.nn.Module],
        optimizers: Mapping[str, Any],
        optimizer_models: Mapping[str, list[str]] | None,
        state: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Load a training state produced by :meth:`state_dict` back into models and optimizers.

        Collective for distributed strategies: rank 0 passes the loaded *state*, the other ranks
        pass ``None``. Returns the non-tensor parts of the state (``meta``, ``grad_scalers``) on
        every rank.
        """


def _optimizer_container(
    models: Mapping[str, torch.nn.Module],
    names: list[str],
) -> torch.nn.Module:
    """Group the models an optimizer owns so DCP can resolve its parameter FQNs."""
    return torch.nn.ModuleDict({n: models[n] for n in names})


class _StateDictMixin:
    """Shared DCP-based state production and loading."""

    _broadcast_on_load = False

    def _options(self, api: Any) -> Any:
        return api.StateDictOptions(full_state_dict=True, cpu_offload=True)

    def _load_options(self, api: Any) -> Any:
        return api.StateDictOptions(full_state_dict=True, broadcast_from_rank0=self._broadcast_on_load)

    def state_dict(
        self,
        models: Mapping[str, torch.nn.Module],
        optimizers: Mapping[str, Any] | None = None,
        optimizer_models: Mapping[str, list[str]] | None = None,
    ) -> dict[str, Any]:
        """Produce wrapper-free model (and optimizer) state dicts. See :class:`DistributedStrategy`."""
        api = _state_dict_api()
        if api is None:
            states: dict[str, Any] = {"models": {n: _strip_wrapper_prefixes(m.state_dict()) for n, m in models.items()}}
            if optimizers is not None:
                states["optimizers"] = {n: o.state_dict() for n, o in optimizers.items()}
            return states
        options = self._options(api)
        states = {"models": {n: api.get_model_state_dict(m, options=options) for n, m in models.items()}}
        if optimizers is not None:
            if optimizer_models:
                states["optimizers"] = {
                    n: api.get_optimizer_state_dict(
                        _optimizer_container(models, optimizer_models[n]), o, options=options
                    )
                    for n, o in optimizers.items()
                }
            else:
                self._require_pairing_or_warn("saving")
                states["optimizers"] = {n: o.state_dict() for n, o in optimizers.items()}
        return states

    def load_state_dict(
        self,
        models: Mapping[str, torch.nn.Module],
        optimizers: Mapping[str, Any],
        optimizer_models: Mapping[str, list[str]] | None,
        state: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Load a saved training state. See :class:`DistributedStrategy`."""
        state = self._share_state(state)
        api = _state_dict_api()
        model_states = state.get("models", {})
        optimizer_states = state.get("optimizers", {})
        if api is None:
            for name, module in models.items():
                module.load_state_dict(model_states[name])
            for name, optimizer in optimizers.items():
                optimizer.load_state_dict(optimizer_states[name])
            return state
        for name, module in models.items():
            api.set_model_state_dict(module, model_states.get(name, {}), options=self._load_options(api))
        if optimizer_models:
            for name, optimizer in optimizers.items():
                self._set_optimizer_state(
                    api,
                    _optimizer_container(models, optimizer_models[name]),
                    optimizer,
                    optimizer_states.get(name, {}),
                )
        elif optimizers:
            self._require_pairing_or_warn("loading")
            for name, optimizer in optimizers.items():
                optimizer.load_state_dict(optimizer_states[name])
        return state

    def _set_optimizer_state(self, api: Any, container: torch.nn.Module, optimizer: Any, osd: dict[str, Any]) -> None:
        """Load one optimizer's saved state, which arrives in full on every rank.

        Optimizer states load without the broadcast option: torch's ``set_optimizer_state_dict``
        must infer a placement device from the *local* optimizer state, and stateless optimizers
        (e.g. plain SGD) never have one. For those, the saved state carries nothing but
        hyperparameters, which are restored directly.
        """
        if any(osd.get("state", {}).values()):
            options = api.StateDictOptions(full_state_dict=True)
            api.set_optimizer_state_dict(container, optimizer, optim_state_dict=osd, options=options)
            return
        for group, saved in zip(optimizer.param_groups, osd.get("param_groups", []), strict=False):
            group.update({k: v for k, v in saved.items() if k != "params"})

    def _share_state(self, state: dict[str, Any] | None) -> dict[str, Any]:
        """Make the non-tensor state parts available on this rank. Overridden by distributed strategies."""
        if state is None:
            raise ValueError("A training state is required to resume from.")
        return state

    def _require_pairing_or_warn(self, action: str) -> None:
        """React to a learner that does not expose its optimizer-to-models pairing."""
        logger.warning(
            "The learner exposes no optimizer_models pairing; %s optimizer state with plain "
            "state_dict keys instead of parameter names.",
            action,
        )


def _shared_meta(state: dict[str, Any] | None) -> dict[str, Any]:
    """Broadcast everything but the model tensors from rank 0 to every rank.

    Model states stay rank-0-only and travel through ``set_model_state_dict``'s efficient
    ``broadcast_from_rank0`` path; optimizer states are object-broadcast here because that path
    cannot infer a device from stateless optimizers such as plain SGD.
    """
    payload = [None if state is None else {k: v for k, v in state.items() if k != "models"}]
    torch.distributed.broadcast_object_list(payload, src=0)
    shared = payload[0] or {}
    if state is None:
        return {"models": {}, **shared}
    return state


@dataclass(kw_only=True)
class SingleDeviceStrategy(_StateDictMixin):
    """Strategy for single-device training: no wrapping, no cross-rank synchronization."""

    device: str
    """Device the run trains on, e.g. ``"cuda:0"`` or ``"cpu"``."""

    local_rank: int = 0
    """Local rank; unused on a single device, accepted for a uniform constructor."""

    grad_scaler_creator: Callable[..., Any] = torch.amp.GradScaler

    def wrap(self, models: "OrderedDict[str, torch.nn.Module]") -> "OrderedDict[str, torch.nn.Module]":
        """Return the models unchanged."""
        return models

    def sync_initial_weights(self, models: Mapping[str, torch.nn.Module]) -> None:
        """Nothing to synchronize on a single device."""


@dataclass(kw_only=True)
class DistributedDataParallelStrategy(_StateDictMixin):
    """Strategy wrapping every model in ``DistributedDataParallel``."""

    device: str
    """Device this rank trains on."""

    local_rank: int = 0
    """Local rank used as the single CUDA device id; ignored on CPU."""

    grad_scaler_creator: Callable[..., Any] = torch.amp.GradScaler

    _broadcast_on_load = True

    def wrap(self, models: "OrderedDict[str, torch.nn.Module]") -> "OrderedDict[str, torch.nn.Module]":
        """Wrap every model in DDP. ``device_ids`` must be ``None`` on CPU, where passing one raises."""
        device_ids = None if "cpu" in self.device else [self.local_rank]
        return OrderedDict(
            (n, torch.nn.parallel.DistributedDataParallel(m, device_ids=device_ids)) for n, m in models.items()
        )

    def sync_initial_weights(self, models: Mapping[str, torch.nn.Module]) -> None:
        """Broadcast rank 0's parameters and buffers, making rank-0-only initializers authoritative.

        DDP's constructor also broadcasts, but that is a side effect of a later call; this method
        owns the guarantee explicitly, on plain pre-wrap tensors, uniformly with FSDP2.
        """
        for module in models.values():
            for tensor in chain(module.parameters(), module.buffers()):
                torch.distributed.broadcast(tensor.data, src=0)

    def _share_state(self, state: dict[str, Any] | None) -> dict[str, Any]:
        return _shared_meta(state)


@dataclass(kw_only=True)
class FullyShardedDataParallelStrategy(_StateDictMixin):
    """Strategy sharding every model in place with ``fully_shard`` (FSDP2).

    Requires torch >= 2.6. fp16 learners built under this strategy use the plain
    ``torch.amp.GradScaler``: since torch 2.5 the DTensor dispatcher all-reduces ``found_inf``
    across ranks inside ``unscale_``, so no sharded scaler class is needed.
    """

    device: str
    """Device this rank trains on."""

    local_rank: int = 0
    """Local rank; the device mesh is derived from the default process group."""

    reshard_after_forward: bool = True
    """Whether to reshard parameters after forward, trading memory for an extra all-gather."""

    mp_policy: dict[str, str] | None = None
    """Mixed precision policy dtypes by ``MixedPrecisionPolicy`` field name, e.g. ``{"param_dtype": "bfloat16"}``."""

    grad_scaler_creator: Callable[..., Any] = torch.amp.GradScaler

    _broadcast_on_load = True
    _mesh: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        """Fail loud when the installed torch has no stable ``fully_shard``."""
        if _fully_shard is None:
            raise ImportError(
                "FullyShardedDataParallelStrategy requires torch>=2.6 for the stable fully_shard "
                "(FSDP2) API; the installed torch does not provide torch.distributed.fsdp.fully_shard."
            )

    def wrap(self, models: "OrderedDict[str, torch.nn.Module]") -> "OrderedDict[str, torch.nn.Module]":
        """Shard every model in place with ``fully_shard`` and return the same (now sharded) modules."""
        if self._mesh is None:
            # Without an explicit mesh, fully_shard follows the accelerator, which reports CUDA on
            # CUDA-enabled builds even when this strategy trains on CPU — the mesh must follow the
            # strategy's device instead.
            device_type = "cpu" if "cpu" in self.device else self.device.split(":")[0]
            self._mesh = _init_device_mesh(device_type, (torch.distributed.get_world_size(),))
        kwargs: dict[str, Any] = {"reshard_after_forward": self.reshard_after_forward, "mesh": self._mesh}
        if self.mp_policy:
            kwargs["mp_policy"] = _MIXED_PRECISION_POLICY(**{k: _DTYPES[v] for k, v in self.mp_policy.items()})
        return OrderedDict((n, _fully_shard(m, **kwargs)) for n, m in models.items())

    def sync_initial_weights(self, models: Mapping[str, torch.nn.Module]) -> None:
        """Broadcast rank 0's parameters and buffers; ``fully_shard`` performs no such synchronization."""
        for module in models.values():
            for tensor in chain(module.parameters(), module.buffers()):
                torch.distributed.broadcast(tensor.data, src=0)

    def _share_state(self, state: dict[str, Any] | None) -> dict[str, Any]:
        return _shared_meta(state)

    def _require_pairing_or_warn(self, action: str) -> None:
        raise ValueError(
            f"{action.capitalize()} optimizer state under FSDP2 requires the learner to expose an "
            "optimizer_models pairing (optimizer name -> model names): sharded optimizer state can "
            "only be resolved through parameter FQNs."
        )


__all__ = [
    "DistributedDataParallelStrategy",
    "DistributedStrategy",
    "FullyShardedDataParallelStrategy",
    "SingleDeviceStrategy",
    "sync_gate",
]


if not TYPE_CHECKING:
    import sys

    from structcast.utils.lazy_import import LazySelectedImporter

    sys.modules[__name__] = LazySelectedImporter(__name__, globals())
