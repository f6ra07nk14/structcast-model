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
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from itertools import chain
from logging import getLogger
import os
import re
from typing import Any, Literal, overload

from structcast.utils.lazy_import import try_import
from timm.utils.distributed import init_distributed_device_so, is_distributed_env, world_info_from_env
from typing_extensions import Protocol, runtime_checkable

from structcast_model.torch.utils import get_torch_device, get_torch_device_type
import torch

logger = getLogger(__name__)

with try_import() as _fsdp_imports:  # torch >= 2.6 ships the stable per-parameter sharding (FSDP2) API.
    from torch.distributed.fsdp import FSDPModule, MixedPrecisionPolicy, fully_shard

with try_import() as _dcp_imports:  # torch >= 2.2; older builds admitted by the torch-cpu extra floor lack both.
    from torch.distributed.checkpoint import state_dict as _dcp_state_dict
    from torch.distributed.device_mesh import init_device_mesh

_DTYPES = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}

_WRAPPER_PREFIXES = ("module.", "_orig_mod.")


def _state_dict_api() -> Any:
    """Return the ``torch.distributed.checkpoint.state_dict`` module, or ``None`` on old torch."""
    return _dcp_state_dict if _dcp_imports.is_successful else None


@overload
def initial_distributed_env(
    device: str | None = None,
    dist_backend: str | None = None,
    dist_url: str | None = None,
    *,
    return_dict: Literal[True] = True,
) -> dict[str, Any]: ...


@overload
def initial_distributed_env(
    device: str | None = None,
    dist_backend: str | None = None,
    dist_url: str | None = None,
    *,
    return_dict: Literal[False] = False,
) -> tuple[str, int, int, int, bool]: ...


def initial_distributed_env(
    device: str | None = None,
    dist_backend: str | None = None,
    dist_url: str | None = None,
    *,
    return_dict: bool = True,
) -> dict[str, Any] | tuple[str, int, int, int, bool]:
    """Initialize the distributed environment.

    One process group serves every strategy: DDP all-reduces over it, and
    :class:`FullyShardedDataParallelStrategy` derives its device mesh from it at wrap time, so
    there is no FSDP2-specific environment setup.

    Args:
        device (str | None): The device to run the model on, e.g., 'cuda' or 'cpu'.
        dist_backend (str | None): The backend to use for distributed training.
            If None, the backend will be automatically selected based on the device.
        dist_url (str | None): The URL to use for distributed training initialization.
            If None, the URL will be automatically generated based on the environment.
        return_dict (bool): Whether to return the result as a dictionary.

    Returns:
        If return_dict is False, returns a tuple of (device, global_rank, local_rank, world_size, distributed).
        If return_dict is True, returns a dictionary with device, global_rank, local_rank, world_size, distributed keys.
    """
    if is_distributed_env() and torch.distributed.is_initialized():
        if "SLURM_PROCID" in os.environ:
            local_rank, global_rank, world_size = world_info_from_env()
        else:
            local_rank, _, _ = world_info_from_env()
            world_size = torch.distributed.get_world_size()
            global_rank = torch.distributed.get_rank()
        device_type = get_torch_device_type(device)
        result = {
            "device": f"{device_type}:{local_rank}" if device_type != "cpu" else "cpu",
            "global_rank": global_rank,
            "local_rank": local_rank,
            "world_size": world_size,
            "distributed": True,
        }
    else:
        device = get_torch_device(device)
        result = init_distributed_device_so(device=device, dist_backend=dist_backend, dist_url=dist_url)
    if return_dict:
        return result
    return result["device"], result["global_rank"], result["local_rank"], result["world_size"], result["distributed"]


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


def sync_gate(module: Any, armed: bool) -> None:
    """Arm or disarm *module*'s next gradient synchronization; a no-op for plain modules.

    Generated training steps call this immediately before every model invocation. ``armed`` is
    computed at code-generation time as "the model is owned by the current optimizer segment AND
    this is its last invocation in the segment", multiplied at runtime by the update-step flag, so
    gradient all-reduce (DDP) or reduce-scatter (FSDP2) fires exactly once per update, on the
    final backward.

    The flag is deliberately left in place rather than scoped: DDP reads it when the forward
    prepares the reducer, but FSDP2 reads it at backward time, after the gated invocation — any
    restore in between would re-enable reduce-scatter before the backward ran. The next gate on
    the same module overwrites the flag, so nothing ever needs restoring. On plain modules both
    branches are false, keeping single-device flow functions fully traceable.
    """
    if isinstance(module, torch.nn.parallel.DistributedDataParallel):
        module.require_backward_grad_sync = armed
    elif _fsdp_imports.is_successful and isinstance(module, FSDPModule):
        module.set_requires_gradient_sync(armed)


@runtime_checkable
class DistributedStrategy(Protocol):
    """Protocol for the strategy owning the distributed lifecycle of a training run.

    ``wrap`` must run after shape resolution and initializers but before the learner is
    constructed, because generated learner step closures and optimizers capture the exact module
    objects handed to ``__init__``. ``sync_initial_weights`` must run before ``wrap`` so all
    implementations broadcast plain tensors.
    """

    @property
    def grad_scaler_creator(self) -> Callable[..., Any]:
        """Callable creating gradient scalers for fp16 learners built under this strategy."""
        ...

    def wrap(self, models: "OrderedDict[str, torch.nn.Module]") -> "OrderedDict[str, torch.nn.Module]":
        """Wrap the models for this strategy and return the wrapped mapping."""

    def compile(self, module: Any, compile_kw: Mapping[str, Any] | None) -> Any:
        """Compile *module* where this strategy wants its compile units, and return what to use.

        ``compile_kw`` of ``None`` returns *module* unchanged. Modules are compiled in place, so the
        returned object is the one handed in; plain callables (the generated flow functions) have no
        in-place form and come back wrapped.
        """

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


def _innermost_module(module: torch.nn.Module) -> torch.nn.Module:
    """Peel DDP and torch.compile wrappers off *module*, in whatever order they were applied."""
    while True:
        if isinstance(module, torch.nn.parallel.DistributedDataParallel):
            module = module.module
        elif isinstance(inner := getattr(module, "_orig_mod", None), torch.nn.Module):
            module = inner
        else:
            return module


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
            if not optimizer_models and optimizers:
                self._require_pairing_or_warn("saving")
            optimizer_states: dict[str, Any] = {}
            for name, optimizer in optimizers.items():
                if optimizer_models and self._dcp_handles(optimizer, "saving"):
                    optimizer_states[name] = api.get_optimizer_state_dict(
                        _optimizer_container(models, optimizer_models[name]), optimizer, options=options
                    )
                else:
                    optimizer_states[name] = optimizer.state_dict()
            states["optimizers"] = optimizer_states
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
            # The old-torch fallback saved wrapper-free keys, so it must load into the innermost
            # module of whatever wrappers the CLI applied.
            for name, module in models.items():
                _innermost_module(module).load_state_dict(model_states[name])
            for name, optimizer in optimizers.items():
                optimizer.load_state_dict(optimizer_states[name])
            return state
        for name, module in models.items():
            api.set_model_state_dict(module, model_states.get(name, {}), options=self._load_options(api))
        if not optimizer_models and optimizers:
            self._require_pairing_or_warn("loading")
        for name, optimizer in optimizers.items():
            if optimizer_models and self._dcp_handles(optimizer, "loading"):
                self._set_optimizer_state(
                    api,
                    _optimizer_container(models, optimizer_models[name]),
                    optimizer,
                    optimizer_states.get(name, {}),
                )
            else:
                optimizer.load_state_dict(optimizer_states[name])
        return state

    def _dcp_handles(self, optimizer: Any, action: str) -> bool:
        """Whether DCP can key this optimizer's state by parameter FQNs.

        Optimizer proxies (e.g. the example ``AdamWWithCosine``) are not ``torch.optim.Optimizer``
        instances and merge scheduler state into their own ``state_dict``; DCP would reject the
        object and drop that merged state, so they save and load through their own state dicts.
        """
        return isinstance(optimizer, torch.optim.Optimizer)

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


class _CompileMixin:
    """Default compilation placement: the model root itself is the compile unit."""

    def compile(self, module: Any, compile_kw: Mapping[str, Any] | None) -> Any:
        """Compile in place when *module* is an ``nn.Module``, else wrap with ``torch.compile``.

        In-place compilation (``nn.Module.compile``) keeps the object identity: no ``OptimizedModule``
        wrapper shifts ``named_modules()`` paths or prefixes checkpoint keys with ``_orig_mod.``. Plain
        callables (the generated flow functions) have no in-place form, so they keep the wrapper.
        """
        if compile_kw is None:
            return module
        if isinstance(module, torch.nn.Module):
            module.compile(**compile_kw)
            return module
        return torch.compile(module, **compile_kw)


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
class SingleDeviceStrategy(_CompileMixin, _StateDictMixin):
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


class _MultiRankMixin:
    """Behavior shared by the strategies that train across ranks.

    ``initial_distributed_env`` is exposed on the classes because initializing the process group is
    the first step of every run trained under them (the CLI calls it before the strategy is
    instantiated, since the strategy's constructor arguments come from its result).
    """

    initial_distributed_env = staticmethod(initial_distributed_env)

    _broadcast_on_load = True

    def sync_initial_weights(self, models: Mapping[str, torch.nn.Module]) -> None:
        """Broadcast rank 0's parameters and buffers, making rank-0-only initializers authoritative.

        The broadcast runs on plain pre-wrap tensors, so one implementation serves DDP (whose
        constructor's own broadcast is a side effect nobody owns) and FSDP2 (which performs none).
        """
        for module in models.values():
            for tensor in chain(module.parameters(), module.buffers()):
                torch.distributed.broadcast(tensor.data, src=0)

    def _share_state(self, state: dict[str, Any] | None) -> dict[str, Any]:
        return _shared_meta(state)


@dataclass(kw_only=True)
class DistributedDataParallelStrategy(_MultiRankMixin, _CompileMixin, _StateDictMixin):
    """Strategy wrapping every model in ``DistributedDataParallel``."""

    device: str
    """Device this rank trains on."""

    local_rank: int = 0
    """Local rank used as the single CUDA device id; ignored on CPU."""

    grad_scaler_creator: Callable[..., Any] = torch.amp.GradScaler

    def wrap(self, models: "OrderedDict[str, torch.nn.Module]") -> "OrderedDict[str, torch.nn.Module]":
        """Wrap every model in DDP. ``device_ids`` must be ``None`` on CPU, where passing one raises."""
        device_ids = None if "cpu" in self.device else [self.local_rank]
        return OrderedDict(
            (n, torch.nn.parallel.DistributedDataParallel(m, device_ids=device_ids)) for n, m in models.items()
        )


def _compile_shard_pattern(pattern: str) -> "re.Pattern[str]":
    """Compile a shard_modules glob into a regex whose ``*``/``?`` stay within one path segment."""
    parts = []
    for char in pattern:
        if char == "*":
            parts.append("[^.]*")
        elif char == "?":
            parts.append("[^.]")
        else:
            parts.append(re.escape(char))
    return re.compile("".join(parts) + r"\Z")


def _matched_in_model(
    model: torch.nn.Module,
    patterns: Sequence[str],
) -> tuple[list[tuple[str, torch.nn.Module]], set[str]]:
    """Return *model*'s matching ``named_modules()`` entries and the patterns that hit at least one.

    Never raises on a pattern matching nothing: one model matching none of them is normal (a CycleGAN
    discriminator has no blocks). Only :func:`matched_shard_modules`, which sees every model, can tell
    a legitimate miss from a typo.
    """
    compiled = {p: _compile_shard_pattern(p) for p in patterns}
    entries: list[tuple[str, torch.nn.Module]] = []
    hits: set[str] = set()
    for path, submodule in model.named_modules():
        stripped = path.removeprefix("_orig_mod.")
        # The root (or its compile wrapper's inner module) is never a match: wrap shards it
        # last unconditionally, and a catch-all pattern must not shard it twice.
        if not stripped or stripped == "_orig_mod":
            continue
        matching = {p for p, rx in compiled.items() if rx.match(stripped)}
        if matching:
            entries.append((path, submodule))
            hits |= matching
    return entries, hits


def matched_shard_modules(
    models: Mapping[str, torch.nn.Module],
    patterns: Sequence[str],
) -> "OrderedDict[str, list[tuple[str, torch.nn.Module]]]":
    """Return, per model, the ``named_modules()`` entries whose path matches one of *patterns*.

    Patterns are globs whose ``*`` and ``?`` never cross a ``.``: ``backbone.block*`` matches
    ``backbone.block0`` but not its children (``backbone.block0.layer_norm``, ...). ``fnmatch``
    semantics, where ``*`` crosses ``.``, would silently match a block's whole subtree and shard
    every leaf as its own communication group — the opposite of one all-gather per block.

    Entries keep ``named_modules()`` (pre-order DFS) order, which the per-block wrap reverses to
    shard descendants before ancestors. A leading ``_orig_mod.`` is stripped before matching so the
    same patterns keep working on a root already wrapped by ``torch.compile``.

    Raises:
        ValueError: if a pattern matches no module in any of the models. A pattern matching nothing
            anywhere is a typo; matching nothing in *one* model is normal for multi-model learners
            (a CycleGAN discriminator has no blocks).
    """
    matched: OrderedDict[str, list[tuple[str, torch.nn.Module]]] = OrderedDict()
    unmatched = set(patterns)
    for name, model in models.items():
        matched[name], hits = _matched_in_model(model, patterns)
        unmatched -= hits
    if unmatched:
        available = [p for m in models.values() for p, _ in m.named_modules() if p][:10]
        raise ValueError(
            f"shard_modules pattern(s) {sorted(unmatched)} matched no module; "
            f"available module paths include {available}."
        )
    return matched


def _check_tied_parameters(model: torch.nn.Module, paths: Sequence[str]) -> None:
    """Refuse to shard *model* when a tied parameter would land in two ``fully_shard`` groups.

    Each parameter belongs to the innermost matched module containing it, or to the root group when
    no matched path contains it. ``fully_shard`` replaces a group's parameters with its own sharded
    copies and checks nothing, so a tie split across two groups silently becomes two parameters that
    drift apart. A tie staying inside one group is untouched and fine.
    """
    groups: dict[int, tuple[str, str]] = {}
    for occurrence, parameter in model.named_parameters(remove_duplicate=False):
        group = max((p for p in paths if occurrence.startswith(f"{p}.")), key=len, default="")
        owner, first = groups.setdefault(id(parameter), (group, occurrence))
        if owner != group:
            raise RuntimeError(
                f"Tied parameter {first!r} and {occurrence!r} would be sharded into different "
                f"fully_shard groups ({owner or '<root>'} and {group or '<root>'}); "
                "shard_modules must keep tied parameters inside one group."
            )


@dataclass(kw_only=True)
class FullyShardedDataParallelStrategy(_MultiRankMixin, _CompileMixin, _StateDictMixin):
    """Strategy sharding every model in place with ``fully_shard`` (FSDP2).

    Each model is one communication group by default; ``shard_modules`` splits it into one group per
    matched submodule, so a block's parameters are all-gathered right before it runs and freed right
    after instead of the whole model being resident for the whole forward.

    Requires torch >= 2.6. fp16 learners built under this strategy use the plain
    ``torch.amp.GradScaler``: since torch 2.5 the DTensor dispatcher all-reduces ``found_inf``
    across ranks inside ``unscale_``, so no sharded scaler class is needed.

    Environment initialization is identical to DDP's — one default process group serves both;
    the FSDP2-specific device mesh is derived from it lazily at wrap time.
    """

    device: str
    """Device this rank trains on."""

    local_rank: int = 0
    """Local rank; the device mesh is derived from the default process group."""

    reshard_after_forward: bool = True
    """Whether to reshard parameters after forward, trading memory for an extra all-gather."""

    mp_policy: dict[str, str] | None = None
    """Mixed precision policy dtypes by ``MixedPrecisionPolicy`` field name, e.g. ``{"param_dtype": "bfloat16"}``."""

    shard_modules: list[str] | None = None
    """Glob patterns over ``named_modules()`` paths to shard as their own communication groups,
    e.g. ``["backbone.block*"]``; ``*`` and ``?`` never cross a ``.``. ``None`` shards each model
    as a single group."""

    grad_scaler_creator: Callable[..., Any] = torch.amp.GradScaler

    _broadcast_on_load = True
    _mesh: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        """Fail loud when the installed torch has no stable ``fully_shard``."""
        if not _fsdp_imports.is_successful:
            raise ImportError(
                "FullyShardedDataParallelStrategy requires torch>=2.6 for the stable fully_shard "
                "(FSDP2) API; the installed torch does not provide torch.distributed.fsdp.fully_shard."
            )

    def wrap(self, models: "OrderedDict[str, torch.nn.Module]") -> "OrderedDict[str, torch.nn.Module]":
        """Shard every model in place with ``fully_shard`` and return the same (now sharded) modules.

        With ``shard_modules`` set, the matched submodules are sharded first and the root last: the
        root's managed-module walk stops at children that are already ``FSDPModule``s, so the root
        group ends up holding exactly the parameters no matched submodule claimed.
        """
        if self._mesh is None:
            # Without an explicit mesh, fully_shard follows the accelerator, which reports CUDA on
            # CUDA-enabled builds even when this strategy trains on CPU — the mesh must follow the
            # strategy's device instead.
            device_type = "cpu" if "cpu" in self.device else self.device.split(":")[0]
            self._mesh = init_device_mesh(device_type, (torch.distributed.get_world_size(),))
        kwargs: dict[str, Any] = {"reshard_after_forward": self.reshard_after_forward, "mesh": self._mesh}
        if self.mp_policy:
            # Any-valued because a dict[str, dtype] unpacked as **kwargs is checked against every
            # MixedPrecisionPolicy field, including the bool cast_forward_inputs no dtype ever fills.
            dtypes: dict[str, Any] = {k: _DTYPES[v] for k, v in self.mp_policy.items()}
            kwargs["mp_policy"] = MixedPrecisionPolicy(**dtypes)
        if self.shard_modules:
            matched = matched_shard_modules(models, self.shard_modules)
            # Every model is validated before any is sharded: a tie violation surfacing halfway
            # would leave the earlier models already irrecoverably sharded.
            for name, model in models.items():
                _check_tied_parameters(model, [path for path, _ in matched[name]])
            for name in models:
                # Reversed pre-order shards descendants before ancestors; the other order makes an
                # ancestor claim its whole subtree and re-sharding the descendant then throws.
                for _, submodule in reversed(matched[name]):
                    fully_shard(submodule, **kwargs)
        # fully_shard shards in place and hands back the very module it was given; its declared
        # FSDPModule return type is the runtime-injected mixin, which is not statically an nn.Module.
        for model in models.values():
            fully_shard(model, **kwargs)
        return OrderedDict(models)

    def compile(self, module: Any, compile_kw: Mapping[str, Any] | None) -> Any:
        """Compile the sharded submodules in place, so compile units follow the shard boundaries.

        Compiling the root instead would bury the per-block all-gather/reduce-scatter hooks inside one
        graph (ADR-0004). A module none of the patterns match keeps the default root compile: matching
        nothing in one model is normal for a multi-model learner, unlike matching nothing anywhere,
        which :func:`matched_shard_modules` rejects at wrap time.
        """
        if compile_kw is None or not self.shard_modules or not isinstance(module, torch.nn.Module):
            return super().compile(module, compile_kw)
        matched, _ = _matched_in_model(module, self.shard_modules)
        if not matched:
            return super().compile(module, compile_kw)
        for _, submodule in matched:
            submodule.compile(**compile_kw)
        return module

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

    def _dcp_handles(self, optimizer: Any, action: str) -> bool:
        if not isinstance(optimizer, torch.optim.Optimizer):
            raise ValueError(
                f"{action.capitalize()} optimizer state under FSDP2 requires torch.optim.Optimizer "
                f"instances, got {type(optimizer).__name__}: an optimizer proxy's own state dict "
                "would hold unsharded DTensor fragments that no single artifact can represent."
            )
        return True


__all__ = [
    "DistributedDataParallelStrategy",
    "DistributedStrategy",
    "FullyShardedDataParallelStrategy",
    "SingleDeviceStrategy",
    "initial_distributed_env",
    "matched_shard_modules",
    "sync_gate",
]


# Unlike the package's other modules, this one is NOT replaced by LazySelectedImporter: generated
# flow functions call sync_gate inside torch.compile'd regions, and dynamo introspects the
# function's module through sys.modules — the shim raises on dunders (`__class__`) and breaks
# tracing (InternalTorchDynamoError). A plain module traces cleanly.
