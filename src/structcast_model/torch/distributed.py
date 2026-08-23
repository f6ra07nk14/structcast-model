"""Distributed strategies: how models are wrapped, synchronized, and turned into checkpointable state.

A distributed strategy is the replaceable unit deciding how models are wrapped for a training run,
how their initial weights are made identical across ranks, whether their ``BatchNorm`` layers are
converted to ``SyncBatchNorm``, and how model/optimizer state becomes a loadable checkpoint. Exactly
one strategy is active per run; single-device training uses :class:`SingleDeviceStrategy` rather than
a special case.

Every ``state_dict``/``load_state_dict`` implementation routes through
``torch.distributed.checkpoint.state_dict`` so checkpoint keys are identical for raw,
``torch.compile``'d, DDP-wrapped, and ``fully_shard``'d models.
"""

from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from itertools import chain
from logging import getLogger
import os
import re
from typing import Any, Literal, overload

from structcast.utils.lazy_import import try_import
from timm.layers import convert_sync_batchnorm
from timm.utils.distributed import init_distributed_device_so, is_distributed_env, world_info_from_env
from torch.nn.modules.batchnorm import _BatchNorm
from typing_extensions import Protocol, runtime_checkable

from structcast_model.torch.utils import get_torch_device, get_torch_device_type
import torch

logger = getLogger(__name__)

with try_import() as _fsdp_imports:  # torch >= 2.6 ships the stable per-parameter sharding (FSDP2) API.
    from torch.distributed.fsdp import FSDPModule, MixedPrecisionPolicy, fully_shard

with try_import() as _dcp_imports:  # torch >= 2.2; older builds admitted by the torch-cpu extra floor lack both.
    from torch.distributed.checkpoint import state_dict as _dcp_state_dict
    from torch.distributed.device_mesh import init_device_mesh

with try_import() as _tp_imports:  # torch >= 2.4 ships the DTensor tensor-parallel styles at this path.
    from torch.distributed.tensor.parallel import (
        ColwiseParallel,
        RowwiseParallel,
        SequenceParallel,
        parallelize_module,
    )

_DTYPES = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}

PARALLEL_STYLES = ("column", "row", "sequence", "column_heads")
"""The tensor-parallel styles a ``parallel_modules`` entry may name; any other value is used as the
``ParallelStyle`` instance itself, which is the escape hatch for the styles this vocabulary lacks."""


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


def split_mixed_param_groups(optimizer: Any) -> None:
    """Rewrite *optimizer*'s parameter groups so none of them mixes ``DTensor``s with plain tensors.

    Tensor parallelism converts only the parameters its ``parallel_modules`` globs name into
    ``DTensor``s; everything else -- a vision transformer's patch embedding, class token, layer norms
    and head -- stays a plain tensor, and a learner puts all of them into one parameter group. torch's
    default multi-tensor path then hands that group to a single ``_foreach_*`` call, whose dispatcher
    refuses a list holding both kinds, so a stock tensor-parallel run dies on its first
    ``optimizer.step()`` with ``aten._foreach_lerp_.Scalar got mixed torch.Tensor and DTensor``.

    A mixed group becomes one subgroup per kind, carrying the same hyperparameters and the same
    parameters in the same order. A group that is already uniform is left as the very object it was,
    which makes this an identity for every run with no ``DTensor`` in it -- which is why it runs for
    every optimizer rather than under the tensor-parallel strategies alone: a hand-written strategy,
    or a wrapper added later, produces the same mixture and needs the same protection.

    Splitting is the fix that changes no arithmetic: an optimizer's update is independent per
    parameter, and the grouping decides only which of them are fused into one kernel call. Passing
    ``foreach=False`` would clear the crash too, but it gives up the fused path for the ``DTensor``
    majority as well and measurably moves the numerics; making every non-parallelized parameter a
    replicated ``DTensor`` instead breaks the forward pass, where those modules would then run
    ``DTensor`` parameters against plain activations.

    Must run before anything reads the group list back: a checkpoint written from a split optimizer
    has to load into an identically split one, and an LR scheduler snapshots one base rate per group
    when it is constructed.

    Args:
        optimizer (Any): The optimizer whose ``param_groups`` are rewritten in place. Duck-typed,
            because an optimizer proxy delegates ``param_groups`` to the optimizer it wraps, and that
            is the list that has to change.
    """
    groups: list[dict[str, Any]] = []
    for group in optimizer.param_groups:
        kinds: dict[bool, list[Any]] = {}
        for parameter in group["params"]:
            # Plain by exact type rather than ``isinstance(parameter, DTensor)``: the public DTensor
            # path only exists from torch 2.5, while the tensor-parallel API that produces the
            # mixture ships in 2.4 -- and a plain parameter is always exactly a Tensor or a Parameter.
            plain = type(parameter) in (torch.Tensor, torch.nn.Parameter)
            kinds.setdefault(plain, []).append(parameter)
        if len(kinds) < 2:
            groups.append(group)
        else:
            groups.extend({**group, "params": params} for params in kinds.values())
    if len(groups) == len(optimizer.param_groups):
        return
    logger.info(
        "Splitting %d optimizer parameter group(s) into %d so that none of them mixes DTensor with "
        "plain parameters, which the multi-tensor optimizer path cannot fuse.",
        len(optimizer.param_groups),
        len(groups),
    )
    optimizer.param_groups[:] = groups


@runtime_checkable
class DistributedStrategy(Protocol):
    """Protocol for the strategy owning the distributed lifecycle of a training run.

    ``wrap`` must run after shape resolution and initializers but before the learner is
    constructed, because generated learner step closures and optimizers capture the exact module
    objects handed to ``__init__``. ``sync_initial_weights`` must run before ``wrap`` so all
    implementations broadcast plain tensors.

    Checkable with ``isinstance`` only: ``data_rank`` and ``data_world_size`` are non-method members,
    and ``issubclass`` against a protocol that has any of those raises ``TypeError`` by design --
    attribute-style annotations are counted the same way, so there is no spelling that keeps it.
    """

    @property
    def data_rank(self) -> int:
        """Which slice of the dataset this rank must consume, and what its seed is derived from.

        The global rank wherever every rank holds its own replica (DDP, FSDP2), but 0 on every rank
        of a tensor-parallel group: those ranks split one model and must run the identical batch
        through it under the identical dropout mask (``docs/adr/0022``). A run reading the global
        rank instead would feed a tensor-parallel group as many different batches as it has ranks,
        and every shard would draw its own mask -- both silently wrong rather than failing.
        """

    @property
    def data_world_size(self) -> int:
        """How many distinct dataset slices the run is split into: the data axis of the mesh."""

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
    """Peel DDP and torch.compile wrappers off *module*, in whatever order they were applied.

    By type, never by attribute name: an `AveragedModel` keeps the module it averages under `.module`
    too, and peeling that one would save a fragment of the average and load it back into nothing.
    """
    while True:
        if isinstance(module, torch.nn.parallel.DistributedDataParallel):
            module = module.module
        elif isinstance(inner := getattr(module, "_orig_mod", None), torch.nn.Module):
            module = inner
        else:
            return module


@dataclass(kw_only=True)
class _StateDictMixin:
    """Shared DCP-based state production and loading."""

    _broadcast_on_load = False

    strict_optimizer_load: bool = True
    """Whether a resumed optimizer state must cover every trainable parameter. ``True`` keeps torch's
    rejection of a gap (descriptive on torch >= 2.10, a bare ``KeyError`` before); ``False`` accepts
    any coverage — even none — leaving unmatched parameters the zeroed state torch materializes.
    Missing state is never synthesized."""

    _api: Any = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Resolve the ``torch.distributed.checkpoint.state_dict`` module, or ``None`` on old torch.

        ``None`` selects the wrapper-free fallback in :meth:`state_dict` and :meth:`load_state_dict`.
        Every strategy defining its own ``__post_init__`` must chain into this one, or it inherits
        the field without ever getting a value.
        """
        # `object` has no `__post_init__`, so the chain up ends here rather than at a bare super() call.
        post = getattr(super(), "__post_init__", None)
        if post is not None:
            post()
        self._api = _dcp_state_dict if _dcp_imports.is_successful else None

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
        api = self._api
        if api is None:
            # Saved from the module the wrappers hold, which is exactly what the fallback load path
            # writes back into: stripping the prefixes off the wrapper's own keys instead would also
            # strip a `module.` a model owns itself.
            states: dict[str, Any] = {"models": {n: _innermost_module(m).state_dict() for n, m in models.items()}}
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
        # Whether this rank was handed the state: the model tensors stay rank-0-only and reach the
        # others through `broadcast_from_rank0`, so only the rank holding them can say what is missing.
        holds_state = state is not None
        state = self._share_state(state)
        api = self._api
        model_states = state.get("models", {})
        optimizer_states = state.get("optimizers", {})
        # Checked before anything is written: torch reports a model it was handed an empty state for
        # as a process-group failure, and a wrapped one accepts it silently and keeps its
        # construction weights. A state holding models the learner no longer has is simply ignored.
        if holds_state:
            for name in models:
                if name not in model_states:
                    raise ValueError(
                        f'The saved training state carries no state for model "{name}": it was written '
                        "before that model was declared -- an EMA shadow added to the learner since, most "
                        "likely. Resume with the learner the checkpoint was saved from, or start a fresh run."
                    )
        if api is None:
            # The old-torch fallback saved wrapper-free keys, so it must load into the innermost
            # module of whatever wrappers the CLI applied.
            for name, module in models.items():
                _innermost_module(module).load_state_dict(model_states[name])
            for name, optimizer in optimizers.items():
                optimizer.load_state_dict(optimizer_states[name])
            return state
        for name, module in models.items():
            # `.get`, because the ranks the tensors are broadcast to hold none of them: what the
            # state must carry was checked above, on the rank that was handed it.
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

        State keyed by parameter index comes from a save that had no pairing; this path matches
        entries by parameter name and cannot resolve positions, so such a state is refused instead
        of being silently discarded or half-applied (ADR-0008). Name-keyed state loads under
        ``strict_optimizer_load``, which decides whether it must cover every trainable parameter.

        Raises:
            ValueError: if the saved optimizer state is keyed by parameter index.
        """
        state = osd.get("state", {})
        if state and all(isinstance(key, int) for key in state):
            raise ValueError(
                "The saved optimizer state is keyed by parameter index, not by parameter name: it was "
                "saved without an optimizer_models pairing, so its entries cannot be matched to the "
                "parameter names of this run. Resume from a training state saved with the pairing, "
                "or restart training."
            )
        if any(state.values()):
            options = api.StateDictOptions(full_state_dict=True, strict=self.strict_optimizer_load)
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


@dataclass(kw_only=True)
class SingleDeviceStrategy(_CompileMixin, _StateDictMixin):
    """Strategy for single-device training: no wrapping, no cross-rank synchronization."""

    device: str
    """Device the run trains on, e.g. ``"cuda:0"`` or ``"cpu"``."""

    local_rank: int = 0
    """Local rank; unused on a single device, accepted for a uniform constructor."""

    @property
    def data_rank(self) -> int:
        """The only slice there is."""
        return 0

    @property
    def data_world_size(self) -> int:
        """One device consumes the whole dataset."""
        return 1

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

    @property
    def data_rank(self) -> int:
        """The global rank: every rank holds a full replica and consumes its own slice."""
        return torch.distributed.get_rank() if torch.distributed.is_initialized() else 0

    @property
    def data_world_size(self) -> int:
        """The world size: one replica per rank. Outside a process group there is one of everything."""
        return torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1

    def sync_initial_weights(self, models: Mapping[str, torch.nn.Module]) -> None:
        """Broadcast rank 0's parameters and buffers, making rank-0-only initializers authoritative.

        The broadcast runs on plain pre-wrap tensors, so one implementation serves DDP (whose
        constructor's own broadcast is a side effect nobody owns) and FSDP2 (which performs none).
        """
        for module in models.values():
            for tensor in chain(module.parameters(), module.buffers()):
                torch.distributed.broadcast(tensor.data, src=0)

    def _share_state(self, state: dict[str, Any] | None) -> dict[str, Any]:
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


def _convert_module_sync_batchnorm(module: torch.nn.Module) -> torch.nn.Module:
    """Return *module* with every ``_BatchNorm`` layer that is not already synchronized converted.

    Walks the tree instead of handing the root to timm's converter, because that converter is not
    idempotent: it matches ``_BatchNorm``, and ``SyncBatchNormAct`` — timm's fused variant — is a
    ``torch.nn.SyncBatchNorm`` subclass but not a ``BatchNormAct2d``, so a second pass rebuilds it as a
    plain ``SyncBatchNorm`` and silently drops the activation, leaving the ``state_dict`` keys
    unchanged. A layer that already is a ``torch.nn.SyncBatchNorm`` is therefore left completely
    untouched — same object, same hooks, same ``process_group``.

    Every other ``_BatchNorm`` goes through timm's converter, so timm's fused norm-act layers survive:
    ``BatchNormAct2d`` becomes a ``SyncBatchNormAct`` that keeps running its activation, where torch's
    converter would replace it with a plain ``SyncBatchNorm`` and drop it. Other third-party
    ``_BatchNorm`` subclasses are still flattened to plain ``SyncBatchNorm``; the off-switch is their
    escape.
    """
    if isinstance(module, torch.nn.SyncBatchNorm):
        return module
    if isinstance(module, _BatchNorm):
        return convert_sync_batchnorm(module)
    for name, child in module.named_children():
        converted = _convert_module_sync_batchnorm(child)
        if converted is not child:
            module.add_module(name, converted)
    return module


def _convert_sync_batchnorm(
    models: "OrderedDict[str, torch.nn.Module]",
    device: str,
) -> "OrderedDict[str, torch.nn.Module]":
    """Return *models* with every ``_BatchNorm`` layer replaced by its ``SyncBatchNorm`` equivalent.

    Runs before any wrapping or sharding: the wrapper must see the final module tree, and per-block
    sharding matches paths on it. The conversion's return value is what callers must use — containers
    are converted in place, but a model that *is* a ``_BatchNorm`` comes back as a new module.
    Parameters and buffers are carried over by reference, so the rank-0 broadcast that ran before
    wrapping survives the conversion.

    The conversion is idempotent: layers that already are ``torch.nn.SyncBatchNorm`` instances, timm's
    fused ones included, are skipped along with their ``process_group``, so a pre-converted model comes
    back as it went in.

    Skipped on CPU devices, where ``SyncBatchNorm``'s training forward rejects the input outright
    once a process group is initialized, even with a single rank.
    """
    if "cpu" in device:
        return models
    return OrderedDict((n, _convert_module_sync_batchnorm(m)) for n, m in models.items())


@dataclass(kw_only=True)
class DistributedDataParallelStrategy(_MultiRankMixin, _CompileMixin, _StateDictMixin):
    """Strategy wrapping every model in ``DistributedDataParallel``."""

    device: str
    """Device this rank trains on."""

    local_rank: int = 0
    """Local rank used as the single CUDA device id; ignored on CPU."""

    sync_batchnorm: bool = True
    """Whether to convert ``BatchNorm`` layers to ``SyncBatchNorm`` before wrapping; a no-op on CPU."""

    def wrap(self, models: "OrderedDict[str, torch.nn.Module]") -> "OrderedDict[str, torch.nn.Module]":
        """Wrap every model in DDP. ``device_ids`` must be ``None`` on CPU, where passing one raises."""
        if self.sync_batchnorm:
            models = _convert_sync_batchnorm(models, self.device)
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
    *,
    option: str = "shard_modules",
) -> "OrderedDict[str, list[tuple[str, torch.nn.Module]]]":
    """Return, per model, the ``named_modules()`` entries whose path matches one of *patterns*.

    Patterns are globs whose ``*`` and ``?`` never cross a ``.``: ``backbone.block*`` matches
    ``backbone.block0`` but not its children (``backbone.block0.layer_norm``, ...). ``fnmatch``
    semantics, where ``*`` crosses ``.``, would silently match a block's whole subtree and shard
    every leaf as its own communication group — the opposite of one all-gather per block.

    Entries keep ``named_modules()`` (pre-order DFS) order, which the per-block wrap reverses to
    shard descendants before ancestors. A leading ``_orig_mod.`` is stripped before matching so the
    same patterns keep working on a root already wrapped by ``torch.compile``. *option* is the name
    of the strategy field the patterns came from, so the rejection below names the one to fix --
    ``parallel_modules`` compiles its globs through here too.

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
            f"{option} pattern(s) {sorted(unmatched)} matched no module; available module paths include {available}."
        )
    return matched


def _check_tied_parameters(model: torch.nn.Module, paths: Sequence[str], option: str = "shard_modules") -> None:
    """Refuse to shard *model* when a tied parameter would land in two sharding groups.

    Each parameter belongs to the innermost matched module containing it, or to the root group when
    no matched path contains it. ``fully_shard`` replaces a group's parameters with its own sharded
    copies and checks nothing, so a tie split across two groups silently becomes two parameters that
    drift apart; ``parallelize_module`` does the same with the styles' own DTensors, and can even
    give the two ends different placements. A tie staying inside one group is untouched and fine.
    """
    groups: dict[int, tuple[str, str]] = {}
    for occurrence, parameter in model.named_parameters(remove_duplicate=False):
        group = max((p for p in paths if occurrence.startswith(f"{p}.")), key=len, default="")
        owner, first = groups.setdefault(id(parameter), (group, occurrence))
        if owner != group:
            raise RuntimeError(
                f"Tied parameter {first!r} and {occurrence!r} would be sharded into different "
                f"groups ({owner or '<root>'} and {group or '<root>'}); "
                f"{option} must keep tied parameters inside one group."
            )


def _device_mesh(device: str, **dims: int) -> Any:
    """Build a named device mesh over the process group's ranks, on *device*'s own device type.

    Without an explicit mesh, ``fully_shard`` follows the accelerator, which reports CUDA on
    CUDA-enabled builds even when the strategy trains on CPU -- the mesh must follow the strategy's
    device instead. Every dimension is named, so a two-dimensional mesh hands out its submeshes by
    axis (``mesh["dp"]``, ``mesh["tp"]``) and the keyword order is the mesh order.
    """
    device_type = "cpu" if "cpu" in device else device.split(":", maxsplit=1)[0]
    return init_device_mesh(device_type, tuple(dims.values()), mesh_dim_names=tuple(dims))


def _parallel_style(style: Any) -> Any:
    """Return the ``ParallelStyle`` a plan entry names, or *style* itself when it is not a name.

    The vocabulary covers the four shapes a transformer needs; anything else -- ``PrepareModuleInput``,
    a custom style, a ``ColwiseParallel`` with hand-picked layouts -- is written as an object pattern
    in the strategy pattern, which the CLI instantiates before the strategy sees it.

    Raises:
        ValueError: if *style* is a string the vocabulary does not have.
    """
    if not isinstance(style, str):
        return style
    if style == "column":
        return ColwiseParallel()
    if style == "row":
        return RowwiseParallel()
    if style == "sequence":
        return SequenceParallel()
    if style == "column_heads":
        # The attention shape: the projection's output stays a DTensor, so the head reshape that
        # consumes it sees the sharded head count instead of the full one.
        return ColwiseParallel(use_local_output=False)
    raise ValueError(f"Unknown parallel style {style!r}. Available styles: {', '.join(PARALLEL_STYLES)}.")


def _parallelize_models(
    models: Mapping[str, torch.nn.Module],
    mesh: Any,
    parallel_modules: Sequence[tuple[str, Any]],
) -> None:
    """Apply *parallel_modules* to every model in place, over the one-dimensional *mesh*.

    The globs are the ``shard_modules`` ones -- ``*`` and ``?`` never cross a ``.`` -- and go through
    the same machinery: a pattern matching nothing anywhere is a typo and is refused, and every model
    is checked for a tie split across two matched modules before any of them is parallelized. The
    first entry matching a path decides its style, as the Flax and Keras rule tables do.
    """
    matched = matched_shard_modules(models, [pattern for pattern, _ in parallel_modules], option="parallel_modules")
    for name, model in models.items():
        _check_tied_parameters(model, [path for path, _ in matched[name]], "parallel_modules")
    for model in models.values():
        plan: dict[str, Any] = {}
        for pattern, style in parallel_modules:
            instance = _parallel_style(style)
            for path, _ in _matched_in_model(model, [pattern])[0]:
                plan.setdefault(path, instance)
        if plan:
            parallelize_module(model, mesh, plan)


@dataclass(kw_only=True)
class TensorParallelStrategy(_MultiRankMixin, _CompileMixin, _StateDictMixin):
    """Strategy splitting the matched layers of every model across all ranks (tensor parallelism).

    One tensor-parallel group spans the whole world: each rank holds a shard of every matched layer
    and runs the *same* batch through it, so the run has a single data slice
    (:attr:`data_rank` 0 of :attr:`data_world_size` 1) and its loader must hand every rank identical
    items -- a ``DistributedSampler`` keyed on the global rank would give each shard a different
    batch and quietly train nonsense.

    Nothing is gated: DTensor inserts each layer's collective inside the operation that needs it, so
    there is no deferred bucket for :func:`sync_gate` to arm, and a parallelized module is a plain
    ``nn.Module`` neither of its branches fires on. ``BatchNorm`` is likewise left alone -- every rank
    already computes the statistics of the same batch, so there is nothing to synchronize.

    Requires torch >= 2.4. ``loss_parallel`` is out of scope (``docs/adr/0022``): it constrains the
    loss to cross-entropy with a mean reduction, which the generated learners do not promise.
    """

    device: str
    """Device this rank trains on."""

    local_rank: int = 0
    """Local rank; the mesh is derived from the default process group."""

    parallel_modules: Sequence[tuple[str, Any]] = ()
    """Ordered (glob over ``named_modules()`` paths, style) pairs, e.g. ``[("*.wq", "column_heads")]``.
    A style is one of :data:`PARALLEL_STYLES` or a ``ParallelStyle`` instance written as an object
    pattern; the first entry matching a path wins."""

    _mesh: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        """Fail loud when the installed torch has no tensor-parallel API, or the plan is empty.

        Raises:
            ImportError: if ``torch.distributed.tensor.parallel`` is missing.
            ValueError: if no ``parallel_modules`` entry was given, which would parallelize nothing.
        """
        super().__post_init__()
        if not _tp_imports.is_successful:
            raise ImportError(
                "TensorParallelStrategy requires torch>=2.4 for the DTensor tensor-parallel API; the "
                "installed torch does not provide torch.distributed.tensor.parallel.parallelize_module."
            )
        if not self.parallel_modules:
            raise ValueError(
                "TensorParallelStrategy needs a parallel_modules plan naming which submodules to split "
                "and how; without one it would run every rank on the whole model and the whole batch."
            )

    @property
    def data_rank(self) -> int:
        """0 on every rank: the ranks of a tensor-parallel group consume one and the same slice."""
        return 0

    @property
    def data_world_size(self) -> int:
        """One slice: the whole world is a single tensor-parallel group."""
        return 1

    def wrap(self, models: "OrderedDict[str, torch.nn.Module]") -> "OrderedDict[str, torch.nn.Module]":
        """Parallelize every model's matched submodules in place and return the same modules."""
        if self._mesh is None:
            self._mesh = _device_mesh(self.device, tp=torch.distributed.get_world_size())
        _parallelize_models(models, self._mesh, self.parallel_modules)
        return models


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

    sync_batchnorm: bool = True
    """Whether to convert ``BatchNorm`` layers to ``SyncBatchNorm`` before sharding; a no-op on CPU."""

    _broadcast_on_load = True
    _mesh: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        """Fail loud when the installed torch has no stable ``fully_shard``."""
        super().__post_init__()
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

        The ``SyncBatchNorm`` conversion runs before everything else, so the patterns match — and
        ``fully_shard`` shards — the converted tree rather than modules about to be replaced.
        """
        if self.sync_batchnorm:
            models = _convert_sync_batchnorm(models, self.device)
        if self._mesh is None:
            self._mesh = _device_mesh(self.device, dp=torch.distributed.get_world_size())
        return self._shard(models)

    def _shard(self, models: "OrderedDict[str, torch.nn.Module]") -> "OrderedDict[str, torch.nn.Module]":
        """Shard *models* over ``self._mesh``: the matched submodules first, then every root.

        Split out of :meth:`wrap` so the combination with tensor parallelism can put its own two
        steps -- a two-dimensional mesh and ``parallelize_module`` -- in front of exactly this.
        """
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


@dataclass(kw_only=True)
class FullyShardedTensorParallelStrategy(FullyShardedDataParallelStrategy):
    """Strategy combining FSDP2 and tensor parallelism on a two-dimensional ``(dp, tp)`` mesh.

    The ranks are laid out row-major, so rank ``r`` is tensor-parallel shard
    ``r % tensor_parallel_size`` of data replica ``r // tensor_parallel_size``: the ranks of one
    tensor-parallel group are adjacent and share :attr:`data_rank`, which is what the loader and the
    seed derivation must follow.

    ``parallelize_module`` runs first and ``fully_shard`` wraps its result on the data submesh --
    torchtitan's order. Reversing it hands the tensor-parallel styles the ``fully_shard`` parameters
    instead of the module's own, and the sharded model is then not the one the plan describes.
    :func:`sync_gate` keeps working unchanged: ``fully_shard`` still runs on the model root, so the
    root is an ``FSDPModule`` whose reduce-scatter -- over the data axis alone -- the gate arms.
    """

    tensor_parallel_size: int = 1
    """How many ranks one tensor-parallel group spans; the world size divided by it is the data axis."""

    parallel_modules: Sequence[tuple[str, Any]] = ()
    """The tensor-parallel plan, in :class:`TensorParallelStrategy`'s form."""

    _tp_mesh: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        """Fail loud when FSDP2 or the tensor-parallel API is missing, or the plan or degree is unusable.

        Raises:
            ImportError: if either API is unavailable in the installed torch.
            ValueError: if no ``parallel_modules`` entry was given, or the degree is not a usable one.
        """
        super().__post_init__()
        if not _tp_imports.is_successful:
            raise ImportError(
                "FullyShardedTensorParallelStrategy requires torch>=2.4 for the DTensor tensor-parallel "
                "API; the installed torch does not provide "
                "torch.distributed.tensor.parallel.parallelize_module."
            )
        if not self.parallel_modules:
            raise ValueError(
                "FullyShardedTensorParallelStrategy needs a parallel_modules plan naming which submodules "
                "to split and how; without one, select FullyShardedDataParallelStrategy instead."
            )
        self._check_degree()

    def _check_degree(self) -> None:
        """Refuse a tensor-parallel degree no ``(dp, tp)`` mesh can express.

        Runs at construction and again at wrap. Construction is where it matters: the CLI builds the
        strategy inside an already-initialized process group and reads :attr:`data_rank` immediately
        after, which divides by the degree -- a zero would surface as a bare ``ZeroDivisionError``
        from the seeding line and a negative one would publish a negative world size. A strategy
        built before the group can only be checked against a world size at wrap.

        Raises:
            ValueError: if the degree is below 1, or does not divide the world size.
        """
        if self.tensor_parallel_size < 1:
            raise ValueError(
                f"tensor_parallel_size must be at least 1, got {self.tensor_parallel_size}: it is how many "
                "ranks one tensor-parallel group spans, and the world size divided by it is the data axis."
            )
        if not torch.distributed.is_initialized():
            return
        world_size = torch.distributed.get_world_size()
        if world_size % self.tensor_parallel_size:
            raise ValueError(
                f"A tensor_parallel_size of {self.tensor_parallel_size} does not divide the world size "
                f"{world_size}: the (dp, tp) mesh needs every rank in exactly one group of each axis."
            )

    @property
    def data_rank(self) -> int:
        """The data-axis coordinate: every rank of one tensor-parallel group reports the same one."""
        return super().data_rank // self.tensor_parallel_size

    @property
    def data_world_size(self) -> int:
        """The size of the data axis: the world size divided by the tensor-parallel degree."""
        return super().data_world_size // self.tensor_parallel_size

    def wrap(self, models: "OrderedDict[str, torch.nn.Module]") -> "OrderedDict[str, torch.nn.Module]":
        """Parallelize the matched submodules on the model axis, then shard on the data axis.

        Raises:
            ValueError: if the tensor-parallel degree does not divide the world size, which no mesh
                can express and which would otherwise leave ranks out of the run.
        """
        if self.sync_batchnorm:
            models = _convert_sync_batchnorm(models, self.device)
        if self._mesh is None:
            self._check_degree()
            world_size = torch.distributed.get_world_size()
            mesh = _device_mesh(self.device, dp=world_size // self.tensor_parallel_size, tp=self.tensor_parallel_size)
            self._tp_mesh, self._mesh = mesh["tp"], mesh["dp"]
        _parallelize_models(models, self._tp_mesh, self.parallel_modules)
        return self._shard(models)


__all__ = [
    "PARALLEL_STYLES",
    "DistributedDataParallelStrategy",
    "DistributedStrategy",
    "FullyShardedDataParallelStrategy",
    "FullyShardedTensorParallelStrategy",
    "SingleDeviceStrategy",
    "TensorParallelStrategy",
    "initial_distributed_env",
    "matched_shard_modules",
    "split_mixed_param_groups",
    "sync_gate",
]


# Unlike the package's other modules, this one is NOT replaced by LazySelectedImporter: generated
# flow functions call sync_gate inside torch.compile'd regions, and dynamo introspects the
# function's module through sys.modules — the shim raises on dunders (`__class__`) and breaks
# tracing (InternalTorchDynamoError). A plain module traces cleanly.
