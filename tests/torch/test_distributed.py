"""Tests for the distributed strategies and the sync gate."""

from collections import OrderedDict
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any, cast

import pytest
from timm.layers import BatchNormAct2d, SyncBatchNormAct

from structcast_model.torch import distributed
from structcast_model.torch.distributed import (
    DistributedDataParallelStrategy,
    DistributedStrategy,
    FullyShardedDataParallelStrategy,
    FullyShardedTensorParallelStrategy,
    SingleDeviceStrategy,
    TensorParallelStrategy,
    matched_shard_modules,
    split_mixed_param_groups,
    sync_gate,
)
import torch

# ---------------------------------------------------------------------------
# sync_gate
# ---------------------------------------------------------------------------


def test_sync_gate_is_a_no_op_for_plain_modules() -> None:
    """Plain modules have no gradient sync to gate, so the gate must leave them untouched."""
    model = torch.nn.Linear(2, 2)
    sync_gate(model, armed=True)
    sync_gate(model, armed=False)
    assert not hasattr(model, "require_backward_grad_sync")


def test_sync_gate_sets_the_ddp_flag_and_leaves_it_for_the_next_gate(single_process_gloo: None) -> None:
    """The gate sets the sync flag and nothing restores it afterwards.

    FSDP2 reads its flag at backward time, after the gated invocation; any restore in between
    would re-enable gradient sync before the backward ran. The next gate on the same module
    overwrites the flag instead.
    """
    ddp = torch.nn.parallel.DistributedDataParallel(torch.nn.Linear(2, 2))
    sync_gate(ddp, armed=False)
    assert ddp.require_backward_grad_sync is False
    sync_gate(ddp, armed=True)
    assert ddp.require_backward_grad_sync is True


def test_sync_gate_sets_fsdp2_gradient_sync(single_process_gloo: None) -> None:
    """The FSDP2 branch must route through set_requires_gradient_sync, once per gate."""
    fsdp = pytest.importorskip("torch.distributed.fsdp")
    model = fsdp.fully_shard(torch.nn.Linear(2, 2))
    calls: list[bool] = []
    original = model.set_requires_gradient_sync

    def _recording(value: bool, **kwargs: Any) -> None:
        calls.append(value)
        original(value, **kwargs)

    model.set_requires_gradient_sync = _recording
    sync_gate(model, armed=False)
    assert calls == [False]
    sync_gate(model, armed=True)
    assert calls == [False, True]


def test_sync_gate_traces_under_torch_compile_without_graph_breaks() -> None:
    """Generated flow functions call the package-imported gate inside compiled regions.

    This only works because structcast_model.torch.distributed is exempt from the package's
    LazySelectedImporter tail: the shim raises on dunder lookups and dynamo's tracer dies on it
    (InternalTorchDynamoError). Restoring the tail keeps every eager test green and breaks
    --compile at runtime — this fullgraph trace is the pin.
    """
    model = torch.nn.Linear(2, 2)

    def flow(x: torch.Tensor) -> torch.Tensor:
        sync_gate(model, armed=True)
        return model(x).sum()

    compiled = torch.compile(flow, fullgraph=True, backend="eager")
    assert torch.isfinite(compiled(torch.randn(3, 2)))


# ---------------------------------------------------------------------------
# SingleDeviceStrategy
# ---------------------------------------------------------------------------


def _linear_models() -> "OrderedDict[str, torch.nn.Module]":
    torch.manual_seed(0)
    return OrderedDict(model=torch.nn.Linear(4, 2))


def test_single_device_strategy_satisfies_the_protocol() -> None:
    """All strategies are used through the DistributedStrategy protocol by the CLI."""
    strategy = SingleDeviceStrategy(device="cpu")
    assert isinstance(strategy, DistributedStrategy)


def test_the_protocol_is_checkable_by_instance_only() -> None:
    """`data_rank` and `data_world_size` made it a data protocol, and those refuse `issubclass`.

    Pinned rather than worked around: no spelling of the two members keeps `issubclass` working
    (attribute annotations are counted the same as properties), so a caller reaching for it has to
    read this instead of discovering a `TypeError` at runtime.
    """
    with pytest.raises(TypeError, match="non-method members"):
        # mypy rejects the call statically for the same reason the runtime does, which is the point.
        issubclass(SingleDeviceStrategy, DistributedStrategy)  # type: ignore[misc]


def test_single_device_wrap_and_sync_are_no_ops() -> None:
    """A single device has nothing to wrap and no ranks to synchronize."""
    strategy = SingleDeviceStrategy(device="cpu")
    models = _linear_models()
    assert strategy.wrap(models) is models
    strategy.sync_initial_weights(models)  # must not require a process group


def _compiled_module(module: torch.nn.Module) -> torch.nn.Module:
    """Compile *module* into its ``OptimizedModule`` wrapper, which torch.compile only types as a callable."""
    return cast(torch.nn.Module, torch.compile(module))


def test_single_device_state_dict_strips_the_compile_wrapper() -> None:
    """Checkpoints must stay loadable no matter whether the model was compiled when saved."""
    strategy = SingleDeviceStrategy(device="cpu")
    models = _linear_models()
    compiled = OrderedDict(model=_compiled_module(models["model"]))
    states = strategy.state_dict(compiled)
    assert set(states["models"]["model"]) == {"weight", "bias"}


def test_single_device_round_trips_models_and_optimizers() -> None:
    """A saved training state must restore weights and optimizer hyperparameters exactly."""
    strategy = SingleDeviceStrategy(device="cpu")
    models = _linear_models()
    optimizer = torch.optim.SGD(models["model"].parameters(), lr=0.125, momentum=0.9)
    pairing = {"opt": ["model"]}
    models["model"](torch.randn(1, 4)).sum().backward()
    optimizer.step()
    state = strategy.state_dict(models, {"opt": optimizer}, pairing)

    torch.manual_seed(123)
    fresh_models: OrderedDict[str, torch.nn.Module] = OrderedDict(model=torch.nn.Linear(4, 2))
    fresh_optimizer = torch.optim.SGD(fresh_models["model"].parameters(), lr=0.5, momentum=0.9)
    returned = strategy.load_state_dict(fresh_models, {"opt": fresh_optimizer}, pairing, state)
    assert torch.equal(fresh_models["model"].get_parameter("weight"), models["model"].get_parameter("weight"))
    assert fresh_optimizer.param_groups[0]["lr"] == 0.125
    assert returned is state


def test_default_compile_returns_the_module_when_compilation_is_off() -> None:
    """`compile_kw` of None is how the CLI says "no --compile"; every strategy must pass through."""
    module = torch.nn.Linear(4, 2)
    assert SingleDeviceStrategy(device="cpu").compile(module, None) is module


def test_default_compile_compiles_modules_in_place_and_wraps_callables() -> None:
    """Modules keep their identity; plain callables get the torch.compile wrapper.

    An OptimizedModule wrapper would shift named_modules() paths (which `wrap` matches on) and prefix
    checkpoint keys with '_orig_mod.'; the generated flow functions have no in-place form.
    """
    strategy = SingleDeviceStrategy(device="cpu")
    module = torch.nn.Linear(4, 2)
    compiled = strategy.compile(module, {})
    assert compiled is module
    assert module._compiled_call_impl is not None  # noqa: SLF001  # the only marker .compile() leaves
    assert set(compiled.state_dict()) == {"weight", "bias"}

    def flow(x: torch.Tensor) -> torch.Tensor:
        return x + 1

    assert strategy.compile(flow, {}) is not flow


class _ProxyOptimizer:
    """An optimizer proxy in the AdamWWithCosine shape: not a torch.optim.Optimizer, own state dict."""

    def __init__(self) -> None:
        self.state = {"scale": 1.0, "scheduler": {"last_epoch": 4}}
        self.param_groups = [{"lr": 0.25, "params": []}]

    def state_dict(self) -> dict:
        return dict(self.state)

    def load_state_dict(self, state: dict) -> None:
        self.state = dict(state)


def test_single_device_round_trips_proxy_optimizers_through_their_own_state() -> None:
    """Optimizer proxies merge scheduler state that DCP would reject and drop; it must survive."""
    strategy = SingleDeviceStrategy(device="cpu")
    models = _linear_models()
    proxy = _ProxyOptimizer()
    state = strategy.state_dict(models, {"opt": proxy}, {"opt": ["model"]})
    assert state["optimizers"]["opt"] == {"scale": 1.0, "scheduler": {"last_epoch": 4}}

    fresh = _ProxyOptimizer()
    fresh.state = {}
    strategy.load_state_dict(models, {"opt": fresh}, {"opt": ["model"]}, state)
    assert fresh.state == {"scale": 1.0, "scheduler": {"last_epoch": 4}}


def test_fsdp2_strategy_refuses_proxy_optimizers(single_process_gloo: None) -> None:
    """A proxy's own state dict cannot represent sharded parameters; FSDP2 must fail loud."""
    pytest.importorskip("torch.distributed.fsdp")
    strategy = FullyShardedDataParallelStrategy(device="cpu")
    wrapped = strategy.wrap(_linear_models())
    with pytest.raises(ValueError, match="torch.optim.Optimizer"):
        strategy.state_dict(wrapped, {"opt": _ProxyOptimizer()}, {"opt": ["model"]})


def test_single_device_load_without_state_fails_loud() -> None:
    """Resuming without a training state is a caller bug, not something to paper over."""
    strategy = SingleDeviceStrategy(device="cpu")
    with pytest.raises(ValueError, match="training state"):
        strategy.load_state_dict(_linear_models(), {}, None, None)


@pytest.mark.parametrize(
    "optimizer_creator",
    [partial(torch.optim.SGD, lr=0.1, momentum=0.9), partial(torch.optim.Adam, lr=0.1)],
    ids=["sgd-momentum", "adam"],
)
def test_single_device_refuses_index_keyed_optimizer_state(
    optimizer_creator: Callable[..., torch.optim.Optimizer], tmp_path: Path
) -> None:
    """A state saved without a pairing keys optimizer state by position and must be refused, not restored.

    The name-keyed load path cannot resolve positions, and today it fails differently per optimizer:
    SGD momentum is silently discarded, so the run resumes with fresh moments, while Adam dies with an
    opaque ``KeyError: 'state.0.step'``. Both must become one explicit refusal (ADR-0008).
    """
    strategy = SingleDeviceStrategy(device="cpu")
    models = _linear_models()
    optimizer = optimizer_creator(models["model"].parameters())
    models["model"](torch.randn(1, 4)).sum().backward()
    optimizer.step()
    saved = strategy.state_dict(models, {"opt": optimizer})  # no pairing -> state keyed 0, 1, ...
    path = tmp_path / "legacy.pt"
    torch.save(saved, path)
    state = torch.load(path, map_location="cpu", weights_only=True)
    assert set(state["optimizers"]["opt"]["state"]) == {0, 1}

    fresh = _linear_models()
    fresh_optimizer = optimizer_creator(fresh["model"].parameters())
    with pytest.raises(ValueError, match="keyed by parameter index"):
        strategy.load_state_dict(fresh, {"opt": fresh_optimizer}, {"opt": ["model"]}, state)


def _trunk_and_head() -> "OrderedDict[str, torch.nn.Module]":
    """Two paired models of which a warmup step exercises only the trunk, leaving the head unstepped."""
    torch.manual_seed(0)
    return OrderedDict(trunk=torch.nn.Linear(4, 2), head=torch.nn.Linear(2, 1))


def test_single_device_resumes_partial_optimizer_state_only_when_asked(tmp_path: Path) -> None:
    """A parameter that has not been stepped yet has no optimizer state, and torch refuses the gap.

    One optimizer pairs a trunk and a head that a warmup phase never runs, so a normal mid-run save
    covers the trunk only. torch's default strict load rejects such a training state outright, which
    would make the run unresumable; ``strict_optimizer_load=False`` accepts the gap and lets the
    uncovered parameters start fresh. Missing state is never synthesized, so a zero-filled moment can
    never masquerade as a restored one.
    """
    strategy = SingleDeviceStrategy(device="cpu")
    models = _trunk_and_head()
    pairing = {"opt": ["trunk", "head"]}
    optimizer = torch.optim.SGD([p for m in models.values() for p in m.parameters()], lr=0.1, momentum=0.9)
    models["trunk"](torch.randn(3, 4)).sum().backward()
    optimizer.step()
    saved = strategy.state_dict(models, {"opt": optimizer}, pairing)
    assert set(saved["optimizers"]["opt"]["state"]) == {"trunk.weight", "trunk.bias"}
    path = tmp_path / "partial.pt"
    torch.save(saved, path)
    covered = {n: e["momentum_buffer"].clone() for n, e in saved["optimizers"]["opt"]["state"].items()}

    strict_models = _trunk_and_head()
    strict_optimizer = torch.optim.SGD([p for m in strict_models.values() for p in m.parameters()], lr=0.5)
    with pytest.raises(RuntimeError, match="Missing optimizer state"):
        strategy.load_state_dict(
            strict_models,
            {"opt": strict_optimizer},
            pairing,
            torch.load(path, map_location="cpu", weights_only=True),
        )

    lenient = SingleDeviceStrategy(device="cpu", strict_optimizer_load=False)
    resumed = _trunk_and_head()
    resumed_optimizer = torch.optim.SGD([p for m in resumed.values() for p in m.parameters()], lr=0.5, momentum=0.9)
    lenient.load_state_dict(
        resumed,
        {"opt": resumed_optimizer},
        pairing,
        torch.load(path, map_location="cpu", weights_only=True),
    )
    parameters = {f"trunk.{n}": p for n, p in resumed["trunk"].named_parameters()}
    for name, buffer in covered.items():
        assert torch.equal(resumed_optimizer.state[parameters[name]]["momentum_buffer"], buffer)
    # Nothing is synthesized for the head, but its entries are not absent either: torch materializes
    # state for every parameter (a step with lr=0) before loading, so the head keeps zero-filled
    # moments. For SGD momentum that is arithmetically the unstepped state (buf = 0 * momentum + grad),
    # so no restored-looking moment can reach it.
    head_buffers = [resumed_optimizer.state[p]["momentum_buffer"] for p in resumed["head"].parameters()]
    assert len(head_buffers) == 2
    assert not any(buffer.any() for buffer in head_buffers)
    assert resumed_optimizer.param_groups[0]["lr"] == 0.1
    resumed["trunk"](torch.randn(3, 4)).sum().backward()
    resumed_optimizer.step()


# ---------------------------------------------------------------------------
# DistributedDataParallelStrategy
# ---------------------------------------------------------------------------


def test_ddp_strategy_wraps_on_cpu_without_device_ids(single_process_gloo: None) -> None:
    """DDP raises on device_ids=['cpu']; the strategy must construct CPU DDP with device_ids=None."""
    strategy = DistributedDataParallelStrategy(device="cpu")
    wrapped = strategy.wrap(_linear_models())
    assert isinstance(wrapped["model"], torch.nn.parallel.DistributedDataParallel)


def test_ddp_strategy_sync_initial_weights_broadcasts_rank0(single_process_gloo: None) -> None:
    """The broadcast must run on plain pre-wrap tensors; with one rank it is an exact no-op."""
    strategy = DistributedDataParallelStrategy(device="cpu")
    models = _linear_models()
    before = models["model"].get_parameter("weight").clone()
    strategy.sync_initial_weights(models)
    assert torch.equal(models["model"].get_parameter("weight"), before)


def test_ddp_strategy_state_dict_has_wrapper_free_keys(single_process_gloo: None) -> None:
    """DDP prefixes every key with 'module.'; checkpoints must not leak that wrapper detail."""
    strategy = DistributedDataParallelStrategy(device="cpu")
    wrapped = strategy.wrap(_linear_models())
    states = strategy.state_dict(wrapped)
    assert not any(key.startswith("module.") for key in states["models"]["model"])


def test_ddp_strategy_round_trips_through_wrapped_models(single_process_gloo: None) -> None:
    """A state saved from wrapped models must load back into freshly wrapped models."""
    strategy = DistributedDataParallelStrategy(device="cpu")
    wrapped = strategy.wrap(_linear_models())
    optimizer = torch.optim.SGD(wrapped["model"].parameters(), lr=0.25)
    state = strategy.state_dict(wrapped, {"opt": optimizer}, {"opt": ["model"]})

    torch.manual_seed(7)
    fresh = strategy.wrap(OrderedDict(model=torch.nn.Linear(4, 2)))
    fresh_optimizer = torch.optim.SGD(fresh["model"].parameters(), lr=0.75)
    strategy.load_state_dict(fresh, {"opt": fresh_optimizer}, {"opt": ["model"]}, state)
    assert torch.equal(fresh["model"].get_parameter("module.weight"), wrapped["model"].get_parameter("module.weight"))
    assert fresh_optimizer.param_groups[0]["lr"] == 0.25


# ---------------------------------------------------------------------------
# FullyShardedDataParallelStrategy
# ---------------------------------------------------------------------------


class _FailedImports:
    """Stand-in for a try_import context whose imports failed."""

    is_successful = False


def test_fsdp2_strategy_requires_fully_shard(monkeypatch: pytest.MonkeyPatch) -> None:
    """Selecting FSDP2 on a torch without fully_shard must fail with an actionable message."""
    # The lazy-import shim hides module privates, so patch the globals the class actually reads.
    monkeypatch.setitem(FullyShardedDataParallelStrategy.__post_init__.__globals__, "_fsdp_imports", _FailedImports())
    with pytest.raises(ImportError, match="torch>=2.6"):
        FullyShardedDataParallelStrategy(device="cpu")


def test_fsdp2_strategy_shards_in_place_and_saves_plain_tensors(single_process_gloo: None) -> None:
    """fully_shard swaps parameters to DTensor; checkpoints must still hold plain tensors."""
    pytest.importorskip("torch.distributed.fsdp")
    strategy = FullyShardedDataParallelStrategy(device="cpu")
    models = _linear_models()
    reference = models["model"].get_parameter("weight").clone()
    wrapped = strategy.wrap(models)
    states = strategy.state_dict(wrapped)
    weight = states["models"]["model"]["weight"]
    assert type(weight) is torch.Tensor
    assert torch.equal(weight, reference)


def test_fsdp2_strategy_round_trips_sharded_models(single_process_gloo: None) -> None:
    """A gathered FSDP2 state must load back into a freshly sharded model."""
    pytest.importorskip("torch.distributed.fsdp")
    strategy = FullyShardedDataParallelStrategy(device="cpu")
    wrapped = strategy.wrap(_linear_models())
    optimizer = torch.optim.SGD(wrapped["model"].parameters(), lr=0.25, momentum=0.9)
    wrapped["model"](torch.randn(1, 4)).sum().backward()
    optimizer.step()
    state = strategy.state_dict(wrapped, {"opt": optimizer}, {"opt": ["model"]})
    # set_model_state_dict(broadcast_from_rank0=True) mutates the given state in place, so keep
    # the expected weight aside before loading.
    expected_weight = state["models"]["model"]["weight"].clone()

    torch.manual_seed(11)
    fresh = strategy.wrap(OrderedDict(model=torch.nn.Linear(4, 2)))
    fresh_optimizer = torch.optim.SGD(fresh["model"].parameters(), lr=0.75, momentum=0.9)
    strategy.load_state_dict(fresh, {"opt": fresh_optimizer}, {"opt": ["model"]}, state)
    restored = strategy.state_dict(fresh)["models"]["model"]["weight"]
    assert torch.equal(restored, expected_weight)
    assert fresh_optimizer.param_groups[0]["lr"] == 0.25


def test_fsdp2_strategy_refuses_optimizer_state_without_pairing(single_process_gloo: None) -> None:
    """Sharded optimizer state is resolvable only through parameter FQNs, which need the pairing."""
    pytest.importorskip("torch.distributed.fsdp")
    strategy = FullyShardedDataParallelStrategy(device="cpu")
    wrapped = strategy.wrap(_linear_models())
    optimizer = torch.optim.SGD(wrapped["model"].parameters(), lr=0.25)
    with pytest.raises(ValueError, match="optimizer_models"):
        strategy.state_dict(wrapped, {"opt": optimizer}, None)


def test_fsdp2_strategy_refuses_index_keyed_optimizer_state(single_process_gloo: None) -> None:
    """The shared refusal must reach FSDP2, where a positional load would corrupt sharded state silently.

    Today an index-keyed state passes torch's int-key passthrough and installs unsharded tensors
    beside DTensor parameters without an error (ADR-0008) — the worst of the silent outcomes, so the
    mixin guard must fire here and not be shadowed by FSDP2's own overrides.
    """
    pytest.importorskip("torch.distributed.fsdp")
    single = SingleDeviceStrategy(device="cpu")
    models = _linear_models()
    optimizer = torch.optim.SGD(models["model"].parameters(), lr=0.1, momentum=0.9)
    models["model"](torch.randn(1, 4)).sum().backward()
    optimizer.step()
    legacy = single.state_dict(models, {"opt": optimizer})  # no pairing -> state keyed 0, 1, ...

    strategy = FullyShardedDataParallelStrategy(device="cpu")
    wrapped = strategy.wrap(_linear_models())
    sharded_optimizer = torch.optim.SGD(wrapped["model"].parameters(), lr=0.1, momentum=0.9)
    with pytest.raises(ValueError, match="keyed by parameter index"):
        strategy.load_state_dict(wrapped, {"opt": sharded_optimizer}, {"opt": ["model"]}, legacy)


# ---------------------------------------------------------------------------
# Per-block sharding
# ---------------------------------------------------------------------------


class _BlockModel(torch.nn.Module):
    """A blocked model in the shape per-block sharding targets: two blocks feeding a shared head."""

    def __init__(self) -> None:
        super().__init__()
        self.block0 = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Linear(4, 4))
        self.block1 = torch.nn.Linear(4, 2)
        self.head = torch.nn.Linear(2, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the blocks and the head in sequence."""
        return self.head(self.block1(self.block0(x)))


def _block_models() -> "OrderedDict[str, torch.nn.Module]":
    torch.manual_seed(0)
    return OrderedDict(model=_BlockModel())


def _param_group(module: Any) -> Any:
    """*module*'s own ``fully_shard`` group. Reached through private FSDP2 state, which torch does not type."""
    return module._get_fsdp_state()._fsdp_param_group


def _group_size(module: torch.nn.Module) -> int:
    """Number of parameters in *module*'s own ``fully_shard`` group.

    FSDP2 exposes no public view of a group's membership, and membership is the whole point of
    per-block sharding: the root must hold only what no matched block claimed.
    """
    return len(_param_group(module).fsdp_params)


def test_matched_shard_modules_keeps_named_modules_order() -> None:
    """The wrap reverses this order to shard descendants first, so ancestors must come first here."""
    matched = matched_shard_modules(_block_models(), ["block0", "block0.1", "block1"])
    assert [path for path, _ in matched["model"]] == ["block0", "block0.1", "block1"]


def test_matched_shard_modules_sees_through_the_compile_wrapper() -> None:
    """torch.compile prefixes every path with '_orig_mod.'; patterns must not have to know that."""
    models: OrderedDict[str, torch.nn.Module] = OrderedDict(model=_compiled_module(_BlockModel()))
    matched = matched_shard_modules(models, ["block?"])
    assert [path for path, _ in matched["model"]] == ["_orig_mod.block0", "_orig_mod.block1"]


def test_matched_shard_modules_globs_stay_within_one_path_segment() -> None:
    """'block*' must match the blocks, not their contents.

    fnmatch's dot-crossing '*' would shard every leaf module as its own communication group and
    leave the block groups empty.
    """
    matched = matched_shard_modules(_block_models(), ["block*"])
    assert [path for path, _ in matched["model"]] == ["block0", "block1"]
    assert [path for path, _ in matched_shard_modules(_block_models(), ["block?.?"])["model"]] == [
        "block0.0",
        "block0.1",
    ]


def test_matched_shard_modules_fails_loud_on_a_pattern_matching_nothing() -> None:
    """A pattern matching nothing anywhere is a typo that would silently train unsharded blocks."""
    with pytest.raises(ValueError, match="blcok"):
        matched_shard_modules(_block_models(), ["block*", "blcok*"])


def test_matched_shard_modules_never_matches_the_root() -> None:
    """Wrap shards the root last unconditionally; a catch-all pattern must not shard it twice."""
    matched = matched_shard_modules(_block_models(), ["*"])
    assert "" not in [path for path, _ in matched["model"]]
    compiled: OrderedDict[str, torch.nn.Module] = OrderedDict(model=_compiled_module(_BlockModel()))
    assert "_orig_mod" not in [path for path, _ in matched_shard_modules(compiled, ["*"])["model"]]


def test_fsdp2_per_block_wrap_refuses_a_tie_across_sibling_blocks(single_process_gloo: None) -> None:
    """fully_shard replaces each group's parameters, so a tie split across groups silently diverges."""
    pytest.importorskip("torch.distributed.fsdp")
    models = _block_models()
    models["model"].get_submodule("block0.1").weight = models["model"].get_parameter("block0.0.weight")
    strategy = FullyShardedDataParallelStrategy(device="cpu", shard_modules=["block0.0", "block0.1"])
    with pytest.raises(RuntimeError, match="Tied parameter"):
        strategy.wrap(models)
    # The guard must run before any sharding: a half-sharded model cannot be recovered from.
    assert type(models["model"].get_parameter("block0.0.weight")) is torch.nn.Parameter


def test_fsdp2_per_block_wrap_refuses_a_tie_with_an_unmatched_module(single_process_gloo: None) -> None:
    """A tie to a module outside every pattern lands in the root group, which is a different group."""
    pytest.importorskip("torch.distributed.fsdp")
    models = _block_models()
    models["model"].get_submodule("head").weight = models["model"].get_parameter("block1.weight")
    strategy = FullyShardedDataParallelStrategy(device="cpu", shard_modules=["block1"])
    with pytest.raises(RuntimeError, match="Tied parameter"):
        strategy.wrap(models)


def test_fsdp2_per_block_wrap_allows_a_tie_inside_one_block(single_process_gloo: None) -> None:
    """A tie both of whose ends land in the same group is sharded once and stays tied."""
    pytest.importorskip("torch.distributed.fsdp")
    models = _block_models()
    models["model"].get_submodule("block0.1").weight = models["model"].get_parameter("block0.0.weight")
    strategy = FullyShardedDataParallelStrategy(device="cpu", shard_modules=["block0"])
    wrapped = strategy.wrap(models)["model"]
    assert wrapped.get_parameter("block0.1.weight") is wrapped.get_parameter("block0.0.weight")


def test_fsdp2_per_block_wrap_gives_every_matched_module_its_own_group(single_process_gloo: None) -> None:
    """Matched blocks become their own groups and the root keeps only the parameters left over."""
    fsdp = pytest.importorskip("torch.distributed.fsdp")
    strategy = FullyShardedDataParallelStrategy(device="cpu", shard_modules=["block?"])
    wrapped = strategy.wrap(_block_models())["model"]
    assert isinstance(wrapped, fsdp.FSDPModule)
    assert isinstance(wrapped.block0, fsdp.FSDPModule)
    assert isinstance(wrapped.block1, fsdp.FSDPModule)
    assert _group_size(wrapped.block0) == 4  # two linears
    assert _group_size(wrapped.block1) == 2
    assert _group_size(wrapped) == 2  # the head only; the blocks' parameters are not re-collected
    wrapped(torch.randn(2, 4)).sum().backward()  # the groups must still compose into one model


def test_fsdp2_wrap_without_patterns_stays_a_single_group(single_process_gloo: None) -> None:
    """The default must keep the pre-existing behavior: one group holding the whole model."""
    fsdp = pytest.importorskip("torch.distributed.fsdp")
    strategy = FullyShardedDataParallelStrategy(device="cpu")
    wrapped = strategy.wrap(_block_models())["model"]
    assert not isinstance(wrapped.block0, fsdp.FSDPModule)
    assert _group_size(wrapped) == 8


def test_fsdp2_sync_gate_on_the_root_reaches_the_block_groups(single_process_gloo: None) -> None:
    """Generated steps gate the root model only; per-block groups must follow it, or they reduce early."""
    pytest.importorskip("torch.distributed.fsdp")
    strategy = FullyShardedDataParallelStrategy(device="cpu", shard_modules=["block?"])
    wrapped = strategy.wrap(_block_models())["model"]
    sync_gate(wrapped, armed=False)
    assert _param_group(wrapped.block0).reduce_grads is False


def _is_compiled(module: torch.nn.Module) -> bool:
    """Whether ``.compile()`` ran on *module* itself; the compiled call impl is the only marker it leaves."""
    return module._compiled_call_impl is not None


def test_fsdp2_compile_compiles_the_matched_blocks_and_not_the_root() -> None:
    """Compile units follow the shard boundaries: a root graph buries the per-block hooks (ADR-0004)."""
    pytest.importorskip("torch.distributed.fsdp")
    strategy = FullyShardedDataParallelStrategy(device="cpu", shard_modules=["block?"])
    model = _block_models()["model"]
    assert strategy.compile(model, {}) is model
    assert _is_compiled(model.block0)
    assert _is_compiled(model.block1)
    assert not _is_compiled(model)


def test_fsdp2_compile_without_patterns_compiles_the_root() -> None:
    """One group, one compile unit: without shard_modules the root is the only boundary there is."""
    pytest.importorskip("torch.distributed.fsdp")
    strategy = FullyShardedDataParallelStrategy(device="cpu")
    model = _block_models()["model"]
    assert strategy.compile(model, {}) is model
    assert _is_compiled(model)
    assert not _is_compiled(model.block0)


def test_fsdp2_compile_falls_back_to_the_root_when_the_patterns_match_nothing() -> None:
    """A blockless model (a CycleGAN discriminator) is normal per module, so compile must not raise.

    Only wrap, which sees every model at once, can tell a legitimate miss from a typo'd pattern.
    """
    pytest.importorskip("torch.distributed.fsdp")
    strategy = FullyShardedDataParallelStrategy(device="cpu", shard_modules=["block?"])
    model = torch.nn.Linear(4, 2)
    assert strategy.compile(model, {}) is model
    assert _is_compiled(model)


# ---------------------------------------------------------------------------
# Tensor parallelism
# ---------------------------------------------------------------------------


class _MLPModel(torch.nn.Module):
    """The shape a tensor-parallel plan targets: a column-parallel layer feeding a row-parallel one."""

    def __init__(self) -> None:
        super().__init__()
        self.up = torch.nn.Linear(4, 8)
        self.down = torch.nn.Linear(8, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the pair."""
        return self.down(torch.relu(self.up(x)))


_PLAN = [("up", "column"), ("down", "row")]


def _mlp_models() -> "OrderedDict[str, torch.nn.Module]":
    torch.manual_seed(0)
    return OrderedDict(model=_MLPModel())


def _placements(model: torch.nn.Module, name: str) -> Any:
    """One parameter's DTensor placements; ``parallelize_module`` replaces the parameter with one."""
    return cast(Any, model.get_parameter(name)).placements


def test_tensor_parallel_strategy_satisfies_the_protocol_and_reports_one_data_slice() -> None:
    """The ranks of a tensor-parallel group split one model, so they consume one and the same batch.

    A strategy reporting the global rank here would have the CLI seed each rank differently and a
    rank-aware loader hand each of them different items — two silently wrong runs, not two errors.
    """
    strategy = TensorParallelStrategy(device="cpu", parallel_modules=_PLAN)

    assert isinstance(strategy, DistributedStrategy)
    assert (strategy.data_rank, strategy.data_world_size) == (0, 1)


@pytest.mark.parametrize(
    ("strategy", "expected"),
    [
        (SingleDeviceStrategy(device="cpu"), (0, 1)),
        (DistributedDataParallelStrategy(device="cpu"), (0, 1)),
    ],
    ids=["single", "ddp"],
)
def test_data_coordinates_of_the_replicating_strategies(
    strategy: DistributedStrategy, expected: tuple[int, int]
) -> None:
    """One replica per rank: outside a process group that is one slice, and the seed is the plain seed."""
    assert (strategy.data_rank, strategy.data_world_size) == expected


def test_tensor_parallel_strategy_requires_the_torch_tensor_parallel_api(monkeypatch: pytest.MonkeyPatch) -> None:
    """Selecting it on a torch without parallelize_module must fail with an actionable message."""
    # The lazy-import shim hides module privates, so patch the globals the class actually reads.
    monkeypatch.setitem(TensorParallelStrategy.__post_init__.__globals__, "_tp_imports", _FailedImports())
    with pytest.raises(ImportError, match="torch>=2.4"):
        TensorParallelStrategy(device="cpu", parallel_modules=_PLAN)


def test_tensor_parallel_strategy_refuses_an_empty_plan() -> None:
    """Without a plan the strategy would parallelize nothing and run every rank on the whole model."""
    with pytest.raises(ValueError, match="parallel_modules"):
        TensorParallelStrategy(device="cpu")


def test_a_parallel_modules_pattern_matching_nothing_is_refused(single_process_gloo: None) -> None:
    """A typo'd path would silently leave the layer unsplit; the message must name the option to fix."""
    strategy = TensorParallelStrategy(device="cpu", parallel_modules=[("up", "column"), ("dwon", "row")])
    with pytest.raises(ValueError, match="parallel_modules pattern"):
        strategy.wrap(_mlp_models())


def test_an_unknown_parallel_style_is_refused(single_process_gloo: None) -> None:
    """A mistyped style is a configuration error, not a reason to leave the layer replicated."""
    strategy = TensorParallelStrategy(device="cpu", parallel_modules=[("up", "colwise")])
    with pytest.raises(ValueError, match="Unknown parallel style"):
        strategy.wrap(_mlp_models())


def test_tensor_parallel_refuses_a_tie_across_two_parallelized_modules(single_process_gloo: None) -> None:
    """The two ends would become separately placed DTensors — the same silent split fully_shard has."""
    models = _mlp_models()
    # The shapes do not match, which is irrelevant: the guard runs before anything is placed or run.
    models["model"].get_submodule("down").weight = models["model"].get_parameter("up.weight")
    strategy = TensorParallelStrategy(device="cpu", parallel_modules=_PLAN)
    with pytest.raises(RuntimeError, match="Tied parameter"):
        strategy.wrap(models)
    assert type(models["model"].get_parameter("up.weight")) is torch.nn.Parameter  # nothing was parallelized


def test_tensor_parallel_places_the_styles_the_vocabulary_names(single_process_gloo: None) -> None:
    """Column splits a weight by its output dimension and row by its input one, bias replicated.

    The bias is what the check is for: a row-parallel layer all-reduces its partial products, so a
    split bias would be added once per shard and counted as many times as the group is wide.
    """
    strategy = TensorParallelStrategy(device="cpu", parallel_modules=_PLAN)
    model = strategy.wrap(_mlp_models())["model"]

    placements = {name: _placements(model, name) for name, _ in model.named_parameters()}
    assert placements["up.weight"] == (torch.distributed.tensor.Shard(0),)
    assert placements["up.bias"] == (torch.distributed.tensor.Shard(0),)
    assert placements["down.weight"] == (torch.distributed.tensor.Shard(1),)
    assert placements["down.bias"] == (torch.distributed.tensor.Replicate(),)


def test_the_gate_stays_a_no_op_on_a_tensor_parallel_model(single_process_gloo: None) -> None:
    """Pure tensor parallelism has no deferred bucket, so the generated steps' gate must find nothing.

    ``parallelize_module`` returns the plain ``nn.Module`` it was given -- neither a DDP wrapper nor
    an ``FSDPModule`` -- and DTensor emits each layer's collective inside the operation that needs
    it, so gradients arrive with the gate never having been armed.
    """
    strategy = TensorParallelStrategy(device="cpu", parallel_modules=_PLAN)
    model = strategy.wrap(_mlp_models())["model"]

    sync_gate(model, armed=False)

    assert not hasattr(model, "require_backward_grad_sync")
    model(torch.randn(2, 4)).sum().backward()
    assert model.get_parameter("up.weight").grad is not None


def test_tensor_parallel_takes_a_style_instance_from_the_plan(single_process_gloo: None) -> None:
    """The escape hatch for the styles the vocabulary lacks: an object pattern the CLI instantiated."""
    parallel = pytest.importorskip("torch.distributed.tensor.parallel")
    strategy = TensorParallelStrategy(
        device="cpu",
        parallel_modules=[("up", parallel.RowwiseParallel(input_layouts=torch.distributed.tensor.Replicate()))],
    )
    model = strategy.wrap(_mlp_models())["model"]

    assert _placements(model, "up.weight") == (torch.distributed.tensor.Shard(1),)


def test_the_column_heads_style_keeps_its_output_a_dtensor() -> None:
    """An attention head reshape must see the sharded head count, which a local tensor hides."""
    parallel = pytest.importorskip("torch.distributed.tensor.parallel")
    style = distributed._parallel_style("column_heads")

    assert isinstance(style, parallel.ColwiseParallel)
    assert style.use_local_output is False


def test_tensor_parallel_state_dict_gathers_plain_tensors(single_process_gloo: None) -> None:
    """A checkpoint must not depend on the tensor-parallel degree that wrote it."""
    strategy = TensorParallelStrategy(device="cpu", parallel_modules=_PLAN)
    models = _mlp_models()
    reference = models["model"].get_parameter("up.weight").clone()
    states = strategy.state_dict(strategy.wrap(models))

    weight = states["models"]["model"]["up.weight"]
    assert type(weight) is torch.Tensor
    assert torch.equal(weight, reference)


def test_fsdp2_tensor_parallel_shards_the_parallelized_models_and_arms_the_gate(single_process_gloo: None) -> None:
    """fully_shard must land on the parallelized root, or the gate has nothing to arm.

    Generated steps gate the model root, and only an ``FSDPModule`` reads that flag: if the
    combination stopped at ``parallelize_module``, gradient synchronization would never be deferred
    and accumulation would reduce on every micro-step.
    """
    fsdp = pytest.importorskip("torch.distributed.fsdp")
    strategy = FullyShardedTensorParallelStrategy(device="cpu", tensor_parallel_size=1, parallel_modules=_PLAN)
    model = strategy.wrap(_mlp_models())["model"]

    assert isinstance(model, fsdp.FSDPModule)
    sync_gate(model, armed=False)
    assert _param_group(model).reduce_grads is False
    model(torch.randn(2, 4)).sum().backward()  # the two wrappers must still compose into one model


def test_fsdp2_tensor_parallel_refuses_a_degree_the_world_does_not_divide(single_process_gloo: None) -> None:
    """A degree that leaves a partial group cannot be a mesh, and would drop ranks out of the run.

    Refused at construction, where the process group already exists: the CLI reads `data_rank` on
    the next line, so a degree that only failed at wrap would have gone through the seeding first.
    """
    with pytest.raises(ValueError, match="does not divide the world size"):
        FullyShardedTensorParallelStrategy(device="cpu", tensor_parallel_size=3, parallel_modules=_PLAN)


@pytest.mark.parametrize("degree", [0, -1], ids=["zero", "negative"])
def test_fsdp2_tensor_parallel_refuses_a_degree_below_one(degree: int) -> None:
    """The data coordinates divide by the degree, so a degree below 1 is arithmetic, not a strategy.

    Zero reached the CLI's seeding line as a bare ZeroDivisionError and -1 published a negative data
    world size for the loader to shard on; both are configuration errors that must say so.
    """
    with pytest.raises(ValueError, match="at least 1"):
        FullyShardedTensorParallelStrategy(device="cpu", tensor_parallel_size=degree, parallel_modules=_PLAN)


def test_fsdp2_tensor_parallel_refuses_an_empty_plan() -> None:
    """Without a plan the combination is plain FSDP2, and saying so beats pretending otherwise."""
    with pytest.raises(ValueError, match="parallel_modules"):
        FullyShardedTensorParallelStrategy(device="cpu", tensor_parallel_size=2)


# ---------------------------------------------------------------------------
# The shared state-dict API field
# ---------------------------------------------------------------------------


def test_every_strategy_resolves_the_state_dict_api_at_construction() -> None:
    """The DCP module is resolved once into a field, so every constructor must reach the mixin's hook.

    `TensorParallelStrategy` and `FullyShardedDataParallelStrategy` define a `__post_init__` of
    their own, which shadows the mixin's unless it chains: the field would then stay unset and every
    save and load raise `AttributeError` at the first checkpoint rather than at construction.
    """
    dcp = pytest.importorskip("torch.distributed.checkpoint.state_dict")
    pytest.importorskip("torch.distributed.fsdp")
    # Annotated at the mixin owning `_api`: the join of the five classes is `_CompileMixin`, which does not.
    strategies: list[distributed._StateDictMixin] = [
        SingleDeviceStrategy(device="cpu"),
        DistributedDataParallelStrategy(device="cpu"),
        TensorParallelStrategy(device="cpu", parallel_modules=_PLAN),
        FullyShardedDataParallelStrategy(device="cpu"),
        FullyShardedTensorParallelStrategy(device="cpu", tensor_parallel_size=1, parallel_modules=_PLAN),
    ]

    assert [strategy._api for strategy in strategies] == [dcp] * len(strategies)


# ---------------------------------------------------------------------------
# split_mixed_param_groups
# ---------------------------------------------------------------------------

_MIXED_PLAN = [("up", "column")]
"""A plan naming one of the two layers; the other one's parameters are what stays plain."""

_MIXED_INPUT = torch.linspace(-1.0, 1.0, 8).reshape(2, 4)
"""A fixed batch, so two optimizers stepped separately see the very same gradients."""

_MIXED_PARAMETERS = ("up.weight", "up.bias", "down.weight", "down.bias")


def _mixed_optimizer(*, foreach: bool) -> tuple[torch.nn.Module, torch.optim.AdamW]:
    """A partly parallelized model and one AdamW holding all of its parameters in one group.

    The shape of every tensor-parallel run: the plan converts the layers it names to DTensors, and
    the learner hands the whole model -- parallelized layers and untouched ones alike -- to one
    optimizer.
    """
    model = TensorParallelStrategy(device="cpu", parallel_modules=_MIXED_PLAN).wrap(_mlp_models())["model"]
    return model, torch.optim.AdamW(model.parameters(), lr=0.1, foreach=foreach)


def _stepped(model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> None:
    """Run the backward and the optimizer step the mixed group crashes on."""
    model(_MIXED_INPUT).sum().backward()
    optimizer.step()


def _kinds(group: dict[str, Any]) -> set[str]:
    """The tensor types one parameter group holds; a fused ``_foreach_*`` call admits exactly one."""
    return {type(parameter).__name__ for parameter in group["params"]}


def _local(tensor: torch.Tensor) -> torch.Tensor:
    """The rank-local values of a tensor, so a DTensor and a plain one compare the same way."""
    to_local = getattr(tensor, "to_local", None)
    return cast(torch.Tensor, to_local()) if to_local else tensor


def test_a_group_mixing_dtensor_and_plain_parameters_crashes_the_first_step(single_process_gloo: None) -> None:
    """The shipped defect: `cfg/torch/strategies/tp.yaml` plus any learner dies on `optimizer.step()`.

    torch's default multi-tensor path fuses a parameter group into one ``_foreach_*`` call and the
    DTensor dispatcher refuses a list holding both kinds, so the run never reports a loss. Every plan
    leaves something plain -- an untouched head is the least a transformer has -- so this is not an
    exotic configuration but the only one tensor parallelism produces.
    """
    model, optimizer = _mixed_optimizer(foreach=True)
    model(_MIXED_INPUT).sum().backward()

    with pytest.raises(RuntimeError, match="mixed torch.Tensor and DTensor"):
        optimizer.step()


def test_splitting_makes_the_mixed_group_uniform_and_keeps_its_hyperparameters(single_process_gloo: None) -> None:
    """The split must buy the step back without moving a parameter between hyperparameter sets.

    A subgroup that lost the group's learning rate or weight decay would train the head on the
    optimizer's defaults instead of the configured schedule -- silently, and only under tensor
    parallelism.
    """
    model, optimizer = _mixed_optimizer(foreach=True)
    settings = {k: v for k, v in optimizer.param_groups[0].items() if k != "params"}

    split_mixed_param_groups(optimizer)

    assert [_kinds(group) for group in optimizer.param_groups] == [{"DTensor"}, {"Parameter"}]
    assert [{k: v for k, v in g.items() if k != "params"} for g in optimizer.param_groups] == [settings, settings]
    assert [id(p) for group in optimizer.param_groups for p in group["params"]] == [
        id(model.get_parameter(name)) for name in _MIXED_PARAMETERS
    ]
    _stepped(model, optimizer)  # the crash the split exists for


def test_splitting_leaves_an_already_uniform_optimizer_exactly_as_it_was() -> None:
    """Every run calls this, tensor-parallel or not, so anywhere it is not needed it must do nothing.

    Rebuilding the groups unconditionally would hand every single-device and DDP run new dictionaries
    -- and anything holding on to one, an LR scheduler above all, would then be writing into objects
    the optimizer no longer reads.
    """
    model = _mlp_models()["model"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
    groups, group = optimizer.param_groups, optimizer.param_groups[0]

    split_mixed_param_groups(optimizer)

    assert optimizer.param_groups is groups
    assert len(optimizer.param_groups) == 1
    assert optimizer.param_groups[0] is group


def test_the_split_optimizer_updates_exactly_as_an_unfused_one(single_process_gloo: None) -> None:
    """The claim that makes splitting the fix rather than a workaround: grouping is fusion, not math.

    ``foreach=False`` is the other way out of the crash and is the arithmetic reference here; it is
    not the fix, because it also unfuses the DTensor majority the strategy exists for, and the H200
    isolation measured its numerics moving.
    """
    split_model, split_optimizer = _mixed_optimizer(foreach=True)
    split_mixed_param_groups(split_optimizer)
    reference_model, reference_optimizer = _mixed_optimizer(foreach=False)
    initial = _local(reference_model.get_parameter("down.weight")).clone()

    _stepped(split_model, split_optimizer)
    _stepped(reference_model, reference_optimizer)

    assert not torch.equal(_local(reference_model.get_parameter("down.weight")), initial)  # a step really landed
    for name in _MIXED_PARAMETERS:
        updated = _local(split_model.get_parameter(name))
        assert torch.allclose(updated, _local(reference_model.get_parameter(name)), rtol=0.0, atol=1e-7)


def test_a_state_saved_from_a_split_optimizer_resumes_into_a_freshly_split_one(single_process_gloo: None) -> None:
    """The CLI splits before `restore_training_state`, so both ends of a resume have the same groups.

    Optimizer state is keyed by parameter name rather than by group, so the split changes nothing a
    checkpoint carries -- which is what lets the fix be unconditional instead of a checkpoint format
    change. Splitting after the load would instead hand torch a group count the saved state lacks.
    """
    strategy = TensorParallelStrategy(device="cpu", parallel_modules=_MIXED_PLAN)
    pairing = {"opt": ["model"]}
    model, optimizer = _mixed_optimizer(foreach=True)
    split_mixed_param_groups(optimizer)
    _stepped(model, optimizer)
    state = strategy.state_dict(OrderedDict(model=model), {"opt": optimizer}, pairing)
    assert set(state["optimizers"]["opt"]["state"]) == {f"model.{name}" for name in _MIXED_PARAMETERS}

    fresh_model, fresh_optimizer = _mixed_optimizer(foreach=True)
    split_mixed_param_groups(fresh_optimizer)
    strategy.load_state_dict(OrderedDict(model=fresh_model), {"opt": fresh_optimizer}, pairing, state)

    assert len(fresh_optimizer.param_groups) == 2
    for name in _MIXED_PARAMETERS:
        restored = fresh_optimizer.state[fresh_model.get_parameter(name)]
        assert torch.equal(_local(restored["exp_avg"]), _local(optimizer.state[model.get_parameter(name)]["exp_avg"]))
    _stepped(fresh_model, fresh_optimizer)  # the resumed optimizer still steps


# ---------------------------------------------------------------------------
# SyncBatchNorm conversion
# ---------------------------------------------------------------------------


class _BatchNormModel(torch.nn.Module):
    """A model carrying a nested BatchNorm layer, the shape the conversion targets."""

    def __init__(self) -> None:
        super().__init__()
        self.body = torch.nn.Sequential(torch.nn.Conv2d(2, 2, 1), torch.nn.BatchNorm2d(2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the body."""
        return self.body(x)


def _unreachable_conversion(
    models: "OrderedDict[str, torch.nn.Module]",
    device: str,
) -> "OrderedDict[str, torch.nn.Module]":
    """Stand in for the conversion in the tests that require wrap never to reach it."""
    raise AssertionError("wrap must not convert when sync_batchnorm is off")


def test_convert_sync_batchnorm_replaces_nested_layers_and_keeps_the_tensors() -> None:
    """Converted layers must reuse the parameter objects, or the pre-wrap rank-0 broadcast is discarded."""
    models: OrderedDict[str, torch.nn.Module] = OrderedDict(model=_BatchNormModel())
    weight = models["model"].get_parameter("body.1.weight")
    converted = distributed._convert_sync_batchnorm(models, "cuda:0")
    layer = converted["model"].get_submodule("body.1")
    assert isinstance(layer, torch.nn.SyncBatchNorm)
    assert layer.weight is weight


def test_convert_sync_batchnorm_keeps_the_activation_of_timms_fused_norm_act_layers() -> None:
    """Fused ``BatchNormAct2d`` layers from timm must keep running their activation after the conversion.

    torch's stock converter replaces it with a plain ``SyncBatchNorm`` whose forward never calls the
    fused activation, and the ``state_dict`` keys stay identical — the model silently trains without
    the activation. This pins the timm converter that keeps it.
    """
    fused = BatchNormAct2d(4, act_layer=torch.nn.ReLU)
    weight = fused.weight
    models: OrderedDict[str, torch.nn.Module] = OrderedDict(
        model=torch.nn.Sequential(fused, torch.nn.BatchNorm2d(4)),
    )
    converted = distributed._convert_sync_batchnorm(models, "cuda:0")
    sync_fused = converted["model"].get_submodule("0")
    assert isinstance(sync_fused, SyncBatchNormAct)
    assert isinstance(sync_fused, torch.nn.SyncBatchNorm)
    assert sync_fused.weight is weight
    assert type(converted["model"].get_submodule("1")) is torch.nn.SyncBatchNorm
    inputs = torch.linspace(-1.0, 1.0, 8).reshape(1, 4, 2, 1)
    assert inputs.min() < 0  # eval mode leaves the values untouched, so only the ReLU can clamp them
    assert sync_fused.eval()(inputs).min() >= 0


def test_convert_sync_batchnorm_is_idempotent_for_timms_fused_layers() -> None:
    """Converting twice must not undo the first conversion, which timm's raw converter would.

    ``SyncBatchNormAct`` subclasses ``torch.nn.SyncBatchNorm`` but not ``BatchNormAct2d``, so a second
    pass through timm's converter rebuilds it as a plain ``SyncBatchNorm`` and silently drops the fused
    activation while the ``state_dict`` keys stay identical. A model converted by hand, or wrapped a
    second time, must keep its activation.
    """
    models: OrderedDict[str, torch.nn.Module] = OrderedDict(
        model=torch.nn.Sequential(BatchNormAct2d(4, act_layer=torch.nn.ReLU)),
    )
    once = distributed._convert_sync_batchnorm(models, "cuda:0")
    converted_layer = once["model"].get_submodule("0")
    twice = distributed._convert_sync_batchnorm(once, "cuda:0")
    layer = twice["model"].get_submodule("0")
    assert layer is converted_layer  # the second pass left the already-converted layer alone
    assert isinstance(layer, SyncBatchNormAct)
    inputs = torch.linspace(-1.0, 1.0, 8).reshape(1, 4, 2, 1)
    assert inputs.min() < 0  # eval mode leaves the values untouched, so only the ReLU can clamp them
    assert layer.eval()(inputs).min() >= 0


def test_convert_sync_batchnorm_leaves_an_existing_sync_batch_norm_and_its_process_group_untouched() -> None:
    """A hand-converted layer must pass through as the very same object, process group included.

    Re-creating it would reset ``process_group`` to the default group, silently discarding a hand-built
    subgroup, and would drop everything else attached to the layer.
    """
    group = object()
    layer = torch.nn.SyncBatchNorm(4, process_group=group)
    converted = distributed._convert_sync_batchnorm(OrderedDict(model=layer), "cuda:0")
    assert converted["model"] is layer
    assert layer.process_group is group


def test_convert_sync_batchnorm_leaves_a_model_without_batch_norm_untouched() -> None:
    """A model with no ``BatchNorm`` must come out identical: every distributed run walks through this.

    The conversion is on by default, so models that have nothing to convert must keep their identity and
    their layer types — the strategies wrap whatever it returns.
    """
    model = torch.nn.Sequential(torch.nn.Linear(4, 2), torch.nn.LayerNorm(2))
    models: OrderedDict[str, torch.nn.Module] = OrderedDict(model=model)
    converted = distributed._convert_sync_batchnorm(models, "cuda:0")
    assert converted["model"] is model
    assert [type(child) for child in model] == [torch.nn.Linear, torch.nn.LayerNorm]


def test_convert_sync_batchnorm_returns_a_new_module_when_the_root_is_a_batch_norm() -> None:
    """A root BatchNorm cannot be converted in place, so the strategies must wrap the returned object."""
    root = torch.nn.BatchNorm1d(2)
    converted = distributed._convert_sync_batchnorm(OrderedDict(model=root), "cuda:0")
    assert isinstance(converted["model"], torch.nn.SyncBatchNorm)
    assert converted["model"] is not root


def test_convert_sync_batchnorm_skips_cpu_devices() -> None:
    """SyncBatchNorm's training forward rejects CPU input once a process group exists, even with one rank."""
    models: OrderedDict[str, torch.nn.Module] = OrderedDict(model=_BatchNormModel())
    assert distributed._convert_sync_batchnorm(models, "cpu") is models
    assert type(models["model"].get_submodule("body.1")) is torch.nn.BatchNorm2d


def test_ddp_wrap_wraps_the_conversion_result_and_converts_for_its_own_device(
    single_process_gloo: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """DDP must wrap what the conversion returned, and convert for the device the strategy trains on.

    A model that is itself a BatchNorm comes back as a new object, so wrapping the input would wrap a
    discarded module. The device decides whether the conversion runs at all, so a hardcoded one would
    convert (or skip) against hardware this rank does not train on.
    """
    converted = torch.nn.Linear(4, 2)
    seen: list[str] = []

    def _spy(
        models: "OrderedDict[str, torch.nn.Module]",
        device: str,
    ) -> "OrderedDict[str, torch.nn.Module]":
        """Record the device wrap passes down and hand back a different module."""
        seen.append(device)
        return OrderedDict(model=converted)

    monkeypatch.setattr(distributed, "_convert_sync_batchnorm", _spy)
    wrapped = DistributedDataParallelStrategy(device="cpu:0").wrap(_linear_models())
    assert wrapped["model"].get_submodule("module") is converted
    assert seen == ["cpu:0"]  # the strategy's own device, not a hardcoded "cpu"


def test_ddp_wrap_leaves_the_models_alone_when_sync_batchnorm_is_off(
    single_process_gloo: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The YAML off-switch (`_bind_: {sync_batchnorm: false}`) is the only way out, so it must really opt out."""
    monkeypatch.setattr(distributed, "_convert_sync_batchnorm", _unreachable_conversion)
    models = _linear_models()
    original = models["model"]
    wrapped = DistributedDataParallelStrategy(device="cpu", sync_batchnorm=False).wrap(models)
    assert wrapped["model"].get_submodule("module") is original


def test_fsdp2_wrap_converts_before_the_mesh_and_shards_the_converted_tree(
    single_process_gloo: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Conversion must come first: the mesh and the shard matching would otherwise see replaced modules.

    The device it runs for is the strategy's own; a hardcoded one would convert (or skip) against
    hardware this rank does not train on.
    """
    fsdp = pytest.importorskip("torch.distributed.fsdp")
    strategy = FullyShardedDataParallelStrategy(device="cpu:0", shard_modules=["block?"])
    converted = _block_models()
    seen: list[Any] = []

    def _spy(
        models: "OrderedDict[str, torch.nn.Module]",
        device: str,
    ) -> "OrderedDict[str, torch.nn.Module]":
        """Record the mesh state and device the conversion runs under, and hand back a different module tree."""
        seen.append((strategy._mesh, device))
        return converted

    monkeypatch.setattr(distributed, "_convert_sync_batchnorm", _spy)
    wrapped = strategy.wrap(_block_models())
    assert seen == [(None, "cpu:0")]  # ran before the mesh was derived, for the strategy's own device
    assert wrapped["model"] is converted["model"]
    assert isinstance(converted["model"].block0, fsdp.FSDPModule)


def test_fsdp2_wrap_leaves_the_models_alone_when_sync_batchnorm_is_off(
    single_process_gloo: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The off-switch must reach FSDP2 too, its field being a separate one from DDP's."""
    pytest.importorskip("torch.distributed.fsdp")
    monkeypatch.setattr(distributed, "_convert_sync_batchnorm", _unreachable_conversion)
    strategy = FullyShardedDataParallelStrategy(device="cpu", sync_batchnorm=False)
    models = _linear_models()
    assert strategy.wrap(models)["model"] is models["model"]


def test_single_device_wrap_never_converts_batch_norm() -> None:
    """A single device has no ranks to synchronize statistics across, so SyncBatchNorm is pure overhead.

    The device is the only thing the conversion gates on, so a non-CPU one here would convert if the
    single-device strategy ever grew the call. ``wrap`` never touches the device itself.
    """
    models: OrderedDict[str, torch.nn.Module] = OrderedDict(model=_BatchNormModel())
    wrapped = SingleDeviceStrategy(device="cuda:0").wrap(models)
    assert wrapped is models
    assert type(models["model"].get_submodule("body.1")) is torch.nn.BatchNorm2d
