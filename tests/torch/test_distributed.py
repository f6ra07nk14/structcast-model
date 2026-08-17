"""Tests for the distributed strategies and the sync gate."""

from collections import OrderedDict
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any, cast

import pytest

from structcast_model.torch.distributed import (
    DistributedDataParallelStrategy,
    DistributedStrategy,
    FullyShardedDataParallelStrategy,
    SingleDeviceStrategy,
    matched_shard_modules,
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
    assert strategy.grad_scaler_creator is torch.amp.GradScaler


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
