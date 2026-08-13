"""Tests for the distributed strategies and the sync gate."""

from collections import OrderedDict
from typing import Any

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


def test_sync_gate_is_a_null_context_for_plain_modules() -> None:
    """Plain modules have no gradient sync to gate, so the gate must be a no-op either way."""
    model = torch.nn.Linear(2, 2)
    with sync_gate(model, armed=True):
        pass
    with sync_gate(model, armed=False):
        assert not hasattr(model, "require_backward_grad_sync")


def test_sync_gate_sets_the_ddp_flag_and_leaves_it_for_the_next_gate(single_process_gloo: None) -> None:
    """The gate sets the sync flag on entry and must NOT restore it on exit.

    FSDP2 reads its flag at backward time, which happens after the gated forward exits; a
    forward-scoped restore would re-enable gradient sync before any backward ran. The next gate on
    the same module overwrites the flag instead.
    """
    ddp = torch.nn.parallel.DistributedDataParallel(torch.nn.Linear(2, 2))
    with sync_gate(ddp, armed=False):
        assert ddp.require_backward_grad_sync is False
    assert ddp.require_backward_grad_sync is False
    with sync_gate(ddp, armed=True):
        assert ddp.require_backward_grad_sync is True
    assert ddp.require_backward_grad_sync is True


def test_sync_gate_sets_fsdp2_gradient_sync_without_restoring(single_process_gloo: None) -> None:
    """The FSDP2 branch must route through set_requires_gradient_sync and leave the flag in place."""
    fsdp = pytest.importorskip("torch.distributed.fsdp")
    model = fsdp.fully_shard(torch.nn.Linear(2, 2))
    calls: list[bool] = []
    original = model.set_requires_gradient_sync

    def _recording(value: bool, **kwargs: Any) -> None:
        calls.append(value)
        original(value, **kwargs)

    model.set_requires_gradient_sync = _recording
    with sync_gate(model, armed=False):
        assert calls == [False]
    assert calls == [False]
    with sync_gate(model, armed=True):
        pass
    assert calls == [False, True]


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


def test_single_device_state_dict_strips_the_compile_wrapper() -> None:
    """Checkpoints must stay loadable no matter whether the model was compiled when saved."""
    strategy = SingleDeviceStrategy(device="cpu")
    models = _linear_models()
    compiled = OrderedDict(model=torch.compile(models["model"]))
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
    assert torch.equal(fresh_models["model"].weight, models["model"].weight)
    assert fresh_optimizer.param_groups[0]["lr"] == 0.125
    assert returned is state


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
    before = models["model"].weight.clone()
    strategy.sync_initial_weights(models)
    assert torch.equal(models["model"].weight, before)


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
    assert torch.equal(fresh["model"].module.weight, wrapped["model"].module.weight)
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
    reference = models["model"].weight.clone()
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


def _group_size(module: torch.nn.Module) -> int:
    """Number of parameters in *module*'s own ``fully_shard`` group.

    FSDP2 exposes no public view of a group's membership, and membership is the whole point of
    per-block sharding: the root must hold only what no matched block claimed.
    """
    return len(module._get_fsdp_state()._fsdp_param_group.fsdp_params)


def test_matched_shard_modules_keeps_named_modules_order() -> None:
    """The wrap reverses this order to shard descendants first, so ancestors must come first here."""
    matched = matched_shard_modules(_block_models(), ["block0", "block0.1", "block1"])
    assert [path for path, _ in matched["model"]] == ["block0", "block0.1", "block1"]


def test_matched_shard_modules_sees_through_the_compile_wrapper() -> None:
    """torch.compile prefixes every path with '_orig_mod.'; patterns must not have to know that."""
    models: OrderedDict[str, torch.nn.Module] = OrderedDict(model=torch.compile(_BlockModel()))
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
    compiled: OrderedDict[str, torch.nn.Module] = OrderedDict(model=torch.compile(_BlockModel()))
    assert "_orig_mod" not in [path for path, _ in matched_shard_modules(compiled, ["*"])["model"]]


def test_fsdp2_per_block_wrap_refuses_a_tie_across_sibling_blocks(single_process_gloo: None) -> None:
    """fully_shard replaces each group's parameters, so a tie split across groups silently diverges."""
    pytest.importorskip("torch.distributed.fsdp")
    models = _block_models()
    models["model"].block0[1].weight = models["model"].block0[0].weight
    strategy = FullyShardedDataParallelStrategy(device="cpu", shard_modules=["block0.0", "block0.1"])
    with pytest.raises(RuntimeError, match="Tied parameter"):
        strategy.wrap(models)
    # The guard must run before any sharding: a half-sharded model cannot be recovered from.
    assert type(models["model"].block0[0].weight) is torch.nn.Parameter


def test_fsdp2_per_block_wrap_refuses_a_tie_with_an_unmatched_module(single_process_gloo: None) -> None:
    """A tie to a module outside every pattern lands in the root group, which is a different group."""
    pytest.importorskip("torch.distributed.fsdp")
    models = _block_models()
    models["model"].head.weight = models["model"].block1.weight
    strategy = FullyShardedDataParallelStrategy(device="cpu", shard_modules=["block1"])
    with pytest.raises(RuntimeError, match="Tied parameter"):
        strategy.wrap(models)


def test_fsdp2_per_block_wrap_allows_a_tie_inside_one_block(single_process_gloo: None) -> None:
    """A tie both of whose ends land in the same group is sharded once and stays tied."""
    pytest.importorskip("torch.distributed.fsdp")
    models = _block_models()
    models["model"].block0[1].weight = models["model"].block0[0].weight
    strategy = FullyShardedDataParallelStrategy(device="cpu", shard_modules=["block0"])
    wrapped = strategy.wrap(models)["model"]
    assert wrapped.block0[1].weight is wrapped.block0[0].weight


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
    with sync_gate(wrapped, armed=False):
        pass
    assert wrapped.block0._get_fsdp_state()._fsdp_param_group.reduce_grads is False
