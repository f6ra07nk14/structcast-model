"""Tests for the distributed strategies and the sync gate."""

from collections import OrderedDict
from typing import Any

import pytest

from structcast_model.torch.distributed import (
    DistributedDataParallelStrategy,
    DistributedStrategy,
    FullyShardedDataParallelStrategy,
    SingleDeviceStrategy,
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


def test_sync_gate_disarms_ddp_reducer_until_exit(single_process_gloo: None) -> None:
    """An unarmed gate must stop DDP from all-reducing the next backward, and restore afterwards."""
    ddp = torch.nn.parallel.DistributedDataParallel(torch.nn.Linear(2, 2))
    with sync_gate(ddp, armed=False):
        assert ddp.require_backward_grad_sync is False
    assert ddp.require_backward_grad_sync is True


def test_sync_gate_armed_leaves_ddp_untouched(single_process_gloo: None) -> None:
    """An armed gate must leave the DDP reducer live so the final backward synchronizes."""
    ddp = torch.nn.parallel.DistributedDataParallel(torch.nn.Linear(2, 2))
    with sync_gate(ddp, armed=True):
        assert ddp.require_backward_grad_sync is True


def test_sync_gate_toggles_fsdp2_gradient_sync(single_process_gloo: None) -> None:
    """The FSDP2 branch must route through set_requires_gradient_sync and restore it on exit."""
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
    assert set(states["models"]["model"]) == {"model.weight", "model.bias"} or set(states["models"]["model"]) == {
        "weight",
        "bias",
    }
    assert not any(key.startswith("_orig_mod.") for key in states["models"]["model"])


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


def test_fsdp2_strategy_requires_fully_shard(monkeypatch: pytest.MonkeyPatch) -> None:
    """Selecting FSDP2 on a torch without fully_shard must fail with an actionable message."""
    # The lazy-import shim hides module privates, so patch the globals the class actually reads.
    monkeypatch.setitem(FullyShardedDataParallelStrategy.__post_init__.__globals__, "_fully_shard", None)
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
