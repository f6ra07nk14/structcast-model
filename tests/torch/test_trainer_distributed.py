"""Two-rank gloo tests for the cross-rank semantics no single-process test can observe.

Each test spawns 2 CPU worker processes, has them write their observations into ``tmp_path``
and asserts on the collected files, so the assertions cover what actually crossed the ranks:
deferred gradient all-reduce, rank-0 authority, tracker averaging, checkpoint broadcast,
``found_inf`` propagation through the DTensor dispatcher, and what a tensor-parallel split computes.
"""

from collections import OrderedDict
from collections.abc import Callable
from datetime import timedelta
import pathlib
import traceback
from typing import Any

import pytest
import torch.multiprocessing as mp

from structcast_model.torch.distributed import (
    DistributedDataParallelStrategy,
    FullyShardedDataParallelStrategy,
    FullyShardedTensorParallelStrategy,
    SingleDeviceStrategy,
    TensorParallelStrategy,
    sync_gate,
)
from structcast_model.torch.trainer import TorchTracker
import torch
import torch.distributed as dist

WORLD_SIZE = 2


def _init(rank: int, world_size: int, init_file: str) -> None:
    """Join the 2-process gloo group backing one spawned test."""
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=60),
    )


def _report(result_dir: str, rank: int, payload: dict[str, Any]) -> None:
    """Hand one rank's observations back to the parent process."""
    torch.save(payload, str(pathlib.Path(result_dir) / f"rank{rank}.pt"))


def _spawn(
    worker: Callable[..., None],
    tmp_path: pathlib.Path,
    world_size: int = WORLD_SIZE,
) -> list[dict[str, Any]]:
    """Run *worker* on *world_size* gloo ranks and return their reported payloads, rank-ordered."""
    mp.spawn(worker, args=(world_size, str(tmp_path / "dist_init"), str(tmp_path)), nprocs=world_size, join=True)
    return [torch.load(str(tmp_path / f"rank{r}.pt"), weights_only=False) for r in range(world_size)]


# ---------------------------------------------------------------------------
# sync_gate accumulation semantics
# ---------------------------------------------------------------------------


def _sync_gate_worker(rank: int, world_size: int, init_file: str, result_dir: str) -> None:
    """Run one unarmed and one armed micro-step, reporting the gradient after each."""
    _init(rank, world_size, init_file)
    try:
        model = torch.nn.Linear(2, 1, bias=False)
        with torch.no_grad():
            model.weight.zero_()
        ddp = torch.nn.parallel.DistributedDataParallel(model, device_ids=None)
        x = torch.tensor([[1.0, 0.0]]) if rank == 0 else torch.tensor([[0.0, 1.0]])
        target = torch.tensor([[1.0]]) if rank == 0 else torch.tensor([[2.0]])

        sync_gate(ddp, False)
        out = ddp(x)
        ((out - target) ** 2).sum().backward()
        assert model.weight.grad is not None, "the unarmed backward must still accumulate a local gradient"
        unarmed = model.weight.grad.detach().clone()

        sync_gate(ddp, True)
        out = ddp(x)
        ((out - target) ** 2).sum().backward()
        assert model.weight.grad is not None
        armed = model.weight.grad.detach().clone()

        _report(result_dir, rank, {"unarmed": unarmed, "armed": armed})
    except Exception:
        traceback.print_exc()
        raise
    finally:
        dist.destroy_process_group()


def test_sync_gate_defers_ddp_all_reduce_until_armed(tmp_path: pathlib.Path) -> None:
    """The unarmed gate must keep gradients local; the armed backward averages what accumulated."""
    rank0, rank1 = _spawn(_sync_gate_worker, tmp_path)

    assert torch.equal(rank0["unarmed"], torch.tensor([[-2.0, 0.0]]))
    assert torch.equal(rank1["unarmed"], torch.tensor([[0.0, -4.0]]))
    assert not torch.equal(rank0["unarmed"], rank1["unarmed"])

    # Accumulated per-rank gradients are [-4, 0] and [0, -8]; the armed backward all-reduces the AVG.
    expected = torch.tensor([[-2.0, -4.0]])
    assert torch.equal(rank0["armed"], expected)
    assert torch.equal(rank1["armed"], expected)


def _fsdp2_sync_gate_worker(rank: int, world_size: int, init_file: str, result_dir: str) -> None:
    """Run one unarmed and one armed micro-step under fully_shard, reporting the gradient state."""
    _init(rank, world_size, init_file)
    try:
        torch.manual_seed(0)
        model = torch.nn.Linear(2, 1, bias=False)
        with torch.no_grad():
            model.weight.zero_()
        strategy = FullyShardedDataParallelStrategy(device="cpu")
        wrapped = strategy.wrap(OrderedDict(model=model))["model"]
        x = torch.tensor([[1.0, 0.0]]) if rank == 0 else torch.tensor([[0.0, 1.0]])
        target = torch.tensor([[1.0]]) if rank == 0 else torch.tensor([[2.0]])

        sync_gate(wrapped, False)
        out = wrapped(x)
        ((out - target) ** 2).sum().backward()
        # With sync deferred, no reduce-scatter ran, so the sharded parameter has no gradient yet;
        # the accumulating gradients live on the unsharded parameters until the armed backward.
        unarmed_grad_missing = wrapped.weight.grad is None

        sync_gate(wrapped, True)
        out = wrapped(x)
        ((out - target) ** 2).sum().backward()
        grad = wrapped.get_parameter("weight").grad
        assert grad is not None, "the armed backward must reduce-scatter a gradient onto the sharded parameter"
        armed = (
            grad.full_tensor().detach().clone()
            if isinstance(grad, torch.distributed.tensor.DTensor)
            else grad.detach().clone()
        )

        _report(result_dir, rank, {"unarmed_grad_missing": unarmed_grad_missing, "armed": armed})
    except Exception:
        traceback.print_exc()
        raise
    finally:
        dist.destroy_process_group()


def test_sync_gate_defers_fsdp2_reduce_scatter_until_armed(tmp_path: pathlib.Path) -> None:
    """The gate's flag must survive past the forward: FSDP2 reads it at backward time.

    A forward-scoped restore would re-enable reduce-scatter before any backward ran, making the
    unarmed backward reduce immediately (a sharded gradient would appear after micro-step one) —
    the bug this test pins.
    """
    pytest.importorskip("torch.distributed.fsdp")
    rank0, rank1 = _spawn(_fsdp2_sync_gate_worker, tmp_path)

    assert rank0["unarmed_grad_missing"] is True
    assert rank1["unarmed_grad_missing"] is True

    # Accumulated per-rank gradients are [-4, 0] and [0, -8]; the armed backward reduces the AVG.
    expected = torch.tensor([[-2.0, -4.0]])
    assert torch.equal(rank0["armed"], expected)
    assert torch.equal(rank1["armed"], expected)


# ---------------------------------------------------------------------------
# sync_initial_weights
# ---------------------------------------------------------------------------

_PATTERN = torch.tensor([[1.0, 2.0], [3.0, 4.0]])


def _sync_initial_weights_worker(rank: int, world_size: int, init_file: str, result_dir: str) -> None:
    """Start from divergent weights and report what ``sync_initial_weights`` left behind."""
    _init(rank, world_size, init_file)
    try:
        model = torch.nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            model.weight.copy_(_PATTERN if rank == 0 else torch.full((2, 2), 123.0))
        DistributedDataParallelStrategy(device="cpu").sync_initial_weights({"model": model})
        _report(result_dir, rank, {"weight": model.weight.detach().clone()})
    except Exception:
        traceback.print_exc()
        raise
    finally:
        dist.destroy_process_group()


def test_sync_initial_weights_makes_rank0_authoritative(tmp_path: pathlib.Path) -> None:
    """Rank 0's initializer decides the starting weights; every other rank's are overwritten."""
    rank0, rank1 = _spawn(_sync_initial_weights_worker, tmp_path)

    assert torch.equal(rank0["weight"], _PATTERN)
    assert torch.equal(rank1["weight"], _PATTERN)


# ---------------------------------------------------------------------------
# TorchTracker
# ---------------------------------------------------------------------------


def _tracker_worker(rank: int, world_size: int, init_file: str, result_dir: str) -> None:
    """Feed one rank-specific loss value into a distributed tracker and report the average."""
    _init(rank, world_size, init_file)
    try:
        tracker = TorchTracker.from_criteria(["loss"], distributed=True)
        _report(result_dir, rank, tracker(loss=torch.tensor(float(rank))))
    except Exception:
        traceback.print_exc()
        raise
    finally:
        dist.destroy_process_group()


def test_torch_tracker_all_reduce_averages_across_ranks(tmp_path: pathlib.Path) -> None:
    """A distributed tracker must report the cross-rank mean, not the rank-local value."""
    rank0, rank1 = _spawn(_tracker_worker, tmp_path)

    assert rank0 == {"loss": 0.5}
    assert rank1 == {"loss": 0.5}


# ---------------------------------------------------------------------------
# Checkpoint round trips
# ---------------------------------------------------------------------------


def _train_one_step(model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> None:
    """Run one forward/backward/step on data identical across ranks."""
    x = torch.ones(3, 4)
    loss = ((model(x) - torch.zeros(3, 2)) ** 2).sum()
    loss.backward()
    optimizer.step()


def _full_weight(model: torch.nn.Module) -> torch.Tensor:
    """Return the (possibly sharded) linear weight as a plain full tensor."""
    weight = model.get_parameter("weight" if hasattr(model, "weight") else "module.weight")
    full = weight.full_tensor() if isinstance(weight, torch.distributed.tensor.DTensor) else weight
    return full.detach().clone()


def _round_trip_worker(rank: int, world_size: int, init_file: str, result_dir: str, strategy: Any) -> None:
    """Save a trained state on rank 0 only, then load it back into freshly initialized models."""
    _init(rank, world_size, init_file)
    try:
        torch.manual_seed(0)
        models = strategy.wrap(OrderedDict(model=torch.nn.Linear(4, 2)))
        optimizer = torch.optim.SGD(models["model"].parameters(), lr=0.1, momentum=0.9)
        _train_one_step(models["model"], optimizer)
        saved = strategy.state_dict(models, {"opt": optimizer}, {"opt": ["model"]})
        trained = _full_weight(models["model"])

        torch.manual_seed(7 + rank)
        fresh = strategy.wrap(OrderedDict(model=torch.nn.Linear(4, 2)))
        fresh_optimizer = torch.optim.SGD(fresh["model"].parameters(), lr=0.1, momentum=0.9)
        before_load = _full_weight(fresh["model"])
        strategy.load_state_dict(fresh, {"opt": fresh_optimizer}, {"opt": ["model"]}, saved if rank == 0 else None)

        _report(
            result_dir,
            rank,
            {
                "model_state_empty": saved["models"]["model"] == {},
                "trained": trained,
                "before_load": before_load,
                "restored": _full_weight(fresh["model"]),
            },
        )
    except Exception:
        traceback.print_exc()
        raise
    finally:
        dist.destroy_process_group()


def _ddp_round_trip_worker(rank: int, world_size: int, init_file: str, result_dir: str) -> None:
    """Round-trip a DDP-wrapped model's checkpoint."""
    _round_trip_worker(rank, world_size, init_file, result_dir, DistributedDataParallelStrategy(device="cpu"))


def _fsdp2_round_trip_worker(rank: int, world_size: int, init_file: str, result_dir: str) -> None:
    """Round-trip a ``fully_shard``ed model's checkpoint."""
    _round_trip_worker(rank, world_size, init_file, result_dir, FullyShardedDataParallelStrategy(device="cpu"))


def _assert_round_trip(rank0: dict[str, Any], rank1: dict[str, Any]) -> None:
    """Assert rank 0 owns the saved tensors and both ranks end up holding them again."""
    assert rank0["model_state_empty"] is False
    assert rank1["model_state_empty"] is True
    assert not torch.equal(rank0["trained"], rank0["before_load"])
    assert torch.equal(rank0["restored"], rank0["trained"])
    assert torch.equal(rank1["restored"], rank0["trained"])


def test_ddp_strategy_state_round_trips_across_two_ranks(tmp_path: pathlib.Path) -> None:
    """DDP checkpoints are rank-0-only on save and must be broadcast back to every rank on load."""
    _assert_round_trip(*_spawn(_ddp_round_trip_worker, tmp_path))


def test_fsdp2_strategy_trains_and_round_trips_across_two_ranks(tmp_path: pathlib.Path) -> None:
    """Sharded parameters must be gathered into a full rank-0 checkpoint and re-sharded on load.

    Runs without hiding CUDA: the strategy builds its mesh from its own `device` field, so a
    CUDA-enabled build training on CPU must not follow the accelerator (regression test).
    """
    pytest.importorskip("torch.distributed.fsdp")
    _assert_round_trip(*_spawn(_fsdp2_round_trip_worker, tmp_path))


class _BlockModel(torch.nn.Module):
    """A two-block model, the shape ``shard_modules`` splits into per-block groups."""

    def __init__(self) -> None:
        super().__init__()
        self.block0 = torch.nn.Linear(4, 4)
        self.block1 = torch.nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run both blocks in sequence."""
        return self.block1(self.block0(x))


def _per_block_round_trip_worker(rank: int, world_size: int, init_file: str, result_dir: str) -> None:
    """Round-trip a per-block sharded model: every group must gather into one rank-0 checkpoint."""
    _init(rank, world_size, init_file)
    try:
        torch.manual_seed(0)
        strategy = FullyShardedDataParallelStrategy(device="cpu", shard_modules=["block?"])
        models = strategy.wrap(OrderedDict(model=_BlockModel()))
        optimizer = torch.optim.SGD(models["model"].parameters(), lr=0.1, momentum=0.9)
        _train_one_step(models["model"], optimizer)
        saved = strategy.state_dict(models, {"opt": optimizer}, {"opt": ["model"]})
        trained = _full_weight(models["model"].get_submodule("block1"))

        torch.manual_seed(7 + rank)
        fresh = strategy.wrap(OrderedDict(model=_BlockModel()))
        fresh_optimizer = torch.optim.SGD(fresh["model"].parameters(), lr=0.1, momentum=0.9)
        before_load = _full_weight(fresh["model"].get_submodule("block1"))
        strategy.load_state_dict(fresh, {"opt": fresh_optimizer}, {"opt": ["model"]}, saved if rank == 0 else None)

        _report(
            result_dir,
            rank,
            {
                "model_state_empty": saved["models"]["model"] == {},
                "keys": sorted(saved["models"]["model"]),
                "trained": trained,
                "before_load": before_load,
                "restored": _full_weight(fresh["model"].get_submodule("block1")),
            },
        )
    except Exception:
        traceback.print_exc()
        raise
    finally:
        dist.destroy_process_group()


def test_fsdp2_per_block_sharding_round_trips_across_two_ranks(tmp_path: pathlib.Path) -> None:
    """Per-block groups must not change the checkpoint: same keys, gathered on rank 0, re-sharded on load."""
    pytest.importorskip("torch.distributed.fsdp")
    rank0, rank1 = _spawn(_per_block_round_trip_worker, tmp_path)

    assert rank0["keys"] == ["block0.bias", "block0.weight", "block1.bias", "block1.weight"]
    _assert_round_trip(rank0, rank1)


# ---------------------------------------------------------------------------
# Tensor parallelism
# ---------------------------------------------------------------------------


class _MLPModel(torch.nn.Module):
    """A column-parallel layer feeding a row-parallel one: the pair a plan is written in."""

    def __init__(self) -> None:
        super().__init__()
        self.up = torch.nn.Linear(4, 8)
        self.down = torch.nn.Linear(8, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the pair."""
        return self.down(torch.relu(self.up(x)))


_PLAN = [("up", "column"), ("down", "row")]
_TP_INPUT = torch.linspace(-1.0, 1.0, 12).reshape(3, 4)
"""The batch every rank of a tensor-parallel group must see: one model, one batch (ADR-0022)."""

_TP_TARGET = torch.linspace(1.0, -1.0, 12).reshape(3, 4)


def _single_process_step() -> tuple[float, dict[str, torch.Tensor]]:
    """Run the same model, batch and step on one process, as the yardstick the ranks must match."""
    torch.manual_seed(0)
    model = _MLPModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    loss = ((model(_TP_INPUT) - _TP_TARGET) ** 2).sum()
    loss.backward()
    optimizer.step()
    return float(loss.detach()), {name: p.detach().clone() for name, p in model.named_parameters()}


def _full(parameter: torch.Tensor) -> torch.Tensor:
    """Gather a (possibly sharded) parameter into a plain tensor; collective, so every rank calls it."""
    if isinstance(parameter, torch.distributed.tensor.DTensor):
        parameter = parameter.full_tensor()
    return parameter.detach().clone()


def _tensor_parallel_step(strategy: Any, rank: int, result_dir: str) -> None:
    """Train one step under *strategy* and report the loss and the gathered parameters."""
    # Both ranks build from the same seed, which is what sync_initial_weights does in a real run.
    torch.manual_seed(0)
    model = strategy.wrap(OrderedDict(model=_MLPModel()))["model"]
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    loss = ((model(_TP_INPUT) - _TP_TARGET) ** 2).sum()
    loss.backward()
    optimizer.step()
    _report(
        result_dir,
        rank,
        {
            "loss": float(loss.detach()),
            "data": (strategy.data_rank, strategy.data_world_size),
            # One entry per mesh axis: the combination places on the data axis and the model axis at
            # once, so the tensor-parallel placement is the last one either way.
            "placements": {name: [repr(x) for x in p.placements] for name, p in model.named_parameters()},
            "parameters": {name: _full(p) for name, p in model.named_parameters()},
        },
    )


def _tensor_parallel_worker(rank: int, world_size: int, init_file: str, result_dir: str) -> None:
    """Split a two-layer MLP over both ranks and train one step on the identical batch."""
    _init(rank, world_size, init_file)
    try:
        _tensor_parallel_step(TensorParallelStrategy(device="cpu", parallel_modules=_PLAN), rank, result_dir)
    except Exception:
        traceback.print_exc()
        raise
    finally:
        dist.destroy_process_group()


def _fsdp2_tensor_parallel_worker(rank: int, world_size: int, init_file: str, result_dir: str) -> None:
    """The same step with ``fully_shard`` on the data axis of a (1, 2) mesh."""
    _init(rank, world_size, init_file)
    try:
        _tensor_parallel_step(
            FullyShardedTensorParallelStrategy(device="cpu", tensor_parallel_size=2, parallel_modules=_PLAN),
            rank,
            result_dir,
        )
    except Exception:
        traceback.print_exc()
        raise
    finally:
        dist.destroy_process_group()


def _assert_matches_one_process(results: list[dict[str, Any]]) -> None:
    """Assert every rank trained the model one process would have, and holds the same weights."""
    loss, expected = _single_process_step()
    for result in results:
        assert result["loss"] == pytest.approx(loss, rel=1e-5)
        for name, value in expected.items():
            # Loose: the split reduces the products in a different order, and a step's worth of that
            # difference is what a tensor-parallel run is allowed to cost (ADR-0014's tolerance).
            assert torch.allclose(result["parameters"][name], value, atol=1e-6), name


def test_tensor_parallel_trains_the_model_one_process_would_have(tmp_path: pathlib.Path) -> None:
    """A split model must compute what the whole one does, or the whole strategy is a silent defect.

    The placements are asserted beside the numbers because replicating everything would pass the
    numeric comparison too: what makes it tensor parallelism is that each rank holds half of each
    weight, and that the row-parallel bias is *not* split -- a split one is added once per shard and
    counted twice by the all-reduce that follows.
    """
    results = _spawn(_tensor_parallel_worker, tmp_path)

    for result in results:
        assert result["data"] == (0, 1)
        assert result["placements"]["up.weight"] == ["Shard(dim=0)"]
        assert result["placements"]["down.weight"] == ["Shard(dim=1)"]
        assert result["placements"]["down.bias"] == ["Replicate()"]
    _assert_matches_one_process(results)


def test_fsdp2_tensor_parallel_trains_the_model_one_process_would_have(tmp_path: pathlib.Path) -> None:
    """The two wrappers must compose: parallelize first, ``fully_shard`` the result on the data axis.

    Two CPU ranks only reach a ``(1, 2)`` mesh, so this pins the composition -- the mesh, the order,
    and one training step through both -- and not a data axis wider than one. A real ``(2, 2)`` needs
    four ranks and is left to the GPU validation the ADR asks for.
    """
    pytest.importorskip("torch.distributed.fsdp")
    results = _spawn(_fsdp2_tensor_parallel_worker, tmp_path)

    for result in results:
        assert result["data"] == (0, 1)
        assert result["placements"]["up.weight"][-1] == "Shard(dim=0)"
        assert result["placements"]["down.weight"][-1] == "Shard(dim=1)"
    _assert_matches_one_process(results)


def _data_coordinates_worker(rank: int, world_size: int, init_file: str, result_dir: str) -> None:
    """Report what each strategy calls this rank's data slice, without wrapping anything."""
    _init(rank, world_size, init_file)
    try:
        strategies: dict[str, Any] = {
            "single": SingleDeviceStrategy(device="cpu"),
            "ddp": DistributedDataParallelStrategy(device="cpu"),
            "fsdp2": FullyShardedDataParallelStrategy(device="cpu"),
            "tp": TensorParallelStrategy(device="cpu", parallel_modules=_PLAN),
            "fsdp2_tp": FullyShardedTensorParallelStrategy(
                device="cpu", tensor_parallel_size=2, parallel_modules=_PLAN
            ),
        }
        _report(result_dir, rank, {n: (s.data_rank, s.data_world_size) for n, s in strategies.items()})
    except Exception:
        traceback.print_exc()
        raise
    finally:
        dist.destroy_process_group()


def test_data_coordinates_follow_the_mesh_not_the_global_rank(tmp_path: pathlib.Path) -> None:
    """The CLI seeds from these and publishes them to the loader, so each strategy must report its own.

    Under the replicating strategies a rank is its own data slice; under the tensor-parallel ones the
    two ranks share one model and therefore one slice, which is exactly what the global rank cannot
    express.
    """
    pytest.importorskip("torch.distributed.fsdp")
    rank0, rank1 = _spawn(_data_coordinates_worker, tmp_path)

    assert rank0 == {"single": (0, 1), "ddp": (0, 2), "fsdp2": (0, 2), "tp": (0, 1), "fsdp2_tp": (0, 1)}
    assert rank1 == {"single": (0, 1), "ddp": (1, 2), "fsdp2": (1, 2), "tp": (0, 1), "fsdp2_tp": (0, 1)}


# ---------------------------------------------------------------------------
# fp16 gradient scaler under FSDP2
# ---------------------------------------------------------------------------


def _grad_scaler_worker(rank: int, world_size: int, init_file: str, result_dir: str) -> None:
    """Poison only rank 1's gradient shard and report whether this rank's step was skipped."""
    _init(rank, world_size, init_file)
    try:
        torch.manual_seed(0)
        models = FullyShardedDataParallelStrategy(device="cpu").wrap(OrderedDict(model=torch.nn.Linear(4, 2)))
        model = models["model"]
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        scaler = torch.amp.GradScaler(device="cpu", init_scale=2.0**8)

        before = _full_weight(model)
        scaler.scale(((model(torch.ones(3, 4)) - torch.zeros(3, 2)) ** 2).sum()).backward()
        if rank == 1:
            grad = model.get_parameter("weight").grad
            assert isinstance(grad, torch.distributed.tensor.DTensor), "fully_shard must keep the gradient sharded"
            local_grad = grad.to_local()
            assert local_grad.numel() > 0, "rank 1 holds no shard of the weight to poison"
            local_grad.mul_(float("inf"))

        scale_before = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()

        _report(
            result_dir,
            rank,
            {
                "step_skipped": torch.equal(_full_weight(model), before),
                "scale_before": scale_before,
                "scale_after": scaler.get_scale(),
            },
        )
    except Exception:
        traceback.print_exc()
        raise
    finally:
        dist.destroy_process_group()


def test_fsdp2_fp16_grad_scaler_synchronizes_found_inf(tmp_path: pathlib.Path) -> None:
    """An inf on one rank's shard must skip the step and halve the scale on *every* rank."""
    pytest.importorskip("torch.distributed.fsdp")
    for rank_result in _spawn(_grad_scaler_worker, tmp_path):
        assert rank_result["step_skipped"] is True
        assert rank_result["scale_before"] == 2.0**8
        assert rank_result["scale_after"] == 2.0**7
