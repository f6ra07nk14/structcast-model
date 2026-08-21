"""Tests for the Flax distributed strategy.

Everything a single device can prove runs in process. The rest -- what a preset actually shards, and
that a run gives the same losses on four devices as on one -- needs more devices than a JAX process
can gain after it started, so those tests re-run the work in a subprocess that sets
``jax_num_cpu_devices`` before importing anything, the way ``tests/torch`` spawns worker processes.
"""

from collections import OrderedDict
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import jax
import jax.numpy as jnp
import optax
import pytest

from flax import nnx
from structcast_model.builders.flax import FlaxBuilder, FlaxLearnerBuilder
from structcast_model.flax.distributed import FlaxDistributedStrategy
from structcast_model.torch.distributed import DistributedStrategy
from tests import FIXTURES_DIR

CFG_DIR = FIXTURES_DIR / "cfg" / "flax"
TIMEOUT = 300
"""Seconds a subprocess script may take; importing JAX and tracing a step is a few of them."""


@pytest.fixture(autouse=True)
def _clear_mesh() -> Any:
    """Unset the mesh a strategy activated, so it does not leak into unrelated tests.

    ``jax.set_mesh`` is a process-wide setter, and a strategy is expected to keep its mesh active
    for the whole run, so the test is what has to clean up after it.
    """
    yield
    jax.set_mesh(None)


class _Model(nnx.Module):
    """A linear layer, a batch norm and a dropout: parameters, statistics and RNG state in one model."""

    def __init__(self, features: int = 4, *, rngs: nnx.Rngs) -> None:
        """Build the layers."""
        self.fc = nnx.Linear(features, 2, rngs=rngs)
        self.bn = nnx.BatchNorm(2, rngs=rngs)
        self.dropout = nnx.Dropout(0.5, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        """Run the layers in training mode."""
        return self.dropout(self.bn(self.fc(x), use_running_average=False), deterministic=False)


def _build() -> tuple[_Model, nnx.Optimizer]:
    """Build a model and the optimizer that owns it, in the order a run does."""
    model = _Model(rngs=nnx.Rngs(params=jax.random.key(0), dropout=jax.random.key(1)))
    return model, nnx.Optimizer(model, tx=optax.adam(learning_rate=0.1), wrt=nnx.Param)


def _step_count(optimizer: Any) -> int:
    """Read an optimizer's update count, which decides the bias correction of its next update."""
    return int(jnp.asarray(nnx.state(optimizer).step[...]))


def _leaves(obj: Any) -> list[jax.Array]:
    """Read an nnx object's state as a flat list of arrays, in a stable order."""
    return jax.tree.leaves(nnx.to_pure_dict(nnx.state(obj)))


def _run_script(source: str, tmp_path: Path, *args: str, devices: int) -> dict[str, Any]:
    """Run *source* in a subprocess configured for *devices* CPU devices and parse its JSON output."""
    script = tmp_path / "script.py"
    script.write_text(f"import jax\njax.config.update('jax_num_cpu_devices', {devices})\n{source}")
    result = subprocess.run(
        [sys.executable, str(script), *args], capture_output=True, text=True, timeout=TIMEOUT, check=False
    )
    assert result.returncode == 0, result.stderr
    return dict(json.loads(result.stdout.splitlines()[-1]))


SHARDING_SCRIPT = """
import json, sys
import jax, optax
from flax import nnx
from structcast_model.flax.distributed import FlaxDistributedStrategy
from structcast_model.flax.optimizers import unwrap_variables

strategy = FlaxDistributedStrategy(preset=sys.argv[1], min_size=2**20)
# 1024x512 floats are 2 MiB and divide by four; the 1023-row kernel does not divide; the wide
# kernel's largest dimension is not the one FSDP shards; the 8x8 kernel is 256 bytes.
models = nnx.Dict(
    divisible=nnx.Linear(1024, 512, rngs=nnx.Rngs(0)),
    odd=nnx.Linear(1023, 512, rngs=nnx.Rngs(0)),
    wide=nnx.Linear(512, 1024, rngs=nnx.Rngs(0)),
    tiny=nnx.Linear(8, 8, rngs=nnx.Rngs(0)),
)
strategy.wrap({"model": models})
optimizer = nnx.Optimizer(models, tx=optax.adam(learning_rate=0.1), wrt=nnx.Param)

def specs(tree):
    return {jax.tree_util.keystr(p, simple=True, separator="."): str(v.sharding.spec)
            for p, v in jax.tree_util.tree_flatten_with_path(tree)[0]}

batch = strategy.shard_batch({"x": jax.numpy.ones((8, 1024)), "y": jax.numpy.ones((8,))})
print(json.dumps({
    "mesh": strategy.mesh.size,
    "params": specs(nnx.to_pure_dict(nnx.state(models, nnx.Param))),
    "optimizer": specs(unwrap_variables(optimizer.opt_state)[0].mu),
    "batch": {k: str(v.sharding.spec) for k, v in batch.items()},
}))
"""

PARITY_SCRIPT = """
import json, sys
from importlib.util import module_from_spec, spec_from_file_location
import jax, jax.numpy as jnp
from flax import nnx
from structcast_model.flax.distributed import FlaxDistributedStrategy

def load(path, name):
    spec = spec_from_file_location(name, path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

directory, preset = sys.argv[1], sys.argv[2]
strategy = FlaxDistributedStrategy(preset=preset, min_size=0)
model = load(directory + "/model.py", "generated_model").Model(rngs=nnx.Rngs(params=jax.random.key(0)))
strategy.wrap({"model": model})
learner = load(directory + "/learner.py", "generated_learner").Learner(model)
learner._training_step = strategy.compile(
    learner._training_step,
    {"static_argnames": "need_update", "donate_argnames": ("models", "optimizers", "acc_grads")},
)
x = jnp.asarray([[1.0, 0.5, -0.5, 2.0], [0.0, 1.0, 1.0, -1.0], [2.0, 0.0, 1.0, 0.5], [-1.0, 1.0, 0.0, 1.0]] * 2)
y = jnp.asarray([[1.0, -1.0], [0.5, 0.25], [0.0, 1.0], [-0.5, 0.5]] * 2)
batch = strategy.shard_batch({"x": x, "y": y})

losses, accumulator = [], []
for step in range(6):
    learner.need_update = step % 2 == 1
    losses.append(float(learner.training_step(**batch)["loss"]))
    accumulator.append([str(v.sharding.spec) for v in jax.tree.leaves(learner._acc_grads["optimizer"])])
print(json.dumps({"losses": losses, "accumulator": accumulator}))
"""

STATE_PRELUDE = """
import json, pickle, sys
import jax, jax.numpy as jnp, optax
from flax import nnx
from structcast_model.flax.distributed import FlaxDistributedStrategy
from structcast_model.flax.optimizers import unwrap_variables

class Model(nnx.Module):
    def __init__(self, seed):
        rngs = nnx.Rngs(params=jax.random.key(seed), dropout=jax.random.key(seed + 1))
        self.fc = nnx.Linear(1024, 4, rngs=rngs)
        self.bn = nnx.BatchNorm(4, rngs=rngs)
        self.dropout = nnx.Dropout(0.5, rngs=rngs)

def build(seed):
    model = Model(seed)
    FlaxDistributedStrategy(preset="fsdp", min_size=0).wrap({"model": model})
    # A scheduled learning rate so the optimizer carries a count of its own beside nnx's step.
    tx = optax.adam(learning_rate=optax.linear_schedule(0.1, 0.0, 10))
    return model, nnx.Optimizer(model, tx=tx, wrt=nnx.Param)

def fingerprint(model, optimizer):
    kernel = model.fc.kernel[...]
    return {
        "params": [v.tolist() for v in jax.tree.leaves(nnx.to_pure_dict(nnx.state(model, nnx.Param)))],
        "batch_stats": [v.tolist() for v in jax.tree.leaves(nnx.to_pure_dict(nnx.state(model, nnx.BatchStat)))],
        # Raw key data, so a key restored as plain uint32 would still be comparable -- and typed,
        # so the run's next draw comes from the stream the saved run was on.
        "key": jax.random.key_data(model.dropout.rngs.key[...]).tolist(),
        "typed_key": bool(jnp.issubdtype(model.dropout.rngs.key[...].dtype, jax.dtypes.prng_key)),
        "opt_state": [jnp.asarray(v).tolist() for v in jax.tree.leaves(unwrap_variables(optimizer.opt_state))],
        "step": int(jnp.asarray(nnx.state(optimizer).step[...])),
        "kernel_spec": str(kernel.sharding.spec),
        "kernel_devices": kernel.sharding.num_devices,
    }
"""

SAVE_SCRIPT = (
    STATE_PRELUDE
    + """
model, optimizer = build(0)
# Move every kind of state off its initial value: parameters and the optimizer's moments through an
# update, the batch statistics through a training-mode forward, the dropout key by hand.
optimizer.update(model, jax.tree.map(jnp.ones_like, nnx.state(model, nnx.Param)))
model.bn(model.fc(jnp.ones((8, 1024))), use_running_average=False)
model.dropout.rngs.key[...] = jax.random.fold_in(jax.random.key(1), 7)

with open(sys.argv[1], "wb") as handle:
    pickle.dump(FlaxDistributedStrategy(preset="fsdp", min_size=0).state_dict({"model": model},
                                                                             {"optimizer": optimizer}), handle)
print(json.dumps(fingerprint(model, optimizer)))
"""
)

RESTORE_SCRIPT = (
    STATE_PRELUDE
    + """
strategy = FlaxDistributedStrategy(preset="fsdp", min_size=0)
model, optimizer = build(2)
with open(sys.argv[1], "rb") as handle:
    strategy.load_state_dict({"model": model}, {"optimizer": optimizer}, None, pickle.load(handle))
print(json.dumps(fingerprint(model, optimizer)))
"""
)


def test_the_strategy_satisfies_the_distributed_strategy_protocol() -> None:
    """The trainer and the checkpoint callbacks accept any strategy structurally, so this is the contract."""
    assert isinstance(FlaxDistributedStrategy(), DistributedStrategy)


def test_the_single_preset_builds_a_size_one_explicit_mesh() -> None:
    """Single-device training is a one-device mesh, not a branch: the sharding path is always the same one."""
    strategy = FlaxDistributedStrategy(preset="single", device="cpu:0")

    assert strategy.mesh.shape == {"data": 1}
    assert strategy.mesh.axis_types == (jax.sharding.AxisType.Explicit,)


def test_an_unknown_preset_is_refused() -> None:
    """A typo'd preset must fail at configuration time, not silently train unsharded."""
    with pytest.raises(ValueError, match="Unknown preset"):
        FlaxDistributedStrategy(preset="zero2")  # type: ignore[arg-type]  # the point is the runtime guard


def test_an_unknown_tactic_is_refused() -> None:
    """A custom rule table naming a tactic that does not exist must fail before any model is placed."""
    with pytest.raises(ValueError, match="Unknown sharding tactic"):
        FlaxDistributedStrategy(rules=[(".*", "zero1")])


def test_a_device_count_outside_the_available_range_is_refused() -> None:
    """A negative count silently drops devices from the tail, so the run would train on fewer than asked."""
    with pytest.raises(ValueError, match="must be between 1 and"):
        FlaxDistributedStrategy(preset="dp", devices=-1)


def test_wrap_returns_the_same_model_objects() -> None:
    """The step closures capture the modules handed to the learner, so wrap must not replace them."""
    strategy = FlaxDistributedStrategy()
    model, _ = _build()
    models: OrderedDict[str, nnx.Module] = OrderedDict(model=model)

    wrapped = strategy.wrap(models)

    assert wrapped is models
    assert wrapped["model"] is model
    assert wrapped["model"].fc is model.fc


def test_shard_batch_commits_every_entry_to_the_mesh() -> None:
    """Uncommitted inputs would place the step's computation by chance, not by the mesh."""
    strategy = FlaxDistributedStrategy()

    batch = strategy.shard_batch({"x": jnp.ones((4, 3)), "y": jnp.ones((4,))})

    assert all(value.committed for value in batch.values())
    assert all(value.sharding.mesh is strategy.mesh for value in batch.values())


def test_shard_batch_refuses_an_entry_the_mesh_cannot_split() -> None:
    """A scalar entry has no batch dimension to split, and the error must name the entry."""
    with pytest.raises(ValueError, match='"y"'):
        FlaxDistributedStrategy().shard_batch({"x": jnp.ones((4, 3)), "y": jnp.asarray(1.0)})


def test_state_dict_carries_the_full_state_to_host_memory() -> None:
    """Parameters alone would resume a different run: batch statistics and RNG state travel too."""
    model, optimizer = _build()

    states = FlaxDistributedStrategy().state_dict({"model": model}, {"optimizer": optimizer})

    assert set(states) == {"models", "optimizers"}
    assert set(states["models"]["model"]) == {"fc", "bn", "dropout"}
    assert set(states["models"]["model"]["bn"]) == {"bias", "scale", "mean", "var"}
    # The typed RNG key travels as its raw key data, which is what a host-side format can hold.
    assert states["models"]["model"]["dropout"]["rngs"]["key"].dtype == jnp.uint32
    assert all(not isinstance(leaf, jax.Array) for leaf in jax.tree.leaves(states))


def test_a_state_round_trips_into_rebuilt_models_and_optimizers() -> None:
    """Resuming rebuilds every object from configuration, so the load must restore into new instances."""
    strategy = FlaxDistributedStrategy()
    model, optimizer = _build()
    optimizer.update(model, jax.tree.map(jnp.ones_like, nnx.state(model, nnx.Param)))
    states = strategy.state_dict({"model": model}, {"optimizer": optimizer})
    states["meta"] = {"epoch": 3}

    restored_model, restored_optimizer = _build()
    returned = strategy.load_state_dict({"model": restored_model}, {"optimizer": restored_optimizer}, None, states)

    assert returned["meta"] == {"epoch": 3}
    assert all(jnp.array_equal(a, b) for a, b in zip(_leaves(model), _leaves(restored_model), strict=True))
    assert all(jnp.array_equal(a, b) for a, b in zip(_leaves(optimizer), _leaves(restored_optimizer), strict=True))
    # The optimizer's step count decides the bias correction of the next update, so it must survive.
    assert _step_count(restored_optimizer) == 1


def test_a_restored_rng_key_stays_a_typed_key_and_keeps_drawing_the_same_stream() -> None:
    """A key restored as raw uint32 data would break the first `nnx.Dropout` call after a resume."""
    strategy = FlaxDistributedStrategy()
    model, _ = _build()
    model.dropout.rngs.key[...] = jax.random.fold_in(jax.random.key(1), 7)
    expected = jax.random.normal(model.dropout.rngs.key[...], (3,))

    restored, _ = _build()
    strategy.load_state_dict({"model": restored}, {}, None, strategy.state_dict({"model": model}))

    key = restored.dropout.rngs.key[...]
    assert jnp.issubdtype(key.dtype, jax.dtypes.prng_key)
    assert jnp.array_equal(jax.random.normal(key, (3,)), expected)


def test_load_state_dict_refuses_a_missing_state() -> None:
    """A resume without a state must stop the run, not start it from random weights."""
    with pytest.raises(ValueError, match="required to resume"):
        FlaxDistributedStrategy().load_state_dict({}, {}, None, None)


# ---------------------------------------------------------------------------
# Multi-device behavior, in subprocesses that own their device count
# ---------------------------------------------------------------------------


def test_the_dp_preset_replicates_every_parameter_and_splits_the_batch(tmp_path: Path) -> None:
    """Data parallelism is exactly that: whole parameters everywhere, one slice of the batch each."""
    result = _run_script(SHARDING_SCRIPT, tmp_path, "dp", devices=4)

    assert result["mesh"] == 4
    assert set(result["params"].values()) == {"P()"}
    assert result["batch"] == {"x": "P('data',)", "y": "P('data',)"}


def test_the_fsdp_preset_shards_the_leading_dimension_of_the_large_parameters(tmp_path: Path) -> None:
    """Sharding is worth a collective only for the big parameters whose rows the mesh divides.

    The optimizer is built after `wrap`, and its moments must land on the parameters' shardings: a
    replicated moment tree would undo the memory saving the preset exists for.
    """
    result = _run_script(SHARDING_SCRIPT, tmp_path, "fsdp", devices=4)

    assert result["params"] == {
        "divisible.kernel": "P('data', None)",
        # Sharded on dim 0 even though dim 1 is larger: dim 1 is the axis the batch is already split
        # along, and a result sharded twice on `data` is a trace-time error.
        "wide.kernel": "P('data', None)",
        # Every bias is one-dimensional, the 8x8 kernel is far below the cutoff, and 1023 rows do not
        # divide by four: all of them stay replicated instead of failing the run.
        "divisible.bias": "P()",
        "odd.kernel": "P()",
        "odd.bias": "P()",
        "wide.bias": "P()",
        "tiny.kernel": "P()",
        "tiny.bias": "P()",
    }
    assert result["optimizer"] == result["params"]


@pytest.mark.parametrize("preset", ["dp", "fsdp"])
def test_a_generated_learner_trains_to_the_same_losses_on_four_devices(preset: str, tmp_path: Path) -> None:
    """A run must not depend on how many devices it was given, or a sharded run cannot be trusted.

    The first step differs in the last bits because the reduction order differs; from the second
    step on the compiled program is the same one, so the losses are bit-identical. The gradient
    accumulator's spec has to stay put across the `need_update` flips too: a spec that drifted would
    retrace the step on every flip.
    """
    FlaxBuilder.from_path(CFG_DIR / "Linear.yaml")()(tmp_path / "model.py")
    FlaxLearnerBuilder.from_path(CFG_DIR / "LinearLearner.yaml")(parameters={"DEFAULT": {"accumulate_gradients": 2}})(
        tmp_path / "learner.py"
    )

    one = _run_script(PARITY_SCRIPT, tmp_path, str(tmp_path), preset, devices=1)
    four = _run_script(PARITY_SCRIPT, tmp_path, str(tmp_path), preset, devices=4)

    # Steps 0 and 1 are one accumulation window and report the same untouched-parameter loss.
    assert four["losses"] == pytest.approx(one["losses"], rel=1e-6)
    assert four["losses"][2:] == one["losses"][2:]
    assert four["accumulator"] == [four["accumulator"][0]] * 6


def _assert_same_state(saved: dict[str, Any], restored: dict[str, Any], *, devices: int) -> None:
    """Assert *restored* holds the very state *saved* recorded, placed on *devices* devices."""
    # Every kind of state moved off its initialization first, or "restored == saved" proves nothing.
    assert saved["step"] == 1
    assert saved["batch_stats"] != [[0.0] * 4, [1.0] * 4]
    for entry in ("params", "batch_stats", "key", "opt_state", "step"):
        # Bitwise: these are the numbers the next step reads, and a resharding round trip must not
        # round any of them. `step` and the schedule count inside `opt_state` decide the next
        # update's bias correction and learning rate, so they travel with the moments.
        assert restored[entry] == saved[entry], entry
    assert restored["typed_key"], "the dropout key came back as raw data and would break the next draw"
    # Placement follows the live run, not the checkpoint: the state itself carries no topology.
    # The spec is the preset's either way -- how many devices it spreads over is what changed.
    assert restored["kernel_spec"] == "P('data', None)"
    assert restored["kernel_devices"] == devices


def test_a_state_saved_on_four_devices_restores_on_one(tmp_path: Path) -> None:
    """Checkpoints outlive the machine that wrote them, so a host-memory state must be topology-free.

    The mirror of the test below: a run that shrank its device count keeps every number, and the
    sharded arrays are gathered onto the one device that is left.
    """
    states_path = tmp_path / "states.pkl"
    saved = _run_script(SAVE_SCRIPT, tmp_path, str(states_path), devices=4)
    assert saved["kernel_devices"] == 4

    restored = _run_script(RESTORE_SCRIPT, tmp_path, str(states_path), devices=1)

    _assert_same_state(saved, restored, devices=1)


def test_a_state_restored_on_four_devices_lands_on_the_live_run_s_sharding(tmp_path: Path) -> None:
    """A checkpoint holds host memory and the strategy owns placement, so a restore must reshard.

    Restoring the saved arrays as they are would leave every parameter on one device, undoing the
    memory saving the preset exists for -- and nothing downstream would notice, since the losses
    come out the same either way.
    """
    states_path = tmp_path / "states.pkl"
    saved = _run_script(SAVE_SCRIPT, tmp_path, str(states_path), devices=1)
    assert saved["kernel_devices"] == 1

    restored = _run_script(RESTORE_SCRIPT, tmp_path, str(states_path), devices=4)

    _assert_same_state(saved, restored, devices=4)
