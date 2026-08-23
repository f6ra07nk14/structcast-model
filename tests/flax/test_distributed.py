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
from structcast_model.flax.distributed import MODEL_AXIS, FlaxDistributedStrategy
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


def _spec(parameter: Any) -> str:
    """One parameter's `PartitionSpec` as a string; `spec` lives on `NamedSharding`, not `Sharding`."""
    sharding: Any = parameter[...].sharding
    return str(sharding.spec)


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
from structcast_model.flax.optimizers import unwrap_variables

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
    learner._training_step, {"donate_argnames": ("models", "optimizers")}
)
x = jnp.asarray([[1.0, 0.5, -0.5, 2.0], [0.0, 1.0, 1.0, -1.0], [2.0, 0.0, 1.0, 0.5], [-1.0, 1.0, 0.0, 1.0]] * 2)
y = jnp.asarray([[1.0, -1.0], [0.5, 0.25], [0.0, 1.0], [-0.5, 0.5]] * 2)
batch = strategy.shard_batch({"x": x, "y": y})

losses, accumulator = [], []
for step in range(6):
    losses.append(float(learner.training_step(**batch)["loss"]))
    acc_grads = unwrap_variables(learner.optimizers["optimizer"].opt_state).acc_grads
    accumulator.append([str(v.sharding.spec) for v in jax.tree.leaves(acc_grads)])
print(json.dumps({"losses": losses, "accumulator": accumulator}))
"""

FSDP_TP_SPEC_SCRIPT = """
import json
import jax
from flax import nnx
from structcast_model.flax.distributed import FlaxDistributedStrategy

# Leading dimensions of 2 and 6 divide the data axis but not the four devices of the whole mesh; 3
# divides neither.
strategy = FlaxDistributedStrategy(preset="fsdp_tp", model_devices=2, min_size=0)
models = nnx.Dict(
    two=nnx.Linear(2, 8, rngs=nnx.Rngs(0)),
    six=nnx.Linear(6, 8, rngs=nnx.Rngs(0)),
    odd=nnx.Linear(3, 8, rngs=nnx.Rngs(0)),
)
strategy.wrap({"model": models})
print(json.dumps({
    "mesh": dict(strategy.mesh.shape),
    "params": {jax.tree_util.keystr(p, simple=True, separator="."): str(v.sharding.spec)
               for p, v in jax.tree_util.tree_flatten_with_path(nnx.to_pure_dict(nnx.state(models, nnx.Param)))[0]},
}))
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


TP_SCRIPT = """
import json, sys
import jax, jax.numpy as jnp, optax
from flax import nnx
from structcast_model.flax.distributed import FlaxDistributedStrategy
from structcast_model.flax.utils import dot_general_out

preset, model_devices = sys.argv[1], int(sys.argv[2]) or None
model_axis_mode = sys.argv[3] if len(sys.argv) > 3 else "auto"
rules = [(r"^up\\.", "column"), (r"^down\\.", "row")] if preset in ("tp", "fsdp_tp") else None
if preset == "fsdp_tp":
    rules = rules + [(r".*", "fsdp")]

class MLP(nnx.Module):
    def __init__(self, rngs):
        self.up = nnx.Linear(8, 16, rngs=rngs)
        # The row-parallel layer names the sharding of its own output when the model axis is typed;
        # an out_sharding may name Explicit axes only, so the batch entry stays None while the data
        # axis is Auto -- which is the annotation a template written for a hybrid mesh carries.
        hook = {"dot_general": dot_general_out(None, None)} if model_axis_mode == "explicit" else {}
        self.down = nnx.Linear(16, 8, rngs=rngs, **hook)
        # Named by no rule: under the tp presets it must keep the sharding it was built with.
        self.head = nnx.Linear(8, 8, rngs=rngs)
    def __call__(self, x):
        return self.head(self.down(nnx.relu(self.up(x))))

strategy = FlaxDistributedStrategy(preset=preset, model_devices=model_devices, rules=rules, min_size=0,
                                   model_axis_mode=model_axis_mode)
model = MLP(nnx.Rngs(params=jax.random.key(0)))
strategy.wrap({"model": model})
optimizer = nnx.Optimizer(model, tx=optax.sgd(0.1), wrt=nnx.Param)
batch = strategy.shard_batch({"x": jnp.linspace(-1.0, 1.0, 64).reshape(8, 8),
                              "y": jnp.linspace(1.0, -1.0, 64).reshape(8, 8)})

def spec(array):
    # Canonical: JAX drops trailing replicated entries when it re-derives a spec, so P('model', None)
    # and P('model') are the same sharding and have to compare equal across a step.
    entries = list(array.sharding.spec)
    while entries and entries[-1] is None:
        entries.pop()
    return "P(" + ", ".join("None" if e is None else repr(e) for e in entries) + ")"

def specs():
    return {jax.tree_util.keystr(p, simple=True, separator="."): spec(v)
            for p, v in jax.tree_util.tree_flatten_with_path(nnx.to_pure_dict(nnx.state(model, nnx.Param)))[0]}

def loss_fn(model, x, y):
    return jnp.mean((model(x) - y) ** 2)

def step(model, optimizer, x, y):
    loss, grads = nnx.value_and_grad(loss_fn)(model, x, y)
    optimizer.update(model, grads)
    return loss

placed = specs()
# Through the strategy's own compile seam, which is how a run steps: an eager step is one program
# per operation, and the compiler re-places every output of every one of them.
compiled = strategy.compile(step, {})
losses = [float(compiled(model, optimizer, batch["x"], batch["y"])) for _ in range(3)]
print(json.dumps({
    "mesh": dict(strategy.mesh.shape),
    "axis_types": [t.name for t in strategy.mesh.axis_types],
    "params": placed,
    "trained": specs(),
    "batch": {k: str(v.sharding.spec) for k, v in batch.items()},
    # What one device actually holds: the proof the split factor is the data axis, not the mesh.
    "batch_shard": list(batch["x"].addressable_shards[0].data.shape),
    "losses": losses,
}))
"""


def test_the_strategy_satisfies_the_distributed_strategy_protocol() -> None:
    """The trainer and the checkpoint callbacks accept any strategy structurally, so this is the contract."""
    assert isinstance(FlaxDistributedStrategy(), DistributedStrategy)


def test_the_single_preset_builds_a_size_one_auto_mesh() -> None:
    """Single-device training is a one-device mesh, not a branch: the sharding path is always the same one.

    The axis type is asserted rather than left to `jax.make_mesh`, whose default has changed between
    jax versions: an Explicit data axis refuses ops that mix a sharded array with a replicated one,
    which is a trace failure for models that carry no sharding annotations at all.
    """
    strategy = FlaxDistributedStrategy(preset="single", device="cpu:0")

    assert strategy.mesh.shape == {"data": 1}
    assert strategy.mesh.axis_types == (jax.sharding.AxisType.Auto,)


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


def test_a_model_axis_tactic_is_refused_on_a_preset_without_one() -> None:
    """`column` and `row` name a mesh axis the one-dimensional presets do not build."""
    with pytest.raises(ValueError, match='splits a parameter along the "model" axis'):
        FlaxDistributedStrategy(preset="fsdp", rules=[(".*", "row")])


@pytest.mark.parametrize("preset", ["tp", "fsdp_tp"], ids=["tp", "fsdp_tp"])
def test_a_tensor_parallel_preset_without_rules_is_refused(preset: str) -> None:
    """The `tp` table is empty, and an empty table under these presets splits nothing at all.

    Every parameter would keep its construction sharding, every device would hold the whole model
    and run the whole batch, and the run would report success -- the one failure mode a preset named
    after tensor parallelism must not have.
    """
    with pytest.raises(ValueError, match="splits the layers its rules name"):
        FlaxDistributedStrategy(preset=preset, model_devices=1, rules=[])  # type: ignore[arg-type]  # runtime guard


def test_a_rule_matching_no_parameter_is_refused() -> None:
    """A typo'd rule leaves its layers on the preset's default placement and says nothing.

    The torch globs are refused the same way and for the same reason: matching nothing anywhere is
    never what the author meant, and the cost of the silence is a run that trains unsharded.
    """
    strategy = FlaxDistributedStrategy(preset="tp", rules=[(r"^fc\.", "column"), (r"^fcc\.", "row")])
    model, _ = _build()

    with pytest.raises(ValueError, match=r"matched no parameter"):
        strategy.wrap(OrderedDict(model=model))


def test_a_model_axis_option_is_refused_on_a_preset_without_one() -> None:
    """A model-axis knob bound to `dp` or `fsdp` would do nothing at all, which is worse than an error."""
    with pytest.raises(ValueError, match="model_devices and model_axis_mode"):
        FlaxDistributedStrategy(preset="fsdp", model_devices=2)

    # The counterpart: the data axis is the one axis every preset builds, so its own mode is
    # configurable wherever the model-axis knobs are refused.
    assert FlaxDistributedStrategy(preset="fsdp", data_axis_mode="explicit").mesh.axis_types == (
        jax.sharding.AxisType.Explicit,
    )


def test_the_fsdp_tp_preset_requires_a_model_axis_size() -> None:
    """Without it the data axis would be one device wide, and the preset would shard nothing on it."""
    with pytest.raises(ValueError, match="model_devices"):
        FlaxDistributedStrategy(preset="fsdp_tp", rules=[(".*", "fsdp")])


def test_a_model_axis_the_devices_do_not_divide_is_refused() -> None:
    """A mesh needs both axes to multiply out to the devices; the leftovers would be dropped ranks."""
    with pytest.raises(ValueError, match="must divide it"):
        FlaxDistributedStrategy(preset="tp", devices=1, model_devices=3, rules=[(".*", "column")])


def test_the_explicit_model_axis_mode_refuses_a_row_layer_without_its_own_dot_general() -> None:
    """Under an Explicit axis the compiler may not place a row-parallel result, so the layer must.

    The default `jax.lax.dot_general` cannot, and the failure it produces is a trace-time error deep
    inside the first step; the check moves it to configuration time and names the template line that
    fixes it. Only explicit mode checks: under the default Auto axis the compiler inserts the
    reduction itself, which is what lets a plain layer row-parallelize unchanged.
    """
    strategy = FlaxDistributedStrategy(preset="tp", model_axis_mode="explicit", rules=[(r"^fc\.", "row")])
    model, _ = _build()

    with pytest.raises(ValueError, match="dot_general_out") as error:
        strategy.wrap(OrderedDict(model=model))

    assert "'fc.kernel'" in str(error.value)
    # The line it hands out has to be one that then traces: an `out_sharding` may name Explicit axes
    # only, so under the Auto data axis of this strategy the batch entry is None, not "data".
    assert 'dot_general_out(None, None)"' in str(error.value)


def test_the_explicit_model_axis_mode_refuses_a_row_parameter_owned_by_no_hooked_layer() -> None:
    """Fail closed: a row-matched parameter whose owner cannot show a hook is refused with the rest.

    A parameter held in a container rather than a layer has no `dot_general` to read at all, and the
    check used to skip exactly those -- passing the run through with the one placement explicit mode
    exists to verify left unverified.
    """
    strategy = FlaxDistributedStrategy(preset="tp", model_axis_mode="explicit", rules=[(r"^weights\.", "row")])
    model = nnx.Dict(weights=nnx.List([nnx.Param(jnp.ones((4, 4)))]))

    with pytest.raises(ValueError, match="row-parallelizes") as error:
        strategy.wrap(OrderedDict(model=model))

    assert "'weights.0'" in str(error.value)


def test_the_tp_presets_leave_an_unmatched_parameter_on_its_construction_sharding() -> None:
    """A model template's own annotation must survive a strategy that says nothing about it.

    `dp` and `fsdp` place every parameter, replicating whatever no rule shards, which would silently
    overwrite the annotation; under the tp presets an unmatched parameter is not placed at all. The
    annotation asserted here is one no rule in the table could have produced, so its surviving is the
    only reason it can still be there. (A template that annotates its own parameters through
    `nnx.with_partitioning` is the one case that needs both axes typed: that initializer path is what
    refuses a mesh whose axes do not all have the same type.)
    """
    strategy = FlaxDistributedStrategy(
        preset="tp", data_axis_mode="explicit", model_axis_mode="explicit", rules=[(r"^up\.", "column")]
    )
    model = nnx.Dict(
        up=nnx.Linear(8, 16, rngs=nnx.Rngs(0)),
        head=nnx.Linear(
            8,
            8,
            kernel_init=nnx.with_partitioning(nnx.initializers.lecun_normal(), (MODEL_AXIS, None)),
            rngs=nnx.Rngs(0),
        ),
    )

    strategy.wrap(OrderedDict(model=model))

    assert strategy.mesh.axis_types == (jax.sharding.AxisType.Explicit, jax.sharding.AxisType.Explicit)
    assert _spec(model.head.kernel) == "P('model', None)"
    assert _spec(model.up.kernel) == "P(None, 'model')"


def test_the_auto_model_axis_mode_places_a_plain_row_layer(tmp_path: Path) -> None:
    """The default must not ask a model template for anything, which is why Auto is the default."""
    strategy = FlaxDistributedStrategy(preset="tp", rules=[(r"^fc\.", "row")])
    model, _ = _build()

    strategy.wrap(OrderedDict(model=model))

    assert strategy.mesh.axis_types == (jax.sharding.AxisType.Auto, jax.sharding.AxisType.Auto)
    assert _spec(model.fc.bias) == "P()"


def test_an_embedding_looks_up_a_sharded_batch_of_indices() -> None:
    """`nnx.Embed` gathers a replicated table with the sharded indices of the batch, and must trace.

    The gather is a `jnp.take` inside Flax's own module, so no model template can name the sharding
    of its output: while the data axis was Explicit this raised `ShardingTypeError: Use
    .at[...].get(out_sharding=)` and every template holding an embedding -- `SmallLanguageModel`
    among them -- was unrunnable under every preset. A plain model must train with no sharding
    annotations at all, which is what the Auto data axis buys and what this asserts.
    """
    strategy = FlaxDistributedStrategy(preset="single", device="cpu:0")
    embed = nnx.Embed(num_embeddings=8, features=4, rngs=nnx.Rngs(params=jax.random.key(0)))
    strategy.wrap(OrderedDict(model=embed))
    batch = strategy.shard_batch({"tokens": jnp.zeros((4, 3), dtype=jnp.int32)})

    embedded = strategy.compile(lambda model, tokens: model(tokens), {})(embed, batch["tokens"])

    assert embedded.shape == (4, 3, 4)


def test_the_explicit_data_axis_mode_types_the_data_axis_of_any_preset_and_fails_loud_again() -> None:
    """The opt-in has to restore exactly what the Auto default gave up, or it is not the old mode.

    An Explicit data axis is a type-system property, not a layout, so it applies to every preset --
    `single` here, which builds no model axis at all and therefore takes no `model_axis_mode`. What
    it buys is the refusal above: the very `nnx.Embed` lookup that traces under the Auto default
    raises here, because a typed axis demands an `out_sharding` at a meeting point inside Flax's own
    code. That is the point of the flag, and the reason it is not the default (`docs/adr/0022`).
    """
    strategy = FlaxDistributedStrategy(preset="single", device="cpu:0", data_axis_mode="explicit")
    embed = nnx.Embed(num_embeddings=8, features=4, rngs=nnx.Rngs(params=jax.random.key(0)))
    strategy.wrap(OrderedDict(model=embed))
    batch = strategy.shard_batch({"tokens": jnp.zeros((4, 3), dtype=jnp.int32)})

    assert strategy.mesh.axis_types == (jax.sharding.AxisType.Explicit,)
    # `ShardingTypeError` lives in `jax._src.core` and has no public alias, so the name is asserted
    # rather than imported: the message is the contract a template author reads either way.
    with pytest.raises(Exception, match=r"out_sharding") as error:
        strategy.compile(lambda model, tokens: model(tokens), {})(embed, batch["tokens"])

    assert type(error.value).__name__ == "ShardingTypeError"


def test_a_replicated_class_token_concatenates_onto_a_sharded_activation() -> None:
    """A learned token prepended to the batch mixes a replicated array with a sharded one, and must trace.

    This is `VisionTransformer.yaml`'s class token, whose `jax.numpy.concatenate` is a plain call in
    the generated forward: while the data axis was Explicit it raised `ShardingTypeError: All
    operands should have the same sharding`, because the token is replicated and the activation is
    split along the batch dimension. Neither operand is something a template can annotate, so the
    mesh has to accept the mix -- the compiler resharding the token is the right answer here.
    """
    strategy = FlaxDistributedStrategy(preset="single", device="cpu:0")
    model = nnx.Dict(class_token=nnx.Param(jnp.zeros((1, 1, 4))))
    strategy.wrap(OrderedDict(model=model))
    feature = strategy.shard_batch({"feature": jnp.ones((4, 3, 4))})["feature"]

    def prepend(model: nnx.Dict, feature: jax.Array) -> jax.Array:
        token = jnp.broadcast_to(model.class_token[...], (feature.shape[0], 1, feature.shape[-1]))
        return jnp.concatenate((token, feature), axis=1)

    assert strategy.compile(prepend, {})(model, feature).shape == (4, 4, 4)


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
    """A scalar entry has no batch dimension to split, and the error must name the entry.

    It must also name the divisor it used: the batch is split along the data axis alone, so a
    message pointing at "the mesh size" sends the reader of a two-axis run after the wrong number.
    """
    with pytest.raises(ValueError, match='"y"') as error:
        FlaxDistributedStrategy().shard_batch({"x": jnp.ones((4, 3)), "y": jnp.asarray(1.0)})

    assert 'devices of the "data" axis' in str(error.value)


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
        # along, so every op consuming it would meet `data` on two dimensions and pay a reshard.
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
    step on the compiled program is the same one, so the losses are bit-identical. The `MultiSteps`
    accumulator inside the optimizer state has to stay consistently sharded across the six steps
    too: a spec that drifted would retrace the step mid-run.
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


@pytest.fixture(scope="module")
def tp_reference(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """The same three steps on one device, as the yardstick every tensor-parallel mesh must match."""
    return _run_script(TP_SCRIPT, tmp_path_factory.mktemp("tp_reference"), "single", "0", devices=1)


def test_the_tp_preset_splits_the_layers_its_rules_name_and_pins_the_row_bias(tmp_path: Path) -> None:
    """The tactics carry the one rule a table cannot express: a row-parallel bias stays whole.

    A bias split along the model axis -- or added once per shard -- is counted as many times as the
    axis is wide by the reduction a row-parallel layer ends in, and the run still reports a plausible
    loss. The column bias is the opposite case: it belongs to the output dimension and splits with it.
    """
    result = _run_script(TP_SCRIPT, tmp_path, "tp", "0", devices=4)

    assert result["mesh"] == {"data": 1, "model": 4}
    assert result["params"]["up.kernel"] == "P(None, 'model')"
    assert result["params"]["up.bias"] == "P('model')"
    assert result["params"]["down.kernel"] == "P('model')"
    assert result["params"]["down.bias"] == "P()"
    # And the split survives training: a compiled step that gave the parameters back replicated
    # would undo the preset after one update, at no cost to the losses that are compared below.
    assert {k: v for k, v in result["trained"].items() if not k.startswith("head")} == {
        k: v for k, v in result["params"].items() if not k.startswith("head")
    }


def test_the_tp_preset_splits_the_batch_along_the_data_axis_only(tmp_path: Path) -> None:
    """Every device of one model-axis group runs the same items, so the mesh size is not the divisor.

    With a data axis of one, an eight-row batch stays eight rows on every device; dividing by the
    four devices of the mesh would hand each of them a quarter of it and quietly train on a fraction.
    """
    result = _run_script(TP_SCRIPT, tmp_path, "tp", "0", devices=4)

    assert result["batch"] == {"x": "P('data',)", "y": "P('data',)"}
    assert result["batch_shard"] == [8, 8]


@pytest.mark.parametrize(("preset", "model_devices"), [("tp", "0"), ("fsdp_tp", "2")], ids=["tp-4", "fsdp_tp-2x2"])
def test_a_tensor_parallel_mesh_trains_to_the_losses_of_one_device(
    preset: str, model_devices: str, tmp_path: Path, tp_reference: dict[str, Any]
) -> None:
    """A split model must compute what the whole one does, or a tensor-parallel run cannot be trusted.

    Splitting changes the order the products are reduced in, so the losses agree to the tolerance
    ADR-0014 records for a mesh change rather than bitwise. The `(2, 2)` combination is the same
    assertion with the fsdp rules composed behind the tensor-parallel ones.
    """
    result = _run_script(TP_SCRIPT, tmp_path, preset, model_devices, devices=4)

    assert result["losses"] == pytest.approx(tp_reference["losses"], rel=1e-6)


def test_a_hybrid_mesh_trains_an_annotated_model_to_the_losses_of_one_device(
    tmp_path: Path, tp_reference: dict[str, Any]
) -> None:
    """`(Auto data, Explicit model)` is what the two flags exist for, and it has to compute correctly.

    A template that names the sharding of its own outputs wants the model axis typed -- the mode that
    makes an unnamed row-parallel result an error rather than a compiler guess -- and wants the data
    axis left Auto all the same, because the ops that break under a typed data axis sit inside Flax
    and no annotation reaches them. jax meshes may mix axis types, so nothing at the strategy level
    forbids the combination; this runs it on a four-wide model axis, where a row-parallel layer really
    does hold partial sums, and asserts the same losses as one device.
    """
    result = _run_script(TP_SCRIPT, tmp_path, "tp", "0", "explicit", devices=4)

    assert result["axis_types"] == ["Auto", "Explicit"]
    assert result["losses"] == pytest.approx(tp_reference["losses"], rel=1e-6)


def test_the_fsdp_tactic_divides_by_the_data_axis_and_not_the_whole_mesh(tmp_path: Path) -> None:
    """The `fsdp` tactic shards along the data axis, so the data axis is what has to divide the rows.

    Dividing by the mesh instead leaves every parameter whose leading dimension the data axis alone
    would have split replicated -- a preset that says it shards and quietly does not, on exactly the
    two-axis mesh the combination exists for.
    """
    result = _run_script(FSDP_TP_SPEC_SCRIPT, tmp_path, devices=4)

    assert result["mesh"] == {"data": 2, "model": 2}
    assert result["params"]["two.kernel"] == "P('data', None)"
    assert result["params"]["six.kernel"] == "P('data', None)"
    # Three rows divide neither axis, so this one really is replicated rather than failing the run.
    assert result["params"]["odd.kernel"] == "P()"


def test_the_fsdp_tp_preset_shards_on_the_axis_the_first_matching_rule_names(tmp_path: Path) -> None:
    """Precedence is the rule table's own order, so a parameter lands on one axis or the other.

    The tensor-parallel rules come first and claim their layers; the catch-all `fsdp` rule behind
    them takes everything left over onto the data axis -- which is what makes the combination two
    axes rather than two strategies fighting over the same parameter.
    """
    result = _run_script(TP_SCRIPT, tmp_path, "fsdp_tp", "2", devices=4)

    assert result["mesh"] == {"data": 2, "model": 2}
    assert result["params"]["up.kernel"] == "P(None, 'model')"
    assert result["params"]["down.bias"] == "P()"
    assert result["params"]["head.kernel"] == "P('data')"
    assert result["batch"] == {"x": "P('data',)", "y": "P('data',)"}
    assert result["batch_shard"] == [4, 8]


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
