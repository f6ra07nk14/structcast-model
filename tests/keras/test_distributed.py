"""Tests for the Keras distributed strategy.

Two kinds of test live here. The validation matrix runs in process, against whichever backend the
session resolved Keras on -- a rejected cell is rejected by that backend, so each one is asserted in
the lane that can reach it, and the three lanes of the verification matrix cover all of them.

Everything else needs devices or ranks a process cannot gain after Keras started: JAX fixes its
device count from `XLA_FLAGS` at import, TensorFlow refuses a logical-device configuration once its
runtime is initialized, and torch ranks are separate processes. Those tests therefore run a script
in a subprocess that sets its own `KERAS_BACKEND` before importing anything, the way
`tests/flax/test_distributed.py` re-runs its work for a device count -- which also makes them
independent of the lane they are collected in.
"""

from __future__ import annotations

from collections import OrderedDict
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np
import pytest

import keras
from structcast_model.builders.keras import KerasBuilder, KerasLearnerBuilder
from structcast_model.keras.distributed import REJECTED, KerasDistributedStrategy
from tests import FIXTURES_DIR

CFG_DIR = FIXTURES_DIR / "cfg" / "keras"
BACKEND = keras.backend.backend()
TIMEOUT = 600
"""Seconds a subprocess script may take; importing a framework and tracing a step is a few of them."""

JAX_DEVICES = {"backend": "jax", "XLA_FLAGS": "--xla_force_host_platform_device_count=8"}
"""How a JAX subprocess is started: eight CPU devices, fixed before JAX is imported."""


@pytest.fixture(scope="module")
def generated(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Build the linear model and learner fixtures once, for the subprocess scripts to load."""
    directory = tmp_path_factory.mktemp("generated")
    KerasBuilder.from_path(CFG_DIR / "Linear.yaml")()(directory / "model.py")
    KerasLearnerBuilder.from_path(CFG_DIR / "LinearLearner.yaml")()(directory / "learner.py")
    return directory


def _run(source: str, tmp_path: Path, *args: str, backend: str, **environment: str) -> dict[str, Any]:
    """Run *source* in a subprocess pinned to *backend* and parse the JSON it prints last."""
    script = tmp_path / f"{backend}_script.py"
    script.write_text(source)
    result = subprocess.run(
        [sys.executable, str(script), *args],
        capture_output=True,
        text=True,
        timeout=TIMEOUT,
        check=False,
        # The lane's own backend must not leak in: the script picks the one it is testing.
        env={**os.environ, "KERAS_BACKEND": backend, **environment},
    )
    assert result.returncode == 0, result.stderr
    return dict(json.loads(result.stdout.splitlines()[-1]))


# ---------------------------------------------------------------------------
# Validation: what a strategy refuses before a model exists
# ---------------------------------------------------------------------------


def test_an_unknown_preset_is_refused() -> None:
    """A typo'd preset must fail at configuration time, not silently train on one device."""
    with pytest.raises(ValueError, match="Unknown preset"):
        KerasDistributedStrategy(preset="zero2")  # type: ignore[arg-type]  # the point is the runtime guard


def test_an_unknown_tactic_is_refused() -> None:
    """A custom rule table naming a tactic that does not exist must fail before any model is built."""
    with pytest.raises(ValueError, match="Unknown sharding tactic"):
        KerasDistributedStrategy(preset="fsdp", rules=[(".*", "zero1")])


def test_rules_given_to_a_preset_that_replicates_are_refused() -> None:
    """Only fsdp shards anything, so rules handed to another preset would be silently ignored."""
    with pytest.raises(ValueError, match="Sharding rules only decide"):
        KerasDistributedStrategy(preset="dp", rules=[(".*", "fsdp")])


@pytest.mark.skipif(BACKEND == "jax", reason="fsdp is supported on the jax backend, and asserted below.")
def test_fsdp_is_refused_on_a_backend_that_cannot_shard() -> None:
    """The two rejected cells fail silently otherwise, so each is refused with its own reason.

    TensorFlow would replicate every variable through a `keras.distribution` prototype that does
    nothing, and torch FSDP2 would leave the Keras variables pointing at the parameters it replaced:
    both train on and report success, which is why the preset is refused rather than approximated.
    A refusal is only useful if it says which of the two it is and what to run instead, so the
    reason is asserted by its substance rather than against the constant it was raised from.
    """
    with pytest.raises(ValueError, match="is not available") as error:
        KerasDistributedStrategy(preset="fsdp")

    message = str(error.value)
    assert message == REJECTED[("fsdp", BACKEND)]
    if BACKEND == "tensorflow":
        assert "distribute_value" in message
    else:
        assert "keras-team/keras#23418" in message
        assert "caches" in message
    assert 'Use the "dp" preset' in message


@pytest.mark.skipif(BACKEND != "jax", reason="fsdp is only available on the jax backend.")
def test_fsdp_is_accepted_on_the_jax_backend() -> None:
    """The one supported cell of the sharding column must construct, or the matrix says nothing."""
    assert KerasDistributedStrategy(preset="fsdp").preset == "fsdp"


@pytest.mark.skipif(BACKEND != "torch", reason="Only the torch backend takes its ranks from the launcher.")
def test_a_device_count_is_refused_on_torch() -> None:
    """Torch ranks come from torchrun, so a count bound here would be quietly ignored."""
    with pytest.raises(ValueError, match="takes its number of ranks from the launcher"):
        KerasDistributedStrategy(preset="dp", devices=2)


@pytest.mark.skipif(BACKEND == "torch", reason="The torch backend refuses a device count outright.")
def test_a_device_count_outside_the_available_range_is_refused() -> None:
    """A negative count silently drops devices from the tail, so the run would span fewer than asked.

    The count the message names has to be the machine's own, or it sends the reader after a limit
    nothing here has: a range computed from the already-truncated list reports `1 and 0` for the
    count below, on a session that does have a device.
    """
    available = len(keras.distribution.list_devices())

    with pytest.raises(ValueError, match=f"Keras exposes {available}: the count must be between 1 and {available}"):
        KerasDistributedStrategy(preset="dp", devices=-1)


# ---------------------------------------------------------------------------
# The single preset, which every backend runs
# ---------------------------------------------------------------------------


def test_the_single_preset_touches_nothing() -> None:
    """Single-device training must stay the plain path: nothing activated, nothing wrapped or placed."""
    strategy = KerasDistributedStrategy()
    models: OrderedDict[str, Any] = OrderedDict(model=object())
    batch = {"x": [[1.0, 2.0]], "y": [[3.0]]}

    with strategy.activate():
        pass

    assert strategy.wrap(models) is models
    assert strategy.shard_batch(batch) == batch
    assert strategy.replicas == 1
    assert strategy.is_main


def test_compile_refuses_arguments_it_cannot_honor() -> None:
    """The backend adapter owns step compilation, so arguments handed here would be dropped."""
    strategy = KerasDistributedStrategy()
    step = object()

    assert strategy.compile(step, None) is step
    with pytest.raises(ValueError, match="compiled by the backend adapter"):
        strategy.compile(step, {"jit_compile": True})


def test_load_state_dict_refuses_a_missing_state() -> None:
    """A resume without a state must stop the run, not start it from freshly initialized weights."""
    with pytest.raises(ValueError, match="required to resume"):
        KerasDistributedStrategy().load_state_dict({}, {}, None, None)


# ---------------------------------------------------------------------------
# JAX: keras.distribution, in subprocesses owning their device count
# ---------------------------------------------------------------------------


JAX_SCRIPT = """
import json, sys
from importlib.util import module_from_spec, spec_from_file_location
import numpy as np
import keras
from structcast_model.keras.distributed import KerasDistributedStrategy
from structcast_model.keras.trainer import initial_model

def load(path, name):
    spec = spec_from_file_location(name, path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

directory, preset, devices = sys.argv[1], sys.argv[2], int(sys.argv[3]) or None
strategy = KerasDistributedStrategy(preset=preset, devices=devices)
# Before the model: a JAX variable reads the active distribution while it is created.
with strategy.activate():
    keras.utils.set_random_seed(0)
    model = initial_model(load(directory + "/model.py", "generated_model").Model(), {"x": (4,)})
    learner = load(directory + "/learner.py", "generated_learner").Learner(model=model)
strategy.wrap_steps(learner)

rows = [[1.0, 0.5, -0.5, 2.0], [0.0, 1.0, 1.0, -1.0], [2.0, 0.0, 1.0, 0.5], [-1.0, 1.0, 0.0, 1.0]]
x = np.asarray(rows * 2, "float32")
y = np.asarray([[1.0, -1.0], [0.5, 0.25], [0.0, 1.0], [-0.5, 0.5]] * 2, "float32")
# Read where the strategy placed them, before a step recomputes them.
# The "single" preset activates no distribution, so its arrays report a plain device sharding.
variables = {v.path: str(getattr(v.value.sharding, "spec", "single")) for v in model.variables}
losses = [learner.training_step(x=x, y=y)["loss"] for _ in range(3)]
batch = strategy.shard_batch({"x": x, "y": y})
print(json.dumps({
    "replicas": strategy.replicas,
    "variables": variables,
    # The "single" preset hands the batch back untouched, hence the numpy fallback.
    "batch": {k: str(getattr(getattr(v, "sharding", None), "spec", "single")) for k, v in batch.items()},
    # Floats, not tensors: what a step reports is what the tracker averages and the loggers write.
    "losses": [float(keras.ops.convert_to_numpy(loss)) for loss in losses],
    "kernel": np.asarray(keras.ops.convert_to_numpy(model.variables[0].value)).tolist(),
}))
"""


@pytest.fixture(scope="module")
def jax_reference(tmp_path_factory: pytest.TempPathFactory, generated: Path) -> dict[str, Any]:
    """Run the same three steps on one JAX device, as the yardstick every preset must match."""
    return _run(JAX_SCRIPT, tmp_path_factory.mktemp("jax_reference"), str(generated), "single", "0", **JAX_DEVICES)


@pytest.mark.parametrize("preset", ["dp", "fsdp"])
def test_a_jax_preset_trains_to_the_losses_of_one_device(
    preset: str, tmp_path: Path, generated: Path, jax_reference: dict[str, Any]
) -> None:
    """A run must not depend on how many devices it was given, or a sharded run cannot be trusted.

    The reported criteria are the yardstick twice over: they must be plain floats -- the tracker
    averages floats -- and they must be the *global* values. A step whose reductions stopped at the
    slice its device holds would report a quarter of the batch and log it as the epoch's loss.
    """
    result = _run(JAX_SCRIPT, tmp_path, str(generated), preset, "4", **JAX_DEVICES)

    assert result["replicas"] == 4
    assert result["batch"] == {"x": "P('batch', None)", "y": "P('batch', None)"}
    assert result["losses"] == pytest.approx(jax_reference["losses"], rel=1e-6)
    assert np.allclose(result["kernel"], jax_reference["kernel"], rtol=1e-6)


def test_the_jax_dp_preset_replicates_every_variable(tmp_path: Path, generated: Path) -> None:
    """Data parallelism is exactly that: whole variables everywhere, one slice of the batch each."""
    result = _run(JAX_SCRIPT, tmp_path, str(generated), "dp", "4", **JAX_DEVICES)

    assert set(result["variables"].values()) == {"P(None, None)", "P(None,)"}


def test_the_jax_fsdp_preset_shards_the_leading_dimension_of_what_it_can_divide(
    tmp_path: Path, generated: Path
) -> None:
    """Sharding is what the preset exists for, and only the kernel is a candidate.

    The 4x2 kernel's leading dimension divides by two devices, so it is split; the bias has one
    dimension, which is the batch axis' own, and stays whole on every device.
    """
    result = _run(JAX_SCRIPT, tmp_path, str(generated), "fsdp", "2", **JAX_DEVICES)

    assert result["replicas"] == 2
    assert sorted(result["variables"].values()) == ["P('batch', None)", "P(None,)"]


def test_the_jax_fsdp_preset_replicates_what_the_devices_do_not_divide(tmp_path: Path, generated: Path) -> None:
    """A variable the mesh cannot split falls back to replication rather than failing the run.

    Four rows over eight devices is that case, and the alternative -- an error deep inside variable
    creation -- would make the preset unusable for any model whose shapes are not multiples of the
    device count.
    """
    result = _run(JAX_SCRIPT, tmp_path, str(generated), "fsdp", "8", **JAX_DEVICES)

    assert result["replicas"] == 8
    assert set(result["variables"].values()) == {"P(None, None)", "P(None,)"}


# ---------------------------------------------------------------------------
# TensorFlow: MirroredStrategy, in a subprocess owning its logical devices
# ---------------------------------------------------------------------------


TENSORFLOW_SCRIPT = """
import json, sys
from importlib.util import module_from_spec, spec_from_file_location
import numpy as np
import tensorflow as tf

# Before anything initializes the TensorFlow runtime, which fixes the logical device list.
cpus = tf.config.list_physical_devices("CPU")
tf.config.set_logical_device_configuration(cpus[0], [tf.config.LogicalDeviceConfiguration()] * 2)

import keras
from structcast_model.keras.distributed import KerasDistributedStrategy
from structcast_model.keras.trainer import initial_model

def load(path, name):
    spec = spec_from_file_location(name, path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

directory, preset = sys.argv[1], sys.argv[2]
strategy = KerasDistributedStrategy(preset=preset)
# Before the models: MirroredStrategy mirrors only the variables created inside its scope.
with strategy.activate():
    keras.utils.set_random_seed(0)
    model = initial_model(load(directory + "/model.py", "generated_model").Model(), {"x": (4,)})
    learner = load(directory + "/learner.py", "generated_learner").Learner(model=model)
strategy.wrap_steps(learner)

# The same batch either way: whole on one device, split two rows per replica under "dp".
x = np.asarray([[1.0, 0.5, -0.5, 2.0], [0.0, 1.0, 1.0, -1.0], [2.0, 0.0, 1.0, 0.5], [-1.0, 1.0, 0.0, 1.0]], "f4")
y = np.asarray([[1.0, -1.0], [0.5, 0.25], [0.0, 1.0], [-0.5, 0.5]], "float32")
kernel = np.asarray(keras.ops.convert_to_numpy(model.variables[0].value))
bias = np.asarray(keras.ops.convert_to_numpy(model.variables[1].value))
losses = [float(keras.ops.convert_to_numpy(learner.training_step(x=x, y=y)["loss"])) for _ in range(3)]
print(json.dumps({
    "replicas": strategy.replicas,
    "mirrored": type(model.variables[0].value).__name__,
    "losses": losses,
    # The whole batch's error, computed off the initial weights without TensorFlow: the value the
    # first step has to report if the two replicas' criteria were reduced rather than picked from.
    "expected": float(np.mean((x @ kernel + bias - y) ** 2)),
    "inference": float(keras.ops.convert_to_numpy(learner.inference_step(x=x, y=y)["loss"])),
    "kernel": np.asarray(keras.ops.convert_to_numpy(model.variables[0].value)).tolist(),
}))
"""


@pytest.fixture(scope="module")
def tensorflow_reference(tmp_path_factory: pytest.TempPathFactory, generated: Path) -> dict[str, Any]:
    """Run the same three steps on one TensorFlow device, as the yardstick `dp` must match."""
    directory = tmp_path_factory.mktemp("tensorflow_reference")
    return _run(TENSORFLOW_SCRIPT, directory, str(generated), "single", backend="tensorflow")


def test_the_tensorflow_dp_preset_trains_under_mirrored_strategy(
    tmp_path: Path, generated: Path, tensorflow_reference: dict[str, Any]
) -> None:
    """Two mirrored replicas train, and what they report is one reduced number.

    `MirroredStrategy.run` hands back a per-replica value, which the tracker would turn into a
    meaningless average of one replica's slice -- or fail on outright. The first step's loss is
    therefore compared with the whole batch's error, computed from the initial weights with numpy:
    a value reduced with `ReduceOp.MEAN` matches it, a single replica's half does not.

    The remaining steps are compared with the same batch trained whole on one device, exactly as the
    JAX presets are: `dp` means the mean of the per-replica gradients on every backend, and the
    Keras TensorFlow optimizer all-reduces them with `ReduceOp.SUM` -- so without the strategy's
    scaling this run would take steps twice the size and diverge from the reference by the second.
    """
    result = _run(TENSORFLOW_SCRIPT, tmp_path, str(generated), "dp", backend="tensorflow")

    assert result["replicas"] == 2
    assert result["mirrored"] == "MirroredVariable"
    assert result["losses"][0] == pytest.approx(result["expected"], rel=1e-5)
    assert result["losses"][1] < result["losses"][0]
    assert result["inference"] < result["losses"][0]
    assert result["losses"] == pytest.approx(tensorflow_reference["losses"], rel=1e-5)
    assert np.allclose(result["kernel"], tensorflow_reference["kernel"], rtol=1e-5)


TENSORFLOW_CLI_SCRIPT = """
import json, os, sys

import numpy as np
import tensorflow as tf

# Before anything initializes the TensorFlow runtime, which fixes the logical device list. Guarded
# because the run below loads this same file back as an object pattern, which re-executes it.
try:
    cpus = tf.config.list_physical_devices("CPU")
    tf.config.set_logical_device_configuration(cpus[0], [tf.config.LogicalDeviceConfiguration()] * 2)
except RuntimeError:
    pass

from structcast_model.keras.trainer import KerasTrainer

X = np.asarray([[1.0, 0.5, -0.5, 2.0], [0.0, 1.0, 1.0, -1.0], [2.0, 0.0, 1.0, 0.5], [-1.0, 1.0, 0.0, 1.0]], "f4")
Y = np.asarray([[1.0, -1.0], [0.5, 0.25], [0.0, 1.0], [-0.5, 0.5]], "float32")

def batches():
    return [{"x": X, "y": Y}]

class Recorder(KerasTrainer):
    # Reads what the command built, from inside the loop it hands the built run to.
    def fit(self, *args, **kwargs):
        optimizer = next(iter(self.learner.optimizers.values()))
        variables = {
            "model": type(self.learner.models["model"].variables[0].value).__name__,
            "optimizer": type(optimizer.variables[0].value).__name__,
        }
        with open(os.environ["PROBE_OUT"], "w") as handle:
            json.dump(variables, handle)
        return super().fit(*args, **kwargs)

if __name__ == "__main__":
    import mlflow
    from typer.testing import CliRunner
    from structcast_model.commands.cmd_keras import app

    directory, out = sys.argv[1], sys.argv[2]
    os.environ["PROBE_OUT"] = out + "/variables.json"
    mlflow.set_tracking_uri(out + "/mlruns")
    result = CliRunner().invoke(app, [
        "train",
        "model: [_obj_, {_addr_: Model, _file_: " + directory + "/model.py}, _call_]",
        "--backend", "tensorflow",
        "--shape", "x: [4]",
        "--learner", "[_obj_, {_addr_: Learner, _file_: " + directory + "/learner.py}]",
        "--training-dataset", "[_obj_, {_addr_: batches, _file_: " + __file__ + "}, _call_]",
        "--trainer", "[_obj_, {_addr_: Recorder, _file_: " + __file__ + "}]",
        "--strategy", "dp",
        "--epochs", "1",
        "--lower-criterion", "loss",
        "--experiment", "keras-tf-dp",
        "--ci",
    ])
    assert result.exit_code == 0, result.output + str(result.exception)
    with open(os.environ["PROBE_OUT"]) as handle:
        print(json.dumps(json.load(handle)))
"""


def test_the_training_command_builds_the_learner_under_the_mirrored_scope(tmp_path: Path, generated: Path) -> None:
    """The command's own ordering is what a run gets, and only a scope it builds inside mirrors.

    `MirroredStrategy` mirrors the variables created inside its scope and nothing else, so a command
    that built the models there but the learner afterwards would leave every optimizer variable --
    the step counter, the momenta -- unmirrored next to mirrored weights, which is only visible on
    more than one device. The whole run therefore goes through the CLI here, in the one subprocess
    that has two of them.
    """
    output = tmp_path / "run"
    output.mkdir()

    result = _run(TENSORFLOW_CLI_SCRIPT, tmp_path, str(generated), str(output), backend="tensorflow")

    assert result == {"model": "MirroredVariable", "optimizer": "MirroredVariable"}


# ---------------------------------------------------------------------------
# torch: DistributedDataParallel, in a subprocess spawning two gloo ranks
# ---------------------------------------------------------------------------


TORCH_SCRIPT = """
import json, sys
from importlib.util import module_from_spec, spec_from_file_location
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

X = np.asarray([[1.0, 0.5, -0.5, 2.0], [0.0, 1.0, 1.0, -1.0], [2.0, 0.0, 1.0, 0.5], [-1.0, 1.0, 0.0, 1.0]], "f4")
Y = np.asarray([[1.0, -1.0], [0.5, 0.25], [0.0, 1.0], [-0.5, 0.5]], "float32")

def load(path, name):
    spec = spec_from_file_location(name, path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def kernel_of(model):
    import keras

    return np.asarray(keras.ops.convert_to_numpy(model.variables[0].value)).tolist()

def build(directory, preset, seed):
    import keras
    from structcast_model.keras.distributed import KerasDistributedStrategy
    from structcast_model.keras.trainer import initial_model

    strategy = KerasDistributedStrategy(preset=preset)
    with strategy.activate():
        # One seed per rank, not one for the run: the ranks have to be made identical by the
        # broadcast below, not by the habit of seeding them the same way.
        keras.utils.set_random_seed(seed)
        model = initial_model(load(directory + "/model.py", "generated_model").Model(), {"x": (4,)})
        # A normalization layer nobody trains, whose statistics each rank would otherwise diverge.
        norm = keras.layers.BatchNormalization()
        norm.build((None, 4))
        # A layer whose only variable is the integer RNG state of its seed generator: not a
        # statistic, and not something an all-reduce and a division could even produce.
        drop = keras.layers.Dropout(0.5)
        drop.build((None, 4))
        kernels = {"seeded_kernel": kernel_of(model)}
        strategy.sync_initial_weights({"model": model, "norm": norm})
        kernels["synced_kernel"] = kernel_of(model)
        models = strategy.wrap({"model": model})
        learner = load(directory + "/learner.py", "generated_learner").Learner(**models)
    # The two loose layers are handed to the learner, since a saver reads the models off it.
    learner.models["norm"] = norm
    learner.models["drop"] = drop
    strategy.wrap_steps(learner)
    return strategy, model, norm, learner, kernels

# Saves a training state the way a run does: through the callback the training CLI registers.
def checkpoint(strategy, learner, x, y):
    from structcast_model.base_trainer import SimpleDataProvider
    from structcast_model.keras.trainer import KerasTracker, KerasTrainer, KerasTrainingStateSaver
    from structcast_model.loggers.base import NullLogger

    class Recorder(NullLogger):
        def __init__(self):
            self.states = []

        def log_state_dict(self, states, name):
            self.states.append(states)

    recorder = Recorder()
    saver = KerasTrainingStateSaver(logger=recorder, strategy=strategy)
    saver.on_epoch_end(KerasTrainer(
        learner=learner,
        tracker=KerasTracker.from_criteria(["loss"]),
        data=SimpleDataProvider(training_dataset=[{"x": x, "y": y}]),
        callbacks=[saver],
    ))
    return recorder.states[-1]

# Reads one variable back out of a saved state, which nests it under the segments of its path.
def saved(state, variable):
    for key in variable.path.split("/"):
        state = state[key]
    return np.asarray(state).tolist()

def report(model, norm, loss):
    import keras

    return {
        "loss": float(keras.ops.convert_to_numpy(loss)),
        "kernel": kernel_of(model),
        "moving_mean": np.asarray(keras.ops.convert_to_numpy(norm.moving_mean.value)).tolist(),
    }

def worker(rank, directory, out):
    dist.init_process_group(backend="gloo", init_method="file://" + out + "/init", rank=rank, world_size=2)
    import keras

    strategy, model, norm, learner, kernels = build(directory, "dp", rank)
    half = slice(rank * 2, rank * 2 + 2)
    # Each rank sees its own slice, as its loader would hand it: the strategy places nothing here.
    loss = learner.training_step(x=X[half], y=Y[half])["loss"]
    norm(keras.ops.convert_to_tensor(X[half]), training=True)
    before = np.asarray(keras.ops.convert_to_numpy(norm.moving_mean.value)).tolist()
    state = checkpoint(strategy, learner, X[half], Y[half])
    payload = report(model, norm, loss)
    payload.update(kernels)
    payload["moving_mean_before"] = before
    payload["saved_moving_mean"] = saved(state["models"]["norm"], norm.moving_mean)
    torch.save(payload, out + "/rank" + str(rank) + ".pt")

if __name__ == "__main__":
    directory, out = sys.argv[1], sys.argv[2]
    import keras

    strategy, model, norm, learner, kernels = build(directory, "single", 0)
    reference = report(model, norm, learner.training_step(x=X, y=Y)["loss"])
    mp.spawn(worker, args=(directory, out), nprocs=2, join=True)
    ranks = [torch.load(out + "/rank" + str(rank) + ".pt", weights_only=False) for rank in range(2)]
    print(json.dumps({"reference": reference, "ranks": ranks}))
"""


@pytest.fixture(scope="module")
def torch_ranks(tmp_path_factory: pytest.TempPathFactory, generated: Path) -> dict[str, Any]:
    """Run the two-rank gloo probe once: spawning it is expensive and every assertion reads one run."""
    directory = tmp_path_factory.mktemp("torch_ranks")
    output = directory / "ranks"
    output.mkdir()
    return _run(TORCH_SCRIPT, directory, str(generated), str(output), backend="torch")


def test_the_torch_dp_preset_averages_the_gradients_and_the_criteria(torch_ranks: dict[str, Any]) -> None:
    """Two ranks training half a batch each must land exactly where one rank on the whole batch does.

    That is the whole contract of the preset, and it covers both halves of it: the weights after the
    step prove `DistributedDataParallel` averaged the gradients into the `.grad` the backend adapter
    reads off each Keras variable, and the reported loss proves the criteria were all-reduced --
    without which every rank would log the loss of its own slice.
    """
    result = torch_ranks
    rank0, rank1 = result["ranks"]
    assert rank0["loss"] == pytest.approx(result["reference"]["loss"], rel=1e-5)
    assert rank1["loss"] == pytest.approx(rank0["loss"], rel=1e-6)
    assert np.allclose(rank0["kernel"], result["reference"]["kernel"], rtol=1e-5)
    # Identical weights on both ranks, or the two would drift apart step after step.
    assert np.allclose(rank0["kernel"], rank1["kernel"], rtol=1e-6)


def test_the_torch_dp_preset_starts_every_rank_from_rank_zeros_weights(torch_ranks: dict[str, Any]) -> None:
    """The ranks must be made identical by the broadcast, not by the habit of seeding them the same.

    That habit is exactly what `docs/adr/0003` puts the broadcast behind the strategy for: a run
    whose ranks initialize apart -- a seed derived from the rank, a weight file read on rank 0 only
    -- would otherwise train two different models against one averaged gradient and report neither.
    The ranks here are seeded differently on purpose, so a missing broadcast shows up immediately.
    """
    rank0, rank1 = torch_ranks["ranks"]

    assert rank0["seeded_kernel"] != rank1["seeded_kernel"]
    assert np.array_equal(rank0["synced_kernel"], rank0["seeded_kernel"])
    assert np.array_equal(rank1["synced_kernel"], rank0["seeded_kernel"])


def test_the_torch_dp_preset_reconciles_the_normalization_statistics_when_a_state_is_saved(
    torch_ranks: dict[str, Any],
) -> None:
    """Moving statistics are the one piece of state DDP does not keep in step, so the strategy does.

    A Keras normalization layer keeps them in variables, which the torch backend makes
    `torch.nn.Parameter`s with `requires_grad=False` -- not buffers -- so DDP neither broadcasts nor
    reduces them, and `SyncBatchNorm` conversion cannot see a Keras layer at all. Each rank
    therefore ends an epoch with the statistics of its own slice; they are averaged where the state
    is read, so the checkpoint and the next epoch hold the whole batch's statistics.

    The state is produced by the saver the training CLI registers, not by calling the strategy by
    hand: reconciliation that the run's own checkpoint path does not go through is reconciliation
    that never happens. The assertions pin all three halves: the ranks did diverge, they were
    reconciled in place, and what the saver wrote is the reconciled value.
    """
    rank0, rank1 = torch_ranks["ranks"]

    assert rank0["moving_mean_before"] != rank1["moving_mean_before"]
    assert rank0["moving_mean"] == pytest.approx(rank1["moving_mean"], rel=1e-6)
    expected = [(a + b) / 2 for a, b in zip(rank0["moving_mean_before"], rank1["moving_mean_before"], strict=True)]
    assert rank0["moving_mean"] == pytest.approx(expected, rel=1e-6)
    assert rank0["saved_moving_mean"] == pytest.approx(expected, rel=1e-6)
