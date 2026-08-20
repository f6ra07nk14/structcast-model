"""The distributed strategy of a Keras training run: one preset per backend-native mechanism.

Keras 3 has no distribution mechanism of its own that works everywhere: `keras.distribution` is
implemented for JAX, is an explicitly labelled no-op prototype on TensorFlow (its
`distribute_value` is a bare `pass`), and does not exist on torch. So the strategy here is one
class holding one preset, and the preset is realized by whatever the active backend actually
supports (`docs/adr/0016`):

| preset   | jax                                    | tensorflow                        | torch                    |
| -------- | -------------------------------------- | --------------------------------- | ------------------------ |
| `single` | nothing to activate                    | nothing to activate               | nothing to activate      |
| `dp`     | `keras.distribution.DataParallel`      | `tf.distribute.MirroredStrategy`  | `DistributedDataParallel`|
| `fsdp`   | `keras.distribution.ModelParallel`     | rejected                          | rejected                 |

The backend is read exactly once, in `__post_init__`, and every unsupported cell is rejected there
-- before a model exists -- because each of the two rejected cells fails silently otherwise: a
`keras.distribution` sharding on TensorFlow replicates every variable without a word, and torch
FSDP2 rebinds the parameters a Keras variable caches, so the run keeps training the stale ones.

Activation is process-wide and happens before the models are built, as on the Flax side: JAX reads
the distribution when a variable is created, and a `MirroredStrategy` mirrors only the variables
created inside its scope.
"""

from collections import OrderedDict
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from re import Pattern, compile as re_compile
from typing import TYPE_CHECKING, Any, Literal

import keras
from structcast_model.keras.trainer import apply_state_dict, collect_state_dict, get_keras_device

AXIS = "batch"
"""The single mesh axis every preset builds: batches split along it, FSDP shards along it.

Keras' own `DEFAULT_BATCH_DIM_NAME`, so a `DataParallel` mesh and the `ModelParallel` mesh built
here name their axis identically and a rule table reads the same under both presets.
"""

PRESET_RULES: Mapping[str, tuple[tuple[str, str], ...]] = {
    "single": ((r".*", "replicate"),),
    "dp": ((r".*", "replicate"),),
    "fsdp": ((r".*", "fsdp"),),
}
"""Ordered (variable-path regex, tactic) rules of each preset; the first matching rule wins."""

TACTICS = ("replicate", "fsdp")
"""The tactics a rule may name: keep the variable on every device, or shard it across the mesh."""

REJECTED: Mapping[tuple[str, str], str] = {
    ("fsdp", "tensorflow"): (
        'The "fsdp" preset is not available on the tensorflow Keras backend: keras.distribution is a no-op '
        "prototype there (its backend distribute_value does nothing at all), so a sharded run would replicate "
        'every variable and report success. Use the "dp" preset, which runs on tf.distribute.MirroredStrategy, '
        "or run fsdp on the jax backend."
    ),
    ("fsdp", "torch"): (
        'The "fsdp" preset is not available on the torch Keras backend: the upstream sharding work is unfinished '
        "(keras-team/keras#23418) and torch FSDP2 replaces the parameters a Keras variable caches, so the run "
        'keeps training the stale ones without an error. Use the "dp" preset, which wraps every model in '
        "DistributedDataParallel, or run fsdp on the jax backend."
    ),
}
"""The unsupported (preset, backend) cells, each with the reason the run is refused for."""


class RuleModelParallel(keras.distribution.ModelParallel):
    """A `ModelParallel` whose variable layouts come from (path regex, tactic) rules.

    A `keras.distribution.LayoutMap` holds one fixed axis tuple per path pattern, and that tuple has
    to have exactly the rank of the variable it places -- so the one-line rule tables the presets are
    written as (`".*" -> "fsdp"`) cannot be `LayoutMap` entries: they name a tactic, and the layout it
    turns into depends on the variable's own shape. The rules are applied here instead, where the
    variable is in hand. The `LayoutMap` handed to the base class carries the device mesh and stays
    empty, so `get_tensor_layout` keeps its meaning (no intermediate tensor is constrained).
    """

    def __init__(self, *, layout_map: Any, rules: Sequence[tuple[Pattern[str], str]]) -> None:
        """Build the distribution over *layout_map*'s mesh, placing variables by *rules*."""
        super().__init__(layout_map=layout_map)
        self._rules = tuple(rules)

    def get_variable_layout(self, variable: Any) -> Any:
        """Return the layout of one variable: its leading dimension sharded, or replicated.

        Only the leading dimension of a variable with at least two dimensions is a candidate, and
        only when the mesh divides it. That leaves every bias and normalization statistic replicated
        by construction, and a variable the mesh does not divide falls back to replication rather
        than failing the run -- the same shape the Flax twin's rules have (`docs/adr/0014`).
        """
        if getattr(variable, "_layout", None) is not None:
            return variable._layout  # noqa: SLF001  # The base class reads it first too; a caller may pin a layout.
        mesh = self.device_mesh
        axes: list[str | None] = [None] * len(variable.shape)
        for pattern, tactic in self._rules:
            if pattern.search(variable.path):
                if tactic == "fsdp" and len(axes) > 1 and not variable.shape[0] % mesh.shape[0]:
                    axes[0] = AXIS
                break
        return keras.distribution.TensorLayout(axes, mesh)


@dataclass(kw_only=True)
class KerasDistributedStrategy:
    """Strategy owning how a Keras run spreads over its devices, on whichever backend it runs.

    The surface is the torch `DistributedStrategy` one, minus `grad_scaler_creator`: Keras loss
    scaling lives inside the optimizer (`keras.optimizers.LossScaleOptimizer`), so no Keras learner
    ever asks a strategy for a scaler. Two calls are ordered around the learner, and neither can be
    merged into the other: :meth:`wrap` runs before it is built, since the learner captures the
    model objects and its optimizers are built against their variables, and :meth:`wrap_steps` runs
    after, since it rewires the steps the learner built in its constructor.
    """

    preset: Literal["single", "dp", "fsdp"] = "single"
    """Which mechanism to run on: one device, replicated variables, or sharded variables."""

    device: str | None = None
    """Device the run is checked against, e.g. `"cpu:0"`; the first available one by default.

    It selects the device *type* the multi-device presets span and, on torch, whether the process
    group is NCCL or gloo. Which devices a Keras backend computes on is otherwise the backend's own
    choice, exactly as `scm keras train --device` documents.
    """

    devices: int | None = None
    """How many devices the `dp` and `fsdp` presets span; every available device by default.

    Rejected on the torch backend, where the number of ranks is the launcher's to decide.
    """

    rules: Sequence[tuple[str, str]] | None = None
    """Rules replacing the `fsdp` preset's table, as ordered (variable-path regex, tactic) pairs."""

    _backend: str = field(default="", init=False, repr=False)
    _rules: tuple[tuple[Pattern[str], str], ...] = field(default=(), init=False, repr=False)
    _distribution: Any = field(default=None, init=False, repr=False)
    _replicas: Any = field(default=None, init=False, repr=False)
    _scope: Any = field(default=None, init=False, repr=False)
    _rank: int = field(default=0, init=False, repr=False)
    _world_size: int = field(default=1, init=False, repr=False)

    def __post_init__(self) -> None:
        """Validate the preset against the active backend, its rules and its device count.

        The (preset, backend) matrix, the rules and the device count are checked here rather than at
        activation: a rejected cell must fail before the run has built anything, and the two
        rejected cells are exactly the ones that would otherwise train silently wrong. The one thing
        left to :meth:`activate` is whether a torch run has a process group, which is a
        `RuntimeError` raised there because only the launcher can answer it -- still before a model
        is built.

        Raises:
            ValueError: if the preset is unknown, unsupported on the active backend, its rules name
                an unknown tactic or do not apply, or the device count cannot be honored.
        """
        self._backend = keras.backend.backend()
        if self.preset not in PRESET_RULES:
            raise ValueError(f"Unknown preset {self.preset!r}. Available presets: {', '.join(PRESET_RULES)}.")
        if self.rules is not None and self.preset != "fsdp":
            raise ValueError(
                f'Sharding rules only decide how the "fsdp" preset shards variables, but the preset is '
                f"{self.preset!r}, which replicates them all: drop the rules, or select the fsdp preset."
            )
        rules = PRESET_RULES[self.preset] if self.rules is None else self.rules
        for _, tactic in rules:
            if tactic not in TACTICS:
                raise ValueError(f"Unknown sharding tactic {tactic!r}. Available tactics: {', '.join(TACTICS)}.")
        self._rules = tuple((re_compile(pattern), tactic) for pattern, tactic in rules)
        # After the rule table: a mistyped tactic is wrong on every backend, so it is reported as
        # itself rather than as whatever the active backend happens to say about the preset.
        if (reason := REJECTED.get((self.preset, self._backend))) is not None:
            raise ValueError(reason)
        self.device = get_keras_device(self.device)
        if self.devices is not None:
            if self._backend == "torch":
                raise ValueError(
                    "The torch Keras backend takes its number of ranks from the launcher (torchrun, or an "
                    "already initialized process group), so a strategy cannot pick one: drop devices."
                )
            # Unlimited: `_device_names` applies the very count being validated, so counting its
            # result would report a negative or oversized count as the machine's own device count.
            available = len(self._device_names(limit=False))
            if not 1 <= self.devices <= available:
                raise ValueError(
                    f"Asked for {self.devices} devices, but Keras exposes {available}: "
                    f"the count must be between 1 and {available}."
                )

    @property
    def rank(self) -> int:
        """The rank of this process; always 0 outside a torch process group."""
        return self._rank

    @property
    def world_size(self) -> int:
        """How many processes take part in the run; always 1 outside a torch process group."""
        return self._world_size

    @property
    def is_main(self) -> bool:
        """Whether this process is the one that speaks for the run."""
        return self._rank == 0

    @property
    def replicas(self) -> int:
        """How many replicas the run is spread over, whichever mechanism spreads it."""
        if self._distribution is not None:
            return int(self._distribution.num_model_replicas)
        if self._replicas is not None:
            return int(self._replicas.num_replicas_in_sync)
        return self._world_size

    @contextmanager
    def activate(self) -> Iterator[None]:
        """Make this strategy the process's, for as long as the models are being built.

        The models, their optimizers and every slot variable have to be created inside this block:
        JAX reads the active distribution while a variable is created, and a `MirroredStrategy`
        mirrors only what is created inside its scope. What the block leaves behind differs per
        backend, because only TensorFlow's part is a scope: JAX keeps its distribution active (as
        the mesh `jax.set_mesh` activates does on the Flax side) and torch stays in its process
        group, while the `MirroredStrategy` scope is left on the way out -- a scope that outlived
        the construction would mirror every variable any later caller in the process creates.
        """
        if self.preset == "single":
            yield
            return
        if self._backend == "jax":
            self._distribution = self._jax_distribution()
            keras.distribution.set_distribution(self._distribution)
            yield
            return
        if self._backend == "torch":
            self._activate_torch()
            yield
            return
        import tensorflow as tf  # noqa: PLC0415  # An inactive backend's framework may not be installed.

        self._replicas = tf.distribute.MirroredStrategy(devices=self._device_names())
        # The scope object is held, not dropped: it owns the variable-creator scope it entered, and
        # letting it be collected tears that down early -- after which the models are built as plain,
        # unmirrored variables and the first step fails inside `strategy.run`.
        self._scope = self._replicas.scope()
        with self._scope:
            yield

    def wrap(self, models: "OrderedDict[str, Any]") -> "OrderedDict[str, Any]":
        """Return the models the learner should be built over.

        Only torch has anything to wrap: a Keras variable *is* the `torch.nn.Parameter` it holds, so
        `DistributedDataParallel` averages the gradients straight into the `.grad` the backend
        adapter reads off it. JAX and TensorFlow variables carry their placement from the moment
        they were created under the activated distribution, so their models come back untouched.
        """
        if self._backend != "torch" or self.preset == "single":
            return models
        return OrderedDict((name, _wrap_ddp(model)) for name, model in models.items())

    def sync_initial_weights(self, models: Mapping[str, Any]) -> None:
        """Make every rank start from rank 0's weights. Call on every rank, before :meth:`wrap`.

        Nothing to do outside torch: JAX and TensorFlow runs are single-controller, so one process
        initializes every device.
        """
        if self._backend != "torch" or self.preset == "single":
            return
        import torch.distributed as dist  # noqa: PLC0415  # An inactive backend's framework may not be installed.

        for model in models.values():
            for variable in model.variables:
                dist.broadcast(variable.value.data, src=0)

    def wrap_steps(self, learner: Any) -> None:
        """Rewire the learner's steps so each one runs across the replicas and reports one value.

        This is where `docs/adr/0016`'s rule is honored: what reaches the tracker is already reduced
        across replicas. The three backends need three different things, and none of them can happen
        inside the learner, which is backend-neutral by construction:

        - JAX places the batch across the mesh; the reductions inside the compiled step are then
          global by construction, since XLA reduces a sharded array across every device holding it.
        - TensorFlow splits the batch per replica, runs the step through `MirroredStrategy.run` and
          reduces the per-replica criteria with `ReduceOp.MEAN`.
        - torch runs the step as it is -- the loader owns the rank's slice of the data -- and
          all-reduces the criteria, since each rank only ever saw its own slice.

        The *gradients* are each backend's own business, and the two multi-device backends do not
        agree: `DistributedDataParallel` averages them across ranks, while the Keras TensorFlow
        optimizer sums the per-replica gradients (`_all_reduce_sum_gradients`, the behaviour of
        `Model.fit` under a strategy too), so a TensorFlow `dp` run takes steps as many times larger
        as it has replicas unless the learner's loss is scaled for it. Reproducing `fit` was
        preferred over inventing a third convention.
        """
        if self.preset == "single":
            return
        if self._replicas is not None:
            # The adapter traced each step into a `tf.function`, and a graph applying an optimizer is
            # a synchronization point TensorFlow refuses to nest inside `strategy.run`. The traced
            # steps are therefore unwrapped back to the Python functions they were built from, and
            # re-traced below inside the one graph that wraps the replicated call. `flow_functions`
            # is what a generated learner exposes for exactly this rebinding (as in `cmd_flax`); a
            # hand-written learner has nothing to unwrap.
            for name in getattr(learner, "flow_functions", ()):
                traced = getattr(learner, name)
                if hasattr(traced, "python_function"):
                    setattr(learner, name, traced.python_function)
        for name in ("training_step", "inference_step"):
            setattr(learner, name, self._replicated(getattr(learner, name)))

    def compile(self, module: Any, compile_kw: Mapping[str, Any] | None) -> Any:
        """Return *module* unchanged, or refuse to compile it.

        Keras step compilation belongs to the backend adapter -- `tf.function` on TensorFlow,
        `jax.jit` on JAX, eager on torch -- which owns the arguments each one needs, so there is
        nothing for a strategy to add and a caller asking for something else is asking the wrong
        object.

        Raises:
            ValueError: if compilation arguments are given.
        """
        if compile_kw is None:
            return module
        raise ValueError(
            "A Keras run's steps are compiled by the backend adapter, not by its distributed strategy, so "
            f"compilation arguments cannot be applied here. Got: {dict(compile_kw)}."
        )

    def shard_batch(self, batch: Mapping[str, Any]) -> dict[str, Any]:
        """Place one batch across the replicas, or return it as it is where the loader owns that.

        Args:
            batch (Mapping[str, Any]): The batch to place, keyed by input name.

        Returns:
            dict[str, Any]: The placed batch.

        Raises:
            ValueError: if an entry has no leading dimension, or one the replica count does not divide.
        """
        if self._distribution is None and self._replicas is None:
            # torch and `single`: on torch the loader hands each rank its own slice, exactly as the
            # torch training path's `DistributedSampler` does -- a strategy that split the batch here
            # would hand every rank the same data and quietly train on a fraction of the dataset.
            return dict(batch)
        placed = {}
        for key, value in batch.items():
            tensor = keras.ops.convert_to_tensor(value)
            shape = tuple(tensor.shape)
            if not shape or shape[0] % self.replicas:
                raise ValueError(
                    f'Batch entry "{key}" of shape {shape} cannot be split across {self.replicas} replicas: '
                    "its leading dimension must exist and be divisible by the replica count."
                )
            placed[key] = self._place(tensor, shape)
        return placed

    def state_dict(
        self,
        models: Mapping[str, Any],
        optimizers: Mapping[str, Any] | None = None,
        optimizer_models: Mapping[str, list[str]] | None = None,
    ) -> dict[str, Any]:
        """Produce `{"models": ..., "optimizers": ...}` in host memory, keyed by model and optimizer name.

        *optimizer_models* is accepted for protocol compatibility and unused: a Keras state is keyed
        by `variable.path`, which already says which model a variable belongs to.

        Raises:
            RuntimeError: if a multi-process JAX run is asked for a state, which no single host holds.
        """
        if self._distribution is not None and self._distribution.num_processes > 1:
            raise RuntimeError(
                "A multi-process JAX run cannot be checkpointed from one host: its arrays are not fully "
                "addressable here, so the state would be read from the shards this process happens to hold."
            )
        self._sync_statistics(models)
        return collect_state_dict(models, optimizers)

    def load_state_dict(
        self,
        models: Mapping[str, Any],
        optimizers: Mapping[str, Any],
        optimizer_models: Mapping[str, list[str]] | None,
        state: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Load a state produced by :meth:`state_dict` into the live models and optimizers.

        Returns the state itself, so the caller reads its non-array parts (`meta`, `grad_scalers`).
        The saved arrays are host numpy and the live variables carry the placement this strategy gave
        them, so a state saved on four devices lands on however many this run has: assigning a Keras
        variable re-applies its own layout on JAX, and copies in place on torch, which is what keeps
        a `DistributedDataParallel` wrapper pointing at the parameters it was built over.

        Raises:
            ValueError: if no state was given.
        """
        if state is None:
            raise ValueError("A training state is required to resume from.")
        apply_state_dict(models, optimizers, state)
        return state

    def _replicated(self, step: Any) -> Any:
        """Wrap one learner step so it runs across the replicas and reports reduced criteria."""
        if self._replicas is not None:
            import tensorflow as tf  # noqa: PLC0415  # An inactive backend's framework may not be installed.

            # One `tf.function` around the whole replicated call, which is what makes it legal at
            # all: the adapter already traced the step, and a `tf.function` containing an optimizer
            # application is a synchronization point TensorFlow refuses to nest inside
            # `strategy.run` unless the run itself is being traced.
            @tf.function
            def replicated(batch: Mapping[str, Any]) -> dict[str, Any]:
                criteria = self._replicas.run(step, kwargs=batch)
                return {
                    name: self._replicas.reduce(tf.distribute.ReduceOp.MEAN, value, axis=None)
                    for name, value in criteria.items()
                }

            def tensorflow_step(**batch: Any) -> dict[str, Any]:
                return dict(replicated(self.shard_batch(batch)))

            return tensorflow_step

        if self._distribution is not None:

            def jax_step(**batch: Any) -> dict[str, Any]:
                return step(**self.shard_batch(batch))

            return jax_step

        def torch_step(**batch: Any) -> dict[str, Any]:
            import torch  # noqa: PLC0415  # An inactive backend's framework may not be installed.
            import torch.distributed as dist  # noqa: PLC0415  # Same.

            criteria = {}
            for name, value in step(**batch).items():
                # Detached and copied: the value a step returns is still the one its graph produced,
                # and an in-place all-reduce would rewrite it under the optimizer that just ran.
                reduced = torch.as_tensor(value, dtype=torch.float32).detach().clone()
                dist.all_reduce(reduced)
                criteria[name] = reduced / self._world_size
            return criteria

        return torch_step

    def _place(self, tensor: Any, shape: tuple[int, ...]) -> Any:
        """Place one batch entry: across the JAX mesh, or as one value per TensorFlow replica."""
        if self._distribution is not None:
            return keras.distribution.distribute_tensor(tensor, self._distribution.get_data_layout(shape))
        per_replica = shape[0] // self.replicas
        return self._replicas.experimental_distribute_values_from_function(
            lambda context: tensor[
                context.replica_id_in_sync_group * per_replica : (context.replica_id_in_sync_group + 1) * per_replica
            ]
        )

    def _jax_distribution(self) -> Any:
        """Build the `keras.distribution` the JAX presets run on."""
        names = self._device_names()
        if self.preset == "dp":
            return keras.distribution.DataParallel(devices=names)
        mesh = keras.distribution.DeviceMesh(shape=(len(names),), axis_names=[AXIS], devices=names)
        return RuleModelParallel(layout_map=keras.distribution.LayoutMap(mesh), rules=self._rules)

    def _activate_torch(self) -> None:
        """Join the process group the launcher started, and bind this rank's device.

        Raises:
            RuntimeError: if no process group is available, so the run has no ranks to spread over.
        """
        import torch  # noqa: PLC0415  # An inactive backend's framework may not be installed.
        import torch.distributed as dist  # noqa: PLC0415  # Same.

        if not dist.is_initialized():
            if not dist.is_torchelastic_launched():
                raise RuntimeError(
                    'The "dp" preset on the torch Keras backend spreads a run over the ranks of a process '
                    "group, and this process is in none: launch the command with torchrun, or initialize the "
                    "group before building the strategy."
                )
            dist.init_process_group(backend="nccl" if self._is_gpu else "gloo")
        self._rank = dist.get_rank()
        self._world_size = dist.get_world_size()
        if self._is_gpu:
            torch.cuda.set_device(self._rank % torch.cuda.device_count())

    def _sync_statistics(self, models: Mapping[str, Any]) -> None:
        """Average every floating-point non-trainable variable across the ranks, in place.

        The state a `DistributedDataParallel` run has to repair by hand. A Keras normalization layer
        keeps its moving statistics in variables, and on the torch backend those are
        `torch.nn.Parameter`s with `requires_grad=False` rather than buffers -- so DDP neither
        broadcasts them per forward nor reduces them at backward (and
        `convert_sync_batchnorm` cannot see a Keras layer either), leaving every rank with the
        statistics of the slice it saw. They are reduced here, where a state is read, which is the
        epoch boundary a resume is exact at: the checkpoint and the next epoch both start from the
        statistics of the whole batch instead of rank 0's quarter of it.
        """
        if self._backend != "torch" or self.preset == "single" or self._world_size == 1:
            return
        import torch.distributed as dist  # noqa: PLC0415  # An inactive backend's framework may not be installed.

        for model in models.values():
            for variable in model.variables:
                # Only the statistics: a non-trainable variable is not necessarily one, and the two
                # other kinds must not be averaged. A `keras.random.SeedGenerator` -- which every
                # Dropout or random-augmentation layer holds -- keeps its RNG state in an integer
                # variable, so the division below would raise on it, and an averaged RNG state would
                # be meaningless anyway; the same goes for any other integer counter a layer keeps.
                if variable.trainable or "seed_generator" in variable.path:
                    continue
                if not keras.backend.is_float_dtype(variable.dtype):
                    continue
                value = variable.value.data
                dist.all_reduce(value)
                value /= self._world_size

    def _device_names(self, *, limit: bool = True) -> list[str]:
        """The devices the multi-device presets span, in the spelling the active backend takes.

        With *limit* false, every device the backend exposes, before `devices` cuts the list down.
        """
        if self._backend == "tensorflow":
            import tensorflow as tf  # noqa: PLC0415  # An inactive backend's framework may not be installed.

            names = [device.name for device in tf.config.list_logical_devices(self._device_type.upper())]
        else:
            names = [name for name in keras.distribution.list_devices() if name.startswith(self._device_type)]
        return names[: self.devices] if limit else names

    @property
    def _device_type(self) -> str:
        """The device type of the run, e.g. `"cpu"` or `"gpu"`."""
        return str(self.device).split(":")[0]

    @property
    def _is_gpu(self) -> bool:
        """Whether the run is on accelerators, which decides the torch process-group backend."""
        return self._device_type != "cpu"


def _wrap_ddp(model: Any) -> Any:
    """Wrap one Keras model in a `DistributedDataParallel` that still looks like the model.

    A generated learner reads `trainable_variables` off the models it was handed, and DDP proxies
    none of the Keras surface: an unwrapped attribute would come back as an `AttributeError` inside
    the learner's constructor. Everything DDP does not define itself is therefore forwarded to the
    model, so the wrapper is what the learner calls -- which is what arms the gradient all-reduce,
    since DDP only prepares the backward pass from inside its own forward.
    """
    import torch  # noqa: PLC0415  # An inactive backend's framework may not be installed.

    class _KerasDistributedDataParallel(torch.nn.parallel.DistributedDataParallel):
        """A `DistributedDataParallel` forwarding the Keras half of the surface to the model."""

        def __getattr__(self, name: str) -> Any:
            """Return DDP's own attribute, or the wrapped model's."""
            try:
                return super().__getattr__(name)
            except AttributeError:
                if name == "module":
                    raise
                return getattr(self.module, name)

    return _KerasDistributedDataParallel(model)


# `AXIS`, `PRESET_RULES`, `REJECTED` and `TACTICS` are listed because the LazySelectedImporter tail
# below only exposes the names in `__all__`, and a caller naming a preset or writing a rule table
# reads them.
__all__ = ["AXIS", "PRESET_RULES", "REJECTED", "TACTICS", "KerasDistributedStrategy", "RuleModelParallel"]


if not TYPE_CHECKING:
    import sys

    from structcast.utils.lazy_import import LazySelectedImporter

    sys.modules[__name__] = LazySelectedImporter(__name__, globals())
