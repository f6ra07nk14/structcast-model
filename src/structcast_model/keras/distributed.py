"""The distributed strategy of a Keras training run: one preset per backend-native mechanism.

Keras 3 has no distribution mechanism of its own that works everywhere: `keras.distribution` is
implemented for JAX, is an explicitly labelled no-op prototype on TensorFlow (its
`distribute_value` is a bare `pass`), and does not exist on torch. So the strategy here is one
class holding one preset, and the preset is realized by whatever the active backend actually
supports:

| preset   | jax                                    | tensorflow                        | torch                    |
| -------- | -------------------------------------- | --------------------------------- | ------------------------ |
| `single` | nothing to activate                    | nothing to activate               | nothing to activate      |
| `dp`     | `keras.distribution.DataParallel`      | `tf.distribute.MirroredStrategy`  | `DistributedDataParallel`|
| `fsdp`   | `keras.distribution.ModelParallel`     | rejected                          | rejected                 |
| `tp`     | `ModelParallel` on a 2-D mesh          | rejected                          | rejected                 |

The backend is read exactly once, in `__post_init__`, and every unsupported cell is rejected there
-- before a model exists -- because each rejected cell fails silently otherwise: a
`keras.distribution` sharding on TensorFlow replicates every variable without a word, torch has no
`keras.distribution` implementation at all, and torch FSDP2 rebinds the parameters a Keras variable
caches, so the run keeps training the stale ones.

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
from structcast_model.keras.adapters import AdapterSegment
from structcast_model.keras.utils import apply_state_dict, collect_state_dict, get_keras_device

if TYPE_CHECKING:
    import tensorflow as tf

    import torch
    import torch.distributed as dist
else:
    from structcast.utils.lazy_import import LazyModuleImporter

    # An inactive backend's framework may not be installed, so each one is bound lazily and only
    # resolved inside the preset paths the active backend can reach, as in `loggers.state_backends`.
    tf = LazyModuleImporter("tensorflow")
    torch = LazyModuleImporter("torch")
    dist = LazyModuleImporter("torch.distributed")

AXIS = "batch"
"""The mesh axis every preset builds: batches split along it, FSDP shards along it.

Keras' own `DEFAULT_BATCH_DIM_NAME`, so a `DataParallel` mesh and the `ModelParallel` mesh built
here name their axis identically and a rule table reads the same under both presets.
"""

MODEL_AXIS = "model"
"""The second mesh axis the `tp` preset adds: layers split along it, batches never are."""

PRESET_RULES: Mapping[str, tuple[tuple[str, str], ...]] = {
    "single": ((r".*", "replicate"),),
    "dp": ((r".*", "replicate"),),
    "fsdp": ((r".*", "fsdp"),),
    # No default plan: which layers pair up into a column/row split is the model's own shape, and
    # unlike the Flax twin a Keras variable carries no annotation of its own to fall back on -- so
    # the preset is refused without rules rather than replicating everything and reporting success.
    "tp": (),
}
"""Ordered (variable-path regex, tactic) rules of each preset; the first matching rule wins."""

TACTICS = ("replicate", "fsdp", "column", "row")
"""The tactics a rule may name: keep the variable on every device, shard it along the batch axis, or
split it along the model axis by its last dimension (`column`) or its first one (`row`)."""

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
    ("tp", "tensorflow"): (
        'The "tp" preset is not available on the tensorflow Keras backend: splitting a layer across devices goes '
        "through keras.distribution, which is a no-op prototype there (its backend distribute_value does nothing "
        "at all), so every variable would be replicated, every device would compute the whole layer, and the run "
        'would report success. Use the "dp" preset, which runs on tf.distribute.MirroredStrategy, or run tp on '
        "the jax backend."
    ),
    ("tp", "torch"): (
        'The "tp" preset is not available on the torch Keras backend: keras.distribution has no torch '
        "implementation at all, so the mesh and the variable layouts a split needs do not exist, and the run "
        'would train replicated variables under a strategy that says it split them. Use the "dp" preset, which '
        "wraps every model in DistributedDataParallel, or run tp on the jax backend."
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
        """Return the layout of one variable: one of its dimensions split, or replicated.

        `fsdp` splits the leading dimension along the batch axis. `column` and `row` split along the
        model axis of the two-dimensional `tp` mesh -- the last dimension for `column`, the leading
        one for `row` -- which is the pair a tensor-parallel layer is made of. A one-dimensional
        variable is a candidate for `column` alone, because a column-parallel bias splits with the
        output dimension it belongs to, while a row-parallel one has to stay whole: the reduction
        that follows a row-parallel layer would count a split bias once per shard, which is the one
        tensor-parallel mistake that reports a plausible loss instead of an error.
        A dimension the mesh does not divide falls back to replication rather than failing the run --
        the same shape the Flax twin's rules have.
        """
        if getattr(variable, "_layout", None) is not None:
            return variable._layout  # noqa: SLF001  # The base class reads it first too; a caller may pin a layout.
        mesh = self.device_mesh
        axes: list[str | None] = [None] * len(variable.shape)
        for pattern, tactic in self._rules:
            if pattern.search(variable.path):
                rank = len(axes)
                if tactic == "column" and rank:
                    axis, dim = MODEL_AXIS, rank - 1
                elif tactic in ("fsdp", "row") and rank > 1:
                    axis, dim = AXIS if tactic == "fsdp" else MODEL_AXIS, 0
                else:
                    break
                if not variable.shape[dim] % mesh.shape[mesh.axis_names.index(axis)]:
                    axes[dim] = axis
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

    preset: Literal["single", "dp", "fsdp", "tp"] = "single"
    """Which mechanism to run on: one device, replicated variables, variables sharded along the batch
    axis, or layers split along a model axis."""

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
    """Rules replacing the sharding presets' tables, as ordered (variable-path regex, tactic) pairs.
    Optional under `fsdp`, which has a table of its own, and required under `tp`, which has none."""

    _backend: str = field(default="", init=False, repr=False)
    _rules: tuple[tuple[Pattern[str], str], ...] = field(default=(), init=False, repr=False)
    _distribution: Any = field(default=None, init=False, repr=False)
    _mirrored: Any = field(default=None, init=False, repr=False)
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
        if self.rules is not None and self.preset not in ("fsdp", "tp"):
            raise ValueError(
                f'Sharding rules only decide how the "fsdp" and "tp" presets split variables, but the preset '
                f"is {self.preset!r}, which replicates them all: drop the rules, or select one of those."
            )
        if self.rules is None and self.preset == "tp":
            raise ValueError(
                'The "tp" preset splits the layers its rules name across the model axis, and has no table of '
                "its own -- which layers pair up into a column/row split is the model's own shape. Bind rules "
                'such as [["kernel$", column]], or select the "fsdp" preset, which shards by size.'
            )
        rules = PRESET_RULES[self.preset] if self.rules is None else self.rules
        for _, tactic in rules:
            if tactic not in TACTICS:
                raise ValueError(f"Unknown sharding tactic {tactic!r}. Available tactics: {', '.join(TACTICS)}.")
            if tactic in ("column", "row") and self.preset != "tp":
                raise ValueError(
                    f'The {tactic!r} tactic splits a variable along the "{MODEL_AXIS}" axis, which the '
                    f"{self.preset!r} preset's mesh does not have: select the tp preset."
                )
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
    def data_rank(self) -> int:
        """Which slice of the dataset this process must consume: its rank.

        The same as :attr:`rank`, and a separate member because the `DistributedStrategy` protocol
        asks for it: the model axis of the `tp` preset lives inside one process, so
        no Keras run ever has ranks that must share a slice.
        """
        return self._rank

    @property
    def data_world_size(self) -> int:
        """How many distinct dataset slices the run is split into: its world size."""
        return self._world_size

    @property
    def replicas(self) -> int:
        """How many replicas the run is spread over, whichever mechanism spreads it."""
        if self._distribution is not None:
            return int(self._distribution.num_model_replicas)
        if self._mirrored is not None:
            return int(self._mirrored.num_replicas_in_sync)
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

        self._mirrored = tf.distribute.MirroredStrategy(devices=self._device_names())
        # The scope object is held, not dropped: it owns the variable-creator scope it entered, and
        # letting it be collected tears that down early -- after which the models are built as plain,
        # unmirrored variables and the first step fails inside `strategy.run`.
        self._scope = self._mirrored.scope()
        with self._scope:
            yield

    def wrap(self, models: "OrderedDict[str, Any]") -> "OrderedDict[str, Any]":
        """Return the models the learner should be built over.

        Only torch has anything to wrap: a Keras variable *is* the `torch.nn.Parameter` it holds, so
        `DistributedDataParallel` averages the gradients straight into the `.grad` the backend
        adapter reads off it. JAX and TensorFlow variables carry their placement from the moment
        they were created under the activated distribution, so their models come back untouched.

        It is also the first moment the rule table can be checked against real variables, which is
        why the typo check below lives here rather than in `__post_init__`.

        Raises:
            ValueError: if a rule matched no variable of any model.
        """
        self._check_rules_matched(models)
        if self._backend != "torch" or self.preset == "single":
            return models
        return OrderedDict((name, _wrap_ddp(model)) for name, model in models.items())

    def _check_rules_matched(self, models: Mapping[str, Any]) -> None:
        """Refuse a rule table holding a pattern no variable of any model matches.

        A rule that matches nothing is a typo whose cost is invisible: the variables it meant to
        split stay replicated and the run trains, larger and slower, with nothing to read. Only the
        rules that actually decide a layout are checked -- on TensorFlow and torch the table never
        runs, and `keras.distribution` is not what those presets are realized by.

        Raises:
            ValueError: naming the patterns that matched nothing, with real paths to write instead.
        """
        if not isinstance(self._distribution, RuleModelParallel):
            return
        paths = [variable.path for model in models.values() for variable in model.variables]
        unmatched = [pattern.pattern for pattern, _ in self._rules if not any(pattern.search(p) for p in paths)]
        if unmatched:
            raise ValueError(
                f"Sharding rule pattern(s) {unmatched} matched no variable; available variable paths "
                f"include {paths[:10]}."
            )

    def sync_initial_weights(self, models: Mapping[str, Any]) -> None:
        """Make every rank start from rank 0's weights. Call on every rank, before :meth:`wrap`.

        Nothing to do outside torch: JAX and TensorFlow runs are single-controller, so one process
        initializes every device.
        """
        if self._backend != "torch" or self.preset == "single":
            return

        for model in models.values():
            for variable in model.variables:
                dist.broadcast(variable.value.data, src=0)

    def wrap_steps(self, learner: Any) -> None:
        """Rewire the learner's steps so each one runs across the replicas and reports one value.

        What reaches the tracker must already be reduced across replicas, and this is the place that
        does it. The three backends need three different things, and none of them can happen inside
        the learner, which is backend-neutral by construction:

        - JAX places the batch across the mesh; the reductions inside the compiled step are then
          global by construction, since XLA reduces a sharded array across every device holding it.
        - TensorFlow splits the batch per replica, runs the step through `MirroredStrategy.run` and
          reduces the per-replica criteria with `ReduceOp.MEAN`.
        - torch runs the step as it is -- the loader owns the rank's slice of the data -- and
          all-reduces the criteria, since each rank only ever saw its own slice.

        The *gradients* are made to agree here: `dp` means the mean of the per-replica gradients on
        every backend, which JAX gets from the sharded step and torch from
        `DistributedDataParallel`. TensorFlow gets it from the scaling below, because the Keras
        TensorFlow optimizer all-reduces the per-replica gradients with `ReduceOp.SUM`
        (`_all_reduce_sum_gradients`, the behaviour of `Model.fit` under a strategy too) and an
        unscaled run would therefore take a step as many times larger as it has replicas.

        Raises:
            ValueError: under `MirroredStrategy`, if the learner exposes no `flow_functions`.
        """
        if self.preset == "single":
            return
        if self._mirrored is not None:
            # The adapter traced each step into a `tf.function`, and a graph applying an optimizer is
            # a synchronization point TensorFlow refuses to nest inside `strategy.run`. The traced
            # steps are therefore unwrapped back to the Python functions they were built from, and
            # re-traced below inside the one graph that wraps the replicated call. `flow_functions`
            # is what a learner exposes for exactly this rebinding (as in `cmd_flax`), and what the
            # check below requires here.
            for name in learner.flow_functions:
                traced = getattr(learner, name)
                if hasattr(traced, "python_function"):
                    setattr(learner, name, traced.python_function)
            # The loss, not the gradients: it is the one value the strategy can reach from out here,
            # the segment's flow being what the adapter differentiates, and dividing it by the
            # replica count turns the optimizer's SUM all-reduce into the mean of the per-replica
            # gradients. The criteria the flow reports beside it are untouched, and still reduced
            # with `ReduceOp.MEAN` below. What is scaled here is exactly what the generated shape
            # exposes: one `AdapterSegment` per instance attribute (`docs/adr/0019`). Segments kept
            # any other way -- inside a list, behind `__slots__`, built on the fly -- are invisible
            # to this scan, so a learner holding them like that has to scale its own loss.
            for segment in getattr(learner, "__dict__", {}).values():
                if isinstance(segment, AdapterSegment):
                    segment.flow = _mean_flow(segment.flow, self.replicas)
            flows = list(learner.flow_functions)
            if flows:
                # The generated learner's public steps stay eager: `training_step` owns the host
                # counters and reads the optimizer counter back after the step (`docs/adr/0018`),
                # neither of which can run inside the replicated graph, so the strategy wraps the
                # inner flow steps it just unwrapped instead.
                for name in flows:
                    setattr(learner, name, self._replicated_flow(getattr(learner, name)))
                return
            raise ValueError(
                "A MirroredStrategy needs the learner's flow_functions to keep the host bookkeeping eager: the "
                "public steps own the training counters, which a replicated graph cannot run, so "
                "a hand-written learner must expose its flow callables the way a generated one does."
            )
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
        if self._distribution is None and self._mirrored is None:
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

    def _replicated_flow(self, flow: Any) -> Any:
        """Wrap one inner flow step -- taking the batch by name -- across the TensorFlow replicas.

        `MirroredStrategy`'s answer to what :meth:`_replicated` does on the other two backends, one
        level further in: the learner's public steps keep their host-side counter bookkeeping
        outside the graph, so the replication wraps the flow attribute the public step calls
        through. The batch stays keyword arguments the whole way down, which is what the step
        expects.
        """

        @tf.function
        def replicated(**batch: Any) -> dict[str, Any]:
            # Traced under :func:`_local_batch_losses`: a Keras loss class would otherwise hand back
            # the replica's share of the global batch's loss, which `ReduceOp.MEAN` below cannot
            # tell from the per-replica means every other criterion is.
            with _local_batch_losses():
                criteria = self._mirrored.run(flow, kwargs=batch)
            return {
                name: self._mirrored.reduce(tf.distribute.ReduceOp.MEAN, value, axis=None)
                for name, value in criteria.items()
            }

        def flow_step(**batch: Any) -> dict[str, Any]:
            return dict(replicated(**self.shard_batch(batch)))

        return flow_step

    def _replicated(self, step: Any) -> Any:
        """Wrap one public learner step so it runs across the replicas and reports reduced criteria.

        What :meth:`_replicated_flow` is for TensorFlow, this is for the other two backends -- and
        neither of them needs a graph around the replicated call, so the wrapper stays an eager
        Python function: JAX shards the batch and lets the compiled step reduce it globally, torch
        runs the rank's own slice and all-reduces the criteria afterwards. Wrapping the public step
        is therefore harmless here, and the host bookkeeping it owns still runs
        where it did.
        """
        if self._distribution is not None:

            def jax_step(**batch: Any) -> dict[str, Any]:
                return step(**self.shard_batch(batch))

            return jax_step

        def torch_step(**batch: Any) -> dict[str, Any]:

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
        return self._mirrored.experimental_distribute_values_from_function(
            lambda context: tensor[
                context.replica_id_in_sync_group * per_replica : (context.replica_id_in_sync_group + 1) * per_replica
            ]
        )

    def _jax_distribution(self) -> Any:
        """Build the `keras.distribution` the JAX presets run on.

        The `tp` mesh is two-dimensional with a batch axis of one: every device holds a slice of the
        split layers and runs the whole batch through it, so `num_model_replicas` -- which
        :attr:`replicas` and `shard_batch` read -- is 1 and the batch is placed whole.
        """
        names = self._device_names()
        if self.preset == "dp":
            return keras.distribution.DataParallel(devices=names)
        shape = (1, len(names)) if self.preset == "tp" else (len(names),)
        axis_names = [AXIS, MODEL_AXIS] if self.preset == "tp" else [AXIS]
        mesh = keras.distribution.DeviceMesh(shape=shape, axis_names=axis_names, devices=names)
        return RuleModelParallel(layout_map=keras.distribution.LayoutMap(mesh), rules=self._rules)

    def _activate_torch(self) -> None:
        """Join the process group the launcher started, and bind this rank's device.

        Raises:
            RuntimeError: if no process group is available, so the run has no ranks to spread over.
        """
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


def _mean_flow(flow: Any, replicas: int) -> Any:
    """Return *flow* with the loss it hands the tape divided by *replicas*, its criteria untouched."""

    def mean(**batch: Any) -> tuple[Any, dict[str, Any]]:
        loss, criteria = flow(**batch)
        return loss / replicas, criteria

    return mean


@contextmanager
def _local_batch_losses() -> Iterator[None]:
    """Keep a `keras.losses.Loss` normalizing by the replica's own batch while a flow is traced.

    A Keras loss reduced over the batch divides by the *global* batch under a `tf.distribute`
    strategy: `keras.losses.Loss.__call__` ends in `scale_loss_for_distribution`, which multiplies
    the replica's mean by `1 / num_replicas_in_sync` so that the SUM all-reduce the TensorFlow
    optimizer applies to the gradients lands on the global mean. Keras' own `fit` undoes that
    scaling before it reports the value (`unscale_loss_for_distribution` in its TensorFlow trainer),
    and nothing else a flow computes is scaled at all -- not an accuracy, not a `keras.ops`
    expression, not a loss written as the plain function.

    A flow's loss is both what the tape differentiates and what the tracker logs, beside criteria
    that were never scaled, so the multiply is neutralized here rather than compensated for
    downstream: every value a flow returns is then the replica's own mean, which `ReduceOp.MEAN`
    turns into the global one, and the loss is divided by the replica count exactly once --
    in :func:`_mean_flow`, for the gradients. Without this a `dp` run reported every
    `keras.losses.Loss` criterion divided by the replica count, training and validation alike, and
    trained on gradients that small too.

    The patch spans the trace rather than the step: the multiply is a graph op, written once, when
    the replicated function is traced. `keras.src.losses.loss` is the private module `keras.losses`
    is built from, so `import keras` has already bound it and the name below is the one
    `reduce_values` itself looks up -- a Keras release moving the function raises here, loudly.
    """
    module = keras.src.losses.loss
    original = module.scale_loss_for_distribution
    module.scale_loss_for_distribution = lambda value: value
    try:
        yield
    finally:
        module.scale_loss_for_distribution = original


def _wrap_ddp(model: Any) -> Any:
    """Wrap one Keras model in a `DistributedDataParallel` that still looks like the model.

    A generated learner reads `trainable_variables` off the models it was handed, and DDP proxies
    none of the Keras surface: an unwrapped attribute would come back as an `AttributeError` inside
    the learner's constructor. Everything DDP does not define itself is therefore forwarded to the
    model, so the wrapper is what the learner calls -- which is what arms the gradient all-reduce,
    since DDP only prepares the backward pass from inside its own forward.
    """

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


# The module constants are listed because the LazySelectedImporter tail below only exposes the names
# in `__all__`, and a caller naming a preset or writing a rule table reads them.
__all__ = [
    "AXIS",
    "MODEL_AXIS",
    "PRESET_RULES",
    "REJECTED",
    "TACTICS",
    "KerasDistributedStrategy",
    "RuleModelParallel",
]


if not TYPE_CHECKING:
    import sys

    from structcast.utils.lazy_import import LazySelectedImporter

    sys.modules[__name__] = LazySelectedImporter(__name__, globals())
