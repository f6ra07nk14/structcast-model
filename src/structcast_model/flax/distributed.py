"""The distributed strategy of a Flax (nnx) training run: one mesh and one sharding rule table.

JAX expresses single-device, data-parallel and fully-sharded execution with the same mechanism -- a
device mesh plus a `PartitionSpec` per array -- so there is one strategy class here instead of the
three torch has, and what distinguishes the modes is a preset naming the mesh to build and the rules
deciding each parameter's spec. The mesh is activated process-wide at construction, before the
models are built, because eager sharding reads it there.
"""

from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from re import Pattern, compile as re_compile
from typing import TYPE_CHECKING, Any, Literal

import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec
import numpy as np

from flax import nnx
from structcast_model.flax.utils import get_jax_device

AXIS = "data"
"""The mesh axis every preset builds: batches are split along it, FSDP shards along it."""

MODEL_AXIS = "model"
"""The second mesh axis the `tp` presets add: layers are split along it, batches never are."""

PRESET_RULES: Mapping[str, tuple[tuple[str, str], ...]] = {
    "single": ((r".*", "replicate"),),
    "dp": ((r".*", "replicate"),),
    "fsdp": ((r".*", "fsdp"),),
    # No default plan: which layers pair up into a column/row split is the model's own shape, and an
    # empty table leaves every parameter on the sharding it was constructed with, so a template that
    # annotates itself needs no rules at all.
    "tp": (),
    "fsdp_tp": ((r".*", "fsdp"),),
}
"""Ordered (parameter-path regex, tactic) rules of each preset; the first matching rule wins."""

TACTICS = ("replicate", "fsdp", "column", "row")
"""The tactics a rule may name: keep the parameter on every device, shard it along the data axis, or
split it along the model axis by its last dimension (`column`) or its first one (`row`)."""

TP_PRESETS = ("tp", "fsdp_tp")
"""The presets whose mesh has a model axis, and whose unmatched parameters keep their own sharding."""


def _axis_type(mode: str) -> Any:
    """The mesh axis type one of the `*_axis_mode` fields names."""
    return jax.sharding.AxisType.Explicit if mode == "explicit" else jax.sharding.AxisType.Auto


def _to_host(value: Any) -> Any:
    """Copy one state leaf to host memory, typed RNG keys as their raw key data."""
    if not isinstance(value, jax.Array):
        return value
    if jnp.issubdtype(value.dtype, jax.dtypes.prng_key):
        value = jax.random.key_data(value)
    return np.asarray(value)


def _from_host(live: Any, saved: Any) -> Any:
    """Restore one state leaf, placed on the sharding and dtype the live array currently has.

    Taking the placement from the live object is what makes a checkpoint topology-independent: the
    saved arrays are host numpy, so a state saved on four devices loads onto one and back.
    """
    if not isinstance(live, jax.Array):
        return saved
    if jnp.issubdtype(live.dtype, jax.dtypes.prng_key):
        # A typed key is placed through its raw data: a key array's own sharding describes the
        # physical uint32 array, whose rank is one higher than the key's.
        data = jax.random.key_data(live)
        return jax.random.wrap_key_data(
            jax.device_put(jnp.asarray(saved, dtype=data.dtype), data.sharding), impl=jax.random.key_impl(live)
        )
    return jax.device_put(jnp.asarray(saved, dtype=live.dtype), live.sharding)


def _host_state(obj: Any) -> dict[str, Any]:
    """Return the full state of an nnx object -- parameters, statistics and RNG state -- in host memory."""
    return jax.tree.map(_to_host, nnx.to_pure_dict(nnx.state(obj)))


def _missing_model_state(name: str) -> ValueError:
    """The error a resume raises when the saved state holds nothing for one of the live models.

    Worded as on the torch side: the two frameworks share the checkpoint contract, so a run that
    outgrew its checkpoint has to read the same way in both.
    """
    return ValueError(
        f'The saved training state carries no state for model "{name}": it was written before that model '
        "was declared -- an EMA shadow added to the learner since, most likely. Resume with the learner "
        "the checkpoint was saved from, or start a fresh run."
    )


def _load_pure_state(obj: Any, saved: Mapping[str, Any]) -> None:
    """Write a saved host state back into an nnx object in place, keeping its identity and metadata.

    The live state is read first so every leaf is restored against its current dtype and sharding.
    """
    state = nnx.state(obj)
    restored = jax.tree.map(_from_host, nnx.to_pure_dict(state), dict(saved))
    nnx.replace_by_pure_dict(state, restored)
    nnx.update(obj, state)


@dataclass(kw_only=True)
class FlaxDistributedStrategy:
    """Strategy owning the mesh a Flax run trains on and the sharding of its parameters.

    Satisfies the torch `DistributedStrategy` protocol structurally, so the trainer and the
    checkpointing callbacks treat both backends alike. Constructing the strategy activates its mesh
    for the rest of the process (`jax.set_mesh` is a global setter that takes effect at `__init__`),
    which is what makes models built afterwards land on it.
    """

    preset: Literal["single", "dp", "fsdp", "tp", "fsdp_tp"] = "single"
    """Which mesh and rule table to use: one device, replicated parameters, parameters sharded along
    the data axis, layers split along the model axis, or both at once."""

    device: str | None = None
    """Device the `single` preset runs on, e.g. `"cpu:0"`; the first available device by default."""

    devices: int | None = None
    """How many devices the multi-device presets span; every available device by default."""

    model_devices: int | None = None
    """How many of them the model axis spans under the `tp` presets; every device under `tp`, and
    required under `fsdp_tp`, whose data axis is what is left over."""

    data_axis_mode: Literal["auto", "explicit"] = "auto"
    """Axis type of the data axis, which every preset builds. `auto` lets the compiler propagate the
    shardings no annotation named, which is what lets a model carrying none at all trace;
    `explicit` types the axis, and a typed axis demands an `out_sharding` wherever a replicated array
    meets a sharded one -- including inside flax's own code, where no template can put one:
    `nnx.Embed`'s gather of a replicated table with sharded indices and a class token concatenated
    onto a sharded activation both raise `ShardingTypeError` under it, which is why hardware
    validation demoted it from the default. Opt in for a run whose every model names its own
    shardings. Independent of :attr:`model_axis_mode`, except through `nnx.with_partitioning`, whose
    initializer path refuses a mesh of mixed axis types ("Mesh must have all axes as Explicit or all
    axes as Auto"), so a template built from those sets both fields to `explicit`."""

    model_axis_mode: Literal["auto", "explicit"] = "auto"
    """Axis type of the model axis, under the `tp` presets that build one. `auto` lets the compiler
    place what the rules do not, so plain layers row-parallelize with no model-template change;
    `explicit` types the axis for a template that names the sharding of its own outputs, which needs
    every row-parallel layer to carry its own `dot_general` hook and is verified at :meth:`wrap` time.
    That hook's spec may name Explicit axes only, so it names the data axis --
    `dot_general_out("data", None)` -- when :attr:`data_axis_mode` is `explicit` too, and
    `dot_general_out(None, None)` otherwise. Independent of that field: an annotated tensor-parallel
    template wants exactly the mix, its model axis typed while the data axis keeps propagating
    through the library-internal ops no template can annotate -- with `nnx.with_partitioning`
    initializers the one thing that refuses a mixed mesh."""

    rules: Sequence[tuple[str, str]] | None = None
    """Rules replacing the preset's table, as ordered (parameter-path regex, tactic) pairs."""

    min_size: int = 2**20
    """Parameters smaller than this many bytes stay replicated, so biases and norm scales do not shard.
    The cutoff is the `fsdp` tactic's alone: a `column`/`row` rule names one layer of a plan whose other
    half is named too, and silently dropping one of the pair is worse than sharding a small kernel."""

    _mesh: Any = field(default=None, init=False, repr=False)
    _rules: tuple[tuple[Pattern[str], str], ...] = field(default=(), init=False, repr=False)

    def __post_init__(self) -> None:
        """Validate the preset and its rules, then build and activate the mesh.

        Raises:
            ValueError: if the preset is unknown, a rule names an unknown tactic or one the preset's
                mesh has no axis for, or the number of devices asked for is not one JAX can give.
        """
        if self.preset not in PRESET_RULES:
            raise ValueError(f"Unknown preset {self.preset!r}. Available presets: {', '.join(PRESET_RULES)}.")
        rules = PRESET_RULES[self.preset] if self.rules is None else self.rules
        if self.preset in TP_PRESETS and not rules:
            raise ValueError(
                f"The {self.preset!r} preset splits the layers its rules name across the model axis, and the "
                '"tp" table is empty because which layers pair up into a column/row split is the model\'s own '
                "shape. Without rules every parameter would keep its construction sharding, so the run would "
                "replicate the whole model and report success. Bind rules through the strategy pattern "
                '(cfg/flax/strategies/tp.yaml ships one), or select the "fsdp" preset, which shards by size.'
            )
        for _, tactic in rules:
            if tactic not in TACTICS:
                raise ValueError(f"Unknown sharding tactic {tactic!r}. Available tactics: {', '.join(TACTICS)}.")
            if tactic in ("column", "row") and self.preset not in TP_PRESETS:
                raise ValueError(
                    f'The {tactic!r} tactic splits a parameter along the "{MODEL_AXIS}" axis, which the '
                    f"{self.preset!r} preset's mesh does not have: select one of {', '.join(TP_PRESETS)}."
                )
        self._rules = tuple((re_compile(pattern), tactic) for pattern, tactic in rules)
        if self.preset == "single":
            devices = [get_jax_device(self.device)]
        else:
            available = jax.devices()
            if self.devices is not None and not 1 <= self.devices <= len(available):
                raise ValueError(
                    f"Asked for {self.devices} devices, but JAX exposes {len(available)}: "
                    f"the count must be between 1 and {len(available)}."
                )
            devices = available[: self.devices]
        self._mesh = self._build_mesh(devices)
        jax.set_mesh(self._mesh)

    def _build_mesh(self, devices: Sequence[Any]) -> Any:
        """Build the mesh of this preset over *devices*: one data axis, plus a model axis for `tp`.

        Each axis takes the type its own field names, :attr:`data_axis_mode` and
        :attr:`model_axis_mode`, both Auto by default and free to differ; under an Auto model axis a
        plain layer -- one that names no output sharding of its own -- still row-parallelizes, the
        compiler inserting the reduction. The data axis was Explicit until H200 validation showed
        the price: an Explicit axis demands an `out_sharding` wherever a
        replicated array meets a sharded one, and those meetings happen inside code no template can
        annotate -- `nnx.Embed`'s gather of a replicated table with sharded indices, a class token
        concatenated onto a sharded activation -- so correct models did not trace at all. Under Auto
        the compiler propagates the shardings itself, and one it chooses badly costs a reshard rather
        than a wrong answer.

        Raises:
            ValueError: if a preset without a model axis was given one to configure, if `fsdp_tp` was
                given no model axis size, or if the size does not divide the devices the run spans.
        """
        if self.preset not in TP_PRESETS:
            if self.model_devices is not None or self.model_axis_mode != "auto":
                raise ValueError(
                    f"model_devices and model_axis_mode configure the model axis, which the {self.preset!r} "
                    f"preset's mesh does not have: drop them, or select one of {', '.join(TP_PRESETS)}."
                )
            # Named rather than left to `jax.make_mesh`, whose default axis type has changed between
            # jax versions -- and the type is what decides whether a plain model traces at all.
            return jax.make_mesh(
                (len(devices),), (AXIS,), devices=devices, axis_types=(_axis_type(self.data_axis_mode),)
            )
        if self.preset == "fsdp_tp" and self.model_devices is None:
            raise ValueError(
                'The "fsdp_tp" preset splits its devices between the two axes, so model_devices says how '
                "many of them the model axis spans; without it the data axis would be one device wide and "
                'nothing would be sharded across it. Bind model_devices, or select the "tp" preset.'
            )
        model_devices = self.model_devices or len(devices)
        if not 1 <= model_devices <= len(devices) or len(devices) % model_devices:
            raise ValueError(
                f"Asked for {model_devices} model-axis devices out of {len(devices)}: the count must be "
                "between 1 and the number of devices the run spans, and must divide it."
            )
        return jax.make_mesh(
            (len(devices) // model_devices, model_devices),
            (AXIS, MODEL_AXIS),
            devices=devices,
            axis_types=(_axis_type(self.data_axis_mode), _axis_type(self.model_axis_mode)),
        )

    @property
    def mesh(self) -> Any:
        """The mesh this strategy activated, e.g. for placing a batch or reading its size."""
        return self._mesh

    @property
    def data_rank(self) -> int:
        """0: a JAX run is single-controller, so its one process reads every batch whole."""
        return 0

    @property
    def data_world_size(self) -> int:
        """1: the mesh splits each batch across the devices, not the dataset across processes."""
        return 1

    def wrap(self, models: "OrderedDict[str, nnx.Module]") -> "OrderedDict[str, nnx.Module]":
        """Place every parameter on the sharding its rule asks for, and return the same models.

        The modules are not wrapped: sharding is a property of their arrays, so the placement is an
        in-place `nnx.update` that leaves every module object -- and the step closures capturing them
        -- untouched. Models built under the activated mesh are already replicated, so `single` and
        `dp` move nothing; an optimizer built after this call inherits the parameters' shardings for
        its own state. Under the `tp` presets a parameter no rule matched is left exactly as it was
        built, so a template's own annotations survive a strategy that says nothing about them.

        Raises:
            ValueError: if a rule matched no parameter of any model, or -- under
                `model_axis_mode="explicit"` -- if a row-parallel layer carries no `dot_general` hook
                to name its output sharding with.
        """
        self._check_rules_matched(models)
        for name, model in models.items():
            state = nnx.state(model, nnx.Param)
            pure = nnx.to_pure_dict(state)
            if self.model_axis_mode == "explicit":
                self._check_row_hooks(name, model, pure)
            placed = jax.tree_util.tree_map_with_path(self._place, pure)
            nnx.replace_by_pure_dict(state, placed)
            nnx.update(model, state)
        return models

    def sync_initial_weights(self, models: Mapping[str, nnx.Module]) -> None:
        """Nothing to synchronize: JAX is single-controller, so one process initializes every device."""

    def compile(self, module: Any, compile_kw: Mapping[str, Any] | None) -> Any:
        """Return *module* compiled with `nnx.jit`, or unchanged when *compile_kw* is None.

        The caller owns the compilation arguments, including which of a generated step's arguments
        are static and which are donated.
        """
        if compile_kw is None:
            return module
        return nnx.jit(module, **compile_kw)

    def shard_batch(self, batch: Mapping[str, Any]) -> dict[str, Any]:
        """Split a batch across the mesh along its leading dimension and commit it to the devices.

        Args:
            batch (Mapping[str, Any]): The batch to place, keyed by input name.

        Returns:
            dict[str, Any]: The same batch, placed.

        Raises:
            ValueError: if an entry has no leading dimension, or one the data axis does not divide.
        """
        # The data axis alone: on a two-dimensional mesh every device of one model axis group runs
        # the same items, so a batch split by the whole mesh would be as many times too small.
        size = self._mesh.shape[AXIS]
        for key, value in batch.items():
            for leaf in jax.tree.leaves(value):
                shape = jnp.shape(leaf)
                if not shape or shape[0] % size:
                    raise ValueError(
                        f'Batch entry "{key}" of shape {shape} cannot be split across the {size} devices of '
                        f'the "{AXIS}" axis: its leading dimension must exist and be divisible by that '
                        "axis' size, which is not the mesh's own once a model axis is added."
                    )
        return jax.device_put(dict(batch), NamedSharding(self._mesh, PartitionSpec(AXIS)))

    def state_dict(
        self,
        models: Mapping[str, nnx.Module],
        optimizers: Mapping[str, Any] | None = None,
        optimizer_models: Mapping[str, list[str]] | None = None,
    ) -> dict[str, Any]:
        """Produce `{"models": ..., "optimizers": ...}` in host memory, keyed by model and optimizer name.

        The full state travels -- parameters, batch statistics and RNG state -- so a restored run
        continues rather than restarts. *optimizer_models* is accepted for protocol compatibility and
        unused: nnx optimizer state is already keyed by parameter path.
        """
        states: dict[str, Any] = {"models": {name: _host_state(model) for name, model in models.items()}}
        if optimizers is not None:
            states["optimizers"] = {name: _host_state(optimizer) for name, optimizer in optimizers.items()}
        return states

    def load_state_dict(
        self,
        models: Mapping[str, nnx.Module],
        optimizers: Mapping[str, Any],
        optimizer_models: Mapping[str, list[str]] | None,
        state: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Load a state produced by :meth:`state_dict` into the live models and optimizers.

        Returns the state itself, so the caller reads its non-array parts (`meta`, `grad_scalers`).

        Raises:
            ValueError: if no state was given.
        """
        if state is None:
            raise ValueError("A training state is required to resume from.")
        model_states = state.get("models", {})
        optimizer_states = state.get("optimizers", {})
        for name, model in models.items():
            # A state holding models the learner no longer has is ignored; one missing a model the
            # learner does have is a resume the checkpoint cannot answer, not a `KeyError`.
            if name not in model_states:
                raise _missing_model_state(name)
            _load_pure_state(model, model_states[name])
        for name, optimizer in optimizers.items():
            _load_pure_state(optimizer, optimizer_states[name])
        return state

    def _tactic(self, name: str) -> str | None:
        """The tactic of the first rule matching the parameter path *name*, or None when none does."""
        for pattern, tactic in self._rules:
            if pattern.search(name):
                return tactic
        return None

    def _check_rules_matched(self, models: Mapping[str, Any]) -> None:
        """Refuse a rule table holding a pattern no parameter of any model matches.

        A rule that matches nothing is a typo, and its cost is invisible: the parameters it meant to
        split keep whatever placement the preset gives them and the run trains, slower or larger, with
        no sign that the plan never applied. Matching nothing *anywhere* is the error; matching
        nothing in one model of several is normal, as it is for the torch `shard_modules` globs.

        Raises:
            ValueError: naming the patterns that matched nothing, with real paths to write instead.
        """
        names = [
            jax.tree_util.keystr(path, simple=True, separator=".")
            for model in models.values()
            for path, _ in jax.tree_util.tree_flatten_with_path(nnx.to_pure_dict(nnx.state(model, nnx.Param)))[0]
        ]
        unmatched = [pattern.pattern for pattern, _ in self._rules if not any(pattern.search(n) for n in names)]
        if unmatched:
            raise ValueError(
                f"Sharding rule pattern(s) {unmatched} matched no parameter; available parameter paths "
                f"include {names[:10]}."
            )

    def _check_row_hooks(self, name: str, model: Any, pure: Mapping[str, Any]) -> None:
        """Refuse a row-parallel layer that cannot name the sharding of its own output.

        A row-parallel layer contracts over the axis its kernel is split on, so its result is a
        partial sum every shard holds a piece of. Under an Auto model axis the compiler inserts the
        reduction; under an Explicit one it may not choose, and the layer has to say where the result
        lands -- which is what :func:`structcast_model.flax.utils.dot_general_out` is for. The check
        runs before anything is placed, so the run stops at configuration rather than deep inside the
        first traced step. It fails closed: a row-matched parameter whose layer this walk cannot
        resolve is reported with the rest, because "not found" is not evidence of a hook.

        Raises:
            ValueError: naming every unverifiable layer and the template line that fixes it.
        """
        modules = {
            ".".join(str(part) for part in path): node
            for path, node in nnx.iter_graph(model)
            if isinstance(node, nnx.Module)
        }
        unverified = []
        for path, _ in jax.tree_util.tree_flatten_with_path(pure)[0]:
            parameter = jax.tree_util.keystr(path, simple=True, separator=".")
            if self._tactic(parameter) != "row":
                continue
            layer = modules.get(parameter.rpartition(".")[0])
            # `None` and `jax.lax.dot_general` are both "the default": nnx.Linear stores the function
            # itself, nnx.LinearGeneral stores None and resolves it per call.
            if layer is not None and getattr(layer, "dot_general", None) not in (None, jax.lax.dot_general):
                continue
            unverified.append(parameter)
        if unverified:
            # An `out_sharding` may name Explicit axes only, so the batch dimension of the hook's spec
            # is the data axis under an Explicit data axis and None under the Auto default.
            batch = f"'{AXIS}'" if self.data_axis_mode == "explicit" else "None"
            raise ValueError(
                f'Model "{name}" row-parallelizes {unverified}, whose layers compute with the default '
                f'dot_general (or could not be resolved), and model_axis_mode is "explicit", so the '
                f'"{MODEL_AXIS}" axis is typed and each of those layers must name the sharding its output '
                f"lands on. Give the layer the hook in the model template -- "
                f'dot_general: "eval: dot_general_out({batch}, None)" -- or run the strategy with '
                "model_axis_mode: auto, which lets the compiler place the result itself."
            )

    def _place(self, path: Any, array: Any) -> Any:
        """Place one parameter on the sharding its first matching rule asks for.

        Under the `tp` presets an unmatched parameter is returned untouched, keeping whatever
        sharding it was constructed with; every other preset replicates what no rule shards.
        """
        tactic = self._tactic(jax.tree_util.keystr(path, simple=True, separator="."))
        if tactic is None and self.preset in TP_PRESETS:
            return array
        return jax.device_put(array, NamedSharding(self._mesh, self._spec(tactic, array)))

    def _spec(self, tactic: str | None, array: Any) -> PartitionSpec:
        """Return the spec one tactic asks for on one parameter."""
        if tactic == "fsdp":
            return self._fsdp_spec(array)
        if tactic == "column":
            # The bias of a column-parallel layer splits with the kernel's output dimension, so a
            # one-dimensional array is a candidate here where the other tactics leave it whole.
            return self._model_spec(array, array.ndim - 1)
        if tactic == "row":
            # The bias is pinned replicated, and the tactic is where that lives because a rule table
            # cannot say it: a bias split along the model axis -- or added once per shard -- is
            # counted as many times as the axis is wide by the reduction that follows, the one
            # tensor-parallel mistake that reports a plausible loss instead of an error.
            return PartitionSpec() if array.ndim < 2 else self._model_spec(array, 0)
        return PartitionSpec()

    def _model_spec(self, array: Any, dim: int) -> PartitionSpec:
        """Return the spec splitting *dim* of *array* across the model axis, or replicated.

        A dimension the model axis does not divide falls back to replication rather than failing the
        run, as the `fsdp` tactic does: the result is the same numbers computed without the split.
        """
        if array.shape[dim] % self._mesh.shape[MODEL_AXIS]:
            return PartitionSpec()
        return PartitionSpec(*(MODEL_AXIS if d == dim else None for d in range(array.ndim)))

    def _fsdp_spec(self, array: Any) -> PartitionSpec:
        """Return the FSDP spec of one parameter: its leading dimension split, or replicated.

        Only the leading dimension of a parameter with at least two dimensions is a candidate.
        Sharding any other dimension puts the parameter's own axis on the mesh axis the batch is
        already split along, which is not FSDP: the ops that consume it meet `data` on two different
        dimensions, and the compiler pays a reshard every step to reconcile them. A parameter below
        :attr:`min_size`, or whose leading dimension the data axis does not divide, stays replicated
        rather than failing the run. The divisor is the data axis and not the mesh: on the
        two-dimensional `fsdp_tp` mesh the whole mesh is wider, and dividing by it would leave every
        parameter the data axis alone would have sharded replicated instead.
        """
        if array.ndim < 2 or array.nbytes < self.min_size or array.shape[0] % self._mesh.shape[AXIS]:
            return PartitionSpec()
        return PartitionSpec(AXIS, *([None] * (array.ndim - 1)))


# The module constants are listed because the LazySelectedImporter tail below only exposes the names
# in `__all__`, and a caller naming a preset or writing a rule table reads them.
__all__ = ["AXIS", "MODEL_AXIS", "PRESET_RULES", "TACTICS", "TP_PRESETS", "FlaxDistributedStrategy"]


if not TYPE_CHECKING:
    import sys

    from structcast.utils.lazy_import import LazySelectedImporter

    sys.modules[__name__] = LazySelectedImporter(__name__, globals())
