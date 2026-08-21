"""The distributed strategy of a Flax (nnx) training run: one mesh and one sharding rule table.

JAX expresses single-device, data-parallel and fully-sharded execution with the same mechanism -- a
device mesh plus a `PartitionSpec` per array -- so there is one strategy class here instead of the
three torch has, and what distinguishes the modes is a preset naming the mesh to build and the rules
deciding each parameter's spec (see `docs/adr/0014`). The mesh is activated process-wide at
construction, before the models are built, because eager sharding reads it there.
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
"""The single mesh axis every preset builds: batches are split along it, FSDP shards along it."""

PRESET_RULES: Mapping[str, tuple[tuple[str, str], ...]] = {
    "single": ((r".*", "replicate"),),
    "dp": ((r".*", "replicate"),),
    "fsdp": ((r".*", "fsdp"),),
}
"""Ordered (parameter-path regex, tactic) rules of each preset; the first matching rule wins."""

TACTICS = ("replicate", "fsdp")
"""The tactics a rule may name: keep the parameter on every device, or shard it across the mesh."""


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

    preset: Literal["single", "dp", "fsdp"] = "single"
    """Which mesh and rule table to use: one device, replicated parameters, or sharded parameters."""

    device: str | None = None
    """Device the `single` preset runs on, e.g. `"cpu:0"`; the first available device by default."""

    devices: int | None = None
    """How many devices the `dp` and `fsdp` presets span; every available device by default."""

    rules: Sequence[tuple[str, str]] | None = None
    """Rules replacing the preset's table, as ordered (parameter-path regex, tactic) pairs."""

    min_size: int = 2**20
    """Parameters smaller than this many bytes stay replicated, so biases and norm scales do not shard."""

    _mesh: Any = field(default=None, init=False, repr=False)
    _rules: tuple[tuple[Pattern[str], str], ...] = field(default=(), init=False, repr=False)

    def __post_init__(self) -> None:
        """Validate the preset and its rules, then build and activate the mesh.

        Raises:
            ValueError: if the preset is unknown, a rule names an unknown tactic, or the number of
                devices asked for is not one JAX can give.
        """
        if self.preset not in PRESET_RULES:
            raise ValueError(f"Unknown preset {self.preset!r}. Available presets: {', '.join(PRESET_RULES)}.")
        rules = PRESET_RULES[self.preset] if self.rules is None else self.rules
        for _, tactic in rules:
            if tactic not in TACTICS:
                raise ValueError(f"Unknown sharding tactic {tactic!r}. Available tactics: {', '.join(TACTICS)}.")
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
        self._mesh = jax.make_mesh((len(devices),), (AXIS,), devices=devices)
        jax.set_mesh(self._mesh)

    @property
    def mesh(self) -> Any:
        """The mesh this strategy activated, e.g. for placing a batch or reading its size."""
        return self._mesh

    def wrap(self, models: "OrderedDict[str, nnx.Module]") -> "OrderedDict[str, nnx.Module]":
        """Place every parameter on the sharding its rule asks for, and return the same models.

        The modules are not wrapped: sharding is a property of their arrays, so the placement is an
        in-place `nnx.update` that leaves every module object -- and the step closures capturing them
        -- untouched. Models built under the activated mesh are already replicated, so `single` and
        `dp` move nothing; an optimizer built after this call inherits the parameters' shardings for
        its own state.
        """
        for model in models.values():
            state = nnx.state(model, nnx.Param)
            placed = jax.tree_util.tree_map_with_path(self._place, nnx.to_pure_dict(state))
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
            ValueError: if an entry has no leading dimension, or one the mesh does not divide.
        """
        size = self._mesh.size
        for key, value in batch.items():
            for leaf in jax.tree.leaves(value):
                shape = jnp.shape(leaf)
                if not shape or shape[0] % size:
                    raise ValueError(
                        f'Batch entry "{key}" of shape {shape} cannot be split across {size} devices: '
                        "its leading dimension must exist and be divisible by the mesh size."
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
            _load_pure_state(model, model_states[name])
        for name, optimizer in optimizers.items():
            _load_pure_state(optimizer, optimizer_states[name])
        return state

    def _place(self, path: Any, array: Any) -> Any:
        """Place one parameter on the sharding its first matching rule asks for."""
        name = jax.tree_util.keystr(path, simple=True, separator=".")
        spec = PartitionSpec()
        for pattern, tactic in self._rules:
            if pattern.search(name):
                spec = self._fsdp_spec(array) if tactic == "fsdp" else PartitionSpec()
                break
        return jax.device_put(array, NamedSharding(self._mesh, spec))

    def _fsdp_spec(self, array: Any) -> PartitionSpec:
        """Return the FSDP spec of one parameter: its leading dimension split, or replicated.

        Only the leading dimension of a parameter with at least two dimensions is a candidate.
        Sharding any other dimension puts the parameter's own axis on the mesh axis the batch is
        already split along, and the elementwise ops that consume it then produce a result sharded
        twice on `data`, which explicit-mode JAX rejects at trace time. A parameter below
        :attr:`min_size`, or whose leading dimension the mesh does not divide, stays replicated
        rather than failing the run.
        """
        if array.ndim < 2 or array.nbytes < self.min_size or array.shape[0] % self._mesh.size:
            return PartitionSpec()
        return PartitionSpec(AXIS, *([None] * (array.ndim - 1)))


# `AXIS`, `PRESET_RULES` and `TACTICS` are listed because the LazySelectedImporter tail below only
# exposes the names in `__all__`, and a caller naming a preset or writing a rule table reads them.
__all__ = ["AXIS", "PRESET_RULES", "TACTICS", "FlaxDistributedStrategy"]


if not TYPE_CHECKING:
    import sys

    from structcast.utils.lazy_import import LazySelectedImporter

    sys.modules[__name__] = LazySelectedImporter(__name__, globals())
