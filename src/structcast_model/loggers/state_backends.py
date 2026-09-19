"""Serialization backends the loggers write and read training states through.

A logger knows *where* a training state goes; a backend knows *what it looks like* on disk. Keeping
the two apart is what lets the same logger carry a torch checkpoint for one run and a Flax one for
the next, instead of growing a subclass per framework.
"""

from collections.abc import Mapping
import json
from pathlib import Path
import tarfile
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any

from typing_extensions import Protocol

if TYPE_CHECKING:
    import numpy as np
    import orbax.checkpoint as ocp

    import torch
else:
    from structcast.utils.lazy_import import LazyModuleImporter

    # Every third-party module is bound lazily, as in `loggers.base`: importing this module must not
    # drag torch into a Flax run nor jax into a torch one. They are touched inside `save` and `load`
    # only. The Keras backend needs no keras at all -- a Keras state is numpy on the way out.
    np = LazyModuleImporter("numpy")
    ocp = LazyModuleImporter("orbax.checkpoint")
    torch = LazyModuleImporter("torch")

_JSON_ITEM = "__json__"
"""Archive member the Keras backend stores its non-array items in; `numpy` archives hold arrays only."""


class StateBackend(Protocol):
    """Reader and writer of one training-state file format.

    `load` returns state in **host memory** -- numpy arrays or CPU tensors, never device-resident
    state. Placing a restored state is the distributed strategy's job, because only the strategy
    knows the topology the resumed run has, and a backend that placed arrays itself would tie a
    checkpoint to the machine that wrote it.
    """

    suffix: str
    """Extension of the files this backend writes, leading dot included."""

    def save(self, states: Mapping[str, Any], directory: Path, name: str) -> Path:
        """Write *states* into *directory* as one file named after *name*, and return its path."""

    def load(self, path: Path) -> dict[str, Any]:
        """Read back a state this backend wrote, in host memory."""


class TorchStateBackend:
    """Training states as a single `torch.save` archive, the format every torch run has used."""

    suffix = ".pt"
    """Torch checkpoints are one pickle file."""

    def save(self, states: Mapping[str, Any], directory: Path, name: str) -> Path:
        """Pickle *states* into `directory/name.pt` and return that path."""
        path = directory / f"{name}{self.suffix}"
        torch.save(dict(states), path)
        return path

    def load(self, path: Path) -> dict[str, Any]:
        """Load a `torch.save` archive onto the host."""
        # `weights_only` because the reference is user input, and an unpickled checkpoint executes code.
        return torch.load(path, map_location="cpu", weights_only=True)


class FlaxStateBackend:
    """Training states as one gzipped tar holding an orbax checkpoint.

    Orbax writes a directory, and every transport around the loggers -- the wandb run directory, the
    `wandb://.../<file>` and `runs:/...` references, `log_artifact` -- carries a file, so the
    directory is packed into one archive on the way out and unpacked on the way in.
    """

    suffix = ".tar.gz"
    """An orbax checkpoint is a directory; it travels packed."""

    def save(self, states: Mapping[str, Any], directory: Path, name: str) -> Path:
        """Write *states* as an orbax checkpoint and pack it into `directory/name.tar.gz`.

        Each top-level entry becomes one orbax item: an array tree is saved by the standard handler,
        anything JSON-serializable -- `meta`, and the `grad_scalers` a loss-scaled learner fills with
        plain numbers -- as plain JSON.
        """
        path = directory / f"{name}{self.suffix}"
        # A failed save leaves orbax's own temporary directory behind, and cleaning that up raises
        # `Directory not empty` from `__exit__`, replacing the failure that caused it.
        with TemporaryDirectory(ignore_cleanup_errors=True) as workspace:
            checkpoint = Path(workspace) / name
            with ocp.Checkpointer(ocp.CompositeCheckpointHandler()) as checkpointer:
                items = {item: _save_item(item, value) for item, value in states.items()}
                checkpointer.save(checkpoint, args=ocp.args.Composite(**items))
            with tarfile.open(path, "w:gz") as archive:
                # Sorted, relative names: the archive does not record where it was built.
                for member in sorted(checkpoint.rglob("*")):
                    archive.add(member, arcname=str(member.relative_to(checkpoint)), recursive=False)
        return path

    def load(self, path: Path) -> dict[str, Any]:
        """Unpack an archive this backend wrote and restore it into host numpy arrays.

        The restore names no target and no sharding, so a state saved on four devices comes back on
        any topology; the strategy places it afterwards.
        """
        if not hasattr(tarfile, "data_filter"):
            # The extraction filters landed in 3.11.4, below the interpreters the project floor
            # admits. Without them `extractall` takes no `filter`, and the raw `TypeError` reads as
            # a bug here rather than as the interpreter being too old to extract safely.
            raise RuntimeError(
                "Reading a Flax training state needs the tarfile extraction filters added in Python 3.11.4, "
                "which this interpreter does not have. Extracting without them would let a crafted archive "
                "write outside the destination, so upgrade the interpreter instead."
            )
        with TemporaryDirectory() as workspace:
            checkpoint = Path(workspace) / "checkpoint"
            with tarfile.open(path, "r:gz") as archive:
                # The reference is user input: `filter="data"` is the stdlib guard refusing members
                # that would be written outside the destination.
                archive.extractall(checkpoint, filter="data")
            with ocp.Checkpointer(ocp.CompositeCheckpointHandler()) as checkpointer:
                # Naming no arguments is what makes orbax read each item's handler from the
                # checkpoint's own metadata and raise on one it cannot resolve. It logs a warning per
                # item on the way, which is noise from a supported path, not a failure.
                restored = dict(checkpointer.restore(checkpoint))
        return {item: _sequence_keys(value) for item, value in restored.items()}


class KerasStateBackend:
    """Training states as one compressed numpy archive.

    A Keras state is path-keyed arrays on every backend -- the tree
    `keras.Model.get_state_tree(value_format="numpy_array")` describes -- so the whole payload is a
    tree of numpy arrays and needs no keras to read or write. The archive members are the paths
    joined by "/", which is the separator Keras already uses inside a variable path, so nesting the
    keys back on load reproduces the tree whichever depth it was flattened from. The items holding
    no arrays -- `meta`, and the always-empty `grad_scalers` -- travel as one JSON member.

    The state is backend-portable but a resume is not: `meta` carries the Keras backend that wrote
    it, and the Keras trainer's `restore_training_state` refuses a state from another one.
    """

    suffix = ".npz"
    """A numpy archive is one file."""

    def save(self, states: Mapping[str, Any], directory: Path, name: str) -> Path:
        """Write *states* into `directory/name.npz` and return that path."""
        path = directory / f"{name}{self.suffix}"
        arrays: dict[str, Any] = {}
        plain: dict[str, Any] = {}
        for item, value in states.items():
            try:
                json.dumps(value)
            except TypeError:
                arrays.update(_flatten_arrays(value, item))
            else:
                plain[item] = value
        arrays[_JSON_ITEM] = np.array(json.dumps(plain))
        np.savez_compressed(path, **arrays)
        return path

    def load(self, path: Path) -> dict[str, Any]:
        """Read back an archive this backend wrote, as host numpy arrays nested under their paths."""
        restored: dict[str, Any] = {}
        # `np.load` does not unpickle unless asked to, and it is not asked to: the reference this
        # path is reached with is user input, as in the torch backend's `weights_only`.
        with np.load(path) as archive:
            for key in archive.files:
                if key == _JSON_ITEM:
                    restored.update(json.loads(str(archive[key])))
                    continue
                *parents, leaf = key.split("/")
                branch = restored
                for part in parents:
                    branch = branch.setdefault(part, {})
                branch[leaf] = archive[key]
        return restored


def _flatten_arrays(value: Any, prefix: str) -> dict[str, Any]:
    """Flatten one array item into `numpy` archive members, keyed by their "/"-joined path."""
    if isinstance(value, Mapping):
        return {
            key: item for name, sub in value.items() for key, item in _flatten_arrays(sub, f"{prefix}/{name}").items()
        }
    array = np.asarray(value)
    if array.dtype.isbuiltin == 2:
        # `ml_dtypes` registers bfloat16 as a user dtype, which `numpy.save` stores as an opaque
        # 2-byte void and reads back as one: the values would survive and the type would not.
        raise TypeError(
            f"training-state entry {prefix!r} holds {array.dtype.name} arrays, which a numpy archive cannot "
            "store without losing the element type. Keep the saved variables in a native numpy type "
            "(a keras mixed-precision policy keeps them float32)."
        )
    return {prefix: array}


def _save_item(item: str, value: Any) -> Any:
    """Return the orbax argument saving one top-level entry: JSON for plain data, arrays otherwise.

    An item is one or the other, never both, so a `meta` holding a numpy scalar next to its strings
    is refused here. Left to orbax it dies inside the array handler at the end of the first epoch,
    reporting the strings it cannot store rather than the entry that is actually wrong.
    """
    try:
        json.dumps(value)
    except TypeError as error:
        if (text := _text_path(value, ())) is not None:
            raise TypeError(
                f"training-state item {item!r} holds text at {'.'.join(text)!r}, so it has to be stored as "
                "JSON, but it is not JSON-serializable as a whole. Convert the offending entry to a plain "
                "value."
            ) from error
        return ocp.args.StandardSave(value)
    return ocp.args.JsonSave(value)


def _text_path(value: Any, path: tuple[str, ...]) -> tuple[str, ...] | None:
    """Return the path of the first string inside *value*, or `None` when it holds none."""
    if isinstance(value, str):
        return path
    entries: list[tuple[Any, Any]]
    if isinstance(value, Mapping):
        entries = list(value.items())
    elif isinstance(value, (list, tuple)):
        entries = list(enumerate(value))
    else:
        return None
    for key, item in entries:
        if (found := _text_path(item, (*path, str(key)))) is not None:
            return found
    return None


def _sequence_keys(value: Any) -> Any:
    """Turn the digit keys orbax writes back into the integers the state was keyed by.

    Orbax stores every mapping key as a string, while an nnx pure state keys the entries of a
    sequence -- an optimizer's `opt_state` tuple -- by position, and restoring a state whose keys
    read `"0"` against a live one keyed `0` fails on the structure mismatch. The reverse mapping is
    unambiguous: every other key is a Python attribute name, which can never be all digits. It runs
    over the JSON items too, so a `meta` entry keyed by digits also comes back keyed by integers.
    """
    if not isinstance(value, dict):
        return value
    return {int(key) if key.isascii() and key.isdecimal() else key: _sequence_keys(item) for key, item in value.items()}


__all__ = ["FlaxStateBackend", "KerasStateBackend", "StateBackend", "TorchStateBackend"]


if not TYPE_CHECKING:
    import sys

    from structcast.utils.lazy_import import LazySelectedImporter

    sys.modules[__name__] = LazySelectedImporter(__name__, globals())
