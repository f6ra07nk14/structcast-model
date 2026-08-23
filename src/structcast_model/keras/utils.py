"""Shared Keras device and training-state helpers."""

from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any

import numpy as np

import keras

if TYPE_CHECKING:
    import torch
else:
    from structcast.utils.lazy_import import LazyModuleImporter

    # torch is only touched when the active Keras backend is torch (`get_keras_device`); binding it
    # lazily keeps tensorflow and jax runs from importing it, as in `loggers.state_backends`.
    torch = LazyModuleImporter("torch")


def get_keras_device(device: str | None = None) -> str:
    """Get a list of available Keras devices."""
    if keras.backend.backend() == "torch":
        # The torch backend ships no distribution hooks (`keras.distribution` functions die on a
        # None backend module), so the device list comes from torch itself, in the same
        # "gpu:N" / "cpu:N" spelling the other backends report.
        devices = [f"gpu:{index}" for index in range(torch.cuda.device_count())]
        devices.append("cpu:0")
    else:
        devices = list(keras.distribution.list_devices())
    if not devices:
        raise ValueError("No Keras devices are available.")
    if device is None:
        device = next(iter(devices))
    if device in devices:
        return device
    devices_str = ", ".join(f"{d!r}" for d in devices)
    raise ValueError(f"Specified device {device!r} is not available. Available devices: {devices_str}")


def collect_state_dict(models: Mapping[str, Any], optimizers: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Read every model and optimizer variable back to host numpy, keyed by its Keras path.

    Each named model and optimizer becomes a tree nesting its variables under the segments of
    `variable.path`, so a state is restored by assigning the matching paths back. It is not what
    `keras.Model.get_state_tree` returns -- that one groups by variable category first
    (`trainable_variables`, `optimizer_variables`, ...), which is the only shape
    `keras.Model.set_state_tree` accepts -- because `get_state_tree` reads the optimizer off the
    model, and a generated learner attaches none: its optimizers belong to the segments the adapter
    drives.

    This is the one place a run's state is read, so the distributed strategies rewire state
    collection here and nowhere else.

    Args:
        models (Mapping[str, Any]): The models to read, by name.
        optimizers (Mapping[str, Any] | None): The optimizers to read, by name.

    Returns:
        dict[str, Any]: The `models` and `optimizers` halves of a training-state payload.
    """
    return {
        "models": {name: _variable_tree(model.variables) for name, model in models.items()},
        "optimizers": {name: _variable_tree(optimizer.variables) for name, optimizer in (optimizers or {}).items()},
    }


def apply_state_dict(models: Mapping[str, Any], optimizers: Mapping[str, Any], states: Mapping[str, Any]) -> None:
    """Assign a state :func:`collect_state_dict` produced back into live variables, by path.

    The inverse of the read, and the only writer: every variable of every named model and optimizer
    is looked up under the segments of its own `variable.path` and assigned the saved value. Nothing
    is created, so the optimizers must already be built -- the backend adapter builds them while the
    learner is constructed, before a resume can reach them.

    Args:
        models (Mapping[str, Any]): The live models to restore into, by name.
        optimizers (Mapping[str, Any]): The live optimizers to restore into, by name.
        states (Mapping[str, Any]): A payload holding the `models` and `optimizers` halves.

    Raises:
        ValueError: If the state holds no entry for a live model, optimizer or variable.
    """
    for kind, live in (("model", models), ("optimizer", optimizers)):
        saved = states.get(f"{kind}s", {})
        for name, holder in live.items():
            if name not in saved:
                raise ValueError(
                    f'The training state carries no {kind} named "{name}", only {sorted(saved)}: it was saved '
                    "from a different configuration."
                )
            _assign_variable_tree(holder.variables, saved[name], f'{kind} "{name}"')


def _assign_variable_tree(variables: Iterable[Any], tree: Mapping[str, Any], owner: str) -> None:
    """Assign each variable the value nested under the segments of its path.

    Raises:
        ValueError: If the tree holds no value for one of the variables.
    """
    for variable in variables:
        branch: Any = tree
        for part in variable.path.split("/"):
            if not isinstance(branch, Mapping) or part not in branch:
                raise ValueError(
                    f'The training state holds no value for {owner} variable "{variable.path}": it was saved '
                    "from a different configuration."
                )
            branch = branch[part]
        variable.assign(branch)


def _variable_tree(variables: Iterable[Any]) -> dict[str, Any]:
    """Nest the host-side value of each variable under the segments of its path."""
    tree: dict[str, Any] = {}
    for variable in variables:
        *parents, leaf = variable.path.split("/")
        branch = tree
        for part in parents:
            branch = branch.setdefault(part, {})
        value = variable.value
        # Through `read_value` where the backend variable has one: under `tf.distribute` an
        # optimizer counter is a `MirroredVariable` aggregated ONLY_FIRST_REPLICA, which refuses to
        # become an array at all ("object __array__ method not producing an array") and hands back
        # its primary copy through this call. A plain TensorFlow variable reads the same way, and
        # the JAX and torch backends have no such method.
        raw = value.read_value() if hasattr(value, "read_value") else value
        branch[leaf] = np.asarray(keras.ops.convert_to_numpy(raw))
    return tree


__all__ = ["apply_state_dict", "collect_state_dict", "get_keras_device"]


if not TYPE_CHECKING:
    import sys

    from structcast.utils.lazy_import import LazySelectedImporter

    sys.modules[__name__] = LazySelectedImporter(__name__, globals())
