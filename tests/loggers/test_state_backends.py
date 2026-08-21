"""Unit tests for structcast_model.loggers.state_backends."""

from __future__ import annotations

import io
from pathlib import Path
import tarfile
from typing import Any

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
import optax
import pytest

from flax import nnx
from structcast_model.flax.distributed import FlaxDistributedStrategy
from structcast_model.loggers.state_backends import FlaxStateBackend, KerasStateBackend, TorchStateBackend
import torch


@pytest.fixture(autouse=True)
def _clear_mesh() -> Any:
    """Unset the mesh a strategy activated, so it does not leak into unrelated tests.

    ``jax.set_mesh`` is process-wide, as in ``tests/flax/test_distributed``.
    """
    yield
    jax.set_mesh(None)


class _Model(nnx.Module):
    """A linear layer, a batch norm and a dropout: parameters, statistics and RNG state in one model."""

    def __init__(self, *, rngs: nnx.Rngs) -> None:
        """Build the layers."""
        self.fc = nnx.Linear(4, 2, rngs=rngs)
        self.bn = nnx.BatchNorm(2, rngs=rngs)
        self.dropout = nnx.Dropout(0.5, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        """Run the layers in training mode."""
        return self.dropout(self.bn(self.fc(x), use_running_average=False), deterministic=False)


def _leaf_paths(tree: Any) -> list[str]:
    """Render every leaf's key path, so a key restored as a string instead of an index stands out."""
    return [str(path) for path, _ in jax.tree_util.tree_flatten_with_path(tree)[0]]


def _trained() -> tuple[_Model, nnx.Optimizer]:
    """Build a model and its optimizer, and take one step so neither state is all zeros."""
    model = _Model(rngs=nnx.Rngs(params=jax.random.key(0), dropout=jax.random.key(1)))
    optimizer = nnx.Optimizer(model, tx=optax.adam(learning_rate=0.1), wrt=nnx.Param)
    grads = nnx.grad(lambda module: jnp.mean(module(jnp.ones((2, 4))) ** 2))(model)
    optimizer.update(model, grads)
    return model, optimizer


def test_torch_backend_writes_what_torch_load_reads(tmp_path: Path) -> None:
    """The torch backend moved five call sites; the format it writes must stay `torch.save`'s own.

    A checkpoint written by an older release has to keep loading, so the assertion is against torch
    itself rather than against the backend's own `load`.
    """
    backend = TorchStateBackend()

    path = backend.save({"models": {"weight": torch.ones(2)}}, tmp_path, "training_state")

    assert path == tmp_path / "training_state.pt"
    assert torch.load(path, weights_only=True)["models"]["weight"].tolist() == [1.0, 1.0]
    assert backend.load(path)["models"]["weight"].tolist() == [1.0, 1.0]


def test_flax_backend_round_trips_a_state_the_strategy_can_load(tmp_path: Path) -> None:
    """What comes out of the transport must be exactly what the strategy put in.

    Values, dtypes and the keys of the optimizer's own tree all have to survive the round trip.

    Orbax stores every mapping key as a string, while an optimizer's `opt_state` is keyed by
    position: a round trip that lost that distinction would restore into a structure mismatch,
    which is why the restored state is fed to a live strategy here and not only compared.
    """
    strategy = FlaxDistributedStrategy(preset="single")
    model, optimizer = _trained()
    states: dict[str, Any] = strategy.state_dict({"model": model}, {"optimizer": optimizer})
    states["grad_scalers"] = {}
    states["meta"] = {"epoch": 2, "step": 7, "update": 3, "optimizer_hashes": {"optimizer": "abc"}}

    path = FlaxStateBackend().save(states, tmp_path, "training_state")
    restored = FlaxStateBackend().load(path)

    assert path == tmp_path / "training_state.tar.gz"
    assert restored["meta"] == states["meta"]
    assert restored["grad_scalers"] == {}
    arrays = {key: states[key] for key in ("models", "optimizers")}
    back = {key: restored[key] for key in ("models", "optimizers")}
    assert _leaf_paths(back) == _leaf_paths(arrays)
    for saved, loaded in zip(jax.tree.leaves(arrays), jax.tree.leaves(back), strict=True):
        assert loaded.dtype == saved.dtype
        assert np.array_equal(loaded, saved)
    # A typed RNG key travels as its raw uint32 data, which is what the strategy wraps back.
    assert restored["models"]["model"]["dropout"]["rngs"]["key"].dtype == np.uint32

    fresh_model, fresh_optimizer = _trained()
    strategy.load_state_dict({"model": fresh_model}, {"optimizer": fresh_optimizer}, None, restored)
    assert jnp.array_equal(fresh_model.fc.kernel[...], model.fc.kernel[...])


def test_flax_backend_names_the_entry_when_an_item_mixes_text_and_arrays(tmp_path: Path) -> None:
    """`meta` is filled by the caller, and one non-JSON value in it must not read as an array tree.

    Stored as arrays, the item fails from inside orbax naming a string it cannot store -- a symptom
    of the wrong entry -- at the end of the first epoch, after the whole state has been produced.
    """
    states = {"meta": {"epoch": np.int64(3), "config_hash": "abc"}}

    with pytest.raises(TypeError, match="config_hash"):
        FlaxStateBackend().save(states, tmp_path, "training_state")


def test_flax_backend_refuses_an_archive_writing_outside_the_extraction_directory(tmp_path: Path) -> None:
    """A `--resume` reference is user input, so a crafted archive must not be able to write anywhere."""
    archive_path = tmp_path / "evil.tar.gz"
    payload = b"owned"
    with tarfile.open(archive_path, "w:gz") as archive:
        member = tarfile.TarInfo("../evil")
        member.size = len(payload)
        archive.addfile(member, io.BytesIO(payload))

    with pytest.raises(tarfile.TarError):
        FlaxStateBackend().load(archive_path)

    assert not (tmp_path / "evil").exists()


def test_flax_backend_refuses_to_extract_on_an_interpreter_without_the_filters(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Python 3.11.0-3.11.3 is inside the project floor and ships no `tarfile.data_filter`.

    On such interpreters the path-traversal guard would silently not exist (the `filter` argument
    resolves against whatever the module registered at import time), so the backend must refuse to
    extract at all rather than extract unsafely or die with an error that reads as a backend bug.
    """
    monkeypatch.delattr(tarfile, "data_filter")
    archive_path = tmp_path / "state.tar.gz"
    with tarfile.open(archive_path, "w:gz"):
        pass

    with pytest.raises(RuntimeError, match="Python 3.11.4"):
        FlaxStateBackend().load(archive_path)


def test_keras_backend_round_trips_a_path_keyed_state(tmp_path: Path) -> None:
    """What comes out of the archive must be exactly what the saver put in, `meta` included.

    The Keras payload is plain numpy -- no keras is imported here, and none is needed -- but the
    backend it was produced on is: a resume compares that field and refuses a mismatch, because
    normalization statistics and RNG trajectories are not verified equivalent across the backends.
    """
    states: dict[str, Any] = {
        "models": {"model": {"model": {"fc": {"kernel": np.ones((4, 2), "float32"), "bias": np.zeros(2, "float16")}}}},
        "optimizers": {"optimizer": {"optimizer": {"iteration": np.asarray(3, "int64")}}},
        "grad_scalers": {},
        "meta": {"epoch": 2, "step": 7, "update": 3, "backend": "jax", "seed": 42},
    }

    path = KerasStateBackend().save(states, tmp_path, "training_state")
    restored = KerasStateBackend().load(path)

    assert path == tmp_path / "training_state.npz"
    assert restored["meta"] == states["meta"]
    assert restored["meta"]["backend"] == "jax"
    assert restored["grad_scalers"] == {}
    assert restored["models"]["model"]["model"]["fc"].keys() == {"kernel", "bias"}
    for name, saved in (("kernel", np.ones((4, 2), "float32")), ("bias", np.zeros(2, "float16"))):
        loaded = restored["models"]["model"]["model"]["fc"][name]
        assert loaded.dtype == saved.dtype
        assert np.array_equal(loaded, saved)
    assert restored["optimizers"]["optimizer"]["optimizer"]["iteration"] == 3


def test_keras_backend_refuses_a_dtype_a_numpy_archive_cannot_store(tmp_path: Path) -> None:
    """`numpy` stores an ml_dtypes bfloat16 array as an opaque void and reads it back as one.

    The values would survive and the element type would not, so a state saved from bfloat16
    variables has to fail here rather than resume into arrays whose type no longer says what they
    are.
    """
    states = {"models": {"model": {"kernel": np.ones(2, dtype=ml_dtypes.bfloat16)}}}

    with pytest.raises(TypeError, match="bfloat16"):
        KerasStateBackend().save(states, tmp_path, "training_state")
