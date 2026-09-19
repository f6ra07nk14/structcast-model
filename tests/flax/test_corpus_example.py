"""Unit tests for the corpus example in examples/flax/corpus.py."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import numpy as np


def _load_example_module() -> Any:
    """Load the example module the way a configuration does: by file path, not by import name."""
    path = Path(__file__).resolve().parents[2] / "examples" / "flax" / "corpus.py"
    spec = importlib.util.spec_from_file_location("example_flax_corpus", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_module = _load_example_module()
TinyShakespeare = _module.TinyShakespeare
TinyShakespeareLoader = _module.TinyShakespeareLoader


def _corpus_file(tmp_path: Path) -> Path:
    """Write a local text file, so no download happens during the tests."""
    path = tmp_path / "corpus.txt"
    path.write_text("abcdefghij" * 10, encoding="utf-8")
    return path


def _corpus(tmp_path: Path, split: str = "train", block_size: int = 4) -> Any:
    """Build a dataset over that local text file."""
    return TinyShakespeare(block_size=block_size, split=split, data_path=_corpus_file(tmp_path))


def test_items_are_deterministic_and_shifted_by_one(tmp_path: Path) -> None:
    """Item `i` covers the block at `i * block_size`, and its targets are that block shifted by one.

    The shift is the training signal itself: reading it wrong would train the model to predict the
    token it was just given. The fixed offset keeps a resumed or replayed run over the same corpus
    comparable.
    """
    corpus = _corpus(tmp_path)
    item = corpus[2]

    assert item["tokens"].tolist() == corpus.tokens[8:12].tolist()
    assert item["targets"].tolist() == corpus.tokens[9:13].tolist()
    assert item["tokens"][1:].tolist() == item["targets"][:-1].tolist()
    # int32, not int64: JAX truncates 64-bit integers unless x64 is enabled, so emitting int32 here
    # is what keeps the array the learner sees the array the loader built.
    assert item["tokens"].dtype == np.int32
    assert corpus[2]["tokens"].tolist() == item["tokens"].tolist()


def test_splits_share_one_vocabulary(tmp_path: Path) -> None:
    """Both splits encode a character to the same index, so a model trained on one can read the other."""
    train, validation = _corpus(tmp_path), _corpus(tmp_path, split="val")

    assert train.vocabulary == validation.vocabulary
    assert train.vocab_size == 10
    assert len(train.tokens) == 90
    assert len(validation.tokens) == 10


def test_loader_stacks_whole_batches_of_host_arrays(tmp_path: Path) -> None:
    """The loader owns batching and nothing else: no device placement, no rank sharding.

    `scm flax train` places every batch on the strategy's mesh itself, and that placement rejects a
    leading dimension the mesh does not divide -- which is why the short final batch is dropped and
    why `len()` has to agree with what an epoch actually yields.
    """
    loader = TinyShakespeareLoader(block_size=4, data_path=_corpus_file(tmp_path), batch_size=8)

    batches = list(loader)

    assert len(batches) == len(loader) == 2  # 22 items, drop_last
    assert batches[0]["tokens"].shape == (8, 4)
    assert batches[0]["targets"].dtype == np.int32
    assert isinstance(batches[0]["tokens"], np.ndarray)


def test_shuffling_reorders_every_epoch_and_replays_from_the_seed(tmp_path: Path) -> None:
    """A shuffled loader must not repeat one order, and two loaders on one seed must agree.

    Repeating the order every epoch is the bug `DistributedSampler.set_epoch` exists to prevent on
    the torch side; here the single generator is what avoids it, and seeding that generator is what
    keeps a whole run reproducible.
    """
    path = _corpus_file(tmp_path)
    loader = TinyShakespeareLoader(block_size=4, data_path=path, batch_size=8, shuffle=True, seed=0)
    twin = TinyShakespeareLoader(block_size=4, data_path=path, batch_size=8, shuffle=True, seed=0)

    first, second = next(iter(loader))["tokens"], next(iter(loader))["tokens"]

    assert not np.array_equal(first, second)
    assert np.array_equal(first, next(iter(twin))["tokens"])
