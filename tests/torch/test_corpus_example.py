"""Unit tests for the corpus example in examples/torch/corpus.py."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import torch


def _load_example_module() -> Any:
    """Load the example module the way a configuration does: by file path, not by import name."""
    path = Path(__file__).resolve().parents[2] / "examples" / "torch" / "corpus.py"
    spec = importlib.util.spec_from_file_location("example_corpus", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_module = _load_example_module()
TinyShakespeare = _module.TinyShakespeare
TinyShakespeareLoader = _module.TinyShakespeareLoader


def _corpus(tmp_path: Path, split: str = "train", block_size: int = 4) -> Any:
    """Build a dataset over a local text file, so no download happens during the tests."""
    path = tmp_path / "corpus.txt"
    path.write_text("abcdefghij" * 10, encoding="utf-8")
    return TinyShakespeare(block_size=block_size, split=split, data_path=path)


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
    assert item["tokens"].dtype == torch.int64
    assert corpus[2]["tokens"].tolist() == item["tokens"].tolist()


def test_splits_share_one_vocabulary(tmp_path: Path) -> None:
    """Both splits encode a character to the same index, so a model trained on one can read the other."""
    train, validation = _corpus(tmp_path), _corpus(tmp_path, split="val")

    assert train.vocabulary == validation.vocabulary
    assert train.vocab_size == 10
    assert len(train.tokens) == 90
    assert len(validation.tokens) == 10


def test_loader_batches_land_on_the_resolved_device(tmp_path: Path) -> None:
    """The loader owns device placement.

    The trainer feeds batches to the learner untouched, so a loader that left them on the CPU
    would crash the first CUDA step (device-mismatch in the loss).
    """
    path = tmp_path / "corpus.txt"
    path.write_text("abcdefghij" * 10, encoding="utf-8")
    loader = TinyShakespeareLoader(block_size=4, split="train", data_path=path, batch_size=8)

    batches = list(loader)
    assert len(batches) == len(loader) == 2  # 22 items, drop_last
    device = torch.device(loader.distributed_results["device"])
    assert batches[0]["tokens"].shape == (8, 4)
    assert batches[0]["tokens"].device.type == device.type
    assert batches[0]["targets"].dtype == torch.int64
