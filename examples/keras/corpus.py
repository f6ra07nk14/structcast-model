"""Example character-level corpus for Keras language-model training runs.

The NumPy twin of `examples/torch/corpus.py`: the corpus itself is framework-neutral text handling,
so only the tensor type and the batching change. The package knows nothing about corpora --
`scm keras train` instantiates whatever the `--training-dataset` / `--validation-dataset` patterns
describe, and the training loop takes any iterable of dictionaries -- so this is example code,
referenced from a configuration by file path:

```yaml
_obj_:
  - _addr_: TinyShakespeareLoader
    _file_: examples/keras/corpus.py
  - _call_: {block_size: 256, split: train, batch_size: 16, shuffle: true}
```

Every batch is `{"tokens": ..., "targets": ...}` of `[batch, block_size]` NumPy arrays, and the
trainer passes them to the learner as keyword arguments -- which is why the keys match the inputs of
`cfg/keras/learners/SmallLanguageModel.yaml`.

The loader is single-stream: it hands the whole batch to the trainer and the distributed strategy
splits it across the replicas, which is what `keras.distribution` on jax and
`tf.distribute.MirroredStrategy` on tensorflow do. Under a torchrun launch on the torch backend the
loader owns each rank's slice instead, so that case needs a rank-aware loader of its own.
"""

from collections.abc import Iterator
from functools import cached_property
from pathlib import Path
from typing import Any, Literal
from urllib.request import urlretrieve

import numpy as np
from pydantic import BaseModel

CORPUS_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
"""Source of the Tiny Shakespeare corpus, a single plain text file of about one megabyte."""

CORPUS_PATH = Path("data/tinyshakespeare.txt")
"""Download destination, relative to the working directory, as in the torch twin."""

TRAIN_FRACTION = 0.9
"""Fraction of the corpus used for training, the remainder being the validation split."""


class TinyShakespeare(BaseModel):
    """The Tiny Shakespeare corpus as fixed-length character sequences for next-token prediction.

    The corpus is downloaded once into `CORPUS_PATH`, or read from `data_path` when it is given,
    which is also how an offline run points at its own text file. The vocabulary is built from the
    whole corpus, so the two splits encode characters identically, and both the split boundary and
    the item offsets are deterministic: item `i` is the characters at `i * block_size`, shifted by
    one for the targets.
    """

    block_size: int = 256
    """Number of characters per item, which is the sequence length the model sees."""

    split: Literal["train", "val"] = "train"
    """The part of the corpus to read: the first 90% of the characters, or the remaining 10%."""

    data_path: Path | None = None
    """A text file to read instead of the downloaded corpus."""

    @cached_property
    def text(self) -> str:
        """The whole corpus, downloading it on first use unless `data_path` provides it."""
        if self.data_path is not None:
            return self.data_path.read_text(encoding="utf-8")
        if not CORPUS_PATH.exists():
            CORPUS_PATH.parent.mkdir(parents=True, exist_ok=True)
            urlretrieve(CORPUS_URL, CORPUS_PATH)  # noqa: S310  # the URL is the constant literal above
        return CORPUS_PATH.read_text(encoding="utf-8")

    @cached_property
    def vocabulary(self) -> list[str]:
        """The characters of the whole corpus, sorted, so the encoding does not depend on the split."""
        return sorted(set(self.text))

    @property
    def vocab_size(self) -> int:
        """Number of distinct characters, the `vocab_size` the model must be created with."""
        return len(self.vocabulary)

    @cached_property
    def tokens(self) -> np.ndarray:
        """The characters of this split, encoded as their indices in the vocabulary."""
        index = {character: i for i, character in enumerate(self.vocabulary)}
        data = np.fromiter((index[character] for character in self.text), dtype="int64")
        boundary = int(len(data) * TRAIN_FRACTION)
        return data[:boundary] if self.split == "train" else data[boundary:]

    def __len__(self) -> int:
        """Number of whole blocks in the split, one character being held back for the last target."""
        return (len(self.tokens) - 1) // self.block_size


class TinyShakespeareLoader(BaseModel):
    """Batches `TinyShakespeare` into NumPy arrays, which every Keras backend accepts.

    The blocks are sliced out of the token array in one indexing operation per batch, so no per-item
    collation is needed.
    """

    block_size: int = 256
    """Number of characters per item, passed through to `TinyShakespeare`."""

    split: Literal["train", "val"] = "train"
    """The corpus split to read, passed through to `TinyShakespeare`."""

    data_path: Path | None = None
    """A text file to read instead of the downloaded corpus, passed through to `TinyShakespeare`."""

    batch_size: int = 16
    """Items per batch."""

    shuffle: bool = False
    """Whether to shuffle the items each epoch."""

    drop_last: bool = True
    """Whether to drop the final short batch."""

    seed: int = 42
    """Seed of the shuffle, so a run's data order is reproducible."""

    @cached_property
    def dataset(self) -> TinyShakespeare:
        """The wrapped corpus split."""
        return TinyShakespeare(block_size=self.block_size, split=self.split, data_path=self.data_path)

    @cached_property
    def _generator(self) -> Any:
        """The generator drawing the shuffle order, advanced once per epoch."""
        return np.random.default_rng(self.seed)

    def __len__(self) -> int:
        """Number of batches per epoch."""
        items = len(self.dataset)
        return items // self.batch_size if self.drop_last else -(-items // self.batch_size)

    def __iter__(self) -> Iterator[dict[str, np.ndarray]]:
        """The batches of one epoch, as `{"tokens": ..., "targets": ...}` NumPy arrays."""
        order = np.arange(len(self.dataset))
        if self.shuffle:
            self._generator.shuffle(order)
        tokens = self.dataset.tokens
        for start in range(0, len(self) * self.batch_size, self.batch_size):
            offsets = order[start : start + self.batch_size] * self.block_size
            blocks = tokens[offsets[:, None] + np.arange(self.block_size + 1)]
            yield {"tokens": blocks[:, :-1], "targets": blocks[:, 1:]}
