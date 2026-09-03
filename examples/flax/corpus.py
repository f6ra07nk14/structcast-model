"""Example character-level corpus for Flax (nnx) language-model training runs.

The package knows nothing about text corpora: `scm flax train` instantiates whatever the
`--training-dataset` / `--validation-dataset` patterns describe, and the training loop takes any
iterable of dictionaries. `TinyShakespeare` is therefore example code, referenced from a
configuration by file path, and batched by `TinyShakespeareLoader`:

```yaml
_obj_:
  - _addr_: TinyShakespeareLoader
    _file_: examples/flax/corpus.py
  - _call_: {block_size: 256, split: train, batch_size: 16, shuffle: true}
```

Every batch is `{"tokens": ..., "targets": ...}` of shape `[batch, block_size]`, and the trainer
passes those to the learner as keyword arguments -- which is why the keys match the inputs of
`cfg/flax/learners/SmallLanguageModel.yaml`.

This is the twin of `examples/torch/corpus.py`, minus two things the Flax side must not do.

The batches are NumPy arrays and stay on the host: `scm flax train` places every batch on the
strategy's mesh itself, so a loader that moved them onto a device first would only add a transfer
the placement then undoes.

And there is no rank sharding, by design. The torch twin wraps its dataset in a
`DistributedSampler` because `torchrun` starts one process per rank and each of them must be handed
a different slice of the epoch. JAX is single-controller: one process reads the whole batch and
`FlaxDistributedStrategy.shard_batch` splits it across the mesh along its leading dimension. A
loader that also sharded per rank would double-shard -- each device would see a slice of a slice,
the effective batch would be the configured one divided by the device count squared, and most of
the epoch would never be trained on. So the loader hands over whole batches and lets the strategy
be the only thing that splits them.
"""

from collections.abc import Iterator
from functools import cached_property
from pathlib import Path
from typing import Literal
from urllib.request import urlretrieve

import numpy as np
from pydantic import BaseModel

CORPUS_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
"""Source of the Tiny Shakespeare corpus, a single plain text file of about one megabyte."""

CORPUS_PATH = Path("data/tinyshakespeare.txt")
"""Download destination, relative to the working directory, matching the `data_dir` of the tf.data example."""

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
        """The characters of this split, encoded as their indices in the vocabulary.

        int32, not int64: JAX truncates 64-bit integers to 32 unless `jax_enable_x64` is set, so
        emitting int32 here is what keeps the arrays the model sees the ones the loader built.
        """
        index = {character: i for i, character in enumerate(self.vocabulary)}
        data = np.fromiter((index[character] for character in self.text), dtype=np.int32, count=len(self.text))
        boundary = int(len(data) * TRAIN_FRACTION)
        return data[:boundary] if self.split == "train" else data[boundary:]

    def __len__(self) -> int:
        """Number of whole blocks in the split, one character being held back for the last target."""
        return (len(self.tokens) - 1) // self.block_size

    def __getitem__(self, index: int) -> dict[str, np.ndarray]:
        """The block at `index`, and the same block shifted by one character as its targets."""
        start = index * self.block_size
        block = self.tokens[start : start + self.block_size + 1]
        return {"tokens": block[:-1], "targets": block[1:]}


class TinyShakespeareLoader(BaseModel):
    """Batches `TinyShakespeare` into NumPy arrays the trainer hands straight to the learner.

    Deliberately not a `torch.utils.data.DataLoader` equivalent: there are no workers, because the
    whole epoch is one contiguous int32 array already in memory and slicing it costs less than
    handing slices to a worker would, and no sampler, because the strategy is what splits a batch
    across the devices (see the module docstring). The shuffle order is drawn from one generator
    seeded with `seed`, so a whole run is reproducible while every epoch sees a different order.
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
    """Whether to drop the final short batch. A Flax run wants this: the strategy splits a batch
    across its mesh, so a short one would not divide."""

    seed: int = 42
    """Seed of the shuffle generator, so the sequence of epoch orders is reproducible."""

    @cached_property
    def dataset(self) -> TinyShakespeare:
        """The wrapped corpus split."""
        return TinyShakespeare(block_size=self.block_size, split=self.split, data_path=self.data_path)

    @cached_property
    def generator(self) -> np.random.Generator:
        """The shuffle generator, advanced once per epoch so no two epochs repeat an order."""
        return np.random.default_rng(self.seed)

    def __len__(self) -> int:
        """Number of batches one epoch yields."""
        whole, remainder = divmod(len(self.dataset), self.batch_size)
        return whole if self.drop_last else whole + bool(remainder)

    def __iter__(self) -> Iterator[dict[str, np.ndarray]]:
        """The batches of one epoch, stacked into `[batch, block_size]` NumPy arrays."""
        count = len(self.dataset)
        order = self.generator.permutation(count) if self.shuffle else np.arange(count)
        for start in range(0, len(self) * self.batch_size, self.batch_size):
            items = [self.dataset[int(index)] for index in order[start : start + self.batch_size]]
            yield {key: np.stack([item[key] for item in items]) for key in ("tokens", "targets")}
