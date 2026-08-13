"""Example character-level corpus for PyTorch language-model training runs.

The package knows nothing about text corpora: `scm torch train` instantiates whatever the
`--training-dataset` / `--validation-dataset` patterns describe, and the training loop takes any
iterable of dictionaries. `TinyShakespeare` is therefore example code, referenced from a
configuration by file path, and batched by `TinyShakespeareLoader`, which also owns moving each
batch onto the rank's device and sharding the items across ranks:

```yaml
_obj_:
  - _addr_: TinyShakespeareLoader
    _file_: examples/torch/corpus.py
  - _call_: {block_size: 256, split: train, batch_size: 16, shuffle: true}
```

Every item is `{"tokens": ..., "targets": ...}`, collation stacks those into
`[batch, block_size]` tensors, and the trainer passes them to the learner as keyword arguments --
which is why the keys match the inputs of `cfg/torch/learners/Transformer.yaml`.
"""

from collections.abc import Iterator
from functools import cached_property
from pathlib import Path
from typing import Any, Literal
from urllib.request import urlretrieve

from pydantic import BaseModel

from structcast_model.torch.trainer import initial_distributed_env
from structcast_model.torch.types import Tensor
import torch

CORPUS_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
"""Source of the Tiny Shakespeare corpus, a single plain text file of about one megabyte."""

CORPUS_PATH = Path("data/tinyshakespeare.txt")
"""Download destination, relative to the working directory, matching the `root: data` of the timm example."""

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
    def tokens(self) -> Tensor:
        """The characters of this split, encoded as their indices in the vocabulary."""
        index = {character: i for i, character in enumerate(self.vocabulary)}
        data = torch.tensor([index[character] for character in self.text], dtype=torch.int64)
        boundary = int(len(data) * TRAIN_FRACTION)
        return data[:boundary] if self.split == "train" else data[boundary:]

    def __len__(self) -> int:
        """Number of whole blocks in the split, one character being held back for the last target."""
        return (len(self.tokens) - 1) // self.block_size

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        """The block at `index`, and the same block shifted by one character as its targets."""
        start = index * self.block_size
        block = self.tokens[start : start + self.block_size + 1]
        return {"tokens": block[:-1], "targets": block[1:]}


class TinyShakespeareLoader(BaseModel):
    """Batches `TinyShakespeare` onto the local rank's device, sharding the items across ranks.

    The trainer deliberately owns neither concern: the loader that produced a batch knows its
    device and its ranks (the timm wrapper in `data.py` follows the same convention). A bare
    `DataLoader` would leave batches on the CPU and feed every rank the same items. Under
    distributed execution the shuffle order repeats every epoch (nothing calls
    `DistributedSampler.set_epoch`), which this example accepts.
    """

    block_size: int = 256
    """Number of characters per item, passed through to `TinyShakespeare`."""

    split: Literal["train", "val"] = "train"
    """The corpus split to read, passed through to `TinyShakespeare`."""

    data_path: Path | None = None
    """A text file to read instead of the downloaded corpus, passed through to `TinyShakespeare`."""

    batch_size: int = 16
    """Items per batch, per rank."""

    shuffle: bool = False
    """Whether to shuffle the items each epoch."""

    drop_last: bool = True
    """Whether to drop the final short batch."""

    @cached_property
    def dataset(self) -> TinyShakespeare:
        """The wrapped corpus split."""
        return TinyShakespeare(block_size=self.block_size, split=self.split, data_path=self.data_path)

    @cached_property
    def distributed_results(self) -> dict[str, Any]:
        """The rank's device and world layout, resolved once like the timm wrapper does."""
        return initial_distributed_env()

    @cached_property
    def dataloader(self) -> "torch.utils.data.DataLoader[dict[str, Tensor]]":
        """The underlying loader, sharded with a `DistributedSampler` when ranks exist."""
        sampler = (
            torch.utils.data.distributed.DistributedSampler(self.dataset, shuffle=self.shuffle)
            if self.distributed_results["distributed"]
            else None
        )
        return torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle if sampler is None else False,
            sampler=sampler,
            drop_last=self.drop_last,
        )

    def __len__(self) -> int:
        """Number of batches one rank sees per epoch."""
        return len(self.dataloader)

    def __iter__(self) -> Iterator[dict[str, Tensor]]:
        """The batches of one epoch, moved onto the rank's device."""
        device = torch.device(self.distributed_results["device"])
        for batch in self.dataloader:
            yield {key: value.to(device) for key, value in batch.items()}
