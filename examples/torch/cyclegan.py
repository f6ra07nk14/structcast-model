r"""Example unpaired two-domain image data and replay buffer for PyTorch CycleGAN runs.

`cfg/torch/learners/CycleGAN.yaml` reads four batch names: the two real images `real_A` and
`real_B`, and `fake_A_sample` / `fake_B_sample`, the *generated* images its discriminator segments
train against. The first two are what an unpaired dataset produces; the last two are the
generators' own output from earlier steps, so no dataset can produce them and this file supplies
both halves -- the loader, and the trainer that closes the loop:

```bash
scm torch create model cfg/torch/models/CycleGAN_generator.yaml -o generator.py
scm torch create model cfg/torch/models/CycleGAN_discriminator.yaml -o discriminator.py
scm torch create learner cfg/torch/learners/CycleGAN.yaml -o learner.py

scm torch train \\
    'G_AB: [_obj_, {_addr_: Model, _file_: generator.py}, _call_]' \\
    'G_BA: [_obj_, {_addr_: Model, _file_: generator.py}, _call_]' \\
    'D_A: [_obj_, {_addr_: Model, _file_: discriminator.py}, _call_]' \\
    'D_B: [_obj_, {_addr_: Model, _file_: discriminator.py}, _call_]' \\
    -L '[_obj_, {_addr_: Learner, _file_: learner.py}]' \\
    -s 'image: [3, 256, 256]' \\
    --training-dataset '[_obj_, {_addr_: UnpairedImageLoader, _file_: examples/torch/cyclegan.py},
                         {_call_: {root_A: data/horse2zebra/trainA, root_B: data/horse2zebra/trainB}}]' \\
    --trainer '[_obj_, {_addr_: CycleGANTrainer, _file_: examples/torch/cyclegan.py}]' \\
    -LO loss_G -LO loss_GAN -LO loss_cycle -LO loss_identity -LO loss_D_A -LO loss_D_B \\
    -d cuda -e 200 -LC loss_G -E cyclegan
```

Two details of that command are load-bearing:

* `--trainer` is the seam. `BaseTrainer.update_models` is the one place that sees both the batch on
  its way to the learner and the criteria on the way back, which is exactly what a replay buffer
  needs; a callback or a dataset event receives the trainer alone and never the step's tensors. The
  buffer therefore feeds the *previous* step's generated images rather than the current step's,
  which the reference implementation can use because there the generator forward and the
  discriminator update are separate calls. One step of lag inside a buffer that already reaches
  fifty steps back is not a difference worth a second forward pass.
* `-LO` names the six scalar criteria. The template also outputs `fake_A` and `fake_B` -- it has to,
  since that is how the generated images reach the buffer -- and the tracker sums every criterion it
  is given into a `(1,)` buffer, which a `[batch, 3, height, width]` image does not fit.

No validation dataset: the template's discriminator segments have no `INFERENCE_FLOW`, so an
inference step would want the two buffer samples as well, and a GAN has no held-out scalar worth
selecting a checkpoint on anyway.
"""

from collections.abc import Iterator
from dataclasses import dataclass, field
from functools import cached_property
import os
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from pydantic import BaseModel, DirectoryPath

from structcast_model.torch.trainer import TorchTrainer, initial_distributed_env
from structcast_model.torch.types import Tensor
import torch

EXTENSIONS = frozenset({".bmp", ".jpeg", ".jpg", ".png", ".webp"})
"""Suffixes read as images, lowercased. The horse2zebra tree is JPEG; the others cost nothing."""


def data_shard() -> tuple[int | None, int | None]:
    """The `rank` and `num_replicas` a sampler must shard on, both None when nothing published them.

    `scm torch train` puts the run's coordinates on the *data* axis of the device mesh into
    `DATA_RANK` and `DATA_WORLD_SIZE`; `examples/torch/corpus.py` carries the long version of why
    those are not always the global rank and world size.
    """
    rank, world_size = os.environ.get("DATA_RANK"), os.environ.get("DATA_WORLD_SIZE")
    if rank is None or world_size is None:
        return None, None
    return int(rank), int(world_size)


class UnpairedImages(BaseModel):
    """Two directories of images drawn independently, as unpaired translation is defined.

    Nothing aligns the two domains: item `i` is the `i`-th image of `root_A` and, while training, a
    randomly drawn image of `root_B`. An epoch is the longer of the two directories, so the smaller
    domain repeats within it -- the convention of the reference implementation, and what keeps every
    image of the larger domain seen once per epoch.

    The draw comes from one `numpy.random.Generator` of this object, so a run is reproducible from `seed`
    alone and the pairing is redrawn every epoch as the generator advances. That also means the
    items must be produced in one process: the loader below asks for no worker processes, which a
    directory of a few thousand JPEGs does not need.
    """

    root_A: DirectoryPath
    """Directory of the first domain's images, e.g. `data/horse2zebra/trainA`. Read recursively."""

    root_B: DirectoryPath
    """Directory of the second domain's images, e.g. `data/horse2zebra/trainB`."""

    load_size: int = 286
    """Side length images are resized to before the crop; the paper's 286 for its 256-pixel crop."""

    crop_size: int = 256
    """Side length the models see. A multiple of four: the generator downsamples twice."""

    is_training: bool = True
    """Whether to draw domain B randomly and augment. The other way is deterministic and unaugmented."""

    hflip: bool = True
    """Whether the training augmentation flips images horizontally."""

    seed: int = 42
    """Seed of the domain B draw and of the augmentation."""

    @cached_property
    def files(self) -> tuple[list[Path], list[Path]]:
        """The images of each domain, sorted, so the same tree is numbered the same way twice.

        Raises:
            ValueError: If either directory holds no image, which would make an epoch undefined.
        """
        listings = [
            sorted(path for path in root.rglob("*") if path.suffix.lower() in EXTENSIONS)
            for root in (self.root_A, self.root_B)
        ]
        for root, listing in zip((self.root_A, self.root_B), listings, strict=True):
            if not listing:
                raise ValueError(
                    f'The directory "{root}" holds no image with a suffix in {sorted(EXTENSIONS)}. '
                    "It names one domain's directory of images, not the dataset root: the horse2zebra "
                    "set is two of them, trainA and trainB."
                )
        return listings[0], listings[1]

    @cached_property
    def rng(self) -> np.random.Generator:
        """The generator of the domain B draw and of the augmentation, seeded once per run."""
        return np.random.default_rng(self.seed)

    def _load(self, path: Path) -> Tensor:
        """Read one image as a `[3, crop_size, crop_size]` tensor scaled to [-1, 1].

        The range is what the generator's closing `torch.nn.Tanh` emits, and the identity and cycle
        losses compare a real image against exactly that, so anything else would ask the generator
        for a value it cannot produce.
        """
        with Image.open(path) as opened:
            image = opened.convert("RGB").resize((self.load_size, self.load_size), Image.Resampling.BICUBIC)
        if self.is_training:
            offset = self.load_size - self.crop_size
            left, top = (int(self.rng.integers(offset, endpoint=True)) for _ in range(2))
            image = image.crop((left, top, left + self.crop_size, top + self.crop_size))
            if self.hflip and self.rng.random() < 0.5:
                image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        else:
            image = image.resize((self.crop_size, self.crop_size), Image.Resampling.BICUBIC)
        array = np.asarray(image, dtype=np.float32).transpose(2, 0, 1)
        return torch.from_numpy(array / 127.5 - 1.0)

    def __len__(self) -> int:
        """Items in one epoch: the longer directory, so its every image is seen once."""
        return max(len(listing) for listing in self.files)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        """The `index`-th image of domain A, and an unaligned image of domain B."""
        files_A, files_B = self.files
        other = int(self.rng.integers(len(files_B))) if self.is_training else index % len(files_B)
        return {"real_A": self._load(files_A[index % len(files_A)]), "real_B": self._load(files_B[other])}


class UnpairedImageLoader(BaseModel):
    """Batches `UnpairedImages` onto the local rank's device, sharding the items across ranks.

    The same division of labour as `examples/torch/corpus.py`: the trainer feeds batches to the
    learner untouched, so the loader that produced one owns both its device and its shard. The two
    coordinates differ -- batches land on the *local* rank's device, the shard comes from
    `data_shard` -- and the shuffle order repeats every epoch, since nothing calls
    `DistributedSampler.set_epoch`; the domain B draw is redrawn regardless.
    """

    root_A: DirectoryPath
    """Directory of the first domain's images, passed through to `UnpairedImages`."""

    root_B: DirectoryPath
    """Directory of the second domain's images, passed through to `UnpairedImages`."""

    load_size: int = 286
    """Side length before the crop, passed through to `UnpairedImages`."""

    crop_size: int = 256
    """Side length the models see, passed through to `UnpairedImages`."""

    is_training: bool = True
    """Whether the split is drawn unaligned and augmented, passed through to `UnpairedImages`."""

    hflip: bool = True
    """Whether the augmentation flips horizontally, passed through to `UnpairedImages`."""

    seed: int = 42
    """Seed of the draw and the augmentation, passed through to `UnpairedImages`."""

    batch_size: int = 1
    """Items per batch, per rank. The paper trains CycleGAN with one."""

    shuffle: bool = True
    """Whether to shuffle the domain A order each epoch."""

    drop_last: bool = True
    """Whether to drop the final short batch."""

    @cached_property
    def dataset(self) -> UnpairedImages:
        """The wrapped pair of directories."""
        return UnpairedImages(
            root_A=self.root_A,
            root_B=self.root_B,
            load_size=self.load_size,
            crop_size=self.crop_size,
            is_training=self.is_training,
            hflip=self.hflip,
            seed=self.seed,
        )

    @cached_property
    def distributed_results(self) -> dict[str, Any]:
        """The rank's device and world layout, resolved once as the other torch examples do."""
        return initial_distributed_env()

    @cached_property
    def dataloader(self) -> "torch.utils.data.DataLoader[dict[str, Tensor]]":
        """The underlying loader, sharded with a `DistributedSampler` on the run's data coordinates.

        No worker processes: the unaligned draw comes from one `numpy.random.Generator` living on the
        dataset, which forked workers would each hold their own copy of.
        """
        rank, num_replicas = data_shard()
        sampler: torch.utils.data.distributed.DistributedSampler[dict[str, Tensor]] | None = (
            torch.utils.data.distributed.DistributedSampler(
                self.dataset, shuffle=self.shuffle, rank=rank, num_replicas=num_replicas
            )
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


class ImagePool:
    """The replay buffer of generated images CycleGAN trains its discriminators on.

    Zhu et al., "Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks"
    (ICCV 2017), section 4, following Shrivastava et al. (2017): a discriminator is shown a history
    of images the generator produced rather than only its newest ones, which damps the oscillation
    the two networks otherwise fall into. Fifty images is the paper's size.

    `query` implements the paper's rule per image: until the buffer is full every image is stored
    and returned; afterwards a coin decides between returning the image untouched and swapping it
    for a random stored one, which is then returned in its place. The coin and the index come from
    one `numpy.random.Generator`, so a run is reproducible from `seed`.
    """

    def __init__(self, size: int = 50, seed: int = 42) -> None:
        """Build an empty buffer of at most *size* images, drawing from a generator seeded by *seed*."""
        self.size = size
        self.rng = np.random.default_rng(seed)
        self.images: list[Tensor] = []

    def query(self, images: Tensor) -> Tensor:
        """Return a `[batch, ...]` batch of buffered images, storing *images* in their place.

        Args:
            images (Tensor): The generated images of one batch, which the buffer may keep.

        Returns:
            Tensor: As many images as were given, each either the one given or one buffered earlier.
        """
        if self.size == 0:
            return images
        sampled = []
        for image in images.detach().split(1):
            if len(self.images) < self.size:
                self.images.append(image.clone())
                sampled.append(image)
            elif self.rng.random() > 0.5:
                index = int(self.rng.integers(self.size))
                sampled.append(self.images[index])
                self.images[index] = image.clone()
            else:
                sampled.append(image)
        return torch.cat(sampled)


@dataclass(kw_only=True)
class CycleGANTrainer(TorchTrainer):
    """A `TorchTrainer` that feeds the discriminators a replay buffer of earlier generated images.

    `update_models` is the seam: it is the one method that sees both the batch on its way to the
    learner and the criteria on the way back, so it can add the two `fake_*_sample` names the
    template reads and take the `fake_A` / `fake_B` it outputs. Nothing else in the loop can --
    every event hands a participant the trainer alone, by which time the step's tensors are gone.

    What the buffer is queried with is therefore the *previous* step's generated images: the current
    step's do not exist until the call returns. The first step has none at all and is given zeros --
    a flat grey frame in the [-1, 1] range the loader emits, which is a correct thing to label fake,
    and is one step out of the hundreds of thousands a run takes.
    """

    pool_size: int = 50
    """Images each domain's buffer holds; 0 disables the buffer and feeds the last step's images."""

    pool_seed: int = 42
    """Seed of the two buffers; the second is seeded from the next integer, so they draw apart."""

    previous: dict[str, Tensor] = field(default_factory=dict)
    """The generated images of the last step, keyed `fake_A` / `fake_B`; empty before the first."""

    @cached_property
    def pools(self) -> dict[str, ImagePool]:
        """One buffer per domain, as the paper keeps: a generated A is never shown to D_B."""
        return {
            domain: ImagePool(size=self.pool_size, seed=self.pool_seed + offset) for offset, domain in enumerate("AB")
        }

    def update_models(self, __inputs__: Any) -> dict[str, Any]:
        """Add the two buffer samples to the batch, then keep the generated images the step returned.

        Args:
            __inputs__ (Any): The batch the training dataset produced, holding `real_A` and `real_B`.

        Returns:
            dict[str, Any]: The criteria of the step, unchanged.
        """
        fakes = self.previous or {f"fake_{d}": torch.zeros_like(__inputs__[f"real_{d}"]) for d in self.pools}
        samples = {f"fake_{d}_sample": pool.query(fakes[f"fake_{d}"]) for d, pool in self.pools.items()}
        criteria = super().update_models({**__inputs__, **samples})
        self.previous = {f"fake_{d}": criteria[f"fake_{d}"].detach() for d in self.pools}
        return criteria
