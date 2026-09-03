r"""Example unpaired two-domain `tf.data` pipeline for Flax (nnx) CycleGAN runs.

`cfg/flax/learners/CycleGAN.yaml` declares `INPUTS: [real_A, real_B]` and nothing else: its
discriminator segments train on the images the generator segment just produced, so unlike the torch
twin it needs no replay buffer fed into the batch and this file is only the loader. Two directories,
one per domain, drawn independently -- which is what "unpaired" means:

```bash
scm flax create model cfg/flax/models/CycleGAN_generator.yaml -o generator.py
scm flax create model cfg/flax/models/CycleGAN_discriminator.yaml -o discriminator.py
# steps_per_epoch is what turns the template's epoch counts into the step counts an optax schedule
# reads; it is len(training_dataset), which the command prints before the first epoch.
scm flax create learner cfg/flax/learners/CycleGAN.yaml -p 'DEFAULT: {steps_per_epoch: 1334}' -o learner.py

scm flax train \\
    'G_AB: [_obj_, {_addr_: Model, _file_: generator.py}]' \\
    'G_BA: [_obj_, {_addr_: Model, _file_: generator.py}]' \\
    'D_A: [_obj_, {_addr_: Model, _file_: discriminator.py}]' \\
    'D_B: [_obj_, {_addr_: Model, _file_: discriminator.py}]' \\
    -L '[_obj_, {_addr_: Learner, _file_: learner.py}]' \\
    -s 'image: [256, 256, 3]' \\
    --training-dataset '[_obj_, {_addr_: UnpairedImageLoader, _file_: examples/flax/cyclegan.py},
                         {_call_: {root_A: data/horse2zebra/trainA, root_B: data/horse2zebra/trainB}}]' \\
    -e 200 -LC loss_G -E cyclegan
```

The batches are NHWC NumPy arrays scaled to [-1, 1], which is the layout
`cfg/flax/models/CycleGAN_generator.yaml` reads and the range its closing `flax.nnx.tanh` emits --
the identity and cycle losses compare a real image against exactly that.

As in `examples/flax/data.py` there is no rank sharding: JAX is single-controller, so the strategy
splits each batch across the devices and a loader that also sharded per rank would double-shard.
"""

from collections.abc import Iterator
from functools import cached_property
from pathlib import Path
from typing import Any

from pydantic import BaseModel, DirectoryPath
import tensorflow as tf

EXTENSIONS = frozenset({".bmp", ".jpeg", ".jpg", ".png", ".webp"})
"""Suffixes read as images, lowercased. The horse2zebra tree is JPEG; the others cost nothing."""


def hide_gpus_from_tensorflow() -> None:
    """Take every GPU out of TensorFlow's sight, so it never reserves the memory JAX needs.

    The twin of `examples/flax/data.py`, which carries the long version: called when a loader is
    constructed rather than on import, and a no-op once TensorFlow has initialized its devices.
    """
    try:
        tf.config.set_visible_devices([], "GPU")
    except RuntimeError:
        pass


def list_images(root: Path) -> list[str]:
    """The images under *root*, sorted, so the same tree is read in the same order twice.

    Args:
        root (Path): One domain's directory, read recursively.

    Returns:
        list[str]: The paths as strings, which is what `tf.data` slices into a source.

    Raises:
        ValueError: If the directory holds no image, which would make an epoch undefined.
    """
    paths = sorted(str(path) for path in root.rglob("*") if path.suffix.lower() in EXTENSIONS)
    if not paths:
        raise ValueError(
            f'The directory "{root}" holds no image with a suffix in {sorted(EXTENSIONS)}. It names '
            "one domain's directory of images, not the dataset root: the horse2zebra set is two of "
            "them, trainA and trainB."
        )
    return paths


class UnpairedImageLoader(BaseModel):
    """A `tf.data` pipeline yielding `{"real_A": ..., "real_B": ...}` batches from two directories.

    Nothing aligns the two domains: each is shuffled on its own and the two streams are zipped, so
    an image of A meets a different image of B on every epoch. An epoch is the longer of the two
    directories -- the convention of the reference implementation, which keeps every image of the
    larger domain seen once -- and the shorter one repeats within it.
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
    """Whether the two domains are shuffled and augmented. The other way is deterministic and unaugmented."""

    hflip: bool = True
    """Whether the training augmentation flips images horizontally."""

    batch_size: int = 1
    """Items per batch. The paper trains CycleGAN with one; the final short batch is dropped."""

    seed: int = 42
    """Seed of the two shuffles and of the per-item augmentation draws, so a run is reproducible."""

    def model_post_init(self, context: Any, /) -> None:
        """Hide the GPUs from TensorFlow, now that something in the run is about to read data."""
        hide_gpus_from_tensorflow()

    @cached_property
    def sources(self) -> tuple[tf.data.Dataset, tf.data.Dataset]:
        """The two domains as datasets of file paths, listed once and decoded as they are pulled.

        Paths rather than pixels, as in `examples/flax/data.py`: a directory of any size costs a
        list of strings here, and the decode happens after everything downstream.
        """
        return tuple(tf.data.Dataset.from_tensor_slices(list_images(root)) for root in (self.root_A, self.root_B))

    @cached_property
    def items(self) -> int:
        """Items in one epoch: the longer of the two directories."""
        return max(int(source.cardinality()) for source in self.sources)

    # AutoGraph has nothing to rewrite here -- every branch is decided while the graph is traced --
    # and it fails to introspect a method of a pydantic model.
    @tf.autograph.experimental.do_not_convert
    def _image(self, path: tf.Tensor, seed: tf.Tensor) -> tf.Tensor:
        """Decode one image and return it as `[crop_size, crop_size, 3]` scaled to [-1, 1].

        The random draws are stateless and keyed by the item's position in the stream, so the
        augmentation is a function of that position rather than of a global RNG.

        Args:
            path (tf.Tensor): The image file to read.
            seed (tf.Tensor): The two-element stateless seed of the crop and the flip.

        Returns:
            tf.Tensor: The float32 image the models read.
        """
        image = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
        image = tf.image.resize(image, (self.load_size, self.load_size), method="bicubic")
        if self.is_training:
            image = tf.image.stateless_random_crop(image, (self.crop_size, self.crop_size, 3), seed=seed)
            if self.hflip:
                # A seed of its own, so the flip is not perfectly correlated with the crop offset.
                image = tf.image.stateless_random_flip_left_right(image, seed=seed + 1)
        else:
            image = tf.image.resize(image, (self.crop_size, self.crop_size), method="bicubic")
        return image / 127.5 - 1.0

    @tf.autograph.experimental.do_not_convert
    def _preprocess(self, index: tf.Tensor, paths: tuple[tf.Tensor, tf.Tensor]) -> dict[str, tf.Tensor]:
        """Decode one unaligned pair into the two batch names the learner reads."""
        seed = tf.stack([tf.cast(index, tf.int32) + self.seed, self.seed])
        return {"real_A": self._image(paths[0], seed), "real_B": self._image(paths[1], seed + 2)}

    @cached_property
    def dataset(self) -> tf.data.Dataset:
        """The zipped, decoded, batched pipeline over one epoch.

        Each domain is shuffled and repeated on its own before the zip, so the shorter one wraps
        around inside an epoch and the pairing is redrawn on every one of them.
        """
        sources = self.sources
        if self.is_training:
            sources = tuple(
                source.shuffle(int(source.cardinality()), seed=self.seed + offset)
                for offset, source in enumerate(sources)
            )
        zipped = tf.data.Dataset.zip(tuple(source.repeat() for source in sources)).take(self.items)
        return (
            zipped.enumerate()
            .map(self._preprocess, num_parallel_calls=tf.data.AUTOTUNE)
            .batch(self.batch_size, drop_remainder=True)
            .prefetch(tf.data.AUTOTUNE)
        )

    def __len__(self) -> int:
        """Number of batches one epoch yields, the final short batch being dropped."""
        return self.items // self.batch_size

    def __call__(self) -> Iterator[dict[str, Any]]:
        """Yield one epoch of batches as NumPy arrays, keyed `real_A` and `real_B`."""
        yield from self.dataset.as_numpy_iterator()
