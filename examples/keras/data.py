"""Example Keras input pipeline for image classification runs.

The package knows nothing about datasets: `scm keras train` instantiates whatever the
`--training-dataset` / `--validation-dataset` patterns describe, and the training loop takes any
iterable of dictionaries whose keys are the learner's input names. `KerasImageData` is therefore
example code, referenced from a configuration by file path -- see
`cfg/keras/others/default_keras.yaml`:

```yaml
_obj_:
  - _addr_: KerasImageData
    _file_: examples/keras/data.py
  - _attr_: model_validate
  - - _call_
    - {dataset: cifar10, training: true, batch_size: 32, image_size: [32, 32]}
```

`dataset` is either the name of a `keras.datasets` set, which is downloaded once into the Keras
cache directory and read into memory through the module-level `load_arrays` below, or the path of
one split's directory laid out one folder per class -- the same tree
`cfg/torch/others/default_timm.yaml` points timm at, so one dataset directory on the host serves
both. The path is the only form that scales: a set of ImageNet's size never becomes an array, and
the directory form decodes one batch at a time.

Augmentation happens here, in the `tf.data` pipeline, and never inside the model: Keras' image
preprocessing layers fall back to TensorFlow operations when a `tf.data` pipeline traces them, so
one pipeline feeds a run on any Keras backend, while a layer built into the model would augment
whatever loads that model afterwards. Building the pipeline therefore needs `tensorflow` installed
even for a `jax` or `torch` backend run.

The batch keys are `image_key` and `label_key`, which is where a learner whose inputs are named
differently is served. A `structcast` `FlexSpec` would be the other way to remap them, and it is
deliberately not offered here: it compares each constructed value against a sentinel, which raises
on a NumPy array.
"""

from collections.abc import Iterator
from functools import cached_property
import os
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, DirectoryPath
import tensorflow as tf

import keras

EXTENSIONS = frozenset({".bmp", ".gif", ".jpeg", ".jpg", ".png"})
"""Suffixes read as images, lowercased: the set `keras.utils.image_dataset_from_directory` lists."""


def list_labelled_images(root: Path) -> tuple[list[str], list[int]]:
    """The images under *root* and their class indices, from a tree laid out one folder per class.

    The other seam between the pipeline and where its items come from, the streaming twin of
    `load_arrays`. A label is its folder's position among the sorted folder names, which is what
    `keras.utils.image_dataset_from_directory` and timm both do over the same tree -- so a class
    keeps its index whichever framework reads the directory. Each class folder is read recursively
    and the files are sorted, so the listing is the same list every run: the sharding below cuts
    that order, and an order that varied by itself would hand a rank a different share each epoch.

    Args:
        root (Path): One split's directory, e.g. `.../imagenet/train`, one folder per class.

    Returns:
        tuple[list[str], list[int]]: The file paths and their class indices, paired by position.

    Raises:
        ValueError: If the tree holds no image, which would leave an epoch with nothing in it.
    """
    paths: list[str] = []
    labels: list[int] = []
    for label, folder in enumerate(sorted(entry for entry in root.iterdir() if entry.is_dir())):
        for path in sorted(folder.rglob("*")):
            if path.suffix.lower() in EXTENSIONS:
                paths.append(str(path))
                labels.append(label)
    if not paths:
        raise ValueError(
            f'No class folder under "{root}" holds an image with a suffix in {sorted(EXTENSIONS)}. '
            "The directory form names one split's directory, e.g. .../imagenet/train, whose images "
            "sit one folder per class -- not the dataset root and not a flat folder of files."
        )
    return paths, labels


def load_arrays(dataset: str, training: bool) -> tuple[np.ndarray, np.ndarray]:
    """Load one split of a `keras.datasets` set as (uint8 images, int64 labels).

    The single seam between the pipeline below and where its arrays come from: point it at your own
    corpus and everything downstream -- the sharding, the augmentation, the batching -- is unchanged.

    Args:
        dataset (str): Name of a set in `keras.datasets`, e.g. `cifar10`.
        training (bool): Whether to return the training split rather than the test split.

    Returns:
        tuple[np.ndarray, np.ndarray]: The images as `(n, height, width, channels)` and the labels
            as `(n,)`.
    """
    train, validation = getattr(keras.datasets, dataset).load_data()
    images, labels = train if training else validation
    # MNIST arrives as (n, 28, 28) and the others as (n, 32, 32, 3); this gives both a channel axis.
    return images.reshape(*images.shape[:3], -1).astype("uint8"), labels.reshape(-1).astype("int64")


def rank_and_world() -> tuple[int, int]:
    """Return this process's rank and the number of processes in the launch, or `(0, 1)`.

    Read from `RANK` and `WORLD_SIZE` rather than from a framework, because those are what a
    launcher sets and this file must not import one: only the torch Keras backend runs multi-process
    at all, and the pipeline has to keep working on the other two.

    Returns:
        tuple[int, int]: The rank and the world size.
    """
    return int(os.environ.get("RANK", "0")), int(os.environ.get("WORLD_SIZE", "1"))


class KerasImageData(BaseModel):
    """A `tf.data` pipeline yielding dictionary batches keyed by the model's input names.

    Iterating it yields NumPy arrays rather than TensorFlow tensors, so the batches are equally
    usable on the `jax` and `torch` backends.
    """

    dataset: Literal["mnist", "cifar10", "cifar100"] | DirectoryPath
    """The `keras.datasets` set to read, or the directory of one split, one folder per class.

    One field rather than a name plus a separate directory, because the two are alternatives and a
    pair of fields would let a configuration set both and silently honor one. Pydantic discriminates
    them -- a value that is neither a known name nor an existing directory is refused at
    construction, naming both possibilities -- so nothing here sniffs for a path separator.

    Required: a default would download something unasked. It names one split's directory, not the
    dataset root, so `training` decides how that split is read and not which one it is.
    """

    training: bool = False
    """Whether this is the training split: it is shuffled and augmented, the other one is not."""

    batch_size: int = 32
    """Items per batch. The final short batch is dropped, so every batch has this many items."""

    image_size: tuple[int, int] = (32, 32)
    """The (height, width) the model sees.

    A directory's images are decoded straight to this size. An in-memory image larger than this is
    randomly cropped to it when training and resized to it when not, the usual pair.
    """

    crop_padding: int = 4
    """Pixels of zero padding added on each side before the random crop; 0 disables the crop.

    Pad-then-crop is the small-image recipe. It is not the scale-and-aspect jitter an ImageNet run
    usually trains under -- a directory's images are resized to `image_size` outright, aspect ratio
    and all -- so a run reproducing a published ImageNet number replaces this with a random resized
    crop of its own.
    """

    shuffle_buffer: int = 1024
    """Items the training shuffle of a `keras.datasets` set holds at once, capped by this rank's share.

    A full-split buffer is what a shuffle wants and what an in-memory set cannot have for free: the
    buffer would be a second copy of the array. It does not bound a directory -- there the shuffle
    runs over the file list, before the decode, so this rank's whole share is permuted every epoch
    and the buffer holds paths rather than images.
    """

    seed: int = 42
    """Seed of the shuffle and of the augmentation draws."""

    image_key: str = "image"
    """The batch key the images are stored under, i.e. the model input they feed."""

    label_key: str = "label"
    """The batch key the labels are stored under."""

    @cached_property
    def source(self) -> tf.data.Dataset:
        """The whole split as unbatched, unshuffled items: `(path, label)` pairs, or `(image, label)`.

        Unbatched and unshuffled on purpose, in both forms: the sharding below has to cut the split
        by item, before anything groups or reorders it, so this stage only decides where the items
        come from.

        A directory is listed by `list_labelled_images` and handed on as paths rather than pixels,
        so a set of ImageNet's size costs a list of strings and `_decode` reads a file only once the
        shard and the shuffle below have picked it. A `keras.datasets` name is loaded into memory
        instead, which is the whole point of the small sets.

        The two differ in what they hand over -- a directory yields a path and an int32 label, an
        array set a uint8 image and an int64 one -- and `_decode` plus `_prepare` below are what
        make the batch contract identical either way.
        """
        if isinstance(self.dataset, Path):
            return tf.data.Dataset.from_tensor_slices(list_labelled_images(self.dataset))
        return tf.data.Dataset.from_tensor_slices(load_arrays(self.dataset, self.training))

    @property
    def items(self) -> int:
        """Number of items in the whole split, from the file listing or the array length."""
        return int(self.source.cardinality())

    @property
    def shard_items(self) -> int:
        """Number of items this rank owns: an equal share of the split, the indivisible tail cut."""
        _, world = rank_and_world()
        return self.items // world

    @cached_property
    def augmentation(self) -> list[keras.layers.Layer]:
        """The preprocessing layers applied to one batch, in order.

        Applied one by one rather than through a `keras.Sequential`: these run inside a `tf.data`
        graph, where a preprocessing layer switches to TensorFlow operations by itself but the model
        container around it would not.
        """
        layers: list[keras.layers.Layer] = []
        if self.training:
            layers.append(keras.layers.RandomFlip("horizontal", seed=self.seed))
            if self.crop_padding:
                layers.append(keras.layers.RandomCrop(*self.image_size, seed=self.seed))
        layers.append(keras.layers.Resizing(*self.image_size))
        layers.append(keras.layers.Rescaling(scale=1.0 / 255))
        return layers

    def _decode(self, path: Any, label: Any) -> tuple[Any, Any]:
        """Read one listed file and resize it, exactly as `image_dataset_from_directory` did.

        Args:
            path (Any): The image file to read, as a scalar string tensor.
            label (Any): Its class index, carried through untouched so the pair stays paired.

        Returns:
            tuple[Any, Any]: The float32 image in 0..255 at `image_size` and its label.
        """
        image = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
        return tf.image.resize(image, self.image_size), label

    def _prepare(self, images: Any, labels: Any) -> dict[str, Any]:
        """Augment one batch of images and key it by the model's input names.

        The labels are cast because the two sources disagree on their width; the images need no
        cast, since `Rescaling` ends the chain in float32 either way. Both sources therefore leave
        one batch contract, which is what lets a run swap a small set for a directory unchanged.
        """
        if self.training and self.crop_padding:
            # `tf.pad`, not a Keras layer: the pad-then-crop recipe needs a padding Keras has no
            # preprocessing layer for, and a plain layer would run its own backend's operations
            # inside this TensorFlow graph.
            pad = self.crop_padding
            images = tf.pad(images, [[0, 0], [pad, pad], [pad, pad], [0, 0]])
        for layer in self.augmentation:
            images = layer(images)
        return {self.image_key: images, self.label_key: tf.cast(labels, "int64")}

    @cached_property
    def pipeline(self) -> tf.data.Dataset:
        """The sharded, batched, augmented and prefetched pipeline over this rank's share.

        Under a multi-process launch each rank takes every `WORLD_SIZE`-th item starting at its own
        rank, so the shards are disjoint and every rank sees the same number of batches; the tail
        the world size does not divide is dropped first, as `DistributedSampler` plus `drop_last`
        does on the torch side. Which items a rank owns is fixed for the whole run and only their
        order is reshuffled each epoch -- the same thing `DistributedSampler` does when nobody calls
        `set_epoch`, and what the torch example accepts too. `examples/keras/corpus.py` shards after
        its shuffle instead, so there a rank's items do change between epochs.

        Outside such a launch -- the tensorflow and jax backends, and any single-process run --
        every rank is rank 0 and the whole split is served. That is not an oversight: those backends
        run one process and the distributed strategy splits each batch across the replicas itself,
        so a loader that sharded here as well would hand each replica a shard of a shard.

        The sharding comes before the shuffle and the batching, so it cuts the split by item however
        the items are stored, and the decode of a directory happens after both -- a rank never reads
        a file another rank owns, and never reads one twice in an epoch.

        A directory's shuffle covers this rank's whole share, because there the items being
        reordered are paths: a tree laid out one folder per class is listed class by class, and a
        buffered shuffle of the decoded images could only ever mix a window of that order -- a
        thousand images are 0.08% of ImageNet, which leaves every batch two or three adjacent
        classes (the flax twin of this pipeline went NaN on exactly that, in H200 tier 2 run 10-f).
        An array set arrives decoded and already in memory, so it keeps its bounded `shuffle_buffer`.
        """
        data = self.source
        rank, world = rank_and_world()
        streaming = isinstance(self.dataset, Path)
        if world > 1:
            data = data.take(self.items - self.items % world).shard(world, rank)
        if self.training:
            buffer = self.shard_items if streaming else min(self.shard_items, self.shuffle_buffer)
            data = data.shuffle(buffer, seed=self.seed, reshuffle_each_iteration=True)
        if streaming:
            data = data.map(self._decode, num_parallel_calls=tf.data.AUTOTUNE)
        data = data.batch(self.batch_size, drop_remainder=True)
        return data.map(self._prepare, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)

    def __len__(self) -> int:
        """Number of batches one rank sees per epoch, the short final batch being dropped."""
        return self.shard_items // self.batch_size

    def __iter__(self) -> Iterator[dict[str, Any]]:
        """The batches of one epoch, as NumPy arrays keyed by `image_key` and `label_key`."""
        yield from self.pipeline.as_numpy_iterator()
