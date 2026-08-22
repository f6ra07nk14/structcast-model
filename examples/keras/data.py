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

The arrays come from `keras.datasets`, downloaded once into the Keras cache directory, through the
module-level `load_arrays` below -- which is also where a corpus of your own is plugged in, without
touching the pipeline.

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
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel
import tensorflow as tf

import keras


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

    dataset: Literal["mnist", "cifar10", "cifar100"]
    """The `keras.datasets` set to read. Required: a default would download something unasked."""

    training: bool = False
    """Whether this is the training split: it is shuffled and augmented, the other one is not."""

    batch_size: int = 32
    """Items per batch. The final short batch is dropped, so every batch has this many items."""

    image_size: tuple[int, int] = (32, 32)
    """The (height, width) the model sees.

    A training image larger than this is randomly cropped to it and a validation image is resized to
    it, the usual pair.
    """

    crop_padding: int = 4
    """Pixels of zero padding added on each side before the random crop; 0 disables the crop."""

    seed: int = 42
    """Seed of the shuffle and of the augmentation draws."""

    image_key: str = "image"
    """The batch key the images are stored under, i.e. the model input they feed."""

    label_key: str = "label"
    """The batch key the labels are stored under."""

    @cached_property
    def arrays(self) -> tuple[np.ndarray, np.ndarray]:
        """This process's share of the split, as (uint8 images, int64 labels), loaded once.

        Under a multi-process launch each rank takes every `WORLD_SIZE`-th item starting at its own
        rank, so the shards are disjoint and every rank sees the same number of batches; the tail
        the world size does not divide is dropped, as `DistributedSampler` plus `drop_last` does on
        the torch side. Which items a rank owns is fixed for the whole run and only their order is
        reshuffled each epoch -- the same thing `DistributedSampler` does when nobody calls
        `set_epoch`, and what the torch example accepts too. `examples/keras/corpus.py` shards after
        its shuffle instead, so there a rank's items do change between epochs.

        Outside such a launch -- the tensorflow and jax backends, and any single-process run --
        every rank is rank 0 and the whole split is served. That is not an oversight: those backends
        run one process and the distributed strategy splits each batch across the replicas itself,
        so a loader that sharded here as well would hand each replica a shard of a shard.
        """
        images, labels = load_arrays(self.dataset, self.training)
        rank, world = rank_and_world()
        if world > 1:
            usable = len(labels) - len(labels) % world
            images, labels = images[rank:usable:world], labels[rank:usable:world]
        return images, labels

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

    def _prepare(self, images: Any, labels: Any) -> dict[str, Any]:
        """Augment one batch of images and key it by the model's input names."""
        if self.training and self.crop_padding:
            # `tf.pad`, not a Keras layer: the pad-then-crop recipe needs a padding Keras has no
            # preprocessing layer for, and a plain layer would run its own backend's operations
            # inside this TensorFlow graph.
            pad = self.crop_padding
            images = tf.pad(images, [[0, 0], [pad, pad], [pad, pad], [0, 0]])
        for layer in self.augmentation:
            images = layer(images)
        return {self.image_key: images, self.label_key: labels}

    @cached_property
    def pipeline(self) -> tf.data.Dataset:
        """The batched, augmented and prefetched pipeline over this process's share of the split."""
        images, labels = self.arrays
        data = tf.data.Dataset.from_tensor_slices((images, labels))
        if self.training:
            data = data.shuffle(len(labels), seed=self.seed, reshuffle_each_iteration=True)
        data = data.batch(self.batch_size, drop_remainder=True)
        return data.map(self._prepare, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)

    def __len__(self) -> int:
        """Number of batches one rank sees per epoch, the short final batch being dropped."""
        return len(self.arrays[1]) // self.batch_size

    def __iter__(self) -> Iterator[dict[str, Any]]:
        """The batches of one epoch, as NumPy arrays keyed by `image_key` and `label_key`."""
        yield from self.pipeline.as_numpy_iterator()
