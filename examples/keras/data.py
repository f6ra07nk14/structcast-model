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

Augmentation happens here, in the `tf.data` pipeline, and never inside the model: Keras' image
preprocessing layers fall back to TensorFlow operations when a `tf.data` pipeline traces them, so
one pipeline feeds a run on any Keras backend, while a layer built into the model would augment
whatever loads that model afterwards. Building the pipeline therefore needs `tensorflow` installed
even for a `jax` or `torch` backend run.

The batch keys are `image_key` and `label_key`, which is where a learner whose inputs are named
differently is served. A `structcast` `FlexSpec` would be the other way to remap them, and it is
deliberately not offered here: it compares each constructed value against a sentinel, which raises
on a NumPy array.

`dataset: synthetic` builds a deterministic, learnable set of arrays from `seed` and downloads
nothing, which is what the tests use; the other names are `keras.datasets` sets, downloaded once
into the Keras cache directory.
"""

from collections.abc import Iterator
from functools import cached_property
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel
import tensorflow as tf

import keras


class KerasImageData(BaseModel):
    """A `tf.data` pipeline yielding dictionary batches keyed by the model's input names.

    Iterating it yields NumPy arrays rather than TensorFlow tensors, so the batches are equally
    usable on the `jax` and `torch` backends.
    """

    dataset: Literal["synthetic", "mnist", "cifar10", "cifar100"] = "synthetic"
    """The source of the arrays: a deterministic synthetic set, or a `keras.datasets` set."""

    training: bool = False
    """Whether this is the training split: it is shuffled and augmented, the other one is not."""

    batch_size: int = 32
    """Items per batch. The final short batch is dropped, so every batch has this many items."""

    image_size: tuple[int, int] = (32, 32)
    """The (height, width) the model sees.

    A training image larger than this is randomly cropped to it and a validation image is resized to
    it, the usual pair; the synthetic set is generated at exactly this size.
    """

    image_channels: int = 3
    """Channels of the synthetic images; a `keras.datasets` set brings its own."""

    num_classes: int = 10
    """Number of classes of the synthetic set; a `keras.datasets` set brings its own."""

    samples: int = 1024
    """Number of synthetic items to build, ignored for a `keras.datasets` set."""

    crop_padding: int = 4
    """Pixels of zero padding added on each side before the random crop; 0 disables the crop."""

    seed: int = 42
    """Seed of the synthetic arrays, of the shuffle and of the augmentation draws."""

    image_key: str = "image"
    """The batch key the images are stored under, i.e. the model input they feed."""

    label_key: str = "label"
    """The batch key the labels are stored under."""

    @cached_property
    def arrays(self) -> tuple[np.ndarray, np.ndarray]:
        """The whole split as (uint8 images, int64 labels), built or loaded once."""
        if self.dataset == "synthetic":
            return self._synthetic()
        train, validation = getattr(keras.datasets, self.dataset).load_data()
        images, labels = train if self.training else validation
        # MNIST arrives as (n, 28, 28) and the others as (n, 32, 32, 3); this gives both a channel axis.
        return images.reshape(*images.shape[:3], -1).astype("uint8"), labels.reshape(-1).astype("int64")

    def _synthetic(self) -> tuple[np.ndarray, np.ndarray]:
        """Build a learnable classification set: one random image per class, plus per-item noise."""
        rng = np.random.default_rng(self.seed)
        patterns = rng.random((self.num_classes, *self.image_size, self.image_channels))
        labels = rng.integers(0, self.num_classes, self.samples)
        noise = rng.normal(scale=0.1, size=(self.samples, *self.image_size, self.image_channels))
        return (np.clip(patterns[labels] + noise, 0.0, 1.0) * 255.0).astype("uint8"), labels.astype("int64")

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
        """The batched, augmented and prefetched pipeline over this split."""
        images, labels = self.arrays
        data = tf.data.Dataset.from_tensor_slices((images, labels))
        if self.training:
            data = data.shuffle(len(labels), seed=self.seed, reshuffle_each_iteration=True)
        data = data.batch(self.batch_size, drop_remainder=True)
        return data.map(self._prepare, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)

    def __len__(self) -> int:
        """Number of batches in one epoch, the short final batch being dropped."""
        return len(self.arrays[1]) // self.batch_size

    def __iter__(self) -> Iterator[dict[str, Any]]:
        """The batches of one epoch, as NumPy arrays keyed by `image_key` and `label_key`."""
        yield from self.pipeline.as_numpy_iterator()
