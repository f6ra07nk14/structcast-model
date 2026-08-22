"""Example `tf.data` input pipeline for Flax (nnx) training runs.

The package knows nothing about datasets: `scm flax train` instantiates whatever the
`--training-dataset` / `--validation-dataset` patterns describe, wraps it so every batch lands on
the strategy's mesh, and iterates it. A `tf.data` integration is therefore example code, referenced
from a configuration by file path -- see `cfg/flax/others/default_tfdata.yaml`:

```yaml
_obj_:
  - _addr_: TFDataLoader
    _file_: examples/flax/data.py
  - _attr_: model_validate
  - - _call_
    - spec: {image: 0, label: 1}
      name: cifar10
      split: train
      is_training: true
```

The pipeline is big_vision-shaped and host-side: resize, then a random crop and a horizontal flip
while training or a central crop while evaluating, then the channel-wise normalization. `tf.data`
runs all of it on CPU threads while the device is busy with the previous step, which is what keeps
a JAX run fed. Building a loader takes every GPU out of TensorFlow's sight, so the two frameworks
never fight over device memory.

With `name` left empty the loader serves a deterministic synthetic split instead, so an example, a
smoke test or a CI run needs no download at all; a non-empty `name` is loaded through
`tensorflow_datasets`, which is not a dependency of this package and must be installed separately.

Unlike `examples/torch/data.py` there is no epoch hook and no rank sharding here: `tf.data`
reshuffles on every iteration by itself, and JAX is single-controller, so the strategy splits each
batch across the devices rather than the loader feeding each rank its own slice.
"""

from collections.abc import Iterator
from functools import cached_property
from typing import Any, Literal

from pydantic import BaseModel, Field
import tensorflow as tf

DTYPES = {"float32": tf.float32, "float16": tf.float16, "bfloat16": tf.bfloat16}
"""TensorFlow element types of the supported image element types."""


def hide_gpus_from_tensorflow() -> None:
    """Take every GPU out of TensorFlow's sight, so it never reserves the memory JAX needs.

    Called when a loader is constructed rather than when this module is imported: importing it is
    something a test collector or a neighbouring example may do incidentally, and blinding the
    whole process from an import would be a side effect nobody asked for. TensorFlow refuses the
    change once its devices are initialized, which is the `RuntimeError` swallowed here -- by then
    another part of the process has already decided, and this pipeline is host-side either way.
    """
    try:
        tf.config.set_visible_devices([], "GPU")
    except RuntimeError:
        pass


class TFDataLoader(BaseModel):
    """A `tf.data` pipeline yielding batches of NumPy arrays keyed by the learner's input names.

    Every field has a default that works without any data on disk: the loader then serves
    `synthetic_samples` deterministic random images, which is what makes this file runnable in a
    test. Set `name` to load a real dataset through `tensorflow_datasets` instead.
    """

    spec: dict[str, int] = Field(default_factory=lambda: {"image": 0, "label": 1})
    """Learner input name -> position in the `(image, label)` pair each batch is built from.

    A plain mapping, not the `FlexSpec` the timm example takes: the batches leave this pipeline as
    NumPy arrays -- which is what lets the strategy place them on its mesh in one transfer -- and a
    `FlexSpec` compares each constructed value against a sentinel, which on a NumPy array is an
    elementwise comparison rather than a truth value.
    """

    name: str = ""
    """The `tensorflow_datasets` name to load, e.g. "cifar10". Empty serves the synthetic split."""

    split: str = "train"
    """The split to read, in `tensorflow_datasets` split syntax. Ignored by the synthetic split."""

    data_dir: str | None = None
    """Directory the dataset is read from and downloaded into; the tfds default when None."""

    download: bool = False
    """Whether a missing dataset may be downloaded."""

    is_training: bool = False
    """Whether this is the training split: it decides shuffling, repetition and the augmentation."""

    batch_size: int = 32
    """Items per batch."""

    image_size: int = 224
    """Side length of the square image the pipeline emits."""

    crop_pct: float = 0.875
    """Crop ratio: images are resized to `image_size / crop_pct` before being cropped."""

    mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    """Per-channel mean subtracted after scaling the image to [0, 1]."""

    std: tuple[float, float, float] = (0.229, 0.224, 0.225)
    """Per-channel standard deviation divided out after the mean is subtracted."""

    image_dtype: Literal["float32", "float16", "bfloat16"] = "float32"
    """Element type of the emitted images."""

    hflip: bool = True
    """Whether the training augmentation flips images horizontally."""

    shuffle_buffer: int = 1024
    """Items held for shuffling a training split; the whole split is shuffled when it is smaller."""

    seed: int = 42
    """Seed of the shuffle and of the per-item augmentation draws, so a run is reproducible."""

    num_classes: int = 10
    """Number of classes the synthetic split labels cycle through."""

    synthetic_samples: int = 256
    """Number of items the synthetic split holds."""

    drop_remainder: bool = True
    """Whether the final short batch is dropped. A Flax run wants this: the strategy splits a batch
    across its mesh, so a short one would not divide."""

    def model_post_init(self, context: Any, /) -> None:
        """Hide the GPUs from TensorFlow, now that something in the run is about to read data."""
        hide_gpus_from_tensorflow()

    @property
    def resize_size(self) -> int:
        """Side length the image is resized to before it is cropped to `image_size`."""
        return round(self.image_size / self.crop_pct)

    @cached_property
    def source(self) -> tf.data.Dataset:
        """The undecoded `(uint8 image, int32 label)` pairs of this split.

        The synthetic branch builds them from a fixed seed, so the same items come back on every
        run and nothing is downloaded.
        """
        if not self.name:
            images = tf.random.stateless_uniform(
                (self.synthetic_samples, self.resize_size, self.resize_size, 3),
                seed=(self.seed, self.seed),
                maxval=256,
                dtype=tf.int32,
            )
            labels = tf.range(self.synthetic_samples, dtype=tf.int32) % self.num_classes
            return tf.data.Dataset.from_tensor_slices((tf.cast(images, tf.uint8), labels))
        # Imported here, not at module scope: `tensorflow_datasets` is not a dependency of this
        # package, and the synthetic split above must stay usable without it.
        import tensorflow_datasets as tfds  # noqa: PLC0415  # optional, only the named-dataset path needs it

        return tfds.load(
            self.name,
            split=self.split,
            data_dir=self.data_dir,
            download=self.download,
            as_supervised=True,
            shuffle_files=self.is_training,
        )

    @cached_property
    def num_examples(self) -> int:
        """Number of items in the split, which is what `__len__` divides into batches."""
        if not self.name:
            return self.synthetic_samples
        return int(self.source.cardinality())

    # AutoGraph has nothing to rewrite here -- every branch below is decided while the graph is
    # traced, and the rest is tf ops -- and it fails to introspect a method of a pydantic model.
    @tf.autograph.experimental.do_not_convert
    def _preprocess(self, index: tf.Tensor, item: tuple[tf.Tensor, tf.Tensor]) -> tuple[tf.Tensor, tf.Tensor]:
        """Resize, crop, flip and normalize one image.

        The random draws are stateless and keyed by *index*, so the augmentation of an item is a
        function of its position in the (shuffled) stream rather than of a global RNG the graph
        would carry between workers.

        Args:
            index (tf.Tensor): Position of the item in the stream, which seeds the random draws.
            item (tuple[tf.Tensor, tf.Tensor]): The uint8 image and its label.

        Returns:
            tuple[tf.Tensor, tf.Tensor]: The preprocessed image and its label.
        """
        image, label = item
        seed = tf.stack([tf.cast(index, tf.int32) + self.seed, self.seed])
        image = tf.image.resize(image, (self.resize_size, self.resize_size), method="bicubic")
        if self.is_training:
            image = tf.image.stateless_random_crop(image, (self.image_size, self.image_size, 3), seed=seed)
            if self.hflip:
                # A seed of its own, so the flip is not perfectly correlated with the crop offset.
                image = tf.image.stateless_random_flip_left_right(image, seed=seed + 1)
        else:
            image = tf.image.resize_with_crop_or_pad(image, self.image_size, self.image_size)
        image = (image / 255.0 - tf.constant(self.mean)) / tf.constant(self.std)
        return tf.cast(image, DTYPES[self.image_dtype]), tf.cast(label, tf.int32)

    @cached_property
    def dataset(self) -> tf.data.Dataset:
        """The batched pipeline: shuffle (training only), preprocess, batch, prefetch."""
        dataset = self.source
        if self.is_training:
            dataset = dataset.shuffle(min(self.shuffle_buffer, self.num_examples), seed=self.seed)
        return (
            dataset.enumerate()
            .map(self._preprocess, num_parallel_calls=tf.data.AUTOTUNE)
            .batch(self.batch_size, drop_remainder=self.drop_remainder)
            .prefetch(tf.data.AUTOTUNE)
        )

    def __len__(self) -> int:
        """Number of batches one epoch yields."""
        whole, remainder = divmod(self.num_examples, self.batch_size)
        return whole if self.drop_remainder else whole + bool(remainder)

    def __call__(self) -> Iterator[dict[str, Any]]:
        """Yield one epoch of batches as NumPy arrays, keyed by the learner input names of `spec`."""
        for batch in self.dataset.as_numpy_iterator():
            yield {name: batch[index] for name, index in self.spec.items()}
