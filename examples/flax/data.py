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

`name` is either a `tensorflow_datasets` set, downloaded and cached by that package, or the path of
one split's directory laid out one folder per class -- the same tree `cfg/torch/others/default_timm.yaml`
points timm at and `examples/keras/data.py` reads, so one dataset directory on the host serves all
three frameworks. The path is the form that scales: a set of ImageNet's size never becomes an array,
and the directory form lists the files once and decodes one batch at a time.

`tensorflow_datasets` is not a dependency of this package: install it with
`uv pip install tensorflow-datasets` for the name form, use the directory form, or point
`--training-dataset` at a dataset object of your own, since the training loop takes any iterable of
dictionaries.

The pipeline is big_vision-shaped and host-side: resize, then a random crop and a horizontal flip
while training or a central crop while evaluating, then the channel-wise normalization. `tf.data`
runs all of it on CPU threads while the device is busy with the previous step, which is what keeps
a JAX run fed. Building a loader takes every GPU out of TensorFlow's sight, so the two frameworks
never fight over device memory.

Unlike `examples/torch/data.py` there is no epoch hook and no rank sharding here: `tf.data`
reshuffles on every iteration by itself, and JAX is single-controller, so the strategy splits each
batch across the devices rather than the loader feeding each rank its own slice -- a loader that
also sharded per rank would double-shard (see `examples/flax/corpus.py` for the long version).
"""

from collections.abc import Iterator
from functools import cached_property
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, DirectoryPath, Field, field_validator
from structcast.utils.lazy_import import try_import
import tensorflow as tf

with try_import() as _tfds_imports:  # not a dependency of structcast-model; the use site says so.
    import tensorflow_datasets as tfds

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

    `source` is the only place a dataset comes from: it is a `cached_property`, so a caller with a
    `tf.data.Dataset` already in hand -- a test, or a pipeline of its own -- replaces it by writing
    the instance dictionary, and nothing here downloads anything.
    """

    spec: dict[str, int] = Field(default_factory=lambda: {"image": 0, "label": 1})
    """Learner input name -> position in the `(image, label)` pair each batch is built from.

    A plain mapping, not the `FlexSpec` the timm example takes: the batches leave this pipeline as
    NumPy arrays -- which is what lets the strategy place them on its mesh in one transfer -- and a
    `FlexSpec` compares each constructed value against a sentinel, which on a NumPy array is an
    elementwise comparison rather than a truth value.
    """

    name: Annotated[DirectoryPath | str, Field(union_mode="left_to_right")]
    """The `tensorflow_datasets` set to read, e.g. "cifar10", or the directory of one split, one
    folder per class.

    One field rather than a name plus a separate directory, as in `examples/keras/data.py`: the two
    are alternatives, and a pair of fields would let a configuration set both and silently honor
    one. The keras twin discriminates them by a `Literal` of the three set names it knows; a tfds
    name is any string, so the union is resolved left to right instead -- an existing directory
    wins, anything else is a tfds name. A tfds set whose name is also a directory in the working
    directory would therefore read the directory; nothing else here sniffs for a path separator.

    It names one split's directory, not the dataset root, so `is_training` decides how that split
    is read and not which one it is. Required either way: a default would download something
    unasked, and an empty string is refused at construction.
    """

    split: str = "train"
    """The split to read, in `tensorflow_datasets` split syntax. Ignored by a directory."""

    data_dir: str | None = None
    """Directory tfds reads from and downloads into; its default when None. Ignored by a directory."""

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

    drop_remainder: bool = True
    """Whether the final short batch is dropped. A Flax run wants this: the strategy splits a batch
    across its mesh, so a short one would not divide."""

    @field_validator("name")
    @classmethod
    def _reject_an_unnamed_dataset(cls, name: Path | str) -> Path | str:
        """Refuse an empty name here rather than let tfds fail on it much later."""
        if isinstance(name, str) and not name.strip():
            raise ValueError(
                'A dataset name is required, e.g. name="cifar10": it is looked up with '
                "tensorflow_datasets, whose catalogue is at https://www.tensorflow.org/datasets/catalog. "
                "A path to one split's directory, one folder per class, is read from disk instead."
            )
        return name

    def model_post_init(self, context: Any, /) -> None:
        """Hide the GPUs from TensorFlow, now that something in the run is about to read data."""
        hide_gpus_from_tensorflow()

    @property
    def resize_size(self) -> int:
        """Side length the image is resized to before it is cropped to `image_size`."""
        return round(self.image_size / self.crop_pct)

    @cached_property
    def source(self) -> tf.data.Dataset:
        """The unbatched, unshuffled `(image, label)` pairs of this split.

        A directory is listed once with `keras.utils.image_dataset_from_directory` -- reached here
        as `tf.keras`, which is the same function and spares this file an import it would otherwise
        need for one call -- and its images are decoded as the pipeline pulls them, so a set of
        ImageNet's size costs a list of paths rather than a copy of the pixels. The listing is what
        `num_examples` counts, and the decode happens after everything downstream of this property.
        The labels come from the folder names, sorted, so a rerun over the same tree numbers the
        classes the same way. A tfds name is loaded through `tfds.load` instead.

        The two hand over different things -- a directory float32 images in 0..255 and int32 labels,
        tfds uint8 and int64 -- and `_preprocess` is what makes the batch contract identical either
        way. Images are decoded at `resize_size`, not at `image_size`: the crop that `crop_pct`
        exists for happens in `_preprocess`, and a source that had already resized to the final size
        would leave it nothing to crop.

        Raises:
            ImportError: If `tensorflow_datasets` is not installed and a tfds name was asked for.
        """
        if isinstance(self.name, Path):
            return tf.keras.utils.image_dataset_from_directory(
                self.name,
                labels="inferred",
                label_mode="int",
                image_size=(self.resize_size, self.resize_size),
                batch_size=None,
                shuffle=False,
                verbose=False,
            )
        if not _tfds_imports.is_successful:
            raise ImportError(
                f'Loading the dataset "{self.name}" needs the tensorflow_datasets package, which is not '
                "installed and is not a dependency of structcast-model. Install it with "
                '"uv pip install tensorflow-datasets", or point --training-dataset at a dataset object '
                "of your own -- the training loop takes any iterable of dictionaries."
            )
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
        """Number of items in the split, which is what `__len__` divides into batches.

        Raises:
            ValueError: If the split reports no size, which would make `__len__` meaningless.
        """
        count = int(self.source.cardinality())
        if count < 0:
            raise ValueError(
                f'The split "{self.split}" of "{self.name}" reports no size, so an epoch cannot be counted. '
                'A split written as a slice, e.g. "train[:5%]", reports one.'
            )
        return count

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
