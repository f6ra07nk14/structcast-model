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

`crop_pct` picks the transform. Left unset it is the small-image one this pipeline has always had,
a square resize and then a crop, which is what a CIFAR-sized set wants. Set, it is the one
`examples/torch/data.py` gets from timm, rebuilt on `tf.data` ops: a random resized crop drawn from
the same area and log-aspect distributions, a horizontal flip and brightness/contrast/saturation
jitter while training, a shortest-edge resize and a central crop of the same geometry while
evaluating, then the ImageNet channel statistics either way. The three residuals are named where
they are made: the resize is TensorFlow's bicubic rather than PIL's, the jitter is applied in a
fixed order rather than a drawn one, and it blends in float32 rather than in uint8.

`tf.data` runs all of it on CPU threads while the device is busy with the previous
step, which is what keeps a JAX run fed; `num_parallel_calls` and `prefetch` are the budget it is
allowed, the counterpart of the timm loader's `num_workers`. Building a loader takes every GPU out
of TensorFlow's sight, so the two frameworks never fight over device memory.

Unlike `examples/torch/data.py` there is no epoch hook and no rank sharding here: `tf.data`
reshuffles on every iteration by itself, and JAX is single-controller, so the strategy splits each
batch across the devices rather than the loader feeding each rank its own slice -- a loader that
also sharded per rank would double-shard (see `examples/flax/corpus.py` for the long version).
"""

from collections.abc import Iterator
from functools import cached_property
from math import floor, log
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, DirectoryPath, Field, field_validator
from structcast.utils.lazy_import import try_import
import tensorflow as tf

with try_import() as _tfds_imports:  # not a dependency of structcast-model; the use site says so.
    import tensorflow_datasets as tfds

DTYPES = {"float32": tf.float32, "float16": tf.float16, "bfloat16": tf.bfloat16}
"""TensorFlow element types of the supported image element types."""

DEFAULT_CROP_PCT = 0.875
"""Crop ratio the small-image transform resizes by, and timm's `DEFAULT_CROP_PCT`."""

CROP_ATTEMPTS = 10
"""Draws a random resized crop makes before it falls back to a centre crop, as torchvision does."""

EXTENSIONS = frozenset({".bmp", ".gif", ".jpeg", ".jpg", ".png"})
"""Suffixes read as images, lowercased: the set `keras.utils.image_dataset_from_directory` lists."""


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


def list_labelled_images(root: Path) -> tuple[list[str], list[int]]:
    """The images under *root* and their class indices, from a tree laid out one folder per class.

    A label is its folder's position among the sorted folder names, which is what
    `keras.utils.image_dataset_from_directory` and timm both do over the same tree -- so a class
    keeps its index whichever framework reads the directory. Each class folder is read recursively
    and the files are sorted, so the listing of a tree is the same list every run: it is what the
    shuffle in `TFDataLoader.dataset` permutes, and an order that varied by itself would make a
    seeded run unreplayable.

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

    crop_pct: Annotated[float, Field(gt=0.0, le=1.0)] | None = None
    """Evaluation crop ratio, and the switch between the two transforms; None keeps the small one.

    Set, the shortest edge is resized to `image_size / crop_pct` and the centre `image_size` square
    is cut out of it while evaluating, and `scale`, `ratio` and `color_jitter` describe the training
    half -- the transform `examples/torch/data.py` gets from timm. Left at None, none of those three
    is read and the transform this pipeline has always had runs instead: a square resize to
    `image_size / 0.875` and then a random crop while training or a central one while evaluating.
    That one is what a CIFAR-sized set wants; a random resized crop of a 32-pixel image draws
    windows of nine pixels a side and upsamples them, which is not an augmentation of anything.

    Bounded to (0, 1]: it divides `image_size`, and a ratio above one would ask the crop for more
    pixels than the resize produced.
    """

    scale: tuple[float, float] = (0.08, 1.0)
    """Area fraction of the source image a training crop covers, drawn per item. Needs `crop_pct`."""

    ratio: tuple[float, float] = (3.0 / 4.0, 4.0 / 3.0)
    """Aspect ratio range of a training crop, drawn per item. Needs `crop_pct`."""

    color_jitter: float = 0.4
    """Brightness, contrast and saturation jitter of a training image; 0 turns it off.

    The scalar `torchvision.transforms.ColorJitter` and timm read the same way: each of the three
    factors is drawn uniformly from [1 - this, 1 + this], and hue is left alone. Needs `crop_pct`.
    """

    mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    """Per-channel mean subtracted after scaling the image to [0, 1]."""

    std: tuple[float, float, float] = (0.229, 0.224, 0.225)
    """Per-channel standard deviation divided out after the mean is subtracted."""

    image_dtype: Literal["float32", "float16", "bfloat16"] = "float32"
    """Element type of the emitted images."""

    hflip: bool = True
    """Whether the training augmentation flips images horizontally."""

    shuffle_buffer: int = 1024
    """Items held for shuffling a `tensorflow_datasets` training split, capped by the split's size.

    It does not bound a directory: there the shuffle runs over the file list, before the decode, so
    the whole listing is permuted every epoch and the buffer holds paths rather than images. A tfds
    set arrives decoded, where a buffer of the whole split is what nothing of ImageNet's size can
    afford, so that one stays bounded and its shuffle stays local to a window of the file order.
    """

    seed: int = 42
    """Seed of the shuffle and of the per-item augmentation draws, so a run is reproducible."""

    num_parallel_calls: int | None = None
    """Items decoded and preprocessed at once; None is `tf.data.AUTOTUNE`.

    The read is inside the decode map -- `tf.io.read_file` is the first op of `_decode` -- so this
    one number is both the file-reading and the augmentation budget, the `num_workers` of the timm
    example. AUTOTUNE sizes it from what the pipeline is already achieving, which on a host shared
    with a busy JAX process settles below what the CPU has; a launch configuration that knows the
    machine hands it the count instead.
    """

    prefetch: int | None = None
    """Batches held ready ahead of the device; None is `tf.data.AUTOTUNE`."""

    drop_remainder: bool | None = None
    """Whether the final short batch is dropped; None drops it on the training split only.

    A training batch is split across the strategy's mesh, so a short one would not divide, and one
    batch in an epoch of thousands is no loss. An evaluation epoch is a measurement, so the tail is
    kept instead: dropping it reports a metric over 49 664 of ImageNet's 50 000 validation images,
    which is not the number the torch example prints for the same model. The tail is one extra batch
    shape, so a jitted evaluation step traces a second time for it -- the cheaper half of the choice,
    the other being to pad the batch and carry a mask through the learner. It still has to divide the
    data axis of the mesh; where it does not, the strategy says so and `true` here is the way out.

    Coverage, not weighting: the tail is scored, and the trainer averages per batch, so its items
    carry more weight each than an item of a whole batch. That is what the torch example does too --
    timm keeps the tail and the torch tracker is a mean of batch means as well -- so the three
    frameworks weight it identically and a comparison between them is unaffected.
    """

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
        """Length an image is resized to before a crop cuts `image_size` out of it.

        The shortest edge on the ImageNet path, both edges of the square the small-image path
        resizes to. The ImageNet path floors, which is how `timm.data.transforms_factory`
        computes the same number, so the two examples resize an image to the same pixel count
        before cropping it. Without `crop_pct` the small-image path keeps rounding the ratio it
        always resized by, so leaving the knob unset changes nothing about that transform.
        """
        if self.crop_pct is None:
            return round(self.image_size / DEFAULT_CROP_PCT)
        return floor(self.image_size / self.crop_pct)

    @property
    def drops_remainder(self) -> bool:
        """Whether the final short batch is dropped: `drop_remainder`, or the training split alone."""
        return self.is_training if self.drop_remainder is None else self.drop_remainder

    @cached_property
    def source(self) -> tf.data.Dataset:
        """The unbatched, unshuffled items of this split: `(path, label)` pairs, or tfds' own pairs.

        A directory is listed once by `list_labelled_images` and handed on as paths rather than
        pixels, so a set of ImageNet's size costs a list of strings here and `_decode` reads a file
        only once the shuffle in `dataset` has picked it. That order is the whole point: a tree laid
        out one folder per class is listed class by class, so the file list is the only place where
        shuffling it globally is affordable -- a shuffle after the decode would have to hold images
        to reorder them, and a thousand of them is 0.08% of ImageNet, which leaves every batch two
        or three adjacent classes (the flax run of H200 tier 2 10-f: `ce_loss` was NaN from the
        first epoch on the full tree and clean on a ten-class subset of it).

        The listing is what `num_examples` counts. A tfds name is loaded through `tfds.load`
        instead, which hands over decoded uint8 images and int64 labels where a directory hands over
        a path and an int32 label; `_decode` and `_preprocess` are what make the batch contract
        identical either way. A file is decoded at `resize_size`, not at `image_size`: the crop that
        `crop_pct` exists for happens in `_preprocess`, and an item that had already been resized to
        the final size would leave it nothing to crop.

        Raises:
            ImportError: If `tensorflow_datasets` is not installed and a tfds name was asked for.
        """
        if isinstance(self.name, Path):
            return tf.data.Dataset.from_tensor_slices(list_labelled_images(self.name))
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

    @tf.autograph.experimental.do_not_convert
    def _decode(self, path: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        """Read one listed file, resizing it to a square unless a crop is going to read the source.

        On the small-image path this is what `image_dataset_from_directory` did. On the ImageNet path
        nothing is resized: `_preprocess` cuts its crop out of the source image, as the torch
        example's `RandomResizedCrop` does, and a file already squashed to one square size would
        leave that crop no scale and no aspect ratio to draw. The items therefore stay ragged until
        `_preprocess` makes each of them `image_size` square, which is still before anything batches
        them.

        Args:
            path (tf.Tensor): The image file to read.
            label (tf.Tensor): Its class index, carried through untouched so the pair stays paired.

        Returns:
            tuple[tf.Tensor, tf.Tensor]: The image, at `resize_size` or at its stored size, and its
                label.
        """
        size = self.resize_size
        image = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
        return (image if self.crop_pct is not None else tf.image.resize(image, (size, size))), label

    @tf.autograph.experimental.do_not_convert
    def _random_resized_crop(self, image: tf.Tensor, seed: tf.Tensor) -> tf.Tensor:
        """Cut a window of random area and aspect ratio out of *image* and resize it to `image_size`.

        `RandomResizedCropAndInterpolation.get_params`, which is what timm gives
        `examples/torch/data.py`, drawn from the same two distributions: the area uniformly over
        `scale` times the source area, and the *logarithm* of the aspect ratio uniformly over
        `ratio`, ten times, taking the first draw that fits inside the image; the offset is then
        uniform over the room that draw leaves. After ten misses it falls back to the aspect-clamped
        centre crop torchvision falls back to, not to the whole image.

        Not `tf.image.stateless_sample_distorted_bounding_box`, which is the obvious op and the
        wrong one: its kernel draws the crop *height* uniformly and the aspect ratio uniformly, so
        its area density goes as one over the square root of the area and it crops measurably
        harder -- a mean area of 0.354 of the source against torchvision's 0.435 over this scale
        range, which is a stronger augmentation than the torch run this is compared against.

        The ten draws are made at once rather than in a `tf.while_loop`: the loop would only save
        arithmetic on a tensor of ten floats, and each attempt is independent, so taking the first
        that fits is the same rejection sampler written without a loop to trace.

        Args:
            image (tf.Tensor): The float32 image in 0..255, at its stored size.
            seed (tf.Tensor): The two-element stateless seed of this item's draws.

        Returns:
            tf.Tensor: The `image_size` square crop.
        """
        shape = tf.cast(tf.shape(image)[:2], tf.float32)
        height, width = shape[0], shape[1]
        low, high = min(self.ratio), max(self.ratio)
        area = height * width * tf.random.stateless_uniform((CROP_ATTEMPTS,), seed, *self.scale)
        aspect = tf.exp(tf.random.stateless_uniform((CROP_ATTEMPTS,), seed + 1, log(low), log(high)))
        drawn = tf.round(tf.stack([tf.sqrt(area / aspect), tf.sqrt(area * aspect)]))
        fits = tf.reduce_all((drawn > 0.0) & (drawn <= shape[:, None]), axis=0)
        # The aspect-clamped centre crop of the fallback: the widest window of an allowed aspect
        # ratio that the image holds, which is the whole image when its own ratio is already one.
        inner = width / height
        clamped = tf.where(
            inner < low,
            tf.stack([tf.round(width / low), width]),
            tf.where(inner > high, tf.stack([height, tf.round(height * high)]), shape),
        )
        found = tf.reduce_any(fits)
        size = tf.where(found, drawn[:, tf.argmax(tf.cast(fits, tf.int32), output_type=tf.int32)], clamped)
        room = shape - size
        offset = tf.where(
            found, tf.floor(tf.random.stateless_uniform((2,), seed + 2) * (room + 1.0)), tf.floor(room / 2.0)
        )
        window = tf.slice(
            image, tf.concat([tf.cast(offset, tf.int32), [0]], axis=0), tf.concat([tf.cast(size, tf.int32), [3]], 0)
        )
        resized = tf.image.resize(window, (self.image_size, self.image_size), method="bicubic", antialias=True)
        # Bicubic overshoots at an edge; PIL clips it away in the uint8 it resizes into, and a
        # normalization fed the overshoot would put pixels outside the range the torch run trains on.
        return tf.clip_by_value(resized, 0.0, 255.0)

    @tf.autograph.experimental.do_not_convert
    def _resize_then_crop(self, image: tf.Tensor, seed: tf.Tensor) -> tf.Tensor:
        """Resize to a square and crop `image_size` out of it: the transform without `crop_pct`.

        What this pipeline did before `crop_pct` chose between two transforms, unchanged: a square
        resize -- the aspect ratio goes, which is what makes it the small-image recipe rather than
        the ImageNet one -- then a random crop and a flip while training, a central crop otherwise.

        Args:
            image (tf.Tensor): The float32 image in 0..255.
            seed (tf.Tensor): The two-element stateless seed of this item's draws.

        Returns:
            tf.Tensor: The `image_size` square crop.
        """
        image = tf.image.resize(image, (self.resize_size, self.resize_size), method="bicubic")
        if not self.is_training:
            return tf.image.resize_with_crop_or_pad(image, self.image_size, self.image_size)
        image = tf.image.stateless_random_crop(image, (self.image_size, self.image_size, 3), seed=seed)
        # A seed of its own, so the flip is not perfectly correlated with the crop offset.
        return tf.image.stateless_random_flip_left_right(image, seed=seed + 1) if self.hflip else image

    @tf.autograph.experimental.do_not_convert
    def _jitter(self, image: tf.Tensor, seed: tf.Tensor) -> tf.Tensor:
        """Scale brightness, contrast and saturation by three factors drawn around 1.

        `torchvision.transforms.ColorJitter(f, f, f)`, which is what a scalar `color_jitter` asks
        timm for: each factor is uniform in [1 - f, 1 + f] and blends the image towards black,
        towards its own mean grey and towards its greyscale, clipped back into 0..255 after each.
        The three are applied in a fixed order where torchvision draws a permutation of them; the
        factors and the strength of the augmentation are the same, and a fixed order is one random
        draw fewer to keep in step between two frameworks. The other deviation is that torchvision
        blends in uint8 on a PIL image and quantizes after each of the three, where this stays in
        float32 throughout: about 0.6 to 1.1 of 255 in mean absolute pixel difference, which is
        below the rounding of the decode itself.

        Args:
            image (tf.Tensor): The float32 image in 0..255.
            seed (tf.Tensor): The two-element stateless seed of this item's draws.

        Returns:
            tf.Tensor: The jittered image, still in 0..255.
        """
        low, high = max(0.0, 1.0 - self.color_jitter), 1.0 + self.color_jitter
        factors = tf.random.stateless_uniform((3,), seed=seed, minval=low, maxval=high)
        image = tf.clip_by_value(image * factors[0], 0.0, 255.0)
        grey = tf.reduce_mean(tf.image.rgb_to_grayscale(image))
        image = tf.clip_by_value((image - grey) * factors[1] + grey, 0.0, 255.0)
        greyscale = tf.image.rgb_to_grayscale(image)
        return tf.clip_by_value((image - greyscale) * factors[2] + greyscale, 0.0, 255.0)

    @tf.autograph.experimental.do_not_convert
    def _central_crop(self, image: tf.Tensor) -> tf.Tensor:
        """Resize the shortest edge to `resize_size` and cut the centre `image_size` square out.

        The evaluation transform of `examples/torch/data.py`: `Resize` on the shortest edge with the
        aspect ratio kept, then `CenterCrop`. The long edge is computed in integers, the way
        torchvision truncates it, so no float rounding can leave an edge a pixel short of the crop,
        and the crop offset is rounded the way `CenterCrop` rounds it.

        Args:
            image (tf.Tensor): The float32 image in 0..255, at its stored size.

        Returns:
            tf.Tensor: The `image_size` square centre crop.
        """
        shape = tf.shape(image)[:2]
        long = self.resize_size * tf.reduce_max(shape) // tf.reduce_min(shape)
        short_first = tf.stack([self.resize_size, long])
        resized = tf.where(shape[0] < shape[1], short_first, short_first[::-1])
        image = tf.clip_by_value(tf.image.resize(image, resized, method="bicubic", antialias=True), 0.0, 255.0)
        # Rounded half to even, which is what `CenterCrop` gets from Python's own `round`: flooring
        # instead shifts the window a pixel on the aspect ratios whose margin is an odd number of
        # them, and a pixel of shift is a residual of tens of levels against the torch example.
        offset = tf.cast(tf.round((tf.cast(resized, tf.float32) - self.image_size) / 2.0), tf.int32)
        return tf.slice(image, tf.concat([offset, [0]], axis=0), (self.image_size, self.image_size, 3))

    # AutoGraph has nothing to rewrite here -- every branch below is decided while the graph is
    # traced, and the rest is tf ops -- and it fails to introspect a method of a pydantic model.
    @tf.autograph.experimental.do_not_convert
    def _preprocess(self, index: tf.Tensor, item: tuple[tf.Tensor, tf.Tensor]) -> tuple[tf.Tensor, tf.Tensor]:
        """Crop, flip, jitter and normalize one image, in the order the torch example applies them.

        Which crop is `crop_pct`'s to decide: without it the small-image transform below, with it
        the ImageNet one the torch example trains under.

        The random draws are stateless and keyed by *index*, so the augmentation of an item is a
        function of its position in the (shuffled) stream rather than of a global RNG the graph
        would carry between workers. Each stage takes a seed of its own, so the flip is not
        perfectly correlated with the crop it follows, nor the jitter with either.

        Args:
            index (tf.Tensor): Position of the item in the stream, which seeds the random draws.
            item (tuple[tf.Tensor, tf.Tensor]): The image at its stored size and its label.

        Returns:
            tuple[tf.Tensor, tf.Tensor]: The preprocessed image and its label.
        """
        image, label = item
        seed = tf.stack([tf.cast(index, tf.int32) + self.seed, self.seed])
        image = tf.cast(image, tf.float32)
        if self.crop_pct is None:
            image = self._resize_then_crop(image, seed)
        elif self.is_training:
            image = self._random_resized_crop(image, seed)
            if self.hflip:
                image = tf.image.stateless_random_flip_left_right(image, seed=seed + 3)
            if self.color_jitter:
                image = self._jitter(image, seed + 4)
        else:
            image = self._central_crop(image)
        image = (image / 255.0 - tf.constant(self.mean)) / tf.constant(self.std)
        return tf.cast(image, DTYPES[self.image_dtype]), tf.cast(label, tf.int32)

    @cached_property
    def dataset(self) -> tf.data.Dataset:
        """The batched pipeline: shuffle (training only), decode, preprocess, batch, prefetch.

        A directory is shuffled whole and before its files are read, which is what mixes a
        class-sorted tree; `reshuffle_each_iteration` then makes each epoch a fresh permutation of
        the same seeded stream, so an item's augmentation -- keyed by its position in this stream --
        varies between epochs as well. A tfds set is already decoded by the time it arrives here,
        so it keeps the bounded `shuffle_buffer` it always had.
        """
        dataset = self.source
        streaming = isinstance(self.name, Path)
        calls = tf.data.AUTOTUNE if self.num_parallel_calls is None else self.num_parallel_calls
        if self.is_training:
            buffer = self.num_examples if streaming else min(self.shuffle_buffer, self.num_examples)
            dataset = dataset.shuffle(buffer, seed=self.seed, reshuffle_each_iteration=True)
        if streaming:
            dataset = dataset.map(self._decode, num_parallel_calls=calls)
        return (
            dataset.enumerate()
            .map(self._preprocess, num_parallel_calls=calls)
            .batch(self.batch_size, drop_remainder=self.drops_remainder)
            .prefetch(tf.data.AUTOTUNE if self.prefetch is None else self.prefetch)
        )

    def __len__(self) -> int:
        """Number of batches one epoch yields, the short final one included unless it is dropped."""
        whole, remainder = divmod(self.num_examples, self.batch_size)
        return whole if self.drops_remainder else whole + bool(remainder)

    def __call__(self) -> Iterator[dict[str, Any]]:
        """Yield one epoch of batches as NumPy arrays, keyed by the learner input names of `spec`."""
        for batch in self.dataset.as_numpy_iterator():
            yield {name: batch[index] for name, index in self.spec.items()}
