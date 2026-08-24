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

There are two augmentation recipes, and `crop_pct` picks between them. Left at None it is the
small-image one this file has always had: a horizontal flip and a pad-then-crop through Keras
preprocessing layers, images scaled to 0..1, which is what a CIFAR-sized set trains under. Set, it
is the ImageNet one `examples/torch/data.py` gets from timm, rebuilt on `tf.data` operations: a
random resized crop drawn from the same area and log-aspect distributions, a horizontal flip and
brightness/contrast/saturation jitter while training, a shortest-edge resize and a central crop of
the same geometry while evaluating, ImageNet channel statistics either way. The three residuals are
named where they are made: the resize is TensorFlow's bicubic rather than PIL's, the jitter is
applied in a fixed order rather than a drawn one, and it blends in float32 rather than in uint8.
That is the recipe a run reproducing a published ImageNet number wants.

The batch keys are `image_key` and `label_key`, which is where a learner whose inputs are named
differently is served. A `structcast` `FlexSpec` would be the other way to remap them, and it is
deliberately not offered here: it compares each constructed value against a sentinel, which raises
on a NumPy array.
"""

from collections.abc import Iterator
from functools import cached_property
from math import floor, log
import os
from pathlib import Path
from typing import Annotated, Any, Literal

import numpy as np
from pydantic import BaseModel, DirectoryPath, Field
import tensorflow as tf

import keras

CROP_ATTEMPTS = 10
"""Draws a random resized crop makes before it falls back to a centre crop, as torchvision does."""

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
    """Items per batch.

    The training split drops its final short batch, so every batch it yields has this many items; an
    evaluation epoch keeps the tail instead, because it is a measurement of the whole split.
    """

    image_size: tuple[int, int] = (32, 32)
    """The (height, width) the model sees.

    A directory's images are decoded straight to this size. An in-memory image larger than this is
    randomly cropped to it when training and resized to it when not, the usual pair.
    """

    crop_padding: int = 4
    """Pixels of zero padding added on each side before the random crop; 0 disables the crop.

    Pad-then-crop is the small-image recipe, and it is what `crop_pct: null` selects. It is not the
    scale-and-aspect jitter an ImageNet run trains under -- there a directory's images would be
    resized to `image_size` outright, aspect ratio and all -- so a run reproducing a published
    ImageNet number sets `crop_pct` instead, which replaces this whole path with the one the torch
    example uses. Unused on that path.
    """

    crop_pct: Annotated[float, Field(gt=0.0, le=1.0)] | None = None
    """Evaluation crop ratio, and the switch between the two recipes; None keeps the small-image one.

    Set, the shortest edge is resized to `image_size / crop_pct` and the centre `image_size` window
    is cut out of it while evaluating, and `scale`, `ratio`, `color_jitter`, `mean` and `std` describe
    the training half -- the transform `examples/torch/data.py` builds through timm. Left at None,
    none of those five is read and the pad-then-crop layers below run instead, which is what keeps
    the augmentation of a CIFAR-sized run of this file unchanged. The one thing that changes on that
    path too is the tail: an evaluation epoch now yields its short final batch on both recipes.

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
    """Per-channel mean subtracted after scaling to 0..1; timm's `IMAGENET_DEFAULT_MEAN`.

    Needs `crop_pct`: the small-image path scales to 0..1 and stops there, as it always has.
    """

    std: tuple[float, float, float] = (0.229, 0.224, 0.225)
    """Per-channel standard deviation divided out after the mean; timm's `IMAGENET_DEFAULT_STD`."""

    hflip: bool = True
    """Whether the training augmentation flips images horizontally, on either recipe."""

    shuffle_buffer: int = 1024
    """Items the training shuffle of a `keras.datasets` set holds at once, capped by this rank's share.

    A full-split buffer is what a shuffle wants and what an in-memory set cannot have for free: the
    buffer would be a second copy of the array. It does not bound a directory -- there the shuffle
    runs over the file list, before the decode, so this rank's whole share is permuted every epoch
    and the buffer holds paths rather than images.
    """

    seed: int = 42
    """Seed of the shuffle and of the augmentation draws."""

    num_parallel_calls: int | None = None
    """Items decoded and preprocessed at once; None is `tf.data.AUTOTUNE`.

    The read is inside the decode map -- `tf.io.read_file` is the first op of `_decode` -- so this
    one number is both the file-reading and the augmentation budget, the `num_workers` of the timm
    example. AUTOTUNE sizes it from what the pipeline is already achieving, which on a host shared
    with a busy training process settles below what the CPU has; a launch configuration that knows
    the machine hands it the count instead.
    """

    prefetch: int | None = None
    """Batches held ready ahead of the device; None is `tf.data.AUTOTUNE`."""

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

    @property
    def resize_size(self) -> int:
        """Length the shortest edge is resized to before the evaluation crop cuts `image_size`.

        Floored rather than rounded, which is how `timm.data.transforms_factory` computes the same
        number. Taken off the longer of the two target edges so that both survive the crop; for the
        square `image_size` an ImageNet run asks for, the two are the same number and this is exactly
        what the torch example resizes to. Without `crop_pct` there is no resize-then-crop at all --
        the small-image path resizes straight to `image_size` -- which is the number reported then.
        """
        return floor(max(self.image_size) / self.crop_pct) if self.crop_pct is not None else max(self.image_size)

    def _random_resized_crop(self, image: Any, seed: Any) -> Any:
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
            image (Any): The float32 image in 0..255, at its stored size.
            seed (Any): The two-element stateless seed of this item's draws.

        Returns:
            Any: The `image_size` crop.
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
        resized = tf.image.resize(window, self.image_size, method="bicubic", antialias=True)
        # Bicubic overshoots at an edge; PIL clips it away in the uint8 it resizes into, and a
        # normalization fed the overshoot would put pixels outside the range the torch run trains on.
        return tf.clip_by_value(resized, 0.0, 255.0)

    def _jitter(self, image: Any, seed: Any) -> Any:
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
            image (Any): The float32 image in 0..255.
            seed (Any): The two-element stateless seed of this item's draws.

        Returns:
            Any: The jittered image, still in 0..255.
        """
        low, high = max(0.0, 1.0 - self.color_jitter), 1.0 + self.color_jitter
        factors = tf.random.stateless_uniform((3,), seed=seed, minval=low, maxval=high)
        image = tf.clip_by_value(image * factors[0], 0.0, 255.0)
        grey = tf.reduce_mean(tf.image.rgb_to_grayscale(image))
        image = tf.clip_by_value((image - grey) * factors[1] + grey, 0.0, 255.0)
        greyscale = tf.image.rgb_to_grayscale(image)
        return tf.clip_by_value((image - greyscale) * factors[2] + greyscale, 0.0, 255.0)

    def _central_crop(self, image: Any) -> Any:
        """Resize the shortest edge to `resize_size` and cut the centre `image_size` window out.

        The evaluation transform of `examples/torch/data.py`: `Resize` on the shortest edge with the
        aspect ratio kept, then `CenterCrop`. The long edge is computed in integers, the way
        torchvision truncates it, so no float rounding can leave an edge a pixel short of the crop,
        and the crop offset is rounded the way `CenterCrop` rounds it.

        Args:
            image (Any): The float32 image in 0..255, at its stored size.

        Returns:
            Any: The `image_size` centre crop.
        """
        shape = tf.shape(image)[:2]
        long = self.resize_size * tf.reduce_max(shape) // tf.reduce_min(shape)
        short_first = tf.stack([self.resize_size, long])
        resized = tf.where(shape[0] < shape[1], short_first, short_first[::-1])
        image = tf.clip_by_value(tf.image.resize(image, resized, method="bicubic", antialias=True), 0.0, 255.0)
        # Rounded half to even, which is what `CenterCrop` gets from Python's own `round`: flooring
        # instead shifts the window a pixel on the aspect ratios whose margin is an odd number of
        # them, and a pixel of shift is a residual of tens of levels against the torch example.
        target = tf.constant(self.image_size, tf.float32)
        offset = tf.cast(tf.round((tf.cast(resized, tf.float32) - target) / 2.0), tf.int32)
        return tf.slice(image, tf.concat([offset, [0]], axis=0), (*self.image_size, 3))

    @tf.autograph.experimental.do_not_convert
    def _transform(self, index: Any, item: tuple[Any, Any]) -> tuple[Any, Any]:
        """Crop, flip, jitter and normalize one image, in the order the torch example applies them.

        The ImageNet path only, and per item rather than per batch: a random resized crop reads the
        source image, which is one shape per file, so it has to run before anything batches them.
        The draws are stateless and keyed by *index*, so an item's augmentation follows its position
        in the (shuffled) stream rather than a global RNG, and each stage takes a seed of its own so
        the flip is not perfectly correlated with the crop it follows.

        Args:
            index (Any): Position of the item in the stream, which seeds the random draws.
            item (tuple[Any, Any]): The image at its stored size and its label.

        Returns:
            tuple[Any, Any]: The preprocessed image and its label.
        """
        image, label = item
        seed = tf.stack([tf.cast(index, tf.int32) + self.seed, self.seed])
        image = tf.cast(image, tf.float32)
        if self.training:
            image = self._random_resized_crop(image, seed)
            if self.hflip:
                image = tf.image.stateless_random_flip_left_right(image, seed=seed + 3)
            if self.color_jitter:
                image = self._jitter(image, seed + 4)
        else:
            image = self._central_crop(image)
        return (image / 255.0 - tf.constant(self.mean)) / tf.constant(self.std), label

    @cached_property
    def augmentation(self) -> list[keras.layers.Layer]:
        """The preprocessing layers applied to one batch, in order; empty on the ImageNet path.

        Applied one by one rather than through a `keras.Sequential`: these run inside a `tf.data`
        graph, where a preprocessing layer switches to TensorFlow operations by itself but the model
        container around it would not.

        The small-image path only. With `crop_pct` set, `_transform` has already cropped, flipped,
        jittered and normalized each item before it was batched, so a layer here would augment an
        augmented image and `Rescaling` would divide an already normalized one by 255 again.
        """
        layers: list[keras.layers.Layer] = []
        if self.crop_pct is not None:
            return layers
        if self.training:
            if self.hflip:
                layers.append(keras.layers.RandomFlip("horizontal", seed=self.seed))
            if self.crop_padding:
                layers.append(keras.layers.RandomCrop(*self.image_size, seed=self.seed))
        layers.append(keras.layers.Resizing(*self.image_size))
        layers.append(keras.layers.Rescaling(scale=1.0 / 255))
        return layers

    def _decode(self, path: Any, label: Any) -> tuple[Any, Any]:
        """Read one listed file, resizing it to `image_size` unless a crop is going to read it.

        On the small-image path this is what `image_dataset_from_directory` did, and the resize is
        what makes the items one shape for the batching. On the ImageNet path `_transform` cuts its
        crop out of the source image, as the torch example's `RandomResizedCrop` does, and a file
        already squashed to one size would leave that crop no scale and no aspect ratio to draw --
        so the items stay ragged until `_transform` makes each of them `image_size`, which is still
        before anything batches them.

        Args:
            path (Any): The image file to read, as a scalar string tensor.
            label (Any): Its class index, carried through untouched so the pair stays paired.

        Returns:
            tuple[Any, Any]: The image, at `image_size` or at its stored size, and its label.
        """
        image = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
        return image if self.crop_pct is not None else tf.image.resize(image, self.image_size), label

    def _prepare(self, images: Any, labels: Any) -> dict[str, Any]:
        """Augment one batch of images and key it by the model's input names.

        The labels are cast because the two sources disagree on their width; the images need no
        cast, since `Rescaling` -- or `_transform` on the ImageNet path -- ends the chain in float32
        either way. Both sources therefore leave one batch contract, which is what lets a run swap a
        small set for a directory unchanged.
        """
        if self.crop_pct is None and self.training and self.crop_padding:
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
        does on the torch side; every rank therefore owns the same number of items, so the short
        final batch an evaluation epoch keeps is the same size on all of them. Which items a rank
        owns is fixed for the whole run and only their order is reshuffled each epoch -- the same
        thing `DistributedSampler` does when nobody calls `set_epoch`, and what the torch example
        accepts too. `examples/keras/corpus.py` shards after
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
        calls = tf.data.AUTOTUNE if self.num_parallel_calls is None else self.num_parallel_calls
        if world > 1:
            data = data.take(self.items - self.items % world).shard(world, rank)
        if self.training:
            buffer = self.shard_items if streaming else min(self.shard_items, self.shuffle_buffer)
            data = data.shuffle(buffer, seed=self.seed, reshuffle_each_iteration=True)
        if streaming:
            data = data.map(self._decode, num_parallel_calls=calls)
        if self.crop_pct is not None:
            data = data.enumerate().map(self._transform, num_parallel_calls=calls)
        data = data.batch(self.batch_size, drop_remainder=self.training)
        return data.map(self._prepare, num_parallel_calls=calls).prefetch(
            tf.data.AUTOTUNE if self.prefetch is None else self.prefetch
        )

    def __len__(self) -> int:
        """Number of batches one rank sees per epoch; the short final one counts unless training.

        A training batch is dropped when it is short -- the distributed strategies split a batch
        across their replicas, so a short one would not divide, and one batch in an epoch of
        thousands is no loss. An evaluation epoch keeps its tail instead: dropping it reports a
        metric over 49 664 of ImageNet's 50 000 validation images, which is not what the torch
        example prints for the same model. The tail is one extra batch shape, so a jitted evaluation
        step traces a second time for it -- the cheaper half of the choice, the other being to pad
        the batch and carry a mask through the learner.

        Coverage, not weighting: the tail is scored, and the trainer averages per batch, so its
        items carry more weight each than an item of a whole batch. That is what the torch example
        does too -- timm keeps the tail and the torch tracker is a mean of batch means as well -- so
        the three frameworks weight it identically and a comparison between them is unaffected.
        """
        whole, remainder = divmod(self.shard_items, self.batch_size)
        return whole if self.training else whole + bool(remainder)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        """The batches of one epoch, as NumPy arrays keyed by `image_key` and `label_key`."""
        yield from self.pipeline.as_numpy_iterator()
