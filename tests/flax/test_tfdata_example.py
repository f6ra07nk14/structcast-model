"""Unit tests for the tf.data input pipeline example in examples/flax/data.py.

The loader takes its items from one `cached_property`, `source`, so every case here writes a small
`tf.data.Dataset` of tensor slices into the instance dictionary and leaves `tensorflow_datasets`
alone -- the same seam `tests/torch/test_timm_data_example.py` uses on the timm wrapper. Nothing is
downloaded, and the pipeline under test is the real one, all the way from the raw uint8 pairs.
"""

from __future__ import annotations

import importlib.util
from importlib.util import find_spec
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from structcast_model.base_trainer import SimpleDataProvider

# Before the module is loaded: importing the example imports TensorFlow, which the flax floor
# environment does not install. The pipeline is an example integration rather than a floor concern,
# so skipping the file there is the right outcome -- and a collection-time ImportError would not be.
pytest.importorskip("tensorflow")

import tensorflow as tf  # noqa: E402  # only importable once the skip above has passed


def _load_example_module() -> Any:
    """Load the example module the way a configuration does: by file path, not by import name."""
    path = Path(__file__).resolve().parents[2] / "examples" / "flax" / "data.py"
    spec = importlib.util.spec_from_file_location("example_flax_data", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_EXAMPLE = _load_example_module()
TFDataLoader = _EXAMPLE.TFDataLoader

SMALL: dict[str, Any] = {"image_size": 16, "batch_size": 4}
"""Sixteen-pixel images in batches of four: three whole batches out of the twelve items below."""

ITEMS = 12
"""Items the in-memory source holds."""


def _loader(**kwargs: Any) -> Any:
    """Build a loader whose `source` is twelve fixed images already in memory.

    Writing the instance dictionary is what replaces a `cached_property` before it is first read,
    so `tfds.load` is never reached and the run needs neither a download nor that package.
    """
    loader = TFDataLoader(name="tests/in-memory", **{**SMALL, **kwargs})
    images = tf.random.stateless_uniform((ITEMS, 32, 32, 3), seed=(1, 2), maxval=256, dtype=tf.int32)
    labels = tf.range(ITEMS, dtype=tf.int32) % 3
    loader.__dict__["source"] = tf.data.Dataset.from_tensor_slices((tf.cast(images, tf.uint8), labels))
    return loader


def test_batches_are_keyed_by_the_learner_input_names() -> None:
    """The trainer hands a batch to the learner as keyword arguments, so a bare pair never arrives.

    `spec` is what turns the `(image, label)` pair the pipeline produces into those names, and the
    strategy places a batch by iterating its entries -- so a tuple would not even reach the mesh.
    """
    batch = next(iter(_loader()()))

    assert sorted(batch) == ["image", "label"]
    assert batch["image"].shape == (4, 16, 16, 3)
    assert batch["label"].shape == (4,)
    assert batch["image"].dtype == np.float32
    assert isinstance(batch["image"], np.ndarray)


def test_the_length_matches_the_epoch_it_yields() -> None:
    """`SimpleDataProvider` reports `steps_per_epoch` from `__len__` without iterating the data.

    The progress bar, the epoch boundary and the schedule step counts all read that number, so a
    `__len__` that disagreed with the epoch would silently mis-size every one of them.
    """
    loader = _loader()

    assert len(loader) == len(list(loader())) == 3
    assert SimpleDataProvider(training_dataset=loader).steps_per_epoch == 3


def test_the_epoch_count_refuses_a_split_that_reports_no_size() -> None:
    """A `steps_per_epoch` of -1 would size the whole loop wrong, so it has to fail loudly here."""
    loader = _loader()
    loader.__dict__["source"] = tf.data.Dataset.range(4).repeat()

    with pytest.raises(ValueError, match="reports no size"):
        len(loader)


def test_the_evaluation_split_is_a_deterministic_central_crop() -> None:
    """Validation must be the same measurement every epoch, so nothing on that path may be random.

    A random crop left switched on for the evaluation split would make the validation criterion
    drift between epochs for reasons that have nothing to do with the model.
    """
    loader = _loader()

    first, second = next(iter(loader()))["image"], next(iter(loader()))["image"]

    assert np.array_equal(first, second)


def test_the_training_split_augments_but_stays_reproducible_from_its_seed() -> None:
    """Augmentation must actually vary between epochs, and a seed must still replay a whole run.

    The draws are stateless and keyed by the item's position in the shuffled stream: that is what
    gives both properties at once, where a global RNG would give the first and lose the second.
    """
    loader = _loader(is_training=True)

    first, second = next(iter(loader()))["image"], next(iter(loader()))["image"]

    assert first.shape == (4, 16, 16, 3)
    assert not np.array_equal(first, second)
    assert np.array_equal(first, next(iter(_loader(is_training=True)()))["image"])
    assert not np.array_equal(first, next(iter(_loader(is_training=True, seed=7)()))["image"])


def test_the_crop_ratio_decides_how_much_of_the_image_survives() -> None:
    """`crop_pct` is the one knob relating the resize to the crop, so it must reach the resize.

    A pipeline that resized straight to `image_size` would crop nothing at all, which is a
    different augmentation from the one the configuration asks for.
    """
    assert TFDataLoader(name="cifar10", image_size=224, crop_pct=0.875).resize_size == 256
    assert TFDataLoader(name="cifar10", image_size=16, crop_pct=1.0).resize_size == 16


def _one_picture(loader: Any, picture: Any) -> Any:
    """Point *loader* at `ITEMS` copies of *picture*, so the shuffle cannot change what it reads."""
    images = tf.tile(picture[None], (ITEMS, 1, 1, 1))
    loader.__dict__["source"] = tf.data.Dataset.from_tensor_slices((images, tf.zeros((ITEMS,), tf.int32)))
    return loader


def test_the_pipeline_normalizes_with_the_channel_statistics_it_was_given() -> None:
    """A batch has to leave here on the scale `examples/torch/data.py` trains the same model on.

    Scaling to 0..1 and stopping there is what this catches: it survives every shape and dtype
    assertion, it still trains, and it is a different input distribution from the torch example's --
    so a cross-framework comparison would be measuring the pixels rather than the framework. A flat
    grey image makes the arithmetic readable: every channel has to come out at (128/255 - mean)/std.
    The defaults are pinned with it, because they are the other half of that agreement: they are
    `IMAGENET_DEFAULT_MEAN` and `IMAGENET_DEFAULT_STD`, which is what timm normalizes with.
    """
    loader = _one_picture(
        _loader(crop_pct=0.875, mean=(0.0, 0.25, 0.5), std=(1.0, 0.5, 0.25)),
        tf.fill((32, 32, 3), tf.constant(128, tf.uint8)),
    )

    image = next(iter(loader()))["image"]

    assert np.allclose(image, [(128 / 255), (128 / 255 - 0.25) * 2, (128 / 255 - 0.5) * 4], atol=1e-5)
    assert TFDataLoader(name="cifar10").mean == (0.485, 0.456, 0.406)
    assert TFDataLoader(name="cifar10").std == (0.229, 0.224, 0.225)


def test_the_training_crop_is_a_random_window_of_the_source_image() -> None:
    """The torch example trains on `RandomResizedCrop`; a fixed-offset crop of a squashed image is not it.

    Two draws say which one this is. A whole-area, square-aspect crop has to give back exactly the
    bicubic resize of the source, because that is what "the window is the whole image" means -- a
    pipeline that squashed the image to one square size first and then cropped could not, it would
    still be cutting a smaller square out. And a small-area draw has to give two items two different
    windows, where a fixed crop of an already-resized image would differ by an offset of a few
    pixels at most. The scale range is the augmentation's whole strength, so it has to be the crop's.
    """
    picture = tf.cast(tf.random.stateless_uniform((24, 24, 3), seed=(3, 4), maxval=256, dtype=tf.int32), tf.uint8)
    plain: Any = tf.image.resize(tf.cast(picture, tf.float32), (16, 16), method="bicubic", antialias=True)
    bare = {
        "is_training": True,
        "crop_pct": 0.875,
        "hflip": False,
        "color_jitter": 0.0,
        "mean": (0.0,) * 3,
        "std": (1.0,) * 3,
    }

    whole = next(iter(_one_picture(_loader(**bare, scale=(1.0, 1.0), ratio=(1.0, 1.0)), picture)()))["image"]
    windows = next(iter(_one_picture(_loader(**bare, scale=(0.08, 0.15)), picture)()))["image"]

    assert np.allclose(whole[0], plain.numpy() / 255.0, atol=1e-5)
    assert np.allclose(whole[0], whole[3], atol=1e-6)
    assert not np.allclose(windows[0], windows[1], atol=1e-3)


def test_the_evaluation_crop_is_the_window_torchvision_would_have_taken() -> None:
    """The evaluation window has to be the exact one, not one a pixel off it.

    `Resize` then `CenterCrop` looks like two roundings nobody can get wrong, and both of them can
    be: the long edge has to be truncated and the crop offset has to be rounded half to even, which
    is what `CenterCrop` gets from Python's own `round`. Flooring the offset instead shifts the
    window by one pixel whenever the margin is an odd number of them -- about a quarter of the
    aspect ratios in a real set -- and one pixel of shift on a gradient is a residual of tens of
    levels per channel against the torch example, which is more than the whole of bf16's error.

    A 9x11 source with `image_size=8` and `crop_pct=0.875` is the smallest case that separates the
    two: the shortest edge is already 9, so the margins are 1 and 3 and the correct offsets are
    (0, 2) where flooring gives (0, 1). The source is a gradient because a shift has to be visible.
    """
    picture = tf.cast(tf.reshape(tf.range(9 * 11 * 3), (9, 11, 3)) % 256, tf.uint8)
    bare: Any = {"image_size": 8, "crop_pct": 0.875, "mean": (0.0,) * 3, "std": (1.0,) * 3}
    resized = tf.image.resize(tf.cast(picture, tf.float32), (9, 11), method="bicubic", antialias=True)
    expected = tf.clip_by_value(resized, 0.0, 255.0)[0:8, 2:10].numpy() / 255.0

    window = next(iter(_one_picture(_loader(**bare), picture)()))["image"]

    assert np.allclose(window[0], expected, atol=1e-5)
    # And the whole geometry belongs to the ImageNet path: without `crop_pct` the small-image
    # transform squashes the image to a square first, which is a different window of a different
    # image, so the gate cannot be silently bypassing the recipe a run asked for.
    small = next(iter(_one_picture(_loader(image_size=8, mean=(0.0,) * 3, std=(1.0,) * 3), picture)()))["image"]
    assert not np.allclose(small[0], expected, atol=1e-2)


def test_an_evaluation_epoch_keeps_the_tail_a_training_epoch_drops() -> None:
    """The validation split is a measurement, so it has to be over all of it.

    The dropped tail is silent and small enough to look like noise: 50 000 ImageNet validation
    images in batches of 512 are 97 whole batches and 336 images nobody scored, against the 98 steps
    the torch example runs, so the two frameworks report a metric over different data. Training
    keeps dropping its tail -- a short batch does not divide the strategy's mesh -- and the knob
    still overrides both, which is the way out when a tail does not divide the data axis either.
    """
    validation, training = _loader(batch_size=5), _loader(batch_size=5, is_training=True)

    assert [len(batch["label"]) for batch in validation()] == [5, 5, 2]
    assert [len(batch["label"]) for batch in training()] == [5, 5]
    assert (len(validation), len(training)) == (3, 2)
    assert len(_loader(batch_size=5, drop_remainder=True)) == 2
    assert len(_loader(batch_size=5, is_training=True, drop_remainder=False)) == 3


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_the_image_element_type_is_the_one_that_was_asked_for(dtype: str) -> None:
    """A bfloat16 run wants its batches already narrowed, on the host, before they are placed."""
    batch = next(iter(_loader(image_dtype=dtype)()))

    assert batch["image"].dtype.name == dtype
    assert batch["label"].dtype == np.int32


@pytest.mark.skipif(find_spec("tensorflow_datasets") is not None, reason="tensorflow_datasets is installed")
def test_a_missing_tensorflow_datasets_says_what_to_install() -> None:
    """The one optional dependency of this file must name itself and its install command.

    It is imported at module scope through `try_import`, so the failure is deferred to the read
    that needs it -- and everything else in the file, this whole test module included, keeps
    working without it.
    """
    with pytest.raises(ImportError, match="uv pip install tensorflow-datasets"):
        _ = TFDataLoader(name="cifar10").source


def _image_tree(root: Path, classes: int = 2, per_class: int = 4) -> Path:
    """Write a class-per-folder tree of tiny PNGs, the layout the timm and keras loaders read too.

    The folders are written in class order and read back in it, which is the property every test
    below turns on: a real tree of ImageNet's shape is listed the same way. Every image is one flat
    colour; the label, not the colour, is what says here which file a batch item came from.
    """
    for label in range(classes):
        folder = root / f"class{label}"
        folder.mkdir(parents=True)
        for index in range(per_class):
            image = tf.fill((8, 8, 3), tf.constant((label * per_class + index) % 256, tf.uint8))
            tf.io.write_file(str(folder / f"{index}.png"), tf.io.encode_png(image))
    return root


@pytest.fixture(scope="module")
def sorted_tree(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A class-sorted tree big enough for its listing order to matter: 20 classes of 30 images.

    Module-scoped because it is written once and only ever read: six hundred files is enough that
    a batch of sixteen drawn from the raw listing is one class, and few enough to write in a second.
    """
    return _image_tree(tmp_path_factory.mktemp("sorted"), classes=20, per_class=30)


def _directory_loader(root: Path, **kwargs: Any) -> Any:
    """Build a loader over a class-per-folder tree, at the size the tree was written in."""
    return TFDataLoader(**{"name": root, "batch_size": 2, "image_size": 8, "crop_pct": 1.0, **kwargs})


def _labels(loader: Any) -> list[int]:
    """Every label one epoch of *loader* yields, in order."""
    return [int(label) for batch in loader() for label in batch["label"]]


def test_a_directory_is_told_apart_from_a_dataset_name_by_the_field_itself() -> None:
    """One field carries both sources, so the discrimination has to be the field's, not a sniff.

    A tfds name is any string, unlike the three-name `Literal` the keras twin can discriminate
    against, so the union is resolved left to right: an existing directory wins and everything else
    is a name. A pair of fields would let a configuration set both and silently honor one.

    There is also no default, so an empty name has to fail at construction rather than deep inside
    `tfds.load`, one training run and one model build later.
    """
    with pytest.raises(ValueError, match="A dataset name is required"):
        TFDataLoader(name="   ")

    assert isinstance(TFDataLoader(name="cifar10").name, str)


def test_a_directory_source_yields_the_same_batch_contract_as_a_dataset_name(tmp_path: Path) -> None:
    """A directory and a tfds name must be interchangeable from the learner's side.

    They are not interchangeable underneath -- a directory hands over float32 images in 0..255 where
    tfds hands over uint8 -- so the keys, the shapes and the dtypes are pinned here: a run that
    swapped a small set for the real one would otherwise only find out inside the loss.
    """
    data = _directory_loader(_image_tree(tmp_path))

    batch = next(iter(data()))

    assert sorted(batch) == ["image", "label"]
    assert batch["image"].shape == (2, 8, 8, 3)
    assert batch["image"].dtype == np.float32
    assert batch["label"].shape == (2,)
    assert batch["label"].dtype == np.int32
    assert (data.num_examples, len(data)) == (8, 4)


def test_a_directory_source_decodes_as_the_pipeline_pulls(tmp_path: Path) -> None:
    """The listing is what is read up front; the pixels are not, or ImageNet would not fit.

    The element spec is the proof: a scalar string per item, so the source is a list of paths and
    everything the pipeline puts in front of the decode -- the shuffle, above all -- reorders
    strings. A source that had read the tree into one array would report `(n, ...)` instead, and
    `num_examples` would be counting something already in memory.
    """
    data = _directory_loader(_image_tree(tmp_path))

    assert data.source.element_spec[0] == tf.TensorSpec(shape=(), dtype=tf.string)
    # The batch axis is unknown because an evaluation epoch keeps its short final batch; the pixel
    # axes are not, and they are what says the ragged decoded items were made one shape before this.
    assert tuple(data.dataset.element_spec[0].shape) == (None, 8, 8, 3)
    assert data.num_examples == 8


def test_a_directory_source_numbers_its_classes_by_the_sorted_folder_names(tmp_path: Path) -> None:
    """A rerun over the same tree has to give a class the same index, or a checkpoint means nothing.

    The label of an item is its folder's position in sorted order, which is also what the timm and
    keras loaders do over the same tree -- so a model trained through one can be evaluated by
    another without a relabelling table.
    """
    data = _directory_loader(_image_tree(tmp_path, classes=2, per_class=4))

    labels = [int(label) for batch in data() for label in batch["label"]]

    assert sorted(labels) == [0, 0, 0, 0, 1, 1, 1, 1]
    assert labels == [int(label) for batch in data() for label in batch["label"]]


def test_a_directory_source_shuffles_the_same_way_for_the_same_seed(tmp_path: Path) -> None:
    """A run has to be replayable, and the shuffle is the only thing here that could stop it being.

    Asserted on the labels, one per file here, because they are what the shuffle reorders and the
    augmentation cannot touch. Two pipelines built from one seed must agree epoch for epoch; two
    seeds must not, or the seed is ignored and every run is silently the file order.
    """
    root = _image_tree(tmp_path, classes=8, per_class=1)

    def _epoch(seed: int) -> list[int]:
        data = _directory_loader(root, is_training=True, seed=seed)
        return [int(label) for batch in data() for label in batch["label"]]

    assert _epoch(0) == _epoch(0)
    assert _epoch(0) != _epoch(7)
    assert sorted(_epoch(0)) == list(range(8))


def test_a_directory_batch_mixes_the_classes_of_a_class_sorted_tree(sorted_tree: Path) -> None:
    """A megascale tree is listed class by class, and a batch of one class is what went NaN.

    This is the flax half of H200 tier 2 run 10-f: ConvNeXtV2 over the 1.28M-image ImageNet train
    tree reported `ce_loss = nan` from the first epoch, while the same learner and the same
    thousand-class configuration trained clean on a ten-class subset of it. The measured label
    stream ran [0, 0] -> [6, 10] -> [15, 20] -> [36, 40] over four hundred batches of 128: every
    batch two to five adjacent classes, advancing monotonically, because the only shuffle was a
    thousand-item buffer over the decoded stream -- 0.08% of that tree.

    So the assertion is on the classes of a single batch, which is the thing the run measured.
    `shuffle_buffer=8` over six hundred files is what puts that run's proportions into a tree this
    size: a buffer that holds a window of the listing rather than the whole of it. Raising it is not
    the fix it looks like -- it holds decoded images, ~19 GB per hundred thousand of them -- so the
    buffer no longer reaches a directory at all, and the file list is shuffled whole instead.
    Twenty items drawn from twenty classes give eleven distinct ones that way; a window gives one.
    """
    loader = _directory_loader(sorted_tree, batch_size=20, is_training=True, shuffle_buffer=8)

    first = next(iter(loader()))

    assert len(set(first["label"].tolist())) >= 8


def test_a_directory_listing_is_reshuffled_every_epoch_and_replayed_by_the_seed(sorted_tree: Path) -> None:
    """Two epochs in the same order would decorrelate nothing after the first pass over the data.

    The other half of the contract is that a seed still replays a run: the shuffle is over the file
    list rather than a Python permutation drawn at construction, so `reshuffle_each_iteration` gives
    a fresh order per epoch while a loader rebuilt from the same seed repeats epoch one exactly.
    Every epoch is also still the whole split -- twenty classes of thirty files, which the batch
    size divides exactly -- because a reshuffle that dropped or repeated files would quietly change
    what an epoch means.
    """
    loader = _directory_loader(sorted_tree, batch_size=20, is_training=True)

    first, second = _labels(loader), _labels(loader)

    assert first != second
    assert sorted(first) == sorted(second) == [label for label in range(20) for _ in range(30)]
    assert _labels(_directory_loader(sorted_tree, batch_size=20, is_training=True)) == first


def _graph_parallelism(dataset: Any) -> tuple[list[int], list[int]]:
    """The `num_parallel_calls` of every map and the buffer of every prefetch, source side first.

    Read off the private attributes of the built graph because `tf.data` publishes no reader for
    either, and a field the pipeline accepted and then ignored is indistinguishable from one it
    honoured until the ops themselves are asked.
    """
    calls: list[int] = []
    buffers: list[int] = []

    def _walk(node: Any) -> None:
        for source in node._inputs():
            _walk(source)
        if (parallel := getattr(node, "_num_parallel_calls", None)) is not None:
            calls.append(int(parallel))
        if (buffer := getattr(node, "_buffer_size", None)) is not None:
            buffers.append(int(buffer))

    _walk(dataset)
    return calls, buffers


def test_the_parallelism_knobs_reach_the_ops_that_take_them(tmp_path: Path) -> None:
    """AUTOTUNE is a floor, not a budget, and the timm example is handed `num_workers: 32`.

    On a host whose cores are shared with a busy JAX process AUTOTUNE settles well under what the
    machine has, and a starved input pipeline shows up as a slower run rather than as an error -- so
    a launch configuration has to be able to hand this one an explicit count, the way it hands the
    torch example its worker count. Both maps take it, the decode included: the file read is the
    first op inside the decode, so that map is the read parallelism too. The evaluation split is
    what is walked here, because the training shuffle carries a buffer of its own and this asserts
    on every buffer the graph holds.
    """
    root = _image_tree(tmp_path)

    budgeted, autotuned = _directory_loader(root, num_parallel_calls=6, prefetch=2), _directory_loader(root)

    calls, buffers = _graph_parallelism(budgeted.dataset)
    assert calls, "no map carries a parallelism at all"
    assert set(calls) == {6}, "a map was left on AUTOTUNE"
    assert buffers == [2]
    calls, buffers = _graph_parallelism(autotuned.dataset)
    assert calls
    assert set(calls) == {tf.data.AUTOTUNE}
    assert buffers == [tf.data.AUTOTUNE]
