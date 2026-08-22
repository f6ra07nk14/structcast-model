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


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_the_image_element_type_is_the_one_that_was_asked_for(dtype: str) -> None:
    """A bfloat16 run wants its batches already narrowed, on the host, before they are placed."""
    batch = next(iter(_loader(image_dtype=dtype)()))

    assert batch["image"].dtype.name == dtype
    assert batch["label"].dtype == np.int32


def test_an_unnamed_dataset_is_refused_where_it_is_written() -> None:
    """There is no default dataset, so an empty name has to fail at construction, not at first read.

    A loader that accepted it would only raise deep inside `tfds.load`, one training run and one
    model build later, with a message about a dataset nobody asked for.
    """
    with pytest.raises(ValueError, match="A dataset name is required"):
        TFDataLoader(name="   ")


@pytest.mark.skipif(find_spec("tensorflow_datasets") is not None, reason="tensorflow_datasets is installed")
def test_a_missing_tensorflow_datasets_says_what_to_install() -> None:
    """The one optional dependency of this file must name itself and its install command.

    It is imported at module scope through `try_import`, so the failure is deferred to the read
    that needs it -- and everything else in the file, this whole test module included, keeps
    working without it.
    """
    with pytest.raises(ImportError, match="uv pip install tensorflow-datasets"):
        _ = TFDataLoader(name="cifar10").source
