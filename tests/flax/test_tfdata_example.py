"""Unit tests for the tf.data input pipeline example in examples/flax/data.py.

Every case runs on the synthetic split, so nothing is downloaded and `tensorflow_datasets` is not
needed -- which is exactly the property the synthetic split exists for.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from structcast_model.base_trainer import SimpleDataProvider

# Before the module is loaded: importing the example imports TensorFlow, which the flax floor
# environment does not install. The pipeline is an example integration rather than a floor concern,
# so skipping the file there is the right outcome -- and a collection-time ImportError would not be.
pytest.importorskip("tensorflow")


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

SMALL: dict[str, Any] = {"image_size": 16, "batch_size": 4, "synthetic_samples": 12, "num_classes": 3}
"""A twelve-item synthetic split of sixteen-pixel images: three whole batches."""


def test_batches_are_keyed_by_the_learner_input_names() -> None:
    """The trainer hands a batch to the learner as keyword arguments, so a bare pair never arrives.

    `spec` is what turns the `(image, label)` pair the pipeline produces into those names, and the
    strategy places a batch by iterating its entries -- so a tuple would not even reach the mesh.
    """
    batch = next(iter(TFDataLoader(**SMALL)()))

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
    loader = TFDataLoader(**SMALL)

    assert len(loader) == len(list(loader())) == 3
    assert SimpleDataProvider(training_dataset=loader).steps_per_epoch == 3


def test_the_evaluation_split_is_a_deterministic_central_crop() -> None:
    """Validation must be the same measurement every epoch, so nothing on that path may be random.

    A random crop left switched on for the evaluation split would make the validation criterion
    drift between epochs for reasons that have nothing to do with the model.
    """
    loader = TFDataLoader(**SMALL)

    first, second = next(iter(loader()))["image"], next(iter(loader()))["image"]

    assert np.array_equal(first, second)


def test_the_training_split_augments_but_stays_reproducible_from_its_seed() -> None:
    """Augmentation must actually vary between epochs, and a seed must still replay a whole run.

    The draws are stateless and keyed by the item's position in the shuffled stream: that is what
    gives both properties at once, where a global RNG would give the first and lose the second.
    """
    loader = TFDataLoader(is_training=True, **SMALL)
    twin = TFDataLoader(is_training=True, **SMALL)
    reseeded = TFDataLoader(is_training=True, seed=7, **SMALL)

    first, second = next(iter(loader()))["image"], next(iter(loader()))["image"]

    assert first.shape == (4, 16, 16, 3)
    assert not np.array_equal(first, second)
    assert np.array_equal(first, next(iter(twin()))["image"])
    assert not np.array_equal(first, next(iter(reseeded()))["image"])


def test_the_crop_ratio_decides_how_much_of_the_image_survives() -> None:
    """`crop_pct` is the one knob relating the resize to the crop, so it must reach the resize.

    A pipeline that resized straight to `image_size` would crop nothing at all, which is a
    different augmentation from the one the configuration asks for.
    """
    assert TFDataLoader(image_size=224, crop_pct=0.875).resize_size == 256
    assert TFDataLoader(image_size=16, crop_pct=1.0).resize_size == 16


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_the_image_element_type_is_the_one_that_was_asked_for(dtype: str) -> None:
    """A bfloat16 run wants its batches already narrowed, on the host, before they are placed."""
    batch = next(iter(TFDataLoader(image_dtype=dtype, **SMALL)()))

    assert batch["image"].dtype.name == dtype
    assert batch["label"].dtype == np.int32
