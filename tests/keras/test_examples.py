"""Tests for the Keras example code under `examples/keras`.

Everything here is loaded by file path, the way a configuration reaches it with `_addr_` plus
`_file_`: an example that only works when it is imported as a module is an example the configuration
form cannot use. Nothing downloads anything -- the image pipeline runs on its synthetic split and the
corpus reads a local file through `data_path`.
"""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest

import keras
from structcast_model.base_trainer import Learner

EXAMPLES = Path(__file__).resolve().parents[2] / "examples" / "keras"


def _example(name: str) -> ModuleType:
    """Load one example module by file path, as `_file_` does."""
    spec = spec_from_file_location(f"example_keras_{name}", EXAMPLES / f"{name}.py")
    assert spec is not None
    assert spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


DATA = _example("data")
CORPUS = _example("corpus")
OPTIMIZERS = _example("optimizers")
SIMPLE_TRAINING = _example("simple_training")


def _pipeline(**kwargs: Any) -> Any:
    """Build a tiny synthetic image pipeline, small enough to iterate twice per test."""
    defaults = {"batch_size": 4, "samples": 16, "image_size": (8, 8), "num_classes": 3}
    return DATA.KerasImageData(**{**defaults, **kwargs})


def test_the_image_pipeline_yields_batches_keyed_by_the_model_input_names() -> None:
    """The keys are the learner's keyword arguments, so they are the whole interface of a dataset.

    The trainer calls `training_step(**batch)`, which makes a renamed key a `TypeError` on the first
    batch of a run; `image_key` and `label_key` are what a learner with other input names is served
    by, so the rename has to reach the batch.
    """
    data = _pipeline(training=True, image_key="pixels", label_key="target")

    batch = next(iter(data))

    assert sorted(batch) == ["pixels", "target"]
    assert batch["pixels"].shape == (4, 8, 8, 3)
    assert batch["target"].shape == (4,)
    assert len(data) == 4


def test_the_image_pipeline_normalizes_and_stays_numpy() -> None:
    """Batches leave the pipeline as NumPy in [0, 1], which is what every Keras backend accepts.

    TensorFlow tensors would work on the tensorflow backend and nowhere else, and unscaled uint8
    pixels would train every model in this repository at a hundred times the intended input range.
    """
    batch = next(iter(_pipeline(training=False)))

    assert isinstance(batch["image"], np.ndarray)
    assert batch["image"].dtype == np.float32
    assert batch["image"].min() >= 0.0
    assert batch["image"].max() <= 1.0


def _augment_twice(data: Any) -> tuple[np.ndarray, np.ndarray]:
    """Run one split's preprocessing layers over the same fixed image twice.

    The layers, not the pipeline: a whole-pipeline comparison of the training split would differ
    because of the shuffle whether or not anything augmented, which proves nothing about the draws.
    The image is oversized and asymmetric on purpose -- oversized because `RandomCrop` only moves
    anything when there is more image than crop, which in the pipeline is what `crop_padding`
    provides, and asymmetric because a centered square survives a horizontal flip unchanged.
    """
    fixed = np.zeros((2, 12, 12, 3), dtype="uint8")
    fixed[:, 1:5, 2:4] = 255

    def _apply() -> np.ndarray:
        image: Any = fixed
        for layer in data.augmentation:
            image = layer(image)
        return np.asarray(keras.ops.convert_to_numpy(image))

    return _apply(), _apply()


def test_only_the_training_split_augments_its_images() -> None:
    """Augmentation belongs to the training split alone, and evaluation has to be repeatable.

    An augmented validation split makes every epoch's metric a different measurement, and a training
    split whose augmentation silently did nothing -- a preprocessing layer that fell back to
    inference mode inside `tf.data`, say -- would leave the two draws below identical.
    """
    first, second = _augment_twice(_pipeline(training=True))
    assert not np.array_equal(first, second)

    first, second = _augment_twice(_pipeline(training=False))
    assert np.array_equal(first, second)


def test_the_validation_split_yields_the_same_epoch_every_time() -> None:
    """Evaluation is a measurement, so it has to be the same one: no shuffle and no augmentation."""
    validation = _pipeline(training=False)

    assert np.array_equal(next(iter(validation))["image"], next(iter(validation))["image"])
    assert np.array_equal(next(iter(validation))["label"], next(iter(validation))["label"])


def test_the_corpus_yields_the_shifted_pair_the_language_learner_reads(tmp_path: Path) -> None:
    """`targets` is `tokens` shifted by one character, which is the whole of next-token prediction.

    The keys are the inputs of `cfg/keras/learners/SmallLanguageModel.yaml`, and the shift is what
    the loss compares; an off-by-one here trains a model to predict the character it was just given.
    """
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("abcdefghij" * 20, encoding="utf-8")
    loader = CORPUS.TinyShakespeareLoader(block_size=4, batch_size=2, split="train", data_path=corpus)

    batch = next(iter(loader))

    assert sorted(batch) == ["targets", "tokens"]
    assert batch["tokens"].shape == (2, 4)
    assert loader.dataset.vocab_size == 10
    # Item i is the characters at i * block_size, and the targets are the same block shifted by one.
    assert np.array_equal(batch["tokens"][:, 1:], batch["targets"][:, :-1])


def test_the_corpus_reads_a_local_file_without_downloading(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """`data_path` is what makes the example runnable offline, and the tests depend on it.

    Without it the first read fetches a megabyte over the network into `CORPUS_PATH`, which is
    relative to the working directory -- so the run is moved into a temporary one and that path must
    stay unwritten. The vocabulary and the split boundary come from the file actually read, so both
    are pinned against the one written here.
    """
    monkeypatch.chdir(tmp_path)
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("ab" * 100, encoding="utf-8")

    dataset = CORPUS.TinyShakespeare(block_size=4, split="val", data_path=corpus)

    assert dataset.vocabulary == ["a", "b"]
    assert len(dataset.tokens) == 20
    assert not (tmp_path / CORPUS.CORPUS_PATH).exists()


def test_the_optimizer_factory_exempts_the_named_variables_from_weight_decay() -> None:
    """This is the one thing the factory exists for, so it is the one thing worth testing.

    Keras exempts variables through a method call that must land before the optimizer is built, and
    no object pattern can express it; a factory that accepted the argument and dropped it would
    decay every bias and normalization scale while the configuration said otherwise.
    """
    dense = keras.layers.Dense(2, use_bias=True)
    dense.build((None, 2))
    optimizer = OPTIMIZERS.create_optimizer("AdamW", ["bias"], learning_rate=0.1, weight_decay=0.5)
    optimizer.build(dense.trainable_variables)
    kernel, bias = dense.trainable_variables
    bias.assign(keras.ops.ones_like(bias))
    before = [np.asarray(keras.ops.convert_to_numpy(v.value)) for v in (kernel, bias)]

    # Zero gradients leave weight decay as the only thing that can move a variable: Adam's own
    # update is proportional to the gradient, and the decay is proportional to the variable.
    optimizer.apply([keras.ops.zeros_like(v.value) for v in (kernel, bias)], [kernel, bias])

    after = [np.asarray(keras.ops.convert_to_numpy(v.value)) for v in (kernel, bias)]
    assert not np.array_equal(before[0], after[0])
    assert np.array_equal(before[1], after[1])


def test_the_optimizer_factory_names_the_module_it_could_not_find() -> None:
    """A typo in a configuration must say what was wrong, not fail later on a missing attribute."""
    with pytest.raises(ValueError, match="no optimizer named 'Adamw'"):
        OPTIMIZERS.create_optimizer("Adamw")


def test_the_simple_training_example_trains_end_to_end(capsys: pytest.CaptureFixture[str]) -> None:
    """The tutorial is only worth reading if running it as documented still completes a run."""
    SIMPLE_TRAINING.main()

    assert "Best val_loss" in capsys.readouterr().out


def test_the_hand_written_learner_satisfies_the_trainer_protocol() -> None:
    """The trainer and the CLI drive learners through the protocol only, generated or hand-written.

    `flow_functions` names the compiled steps here, as a generated Keras learner's does, because a
    distributed strategy rebinds a run's steps by walking exactly that mapping: a learner returning
    an empty one would run unreplicated while looking wired up.
    """
    keras.utils.set_random_seed(SIMPLE_TRAINING.SEED)
    learner = SIMPLE_TRAINING.SimpleLearner(SIMPLE_TRAINING.build_model())

    assert isinstance(learner, Learner)
    assert learner.optimizer_models == {"optimizer": ["model"]}
    assert sorted(learner.flow_functions) == ["_inference_step", "_training_step"]
    assert learner.learning_rates == {"optimizer": pytest.approx(0.1)}


def test_the_hand_written_learner_counts_the_applies_its_optimizer_reports() -> None:
    """`updates` is read back off the optimizer, not incremented next to it (`docs/adr/0019`).

    An optimizer accumulating over a window applies on every second step here, and the learner has
    to report that without a line of its own changing -- which is exactly what a hand-written
    learner gets wrong when it counts its own applies instead of reading the optimizer's.
    """
    keras.utils.set_random_seed(SIMPLE_TRAINING.SEED)
    learner = SIMPLE_TRAINING.SimpleLearner(SIMPLE_TRAINING.build_model(), gradient_accumulation_steps=2)
    batch = SIMPLE_TRAINING.make_dataset(1, SIMPLE_TRAINING.SEED)[0]

    flags = [bool(learner.training_step(**batch)) and learner.has_updated for _ in range(4)]

    assert flags == [False, True, False, True]
    assert (learner.steps, learner.updates) == (4, 2)
