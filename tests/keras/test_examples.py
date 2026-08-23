"""Tests for the Keras example code under `examples/keras`.

Everything here is loaded by file path, the way a configuration reaches it with `_addr_` plus
`_file_`: an example that only works when it is imported as a module is an example the configuration
form cannot use.

Nothing downloads anything, and the image pipeline's two sources are each covered on their own
terms. The in-memory one is fed through the `arrays` fixture, which replaces the module-level
`load_arrays` -- the one place that path reaches for data, so everything between that call and the
batch is the real code. The directory one is given a real tree of tiny PNGs under `tmp_path`, since
listing and decoding files is precisely what it is being asked to do. The corpus reads a local file
through `data_path`.
"""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest

import keras
from structcast_model.base_trainer import Learner
from structcast_model.builders.schema import Template
from tests import CFG_DIR

BACKEND = keras.backend.backend()
"""The active Keras backend, as in `tests/keras/test_distributed.py`."""

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


IMAGES = np.arange(16 * 12 * 12 * 3, dtype="uint8").reshape(16, 12, 12, 3)
"""Sixteen 12x12 images, larger than the pipeline's crop so the random crop has room to move."""

LABELS = np.arange(16, dtype="int64")
"""One distinct label per image, so a batch's labels say exactly which items produced it."""


@pytest.fixture
def arrays(monkeypatch: pytest.MonkeyPatch) -> None:
    """Feed the pipeline the arrays above instead of a `keras.datasets` download.

    `load_arrays` is the single seam between the pipeline and where its data comes from, which is
    what makes it the right thing to replace: the sharding, the shuffle, the augmentation, the
    batching and the keys under test are all the real code. Downloading a set would put the network
    and a hundred megabytes of cache into every run of this file.
    """
    monkeypatch.setattr(DATA, "load_arrays", lambda dataset, training: (IMAGES, LABELS))


def _pipeline(**kwargs: Any) -> Any:
    """Build a tiny image pipeline; `dataset` names the set the patched `load_arrays` ignores."""
    defaults = {"dataset": "cifar10", "batch_size": 4, "image_size": (8, 8)}
    return DATA.KerasImageData(**{**defaults, **kwargs})


def test_the_image_pipeline_yields_batches_keyed_by_the_model_input_names(arrays: None) -> None:
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


def test_the_image_pipeline_normalizes_and_stays_numpy(arrays: None) -> None:
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


def test_the_validation_split_yields_the_same_epoch_every_time(arrays: None) -> None:
    """Evaluation is a measurement, so it has to be the same one: no shuffle and no augmentation."""
    validation = _pipeline(training=False)

    assert np.array_equal(next(iter(validation))["image"], next(iter(validation))["image"])
    assert np.array_equal(next(iter(validation))["label"], next(iter(validation))["label"])


def _image_tree(root: Path, classes: int = 2, per_class: int = 4) -> Path:
    """Write a class-per-folder tree of tiny PNGs, the layout the timm loader reads too.

    Every image is one flat colour and no two share it, so the pixel value of a decoded batch says
    exactly which files produced it -- which is what the sharding assertions below rest on.
    """
    for label in range(classes):
        folder = root / f"class{label}"
        folder.mkdir(parents=True)
        for index in range(per_class):
            colour = 10 * (label * per_class + index) + 1
            keras.utils.save_img(folder / f"{index}.png", np.full((8, 8, 3), colour, "uint8"), scale=False)
    return root


def _colours(batches: Any) -> list[int]:
    """The flat colour of every image an epoch yielded, recovered from the rescaled pixels."""
    return [round(float(image[0, 0, 0]) * 255) for batch in batches for image in batch["image"]]


def test_the_streaming_source_yields_the_same_batch_contract_as_the_array_source(tmp_path: Path) -> None:
    """A directory and a `keras.datasets` name must be interchangeable from the learner's side.

    They are not interchangeable underneath -- a directory hands over float32 images in 0..255 and
    int32 labels where an array set hands over uint8 and int64 -- so the keys, the shapes and above
    all the dtypes are pinned here: a run that swapped a small set for the real one would otherwise
    only find out inside the loss, where an int32 label is a different error message.
    """
    data = DATA.KerasImageData(dataset=_image_tree(tmp_path), batch_size=2, image_size=(8, 8))

    batch = next(iter(data))

    assert sorted(batch) == ["image", "label"]
    assert batch["image"].shape == (2, 8, 8, 3)
    assert batch["image"].dtype == np.float32
    assert batch["image"].min() >= 0.0
    assert batch["image"].max() <= 1.0
    assert batch["label"].shape == (2,)
    assert batch["label"].dtype == np.int64
    # Per-item, not per-set: the source decodes as the pipeline pulls, which is what lets a set of
    # ImageNet's size through at all. A source that had read the tree into one array would spec (n, ...).
    assert tuple(data.source.element_spec[0].shape) == (8, 8, 3)
    assert (data.items, len(data)) == (8, 4)


def test_the_streaming_source_shards_its_files_across_the_launcher_ranks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The sharding has to cut the file list, not the decoded images, or every rank decodes the lot.

    Same contract as the in-memory path -- disjoint, covering the split between them, equal length,
    stable for a rank -- asserted on the directory source because the cut happens somewhere else
    there: on the `tf.data` pipeline rather than on a NumPy slice.
    """
    root = _image_tree(tmp_path)
    monkeypatch.setenv("WORLD_SIZE", "2")

    shards = []
    for rank in ("0", "1"):
        monkeypatch.setenv("RANK", rank)
        data = DATA.KerasImageData(dataset=root, batch_size=2, image_size=(8, 8))
        assert len(data) == 2, "four files per rank, two to a batch"
        shards.append(_colours(data))

    assert not set(shards[0]) & set(shards[1])
    assert sorted(shards[0] + shards[1]) == [10 * index + 1 for index in range(8)]
    assert len(shards[0]) == len(shards[1])
    assert _colours(DATA.KerasImageData(dataset=root, batch_size=2, image_size=(8, 8))) == shards[1]


def test_the_streaming_source_shuffles_the_same_way_for_the_same_seed(tmp_path: Path) -> None:
    """A run has to be replayable, and the shuffle is the only thing here that could stop it being.

    Asserted on the labels, one per file here, because they are what the shuffle reorders and the
    augmentation cannot touch: an image would answer for the random crop as well and say nothing
    about the item order. Two pipelines built from one seed must agree epoch for epoch; two seeds
    must not, or the seed is ignored and every run is silently the file order.
    """
    root = _image_tree(tmp_path, classes=8, per_class=1)

    def _epoch(seed: int) -> list[int]:
        data = DATA.KerasImageData(dataset=root, batch_size=2, image_size=(8, 8), training=True, seed=seed)
        return [int(label) for batch in data for label in batch["label"]]

    assert _epoch(0) == _epoch(0)
    assert _epoch(0) != _epoch(7)
    assert sorted(_epoch(0)) == list(range(8))


def _labels_of_rank(rank: int, monkeypatch: pytest.MonkeyPatch) -> np.ndarray:
    """Every label one rank of a two-process launch sees in an epoch, in order."""
    monkeypatch.setenv("RANK", str(rank))
    monkeypatch.setenv("WORLD_SIZE", "2")
    data = _pipeline(training=False)
    assert len(data) == 2, "eight items per rank, four to a batch"
    return np.concatenate([batch["label"] for batch in data])


def test_the_image_pipeline_shards_its_items_across_the_launcher_ranks(
    arrays: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Under a torchrun launch every rank is its own process, so an unsharded loader trains on copies.

    That failure is silent -- the run completes, the loss falls, and every rank has just replayed the
    same epoch -- so the shards are pinned here: disjoint, covering the split between them, the same
    length, and the same on a second construction of the same rank.
    """
    first = _labels_of_rank(0, monkeypatch)
    second = _labels_of_rank(1, monkeypatch)

    assert not set(first) & set(second)
    assert sorted([*first, *second]) == LABELS.tolist()
    assert len(first) == len(second)
    assert np.array_equal(first, _labels_of_rank(0, monkeypatch))


def test_the_image_pipeline_serves_the_whole_split_outside_a_multi_process_launch(arrays: None) -> None:
    """Without RANK and WORLD_SIZE the loader must not shard, which is the tensorflow and jax case.

    Those backends run one process and the distributed strategy splits each batch across the
    replicas itself, so a loader sharding here as well would hand each replica a shard of a shard --
    the same silent correctness bug as not sharding under torchrun, in the other direction.
    """
    data = _pipeline(training=False)

    assert len(data) == 4
    assert sorted(np.concatenate([batch["label"] for batch in data])) == LABELS.tolist()


def test_the_image_pipeline_requires_a_dataset_and_names_the_ones_it_knows() -> None:
    """`dataset` has no default, because any default downloads a set nobody asked for.

    A typo has to be refused where it is written, not at the first batch after a download of the
    wrong thing, and the message has to say what the alternatives are.
    """
    with pytest.raises(ValueError, match="Field required"):
        DATA.KerasImageData()

    with pytest.raises(ValueError, match="'mnist', 'cifar10' or 'cifar100'"):
        DATA.KerasImageData(dataset="synthetic")


def test_the_shipped_dataset_template_refuses_to_render_without_a_dataset() -> None:
    """`scm format` must fail on the missing parameter, not render `{{dataset}}` into the pattern.

    An undefined parameter renders as its own literal here, so without the guard the template would
    produce a file that looks complete and fails one command later; the message has to name the
    parameter and the sets. The rendered pattern is only inspected, never instantiated: `_file_`
    loads a fresh copy of the example module, which the `arrays` fixture cannot reach, so building
    it would download a real set.
    """
    template: Template[Any] = Template.from_path(CFG_DIR / "keras" / "others" / "default_keras.yaml")

    with pytest.raises(ValueError, match="A source is required and has no default"):
        template({})

    pattern = template({"DEFAULT": {"dataset": "cifar100", "batch_size": 8}}).model_dump(mode="json")

    arguments = pattern["_obj_"][-1][-1]
    assert arguments["dataset"] == "cifar100"
    assert arguments["batch_size"] == 8
    assert (arguments["image_key"], arguments["label_key"]) == ("image", "label")


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


def _corpus_loader(path: Path) -> Any:
    """A loader over 23 blocks of four identical characters each, so every block is recognizable."""
    path.write_text("".join(chr(ord("a") + index) * 4 for index in range(23)), encoding="utf-8")
    return CORPUS.TinyShakespeareLoader(block_size=4, batch_size=2, split="train", data_path=path)


def _blocks_of_rank(rank: int, path: Path, monkeypatch: pytest.MonkeyPatch) -> np.ndarray:
    """Every block one rank of a two-process launch sees in an epoch, named by its first token."""
    monkeypatch.setenv("RANK", str(rank))
    monkeypatch.setenv("WORLD_SIZE", "2")
    loader = _corpus_loader(path)
    return np.concatenate([batch["tokens"][:, 0] for batch in loader])


def test_the_corpus_loader_shards_its_blocks_across_the_launcher_ranks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The same defect as in the image pipeline, and the same silence: every rank replays one epoch.

    The blocks are pinned the same way -- disjoint, covering the split between them, the same
    length, and stable for a rank -- with `shuffle` off, so what the shards contain is the sharding
    alone and not a draw.
    """
    corpus = tmp_path / "corpus.txt"
    blocks = len(_corpus_loader(corpus).dataset)

    first = _blocks_of_rank(0, corpus, monkeypatch)
    second = _blocks_of_rank(1, corpus, monkeypatch)

    assert not set(first) & set(second)
    assert sorted([*first, *second]) == list(range(blocks))
    assert len(first) == len(second)
    assert np.array_equal(first, _blocks_of_rank(0, corpus, monkeypatch))


def test_the_corpus_loader_serves_the_whole_split_outside_a_multi_process_launch(tmp_path: Path) -> None:
    """Without RANK and WORLD_SIZE the loader must not shard: the strategy splits the batch instead.

    On the tensorflow and jax backends the run is one process feeding a strategy that spreads each
    batch over the replicas, so sharding here too would give every replica a shard of a shard.
    """
    loader = _corpus_loader(tmp_path / "corpus.txt")

    seen = np.concatenate([batch["tokens"][:, 0] for batch in loader])

    assert sorted(seen) == list(range(len(loader.dataset)))


@pytest.mark.skipif(
    BACKEND == "jax",
    reason="The stateful optimizer.apply this asserts through is a tensorflow and torch path; on jax "
    "the backend adapter applies statelessly inside the jitted step, which no eager call can stand in for.",
)
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
