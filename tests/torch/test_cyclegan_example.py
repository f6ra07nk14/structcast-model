"""Unit tests for the CycleGAN example in examples/torch/cyclegan.py.

`cfg/torch/learners/CycleGAN.yaml` is the only one of the three templates that reads a replay-buffer
sample out of its batch, and nothing in the package can produce one: `fake_A_sample` and
`fake_B_sample` are the generators' own earlier output. The example supplies both halves -- the
unpaired loader and the trainer that closes the loop -- so what is worth testing is that a batch
really carries two unaligned domains, that the buffer follows the paper's rule, and that the shipped
template trains through the real command with them.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
from PIL import Image
import pytest
from typer.testing import CliRunner

from structcast_model.builders.torch import TorchBuilder, TorchLearnerBuilder
from structcast_model.commands.cmd_torch import app
from tests import CFG_DIR
import torch

MODELS = CFG_DIR / "torch" / "models"
LEARNERS = CFG_DIR / "torch" / "learners"

EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "torch" / "cyclegan.py"
"""The example under test, addressed by file path exactly as a configuration addresses it."""

CRITERIA = ["loss_G", "loss_GAN", "loss_cycle", "loss_identity", "loss_D_A", "loss_D_B"]
"""The six scalar criteria of the template. `fake_A` and `fake_B` are outputs too, but images."""

IMAGE_SIZE = 32
"""Side length the models run on: the discriminator strides four times, so 16 leaves it nothing."""


def _load_example_module() -> Any:
    """Load the example module the way a configuration does: by file path, not by import name."""
    spec = importlib.util.spec_from_file_location("example_cyclegan", EXAMPLE)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_module = _load_example_module()
ImagePool = _module.ImagePool
UnpairedImageLoader = _module.UnpairedImageLoader


def _domains(root: Path, count_A: int = 5, count_B: int = 3) -> tuple[Path, Path]:
    """Write two directories of tiny JPEGs, deliberately of different sizes.

    Each image is a flat shade, so the value of a decoded pixel names the file it came from and a
    test can say which domain image ended up in a batch.
    """
    directories = []
    for name, count in (("trainA", count_A), ("trainB", count_B)):
        directory = root / name
        directory.mkdir(parents=True, exist_ok=True)
        for index in range(count):
            Image.fromarray(np.full((8, 8, 3), 10 + 40 * index, dtype=np.uint8)).save(directory / f"{index}.jpg")
        directories.append(directory)
    return directories[0], directories[1]


def _loader(root: Path, **overrides: Any) -> Any:
    """A loader over the two synthetic domains, at a resolution the tests can afford."""
    root_A, root_B = _domains(root)
    return UnpairedImageLoader(
        **{"root_A": root_A, "root_B": root_B, "load_size": 10, "crop_size": 8, "batch_size": 2, "seed": 0, **overrides}
    )


def test_a_batch_carries_both_domains_in_the_range_the_generator_emits(tmp_path: Path) -> None:
    """The batch names and the value range are the model contract, and neither is checked at runtime.

    The learner takes its batch as keyword arguments, so a differently named key is a `TypeError`
    on the first step; the range matters just as quietly, since the generator closes on a `Tanh` and
    the identity and cycle losses compare a real image against exactly that output. A [0, 1] image
    would train against a target half the generator cannot reach.
    """
    batch = next(iter(_loader(tmp_path)))

    assert sorted(batch) == ["real_A", "real_B"]
    assert batch["real_A"].shape == (2, 3, 8, 8)
    assert batch["real_A"].dtype == torch.float32
    assert batch["real_A"].min() >= -1.0
    assert batch["real_A"].max() <= 1.0
    assert batch["real_B"].min() >= -1.0
    assert batch["real_B"].max() <= 1.0


def test_an_epoch_is_the_longer_domain_and_the_two_are_drawn_apart(tmp_path: Path) -> None:
    """Unpaired means the two domains are indexed independently, which is the whole dataset contract.

    A loader that zipped them would stop at the shorter directory -- three of the five horses would
    never be seen -- and would show the same horse the same zebra in every epoch, which is exactly
    the alignment CycleGAN exists to do without.
    """
    dataset = _loader(tmp_path).dataset

    assert len(dataset) == 5  # max(5, 3), so every image of the larger domain is seen once
    drawn = [dataset[index]["real_B"][0, 0, 0].item() for index in range(len(dataset))]
    aligned = [dataset[index]["real_A"][0, 0, 0].item() for index in range(len(dataset))]
    assert len(set(drawn)) > 1  # the smaller domain wraps around, and not in index order
    assert drawn != aligned[: len(drawn)]


def test_the_same_seed_replays_the_same_unaligned_pairing(tmp_path: Path) -> None:
    """The draw is the one part of an unpaired epoch that is not a function of the file listing.

    Reproducing a run means reproducing it, so the pairing has to come from `seed` alone rather than
    from a global RNG some other part of the process may have advanced.
    """
    first = [batch["real_B"] for batch in _loader(tmp_path)]
    torch.rand(10)  # a global draw, which must not reach the dataset's own generator
    second = [batch["real_B"] for batch in _loader(tmp_path)]
    other = [batch["real_B"] for batch in _loader(tmp_path, seed=1)]

    assert all(torch.equal(a, b) for a, b in zip(first, second, strict=True))
    assert any(not torch.equal(a, b) for a, b in zip(first, other, strict=True))


def test_loader_shards_on_the_data_coordinates_rather_than_the_global_rank(
    tmp_path: Path, single_process_gloo: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`scm torch train` publishes the data coordinates, and the shard must follow them.

    Under a tensor-parallel strategy the ranks of one group split a single model and must be fed the
    identical batch; sharding on the global rank there hands each of them a different slice, which
    trains a mismatched model rather than failing.
    """
    monkeypatch.setenv("DATA_RANK", "1")
    monkeypatch.setenv("DATA_WORLD_SIZE", "2")
    loader = _loader(tmp_path, batch_size=1)
    loader.__dict__["distributed_results"] = {"device": "cpu", "distributed": True}

    assert (loader.dataloader.sampler.rank, loader.dataloader.sampler.num_replicas) == (1, 2)
    assert len(loader.dataloader.sampler) == 3  # ceil(5 / 2), the padded half of the longer domain


def _single(value: float) -> torch.Tensor:
    """A one-image batch whose only pixel is *value*, so a buffered image is identifiable."""
    return torch.full((1, 1, 1, 1), value)


def test_the_pool_fills_before_it_swaps_and_never_outgrows_its_size(tmp_path: Path) -> None:
    """Until the buffer is full every image is stored and handed straight back.

    The paper's buffer is a history, and a history that dropped or duplicated its first fifty
    entries would show the discriminator the same few images for the first fifty steps.
    """
    pool = ImagePool(size=4, seed=0)

    returned = [pool.query(_single(float(index))).item() for index in range(4)]

    assert returned == [0.0, 1.0, 2.0, 3.0]
    assert sorted(image.item() for image in pool.images) == [0.0, 1.0, 2.0, 3.0]
    for index in range(20):
        pool.query(_single(100.0 + index))
    assert len(pool.images) == 4


def test_the_pool_returns_a_buffered_image_about_half_the_time(tmp_path: Path) -> None:
    """The coin is the mechanism: half the discriminator's fakes come from the buffer, half are new.

    A buffer that always returned the newest image is no buffer at all, and one that always returned
    a stored image would stop the discriminator from ever seeing what the generator does now.
    Fifty images back is the reach that damps the oscillation the paper describes.
    """
    pool = ImagePool(size=50, seed=0)
    for index in range(50):
        pool.query(_single(float(index)))

    offered = [100.0 + index for index in range(400)]
    returned = [pool.query(_single(value)).item() for value in offered]

    swapped = [given for given, back in zip(offered, returned, strict=True) if given != back]
    assert 0.4 < len(swapped) / len(offered) < 0.6
    # The buffer reaches back, not just one step: images stored before the 400 queries still come out.
    assert min(returned) < 50.0
    assert len(pool.images) == 50


def test_a_pool_of_no_images_hands_every_image_straight_back() -> None:
    """`pool_size: 0` is the off-switch, which a run comparing against the paper's ablation needs."""
    pool = ImagePool(size=0, seed=0)
    images = torch.arange(3.0).reshape(3, 1, 1, 1)

    assert torch.equal(pool.query(images), images)
    assert pool.images == []


@pytest.fixture(scope="module")
def generated(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Generate the two shipped model templates and the learner once, shrunk to a few channels."""
    directory = tmp_path_factory.mktemp("generated")
    TorchBuilder.from_path(MODELS / "CycleGAN_generator.yaml")(
        parameters={"DEFAULT": {"n_residual_blocks": 1, "init_features": 4}}
    )(directory / "generator.py")
    TorchBuilder.from_path(MODELS / "CycleGAN_discriminator.yaml")()(directory / "discriminator.py")
    TorchLearnerBuilder.from_path(LEARNERS / "CycleGAN.yaml")(parameters={"DEFAULT": {"epochs": 2, "decay_epoch": 1}})(
        directory / "learner.py"
    )
    return directory


def _train_arguments(generated: Path, root_A: Path, root_B: Path, *, learner_outputs: list[str]) -> list[str]:
    """The `scm torch train` invocation the example documents, pointed at the synthetic domains."""
    dataset = (
        f"[_obj_, {{_addr_: UnpairedImageLoader, _file_: {EXAMPLE}}}, "
        f"{{_call_: {{root_A: {root_A}, root_B: {root_B}, load_size: {IMAGE_SIZE + 4}, "
        f"crop_size: {IMAGE_SIZE}, batch_size: 2}}}}]"
    )
    outputs = [flag for name in learner_outputs for flag in ("-LO", name)]
    return [
        "train",
        *(
            f"{name}: [_obj_, {{_addr_: Model, _file_: {generated / file}}}, _call_]"
            for name, file in (
                ("G_AB", "generator.py"),
                ("G_BA", "generator.py"),
                ("D_A", "discriminator.py"),
                ("D_B", "discriminator.py"),
            )
        ),
        "-L",
        f"[_obj_, {{_addr_: Learner, _file_: {generated / 'learner.py'}}}]",
        "-s",
        f"image: [3, {IMAGE_SIZE}, {IMAGE_SIZE}]",
        "--training-dataset",
        dataset,
        "--trainer",
        f"[_obj_, {{_addr_: CycleGANTrainer, _file_: {EXAMPLE}}}]",
        *outputs,
        "-d",
        "cpu",
        "-e",
        "1",
        "-LC",
        "loss_G",
        "--ci",
        "-E",
        "cyclegan-example",
    ]


def test_the_template_trains_through_the_command_the_example_documents(
    tmp_path: Path, generated: Path, cli_runner: CliRunner
) -> None:
    """The whole point of the example: `scm torch train` runs the shipped template end to end.

    Nothing short of the command proves it. The loader has to produce the two names the template
    reads, `--trainer` has to be the seam that adds the other two, and the buffer has to hand back
    something of the right shape for the discriminators -- a chain in which every link is only
    exercised together with the rest.
    """
    mlflow.set_tracking_uri(str(tmp_path / "mlruns"))
    root_A, root_B = _domains(tmp_path)

    result = cli_runner.invoke(app, _train_arguments(generated, root_A, root_B, learner_outputs=CRITERIA))

    assert result.exit_code == 0, result.output
    run = mlflow.search_runs(experiment_names=["cyclegan-example"], output_format="list")[0]
    # All six criteria, from all three segments: a segment that never ran reports nothing here.
    assert set(CRITERIA) <= set(run.data.metrics)
    assert all(np.isfinite(run.data.metrics[name]) for name in CRITERIA)


def test_the_generated_images_must_be_kept_out_of_the_tracker(
    tmp_path: Path, generated: Path, cli_runner: CliRunner
) -> None:
    """`-LO` is not decoration: the template outputs the two generated images, and it has to.

    That is how they reach the replay buffer, and it is also why the criterion list must be given
    explicitly -- the tracker sums every criterion into a one-element buffer, which an image does
    not broadcast into. Leaving `-LO` off is the mistake this documents, so the example's command
    carries the six names.
    """
    mlflow.set_tracking_uri(str(tmp_path / "mlruns"))
    root_A, root_B = _domains(tmp_path)

    result = cli_runner.invoke(app, _train_arguments(generated, root_A, root_B, learner_outputs=[]))

    assert result.exit_code != 0
    assert isinstance(result.exception, RuntimeError)
    assert "broadcast" in str(result.exception)
