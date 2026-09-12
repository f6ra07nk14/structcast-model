"""Unit tests for the CycleGAN loader in examples/flax/cyclegan.py.

`cfg/flax/learners/CycleGAN.yaml` declares `INPUTS: [real_A, real_B]` and its discriminator segments
train on what the generator segment just produced, so the example is the loader alone -- no replay
buffer, unlike the torch twin. What is worth testing is therefore the dataset contract the template
cannot state for itself: two domains drawn apart, in the layout and the range its models read, and
that `scm flax train` runs the shipped template over them.
"""

from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
from PIL import Image
import pytest
from typer.testing import CliRunner

from structcast_model.builders.flax import FlaxBuilder, FlaxLearnerBuilder
from structcast_model.commands.cmd_flax import app
from tests import CFG_DIR

MODELS = CFG_DIR / "flax" / "models"
LEARNERS = CFG_DIR / "flax" / "learners"

EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "flax" / "cyclegan.py"
"""The example under test, addressed by file path exactly as a configuration addresses it."""

CRITERIA = ["loss_G", "loss_GAN", "loss_cycle", "loss_identity", "loss_D_A", "loss_D_B"]
"""Every criterion the template's three segments report between them."""

IMAGE_SIZE = 32
"""Side length the models run on: the discriminator strides four times, so 16 leaves it nothing."""


def _load_example_module() -> Any:
    """Load the example module the way a configuration does: by file path, not by import name."""
    spec = spec_from_file_location("example_flax_cyclegan", EXAMPLE)
    assert spec is not None
    assert spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


UnpairedImageLoader = _load_example_module().UnpairedImageLoader


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
        **{
            "root_A": root_A,
            "root_B": root_B,
            "load_size": 10,
            "crop_size": 8,
            "batch_size": 1,
            "seed": 0,
            **overrides,
        }
    )


def _shades(loader: Any, name: str) -> list[float]:
    """The first pixel of every image an epoch yields under *name*, which identifies its file."""
    return [float(batch[name][0, 0, 0, 0]) for batch in loader()]


def test_a_batch_carries_both_domains_in_the_layout_and_range_the_models_read(tmp_path: Path) -> None:
    """NHWC and [-1, 1] are the model contract, and neither is checked at runtime.

    `cfg/flax/models/CycleGAN_generator.yaml` is written against the NHWC layout Flax convolutions
    read, and it closes on `flax.nnx.tanh`, which is what the identity and cycle losses compare a
    real image against. A [0, 1] image would train against a target half the generator cannot reach,
    and a transposed one would be silently convolved over the wrong axes.
    """
    batch = next(iter(_loader(tmp_path)()))

    assert sorted(batch) == ["real_A", "real_B"]
    assert batch["real_A"].shape == (1, 8, 8, 3)
    assert batch["real_A"].dtype == np.float32
    assert batch["real_A"].min() >= -1.0
    assert batch["real_A"].max() <= 1.0
    assert batch["real_B"].min() >= -1.0
    assert batch["real_B"].max() <= 1.0


def test_an_epoch_is_the_longer_domain_and_the_two_are_drawn_apart(tmp_path: Path) -> None:
    """Unpaired means the two domains are indexed independently, which is the dataset contract.

    Zipping them unshuffled would stop at the shorter directory -- two of the five horses would never
    be seen -- and would show the same horse the same zebra in every epoch, which is exactly the
    alignment CycleGAN exists to do without.
    """
    loader = _loader(tmp_path)

    assert len(loader) == 5  # max(5, 3) items, one per batch
    assert len(_shades(loader, "real_A")) == 5
    assert len(set(_shades(loader, "real_B"))) > 1  # the smaller domain wraps, and not in file order
    assert _shades(loader, "real_A") != _shades(loader, "real_B")


def test_the_same_seed_replays_the_same_unaligned_pairing(tmp_path: Path) -> None:
    """Reproducing a run means reproducing it, so the pairing has to come from `seed` alone.

    Both domains are shuffled, and the shuffles must be seeded apart: one seed for both would walk
    the two directories in step and pair image `i` with image `i` after all.
    """
    first, second = _loader(tmp_path), _loader(tmp_path)
    other = _loader(tmp_path, seed=1)

    assert _shades(first, "real_B") == _shades(second, "real_B")
    assert _shades(first, "real_A") != _shades(first, "real_B")
    assert _shades(first, "real_B") != _shades(other, "real_B")


def test_the_pairing_is_redrawn_on_every_epoch(tmp_path: Path) -> None:
    """A fixed pairing is a paired dataset with extra steps.

    Each domain is shuffled and repeated on its own, so the second pass over the loader has to meet
    a different set of partners -- which is what makes the smaller domain's images generalize rather
    than memorize one counterpart.
    """
    loader = _loader(tmp_path)

    assert [_shades(loader, "real_A"), _shades(loader, "real_B")] != [
        _shades(loader, "real_A"),
        _shades(loader, "real_B"),
    ]


@pytest.fixture(scope="module")
def generated(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Generate the two shipped model templates and the learner once, shrunk to a few channels."""
    directory = tmp_path_factory.mktemp("generated")
    FlaxBuilder.from_path(MODELS / "CycleGAN_generator.yaml")(
        parameters={"DEFAULT": {"n_residual_blocks": 1, "init_features": 4}}
    )(directory / "generator.py")
    FlaxBuilder.from_path(MODELS / "CycleGAN_discriminator.yaml")()(directory / "discriminator.py")
    FlaxLearnerBuilder.from_path(LEARNERS / "CycleGAN.yaml")(
        parameters={"DEFAULT": {"epochs": 2, "decay_epoch": 1, "steps_per_epoch": 3}}
    )(directory / "learner.py")
    return directory


def test_the_template_trains_through_the_command_the_example_documents(
    tmp_path: Path, generated: Path, cli_runner: CliRunner
) -> None:
    """The whole point of the example: `scm flax train` runs the shipped template end to end.

    Nothing short of the command proves it. The loader has to produce the two names the template
    declares as its inputs, in the layout its convolutions read, and hand them over as NumPy arrays
    the strategy can place on its mesh -- a chain only exercised as a whole.
    """
    mlflow.set_tracking_uri(str(tmp_path / "mlruns"))
    root_A, root_B = _domains(tmp_path)
    dataset = (
        f"[_obj_, {{_addr_: UnpairedImageLoader, _file_: {EXAMPLE}}}, "
        f"{{_call_: {{root_A: {root_A}, root_B: {root_B}, load_size: {IMAGE_SIZE + 4}, "
        f"crop_size: {IMAGE_SIZE}, batch_size: 2}}}}]"
    )

    result = cli_runner.invoke(
        app,
        [
            "train",
            *(
                f"{name}: [_obj_, {{_addr_: Model, _file_: {generated / file}}}]"
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
            f"image: [{IMAGE_SIZE}, {IMAGE_SIZE}, 3]",
            "--training-dataset",
            dataset,
            "-e",
            "1",
            "-LC",
            "loss_G",
            "--ci",
            "-E",
            "cyclegan-flax-example",
        ],
    )

    assert result.exit_code == 0, result.output
    run = mlflow.search_runs(experiment_names=["cyclegan-flax-example"], output_format="list")[0]
    # All six criteria, from all three segments: a segment that never ran reports nothing here.
    assert set(CRITERIA) <= set(run.data.metrics)
    assert all(np.isfinite(run.data.metrics[name]) for name in CRITERIA)
