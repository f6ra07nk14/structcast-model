"""A complete, runnable PyTorch training program written directly against the trainer API.

Everything the CLI wires up from YAML is written by hand here, so the moving parts stay visible:

* a **Learner** owning the model, the optimizer, and the two step definitions;
* a **DataProvider** carrying the training and validation datasets;
* a **tracker** turning the criteria of one step into plain floats;
* **callbacks** -- `Printer` and `BestCriterion` -- routed into lifecycle events by protocol.

Run it with::

    uv run python examples/torch/simple_training.py

It trains a two-layer MLP on a synthetic, deterministic dataset and finishes in a few seconds on
the CPU.
"""

from typing import Any

from structcast_model.base_trainer import BaseInfo, BestCriterion, Printer, SimpleDataProvider
from structcast_model.torch.trainer import TorchTrainer
import torch

# The datasets below are built once from this generator, so every run sees the same tensors.
SEED = 0
BATCHES = 20
BATCH_SIZE = 32
FEATURES = 8
CLASSES = 3


def make_dataset(batches: int, generator: torch.Generator) -> list[dict[str, torch.Tensor]]:
    """Build a synthetic classification dataset as a list of input dictionaries.

    A dataset is anything iterable that yields dictionaries; the keys become the keyword arguments
    of `training_step` and `inference_step`. A list of pre-built batches is the simplest form and
    needs no downloads.

    Args:
        batches: Number of batches to build.
        generator: The random generator producing the tensors, seeded by the caller.

    Returns:
        One dictionary per batch, each with an `x` feature tensor and a `y` label tensor.
    """
    dataset = []
    for _ in range(batches):
        x = torch.randn(BATCH_SIZE, FEATURES, generator=generator)
        # A learnable rule: the label is the argmax over the first CLASSES features.
        y = x[:, :CLASSES].argmax(dim=1)
        dataset.append({"x": x, "y": y})
    return dataset


class SimpleLearner:
    """A hand-written learner: it owns the model and the optimizer and defines both steps.

    This is the object the redesign asks you to customize per model. It implements the `Learner`
    protocol -- a `models` property, `update`, `training_step`, and `inference_step` -- and nothing
    more is required. Extra methods named after a lifecycle event, such as `on_epoch_end` below,
    are picked up by the trainer automatically.
    """

    def __init__(self, model: torch.nn.Module, learning_rate: float = 0.1) -> None:
        """Create the learner over *model*, with its own optimizer and learning-rate schedule.

        Args:
            model: The model to train.
            learning_rate: The initial learning rate of the SGD optimizer.
        """
        self.model = model
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
        # Composition of an optimizer and a schedule belongs to the learner, not to the package.
        # Stepping it from `on_epoch_end` keeps this example to one object; the alternative is an
        # optimizer wrapper that implements the event protocols itself -- see
        # `examples/torch/optimizers.py` for that variant.
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=0.5)

    @property
    def models(self) -> dict[str, torch.nn.Module]:
        """The models to train, by name. The trainer passes them to every callback as keywords."""
        return {"model": self.model}

    @property
    def optimizers(self) -> dict[str, torch.optim.Optimizer]:
        """The optimizers by name: the trainer scans them for event protocols right after the learner."""
        return {"optimizer": self.optimizer}

    @property
    def learning_rates(self) -> dict[str, float]:
        """The current learning rate per optimizer, shown by `Printer` next to the criteria."""
        return {name: opt.param_groups[0]["lr"] for name, opt in self.optimizers.items()}

    def update(self, step: int) -> bool:
        """Report whether *step* applied the optimizer.

        Returning True on every step means "no gradient accumulation": one step, one update. A
        learner accumulating over N batches would return True only every N-th step, and the trainer
        would fire `on_update` that often.

        Args:
            step: The training step that just ran, counted from 1 across the whole run.

        Returns:
            True, because this learner updates on every step.
        """
        return True

    def training_step(self, x: torch.Tensor, y: torch.Tensor, **kwargs: Any) -> dict[str, torch.Tensor]:
        """Run one training batch: forward, backward, and optimizer step.

        Args:
            x: The input features of the batch.
            y: The target labels of the batch.
            **kwargs: Further dataset keys, ignored here.

        Returns:
            The criteria of this step, which the tracker turns into logged values.
        """
        self.model.train()
        loss = self.criterion(self.model(x), y)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return {"loss": loss.detach(), "accuracy": self._accuracy(x, y)}

    @torch.no_grad()
    def inference_step(self, x: torch.Tensor, y: torch.Tensor, **kwargs: Any) -> dict[str, torch.Tensor]:
        """Run one validation batch, returning the same criteria as the training step.

        Args:
            x: The input features of the batch.
            y: The target labels of the batch.
            **kwargs: Further dataset keys, ignored here.

        Returns:
            The criteria of this step; the trainer prefixes their names with `val_`.
        """
        self.model.eval()
        return {"loss": self.criterion(self.model(x), y), "accuracy": self._accuracy(x, y)}

    @torch.no_grad()
    def _accuracy(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Return the fraction of correctly classified rows of the batch."""
        return (self.model(x).argmax(dim=1) == y).to(torch.float32).mean()

    def on_epoch_end(self, info: BaseInfo, **models: torch.nn.Module) -> None:
        """Advance the learning-rate schedule once the epoch is over.

        The trainer found this method by checking the learner against the `OnEpochEnd` protocol --
        no registration call, no global registry.

        Args:
            info: The trainer itself, exposing `epoch`, `step`, `update`, and `logs()`.
            **models: The models of the learner, by name.
        """
        self.scheduler.step()


def track(**criteria: torch.Tensor) -> dict[str, float]:
    """Turn the criteria of one step into plain floats.

    The trainer calls the tracker once per step with the dictionary the step returned, and writes
    the result into the logs of the current epoch. This one reports the value of the last step;
    `structcast_model.torch.trainer.TorchTracker` averages over the epoch instead.

    Args:
        **criteria: The criteria of the step that just ran.

    Returns:
        The criteria as floats, keyed by the same names.
    """
    return {name: value.item() for name, value in criteria.items()}


def main() -> None:
    """Build every participant, fit for three epochs, and report the best validation loss."""
    torch.manual_seed(SEED)
    generator = torch.Generator().manual_seed(SEED)
    model = torch.nn.Sequential(
        torch.nn.Linear(FEATURES, 32),
        torch.nn.ReLU(),
        torch.nn.Linear(32, CLASSES),
    )

    # A data provider carries both datasets for the whole run, so `fit()` needs no dataset argument.
    data = SimpleDataProvider(
        training_dataset=make_dataset(BATCHES, generator),
        validation_dataset=make_dataset(BATCHES // 4, generator),
    )

    # `BestCriterion` watches one criterion; validation criteria carry the `val_` prefix.
    best = BestCriterion[torch.nn.Module](target="val_loss", mode="min")

    # Every participant -- learner, its optimizers, tracker, data provider, then the callbacks in
    # the order given -- is scanned once here and routed into the events whose protocol it
    # implements.
    trainer = TorchTrainer(
        device="cpu",
        learner=SimpleLearner(model),
        tracker=track,
        data=data,
        callbacks=[Printer(), best],
    )
    print(f"Registered callbacks: {trainer.describe()}\n")

    trainer.fit(epochs=3)

    print(f"\nBest {best.target}: {best.value:.4f} at step {best.step}")


if __name__ == "__main__":
    main()
