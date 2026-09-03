r"""A complete, runnable PyTorch training program written directly against the trainer API.

Everything the CLI wires up from YAML is written by hand here, so the moving parts stay visible:

* a **Learner** owning the model, the optimizer, the flow function, and the two step definitions;
* a **DataProvider** carrying the training and validation datasets;
* a **tracker** turning the criteria of one step into plain floats;
* **callbacks** -- `Printer` and `BestCriterion` -- routed into lifecycle events by protocol.

Run it standalone::

    uv run python examples/torch/simple_training.py

Or hand the same three objects to the CLI, which instantiates them from object patterns and adds
what the tutorial leaves out -- `torch.compile` over the flow function, an experiment logger, and
the checkpoint savers::

    FILE=examples/torch/simple_training.py

    uv run scm torch train "model: [_obj_, {_addr_: build_model, _file_: $FILE}, _call_]" \
        --learner "[_obj_, {_addr_: SimpleLearner, _file_: $FILE}]" \
        --training-dataset "[_obj_, {_addr_: make_dataset, _file_: $FILE}, {_call_: {batches: 20, seed: 0}}]" \
        --validation-dataset "[_obj_, {_addr_: make_dataset, _file_: $FILE}, {_call_: {batches: 5, seed: 1}}]" \
        --device cpu --epochs 3 --compile true --ci -LC val_loss -HC val_accuracy -E simple-training

Nothing is generated for that: `_addr_` names a symbol of this file and `_file_` the path to load
it from, resolved from the working directory, so run it from the repository root. `--logger` needs
the `mlflow` extra (the default) or the `wandb` one.

Both entry points train a two-layer MLP on a synthetic, deterministic dataset and finish in a few
seconds on the CPU.
"""

from typing import Any

from structcast_model.base_trainer import BaseInfo, BestCriterion, Printer, SimpleDataProvider
from structcast_model.torch.distributed import sync_gate
from structcast_model.torch.trainer import TorchTrainer
import torch

# The datasets below are built once from this seed, so every run sees the same tensors.
SEED = 0
BATCHES = 20
BATCH_SIZE = 32
FEATURES = 8
CLASSES = 3


def make_dataset(batches: int, seed: int) -> list[dict[str, torch.Tensor]]:
    """Build a synthetic classification dataset as a list of input dictionaries.

    A dataset is anything iterable that yields dictionaries; the keys become the keyword arguments
    of `training_step` and `inference_step`. A list of pre-built batches is the simplest form and
    needs no downloads.

    The seed, not a generator, is the argument, so `--training-dataset` and `--validation-dataset`
    can call this function with plain YAML values and still get two different datasets.

    Args:
        batches: Number of batches to build.
        seed: Seed of the generator producing the tensors.

    Returns:
        One dictionary per batch, each with an `x` feature tensor and a `y` label tensor.
    """
    generator = torch.Generator().manual_seed(seed)
    dataset = []
    for _ in range(batches):
        x = torch.randn(BATCH_SIZE, FEATURES, generator=generator)
        # A learnable rule: the label is the argmax over the first CLASSES features.
        y = x[:, :CLASSES].argmax(dim=1)
        dataset.append({"x": x, "y": y})
    return dataset


def build_model() -> torch.nn.Sequential:
    """Build the two-layer MLP both entry points train.

    A module-level factory, so `scm torch train` can address it with `_addr_`/`_file_` and pass the
    result to the learner under the name of its model pattern.
    """
    return torch.nn.Sequential(
        torch.nn.Linear(FEATURES, 32),
        torch.nn.ReLU(),
        torch.nn.Linear(32, CLASSES),
    )


class SimpleLearner:
    """A hand-written learner: it owns the model and the optimizer and defines both steps.

    This is the object the redesign asks you to customize per model. It implements the `Learner`
    protocol -- the `models`, `optimizers`, `optimizer_models`, `flow_functions`, `learning_rates`,
    `steps`, `updates`, and `has_updated` properties plus `restore_counters`, `training_step`, and
    `inference_step`. Extra methods named after a lifecycle event, such as `on_epoch_end` below,
    are picked up by the trainer automatically.
    """

    def __init__(self, model: torch.nn.Module, learning_rate: float = 0.1) -> None:
        """Create the learner over *model*, with its own optimizer, flow function, and schedule.

        Args:
            model: The model to train. `scm torch train` passes it under the name of its model
                pattern, which is why this parameter is called `model`.
            learning_rate: The initial learning rate of the SGD optimizer.
        """
        criterion = torch.nn.CrossEntropyLoss()
        self.model = model
        self.optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
        # Composition of an optimizer and a schedule belongs to the learner, not to the package.
        # Stepping it from `on_epoch_end` keeps this example to one object; the alternative is an
        # optimizer wrapper that implements the event protocols itself -- see
        # `examples/torch/optimizers.py` for that variant.
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=0.5)

        def _flow_optimizer(
            __need_update__: bool, x: torch.Tensor, y: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            """Compute the criteria of one batch: the whole graph, and nothing but the graph.

            Free variables only -- the model and the loss, never `self` -- so the traced graph is
            the computation and not the learner object around it.
            """
            # A plain module ignores the gate; a DDP- or FSDP2-wrapped one reads it to fire the
            # gradient synchronization on the last backward of an update.
            sync_gate(model, __need_update__)
            logits = model(x)
            return criterion(logits, y), (logits.argmax(dim=1) == y).to(torch.float32).mean()

        # Bound as an attribute rather than written as a method, exactly as a generated learner
        # binds its `_flow_<optimizer>` closures: `scm torch train --compile` rebinds every name of
        # `flow_functions` with `setattr(learner, name, torch.compile(...))`, and both steps call it
        # back through `self`, so the compiled version is what runs. A generated learner emits one
        # flow per optimizer segment plus a `_flow_inference`; one covers both steps here because
        # they compute the same thing.
        self._flow_optimizer = _flow_optimizer
        # The criteria both steps return. The CLI reads them off the learner to build the tracker
        # and the progress-bar rows, unless `--learner-outputs` overrides them.
        self.outputs = ["loss", "accuracy"]
        # The learner owns the training counters: the trainer only reads them (docs/adr/0018).
        self._steps = 0
        self._updates = 0
        self._has_updated = False

    @property
    def models(self) -> dict[str, torch.nn.Module]:
        """The models to train, by name. The trainer exposes them to every callback as `info.models`."""
        return {"model": self.model}

    @property
    def optimizers(self) -> dict[str, torch.optim.Optimizer]:
        """The optimizers by name: the trainer scans them for event protocols right after the learner."""
        return {"optimizer": self.optimizer}

    @property
    def optimizer_models(self) -> dict[str, list[str]]:
        """The models each optimizer updates, so checkpointing can key its state by parameter names."""
        return {"optimizer": ["model"]}

    @property
    def flow_functions(self) -> dict[str, Any]:
        """The compile units of this learner: the attributes holding its flow functions, by name.

        The flow functions are what `torch.compile` is applied to, never the step (ADR-0004): the
        CLI walks this mapping and rebinds each name with the compiled wrapper. Returning `{}` --
        the easy shortcut when a learner writes its steps whole -- silently opts the run out of
        that stage and leaves only the models compiled.
        """
        return {"_flow_optimizer": self._flow_optimizer}

    @property
    def learning_rates(self) -> dict[str, float]:
        """The current learning rate per optimizer, shown by `Printer` next to the criteria."""
        return {name: opt.param_groups[0]["lr"] for name, opt in self.optimizers.items()}

    @property
    def steps(self) -> int:
        """The number of training steps completed so far, counted by this learner."""
        return self._steps

    @property
    def updates(self) -> int:
        """The number of optimizer applies completed so far -- one per step here."""
        return self._updates

    @property
    def has_updated(self) -> bool:
        """Report whether the step that just finished applied the optimizer.

        True after every step here means "no gradient accumulation": one step, one update. A
        learner accumulating over N batches would report True only after every N-th step, and the
        trainer would fire `on_update` that often.
        """
        return self._has_updated

    def restore_counters(self, steps: int, updates: int) -> None:
        """Seed the counters after a checkpoint restore; a fresh run never calls this.

        Args:
            steps: The number of completed training steps read back from the checkpoint.
            updates: The number of completed optimizer applies read back from the checkpoint.
        """
        self._steps = steps
        self._updates = updates

    def training_step(self, x: torch.Tensor, y: torch.Tensor, **kwargs: Any) -> dict[str, torch.Tensor]:
        """Run one training batch: the flow function, then backward and the optimizer.

        The step stays eager on purpose. It owns the host-side counters and the optimizer calls,
        which a compiled region could only turn into graph breaks and guards; the tensor work sits
        in the flow function instead, which is the unit the CLI compiles (ADR-0004, docs/adr/0018).
        Calling it through the attribute is what makes that compilation take effect.

        Args:
            x: The input features of the batch.
            y: The target labels of the batch.
            **kwargs: Further dataset keys, ignored here.

        Returns:
            The criteria of this step, which the tracker turns into logged values.
        """
        self._steps += 1
        self.model.train()
        # True: every step applies the optimizer here, so every backward must synchronize.
        loss, accuracy = self._flow_optimizer(True, x, y)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        self._updates += 1
        self._has_updated = True
        return {"loss": loss.detach(), "accuracy": accuracy}

    @torch.no_grad()
    def inference_step(self, x: torch.Tensor, y: torch.Tensor, **kwargs: Any) -> dict[str, torch.Tensor]:
        """Run one validation batch through the same flow, returning the same criteria.

        `False` disarms the gradient-sync gate: no backward follows, so there is nothing to
        synchronize.

        Args:
            x: The input features of the batch.
            y: The target labels of the batch.
            **kwargs: Further dataset keys, ignored here.

        Returns:
            The criteria of this step; the trainer prefixes their names with `val_`.
        """
        self.model.eval()
        loss, accuracy = self._flow_optimizer(False, x, y)
        return {"loss": loss, "accuracy": accuracy}

    def on_epoch_end(self, info: BaseInfo) -> None:
        """Advance the learning-rate schedule once the epoch is over.

        The trainer found this method by checking the learner against the `OnEpochEnd` protocol --
        no registration call, no global registry.

        Args:
            info: The trainer itself, exposing `epoch`, `step`, `update`, `logs()`, and `models`.
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
    model = build_model()

    # A data provider carries both datasets for the whole run, so `fit()` needs no dataset argument.
    data = SimpleDataProvider(
        training_dataset=make_dataset(BATCHES, SEED),
        validation_dataset=make_dataset(BATCHES // 4, SEED + 1),
    )

    # `BestCriterion` watches one criterion; validation criteria carry the `val_` prefix.
    best = BestCriterion[torch.nn.Module](target="val_loss", mode="min")

    # Every participant -- learner, its optimizers, tracker, data provider and its datasets, then
    # the callbacks in the order given -- is scanned once on first use and routed into the events
    # whose protocol it implements.
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
