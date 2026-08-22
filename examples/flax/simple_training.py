r"""A complete, runnable Flax (nnx) training program written directly against the trainer API.

Everything the CLI wires up from YAML is written by hand here, so the moving parts stay visible:

* a **Learner** owning the model, the optimizer, the differentiated flow, and the two step
  definitions;
* a **DataProvider** carrying the training and validation datasets;
* a **tracker** turning the criteria of one step into plain floats;
* **callbacks** -- `Printer` and `BestCriterion` -- routed into lifecycle events by protocol.

Run it standalone::

    uv run python examples/flax/simple_training.py

Or hand the same three objects to the CLI, which instantiates them from object patterns and adds
what the tutorial leaves out -- `flax.nnx.jit` over the steps, an experiment logger, and the
checkpoint savers::

    FILE=examples/flax/simple_training.py

    uv run scm flax train "model: [_obj_, {_addr_: build_model, _file_: $FILE}]" \
        --learner "[_obj_, {_addr_: SimpleLearner, _file_: $FILE}]" \
        --training-dataset "[_obj_, {_addr_: make_dataset, _file_: $FILE}, {_call_: {batches: 20, seed: 0}}]" \
        --validation-dataset "[_obj_, {_addr_: make_dataset, _file_: $FILE}, {_call_: {batches: 5, seed: 1}}]" \
        -s "x: [8]" -d cpu:0 --epochs 3 --ci -LC val_loss -HC val_accuracy -E simple-training

Nothing is generated for that: `_addr_` names a symbol of this file and `_file_` the path to load
it from, resolved from the working directory, so run it from the repository root. The model pattern
carries no `_call_`: the command calls the factory itself, with the run's seeded `flax.nnx.Rngs`.
`--logger` needs the `mlflow` extra (the default) or the `wandb` one.

Both entry points train a two-layer MLP on a synthetic, deterministic dataset and finish in a few
seconds on the CPU.
"""

from typing import Any

import jax
import optax

from flax import nnx
from structcast_model.base_trainer import BestCriterion, Printer, SimpleDataProvider
from structcast_model.flax.optimizers import get_learning_rate, gradient_steps
from structcast_model.flax.trainer import FlaxTrainer

# The datasets below are built once from this seed, so every run sees the same arrays.
SEED = 0
BATCHES = 20
BATCH_SIZE = 32
FEATURES = 8
CLASSES = 3


def make_dataset(batches: int, seed: int) -> list[dict[str, jax.Array]]:
    """Build a synthetic classification dataset as a list of input dictionaries.

    A dataset is anything iterable that yields dictionaries; the keys become the keyword arguments
    of `training_step` and `inference_step`. A list of pre-built batches is the simplest form and
    needs no downloads.

    The seed, not a key, is the argument, so `--training-dataset` and `--validation-dataset` can
    call this function with plain YAML values and still get two different datasets.

    Args:
        batches: Number of batches to build.
        seed: Seed of the key producing the arrays.

    Returns:
        One dictionary per batch, each with an `x` feature array and a `y` label array.
    """
    keys = jax.random.split(jax.random.key(seed), batches)
    dataset = []
    for key in keys:
        x = jax.random.normal(key, (BATCH_SIZE, FEATURES))
        # A learnable rule: the label is the argmax over the first CLASSES features.
        y = x[:, :CLASSES].argmax(axis=1)
        dataset.append({"x": x, "y": y})
    return dataset


def build_model(*, rngs: nnx.Rngs) -> nnx.Module:
    """Build the two-layer MLP both entry points train.

    A module-level factory taking `rngs`, so `scm flax train` can address it with `_addr_`/`_file_`
    and call it with the run's seeded RNG streams -- which is how a generated model class is built
    too, and why the pattern must not carry a `_call_` of its own.
    """
    return nnx.Sequential(
        nnx.Linear(FEATURES, 32, rngs=rngs),
        nnx.relu,
        nnx.Linear(32, CLASSES, rngs=rngs),
    )


class SimpleLearner:
    """A hand-written learner: it owns the model and the optimizer and defines both steps.

    This is the object the redesign asks you to customize per model. It implements the `Learner`
    protocol -- the `models`, `optimizers`, `optimizer_models`, `flow_functions`, `learning_rates`,
    `steps`, `updates`, and `has_updated` properties plus `restore_counters`, `training_step`, and
    `inference_step` -- and it follows the conventions a generated Flax learner follows, which is
    what lets the CLI compile and donate it (`docs/adr/0019`). Extra methods named after a lifecycle
    event are picked up by the trainer automatically; there is none here, because the learning-rate
    schedule below is an optax schedule counting updates rather than something an epoch hook steps.
    """

    def __init__(self, model: nnx.Module, learning_rate: float = 0.1, decay: float = 0.5) -> None:
        """Create the learner over *model*, with its own optimizer, flow function, and schedule.

        Args:
            model: The model to train. `scm flax train` passes it under the name of its model
                pattern, which is why this parameter is called `model`.
            learning_rate: The initial learning rate of the SGD optimizer.
            decay: Factor the rate is multiplied by after every epoch of the tutorial's dataset.
        """
        self.model = model
        # `inject_hyperparams` is what makes the rate readable at run time: optax otherwise keeps a
        # constant in the update closure and a schedule's value nowhere at all. `FlaxLearnerBuilder`
        # wraps a template's optimizer pattern in it for the same reason (`docs/adr/0013`).
        schedule = optax.exponential_decay(learning_rate, transition_steps=BATCHES, decay_rate=decay, staircase=True)
        self.optimizer = nnx.Optimizer(
            model, tx=optax.inject_hyperparams(optax.sgd)(learning_rate=schedule), wrt=nnx.Param
        )

        def _flow_optimizer(model: nnx.Module, x: jax.Array, y: jax.Array) -> tuple[jax.Array, tuple[jax.Array, ...]]:
            """Compute the criteria of one batch: the whole graph, and nothing but the graph.

            Free variables only -- never `self` -- so the traced graph is the computation and not
            the learner object around it. The model is a parameter because it is what the enclosing
            step differentiates: its position is the `argnums` of `nnx.value_and_grad`.
            """
            logits = model(x)
            loss = optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()
            return loss, (loss, (logits.argmax(axis=-1) == y).mean())

        def _training_step(model: nnx.Module, optimizer: nnx.Optimizer, *, x: Any, y: Any, **kwargs: Any) -> Any:
            """Differentiate the flow, apply the optimizer, and report what the caller reads back.

            The signature is the donation contract: every model and optimizer is a
            positional-or-keyword parameter and the batch is keyword-only, which is exactly what
            `scm flax train` reads back to decide what `flax.nnx.jit` may donate.
            """
            (_, (loss, accuracy)), grads = nnx.value_and_grad(_flow_optimizer, has_aux=True)(model, x=x, y=y)
            # Read across the update: with an `optax.MultiSteps` window the device decides which
            # step an update lands on. There is none here, so `gradient_steps` reports None and
            # every step applies -- the same code covers both cases.
            before = gradient_steps(optimizer)
            optimizer.update(model, grads)
            after = gradient_steps(optimizer)
            if before is None or after is None:
                # No window at all, so there is no counter to compare and every step applies.
                has_updated: Any = True
            else:
                has_updated = after > before
            # Read inside the step, at trace time: reading the rate afterwards would touch the
            # optimizer buffers a compiled step was handed and is allowed to have donated.
            return {"loss": loss, "accuracy": accuracy}, {"optimizer": get_learning_rate(optimizer)}, has_updated

        def _inference_step(model: nnx.Module, *, x: Any, y: Any, **kwargs: Any) -> dict[str, jax.Array]:
            """Run one validation batch through the same flow, returning the same criteria."""
            _, (loss, accuracy) = _flow_optimizer(model, x, y)
            return {"loss": loss, "accuracy": accuracy}

        # Bound as attributes rather than written as methods, exactly as a generated learner binds
        # its steps: `scm flax train` rebinds every name of `flow_functions` with
        # `setattr(learner, name, nnx.jit(...))`, and the public steps below call them back through
        # `self`, so the compiled version is what runs.
        self._training_step = _training_step
        self._inference_step = _inference_step
        # An inference view shares its arrays with the trained model but reports itself as not
        # training, which is what turns dropout and batch-norm updates off. This MLP has neither;
        # the view is here because every learner needs one the moment a model does.
        self._view_model = nnx.view(
            model, raise_if_not_found=False, training=False, deterministic=True, use_running_average=True
        )
        # The criteria both steps return. The CLI reads them off the learner to build the tracker
        # and the progress-bar rows, unless `--learner-outputs` overrides them.
        self.outputs = ["loss", "accuracy"]
        # The learner owns the training counters: the trainer only reads them (docs/adr/0018).
        self._steps = 0
        self._updates = 0
        self._has_updated = False
        self._learning_rates: dict[str, Any] = {"optimizer": float("nan")}

    @property
    def models(self) -> dict[str, nnx.Module]:
        """The models to train, by name. The trainer exposes them to every callback as `info.models`."""
        return {"model": self.model}

    @property
    def optimizers(self) -> dict[str, Any]:
        """The optimizers by name: the trainer scans them for event protocols right after the learner."""
        return {"optimizer": self.optimizer}

    @property
    def optimizer_models(self) -> dict[str, list[str]]:
        """The models each optimizer updates, so checkpointing can key its state by model name."""
        return {"optimizer": ["model"]}

    @property
    def flow_functions(self) -> dict[str, Any]:
        """The compile units of this learner: the attributes holding its steps, by name.

        Unlike the torch side, where the flow inside the step is compiled (`docs/adr/0004`), a Flax
        run compiles the whole step: `nnx.jit` owns the state transfer, so the optimizer apply
        belongs inside the traced region. The CLI walks this mapping and rebinds each name with the
        compiled wrapper, donating the state parameters of `_training_step` alone.
        """
        return {"_training_step": self._training_step, "_inference_step": self._inference_step}

    @property
    def learning_rates(self) -> dict[str, float]:
        """The current learning rate per optimizer, shown by `Printer` next to the criteria."""
        return {name: float(rate) for name, rate in self._learning_rates.items()}

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
        learner whose optimizer wraps its transformation in `optax.MultiSteps` would report True
        only on the step the device applies, and the trainer would fire `on_update` that often.
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

    def training_step(self, x: Any, y: Any, **kwargs: Any) -> dict[str, jax.Array]:
        """Run one training batch: the counters on the host, everything else inside the step.

        Args:
            x: The input features of the batch.
            y: The target labels of the batch.
            **kwargs: Further dataset keys, ignored here.

        Returns:
            The criteria of this step, which the tracker turns into logged values.
        """
        self._steps += 1
        criteria, learning_rates, has_updated = self._training_step(self.model, self.optimizer, x=x, y=y, **kwargs)
        self._learning_rates = learning_rates
        self._has_updated = bool(has_updated)
        self._updates += int(self._has_updated)
        return criteria

    def inference_step(self, x: Any, y: Any, **kwargs: Any) -> dict[str, jax.Array]:
        """Run one validation batch against the inference view of the model.

        Args:
            x: The input features of the batch.
            y: The target labels of the batch.
            **kwargs: Further dataset keys, ignored here.

        Returns:
            The criteria of this step; the trainer prefixes their names with `val_`.
        """
        return self._inference_step(self._view_model, x=x, y=y, **kwargs)


def track(**criteria: jax.Array) -> dict[str, float]:
    """Turn the criteria of one step into plain floats.

    The trainer calls the tracker once per step with the dictionary the step returned, and writes
    the result into the logs of the current epoch. This one reports the value of the last step;
    `structcast_model.flax.trainer.FlaxTracker` averages over the epoch instead.

    Args:
        **criteria: The criteria of the step that just ran.

    Returns:
        The criteria as floats, keyed by the same names.
    """
    return {name: float(value) for name, value in jax.device_get(criteria).items()}


def main() -> None:
    """Build every participant, fit for three epochs, and report the best validation loss."""
    model = build_model(rngs=nnx.Rngs(params=jax.random.key(SEED), dropout=jax.random.key(SEED + 1)))

    # A data provider carries both datasets for the whole run, so `fit()` needs no dataset argument.
    data = SimpleDataProvider(
        training_dataset=make_dataset(BATCHES, SEED),
        validation_dataset=make_dataset(BATCHES // 4, SEED + 1),
    )

    # `BestCriterion` watches one criterion; validation criteria carry the `val_` prefix.
    best = BestCriterion[nnx.Module](target="val_loss", mode="min")

    # Every participant -- learner, its optimizers, tracker, data provider and its datasets, then
    # the callbacks in the order given -- is scanned once on first use and routed into the events
    # whose protocol it implements.
    trainer = FlaxTrainer(
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
