r"""A complete, runnable Keras training program written directly against the trainer API.

Everything the CLI wires up from YAML is written by hand here, so the moving parts stay visible:

* a **Learner** owning the model, the optimizer segment, the two flows and the two step definitions;
* a **DataProvider** carrying the training and validation datasets;
* a **tracker** turning the criteria of one step into plain floats;
* **callbacks** -- `Printer` and `BestCriterion` -- routed into lifecycle events by protocol.

Run it standalone::

    uv run python examples/keras/simple_training.py

Or hand the same three objects to the CLI, which instantiates them from object patterns and adds
what the tutorial leaves out -- an experiment logger and the checkpoint savers::

    FILE=examples/keras/simple_training.py

    uv run scm keras train "model: [_obj_, {_addr_: build_model, _file_: $FILE}, _call_]" \
        --backend tensorflow \
        --learner "[_obj_, {_addr_: SimpleLearner, _file_: $FILE}]" \
        --training-dataset "[_obj_, {_addr_: make_dataset, _file_: $FILE}, {_call_: {batches: 20, seed: 0}}]" \
        --validation-dataset "[_obj_, {_addr_: make_dataset, _file_: $FILE}, {_call_: {batches: 5, seed: 1}}]" \
        --epochs 3 --ci -LC val_loss -HC val_accuracy -E simple-training

Nothing is generated for that: `_addr_` names a symbol of this file and `_file_` the path to load
it from, resolved from the working directory, so run it from the repository root. `--logger` needs
the `mlflow` extra (the default) or the `wandb` one.

Both entry points train a two-layer MLP on a synthetic, deterministic dataset and finish in a few
seconds on the CPU, on whichever backend `KERAS_BACKEND` selects.
"""

from typing import Any

import numpy as np

import keras
from structcast_model.base_trainer import BaseInfo, BestCriterion, Printer, SimpleDataProvider
from structcast_model.keras.adapters import AdapterSegment, select_backend_adapter
from structcast_model.keras.trainer import KerasTrainer

# The datasets below are built once from this seed, so every run sees the same arrays.
SEED = 0
BATCHES = 20
BATCH_SIZE = 32
FEATURES = 8
CLASSES = 3


def make_dataset(batches: int, seed: int) -> list[dict[str, np.ndarray]]:
    """Build a synthetic classification dataset as a list of input dictionaries.

    A dataset is anything iterable that yields dictionaries; the keys become the keyword arguments
    of `training_step` and `inference_step`. A list of pre-built batches is the simplest form and
    needs no downloads. The arrays are NumPy, which every Keras backend accepts.

    The seed, not a generator, is the argument, so `--training-dataset` and `--validation-dataset`
    can call this function with plain YAML values and still get two different datasets.

    Args:
        batches: Number of batches to build.
        seed: Seed of the generator producing the arrays.

    Returns:
        One dictionary per batch, each with an `x` feature array and a `y` label array.
    """
    generator = np.random.default_rng(seed)
    dataset = []
    for _ in range(batches):
        x = generator.standard_normal((BATCH_SIZE, FEATURES)).astype("float32")
        # A learnable rule: the label is the argmax over the first CLASSES features.
        dataset.append({"x": x, "y": x[:, :CLASSES].argmax(axis=1).astype("int64")})
    return dataset


def build_model() -> keras.Model:
    """Build the two-layer MLP both entry points train.

    A module-level factory, so `scm keras train` can address it with `_addr_`/`_file_` and pass the
    result to the learner under the name of its model pattern. It returns a built `keras.Model`
    rather than a bare layer, because a layer owns no variable until it is traced -- which is what
    the CLI's `--shape` does for a generated layer, and what `keras.Input` does here.
    """
    inputs = keras.Input(shape=(FEATURES,), name="x")
    hidden = keras.layers.Dense(32, activation="relu")(inputs)
    return keras.Model(inputs=inputs, outputs=keras.layers.Dense(CLASSES)(hidden))


class SimpleLearner:
    """A hand-written learner: it owns the model and the optimizer and defines both steps.

    This is the object the redesign asks you to customize per model. It implements the `Learner`
    protocol -- the `models`, `optimizers`, `optimizer_models`, `flow_functions`, `learning_rates`,
    `steps`, `updates`, and `has_updated` properties plus `restore_counters`, `training_step`, and
    `inference_step`. Extra methods named after a lifecycle event, such as `on_epoch_end` below,
    are picked up by the trainer automatically.

    It names no backend, exactly as a generated learner does: the flows below are written in
    `keras.ops`, and the adapter selected in `__init__` owns the gradients, the optimizer
    application and the step compilation.
    """

    def __init__(
        self, model: keras.Model, learning_rate: float = 0.1, gradient_accumulation_steps: int | None = None
    ) -> None:
        """Create the learner over *model*, with its own optimizer, flows, and compiled steps.

        Args:
            model: The model to train. `scm keras train` passes it under the name of its model
                pattern, which is why this parameter is called `model`.
            learning_rate: The initial learning rate of the SGD optimizer.
            gradient_accumulation_steps: Batches to accumulate before one update, or None for one
                update per batch. Accumulation is the Keras optimizer's own, which is why it is a
                keyword here and nothing below has to know about it.
        """
        self.model = model
        self.loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        # One segment: this optimizer, these variables, this flow. A learner training two models
        # under two optimizers -- a GAN, say -- hands the adapter one segment per optimizer.
        self._segment = AdapterSegment(
            name="optimizer",
            flow=self._flow_optimizer,
            optimizer=keras.optimizers.SGD(
                learning_rate=learning_rate, gradient_accumulation_steps=gradient_accumulation_steps
            ),
            variables=list(model.trainable_variables),
            models=[model],
        )
        adapter = select_backend_adapter()
        # `prepare` builds the optimizer against those variables, and it has to run before the step
        # is compiled: JAX refuses an unbuilt optimizer inside a jitted step.
        adapter.prepare([self._segment])
        self._training_step = adapter.build_train_step([self._segment])
        self._inference_step = adapter.build_inference_step(self._flow_inference, models=[model])
        # The criteria both steps return. The CLI reads them off the learner to build the tracker
        # and the progress-bar rows, unless `--learner-outputs` overrides them.
        self.outputs = ["loss", "accuracy"]
        # The learner owns the training counters: the trainer only reads them (docs/adr/0018).
        self._steps = 0
        self._last_updates = 0
        self._has_updated = False

    def _flow_optimizer(self, *, x: Any, y: Any) -> tuple[Any, dict[str, Any]]:
        """Compute the loss to differentiate and the criteria of one training batch.

        Keyword-only, as every generated flow is: the adapters and the distributed strategies pass
        the batch by name, and a positional batch would bind the entries in declaration order.
        """
        logits = self.model(x, training=True)
        loss = self.loss(y_true=y, y_pred=logits)
        accuracy = keras.ops.mean(keras.metrics.sparse_categorical_accuracy(y_true=y, y_pred=logits))
        return loss, {"loss": loss, "accuracy": accuracy}

    def _flow_inference(self, *, x: Any, y: Any) -> dict[str, Any]:
        """Compute the same criteria with the model in inference mode, differentiating nothing."""
        logits = self.model(x, training=False)
        return {
            "loss": self.loss(y_true=y, y_pred=logits),
            "accuracy": keras.ops.mean(keras.metrics.sparse_categorical_accuracy(y_true=y, y_pred=logits)),
        }

    @property
    def models(self) -> dict[str, keras.Model]:
        """The models to train, by name. The trainer exposes them to every callback as `info.models`."""
        return {"model": self.model}

    @property
    def optimizers(self) -> dict[str, Any]:
        """The optimizers by name, read through the segment.

        Through the segment, not through a dictionary built in `__init__`: under a float16 policy
        `prepare` replaces the optimizer with the `LossScaleOptimizer` wrapping it, and a stored
        reference would report the one that never applied anything.
        """
        return {"optimizer": self._segment.optimizer}

    @property
    def optimizer_models(self) -> dict[str, list[str]]:
        """The models each optimizer updates, so checkpointing can key its state by model name."""
        return {"optimizer": ["model"]}

    @property
    def flow_functions(self) -> dict[str, Any]:
        """The compiled steps of this learner, by attribute name.

        A Keras run compiles nothing on top of these -- the backend adapter already built them with
        `tf.function` or `jax.jit` -- so this is what a generated Keras learner reports too, and it
        is the mapping a distributed strategy rebinds when it replicates a step across devices.
        """
        return {"_training_step": self._training_step, "_inference_step": self._inference_step}

    @property
    def learning_rates(self) -> dict[str, float]:
        """The current learning rate per optimizer, shown by `Printer` next to the criteria."""
        return {name: float(keras.ops.convert_to_numpy(o.learning_rate)) for name, o in self.optimizers.items()}

    @property
    def steps(self) -> int:
        """The number of training steps completed so far, counted by this learner."""
        return self._steps

    @property
    def updates(self) -> int:
        """The number of optimizer applies completed so far, read off the optimizer's own counter."""
        return self._last_updates

    @property
    def has_updated(self) -> bool:
        """Report whether the step that just finished applied the optimizer.

        True after every step here means "no gradient accumulation": one step, one update. An
        optimizer built with `gradient_accumulation_steps=N` reports True only after every N-th
        step, and the trainer fires `on_update` that often -- without a line changing here, because
        this is a read of what happened, not a prediction of what will.
        """
        return self._has_updated

    def restore_counters(self, steps: int, updates: int) -> None:
        """Seed the counters after a checkpoint restore; a fresh run never calls this.

        Args:
            steps: The number of completed training steps read back from the checkpoint.
            updates: Ignored: the restored optimizer variables already carry the apply count, and
                re-reading it here keeps the two sources from ever disagreeing.
        """
        self._steps = steps
        self._last_updates = self._iterations()

    def _iterations(self) -> int:
        """Read the optimizer's completed-update count back onto the host."""
        optimizer = getattr(self._segment.optimizer, "inner_optimizer", self._segment.optimizer)
        return int(keras.ops.convert_to_numpy(optimizer.iterations))

    def training_step(self, x: Any, y: Any, **kwargs: Any) -> dict[str, Any]:
        """Run one training batch through the compiled step and update the counters.

        The step the adapter built owns the gradients and the optimizer application; what is left
        here is the host-side bookkeeping, which a compiled region could not hold anyway.

        Args:
            x: The input features of the batch.
            y: The target labels of the batch.
            **kwargs: Further dataset keys, ignored here.

        Returns:
            The criteria of this step, which the tracker turns into logged values.
        """
        self._steps += 1
        criteria = self._training_step(x=x, y=y)
        # A post-step read, not a prediction: the optimizer has applied by now, so a skipped update
        # -- a float16 loss-scale overflow, an accumulation window still filling -- reads back as
        # a counter that did not move, and `has_updated` says so truthfully.
        current = self._iterations()
        self._has_updated = current > self._last_updates
        self._last_updates = current
        return criteria

    def inference_step(self, x: Any, y: Any, **kwargs: Any) -> dict[str, Any]:
        """Run one validation batch through the compiled inference step, returning the same criteria.

        Args:
            x: The input features of the batch.
            y: The target labels of the batch.
            **kwargs: Further dataset keys, ignored here.

        Returns:
            The criteria of this step; the trainer prefixes their names with `val_`.
        """
        return self._inference_step(x=x, y=y)

    def on_epoch_end(self, info: BaseInfo[Any]) -> None:
        """Halve the learning rate once the epoch is over.

        The trainer found this method by checking the learner against the `OnEpochEnd` protocol --
        no registration call, no global registry. A Keras schedule
        (`keras.optimizers.schedules.CosineDecay` and friends) lives inside the optimizer and counts
        steps instead, which is what the learner templates under `cfg/keras/learners` use; assigning
        the rate here is the per-epoch alternative.

        Args:
            info: The trainer itself, exposing `epoch`, `step`, `update`, `logs()`, and `models`.
        """
        self._segment.optimizer.learning_rate = self.learning_rates["optimizer"] * 0.5


def track(**criteria: Any) -> dict[str, float]:
    """Turn the criteria of one step into plain floats.

    The trainer calls the tracker once per step with the dictionary the step returned, and writes
    the result into the logs of the current epoch. This one reports the value of the last step;
    `structcast_model.keras.trainer.KerasTracker` averages over the epoch instead.

    Args:
        **criteria: The criteria of the step that just ran.

    Returns:
        The criteria as floats, keyed by the same names.
    """
    return {name: float(keras.ops.convert_to_numpy(value)) for name, value in criteria.items()}


def main() -> None:
    """Build every participant, fit for three epochs, and report the best validation loss."""
    # Seeds Python, NumPy and the active backend at once, so the model initialization is fixed.
    keras.utils.set_random_seed(SEED)
    model = build_model()

    # A data provider carries both datasets for the whole run, so `fit()` needs no dataset argument.
    data = SimpleDataProvider(
        training_dataset=make_dataset(BATCHES, SEED),
        validation_dataset=make_dataset(BATCHES // 4, SEED + 1),
    )

    # `BestCriterion` watches one criterion; validation criteria carry the `val_` prefix.
    best = BestCriterion[Any](target="val_loss", mode="min")

    # Every participant -- learner, its optimizers, tracker, data provider and its datasets, then
    # the callbacks in the order given -- is scanned once on first use and routed into the events
    # whose protocol it implements.
    trainer = KerasTrainer(
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
