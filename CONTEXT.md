# StructCast Model

Configuration-driven toolkit for generating models and training workflows from YAML templates, with a
framework-agnostic training loop specialized for PyTorch, Keras, and Flax.

## Language

### Training loop

**Learner**:
The object that owns the models being trained and defines how they learn: when an update should happen, how a
training step runs, and how an inference step runs.
_Avoid_: Backward, backward class, backward pass configuration

**Trainer**:
Runs the training loop — epochs, steps, and validation — over a Learner, dispatching lifecycle events to callbacks.

**Step**:
One iteration over a single batch of inputs during training or validation.

**Update**:
One application of the Learner's optimizers. With gradient accumulation, several steps may pass between updates.

**Epoch**:
One full pass over the training dataset, optionally followed by validation.

**Event**:
A named moment in the training lifecycle at which callbacks run: update, training begin/end, training step
begin/end, validation begin/end, validation step begin/end, epoch begin/end.

**Callback**:
An object that reacts to one or more events. Which events it receives is determined by which event protocols it
implements.
_Avoid_: hook, listener

**Criterion**:
A named scalar produced by a training or inference step — a loss or a metric — tracked across steps.
(Plural: criteria.)

**Tracker**:
Averages per-step criteria into per-epoch metric values.
_Avoid_: logger

**Logger**:
Records a training run to an experiment-tracking service (MLflow, wandb): parameters, metrics, artifacts, and
model state.
_Avoid_: tracker

**Best criterion**:
Monitors one criterion for its best value seen so far and notifies its on-best participants after every epoch
that produced the criterion.

### Data

**Dataset**:
An iterable of input dictionaries consumed by training or validation steps, or a callable returning such an
iterable.

**DataProvider**:
Supplies the training dataset, the optional validation dataset, and their step counts (`steps_per_epoch`,
`validation_steps`) for a whole training run; given to a Trainer at construction, which scans the provider and its
datasets for event protocols. The dataset properties must return the same object on every read.
_Avoid_: DataModule, dataset wrapper

### Configuration

**Object pattern**:
A YAML/CLI expression describing how to instantiate an object (`_obj_` / `_addr_` / `_call_`).

