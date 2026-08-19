# StructCast Model

Configuration-driven toolkit for generating models and training workflows from YAML templates, with a
framework-agnostic training loop specialized for PyTorch, Keras, and Flax.

## Language

### Training loop

**Learner**:
The object that owns the models being trained and defines how they learn: when an update should happen, how a
training step runs, and how an inference step runs.
_Avoid_: Backward, backward class, backward pass configuration

**Pairing**:
The Learner's declaration of which models each optimizer updates (`optimizer_models`). What allows a training
state to key optimizer state by parameter name rather than by position.
_Avoid_: optimizer mapping, optimizer-model map

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
model state. Also retrieves a saved training state to resume from — each logger accepts only references to its
own service, or a local path.
_Avoid_: tracker

**Best criterion**:
Monitors one criterion for its best value seen so far and notifies its on-best participants after every epoch
that produced the criterion.

**Distributed strategy**:
The replaceable unit that decides how models are distributed across devices — wrapped or partitioned,
gradient-synchronized, weight-initialized, compiled (where the compile units sit) — and turned into
checkpointable state. Exactly one strategy
is active per training run; single-device training uses a strategy too, not a special case.
_Avoid_: dist_fn, wrapper function, backend

**Strategy preset**:
A named sharding-rule table (`single`, `dp`, `fsdp`) selecting how a distributed strategy partitions
parameters, optimizer state, and batches across the devices of one host.
_Avoid_: ZeRO stage, parallelism mode, sharding config

**State backend**:
The serialization component behind a Logger's training-state methods: it turns a training state into one
artifact file and back into host-memory state. Each framework supplies one; loggers default to the torch
backend.
_Avoid_: serializer, checkpoint writer

**Training state**:
The checkpoint artifact produced at epoch end — model weights, optimizer states, gradient-scaler states, and
progress metadata — sufficient to resume a training run at an epoch boundary.
_Avoid_: checkpoint dict, snapshot

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

**Optimizer segment**:
One `LEARNERS` entry of a learner template: a loss, the flow computing it, the trainable layers that
entry's optimizer owns, and that optimizer. A learner has one segment per optimizer, applied in order.
_Avoid_: optimizer block, learner behavior

