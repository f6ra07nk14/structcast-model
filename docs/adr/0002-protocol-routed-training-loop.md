# The training loop is wired by protocol routing, not a global registry

This ADR records the decisions behind the redesign of `src/structcast_model/base_trainer.py` and its PyTorch
consumers: callback registration, the `Learner` rename, construction-time datasets, per-learner optimizer
ownership, logger-owned run lifecycle, and the learner factory boundary.

## Callbacks are routed by protocol, not registered globally

Callbacks used to reach a trainer through two channels: a process-global `GLOBAL_CALLBACKS` registry that library
internals mutated as a side effect of object construction (scoped by the `callbacks_session` context manager), and
per-event `NamedCallbackList.register()` calls on the trainer instance. The global registry leaked state across
runs, made registration order depend on object construction order, and hid who attached what.

Both channels are replaced by one: a trainer receives participant objects at construction and routes each into
lifecycle events by checking which of the eleven event protocols (`OnUpdate`, `OnTrainingBegin`, …,
`OnEpochEnd`) the object implements. The scan order is fixed — learner, the learner's optimizers, tracker, data
provider, then the explicit `callbacks` sequence in the order given — and the same object is never registered
twice for the same event. A dual-track design (keeping `register()` for ad-hoc lambdas alongside protocol routing)
was rejected: it means two registration mechanisms to learn and maintain, for a single in-repo consumer
(`cmd_torch.py`) whose lambdas collapse naturally into small callback classes (`ProgressBar`, `Printer`).

The protocols are `runtime_checkable`, so `isinstance` only verifies that a method with the right name exists —
it does not inspect the signature (the same accepted weakness as `TensorSpec`'s `TensorInitializer` gate in
ADR-0001). An object with an unrelated `on_update` attribute would be picked up; this is accepted.

## `Backward` is renamed `Learner`

The protocol formerly called `Backward` owns the models and defines when to update, how a training step runs, and
how an inference step runs — far more than a backward pass. It is renamed `Learner`, following precedent in both
supervised learning (fastai's `Learner`) and reinforcement learning (ACME's Learner), keeping the door open for
RL-style learners whose batches come from rollouts rather than datasets. `TrainingModule` (Lightning familiarity,
but "Module" is overloaded next to `torch.nn.Module`), `Optimization`, `StepRunner`, and `LearningProcedure` were
considered and rejected for weaker cross-paradigm fit. The rename cascades everywhere — runtime, CLI options and
subcommands, builder and schema names, configuration keys and the `cfg/torch/learners/` directory — with no
backward-compatibility aliases; the terminology split of a partial rename was judged worse than one breaking
release.

## Datasets arrive at construction through a `DataProvider`

`fit()` used to receive the training and validation datasets on every call. They now come from a `DataProvider`
(training dataset plus optional validation dataset — `None` skips validation) given to the trainer at
construction, so a fully wired trainer is one object and the data side's epoch-synchronization hooks participate
in the same protocol scan as everything else. `fit()` keeps only loop parameters (epochs, start epoch, validation
frequency) and feeds the provider's datasets into `train(dataset)` / `evaluate(dataset)`, whose signatures are
deliberately unchanged so they remain usable standalone. The CLI keeps its two dataset options
(`--training-dataset` / `--validation-dataset`) and composes the provider internally; a single provider pattern
was rejected because it complicates configuration files and loses the ability to combine splits freely at run
time.

## Optimizer and scheduler composition lives with the learner

`create_with_scheduler` built optimizers and registered scheduler steps into the global registry as a side
effect, forcing one package module (`torch/optimizers.py`) to anticipate every optimizer/scheduler combination.
That side channel is deleted. What stays in the package is `create_opt`: regex-based weight-decay and layer-decay
parameter grouping, applied before the optimizer engine, over two engines — native `torch.optim` classes and
`timm.optim.create_optimizer_v2`. Grouping happens on named parameters and emits standard parameter groups, so it
is engine-agnostic by construction.

Optimizer+scheduler combinations (e.g. `AdamWWithCosine`) are example code under `examples/torch/`, referenced
from configuration by file path — they are use-case-specific compositions, not package API. Such a combination is
a transparent proxy: it delegates the `Optimizer` interface to the wrapped optimizer via `__getattr__` (generated
learner code keeps calling `step()` / `zero_grad()` / `param_groups` unchanged, and `GradScaler.step()` works
through the delegation), implements event protocols so the trainer's optimizer scan steps the scheduler at the
right moment, and merges optimizer and scheduler state into one `state_dict` — schedule state was silently lost
on resume under the global-registry design.

## The run lifecycle belongs to the Logger object

`mlflow.start_run` / `wandb.init` happen once per `fit()`, but no event fires exactly once per fit —
`on_training_begin` fires every epoch. Rather than growing the event set with `on_fit_begin` / `on_fit_end`
(which also entangles run teardown with interrupt-time state saving), a Logger is a context manager that owns the
run — start, parameter logging, teardown — and additionally implements event protocols for per-epoch metric
logging. The CLI selects the backend with `--logger mlflow|wandb` (default `mlflow`); state saving and
best-criterion recording go through the Logger interface so both backends receive them.

## The learner factory excludes the tracker and DDP wrapping

The factory that builds a Learner from object patterns covers model instantiation, input-shape resolution,
initializers, learner construction, and step-function compilation. It deliberately does not create the tracker
(a metrics concern, and a separate trainer field) and does not apply `DistributedDataParallel` wrapping, which
stays in `cmd_torch.py`. Consequence, recorded as a known limitation rather than fixed here: models are
DDP-wrapped after the learner captures them, so the learner's step closures call the raw modules and
`TorchTrainer.no_sync` never sees a DDP instance — distributed gradient synchronization does not actually flow
through the wrapper.
