# Examples

Two ways to build the same training program, plus the integrations a configuration loads by file path:

| File                                             | What it shows                                                             |
| ------------------------------------------------ | ------------------------------------------------------------------------- |
| [`torch/simple_training.py`](torch/simple_training.py) | A complete training program written by hand, run standalone or by `scm torch train` |
| [`torch/optimizers.py`](torch/optimizers.py)     | Optimizer + scheduler compositions referenced from YAML by file path      |
| [`torch/data.py`](torch/data.py)                 | timm dataset and dataloader wrappers, referenced from YAML by file path   |
| [`torch/corpus.py`](torch/corpus.py)             | A character-level text corpus, referenced from the CLI by file path       |
| [`torch/cyclegan.py`](torch/cyclegan.py)         | An unpaired two-domain image loader and the generated-image replay buffer |
| [`flax/simple_training.py`](flax/simple_training.py) | The same tutorial against the Flax trainer, run standalone or by `scm flax train` |
| [`flax/data.py`](flax/data.py)                   | A `tf.data` input pipeline, referenced from YAML by file path             |
| [`flax/corpus.py`](flax/corpus.py)               | The same character-level corpus, as NumPy batches                         |
| [`flax/cyclegan.py`](flax/cyclegan.py)           | The `tf.data` twin of the unpaired two-domain image loader                |
| [`keras/simple_training.py`](keras/simple_training.py) | A complete Keras training program written by hand, run standalone or by `scm keras train` |
| [`keras/optimizers.py`](keras/optimizers.py)     | The one optimizer knob no object pattern can express, referenced from YAML by file path |
| [`keras/data.py`](keras/data.py)                 | A `tf.data` image pipeline with core Keras augmentation, referenced from YAML by file path |
| [`keras/corpus.py`](keras/corpus.py)             | The NumPy twin of `torch/corpus.py`, referenced from the CLI by file path |
| [`keras/cyclegan.py`](keras/cyclegan.py)         | The rank-sharded `tf.data` twin of the unpaired two-domain image loader   |

Run the tutorial:

```bash
uv run python examples/torch/simple_training.py
```

It trains a two-layer MLP on a synthetic dataset for three epochs and finishes in a few seconds on
the CPU.

Or run the very same objects under the CLI, which adds the stages the tutorial leaves out —
`torch.compile` over the learner's flow function, an experiment logger, and the checkpoint savers:

```bash
FILE=examples/torch/simple_training.py

uv run scm torch train "model: [_obj_, {_addr_: build_model, _file_: $FILE}, _call_]" \
    --learner "[_obj_, {_addr_: SimpleLearner, _file_: $FILE}]" \
    --training-dataset "[_obj_, {_addr_: make_dataset, _file_: $FILE}, {_call_: {batches: 20, seed: 0}}]" \
    --validation-dataset "[_obj_, {_addr_: make_dataset, _file_: $FILE}, {_call_: {batches: 5, seed: 1}}]" \
    --device cpu --epochs 3 --compile true --ci -LC val_loss -HC val_accuracy -E simple-training
```

Nothing is generated for that: `_addr_` names a symbol of the file and `_file_` the path to load it
from, resolved from the working directory. The learner is hand-written, and the command drives it
exactly as it drives a generated one — which is why the members below are worth writing in full.

## Walkthrough: `torch/simple_training.py`

### The dataset

A dataset is any iterable of dictionaries. `make_dataset` builds a list of pre-made batches from a
`torch.Generator` seeded by its `seed` argument, so no download and no `DataLoader` are involved:

```python
dataset.append({"x": x, "y": y})
```

The keys of each dictionary become the keyword arguments of the learner's steps: the trainer calls
`training_step(**inputs)` for every item it pulls from the dataset. Taking a seed rather than a
generator is what lets `--training-dataset` and `--validation-dataset` build the two splits from
plain YAML values.

### The learner

`SimpleLearner` is the object the redesign asks you to write per model. It implements the `Learner`
protocol — nothing is subclassed, nothing is registered:

- **`models`** — the models by name. Every callback reads them from `info.models`.
- **`steps` / `updates` / `has_updated`** — the training counters the learner owns
  (`docs/adr/0018`): completed steps, completed optimizer applies, and whether the step that just
  ran applied the optimizers. Here all three advance together — one step, one update; a learner
  accumulating gradients over N batches reports `has_updated` only after every N-th step, and the
  trainer fires `on_update` that often. `restore_counters(steps, updates)` seeds them when a run
  resumes from a checkpoint.
- **`flow_functions`** — the compile units, by attribute name: `{"_flow_optimizer": self._flow_optimizer}`.
  `torch.compile` is applied to the flows, never to the step (`docs/adr/0004`), so `scm torch train
  --compile` rebinds every name listed here with the compiled wrapper —
  `setattr(learner, name, torch.compile(...))`. That is why `_flow_optimizer` is a closure bound as
  an attribute in `__init__` and why both steps call it back through `self`: a step calling the
  closure directly would look compiled and train uncompiled, and returning `{}` skips the stage
  altogether, leaving only the models compiled.
- **`training_step(**inputs)`** — the flow function, then backward and the optimizer step. Returns the
  criteria of the step. It stays eager on purpose: it owns the host-side counters and the optimizer
  calls, which inside a compiled region would only become graph breaks and guards.
- **`inference_step(**inputs)`** — the validation counterpart, running the same flow under
  `torch.no_grad()` and returning the same criteria.
- **`outputs`** — the criterion names the steps return. The CLI reads them off the learner to build
  the tracker and the progress-bar rows, unless `--learner-outputs` overrides them.

The flow function takes `__need_update__` first, as every generated one does: it arms the gradient
synchronization of a DDP- or FSDP2-wrapped model on the last backward of an update, and does nothing
on a plain module. A generated learner emits one `_flow_<optimizer>` per optimizer segment plus a
`_flow_inference`; one flow covers both steps here because they compute the same thing.

The optimizer lives on the learner, together with its schedule. `SimpleLearner` also defines
`on_epoch_end`, which advances the schedule after each epoch. The trainer finds that method by
checking the learner against the `OnEpochEnd` protocol, so no registration call is needed.

Three more properties are required by the protocol: `optimizers` (the CLI saves their state between
epochs, and the trainer scans them for event protocols as well), `optimizer_models` (which models
each optimizer updates — `{"optimizer": ["model"]}` here — so checkpointing can pair optimizer state
with those models), and `learning_rates` (printed next to the criteria by `Printer` and logged by the
loggers). Optional extras — `grad_scalers`, `weight_decays`, `param_group_names` — are read by the
toolkit when present.

### The tracker

The trainer calls the tracker once per step with the dictionary the step returned, then writes the
result into the logs of the current epoch:

```python
def track(**criteria: torch.Tensor) -> dict[str, float]:
    return {name: value.item() for name, value in criteria.items()}
```

This one simply reports the last step's values. `structcast_model.torch.trainer.TorchTracker`
averages criteria over the epoch instead, and reduces them across ranks when the run is distributed.

### The data provider

`SimpleDataProvider` carries the training dataset and the optional validation dataset for the whole
run. It is given to the trainer at construction, which is why `fit()` takes no dataset argument.
Passing `validation_dataset=None` skips validation entirely.

### The trainer and its callbacks

```python
trainer = TorchTrainer(
    device="cpu",
    learner=SimpleLearner(model),
    tracker=track,
    data=data,
    callbacks=[Printer(), best],
)
```

Every participant is scanned once on first use, in a fixed order — learner, the learner's optimizers,
tracker, data provider and its datasets, then the `callbacks` sequence in the order given — and routed into each
lifecycle event whose protocol it implements. `trainer.describe()` prints the result:

```text
Registered callbacks: {'on_epoch_end': ['SimpleLearner', 'Printer', 'BestCriterion']}
```

The order matters: the learner steps its schedule before `Printer` reads `learning_rates`, so the
line printed after epoch 1 already shows the rate that epoch 2 will use.

`BestCriterion(target="val_loss", mode="min")` monitors one criterion and keeps its best value and
the step at which it appeared. Validation criteria carry the `val_` prefix (`validation_prefix` on
the trainer), which is why the target is `val_loss` and not `loss`.

### Fitting

```python
trainer.fit(epochs=3)
print(f"\nBest {best.target}: {best.value:.4f} at step {best.step}")
```

`fit()` only takes loop parameters: `epochs`, `start_epoch`, and `validation_frequency`. It feeds the
provider's datasets into `train(dataset)` and `evaluate(dataset)`, which remain usable on their own
when you want a single pass over a dataset rather than a whole run.

## The same program, configuration-driven

The CLI builds the same objects from YAML templates. Instead of writing a learner class, generate one:

```bash
# 1. The model
scm torch create model cfg/torch/models/ConvNeXtV2.yaml -p 'DEFAULT: {backbone: femto}' -o model.py

# 2. The learner: losses, metrics, optimizer, mixed precision, and both flows
scm torch create learner cfg/torch/learners/ConvNeXtV2.yaml -p 'DEFAULT: {epochs: 5}' -o learner.py
```

`learner.py` holds a `Learner` class with exactly the members the tutorial writes by hand — `models`,
`update`, `training_step`, `inference_step`, plus `flow_functions`, `optimizers`, `optimizer_models`,
`grad_scalers`, `learning_rates`, `weight_decays`, and `param_group_names`.

Then render the dataset configurations and train:

```bash
scm format cfg/torch/others/default_timm.yaml \
    -o dataset_train.yaml \
    -p 'DEFAULT: {training: true, epochs: 5, batch_size: 32, dataset: torch/cifar100, num_classes: 100, input_size: [3, 224, 224], image_dtype: bfloat16, download: true}'

scm format cfg/torch/others/default_timm.yaml \
    -o dataset_valid.yaml \
    -p 'DEFAULT: {training: false, epochs: 5, batch_size: 32, dataset: torch/cifar100, num_classes: 100, input_size: [3, 224, 224], image_dtype: bfloat16, download: true}'

scm torch train \
    'model: [_obj_, {_addr_: model.Model, _file_: model.py}, _call_]' \
    -s 'image: [3, 224, 224]' \
    -d cuda \
    --learner '[_obj_, {_addr_: learner.Learner, _file_: learner.py}]' \
    --training-dataset dataset_train.yaml \
    --validation-dataset dataset_valid.yaml \
    -e 5 \
    -LC val_ce_loss \
    -HC val_acc1 \
    -SC val_acc1 \
    --logger mlflow \
    -E Test
```

How the CLI maps onto the objects of the tutorial:

| Tutorial                       | CLI                                                                       |
| ------------------------------ | ------------------------------------------------------------------------- |
| `SimpleLearner(model)`         | `--learner/-L` pattern, called with the models the command instantiated    |
| `SimpleLearner.flow_functions` | `--compile/-c` (single-device), rebinding flows with compiled wrappers     |
| `track`                        | `TorchTracker`, built from the learner's `outputs` or `--learner-outputs`  |
| `SimpleDataProvider(...)`      | `--training-dataset` and `--validation-dataset/-V`, composed into one      |
| `Printer()`                    | `ProgressBar`, or `Printer` when `--ci` is given                           |
| `BestCriterion(...)`           | `--lower-criterion/-LC`, `--higher-criterion/-HC`, `--save-criterion/-SC`  |
| `print(...)` of the best value | `--logger mlflow` or `--logger wandb`, plus `--experiment/-E`              |
| `trainer.fit(epochs=3)`        | `--epochs/-e`, `--start-epoch`, `--validation-frequency/-f`                |

The logger is a context manager owning the run: it starts the run, logs the parameters and the
artifacts given with `--log-artifacts/-A`, records the epoch metrics through its own `on_epoch_end`,
and ends the run. `--logger wandb` needs the `wandb` extra, `--logger mlflow` the `mlflow` extra.
Current MLflow refuses a plain file-store tracking URI (the default when no server is configured)
unless `MLFLOW_ALLOW_FILE_STORE=true` is set in the environment.

## File-addressed optimizer compositions

The package ships `structcast_model.torch.optimizers.create_opt`, which builds an optimizer and
nothing else. Combining it with a learning-rate schedule is use-case specific, so those combinations
are example code — [`torch/optimizers.py`](torch/optimizers.py) — and a learner template references
them by file path with `_file_`:

```yaml
OPTIMIZER:
  - _obj_
  - _addr_: AdamWWithCosine
    _file_: examples/torch/optimizers.py
  - _bind_:
      optimizer_kwargs: {opt: adamw, lr: 0.004, weight_decay: 0.001}
      scheduler_kwargs: {sched: cosine, num_epochs: 300, criterion: ce_loss}
```

`_addr_` names the symbol and `_file_` the Python file to load it from, relative to the working
directory — so the composition can live in your own repository without being packaged. The path is
resolved when the learner is instantiated, which means `scm torch train` must run from a directory
where it resolves; copy the file next to your generated code and adjust `_file_` if it does not.

`AdamWWithCosine` delegates the `Optimizer` interface to the wrapped optimizer through
`__getattr__`, so the generated learner keeps calling `step()`, `zero_grad()`, and `param_groups`
unchanged. It implements `on_update` and `on_epoch_end` itself, and the trainer's scan of the
learner's `optimizers` routes those events to it — the same mechanism the tutorial uses for the
learner's own `on_epoch_end`. `state_dict` merges the optimizer and schedule state into one
dictionary, so a resumed run keeps its schedule.

`OptimizerWithNativeScheduler` in the same file does the same for `torch.optim.lr_scheduler`
schedules, which count in epochs and therefore only need `on_epoch_end`.

## A small language model on Tiny Shakespeare

[`cfg/torch/models/SmallLanguageModel.yaml`](../cfg/torch/models/SmallLanguageModel.yaml) is a
GPT-style language model: a token embedding, a `backbone` of pre-LN blocks named `block0` …
`blockN-1`, a final layer normalization and a linear head over the vocabulary. Its blocks are
therefore addressable as `backbone.block*`, which is what the per-block sharding of
[`cfg/torch/strategies/fsdp2.yaml`](../cfg/torch/strategies/fsdp2.yaml) matches on.

Causal self-attention is not a package layer but a `CausalSelfAttention` section of that
configuration: two `torch.nn.Linear` projections around flow lines that split the fused qkv output
(`Split`), reshape it into heads (`torch.nn.Unflatten` and `Permute`), and call
`torch.nn.functional.scaled_dot_product_attention(..., is_causal=True)`, whose kernel applies the
causal mask without materializing one. Positions come from a rotary embedding computed in the same
section from the sequence length of the actual input, so there is no learned position table and no
length the model cannot run — `max_seq_len` only sizes the `INPUT_SHAPES` dummy forward.

```bash
# 1. The model: -p "DEFAULT: {size: small}" picks the larger preset, and the vocabulary and the
#    sequence length are parameters of the size group, e.g. -p "tiny: {vocab_size: 100}".
scm torch create model cfg/torch/models/SmallLanguageModel.yaml -o model.py

# 2. The learner: next-token cross entropy over the flattened logits and targets, with a native
#    torch.optim.AdamW -- FSDP2's checkpoint path refuses scheduler proxies like AdamWWithCosine
scm torch create learner cfg/torch/learners/SmallLanguageModel.yaml -o learner.py
```

[`torch/corpus.py`](torch/corpus.py) supplies the data: `TinyShakespeare` downloads the corpus once
into `data/tinyshakespeare.txt` (`data_path` reads a local file instead) and yields
`{"tokens": ..., "targets": ...}` blocks, the second being the first shifted by one character. Its
65 characters are the default `vocab_size` of the model configuration; a different corpus needs
`-p "tiny: {vocab_size: ...}"` when creating the model. `TinyShakespeareLoader` batches the items
-- owning the move onto each rank's device and the sharding across ranks, since the training loop
deliberately does neither -- and collation keeps the keys, which is how they reach the learner as
keyword arguments:

```bash
torchrun --nproc_per_node=gpu -m structcast_model.commands.main torch train \
    'model: [_obj_, {_addr_: model.Model, _file_: model.py}, _call_]' \
    -d cuda \
    --learner '[_obj_, {_addr_: learner.Learner, _file_: learner.py}]' \
    --strategy cfg/torch/strategies/fsdp2.yaml \
    --training-dataset '[_obj_, {_addr_: TinyShakespeareLoader, _file_: examples/torch/corpus.py}, {_call_: {block_size: 256, split: train, batch_size: 16, shuffle: true}}]' \
    --validation-dataset '[_obj_, {_addr_: TinyShakespeareLoader, _file_: examples/torch/corpus.py}, {_call_: {block_size: 256, split: val, batch_size: 16}}]' \
    -e 5 \
    -LC val_ce_loss \
    -SC val_ce_loss \
    --logger mlflow \
    -E SmallLanguageModel
```

The model declares its own `INPUT_SHAPES` (`tokens`, `int64`), so `--shape/-s` is not needed. To
shard every block as its own communication group, uncomment `shard_modules` in the strategy
configuration — the patterns are globs over `named_modules()` paths whose `*` and `?` never cross
a `.`, so `"backbone.block*"` matches the blocks this model generates but not their contents.

The same configuration carries a commented `sync_batchnorm: false`, the off-switch for the other
wrap-time knob: under DDP and FSDP2 the strategy converts every `BatchNorm` layer to
`SyncBatchNorm` before the models are wrapped or sharded, so a convolutional model needs no
`convert_sync_batchnorm` call of its own (a model that already made one still works). The
conversion uses timm's converter, which turns a fused `BatchNormAct2d` into a `SyncBatchNormAct`
instead of dropping its activation, and it is skipped on CPU. This language model has no
normalization of that kind, so the knob changes nothing here; bind it to `false` when a model
carries `BatchNorm` layers you want left alone — a third-party `_BatchNorm` subclass timm does not
know, or a run whose `torch.compile` graph must not break on `SyncBatchNorm`.

## File-addressed datasets

[`torch/data.py`](torch/data.py) holds the timm dataset and dataloader wrappers for the same reason:
the package's training loop takes any iterable of dictionaries, so a timm integration is use-case
code. [`cfg/torch/others/default_timm.yaml`](../cfg/torch/others/default_timm.yaml) addresses
`TimmDataLoaderWrapper` there with `_addr_` plus `_file_`, and `scm torch train` stays timm-agnostic:
the trainer scans the provider datasets for event protocols — on every rank, so
`TimmDataLoaderWrapper.on_epoch_begin` reaches the `DistributedSampler` of each process — and
`on_training_begin` applies the mixup cutoff. `TimmDataProvider` in the same file is the
programmatic wiring when you build a trainer by hand instead.

## The Flax side

`examples/flax/` is the same set of files against `flax.nnx` — less `optimizers.py`, whose helpers
were expressible in the templates directly — and the differences are the interesting part.

```bash
uv run python examples/flax/simple_training.py
```

`SimpleLearner` there implements the same `Learner` protocol, but its `flow_functions` are the two
**steps**, not a flow inside them: `nnx.jit` owns the state transfer, so the optimizer apply belongs
inside the traced region, where `torch.compile` wants it outside (`docs/adr/0004`). The signature is
the donation contract — `_training_step(model, optimizer, *, x, y, **kwargs)`, every model and
optimizer positional-or-keyword and the batch keyword-only — and `scm flax train` reads
`donate_argnames` straight off it (`docs/adr/0019`). A hand-written learner opts into donation by
writing that signature and nothing else. There is no `on_epoch_end`: the schedule is an optax
schedule counting updates, and `optax.inject_hyperparams` is what keeps its current value readable
for `Printer`.

Gradient accumulation is the same story. The Flax learner schema has no `ACCUMULATE_GRADIENTS`; the
window is an `optax.MultiSteps` wrapping the whole `tx`, and it must be the **outermost**
transformation — the generated step reads its applied count off the outermost `opt_state`, so a
window buried inside `optax.chain` would accumulate identically and still report an update every
step. `CLIP` is torch-only too: clipping is an `optax.clip_by_global_norm` at the head of the chain.

There is no `examples/flax/optimizers.py`, and that absence is the point. optax already ships
`chain`, `adamw`, `clip_by_global_norm`, `MultiSteps` and every schedule a recipe needs, and
`flax.nnx.Optimizer` only binds the result to a module — so where the torch side needs a wrapper
class to marry an optimizer to a scheduler, the Flax side writes the whole thing in the template:

```yaml
learning_rate:
  - _obj_
  - _addr_: optax.linear_schedule
  - _call_:
      end_value: 0.0
      _jinja_yaml_: |-
        init_value: {{lr}}
        transition_steps: {{(epochs - decay_epoch) * steps_per_epoch}}
        transition_begin: {{(decay_epoch - offset) * steps_per_epoch}}
```

The one thing optax does not do is count epochs: its schedules are functions of the update count,
so `cfg/flax/learners/CycleGAN.yaml` takes a `steps_per_epoch` parameter and does the conversion in
jinja. Nothing checks that number against the run — set it to the "Training dataset size" the
command prints before the first epoch, divided by the accumulation window if one is configured. The
result is not the same curve as the torch `LambdaLR`, only the same envelope: torch steps once per
epoch and holds one rate for all of it, an optax schedule is read on every update and falls
continuously, so the two agree exactly at epoch boundaries and drift inside an epoch.

[`flax/data.py`](flax/data.py) is a `tf.data` pipeline: resize, then a random crop and flip while
training or a central crop while evaluating, then normalization — all on CPU threads. Constructing
a loader takes every GPU out of TensorFlow's sight so it never reserves the memory JAX needs;
importing the module does not, because importing is something a test collector may do incidentally.
The draws are stateless and keyed by each item's position in the shuffled stream, so epochs differ
while a seed still replays a whole run. There is no epoch hook and no rank sharding: `tf.data`
reshuffles by itself, and the strategy is the only thing that splits a batch.

`name` is either a `tensorflow_datasets` set or the path of one split's directory laid out one
folder per class — the same tree `cfg/torch/others/default_timm.yaml` points timm at and
[`keras/data.py`](../examples/keras/data.py) reads, so one dataset directory on the host serves all
three frameworks. One field carries both, as on the Keras side; a tfds name is any string, so the
union is resolved left to right — an existing directory wins, anything else is a name. The
directory form is what scales: the files are listed once and decoded a batch at a time, so a set of
ImageNet's size costs a list of paths rather than a copy of the pixels. The training shuffle runs
over that list, before the decode, because a tree of class folders is listed class by class and a
buffered shuffle of the decoded stream can only ever mix a window of it — a thousand images are
0.08% of ImageNet, which leaves every batch two or three adjacent classes and, measured, a NaN loss
from the first epoch. `shuffle_buffer` therefore bounds a `tensorflow_datasets` set only.
`cfg/flax/others/default_tfdata.yaml` takes `data_dir` for the tree root or `dataset` for the name,
one of the two required, and appends the derived split to the root — so one render per split covers
a whole tree. `tensorflow_datasets` itself loads through `try_import`, and the loader's items come
from one `cached_property`, `source`, which a caller with their own `tf.data.Dataset` replaces.

[`flax/corpus.py`](flax/corpus.py) is the same Tiny Shakespeare corpus as NumPy batches, with no
device placement and — deliberately — no `DistributedSampler`. The torch twin shards per rank
because `torchrun` starts one process per rank and each must be handed a different slice of the
epoch. JAX is single-controller: one process reads the whole batch and the strategy splits it
across the mesh. A loader that also sharded per rank would double-shard, and most of the epoch
would never be trained on.

## The Keras side

`examples/keras/` is the same five files against Keras 3.

```bash
uv run python examples/keras/simple_training.py
```

It trains a two-layer MLP on a synthetic dataset for three epochs and finishes in a few seconds on
the CPU, on whichever backend `KERAS_BACKEND` selects. Or run the very same objects under the CLI,
which adds the experiment logger and the checkpoint savers:

```bash
FILE=examples/keras/simple_training.py

uv run scm keras train "model: [_obj_, {_addr_: build_model, _file_: $FILE}, _call_]" \
    --backend tensorflow \
    --learner "[_obj_, {_addr_: SimpleLearner, _file_: $FILE}]" \
    --training-dataset "[_obj_, {_addr_: make_dataset, _file_: $FILE}, {_call_: {batches: 20, seed: 0}}]" \
    --validation-dataset "[_obj_, {_addr_: make_dataset, _file_: $FILE}, {_call_: {batches: 5, seed: 1}}]" \
    --epochs 3 --ci -LC val_loss -HC val_accuracy -E simple-training
```

`--backend` has no default: Keras resolves its backend once, while it is first imported, so the
command states it rather than inheriting `~/.keras/keras.json`.

`SimpleLearner` implements the same `Learner` protocol, with three Keras-shaped differences:

- **The backend adapter owns the mechanics.** `select_backend_adapter()` returns the adapter of the
  active backend, `prepare` builds the optimizer against the segment's variables, and
  `build_train_step` / `build_inference_step` return the compiled steps. `prepare` has to run first:
  JAX refuses an unbuilt optimizer inside a jitted step (`docs/adr/0016`).
- **There is no compile stage.** `flow_functions` names the two compiled steps rather than a flow
  the CLI would wrap, because the adapter already compiled them with `tf.function` or `jax.jit`;
  it is the mapping a distributed strategy rebinds when it replicates a step.
- **The counters are read, not incremented.** `training_step` counts itself and then reads the
  optimizer's `iterations` back, so an accumulation window or a float16 loss-scale skip reports
  `has_updated is False` truthfully, without a line of the learner changing (`docs/adr/0019`).

Gradient clipping, weight decay, the learning-rate schedule and gradient accumulation are all
keywords of the Keras optimizer, which is why the Keras learner schema has no `CLIP` field and no
`ACCUMULATE_GRADIENTS` field and rejects both with the substitute named.

[`keras/data.py`](keras/data.py) reads from one of two sources, and `dataset` is the single field
naming either: a `keras.datasets` name (`mnist`, `cifar10`, `cifar100`), read into memory, or the
path of one split's directory laid out one folder per class — the same tree
[`cfg/torch/others/default_timm.yaml`](../cfg/torch/others/default_timm.yaml) points timm at, so one
dataset directory on the host serves both frameworks. Pydantic discriminates the two and refuses a
value that is neither a known name nor an existing directory. The directory form is the one that
scales — it lists the files once and decodes a batch at a time, so a set at ImageNet's size costs a
list of paths and never an array — and it is what `data_dir` renders in
[`cfg/keras/others/default_keras.yaml`](../cfg/keras/others/default_keras.yaml), joined with
`train_split` / `validation_split`. Both sources leave one batch contract — `{image: float32 in
[0, 1], label: int64}` as NumPy, keyed by `image_key` and `label_key` — and shard identically per
rank, before the decode. `RandomFlip`, `RandomCrop`, `Resizing` and `Rescaling` apply inside the
`tf.data` pipeline and never inside the model: Keras' image preprocessing layers fall back to
TensorFlow operations when a `tf.data` pipeline traces them, so one pipeline feeds a run on any
backend, while a layer built into the model would augment whatever loads that model afterwards.
Building the pipeline therefore needs `tensorflow` installed even for a `jax` or `torch` run.
`shuffle_buffer` bounds the training shuffle of a `keras.datasets` set, whose buffer would be a
second copy of an array already in memory; a directory is shuffled by its file list instead, after
the shard and before the decode, so a tree listed class by class is mixed whole rather than through
a window of it. Pad-then-crop is the small-image recipe rather than ImageNet's scale-and-aspect
jitter — a run chasing a published number brings its own random resized crop.

[`keras/optimizers.py`](keras/optimizers.py) exists for one knob: weight-decay exemptions are
configured by `optimizer.exclude_from_weight_decay(...)` after construction and before the optimizer
is built, which an object pattern cannot express. Keras names its parameters `kernel`, `bias`,
`gamma`, `beta` and `embeddings`, so those anchor words are the "no decay on biases, normalization
scales and lookup tables" rule.

[`keras/corpus.py`](keras/corpus.py) supplies `{"tokens", "targets"}` NumPy blocks for
[`cfg/keras/models/SmallLanguageModel.yaml`](../cfg/keras/models/SmallLanguageModel.yaml), which —
like the torch twin — writes its own attention section: a fused `qkv_proj` `Dense`, the rotary
rotation of the query and the key, and `keras.ops.dot_product_attention(is_causal=True)` into an
`out_proj` `Dense`. The angles come from the length of the actual input, so `max_seq_len` only sizes
the `INPUT_SHAPES` dummy forward and bounds nothing the model runs on.

Both Keras loaders shard per rank when `RANK` and `WORLD_SIZE` are set, and serve the whole stream
when they are not — read from the environment rather than from a framework, because the file must
not import one. Only the torch Keras backend is multi-process, and there `torchrun` starts one
process per rank: an unsharded loader would hand every rank the same epoch, which completes, lowers
the loss, and is wrong. On tensorflow and jax the run is a single process whose strategy splits
each batch across the replicas itself, so sharding in the loader as well would give each replica a
shard of a shard. The tail the world size does not divide is dropped, so every rank sees the same
number of batches. `keras/data.py` shards before its shuffle, so a rank's items are fixed for the
run and only their order changes — `DistributedSampler` without `set_epoch`, as in the torch twin;
`keras/corpus.py` shards after, so a rank's items change each epoch.

## Unpaired image translation with CycleGAN

The three `CycleGAN` learner templates train four models over three optimizer segments, and every
one of them reads its batch by name. `cyclegan.py` in each framework directory supplies those names
from two directories of images — `trainA` and `trainB` of the horse2zebra set, or any other pair —
drawn independently, since "unpaired" means nothing aligns them. An epoch is the longer directory,
each image is resized to `load_size`, cropped to `crop_size` and scaled to **[-1, 1]**, which is the
range the generators' closing `tanh` emits and therefore the only range the identity and cycle
losses can compare a real image against.

```bash
scm torch create model cfg/torch/models/CycleGAN_generator.yaml -o generator.py
scm torch create model cfg/torch/models/CycleGAN_discriminator.yaml -o discriminator.py
scm torch create learner cfg/torch/learners/CycleGAN.yaml -o learner.py

scm torch train \
    'G_AB: [_obj_, {_addr_: Model, _file_: generator.py}, _call_]' \
    'G_BA: [_obj_, {_addr_: Model, _file_: generator.py}, _call_]' \
    'D_A: [_obj_, {_addr_: Model, _file_: discriminator.py}, _call_]' \
    'D_B: [_obj_, {_addr_: Model, _file_: discriminator.py}, _call_]' \
    -L '[_obj_, {_addr_: Learner, _file_: learner.py}]' \
    -s 'image: [3, 256, 256]' \
    --training-dataset '[_obj_, {_addr_: UnpairedImageLoader, _file_: examples/torch/cyclegan.py},
                         {_call_: {root_A: data/horse2zebra/trainA, root_B: data/horse2zebra/trainB}}]' \
    --trainer '[_obj_, {_addr_: CycleGANTrainer, _file_: examples/torch/cyclegan.py}]' \
    -LO loss_G -LO loss_GAN -LO loss_cycle -LO loss_identity -LO loss_D_A -LO loss_D_B \
    -d cuda -e 200 -LC loss_G -E cyclegan
```

Two flags in that command are load-bearing, and only on the torch side.

`--trainer` is the replay buffer. [`cfg/torch/learners/CycleGAN.yaml`](../cfg/torch/learners/CycleGAN.yaml)
is the only one of the three templates whose discriminators train on `fake_A_sample` /
`fake_B_sample` — the paper's pool of fifty *earlier* generated images, which damps the oscillation
between the two networks — and no dataset can produce those: they are the generators' own output.
`BaseTrainer.update_models` is the one place that sees both the batch on its way to the learner and
the criteria on the way back, so `CycleGANTrainer` overrides it, adds the two pooled samples and
keeps the `fake_A` / `fake_B` the step returned. The buffer is therefore fed the previous step's
images rather than the current step's — one step of lag inside a buffer that already reaches fifty
back. `pool_size: 0` turns it off and feeds the last step's images straight through.

`-LO` names the six scalar criteria. The torch template outputs `fake_A` and `fake_B` as well — it
has to, that is how the generated images reach the buffer — and the tracker sums every criterion it
is handed into a one-element buffer, which a `[batch, 3, height, width]` image does not broadcast
into. Leaving `-LO` off fails on the first step.

Neither applies to Flax or Keras: their templates carry no buffer at all (a Flax segment
differentiates only what its own flow computes, and a Keras segment is called with the batch alone,
so both discriminate a fake image they generate themselves), they declare `INPUTS: [real_A, real_B]`,
and they keep the images out of `OUTPUTS`. Their commands are the plain ones — no `--trainer`, no
`-LO` — over [`flax/cyclegan.py`](flax/cyclegan.py) and [`keras/cyclegan.py`](keras/cyclegan.py),
whose `UnpairedImageLoader` is the same contract as a `tf.data` pipeline yielding NHWC NumPy arrays:

```bash
scm flax create model cfg/flax/models/CycleGAN_generator.yaml -o generator.py
scm flax create model cfg/flax/models/CycleGAN_discriminator.yaml -o discriminator.py
# steps_per_epoch is what turns the template's epoch counts into the step counts an optax or Keras
# schedule reads: len(training_dataset), which the command prints before the first epoch.
scm flax create learner cfg/flax/learners/CycleGAN.yaml -p 'DEFAULT: {steps_per_epoch: 1334}' -o learner.py

scm flax train \
    'G_AB: [_obj_, {_addr_: Model, _file_: generator.py}]' \
    'G_BA: [_obj_, {_addr_: Model, _file_: generator.py}]' \
    'D_A: [_obj_, {_addr_: Model, _file_: discriminator.py}]' \
    'D_B: [_obj_, {_addr_: Model, _file_: discriminator.py}]' \
    -L '[_obj_, {_addr_: Learner, _file_: learner.py}]' \
    -s 'image: [256, 256, 3]' \
    --training-dataset '[_obj_, {_addr_: UnpairedImageLoader, _file_: examples/flax/cyclegan.py},
                         {_call_: {root_A: data/horse2zebra/trainA, root_B: data/horse2zebra/trainB}}]' \
    -e 200 -LC loss_G -E cyclegan
```

The Keras command is the same one with `keras` in place of `flax`, `KERAS_BACKEND` (or `--backend`)
selected, and `_call_` appended to each model pattern.

None of the three takes a validation dataset. The discriminator segments have no `INFERENCE_FLOW`,
so an inference step would want the torch template's two buffer samples as well, and a GAN has no
held-out scalar worth selecting a checkpoint on.

Sharding follows each framework's convention, as it does for the other loaders: the torch loader
builds a `DistributedSampler` on `DATA_RANK` / `DATA_WORLD_SIZE`, the Keras one cuts both domains on
`RANK` / `WORLD_SIZE`, and the Flax one shards nothing, because JAX is single-controller and the
strategy splits each batch across the mesh itself.
