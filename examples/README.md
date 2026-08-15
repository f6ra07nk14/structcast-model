# Examples

Two ways to build the same training program, plus the integrations a configuration loads by file path:

| File                                             | What it shows                                                             |
| ------------------------------------------------ | ------------------------------------------------------------------------- |
| [`torch/simple_training.py`](torch/simple_training.py) | A complete training program written by hand against the trainer API   |
| [`torch/optimizers.py`](torch/optimizers.py)     | Optimizer + scheduler compositions referenced from YAML by file path      |
| [`torch/data.py`](torch/data.py)                 | timm dataset and dataloader wrappers, referenced from YAML by file path   |
| [`torch/corpus.py`](torch/corpus.py)             | A character-level text corpus, referenced from the CLI by file path       |

Run the tutorial:

```bash
uv run python examples/torch/simple_training.py
```

It trains a two-layer MLP on a synthetic dataset for three epochs and finishes in a few seconds on
the CPU.

## Walkthrough: `torch/simple_training.py`

### The dataset

A dataset is any iterable of dictionaries. `make_dataset` builds a list of pre-made batches from a
seeded `torch.Generator`, so no download and no `DataLoader` are involved:

```python
dataset.append({"x": x, "y": y})
```

The keys of each dictionary become the keyword arguments of the learner's steps: the trainer calls
`training_step(**inputs)` for every item it pulls from the dataset.

### The learner

`SimpleLearner` is the object the redesign asks you to write per model. It implements the `Learner`
protocol — nothing is subclassed, nothing is registered:

- **`models`** — the models by name. Every callback reads them from `info.models`.
- **`update(step)`** — whether this step applied the optimizers. Returning `True` on every step means
  "one step, one update"; a learner accumulating gradients over N batches returns `True` only every
  N-th step, and the trainer fires `on_update` that often.
- **`training_step(**inputs)`** — forward, backward, optimizer step. Returns the criteria of the step.
- **`inference_step(**inputs)`** — the validation counterpart, returning the same criteria.

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
`update`, `training_step`, `inference_step`, plus `optimizers`, `optimizer_models`, `grad_scalers`,
`learning_rates`, `weight_decays`, and `param_group_names`.

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
| `track`                        | `TorchTracker`, built from the learner's `outputs` or `--learner-outputs`  |
| `SimpleDataProvider(...)`      | `--training-dataset` and `--validation-dataset/-V`, composed into one      |
| `Printer()`                    | `ProgressBar`, or `Printer` when `--ci` is given                           |
| `BestCriterion(...)`           | `--lower-criterion/-LC`, `--higher-criterion/-HC`, `--save-criterion/-SC`  |
| `print(...)` of the best value | `--logger mlflow` or `--logger wandb`, plus `--experiment/-E`              |
| `trainer.fit(epochs=3)`        | `--epochs/-e`, `--start-epoch`, `--validation-frequency/-f`                |

The logger is a context manager owning the run: it starts the run, logs the parameters and the
artifacts given with `--log-artifacts/-A`, records the epoch metrics through its own `on_epoch_end`,
and ends the run. `--logger wandb` needs the `wandb` extra, `--logger mlflow` the `mlflow` extra.

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

## File-addressed datasets

[`torch/data.py`](torch/data.py) holds the timm dataset and dataloader wrappers for the same reason:
the package's training loop takes any iterable of dictionaries, so a timm integration is use-case
code. [`cfg/torch/others/default_timm.yaml`](../cfg/torch/others/default_timm.yaml) addresses
`TimmDataLoaderWrapper` there with `_addr_` plus `_file_`, and `scm torch train` stays timm-agnostic:
the trainer scans the provider datasets for event protocols — on every rank, so
`TimmDataLoaderWrapper.on_epoch_begin` reaches the `DistributedSampler` of each process — and
`on_training_begin` applies the mixup cutoff. `TimmDataProvider` in the same file is the
programmatic wiring when you build a trainer by hand instead.
