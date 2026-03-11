# StructCast-Model

StructCast-Model is a configuration-driven toolkit for generating PyTorch models and training workflows from YAML templates. It builds on top of StructCast's pattern system, template rendering, and runtime instantiation so that model structure, optimizer logic, dataset wiring, and training orchestration can all be described declaratively.

The implemented workflow in this repository is PyTorch-first. JAX and TensorFlow appear in the dependency matrix and roadmap, but the active CLI, builders, examples, and tests are centered on the PyTorch stack.

## Table of Contents

- [StructCast-Model](#structcast-model)
  - [Table of Contents](#table-of-contents)
  - [What This Project Does](#what-this-project-does)
  - [Installation](#installation)
  - [Project Structure](#project-structure)
  - [Core Workflow](#core-workflow)
  - [StructCast Pattern Basics](#structcast-pattern-basics)
  - [Command Guide](#command-guide)
    - [1. Format Templates](#1-format-templates)
    - [2. Generate a Model Class](#2-generate-a-model-class)
    - [3. Generate Loss, Metric, and Backward Classes](#3-generate-loss-metric-and-backward-classes)
    - [4. Inspect FLOPs and Parameters](#4-inspect-flops-and-parameters)
    - [5. Train a Generated Model](#5-train-a-generated-model)
  - [Configuration Examples Included in This Repository](#configuration-examples-included-in-this-repository)
    - [`cfg/models/ConvNeXtV2.yaml`](#cfgmodelsconvnextv2yaml)
    - [`cfg/backwards/ConvNeXtV2.yaml`](#cfgbackwardsconvnextv2yaml)
    - [`cfg/datasets/default_timm.yaml`](#cfgdatasetsdefault_timmyaml)
  - [API Introduction: `base_trainer.py`](#api-introduction-base_trainerpy)
    - [Utility functions](#utility-functions)
      - [`get_dataset(dataset)`](#get_datasetdataset)
      - [`get_dataset_size(dataset)`](#get_dataset_sizedataset)
      - [`invoke_callback(callbacks, info, *args, **models)`](#invoke_callbackcallbacks-info-args-models)
    - [Protocols](#protocols)
      - [`Forward`](#forward)
      - [`Backward`](#backward)
      - [`Callback` and `BestCallback`](#callback-and-bestcallback)
      - [`InferenceWrapper`](#inferencewrapper)
    - [State and callbacks](#state-and-callbacks)
      - [`BaseInfo`](#baseinfo)
      - [`Callbacks`](#callbacks)
      - [`GLOBAL_CALLBACKS`](#global_callbacks)
    - [Core classes](#core-classes)
      - [`BaseTrainer`](#basetrainer)
      - [`BestCriterion`](#bestcriterion)
  - [API Introduction: `trainer.py`](#api-introduction-trainerpy)
    - [Utility functions](#utility-functions-1)
      - [`create_torch_inputs(shape)`](#create_torch_inputsshape)
      - [`get_torch_device(device=None)`](#get_torch_devicedevicenone)
      - [`initial_model(model, shapes=None, compile_fn=None)`](#initial_modelmodel-shapesnone-compile_fnnone)
      - [`get_autocast(mixed_precision_type, device)`](#get_autocastmixed_precision_type-device)
    - [Step objects](#step-objects)
      - [`TrainingStep`](#trainingstep)
      - [`ValidationStep`](#validationstep)
    - [Tracking and orchestration](#tracking-and-orchestration)
      - [`TorchTracker`](#torchtracker)
      - [`TorchTrainer`](#torchtrainer)
    - [timm integrations](#timm-integrations)
      - [`TimmDatasetWrapper`](#timmdatasetwrapper)
      - [`TimmDataLoaderWrapper`](#timmdataloaderwrapper)
      - [`TimmEmaWrapper`](#timmemawrapper)
  - [Minimal End-to-End Example](#minimal-end-to-end-example)
  - [Development](#development)
  - [Roadmap](#roadmap)

## What This Project Does

- Generate PyTorch model classes from YAML templates.
- Generate backward and optimizer orchestration classes from YAML templates.
- Format reusable YAML templates into concrete runtime configs.
- Compute model complexity with `ptflops` and `calflops`.
- Run end-to-end training with [Automatic Mixed Precision (AMP)](https://docs.pytorch.org/docs/stable/amp.html), [timm](https://github.com/huggingface/pytorch-image-models) datasets and Exponential Moving Average (EMA), optional [`torch.compile`](https://docs.pytorch.org/tutorials/intermediate/torch_compile_tutorial.html), and [MLflow logging](https://mlflow.org/docs/latest/ml/deep-learning/pytorch/).

## Installation

The package exposes the `scm` CLI.

```bash
uv sync --extra torch-cu130 --extra mlflow --extra flops
```

Typical extras:

- `torch-cu130`: PyTorch and torchvision for CUDA 13.0
- `mlflow`: experiment tracking during `scm torch train`
- `flops`: enables both `ptflops` and `calflops`

If you do not need FLOPs inspection or MLflow logging, you can omit those extras.

## Project Structure

```text
structcast-model/
├── cfg/
│   ├── backwards/     # backward, optimizer, scheduler templates
│   ├── datasets/      # reusable dataset/dataloader templates
│   ├── losses/        # loss module templates
│   ├── metrics/       # metric module templates
│   ├── models/        # model architecture templates
│   └── others/        # compile and EMA presets
├── src/structcast_model/
│   ├── builders/      # generic and torch-specific code generators
│   ├── commands/      # Typer CLI entry points
│   ├── torch/         # trainer, layers, optimizer helpers
│   ├── utils/         # shared helpers
│   └── base_trainer.py
├── tests/             # CLI, builder, trainer, and layer tests
└── README.md
```

The main package areas are:

- `builders`: turns validated templates into intermediate representations and then Python source code.
- `commands`: exposes the `scm` CLI.
- `torch`: contains the runtime pieces used by the CLI and by direct Python usage.
- `cfg`: serves as the declarative source of truth for generated models, backwards, datasets, and presets.

## Core Workflow

The repository follows a repeatable workflow:

1. Write or reuse YAML templates under `cfg/`.
2. Render or specialize templates with `scm format` and `-p/--parameter` overrides.
3. Generate Python source files for the model, losses, metrics, and backward logic.
4. Instantiate those generated artifacts through StructCast object patterns.
5. Train through `scm torch train`, which wires datasets, models, losses, metrics, optimizer logic, AMP, EMA, and MLflow together.

## StructCast Pattern Basics

This repository relies heavily on StructCast object patterns. The minimum syntax you need to read the commands is:

| Alias    | Meaning                                   |
| -------- | ----------------------------------------- |
| `_obj_`  | Chain multiple construction steps         |
| `_addr_` | Import a class or function by dotted path |
| `_file_` | Load the symbol from a local Python file  |
| `_call_` | Invoke the current callable               |
| `_bind_` | Partially apply arguments                 |
| `_attr_` | Access an attribute or method             |

Example:

```yaml
[_obj_, {_addr_: model.Model, _file_: model.py}, _call_]
```

This means:

1. Import `Model` from `model.py`.
2. Call `Model()` and return the instance.

That pattern is the bridge between generated source files and runtime commands such as `ptflops`, `calflops`, and `train`.

## Command Guide

### 1. Format Templates

Use `scm format` when a YAML file is parameterized (e.g. [`cfg/datasets/default_timm.yaml`](cfg/datasets/default_timm.yaml)) and you want a concrete config.

```bash
scm format cfg/datasets/default_timm.yaml \
    -o dataset_train.yaml \
    -p 'DEFAULT: {training: true, epochs: 5, batch_size: 32, dataset: torch/cifar100, num_classes: 100, label_smoothing: 0.1, input_size: [3, 224, 224], image_dtype: bfloat16, download: true}'

scm format cfg/datasets/default_timm.yaml \
    -o dataset_valid.yaml \
    -p 'DEFAULT: {training: false, epochs: 5, batch_size: 32, dataset: torch/cifar100, num_classes: 100, input_size: [3, 224, 224], image_dtype: bfloat16, download: true}'
```

What it does:

- Loads the template.
- Merges repeated `-p/--parameter` groups.
- Renders Jinja-based sections.
- Prints the resolved YAML or writes it to `-o/--output`.

### 2. Generate a Model Class

Generate a Python model module from a YAML template (e.g., [`cfg/models/ConvNeXtV2.yaml`](cfg/models/ConvNeXtV2.yaml)).

```bash
scm torch create model cfg/models/ConvNeXtV2.yaml
scm torch create model cfg/models/ConvNeXtV2.yaml -p 'DEFAULT: {backbone: femto}'
scm torch create model cfg/models/ConvNeXtV2.yaml -p 'DEFAULT: {backbone: femto}' -o model.py
```

Useful options:

- `-p/--parameter`: override template parameters
- `-c/--classname`: set the generated class name, default `Model`
- `--no-structured-output`: force tuple-like return behavior instead of a structured output mapping
- `-s/--sublayer`: generate a named sublayer from the template instead of the root model
- `-o/--output`: output file path; if omitted, the file is written to the snake-cased class name in the current directory (e.g., `model.py` for the default class name `Model`)

The ConvNeXtV2 template uses Jinja groups to switch backbone variants such as `atto`, `femto`, `tiny`, and `base`.

### 3. Generate Loss, Metric, and Backward Classes

Losses and metrics are generated through the same model command because they are still layer graphs.

```bash
scm torch create model cfg/losses/cls.yaml -c Loss -o loss.py
scm torch create model cfg/metrics/topk.yaml -c Metric -o metric.py
scm torch create backward cfg/backwards/ConvNeXtV2.yaml -p 'DEFAULT: {epochs: 5}' -o backward.py
```

`scm torch create backward` turns the backward template into a class that owns:

- optimizer construction
- optional gradient scaler creation
- optional gradient clipping
- optional gradient accumulation
- optimizer stepping and zeroing
- learning-rate and parameter-group inspection helpers

### 4. Inspect FLOPs and Parameters

Once a model has been generated, it can be instantiated from a StructCast pattern and inspected with either FLOPs backend.

```bash
scm torch ptflops '[_obj_, {_addr_: model.Model, _file_: model.py}, _call_]' \
    -s 'image: [3, 224, 224]' \
    --backend pytorch

scm torch calflops '[_obj_, {_addr_: model.Model, _file_: model.py}, _call_]' \
    -s 'image: [3, 224, 224]'
```

What these commands do internally:

- Instantiate the model from the `_obj_` pattern.
- Create dummy tensors from `-s/--shape`.
- Run one initialization forward pass with [`initial_model(...)`](src/structcast_model/torch/trainer.py#L67).
- Pass the initialized model to `ptflops` or `calflops`.

### 5. Train a Generated Model

This is the complete example flow from the current repository.

```bash
scm torch train \
    'model: [_obj_, {_addr_: model.Model, _file_: model.py}, _call_]' \
    -s 'image: [3, 224, 224]' \
    -d cuda \
    --ema cfg/others/ema.yaml \
    -l '[_obj_, {_addr_: loss.Loss, _file_: loss.py}, _call_]' \
    -m '[_obj_, {_addr_: metric.Metric, _file_: metric.py}, _call_]' \
    -b '[_obj_, {_addr_: backward.Backward, _file_: backward.py}]' \
    -c cfg/others/compile_default.yaml \
    -e 5 \
    -t dataset_train.yaml \
    -v dataset_valid.yaml \
    -f 1 \
    -LC ce_loss \
    -LC val_ce_loss \
    -HC acc1 \
    -HC val_acc1 \
    -HC acc5 \
    -HC val_acc5 \
    -SC val_acc1 \
    --matmul-precision high \
    -E Test \
    -A model.py \
    -A cfg/others/ema.yaml \
    -A loss.py \
    -A metric.py \
    -A backward.py \
    -A cfg/others/compile_default.yaml \
    -A dataset_train.yaml \
    -A dataset_valid.yaml
```

Key arguments:

- positional model patterns: one or more named model definitions
- `-s/--shape`: dummy input shapes used for model initialization
- `-d/--device`: `cpu` or `cuda`
- `--ema`: boolean, YAML file, or inline dict for `timm.utils.ModelEmaV3`
- `-l/--loss`: StructCast pattern for the loss module
- `-m/--metric`: StructCast pattern for the metric module
- `-b/--backward`: StructCast pattern for the backward class
- `-c/--compile`: boolean, YAML file, or inline dict for `torch.compile`
- `-t/--training-dataset`: training dataset pattern or rendered dataset YAML
- `-v/--validation-dataset`: validation dataset pattern or rendered dataset YAML
- `-LC/--lower-criterion`: criteria where lower is better
- `-HC/--higher-criterion`: criteria where higher is better
- `-SC/--save-criterion`: criteria that should trigger best-model saving
- `-E/--experiment`: MLflow experiment name
- `-A/--log-artifacts`: artifacts to store in MLflow

What the train command does internally:

1. Instantiates datasets and counts their lengths.
2. Initializes the models with optional dummy inputs.
3. Instantiates loss, metric, backward, compile, and EMA objects.
4. Builds a `TorchTracker` from declared output names.
5. Creates a `TorchTrainer` with training and validation step objects.
6. Logs metrics, arguments, model state, optimizer state, gradient scaler state, and best checkpoints to MLflow.

## Configuration Examples Included in This Repository

### `cfg/models/ConvNeXtV2.yaml`

This template demonstrates the model-building style used throughout the project:

- parameter groups for multiple backbone sizes
- nested user-defined layers such as `Backbone`, `Stem`, `DownSample`, and `Block`
- Jinja-driven layer expansion
- separate training and inference flow support
- structured outputs such as `{cls: torch.tensor(...), ...}`

### `cfg/backwards/ConvNeXtV2.yaml`

This template shows how backward logic is configured declaratively:

- `MIXED_PRECISION` for `torch.amp.GradScaler`
- `MIXED_PRECISION_TYPE` for autocast dtype
- `ACCUMULATE_GRADIENTS` for delayed optimizer updates
- optimizer creation through `structcast_model.torch.optimizers.create_with_scheduler`
- optional gradient clipping via `timm.utils.clip_grad.dispatch_clip_grad`

### `cfg/datasets/default_timm.yaml`

This template formats directly into a `TimmDataLoaderWrapper.model_validate(...)` pattern. It covers:

- timm dataset construction
- timm dataloader construction
- device and prefetch settings
- mixup and cutmix options
- train or validation split generation from one template

## API Introduction: `base_trainer.py`

`src/structcast_model/base_trainer.py` provides the framework-agnostic training loop, state management, and callback system that concrete trainers such as `TorchTrainer` build upon.

### Utility functions

#### `get_dataset(dataset)`

Resolves a `DatasetLike` or a zero-argument callable into an actual iterable. This allows lazy dataset construction.

#### `get_dataset_size(dataset)`

Returns the number of batches. Uses `__len__` when available, otherwise iterates to count.

#### `invoke_callback(callbacks, info, *args, **models)`

Iterates over a callback list and calls each entry with `info` and keyword model arguments.

### Protocols

#### `Forward`

Called once per batch during training or validation. Accepts any `inputs` dict and keyword models; returns `dict[str, Any]` of named outputs and criteria.

#### `Backward`

Called once per training step. Receives the step index and criterion keyword arguments; returns `True` when the optimizer has been stepped, `False` during gradient accumulation.

#### `Callback` and `BestCallback`

Lifecycle hooks called with `(info: BaseInfo, **models)`. `BestCallback` additionally receives `target: str` and `best: float` arguments.

#### `InferenceWrapper`

Applied to models before each validation epoch. Returns a remapped model dictionary, e.g., swapping a trained model for its EMA copy.

### State and callbacks

#### `BaseInfo`

Dataclass holding mutable training state:

- `step` — total training steps taken
- `update` — optimizer update count
- `epoch` — current epoch number
- `history` — per-epoch log dictionaries
- `logs(epoch=None)` — returns the log dict for the current (or given) epoch

#### `Callbacks`

Dataclass holding callback lists for each lifecycle hook:

- `on_update` — after each optimizer update
- `on_training_begin` / `on_training_end`
- `on_training_step_begin` / `on_training_step_end`
- `on_validation_begin` / `on_validation_end`
- `on_validation_step_begin` / `on_validation_step_end`
- `on_epoch_begin` / `on_epoch_end`

When `add_global_callbacks=True` (the default), entries from `GLOBAL_CALLBACKS` are copied into each list at construction time.

#### `GLOBAL_CALLBACKS`

A shared `Callbacks[Any]` instance. Callbacks registered here are automatically picked up by every newly created trainer.

### Core classes

#### `BaseTrainer`

The main training loop driver. Inherits both `BaseInfo` and `Callbacks`.

Required fields: `training_step` (`Forward`), `backward` (`Backward`), `tracker` (callable returning `dict[str, float]`).

Optional fields: `validation_step`, `inference_wrapper`, `training_prefix` (default `""`), `validation_prefix` (default `"val_"`).

Key methods:

- `train(dataset, **models)` — runs one training epoch, returns the final step logs
- `evaluate(dataset, **models)` — runs one validation epoch, returns the final step logs
- `fit(epochs, training_dataset, validation_dataset=None, start_epoch=1, validation_frequency=1, **models)` — runs the full loop and returns the complete history dict
- `sync()` — optional synchronization hook, no-op by default (overridden in `TorchTrainer`)

```python
trainer = MyTrainer(
    training_step=my_forward,
    backward=my_backward,
    tracker=my_tracker,
    validation_step=my_val_forward,
)
history = trainer.fit(
    epochs=10,
    training_dataset=train_loader,
    validation_dataset=val_loader,
    model=model,
)
```

#### `BestCriterion`

A callable that monitors a log key and fires `on_best` callbacks whenever a new best is found. Attach it to `on_epoch_end` or `on_validation_end`.

```python
checkpoint = BestCriterion(
    target="val_acc1",
    mode="max",
    on_best=[save_checkpoint],
)
trainer.on_epoch_end.append(checkpoint)
```

Fields: `target` (str), `mode` (`"min"` or `"max"`, default `"min"`), `on_best` (list of `BestCallback`).

## API Introduction: `trainer.py`

`src/structcast_model/torch/trainer.py` contains the reusable runtime layer for PyTorch execution.

### Utility functions

#### `create_torch_inputs(shape)`

Creates dummy `float32` tensors from tuple, list, or dict shape descriptions. It is used for model initialization and FLOPs inspection.

#### `get_torch_device(device=None)`

Returns the runtime device:

- `cuda` when available and requested
- `cpu` otherwise

If `cuda` is requested but unavailable, it falls back to CPU.

#### `initial_model(model, shapes=None, compile_fn=None)`

Walks a module or nested module structure, optionally builds dummy inputs, optionally runs a forward pass, and optionally applies a compile function to each module. It returns:

```python
(initialized_model, inputs, outputs)
```

#### `get_autocast(mixed_precision_type, device)`

Returns either:

- `contextlib.suppress` when AMP is disabled
- a configured `torch.autocast(...)` partial when AMP is enabled

### Step objects

#### `TrainingStep`

`TrainingStep` chains one or more models, updates a shared output dictionary, computes losses, and optionally computes metrics.

```python
step = TrainingStep(
    models=["model"],
    losses=loss_module,
    metrics=metric_module,
    autocast=get_autocast("bfloat16", "cuda"),
)
criteria = step({"image": image, "label": label}, model=model)
```

#### `ValidationStep`

Same interface as `TrainingStep`, but always executes under `torch.no_grad()`.

### Tracking and orchestration

#### `TorchTracker`

Wraps `CriteriaTracker` modules for losses and metrics, resets them through global callbacks, and returns float logs suitable for history storage and MLflow logging.

```python
tracker = TorchTracker.from_criteria(["ce_loss"], ["acc1", "acc5"])
logs = tracker(ce_loss=loss_tensor, acc1=acc1_tensor, acc5=acc5_tensor)
```

#### `TorchTrainer`

`TorchTrainer` extends the generic `BaseTrainer` with PyTorch-specific synchronization.

```python
trainer = TorchTrainer(
    device="cuda",
    training_step=TrainingStep(models=["model"], losses=loss_module, metrics=metric_module),
    validation_step=ValidationStep(models=["model"], losses=loss_module, metrics=metric_module),
    backward=backward,
    tracker=tracker,
)

history = trainer.fit(
    epochs=5,
    training_dataset=train_loader,
    validation_dataset=valid_loader,
    model=model,
)
```

### timm integrations

#### `TimmDatasetWrapper`

Holds validated dataset configuration and lazily calls `timm.data.create_dataset(...)`.

#### `TimmDataLoaderWrapper`

Builds a timm dataloader with support for:

- prefetching
- channels-last conversion
- mixup and cutmix
- train/validation-specific augmentation settings
- distributed device initialization
- optional `FlexSpec` output remapping

The dataset template under `cfg/datasets/default_timm.yaml` formats into this wrapper.

#### `TimmEmaWrapper`

Creates and updates `timm.utils.ModelEmaV3` instances and can swap them into inference-time evaluation when appropriate.

## Minimal End-to-End Example

```bash
uv sync --extra torch-cu130 --extra mlflow --extra flops

scm torch create model cfg/models/ConvNeXtV2.yaml -p 'DEFAULT: {backbone: femto}' -o model.py
scm torch create model cfg/losses/cls.yaml -c Loss -o loss.py
scm torch create model cfg/metrics/topk.yaml -c Metric -o metric.py
scm torch create backward cfg/backwards/ConvNeXtV2.yaml -p 'DEFAULT: {epochs: 5}' -o backward.py

scm format cfg/datasets/default_timm.yaml \
    -o dataset_train.yaml \
    -p 'DEFAULT: {training: true, epochs: 5, batch_size: 32, dataset: torch/cifar100, num_classes: 100, label_smoothing: 0.1, input_size: [3, 224, 224], image_dtype: bfloat16, download: true}'

scm format cfg/datasets/default_timm.yaml \
    -o dataset_valid.yaml \
    -p 'DEFAULT: {training: false, epochs: 5, batch_size: 32, dataset: torch/cifar100, num_classes: 100, input_size: [3, 224, 224], image_dtype: bfloat16, download: true}'

scm torch train \
    'model: [_obj_, {_addr_: model.Model, _file_: model.py}, _call_]' \
    -s 'image: [3, 224, 224]' \
    -d cuda \
    --ema cfg/others/ema.yaml \
    -l '[_obj_, {_addr_: loss.Loss, _file_: loss.py}, _call_]' \
    -m '[_obj_, {_addr_: metric.Metric, _file_: metric.py}, _call_]' \
    -b '[_obj_, {_addr_: backward.Backward, _file_: backward.py}]' \
    -c cfg/others/compile_default.yaml \
    -e 5 \
    -t dataset_train.yaml \
    -v dataset_valid.yaml \
    -f 1 \
    -LC ce_loss \
    -LC val_ce_loss \
    -HC acc1 \
    -HC val_acc1 \
    -HC acc5 \
    -HC val_acc5 \
    -SC val_acc1 \
    --matmul-precision high \
    -E Test
```

## Development

Run the test suite with:

```bash
pytest
```

Run static type checks:

```bash
mypy src
mypy tests
```

Run linting and formatting:

```bash
ruff check src tests
ruff format src tests
```

The repository already includes tests for:

- CLI behavior
- builder generation
- schema validation
- trainer utilities
- timm dataset and dataloader wrappers
- custom torch layers

## Roadmap

- [x] PyTorch model construction from YAML configuration files
- [x] PyTorch training workflow generation from YAML configuration files
- [ ] JAX model construction from YAML configuration files
- [ ] JAX training workflow generation from YAML configuration files
- [ ] TensorFlow model construction from YAML configuration files
- [ ] TensorFlow training workflow generation from YAML configuration files
