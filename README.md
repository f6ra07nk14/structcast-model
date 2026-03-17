# StructCast-Model

StructCast-Model is a configuration-driven toolkit that generates PyTorch models and training workflows from YAML templates. Built on top of [StructCast](https://github.com/f6ra07nk14/structcast), it lets you describe model architecture, optimizer logic, dataset configuration, and training orchestration declaratively — then generates runnable Python code from those descriptions.

The current implementation focuses on PyTorch. JAX and TensorFlow support is planned (see [Roadmap](#roadmap)), but all active CLI commands, code generators, and tests target the PyTorch stack.

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
  - [Distributed Training with `torchrun`](#distributed-training-with-torchrun)
    - [How It Works](#how-it-works)
    - [Single-Node Multi-GPU](#single-node-multi-gpu)
    - [Multi-Node Training](#multi-node-training)
    - [Dataset Configuration](#dataset-configuration)
    - [Distributed Training Notes](#distributed-training-notes)
  - [Configuration Examples](#configuration-examples)
    - [`cfg/models/ConvNeXtV2.yaml`](#cfgmodelsconvnextv2yaml)
    - [`cfg/backwards/ConvNeXtV2.yaml`](#cfgbackwardsconvnextv2yaml)
    - [`cfg/datasets/default_timm.yaml`](#cfgdatasetsdefault_timmyaml)
  - [API Reference: `base_trainer.py`](#api-reference-base_trainerpy)
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
  - [API Reference: `trainer.py`](#api-reference-trainerpy)
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

- **Generate model code** — Produce PyTorch `nn.Module` classes from YAML layer templates.
- **Generate training code** — Produce backward-pass, optimizer, and scheduler orchestration classes from YAML templates.
- **Format reusable templates** — Render parameterized YAML templates into concrete runtime configurations.
- **Inspect model complexity** — Compute FLOPs and parameter counts with [`ptflops`](https://github.com/sovrasov/flops-counter.pytorch) and [`calflops`](https://github.com/MrYxJ/calculate-flops.pytorch).
- **Train end-to-end** — Run training with [Automatic Mixed Precision (AMP)](https://docs.pytorch.org/docs/stable/amp.html), [timm](https://github.com/huggingface/pytorch-image-models) datasets, Exponential Moving Average (EMA), optional [`torch.compile`](https://docs.pytorch.org/tutorials/intermediate/torch_compile_tutorial.html), and [MLflow](https://mlflow.org/docs/latest/ml/deep-learning/pytorch/) experiment logging.

## Installation

StructCast-Model is installed with [uv](https://docs.astral.sh/uv/) and exposes the `scm` CLI entry point.

```bash
uv sync --extra torch-cu130 --extra mlflow --extra flops
```

Each extra installs a group of optional dependencies:

| Extra         | What it provides                                              |
| ------------- | ------------------------------------------------------------- |
| `torch-cu130` | PyTorch and torchvision with CUDA 13.0 support                |
| `mlflow`      | Experiment tracking for `scm torch train`                     |
| `flops`       | Both `ptflops` and `calflops` for model complexity inspection |

Omit any extra you do not need. For example, `uv sync --extra torch-cu130` is sufficient if you only want to generate and train models without FLOPs analysis or MLflow logging.

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

| Directory   | Purpose                                                                                                                               |
| ----------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| `builders/` | Converts validated YAML templates into intermediate representations, then renders Python source code.                                 |
| `commands/` | Exposes the `scm` CLI (built with [Typer](https://typer.tiangolo.com/)).                                                              |
| `torch/`    | Runtime utilities used by the CLI and available for direct Python usage — training steps, trackers, timm wrappers, optimizer helpers. |
| `cfg/`      | Declarative source of truth: YAML templates for models, backward logic, datasets, and runtime presets.                                |

## Core Workflow

The repository follows a repeatable five-step workflow:

1. **Write or reuse** YAML templates under `cfg/`.
2. **Render** templates with `scm format` and `-p/--parameter` overrides to produce concrete configuration files.
3. **Generate** Python source files for the model, loss, metric, and backward logic using `scm torch create`.
4. **Instantiate** those generated modules at runtime through StructCast object patterns (see [StructCast Pattern Basics](#structcast-pattern-basics)).
5. **Train** through `scm torch train`, which wires together datasets, models, losses, metrics, optimizer logic, AMP, EMA, and MLflow.

```text
YAML templates  --->  scm format / scm torch create  --->  Generated .py files
                                                                 |
StructCast patterns  <-------------------------------------------+
       |
       v
scm torch train  --->  MLflow logs + model checkpoints
```

## StructCast Pattern Basics

This repository relies heavily on [StructCast](https://github.com/f6ra07nk14/structcast) object patterns to bridge generated source files and runtime commands. The minimum syntax you need to read the CLI examples is:

| Alias    | Meaning                                   | Example                                    |
| -------- | ----------------------------------------- | ------------------------------------------ |
| `_obj_`  | Chain multiple construction steps         | `[_obj_, ..., ...]`                        |
| `_addr_` | Import a class or function by dotted path | `{_addr_: torch.nn.ReLU}`                  |
| `_file_` | Load the symbol from a local Python file  | `{_addr_: model.Model, _file_: model.py}`  |
| `_call_` | Invoke the current callable               | `_call_` or `{_call_: {out_features: 10}}` |
| `_bind_` | Partially apply arguments                 | `{_bind_: {lr: 0.001}}`                    |
| `_attr_` | Access an attribute or method             | `{_attr_: model_validate}`                 |

**Example:**

```yaml
[_obj_, {_addr_: model.Model, _file_: model.py}, _call_]
```

This pattern does the following:

1. Import `Model` from the local file `model.py`.
2. Call `Model()` with no arguments and return the instance.

This pattern is the bridge between generated source files and runtime commands like `ptflops`, `calflops`, and `train`. For full documentation on StructCast patterns, see the [StructCast README](https://github.com/f6ra07nk14/structcast).

## Command Guide

### 1. Format Templates

Use `scm format` to render a parameterized YAML template (such as [`cfg/datasets/default_timm.yaml`](cfg/datasets/default_timm.yaml)) into a concrete configuration file.

```bash
scm format cfg/datasets/default_timm.yaml \
    -o dataset_train.yaml \
    -p 'DEFAULT: {training: true, epochs: 5, batch_size: 32, dataset: torch/cifar100, num_classes: 100, label_smoothing: 0.1, input_size: [3, 224, 224], image_dtype: bfloat16, download: true}'

scm format cfg/datasets/default_timm.yaml \
    -o dataset_valid.yaml \
    -p 'DEFAULT: {training: false, epochs: 5, batch_size: 32, dataset: torch/cifar100, num_classes: 100, input_size: [3, 224, 224], image_dtype: bfloat16, download: true}'
```

What this does:

1. Loads the YAML template.
2. Merges any repeated `-p/--parameter` groups into a single parameter set.
3. Renders Jinja-based sections within the template.
4. Writes the resolved YAML to `-o/--output` (or prints to stdout if `-o` is omitted).

### 2. Generate a Model Class

Generate a Python `nn.Module` from a YAML layer template (such as [`cfg/models/ConvNeXtV2.yaml`](cfg/models/ConvNeXtV2.yaml)).

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
- `-o/--output`: output file path; if omitted, defaults to the snake-cased class name in the current directory (e.g., `model.py` for the default class name `Model`)

The ConvNeXtV2 template uses Jinja parameter groups to switch between backbone variants such as `atto`, `femto`, `tiny`, and `base`.

### 3. Generate Loss, Metric, and Backward Classes

Losses and metrics use the same `scm torch create model` command because they are also layer graphs.

```bash
scm torch create model cfg/losses/cls.yaml -c Loss -o loss.py
scm torch create model cfg/metrics/topk.yaml -c Metric -o metric.py
scm torch create backward cfg/backwards/ConvNeXtV2.yaml -p 'DEFAULT: {epochs: 5}' -o backward.py
```

The `scm torch create backward` command turns a backward template into a class that manages:

- optimizer construction
- optional gradient scaler creation
- optional gradient clipping
- optional gradient accumulation
- optimizer stepping and zeroing
- learning-rate and parameter-group inspection helpers

### 4. Inspect FLOPs and Parameters

Once a model has been generated, you can instantiate it from a StructCast pattern and measure its computational complexity.

```bash
scm torch ptflops '[_obj_, {_addr_: model.Model, _file_: model.py}, _call_]' \
    -s 'image: [3, 224, 224]' \
    --backend pytorch

scm torch calflops '[_obj_, {_addr_: model.Model, _file_: model.py}, _call_]' \
    -s 'image: [3, 224, 224]'
```

What these commands do internally:

1. Instantiate the model from the `_obj_` pattern.
2. Create dummy tensors from the `-s/--shape` specification.
3. Run one initialization forward pass via [`initial_model(...)`](src/structcast_model/torch/trainer.py).
4. Pass the initialized model to `ptflops` or `calflops` for complexity analysis.

### 5. Train a Generated Model

Below is the complete training command from the included ConvNeXtV2 example.

```bash
scm torch train \
    'model: [_obj_, {_addr_: model.Model, _file_: model.py}, _call_]' \
    -s 'image: [3, 224, 224]' \
    -d cuda \
    --ema cfg/others/ema.yaml \
    -L '[_obj_, {_addr_: loss.Loss, _file_: loss.py}, _call_]' \
    -M '[_obj_, {_addr_: metric.Metric, _file_: metric.py}, _call_]' \
    -B '[_obj_, {_addr_: backward.Backward, _file_: backward.py}]' \
    -c cfg/others/compile_default.yaml \
    -e 5 \
    -T dataset_train.yaml \
    -V dataset_valid.yaml \
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
- `-L/--loss`: StructCast pattern for the loss module
- `-M/--metric`: StructCast pattern for the metric module
- `-B/--backward`: StructCast pattern for the backward class
- `-c/--compile`: boolean, YAML file, or inline dict for `torch.compile`
- `-T/--training-dataset`: training dataset pattern or rendered dataset YAML
- `-V/--validation-dataset`: validation dataset pattern or rendered dataset YAML
- `-LC/--lower-criterion`: criteria where lower is better
- `-HC/--higher-criterion`: criteria where higher is better
- `-SC/--save-criterion`: criteria that should trigger best-model saving
- `-E/--experiment`: MLflow experiment name
- `-A/--log-artifacts`: artifacts to store in MLflow

What the train command does internally:

1. Instantiates datasets and determines their lengths.
2. Initializes models with optional dummy-input forward passes.
3. Instantiates loss, metric, backward, compile, and EMA objects.
4. Builds a `TorchTracker` from the declared output names.
5. Creates a `TorchTrainer` with training and validation step objects.
6. Logs metrics, arguments, model states, optimizer states, gradient scaler states, and best checkpoints to MLflow.

## Distributed Training with `torchrun`

`scm torch train` supports multi-GPU and multi-node [distributed data parallel (DDP)](https://docs.pytorch.org/tutorials/beginner/dist_overview.html) training out of the box via [`torchrun`](https://docs.pytorch.org/docs/stable/elastic/run.html). No changes to your generated code, YAML templates, or dataset configurations are required — the same `scm torch train` command works for both single-GPU and distributed training.

> **⚠️ SyncBatchNorm Warning**
>
> When using multi-GPU training with `DistributedDataParallel`, `scm torch train` does **not** automatically convert `BatchNorm` layers to [`SyncBatchNorm`](https://docs.pytorch.org/docs/stable/generated/torch.nn.SyncBatchNorm.html). Standard `BatchNorm` computes statistics per-GPU, which can cause inconsistent behavior across ranks — especially with small per-GPU batch sizes. If your model contains `BatchNorm` layers and you are training with DDP, consider applying [`torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)`](https://docs.pytorch.org/docs/stable/generated/torch.nn.SyncBatchNorm.html#torch.nn.SyncBatchNorm.convert_sync_batchnorm) to the model **before** wrapping it with `DistributedDataParallel`. This conversion must happen in user code or in the model definition; the CLI will not perform it for you.

### How It Works

When launched through `torchrun`, the environment variables `RANK`, `LOCAL_RANK`, `WORLD_SIZE`, `MASTER_ADDR`, and `MASTER_PORT` are set automatically. `scm torch train` detects these and enables distributed mode:

1. **Process group initialization** — The NCCL backend is initialized via [`torch.distributed.init_process_group`](https://docs.pytorch.org/docs/stable/distributed.html#torch.distributed.init_process_group).
2. **Per-rank device assignment** — Each process is assigned to `cuda:<LOCAL_RANK>`.
3. **DDP model wrapping** — All models are wrapped with [`DistributedDataParallel`](https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html).
4. **Distributed data loading** — `TimmDataLoaderWrapper` automatically creates a [`DistributedSampler`](https://docs.pytorch.org/docs/stable/data.html#torch.utils.data.distributed.DistributedSampler) when a distributed environment is detected. The sampler's `set_epoch()` is called each epoch for proper shuffling.
5. **Metric synchronization** — `TorchTracker` uses [`all_reduce`](https://docs.pytorch.org/docs/stable/distributed.html#torch.distributed.all_reduce) to average loss and metric values across all ranks.
6. **Rank-0 logging** — MLflow logging, progress bars, and checkpoint saving are performed only on rank 0.
7. **Gradient sync optimization** — During gradient accumulation steps, DDP gradient synchronization is disabled to reduce communication overhead.
8. **EMA handling** — `TimmEmaWrapper` automatically unwraps the DDP module before updating EMA weights.
9. **Cleanup** — `torch.distributed.destroy_process_group()` is called when training finishes.

### Single-Node Multi-GPU

To train on all GPUs of a single machine, prefix your `scm torch train` command with `torchrun`:

```bash
# Use all available GPUs on the current machine
torchrun --nproc_per_node=gpu \
    -m structcast_model.commands.main \
    torch train \
    'model: [_obj_, {_addr_: model.Model, _file_: model.py}, _call_]' \
    -s 'image: [3, 224, 224]' \
    -d cuda \
    --ema cfg/others/ema.yaml \
    -L '[_obj_, {_addr_: loss.Loss, _file_: loss.py}, _call_]' \
    -M '[_obj_, {_addr_: metric.Metric, _file_: metric.py}, _call_]' \
    -B '[_obj_, {_addr_: backward.Backward, _file_: backward.py}]' \
    -c cfg/others/compile_default.yaml \
    -e 5 \
    -T dataset_train.yaml \
    -V dataset_valid.yaml \
    -f 1 \
    -LC ce_loss -LC val_ce_loss \
    -HC acc1 -HC val_acc1 -HC acc5 -HC val_acc5 \
    -SC val_acc1 \
    --matmul-precision high \
    -E Test
```

Or specify an exact GPU count:

```bash
# Use exactly 4 GPUs
torchrun --nproc_per_node=4 \
    -m structcast_model.commands.main \
    torch train ...
```

> **Note:** `torchrun` launches the training script as a Python module (`-m structcast_model.commands.main`) rather than through the `scm` entry point. This is because `torchrun` requires a module or script path, not a console script wrapper.

### Multi-Node Training

For training across multiple machines, provide the node topology to `torchrun` on each node:

```bash
# On node 0 (master)
torchrun \
    --nproc_per_node=4 \
    --nnodes=2 \
    --node_rank=0 \
    --master_addr=192.168.1.100 \
    --master_port=29500 \
    -m structcast_model.commands.main \
    torch train ...

# On node 1
torchrun \
    --nproc_per_node=4 \
    --nnodes=2 \
    --node_rank=1 \
    --master_addr=192.168.1.100 \
    --master_port=29500 \
    -m structcast_model.commands.main \
    torch train ...
```

This creates 8 total processes (4 GPUs × 2 nodes) training with DDP.

`torchrun` parameters:

| Parameter          | Description                                                      |
| ------------------ | ---------------------------------------------------------------- |
| `--nproc_per_node` | Number of processes per node. Use `gpu` for all available GPUs.  |
| `--nnodes`         | Total number of nodes. Defaults to `1` for single-node training. |
| `--node_rank`      | Rank of the current node (0-indexed).                            |
| `--master_addr`    | IP address of the master node.                                   |
| `--master_port`    | Port for inter-node communication.                               |

`scm torch train` distributed-related options:

| Option           | Description                                                                                               |
| ---------------- | --------------------------------------------------------------------------------------------------------- |
| `--dist-backend` | Distributed backend (`nccl`, `gloo`). Auto-selected if omitted. Also settable via `DIST_BACKEND` env var. |
| `--dist-url`     | URL for distributed setup. Defaults to `env://`. Also settable via `DIST_URL` env var.                    |
| `--ci`           | Disables `tqdm` progress bars — useful in cluster job logs.                                               |

### Dataset Configuration

Dataset YAML files do **not** need per-rank customization. A single `device: cuda` value in the dataset configuration works for all ranks — `TimmDataLoaderWrapper` internally resolves it to the correct `cuda:<LOCAL_RANK>` device for each process.

```bash
# The same dataset YAML works for single-GPU and distributed training
scm format cfg/datasets/default_timm.yaml \
    -o dataset_train.yaml \
    -p 'DEFAULT: {training: true, epochs: 5, batch_size: 32, dataset: torch/cifar100, num_classes: 100, label_smoothing: 0.1, input_size: [3, 224, 224], image_dtype: bfloat16, download: true}'
```

> **Tip:** The `batch_size` in the dataset template is the **per-GPU** batch size. With 4 GPUs and `batch_size: 32`, the effective global batch size is 128.

### Distributed Training Notes

- **Seed reproducibility** — Each rank's random seed is offset by `global_rank` to ensure different data augmentation across processes while remaining reproducible.
- **Learning rate scaling** — When scaling to multiple GPUs, consider adjusting the learning rate. A common practice is [linear scaling](https://arxiv.org/abs/1706.02677): multiply the base learning rate by the number of GPUs. This must be configured in the backward template or optimizer settings — `scm torch train` does not scale the learning rate automatically.
- **SyncBatchNorm** — `scm torch train` does **not** automatically convert `BatchNorm` layers to [`SyncBatchNorm`](https://docs.pytorch.org/docs/stable/generated/torch.nn.SyncBatchNorm.html). If your model uses `BatchNorm` and you are training with DDP, consider applying `torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)` in the model definition. See the [SyncBatchNorm warning](#5-train-a-generated-model) for details.
- **`torch.compile` and DDP** — When both `--compile` and DDP are active, `torch.compile` is applied **before** DDP wrapping.
- **Checkpoint saving** — Only rank 0 saves checkpoints and logs to MLflow. When resuming from a checkpoint in a distributed setting, all ranks load the same checkpoint.

## Configuration Examples

The `cfg/` directory contains working YAML templates that demonstrate each part of the workflow.

### `cfg/models/ConvNeXtV2.yaml`

Demonstrates the model-building style used throughout the project:

- parameter groups for multiple backbone sizes
- nested user-defined layers such as `Backbone`, `Stem`, `DownSample`, and `Block`
- Jinja-driven layer expansion
- separate training and inference flow support
- structured outputs such as `{cls: torch.tensor(...), ...}`

### `cfg/backwards/ConvNeXtV2.yaml`

Demonstrates how backward logic is configured declaratively:

- `MIXED_PRECISION` for `torch.amp.GradScaler`
- `MIXED_PRECISION_TYPE` for autocast dtype
- `ACCUMULATE_GRADIENTS` for delayed optimizer updates
- optimizer creation through `structcast_model.torch.optimizers.create_with_scheduler`
- optional gradient clipping via `timm.utils.clip_grad.dispatch_clip_grad`

### `cfg/datasets/default_timm.yaml`

Formats directly into a `TimmDataLoaderWrapper.model_validate(...)` pattern. Covers:

- timm dataset construction
- timm dataloader construction
- device and prefetch settings
- mixup and cutmix options
- train or validation split generation from one template

## API Reference: `base_trainer.py`

[`src/structcast_model/base_trainer.py`](src/structcast_model/base_trainer.py) provides the framework-agnostic training loop, state management, and callback system. Concrete trainers such as `TorchTrainer` build on top of these abstractions.

### Utility functions

#### `get_dataset(dataset)`

Resolves a `DatasetLike` or a zero-argument callable into an actual iterable. This allows lazy dataset construction.

#### `get_dataset_size(dataset)`

Returns the number of batches. Uses `__len__` when available, otherwise iterates to count.

#### `invoke_callback(callbacks, info, *args, **models)`

Iterates over a callback list and calls each entry with `info` and keyword model arguments.

### Protocols

#### `Forward`

Called once per batch during training or validation. Accepts an `inputs` dictionary and keyword model arguments; returns a `dict[str, Any]` of named outputs and criteria.

#### `Backward`

Called once per training step. Receives the step index and criterion keyword arguments; returns `True` when the optimizer has stepped, `False` when gradients are being accumulated.

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

## API Reference: `trainer.py`

[`src/structcast_model/torch/trainer.py`](src/structcast_model/torch/trainer.py) contains the PyTorch-specific runtime layer.

### Utility functions

#### `create_torch_inputs(shape)`

Creates dummy `float32` tensors from tuple, list, or dict shape descriptions. Used for model initialization and FLOPs inspection.

#### `get_torch_device(device=None)`

Returns the runtime device. Selects `cuda` when available and requested, otherwise falls back to `cpu`.

#### `initial_model(model, shapes=None, compile_fn=None)`

Walks a module or nested module structure, optionally builds dummy inputs, runs a forward pass, and applies a compile function to each module. Returns:

```python
(initialized_model, inputs, outputs)
```

#### `get_autocast(mixed_precision_type, device)`

Returns a context manager for automatic mixed precision:

- `contextlib.suppress` when AMP is disabled.
- A configured `torch.autocast(...)` partial when AMP is enabled.

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

Wraps `CriteriaTracker` instances for losses and metrics, resets them through global callbacks, and returns float-valued logs suitable for history storage and MLflow logging.

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

- Prefetching
- Channels-last memory format conversion
- Mixup and cutmix data augmentation
- Train/validation-specific augmentation settings
- Distributed device initialization
- Optional `FlexSpec` output remapping

The dataset template at `cfg/datasets/default_timm.yaml` formats into this wrapper.

#### `TimmEmaWrapper`

Creates and updates `timm.utils.ModelEmaV3` instances and swaps them into inference-time evaluation when appropriate.

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
    -L '[_obj_, {_addr_: loss.Loss, _file_: loss.py}, _call_]' \
    -M '[_obj_, {_addr_: metric.Metric, _file_: metric.py}, _call_]' \
    -B '[_obj_, {_addr_: backward.Backward, _file_: backward.py}]' \
    -c cfg/others/compile_default.yaml \
    -e 5 \
    -T dataset_train.yaml \
    -V dataset_valid.yaml \
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

Set up the development environment with:

```bash
uv sync --extra torch-cpu --dev --group tox
```

Run the test suite:

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

Run all checks in parallel with:

```bash
tox run-parallel --parallel all
```

The repository includes tests for:

- CLI behavior
- Builder code generation
- Schema validation
- Trainer utilities
- timm dataset and dataloader wrappers
- Custom torch layers

## Roadmap

- [x] PyTorch model construction from YAML configuration files
- [x] PyTorch training workflow generation from YAML configuration files
- [ ] JAX model construction from YAML configuration files
- [ ] JAX training workflow generation from YAML configuration files
- [ ] TensorFlow model construction from YAML configuration files
- [ ] TensorFlow training workflow generation from YAML configuration files
