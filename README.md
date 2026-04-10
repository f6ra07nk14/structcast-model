# StructCast-Model

StructCast-Model is a configuration-driven toolkit that generates [PyTorch](https://pytorch.org/), [Flax (JAX)](https://flax.readthedocs.io/en/stable/), and [Keras](https://keras.io/) models — plus PyTorch training workflows — from YAML templates. Built on top of [StructCast](https://github.com/f6ra07nk14/structcast), it lets you describe model architecture, optimizer logic, dataset configuration, and training orchestration declaratively — then generates runnable Python code from those descriptions.

Model code generation is available for all three frameworks. Training workflow generation and the full training CLI (`scm torch train`) are currently PyTorch-only; Flax and Keras training support is planned (see [Roadmap](#roadmap)).

## Table of Contents

- [StructCast-Model](#structcast-model)
  - [Table of Contents](#table-of-contents)
  - [What This Project Does](#what-this-project-does)
  - [Installation](#installation)
      - [PyTorch](#pytorch)
      - [JAX / Flax](#jax--flax)
      - [TensorFlow](#tensorflow)
      - [Keras (multi-backend)](#keras-multi-backend)
      - [Bundles](#bundles)
      - [Tools](#tools)
  - [Project Structure](#project-structure)
  - [Core Workflow](#core-workflow)
  - [StructCast Pattern Basics](#structcast-pattern-basics)
  - [Command Guide](#command-guide)
    - [1. Format Templates](#1-format-templates)
    - [2. Generate a Model Class](#2-generate-a-model-class)
      - [PyTorch](#pytorch-1)
      - [Flax](#flax)
      - [Keras](#keras)
      - [Common options](#common-options)
    - [3. Generate Loss, Metric, and Backward Classes](#3-generate-loss-metric-and-backward-classes)
    - [4. Inspect FLOPs and Parameters](#4-inspect-flops-and-parameters)
    - [5. Measure Inference Time](#5-measure-inference-time)
      - [PyTorch](#pytorch-2)
      - [Flax](#flax-1)
      - [Keras](#keras-1)
    - [6. Train a Generated Model](#6-train-a-generated-model)
  - [Distributed Training with `torchrun`](#distributed-training-with-torchrun)
    - [How It Works](#how-it-works)
    - [Single-Node Multi-GPU](#single-node-multi-gpu)
    - [Multi-Node Training](#multi-node-training)
    - [Dataset Configuration](#dataset-configuration)
    - [Distributed Training Notes](#distributed-training-notes)
  - [Configuration Examples](#configuration-examples)
    - [PyTorch](#pytorch-3)
      - [`cfg/torch/models/ConvNeXtV2.yaml`](#cfgtorchmodelsconvnextv2yaml)
      - [`cfg/torch/backwards/ConvNeXtV2.yaml`](#cfgtorchbackwardsconvnextv2yaml)
      - [`cfg/torch/datasets/default_timm.yaml`](#cfgtorchdatasetsdefault_timmyaml)
    - [Flax](#flax-2)
      - [`cfg/flax/models/ConvNeXtV2.yaml`](#cfgflaxmodelsconvnextv2yaml)
    - [Keras](#keras-2)
      - [`cfg/keras/models/ConvNeXtV2.yaml`](#cfgkerasmodelsconvnextv2yaml)
  - [Schema Reference](#schema-reference)
    - [Template Parameters](#template-parameters)
      - [`PARAMETERS`](#parameters)
      - [`DEFAULT`](#default)
      - [`SHARED`](#shared)
      - [Named groups](#named-groups)
      - [`_jinja_yaml_`](#_jinja_yaml_)
      - [`_jinja_group_`](#_jinja_group_)
    - [Model Template Schema](#model-template-schema)
      - [`IMPORTS`](#imports)
      - [`INPUTS`](#inputs)
      - [`OUTPUTS`](#outputs)
      - [`STRUCTURED_OUTPUT`](#structured_output)
      - [`FLOW` and `INFERENCE_FLOW`](#flow-and-inference_flow)
      - [`FLOW` entry format](#flow-entry-format)
      - [`NAME`](#name)
      - [`LAYER`](#layer)
      - [`TYPE`, `PARAM`, and `CFG`](#type-param-and-cfg)
    - [Backward Template Schema](#backward-template-schema)
      - [`IMPORTS`](#imports-1)
      - [`MIXED_PRECISION`](#mixed_precision)
      - [`MIXED_PRECISION_TYPE`](#mixed_precision_type)
      - [`ACCUMULATE_GRADIENTS`](#accumulate_gradients)
      - [`BACKWARDS`](#backwards)
      - [`LOSSES` and `MODELS`](#losses-and-models)
      - [`BACKWARDS` entry keys](#backwards-entry-keys)
      - [`OPTIMIZERS` entry keys](#optimizers-entry-keys)
  - [API Reference: `base_trainer.py`](#api-reference-base_trainerpy)
    - [Utility functions](#utility-functions)
      - [`get_dataset(dataset)`](#get_datasetdataset)
      - [`get_dataset_size(dataset)`](#get_dataset_sizedataset)
      - [`invoke_callback(callbacks, info, *args, **models)`](#invoke_callbackcallbacks-info-args-models)
    - [Protocols](#protocols)
      - [`Forward`](#forward)
      - [`Backward`](#backward)
      - [`Callback` and `BestCallback`](#callback-and-bestcallback)
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
  - [Minimal End-to-End Example](#minimal-end-to-end-example)
  - [Development](#development)
  - [Roadmap](#roadmap)

## What This Project Does

- **Generate model code** — Produce PyTorch [`nn.Module`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html), Flax [`nnx.Module`](https://flax.readthedocs.io/en/stable/api_reference/flax.nnx/module.html), and Keras [`Layer`](https://keras.io/api/layers/base_layer/) classes from YAML layer templates.
- **Generate training code** — Produce backward-pass, optimizer, and scheduler orchestration classes from YAML templates (PyTorch only).
- **Format reusable templates** — Render parameterized YAML templates into concrete runtime configurations.
- **Inspect model complexity** — Compute FLOPs and parameter counts with [`ptflops`](https://github.com/sovrasov/flops-counter.pytorch) and [`calflops`](https://github.com/MrYxJ/calculate-flops.pytorch) (PyTorch only).
- **Measure inference time** — Benchmark average forward-pass latency of generated models across all three frameworks via `scm [torch/flax/keras] time`.
- **Train end-to-end** — Run PyTorch training with [Automatic Mixed Precision (AMP)](https://docs.pytorch.org/docs/stable/amp.html), [timm](https://github.com/huggingface/pytorch-image-models) datasets, optional [`torch.compile`](https://docs.pytorch.org/tutorials/intermediate/torch_compile_tutorial.html), and [MLflow](https://mlflow.org/docs/latest/ml/deep-learning/pytorch/) experiment logging.

## Installation

StructCast-Model is installed with [uv](https://docs.astral.sh/uv/) and exposes the `scm` CLI entry point.

```bash
uv sync --extra torch-cu130 --extra mlflow --extra flops
```

Each extra installs a group of optional dependencies. Pick the extras that match your target framework and accelerator.

#### PyTorch

| Extra         | What it provides                               |
| ------------- | ---------------------------------------------- |
| `torch-cpu`   | PyTorch and torchvision (CPU only)             |
| `torch-cu118` | PyTorch and torchvision with CUDA 11.8 support |
| `torch-cu126` | PyTorch and torchvision with CUDA 12.6 support |
| `torch-cu128` | PyTorch and torchvision with CUDA 12.8 support |
| `torch-cu130` | PyTorch and torchvision with CUDA 13.0 support |

#### JAX / Flax

| Extra      | What it provides                                                                                     |
| ---------- | ---------------------------------------------------------------------------------------------------- |
| `jax-cpu`  | [JAX](https://docs.jax.dev/en/latest/) and [Flax](https://flax.readthedocs.io/en/stable/) (CPU only) |
| `jax-cu12` | JAX and Flax with CUDA 12 support                                                                    |
| `jax-cu13` | JAX and Flax with CUDA 13 support                                                                    |

#### TensorFlow

| Extra     | What it provides                |
| --------- | ------------------------------- |
| `tf-cpu`  | TensorFlow (CPU only)           |
| `tf-cu12` | TensorFlow with CUDA 12 support |

#### Keras (multi-backend)

[Keras](https://keras.io/) runs on top of JAX, PyTorch, or TensorFlow. Choose the extra that matches your preferred backend:

| Extra               | Backend + accelerator           |
| ------------------- | ------------------------------- |
| `keras-jax-cpu`     | Keras with JAX (CPU)            |
| `keras-jax-cu12`    | Keras with JAX (CUDA 12)        |
| `keras-jax-cu13`    | Keras with JAX (CUDA 13)        |
| `keras-torch-cpu`   | Keras with PyTorch (CPU)        |
| `keras-torch-cu118` | Keras with PyTorch (CUDA 11.8)  |
| `keras-torch-cu126` | Keras with PyTorch (CUDA 12.6)  |
| `keras-torch-cu128` | Keras with PyTorch (CUDA 12.8)  |
| `keras-torch-cu130` | Keras with PyTorch (CUDA 13.0)  |
| `keras-tf-cpu`      | Keras with TensorFlow (CPU)     |
| `keras-tf-cu12`     | Keras with TensorFlow (CUDA 12) |

#### Bundles

| Extra      | What it provides                                                               |
| ---------- | ------------------------------------------------------------------------------ |
| `all-cpu`  | JAX + Flax, PyTorch + torchvision + timm, TensorFlow, and Keras — all CPU-only |
| `all-cuda` | Same as `all-cpu` but with CUDA acceleration for every backend                 |

#### Tools

| Extra      | What it provides                                                                                                |
| ---------- | --------------------------------------------------------------------------------------------------------------- |
| `ptflops`  | [`ptflops`](https://github.com/sovrasov/flops-counter.pytorch) for model complexity inspection                  |
| `calflops` | [`calflops`](https://github.com/MrYxJ/calculate-flops.pytorch) and Transformers for model complexity inspection |
| `flops`    | Both `ptflops` and `calflops`                                                                                   |
| `mlflow`   | [MLflow](https://mlflow.org/) experiment tracking for `scm torch train`                                         |

Omit any extra you do not need. For example, `uv sync --extra torch-cu130` is sufficient if you only want to generate and train PyTorch models without FLOPs analysis or MLflow logging. To work with all three model frameworks on CPU:

```bash
uv sync --extra all-cpu
```

## Project Structure

```text
structcast-model/
├── cfg/
│   ├── torch/
│   │   ├── backwards/     # backward, optimizer, scheduler templates
│   │   ├── datasets/      # reusable dataset/dataloader templates
│   │   ├── losses/        # loss module templates
│   │   ├── metrics/       # metric module templates
│   │   ├── models/        # model architecture templates
│   │   └── others/        # misc templates (e.g. for `torch.compile` options)
│   ├── flax/
│   │   └── models/        # Flax model architecture templates
│   └── keras/
│       └── models/        # Keras model architecture templates
├── src/structcast_model/
│   ├── builders/      # generic and framework-specific code generators
│   ├── commands/      # Typer CLI entry points
│   ├── torch/         # trainer, layers, optimizer helpers
│   ├── flax/          # Flax layers and inference utilities
│   ├── keras/         # Keras layers and inference utilities
│   ├── utils/         # shared helpers
│   └── base_trainer.py
├── tests/             # CLI, builder, trainer, and layer tests
└── README.md
```

The main package areas are:

| Directory    | Purpose                                                                                                                               |
| ------------ | ------------------------------------------------------------------------------------------------------------------------------------- |
| `builders/`  | Converts validated YAML templates into intermediate representations, then renders Python source code for PyTorch, Flax, and Keras.    |
| `commands/`  | Exposes the `scm` CLI (built with [Typer](https://typer.tiangolo.com/)) with `torch`, `flax`, and `keras` sub-commands.               |
| `torch/`     | Runtime utilities used by the CLI and available for direct Python usage — training steps, trackers, timm wrappers, optimizer helpers. |
| `flax/`      | Flax-specific layers (e.g. `GlobalResponseNorm`) and JAX inference helpers.                                                           |
| `keras/`     | Keras-specific layers (e.g. `GlobalResponseNormalization`) and backend-agnostic inference helpers.                                    |
| `cfg/torch/` | Declarative source of truth: YAML templates for PyTorch models, backward logic, datasets, and runtime presets.                        |
| `cfg/flax/`  | YAML templates for Flax model architectures.                                                                                          |
| `cfg/keras/` | YAML templates for Keras model architectures.                                                                                         |

## Core Workflow

The repository follows a repeatable workflow:

1. **Write or reuse** YAML templates under `cfg/[torch/flax/keras]/`.
2. **Render** templates with `scm format` and `-p/--parameter` overrides to produce concrete configuration files.
3. **Generate** Python source files for the model (and, for PyTorch, loss, metric, and backward logic) using `scm [torch/flax/keras] create`.
4. **Instantiate** those generated modules at runtime through StructCast object patterns (see [StructCast Pattern Basics](#structcast-pattern-basics)).
5. **Benchmark** inference latency with `scm [torch/flax/keras] time`.
6. *(PyTorch only)* **Train** through `scm torch train`, which wires together datasets, models, losses, metrics, optimizer logic, AMP, and MLflow.

```text
YAML templates  --->  scm format / scm [torch/flax/keras] create  --->  Generated .py files
                                                                              |
StructCast patterns  <--------------------------------------------------------+
       |
       v
scm [torch/flax/keras] time  --->  Inference benchmarks
scm torch train              --->  MLflow logs + model checkpoints
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

Use `scm format` to render a parameterized YAML template (such as [`cfg/torch/datasets/default_timm.yaml`](cfg/torch/datasets/default_timm.yaml)) into a concrete configuration file.

```bash
scm format cfg/torch/datasets/default_timm.yaml \
    -o dataset_train.yaml \
    -p 'DEFAULT: {training: true, epochs: 5, batch_size: 32, dataset: torch/cifar100, num_classes: 100, label_smoothing: 0.1, input_size: [3, 224, 224], image_dtype: bfloat16, download: true}'

scm format cfg/torch/datasets/default_timm.yaml \
    -o dataset_valid.yaml \
    -p 'DEFAULT: {training: false, epochs: 5, batch_size: 32, dataset: torch/cifar100, num_classes: 100, input_size: [3, 224, 224], image_dtype: bfloat16, download: true}'
```

What this does:

1. Loads the YAML template.
2. Merges any repeated `-p/--parameter` groups into a single parameter set.
3. Renders Jinja-based sections within the template.
4. Writes the resolved YAML to `-o/--output` (or prints to stdout if `-o` is omitted).

### 2. Generate a Model Class

Each framework has its own `create model` command that reads a YAML layer template and generates a framework-native module.

#### PyTorch

Generate a PyTorch [`nn.Module`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html) from a YAML layer template (such as [`cfg/torch/models/ConvNeXtV2.yaml`](cfg/torch/models/ConvNeXtV2.yaml)).

```bash
scm torch create model cfg/torch/models/ConvNeXtV2.yaml
scm torch create model cfg/torch/models/ConvNeXtV2.yaml -p 'DEFAULT: {backbone: femto}'
scm torch create model cfg/torch/models/ConvNeXtV2.yaml -p 'DEFAULT: {backbone: atto}' -o torch_model.py
```

#### Flax

Generate a [Flax `nnx.Module`](https://flax.readthedocs.io/en/stable/api_reference/flax.nnx/module.html) from a YAML layer template (such as [`cfg/flax/models/ConvNeXtV2.yaml`](cfg/flax/models/ConvNeXtV2.yaml)).

```bash
scm flax create model cfg/flax/models/ConvNeXtV2.yaml
scm flax create model cfg/flax/models/ConvNeXtV2.yaml -p 'DEFAULT: {backbone: femto}'
scm flax create model cfg/flax/models/ConvNeXtV2.yaml -p 'DEFAULT: {backbone: atto}' -o flax_model.py
```

#### Keras

Generate a [Keras `Layer`](https://keras.io/api/layers/base_layer/) from a YAML layer template (such as [`cfg/keras/models/ConvNeXtV2.yaml`](cfg/keras/models/ConvNeXtV2.yaml)).

```bash
scm keras create model cfg/keras/models/ConvNeXtV2.yaml
scm keras create model cfg/keras/models/ConvNeXtV2.yaml -p 'DEFAULT: {backbone: femto}'
scm keras create model cfg/keras/models/ConvNeXtV2.yaml -p 'DEFAULT: {backbone: atto}' -o keras_model.py
```

#### Common options

All three commands share the same options:

- `-p/--parameter`: override template parameters
- `-c/--classname`: set the generated class name, default `Model`
- `--no-structured-output`: force tuple-like return behavior instead of a structured output mapping
- `-s/--sublayer`: generate a named sublayer from the template instead of the root model
- `-o/--output`: output file path; if omitted, defaults to the snake-cased class name in the current directory (e.g., `model.py` for the default class name `Model`)

The ConvNeXtV2 template uses Jinja parameter groups to switch between backbone variants such as `atto`, `femto`, `tiny`, and `base`.

### 3. Generate Loss, Metric, and Backward Classes

Losses and metrics use the same `scm torch create model` command because they are also layer graphs.

```bash
scm torch create model cfg/torch/losses/cls.yaml -c Loss -o loss.py
scm torch create model cfg/torch/metrics/topk.yaml -c Metric -o metric.py
scm torch create backward cfg/torch/backwards/ConvNeXtV2.yaml -p 'DEFAULT: {epochs: 5}' -o backward.py
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

### 5. Measure Inference Time

Use `scm [torch/flax/keras] time` to benchmark the average forward-pass latency of a generated model. All three frameworks share the same basic options:

| Option             | Description                                                       |
| ------------------ | ----------------------------------------------------------------- |
| positional pattern | StructCast object pattern to instantiate the model                |
| `-s/--shape`       | Input tensor shapes, e.g. `'image: [3, 224, 224]'`                |
| `-d/--device`      | Computation device (`cpu`, `cuda`, `gpu:0`, …)                    |
| `-c/--compile`     | Compile the model before measurement (`true`, YAML path, or dict) |
| `--training-mode`  | Measure in training mode instead of evaluation mode               |
| `-w/--warmup-runs` | Number of warmup iterations (default: 2)                          |
| `-t/--times`       | Number of timed iterations (default: 10)                          |
| `-b/--batch-size`  | Batch size for dummy inputs (default: 1)                          |

#### PyTorch

```bash
scm torch create model cfg/torch/models/ConvNeXtV2.yaml \
    -p 'DEFAULT: {backbone: atto}' -o torch_model.py

scm torch time \
    '[_obj_, {_addr_: model.Model, _file_: torch_model.py}, _call_]' \
    -s 'image: [3, 224, 224]' \
    -c cfg/torch/others/compile_default.yaml \
    -d cuda
```

PyTorch-specific option: `--matmul-precision` (`highest`, `high`, `medium`) controls [`torch.set_float32_matmul_precision`](https://docs.pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html).

#### Flax

```bash
scm flax create model cfg/flax/models/ConvNeXtV2.yaml \
    -p 'DEFAULT: {backbone: atto}' -o flax_model.py

scm flax time \
    '[_obj_, {_addr_: model.Model, _file_: flax_model.py}, {_call_: {rngs: [_obj_, _addr_: flax.nnx.Rngs, _call_: {params: 0, dropout: 1}]}}]' \
    -s 'image: [224, 224, 3]' \
    -c true \
    -d gpu:0
```

Flax-specific option: `--training-mode-kwargs` lets you override the keyword arguments passed to [`nnx.view`](https://flax.readthedocs.io/en/stable/api_reference/flax.nnx/transforms.html) when `--training-mode` is set (e.g. `'{deterministic: false, use_running_average: false}'`).

> **Note:** Flax uses channel-last tensor layout. The shape `'image: [224, 224, 3]'` corresponds to *H × W × C*.

#### Keras

```bash
scm keras create model cfg/keras/models/ConvNeXtV2.yaml \
    -p 'DEFAULT: {backbone: atto}' -o keras_model.py

# Keras with JAX backend may need NVIDIA shared libraries on the path
export LD_LIBRARY_PATH=$(find .venv -name "*.so*" | grep nvidia | xargs dirname | sort -u | paste -d ":" -s -)

scm keras time \
    '[_obj_, {_addr_: model.Model, _file_: keras_model.py}, _call_]' \
    -s 'image: [224, 224, 3]' \
    -c true \
    -d gpu:0
```

Compilation for Keras uses [`keras.Model.compile`](https://keras.io/api/models/model_training_apis/#compile-method). The `--compile/-c` option accepts `true`/`false`, a YAML file path, or an inline dict of keyword arguments.

> **Note:** Keras also uses channel-last layout by default. The shape `'image: [224, 224, 3]'` corresponds to *H × W × C*.

### 6. Train a Generated Model

Below is the complete training command from the included ConvNeXtV2 example.

```bash
scm torch train \
    'model: [_obj_, {_addr_: model.Model, _file_: model.py}, _call_]' \
    -s 'image: [3, 224, 224]' \
    -d cuda \
    -L '[_obj_, {_addr_: loss.Loss, _file_: loss.py}, _call_]' \
    -M '[_obj_, {_addr_: metric.Metric, _file_: metric.py}, _call_]' \
    -B '[_obj_, {_addr_: backward.Backward, _file_: backward.py}]' \
    -c cfg/torch/others/compile_default.yaml \
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
    -A loss.py \
    -A metric.py \
    -A backward.py \
    -A cfg/torch/others/compile_default.yaml \
    -A dataset_train.yaml \
    -A dataset_valid.yaml
```

Key arguments:

- positional model patterns: one or more named model definitions
- `-s/--shape`: dummy input shapes used for model initialization
- `-d/--device`: `cpu` or `cuda`
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
3. Instantiates loss, metric, backward, and compile objects.
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
8. **Cleanup** — `torch.distributed.destroy_process_group()` is called when training finishes.

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
    -L '[_obj_, {_addr_: loss.Loss, _file_: loss.py}, _call_]' \
    -M '[_obj_, {_addr_: metric.Metric, _file_: metric.py}, _call_]' \
    -B '[_obj_, {_addr_: backward.Backward, _file_: backward.py}]' \
    -c cfg/torch/others/compile_default.yaml \
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
scm format cfg/torch/datasets/default_timm.yaml \
    -o dataset_train.yaml \
    -p 'DEFAULT: {training: true, epochs: 5, batch_size: 32, dataset: torch/cifar100, num_classes: 100, label_smoothing: 0.1, input_size: [3, 224, 224], image_dtype: bfloat16, download: true}'
```

> **Tip:** The `batch_size` in the dataset template is the **per-GPU** batch size. With 4 GPUs and `batch_size: 32`, the effective global batch size is 128.

### Distributed Training Notes

- **Seed reproducibility** — Each rank's random seed is offset by `global_rank` to ensure different data augmentation across processes while remaining reproducible.
- **Learning rate scaling** — When scaling to multiple GPUs, consider adjusting the learning rate. A common practice is [linear scaling](https://arxiv.org/abs/1706.02677): multiply the base learning rate by the number of GPUs. This must be configured in the backward template or optimizer settings — `scm torch train` does not scale the learning rate automatically.
- **SyncBatchNorm** — `scm torch train` does **not** automatically convert `BatchNorm` layers to [`SyncBatchNorm`](https://docs.pytorch.org/docs/stable/generated/torch.nn.SyncBatchNorm.html). If your model uses `BatchNorm` and you are training with DDP, consider applying `torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)` in the model definition. See the [SyncBatchNorm warning](#6-train-a-generated-model) for details.
- **`torch.compile` and DDP** — When both `--compile` and DDP are active, `torch.compile` is applied **before** DDP wrapping.
- **Checkpoint saving** — Only rank 0 saves checkpoints and logs to MLflow. When resuming from a checkpoint in a distributed setting, all ranks load the same checkpoint.

## Configuration Examples

The `cfg/` directory contains working YAML templates that demonstrate each part of the workflow. Templates are organized by framework under `cfg/torch/`, `cfg/flax/`, and `cfg/keras/`.

### PyTorch

#### `cfg/torch/models/ConvNeXtV2.yaml`

Demonstrates the model-building style used throughout the project:

- parameter groups for multiple backbone sizes
- nested user-defined layers such as `Backbone`, `Stem`, `DownSample`, and `Block`
- Jinja-driven layer expansion
- separate training and inference flow support
- structured outputs such as `{cls: torch.tensor(...), ...}`

#### `cfg/torch/backwards/ConvNeXtV2.yaml`

Demonstrates how backward logic is configured declaratively:

- `MIXED_PRECISION` for `torch.amp.GradScaler`
- `MIXED_PRECISION_TYPE` for autocast dtype
- `ACCUMULATE_GRADIENTS` for delayed optimizer updates
- optimizer creation through `structcast_model.torch.optimizers.create_with_scheduler`
- optional gradient clipping via `timm.utils.clip_grad.dispatch_clip_grad`

#### `cfg/torch/datasets/default_timm.yaml`

Formats directly into a `TimmDataLoaderWrapper.model_validate(...)` pattern. Covers:

- timm dataset construction
- timm dataloader construction
- device and prefetch settings
- mixup and cutmix options
- train or validation split generation from one template

### Flax

#### `cfg/flax/models/ConvNeXtV2.yaml`

Generates a [Flax `nnx.Module`](https://flax.readthedocs.io/en/stable/api_reference/flax.nnx/module.html) equivalent of the PyTorch ConvNeXtV2 model. The template mirrors the same parameter groups (`atto` through `huge`) and uses [`GlobalResponseNorm`](src/structcast_model/flax/layers/grn.py) as a custom Flax layer. Key differences from the PyTorch variant:

- uses channel-last tensor layout (*H × W × C*)
- constructor accepts a `rngs: flax.nnx.Rngs` argument for parameter initialization
- `__call__` propagates a `training` flag to sub-modules

### Keras

#### `cfg/keras/models/ConvNeXtV2.yaml`

Generates a [Keras `Layer`](https://keras.io/api/layers/base_layer/) equivalent of the ConvNeXtV2 model. Shares the same backbone parameter groups and uses [`GlobalResponseNormalization`](src/structcast_model/keras/layers/grn.py) as a custom Keras layer. Key differences:

- uses channel-last tensor layout (*H × W × C*)
- follows the Keras `call(self, ..., *, training=None, **kwargs)` convention
- runs on any [Keras backend](https://keras.io/getting_started/#configuring-your-backend) (JAX, PyTorch, or TensorFlow)

## Schema Reference

All configuration templates under `cfg/` follow a shared schema that controls how YAML files are parsed, rendered, and validated by the code generators. This section explains every top-level key and sub-key that appears in these templates.

### Template Parameters

Every YAML template may begin with an optional top-level `PARAMETERS` block that declares named sets of values consumed by the Jinja rendering engine.

#### `PARAMETERS`

The top-level container for all template variable groups. Any key nested inside `PARAMETERS` (other than `DEFAULT` and `SHARED`) is treated as a named group that can be selected at render time.

```yaml
PARAMETERS:
  DEFAULT:
    backbone: atto
  SHARED:
    drop_path_rate: 0.0
    num_classes: 1000
  atto:
    dims: [40, 80, 160, 320]
    depths: [2, 2, 6, 2]
  femto:
    dims: [48, 96, 192, 384]
    depths: [2, 2, 6, 2]
```

#### `DEFAULT`

Defines the default template variables. These values are active when no named group is selected and can be overridden at the command line with `-p 'DEFAULT: {key: value}'`.

```yaml
DEFAULT:
  backbone: atto
  epochs: 300
  lr: 4.0e-3
```

#### `SHARED`

Defines variables that are merged into **every** named group (including `DEFAULT`). Use `SHARED` for constants that apply to all backbone or variant choices.

```yaml
SHARED:
  stem_kernel_size: 4
  kernel_size: 7
  norm_eps: 1.0e-6
```

#### Named groups

Any key in `PARAMETERS` that is not `DEFAULT` or `SHARED` is a named parameter group — for example `atto`, `femto`, `tiny`, or `base`. A named group is activated via `_jinja_group_` and its variables (merged with `SHARED`) replace the template variables for that rendering scope.

```yaml
atto:
  dims: [40, 80, 160, 320]
  depths: [2, 2, 6, 2]
femto:
  dims: [48, 96, 192, 384]
  depths: [2, 2, 6, 2]
```

#### `_jinja_yaml_`

Embeds an inline Jinja template that is rendered and merged back into the surrounding YAML. The rendered result must itself be valid YAML. `_jinja_yaml_` blocks are evaluated with the currently active template variables and can emit any number of sibling YAML keys or list entries.

```yaml
_jinja_yaml_: |-
  {% if accumulate_gradients is none %}
  ACCUMULATE_GRADIENTS: null
  {% else %}
  ACCUMULATE_GRADIENTS: {{accumulate_gradients}}
  {% endif %}
```

Inside a `_jinja_yaml_` block you can also use standard Jinja control structures (`{% for %}`, `{% if %}`, `{% set %}`, etc.) as well as the custom filter `cumsum` (provided by `structcast_model.builders.jinja_filters`).

#### `_jinja_group_`

Selects a named parameter group from `PARAMETERS`, merging its values (together with `SHARED`) into the template variable scope for the enclosing block. `_jinja_group_` must appear alongside a `_jinja_yaml_` sibling that consumes the newly activated variables.

```yaml
- _jinja_group_: {{backbone}}
  _jinja_yaml_: |-
    - [_, cls, head, [_obj_, {_addr_: torch.nn.LazyLinear}, {_call_: {out_features: {{num_classes}}}}]]
```

When `backbone` resolves to `atto`, the `atto` group from `PARAMETERS` (merged with `SHARED`) becomes the local variable scope for the inner `_jinja_yaml_` block.

---

### Model Template Schema

The following keys appear in model configuration files such as [`cfg/torch/models/ConvNeXtV2.yaml`](cfg/torch/models/ConvNeXtV2.yaml). Each top-level key that is not `PARAMETERS` or a Jinja directive defines either the **root model** (using the reserved keys below) or a **named sublayer** (an arbitrary key whose value follows the same schema).

#### `IMPORTS`

Additional Python imports to inject at the top of the generated file. Accepts a dict mapping module names to lists of names to import, or an empty dict `{}` when no extra imports are needed.

```yaml
IMPORTS: {}
# or
IMPORTS:
  torch.nn: [Module, Linear]
  my_package.utils: null  # imports the entire module
```

#### `INPUTS`

Ordered list of tensor names that the generated `forward()` method accepts as keyword arguments. These names correspond to the first element of each `FLOW` entry and to the keys in the `inputs` dict passed at runtime.

```yaml
INPUTS: [image]
```

#### `OUTPUTS`

Ordered list of tensor names produced by the generated `forward()` method. When `STRUCTURED_OUTPUT` is `true`, these names become the keys of the returned dict; otherwise, they determine the order of the returned tuple.

```yaml
OUTPUTS: [cls]
# or, for a multi-output model:
OUTPUTS: [feat1, feat2, feat3, feat4]
```

#### `STRUCTURED_OUTPUT`

Controls the return type of the generated `forward()` method.

| Value             | Behavior                                                                 |
| ----------------- | ------------------------------------------------------------------------ |
| `true`            | Returns `{"cls": tensor, ...}` — a dict keyed by the names in `OUTPUTS`. |
| `false` (default) | Returns a plain tuple in the order of `OUTPUTS`.                         |

```yaml
STRUCTURED_OUTPUT: true
```

#### `FLOW` and `INFERENCE_FLOW`

`FLOW` is the training-time execution graph: an ordered list of `LayerBehavior` entries (see [`FLOW` entry format](#flow-entry-format) below) that describes how tensors are routed through the model's submodules.

`INFERENCE_FLOW` is an optional alternative graph used only during inference — for example, to skip `DropPath` or other training-only layers. When `INFERENCE_FLOW` is absent, inference uses `FLOW` unchanged. Both fields must produce the same `INPUTS` and `OUTPUTS`.

```yaml
FLOW:
  - [image, {feature: feat4}, backbone, {TYPE: Backbone}]
  - [feature, _, [_obj_, {_addr_: torch.nn.AdaptiveAvgPool2d}, {_call_: {output_size: 1}}]]
  - [_, cls, head, [_obj_, {_addr_: torch.nn.LazyLinear}, {_call_: {out_features: 1000}}]]

# DropPath sublayer uses a simpler inference path
DropPath:
  FLOW: [[inp, out, [_obj_, {_addr_: timm.layers.DropPath}, {_call_: {drop_prob: 0.1}}]]]
  INFERENCE_FLOW: [[inp, out]]
```

#### `FLOW` entry format

Each entry in `FLOW` or `INFERENCE_FLOW` is a `LayerBehavior` — a list of 2 to 4 elements:

```
[INPUTS, OUTPUTS]
[INPUTS, OUTPUTS, NAME_or_LAYER]
[INPUTS, OUTPUTS, NAME, LAYER]
```

| Position | Field     | Description                                                                                                                                                                                                                                                              |
| -------- | --------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 0        | `INPUTS`  | Input variable name(s) for this step. A plain string (`image`, `feat1`) reads a named tensor from the current scope. Use `_` to pass the previous step's output forward. A nested list `[[a, b]]` collects tensors from multiple sources (e.g., for residual additions). |
| 1        | `OUTPUTS` | Output variable name(s) produced by this step. Use `_` for intermediate values that need not be named. A dict `{alias: real_name}` renames the output in the current scope.                                                                                              |
| 2        | `NAME`    | (optional) A unique identifier for the generated submodule attribute. Auto-generated when omitted. Must be a valid Python identifier.                                                                                                                                    |
| 2 or 3   | `LAYER`   | (optional) The layer definition — either a StructCast `ObjectPattern` (e.g., `[_obj_, {_addr_: torch.nn.ReLU}, _call_]`) or a `UserLayer` dict (see [`TYPE`, `PARAM`, and `CFG`](#type-param-and-cfg)).                                                                  |

```yaml
FLOW:
  - [image, {feature: feat4}, backbone, {TYPE: Backbone}]
  - [feature, _, [_obj_, {_addr_: torch.nn.AdaptiveAvgPool2d}, {_call_: {output_size: 1}}]]
  - [_, _, [_obj_, {_addr_: torch.nn.Flatten}, _call_]]
  - [_, cls, head, [_obj_, {_addr_: torch.nn.LazyLinear}, {_call_: {out_features: 1000}}]]
```

#### `NAME`

`NAME` appears in two contexts:

1. **As the third element of a `FLOW` entry** — sets the Python attribute name of the generated submodule (e.g., `"block0"`, `"head"`). Must be a valid Python identifier.
2. **As a key in a `BACKWARDS` or `OPTIMIZERS` entry** — sets the generated method name for that backward or optimizer step.

```yaml
# In FLOW:
- [feat1, feat1, "block0", {TYPE: Block, PARAM: {DEFAULT: {fout: 40}}}]

# In BACKWARDS:
BACKWARDS:
  - NAME: backward
    LOSS: ce_loss
    OPTIMIZERS:
      - NAME: optimizer
        ...
```

#### `LAYER`

The fourth (or third) element of a `FLOW` entry. Defines how the submodule for this step is constructed. Two forms are accepted:

- **StructCast `ObjectPattern`** — an `[_obj_, ...]` list that constructs a standard PyTorch module:

  ```yaml
  [_obj_, {_addr_: torch.nn.LazyConv2d}, {_call_: {out_channels: 40, kernel_size: 4, stride: 4}}]
  ```

- **`UserLayer` dict** — references a sublayer defined elsewhere in the same file (via `TYPE`) or in an external file (via `CFG`):

  ```yaml
  {TYPE: Backbone}
  {TYPE: Block, PARAM: {DEFAULT: {fout: 40, drop_path: 0.0}}}
  {CFG: cfg/torch/models/my_sublayer.yaml, TYPE: MySublayer}
  ```

#### `TYPE`, `PARAM`, and `CFG`

These three keys form the `UserLayer` dict that activates a named sublayer:

| Key     | Type              | Description                                                                                                                                                                           |
| ------- | ----------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `TYPE`  | `str`             | Name of a sublayer defined as a top-level key in the same YAML file (e.g., `Backbone`, `Block`, `Stem`). The code generator expands it into a nested `nn.Module` subclass.            |
| `PARAM` | `PARAMETERS` dict | Template variable overrides passed when rendering the sublayer. Uses the same `DEFAULT` / `SHARED` / named-group structure as the top-level `PARAMETERS` block.                       |
| `CFG`   | file path         | Path to an external YAML file that defines the sublayer. Allows sublayer reuse across multiple model templates. When `CFG` is set, `TYPE` selects the sublayer name within that file. |

```yaml
# References Backbone sublayer defined in the same file, no parameter overrides
- [image, {feature: feat4}, backbone, {TYPE: Backbone}]

# References Block sublayer with per-instance parameter overrides
- [feat1, feat1, "block0", {TYPE: Block, PARAM: {DEFAULT: {fout: 40, drop_path: 0.0}}}]
```

---

### Backward Template Schema

The following keys appear in backward configuration files such as [`cfg/torch/backwards/ConvNeXtV2.yaml`](cfg/torch/backwards/ConvNeXtV2.yaml).

#### `IMPORTS`

Same format as in the model schema. Injects additional Python imports into the generated backward file.

```yaml
IMPORTS: {}
```

#### `MIXED_PRECISION`

Controls `torch.amp.GradScaler` for automatic mixed-precision training.

| Value             | Behavior                                                                                |
| ----------------- | --------------------------------------------------------------------------------------- |
| `false` (default) | AMP disabled; no `GradScaler` is created.                                               |
| `true`            | AMP enabled with default `GradScaler` settings.                                         |
| `dict`            | AMP enabled; the dict is forwarded as keyword arguments to `torch.amp.GradScaler(...)`. |

```yaml
MIXED_PRECISION:
  init_scale: "eval: 2.0**16"
  growth_factor: 2.0
  backoff_factor: 0.5
  growth_interval: 2000
  enabled: True
```

#### `MIXED_PRECISION_TYPE`

The dtype forwarded to `torch.autocast` when mixed precision is enabled. Accepts `"bfloat16"` or `"float16"`. Has no effect when `MIXED_PRECISION` is `false`.

```yaml
MIXED_PRECISION_TYPE: bfloat16
```

#### `ACCUMULATE_GRADIENTS`

The number of forward–backward steps to accumulate before calling the optimizer. Set to `null` to disable accumulation (optimizer steps every batch). When set to a positive integer `n`, `optimizer.step()` and `optimizer.zero_grad()` are called once every `n` batches.

```yaml
ACCUMULATE_GRADIENTS: null   # disabled
ACCUMULATE_GRADIENTS: 4      # accumulate over 4 steps
```

#### `BACKWARDS`

An ordered list of `BackwardBehavior` entries. Each entry defines one backward pass — i.e., one loss to differentiate and one set of optimizers to update. Multiple entries are used for multi-loss or GAN-style training where different optimizers are stepped independently.

```yaml
BACKWARDS:
  - NAME: backward
    LOSS: ce_loss
    OPTIMIZERS:
      - NAME: optimizer
        OPTIMIZER: [_obj_, ...]
        LAYERS: model
        CLIP: null
```

#### `LOSSES` and `MODELS`

Both fields default to `[]`, which instructs the code generator to infer their values automatically from the `BACKWARDS` entries.

| Key      | Type        | Description                                                                                                                                                           |
| -------- | ----------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `LOSSES` | `list[str]` | Explicit list of loss key names that the generated backward class tracks. Auto-inferred from `BACKWARDS[*].LOSS` when left as `[]`.                                   |
| `MODELS` | `list[str]` | Explicit list of model names the generated backward class expects as constructor arguments. Auto-inferred from `BACKWARDS[*].OPTIMIZERS[*].LAYERS` when left as `[]`. |

```yaml
LOSSES: []   # auto-inferred
MODELS: []   # auto-inferred
```

#### `BACKWARDS` entry keys

Each entry in `BACKWARDS` is a `BackwardBehavior` with the following fields:

| Key          | Type   | Description                                                                                                                                           |
| ------------ | ------ | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| `NAME`       | `str`  | Optional identifier for this backward pass. Used as the generated class or method name. Must be a valid Python identifier.                            |
| `LOSS`       | `str`  | The loss key (matching a key returned by the loss module) that this backward pass differentiates.                                                     |
| `OPTIMIZERS` | `list` | One or more `OptimizerBehavior` entries that are executed in order during each training step (see [`OPTIMIZERS` entry keys](#optimizers-entry-keys)). |

```yaml
BACKWARDS:
  - NAME: backward
    LOSS: ce_loss
    OPTIMIZERS:
      - NAME: optimizer
        OPTIMIZER: [_obj_, ...]
        LAYERS: model
```

#### `OPTIMIZERS` entry keys

Each entry in `OPTIMIZERS` is an `OptimizerBehavior` with the following fields:

| Key         | Type                         | Description                                                                                                                                                                                                                                       |
| ----------- | ---------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `NAME`      | `str`                        | Optional identifier for this optimizer entry. Used as the generated attribute name. Must be a valid Python identifier.                                                                                                                            |
| `OPTIMIZER` | StructCast pattern           | A StructCast `ObjectPattern` that constructs the optimizer (and optionally its learning-rate scheduler). Commonly uses `structcast_model.torch.optimizers.create_with_scheduler` with `_bind_` to pass `optimizer_kwargs` and `scheduler_kwargs`. |
| `LAYERS`    | `str` or `list[str]`         | Model parameter paths that this optimizer manages. Each value must be a valid Python attribute expression (e.g., `model` or `model.backbone`). The generated backward class calls `optimizer.param_groups` over these paths.                      |
| `CLIP`      | StructCast pattern or `null` | Optional gradient-clipping callable. When non-null, the pattern is bound once and called before each optimizer step with the parameters identified by `LAYERS`. Set to `null` to disable gradient clipping.                                       |

```yaml
OPTIMIZERS:
  - NAME: optimizer
    OPTIMIZER:
      - _obj_
      - _addr_: structcast_model.torch.optimizers.create_with_scheduler
      - _bind_:
          optimizer_kwargs:
            opt: adamw
            lr: 4.0e-3
            weight_decay: 0.001
          scheduler_kwargs:
            name: cosine
            num_epochs: 300
    LAYERS: model
    CLIP:
      - _obj_
      - _addr_: timm.utils.clip_grad.dispatch_clip_grad
      - _bind_: {value: 1.0, mode: norm, norm_type: 2.0}
```

---

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

Optional fields: `validation_step`, `training_prefix` (default `""`), `validation_prefix` (default `"val_"`).

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

The dataset template at `cfg/torch/datasets/default_timm.yaml` formats into this wrapper.

## Minimal End-to-End Example

```bash
uv sync --extra torch-cu130 --extra mlflow --extra flops

scm torch create model cfg/torch/models/ConvNeXtV2.yaml -p 'DEFAULT: {backbone: femto}' -o model.py
scm torch create model cfg/torch/losses/cls.yaml -c Loss -o loss.py
scm torch create model cfg/torch/metrics/topk.yaml -c Metric -o metric.py
scm torch create backward cfg/torch/backwards/ConvNeXtV2.yaml -p 'DEFAULT: {epochs: 5}' -o backward.py

scm format cfg/torch/datasets/default_timm.yaml \
    -o dataset_train.yaml \
    -p 'DEFAULT: {training: true, epochs: 5, batch_size: 32, dataset: torch/cifar100, num_classes: 100, label_smoothing: 0.1, input_size: [3, 224, 224], image_dtype: bfloat16, download: true}'

scm format cfg/torch/datasets/default_timm.yaml \
    -o dataset_valid.yaml \
    -p 'DEFAULT: {training: false, epochs: 5, batch_size: 32, dataset: torch/cifar100, num_classes: 100, input_size: [3, 224, 224], image_dtype: bfloat16, download: true}'

scm torch train \
    'model: [_obj_, {_addr_: model.Model, _file_: model.py}, _call_]' \
    -s 'image: [3, 224, 224]' \
    -d cuda \
    -L '[_obj_, {_addr_: loss.Loss, _file_: loss.py}, _call_]' \
    -M '[_obj_, {_addr_: metric.Metric, _file_: metric.py}, _call_]' \
    -B '[_obj_, {_addr_: backward.Backward, _file_: backward.py}]' \
    -c cfg/torch/others/compile_default.yaml \
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
- [x] JAX (Flax) model construction from YAML configuration files
- [ ] JAX (Flax) training workflow generation from YAML configuration files
- [x] Keras model construction from YAML configuration files
- [ ] Keras training workflow generation from YAML configuration files
