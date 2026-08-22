# StructCast-Model

StructCast-Model is a configuration-driven toolkit that generates [PyTorch](https://pytorch.org/), [Flax (JAX)](https://flax.readthedocs.io/en/stable/), and [Keras](https://keras.io/) models — plus PyTorch training workflows — from YAML templates. Built on top of [StructCast](https://github.com/f6ra07nk14/structcast), it lets you describe model architecture, optimizer logic, dataset configuration, and training orchestration declaratively — then generates runnable Python code from those descriptions.

Model code generation, training workflow generation and the full training CLI are available for all three frameworks (`scm torch train`, `scm flax train`, `scm keras train`); a Keras run names the backend it executes on with `--backend`.

## Table of Contents

- [StructCast-Model](#structcast-model)
  - [Table of Contents](#table-of-contents)
  - [What This Project Does](#what-this-project-does)
  - [Installation](#installation)
  - [Project Structure](#project-structure)
  - [Core Workflow](#core-workflow)
  - [StructCast Pattern Basics](#structcast-pattern-basics)
  - [Quick Start](#quick-start)
  - [Command Guide](#command-guide)
    - [1. Format Templates](#1-format-templates)
    - [2. Generate a Model Class](#2-generate-a-model-class)
    - [3. Generate a Learner Class](#3-generate-a-learner-class)
    - [4. Inspect FLOPs and Parameters](#4-inspect-flops-and-parameters)
    - [5. Measure Inference Time](#5-measure-inference-time)
    - [6. Train a Generated Model](#6-train-a-generated-model)
      - [Distributed Training with `torchrun`](#distributed-training-with-torchrun)
        - [How It Works](#how-it-works)
        - [Single-Node Multi-GPU](#single-node-multi-gpu)
        - [Multi-Node Training](#multi-node-training)
        - [Dataset Configuration](#dataset-configuration)
        - [Distributed Training Notes](#distributed-training-notes)
    - [7. Train a Flax Model](#7-train-a-flax-model)
    - [8. Train a Keras Model](#8-train-a-keras-model)
  - [Training Loop Anatomy](#training-loop-anatomy)
  - [Configuration Examples](#configuration-examples)
    - [PyTorch](#pytorch)
    - [Flax](#flax)
    - [Keras](#keras)
  - [Development](#development)
  - [Migration Notes](#migration-notes)
    - [Upgrading from v1.x](#upgrading-from-v1x)
  - [Roadmap](#roadmap)

## What This Project Does

- **Generate model code** — Produce PyTorch [`nn.Module`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html), Flax [`nnx.Module`](https://flax.readthedocs.io/en/stable/api_reference/flax.nnx/module.html), and Keras [`Layer`](https://keras.io/api/layers/base_layer/) classes from YAML layer templates.
- **Generate training code** — Produce learner classes — the object owning the models, the optimizers, and the training and inference steps — from YAML templates (PyTorch, Flax and Keras).
- **Format reusable templates** — Render parameterized YAML templates into concrete runtime configurations.
- **Inspect model complexity** — Compute FLOPs and parameter counts with [`ptflops`](https://github.com/sovrasov/flops-counter.pytorch) and [`calflops`](https://github.com/MrYxJ/calculate-flops.pytorch) (PyTorch only).
- **Measure inference time** — Benchmark average forward-pass latency of generated models across all three frameworks via `scm [torch/flax/keras] time`.
- **Train end-to-end** — Run PyTorch training with [Automatic Mixed Precision (AMP)](https://docs.pytorch.org/docs/stable/amp.html), [timm](https://github.com/huggingface/pytorch-image-models) datasets, optional [`torch.compile`](https://docs.pytorch.org/tutorials/intermediate/torch_compile_tutorial.html), and [MLflow](https://mlflow.org/docs/latest/ml/deep-learning/pytorch/) or [Weights & Biases](https://docs.wandb.ai/) experiment logging — or Flax training on a JAX device mesh with [`nnx.jit`](https://flax.readthedocs.io/en/stable/api_reference/flax.nnx/transforms.html) compilation and the same loggers.
- **Train programmatically** — Use the same trainer directly from Python, without any YAML. See [`examples/`](examples/) for a runnable tutorial.

## Installation

StructCast-Model is installed with [uv](https://docs.astral.sh/uv/) and exposes the `scm` CLI entry point.

```bash
uv sync --extra torch-cu130 --extra mlflow --extra flops
```

Each extra installs a group of optional dependencies. Pick the extras that match your target framework and accelerator. [Keras](https://keras.io/) is multi-backend and runs on top of JAX, PyTorch, or TensorFlow.

| Category   | Extra           | What it provides                                                               |
| ---------- | --------------- | ------------------------------------------------------------------------------ |
| PyTorch    | `torch-cpu`     | PyTorch and torchvision (CPU only)                                             |
|            | `torch-cu130`   | PyTorch and torchvision with CUDA 13.0 support                                 |
| JAX / Flax | `jax-cpu`       | JAX and Flax (CPU only)                                                        |
| Keras      | `keras-jax-cpu` | Keras with JAX (CPU)                                                           |
| Bundles    | `all-cpu`       | JAX + Flax, PyTorch + torchvision + timm, TensorFlow, and Keras — all CPU-only |
|            | `all-cuda`      | Same as `all-cpu` but with CUDA acceleration for every backend                 |
| Tools      | `flops`         | Both `ptflops` and `calflops` for complexity inspection                        |
|            | `mlflow`        | MLflow experiment tracking for `scm torch train --logger mlflow`               |
|            | `wandb`         | Weights & Biases tracking for `scm torch train --logger wandb`                 |

<details>
<summary><strong>All available extras</strong></summary>

| Category   | Extra               | What it provides                                                               |
| ---------- | ------------------- | ------------------------------------------------------------------------------ |
| PyTorch    | `torch-cpu`         | PyTorch and torchvision (CPU only)                                             |
|            | `torch-cu118`       | PyTorch and torchvision with CUDA 11.8 support                                 |
|            | `torch-cu126`       | PyTorch and torchvision with CUDA 12.6 support                                 |
|            | `torch-cu128`       | PyTorch and torchvision with CUDA 12.8 support                                 |
|            | `torch-cu130`       | PyTorch and torchvision with CUDA 13.0 support                                 |
| JAX / Flax | `jax-cpu`           | JAX and Flax (CPU only)                                                        |
|            | `jax-cu12`          | JAX and Flax with CUDA 12 support                                              |
|            | `jax-cu13`          | JAX and Flax with CUDA 13 support                                              |
| TensorFlow | `tf-cpu`            | TensorFlow (CPU only)                                                          |
|            | `tf-cu12`           | TensorFlow with CUDA 12 support                                                |
| Keras      | `keras-jax-cpu`     | Keras with JAX (CPU)                                                           |
|            | `keras-jax-cu12`    | Keras with JAX (CUDA 12)                                                       |
|            | `keras-jax-cu13`    | Keras with JAX (CUDA 13)                                                       |
|            | `keras-torch-cpu`   | Keras with PyTorch (CPU)                                                       |
|            | `keras-torch-cu118` | Keras with PyTorch (CUDA 11.8)                                                 |
|            | `keras-torch-cu126` | Keras with PyTorch (CUDA 12.6)                                                 |
|            | `keras-torch-cu128` | Keras with PyTorch (CUDA 12.8)                                                 |
|            | `keras-torch-cu130` | Keras with PyTorch (CUDA 13.0)                                                 |
|            | `keras-tf-cpu`      | Keras with TensorFlow (CPU)                                                    |
|            | `keras-tf-cu12`     | Keras with TensorFlow (CUDA 12)                                                |
| Bundles    | `all-cpu`           | JAX + Flax, PyTorch + torchvision + timm, TensorFlow, and Keras — all CPU-only |
|            | `all-cuda`          | Same as `all-cpu` but with CUDA acceleration for every backend                 |
| Tools      | `ptflops`           | `ptflops` for model complexity inspection                                      |
|            | `calflops`          | `calflops` and Transformers for complexity inspection                          |
|            | `flops`             | Both `ptflops` and `calflops`                                                  |
|            | `mlflow`            | MLflow experiment tracking for `scm torch train --logger mlflow`               |
|            | `wandb`             | Weights & Biases tracking for `scm torch train --logger wandb`                 |

> - [**ptflops**](https://github.com/sovrasov/flops-counter.pytorch): a popular FLOPs and parameter counting library for PyTorch models. It provides detailed breakdowns of computational complexity per layer and supports custom layer definitions through a registration mechanism. StructCast-Model uses `ptflops` to analyze generated PyTorch models and report their FLOPs and parameter counts.
> - [**calflops**](https://github.com/MrYxJ/calculate-flops.pytorch): a FLOPs and parameter counting library for PyTorch models, similar to `ptflops`.
> - [**MLflow**](https://mlflow.org/): an open-source platform for managing the ML lifecycle, including experimentation, reproducibility, and deployment. StructCast-Model integrates with MLflow to log training metrics, model checkpoints, and configuration artifacts from `scm torch train`.
> - [**Weights & Biases**](https://wandb.ai/): a hosted experiment tracking service. It is the alternative backend of `scm torch train`, selected with `--logger wandb`, and receives the same metrics, artifacts, and state dictionaries as the MLflow backend.

</details>

Omit any extra you do not need. For example, `uv sync --extra torch-cu130` is sufficient if you only want to generate and train PyTorch models without FLOPs analysis or MLflow logging. To work with all three model frameworks on CPU:

```bash
uv sync --extra all-cpu
```

## Project Structure

```text
structcast-model/
├── cfg/
│   ├── torch/
│   │   ├── learners/      # learner, optimizer, loss, and metric templates
│   │   ├── models/        # model architecture templates
│   │   └── others/        # dataset, compile options, and other templates
│   ├── flax/
│   │   ├── learners/      # Flax learner, optimizer, and criterion templates
│   │   ├── models/        # Flax model architecture templates
│   │   └── strategies/    # device-mesh strategy patterns for `scm flax train`
│   └── keras/
│       ├── models/        # Keras model architecture templates
│       └── strategies/    # distributed strategy patterns for `scm keras train`
├── examples/
│   └── torch/         # runnable training tutorial and optimizer compositions
├── src/structcast_model/
│   ├── builders/      # generic and framework-specific code generators
│   ├── commands/      # Typer CLI entry points
│   ├── torch/         # trainer, layers, optimizer helpers
│   ├── flax/          # trainer, distributed strategy, layers, optimizer helpers
│   ├── keras/         # trainer, backend adapters, distributed strategy, layers
│   ├── loggers/       # experiment-tracking loggers and training-state backends
│   ├── utils/         # shared helpers
│   └── base_trainer.py
├── tests/             # CLI, builder, trainer, and layer tests
└── README.md
```

The main package areas are:

- **`builders/`** — Converts validated YAML templates into intermediate representations, then renders Python source code for PyTorch, Flax, and Keras.
- **`commands/`** — Exposes the `scm` CLI (built with [Typer](https://typer.tiangolo.com/)) with `torch`, `flax`, and `keras` sub-commands.
- **`torch/`** — Runtime utilities used by the CLI and available for direct Python usage — training steps, trackers, timm wrappers, optimizer helpers.
- **`flax/`** — Runtime for Flax runs: the trainer and its tracker, the device-mesh strategy, optimizer-state helpers, Flax-specific layers (e.g. `GlobalResponseNorm`), and JAX inference helpers.
- **`keras/`** — Runtime for Keras runs: the trainer and its tracker, the per-backend adapters the training step runs through, the distributed strategy, Keras-specific layers (e.g. `GlobalResponseNormalization`), and backend-agnostic inference helpers.
- **`loggers/`** — The MLflow and Weights & Biases loggers owning a run, and the state backends deciding what a saved training state looks like on disk.
- **`cfg/torch/`** — Declarative source of truth: YAML templates for PyTorch models, learners, datasets, and runtime presets.
- **`examples/torch/`** — Runnable example code: a programmatic training tutorial, and optimizer + scheduler compositions that templates reference by file path.
- **`cfg/flax/`** — YAML templates for Flax models, learners, and device-mesh strategies.
- **`cfg/keras/`** — YAML templates for Keras models, learners, datasets, and distributed strategies.
- **`examples/keras/`** — Runnable example code: a programmatic training tutorial, a `tf.data` input pipeline, a text corpus, and the optimizer factory templates reference by file path.

## Core Workflow

The repository follows a repeatable workflow:

1. **Write or reuse** YAML templates under `cfg/[torch/flax/keras]/`.
2. **Render** templates with `scm format` and `-p/--parameter` overrides to produce concrete configuration files.
3. **Generate** Python source files for the model and the learner using `scm [torch/flax/keras] create`.
4. **Instantiate** those generated modules at runtime through StructCast object patterns (see [StructCast Pattern Basics](#structcast-pattern-basics)).
5. **Benchmark** inference latency with `scm [torch/flax/keras] time`.
6. **Train** through `scm torch train`, `scm flax train` or `scm keras train`, which wires together datasets, models, the learner, the device placement, and the experiment logger.

```text
YAML templates  --->  scm format / scm [torch/flax/keras] create  --->  Generated .py files
                                                                              |
StructCast patterns  <--------------------------------------------------------+
       |
       v
scm [torch/flax/keras] time  --->  Inference benchmarks
scm [torch/flax] train       --->  MLflow / wandb logs + model checkpoints
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

## Quick Start

The following commands generate a ConvNeXtV2 model along with its learner and dataset configurations, then launch a training run on CIFAR-100.

```bash
# 1. Install
uv sync --extra torch-cu130 --extra mlflow --extra flops

# 2. Generate the model and the learner classes
scm torch create model cfg/torch/models/ConvNeXtV2.yaml -p 'DEFAULT: {backbone: femto}' -o model.py
scm torch create learner cfg/torch/learners/ConvNeXtV2.yaml -p 'DEFAULT: {epochs: 5}' -o learner.py

# 3. Render dataset configurations from templates
scm format cfg/torch/others/default_timm.yaml \
    -o dataset_train.yaml \
    -p 'DEFAULT: {training: true, epochs: 5, batch_size: 32, dataset: torch/cifar100, num_classes: 100, label_smoothing: 0.1, input_size: [3, 224, 224], image_dtype: bfloat16, download: true}'

scm format cfg/torch/others/default_timm.yaml \
    -o dataset_valid.yaml \
    -p 'DEFAULT: {training: false, epochs: 5, batch_size: 32, dataset: torch/cifar100, num_classes: 100, input_size: [3, 224, 224], image_dtype: bfloat16, download: true}'

# 4. Train
scm torch train \
    'model: [_obj_, {_addr_: model.Model, _file_: model.py}, _call_]' \
    -s 'image: [3, 224, 224]' \
    -d cuda \
    -L '[_obj_, {_addr_: learner.Learner, _file_: learner.py}]' \
    -c cfg/torch/others/compile_default.yaml \
    -e 5 \
    --training-dataset dataset_train.yaml \
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

Each step is explained in detail under [Command Guide](#command-guide). To see the same training run built in plain Python instead of YAML, start from [`examples/`](examples/) and run `uv run python examples/torch/simple_training.py`.

## Command Guide

### 1. Format Templates

Use `scm format` to render a parameterized YAML template (such as [`cfg/torch/others/default_timm.yaml`](cfg/torch/others/default_timm.yaml)) into a concrete configuration file.

```bash
scm format cfg/torch/others/default_timm.yaml \
    -o dataset_train.yaml \
    -p 'DEFAULT: {training: true, epochs: 5, batch_size: 32, dataset: torch/cifar100, num_classes: 100, label_smoothing: 0.1, input_size: [3, 224, 224], image_dtype: bfloat16, download: true}'

scm format cfg/torch/others/default_timm.yaml \
    -o dataset_valid.yaml \
    -p 'DEFAULT: {training: false, epochs: 5, batch_size: 32, dataset: torch/cifar100, num_classes: 100, input_size: [3, 224, 224], image_dtype: bfloat16, download: true}'
```

What this does:

1. Loads the YAML template.
2. Merges any repeated `-p/--parameter` groups into a single parameter set.
3. Renders Jinja-based sections within the template.
4. Writes the resolved YAML to `-o/--output` (or prints to stdout if `-o` is omitted).

### 2. Generate a Model Class

Each framework has its own `create model` command that reads a YAML layer template and generates a framework-native module. The examples below use PyTorch; Flax and Keras share the same interface with minor differences noted afterward.

```bash
scm torch create model cfg/torch/models/ConvNeXtV2.yaml
scm torch create model cfg/torch/models/ConvNeXtV2.yaml -p 'DEFAULT: {backbone: femto}'
scm torch create model cfg/torch/models/ConvNeXtV2.yaml -p 'DEFAULT: {backbone: atto}' -o torch_model.py
```

**Common options** — All three framework commands share the same options:

- `-p/--parameter`: override template parameters
- `-n/--classname`: set the generated class name, default `Model`
- `--structured-output/--no-structured-output`: force the root model's return type. `scm torch` defaults to the template's `STRUCTURED_OUTPUT` (a plain tuple-like return unless the template sets it); `scm flax` and `scm keras` default to a structured output mapping
- `--sublayer`: generate a named sublayer from the template instead of the root model
- `-o/--output`: output file path; if omitted, defaults to the snake-cased class name in the current directory (e.g., `model.py` for the default class name `Model`)

The ConvNeXtV2 template uses Jinja parameter groups to switch between backbone variants such as `atto`, `femto`, `tiny`, and `base`.

> **Flax and Keras** — Replace `scm torch` with `scm flax` or `scm keras`. Templates live under `cfg/flax/models/` and `cfg/keras/models/` respectively. Flax generates [`nnx.Module`](https://flax.readthedocs.io/en/stable/api_reference/flax.nnx/module.html) classes; Keras generates [`Layer`](https://keras.io/api/layers/base_layer/) classes. Both use channel-last tensor layout (*H × W × C*) instead of PyTorch's channel-first (*C × H × W*).

### 3. Generate a Learner Class

The learner is the object that owns the models and defines how they learn: when an update happens, how a training step runs, and how an inference step runs. Losses and metrics are part of it — they are declared inline in the learner's flow, so there is no separate loss or metric command.

```bash
scm torch create learner cfg/torch/learners/ConvNeXtV2.yaml -p 'DEFAULT: {epochs: 5}' -o learner.py
```

Options: `-p/--parameter` overrides template parameters, `-n/--classname` sets the generated class name (default `Learner`), and `-o/--output` sets the output path.

The generated class manages:

- a training-time execution graph (`FLOW`) and an inference-time execution graph (`INFERENCE_FLOW`) per learner entry
- inline layer instantiation (loss layers, metric layers, and arbitrary modules can be defined directly in the flow)
- one or more `LEARNERS` entries, each with its own optimizer and trainable layers — enabling multi-optimizer training (e.g., GAN generator + discriminator)
- optimizer construction via StructCast patterns, including file-addressed optimizer + scheduler compositions such as [`examples/torch/optimizers.py`](examples/torch/optimizers.py)
- optional gradient scaler creation (`MIXED_PRECISION`)
- optional gradient clipping (`CLIP`)
- optional gradient accumulation (`ACCUMULATE_GRADIENTS`, a torch-only key — see [`docs/adr/0017`](docs/adr/0017-accumulation-gating-follows-each-backends-native-mechanism.md))
- optimizer stepping, zeroing, and automatic train/eval mode switching
- learning-rate and parameter-group inspection helpers

The result implements the `Learner` protocol — the `models`, `optimizers`, `optimizer_models`, `flow_functions`, `learning_rates`, `steps`, `updates`, and `has_updated` properties plus `restore_counters`, `training_step`, and `inference_step` — and the optional `grad_scalers`, `weight_decays`, and `param_group_names` properties the toolkit reads when present (the loggers merge `learning_rates` and `weight_decays` into the epoch metrics). Any object with those members can be trained, generated or hand-written; see [`examples/torch/simple_training.py`](examples/torch/simple_training.py).

For example, a CycleGAN learner template defines three `LEARNERS` entries — one for the generator pair and one for each discriminator — each with its own flow, optimizer, and trainable layers:

```bash
scm torch create learner cfg/torch/learners/CycleGAN.yaml -o learner.py
```

> **Flax** — `scm flax create learner` reads a template under [`cfg/flax/learners/`](cfg/flax/learners/) and generates the Flax counterpart:
>
> ```bash
> scm flax create learner cfg/flax/learners/ConvNeXtV2.yaml -o learner.py
> ```
>
> It takes the same `-p/--parameter`, `-n/--classname`, and `-o/--output` options. The template schema is the shared one minus the torch-only keys: `CLIP` and `MIXED_PRECISION` are rejected, because in Flax clipping is a stage of the [optax](https://optax.readthedocs.io/) chain written inside `OPTIMIZER` and precision is a model-construction property. `ACCUMULATE_GRADIENTS` is rejected too: gradient accumulation is an `optax.MultiSteps` wrapper in that same chain, whose window the generated `__init__` reads back from the built optimizers to validate; the learner then counts real applies by reading `MultiStepsState.gradient_step` after each step ([`docs/adr/0017`](docs/adr/0017-accumulation-gating-follows-each-backends-native-mechanism.md), [`docs/adr/0018`](docs/adr/0018-the-learner-owns-the-training-counters.md)). The `OPTIMIZER` pattern builds an [`nnx.Optimizer`](https://flax.readthedocs.io/en/stable/api_reference/flax.nnx/training/optimizer.html) over the entry's trainable layers, and the builder wraps the factory carrying `learning_rate` in [`optax.inject_hyperparams`](https://optax.readthedocs.io/en/latest/api/utilities.html#optax.inject_hyperparams) so the rate stays readable at run time. The generated class implements the `Learner` protocol — its `flow_functions` being the module-level step functions a trainer compiles — and adds `outputs`, but no `grad_scalers`, `weight_decays`, or `param_group_names`.

> **Keras** — `scm keras create learner` generates a backend-neutral learner class from the same schema:
>
> ```bash
> scm keras create learner learner.yaml -o learner.py
> ```
>
> It takes the same `-p/--parameter`, `-n/--classname`, and `-o/--output` options, and no `--backend`: nothing it runs imports Keras, so the generated file is the same on all three backends and the backend is chosen when it is trained. `CLIP` and `ACCUMULATE_GRADIENTS` are rejected — clipping and gradient accumulation are both keyword arguments of the Keras optimizer written inside `OPTIMIZER`; accumulation is `gradient_accumulation_steps`, whose applies the generated learner detects by reading the optimizer's own step counter after each step, so the update count tracks real applies ([`docs/adr/0017`](docs/adr/0017-accumulation-gating-follows-each-backends-native-mechanism.md), [`docs/adr/0018`](docs/adr/0018-the-learner-owns-the-training-counters.md)) — while `MIXED_PRECISION` turns on a global [`keras.mixed_precision`](https://keras.io/api/mixed_precision/) policy of `MIXED_PRECISION_TYPE` (`float16` loss-scales the optimizer, `bfloat16` does not). The [Keras configuration example](#keras) below shows the shape of a template.

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

**PyTorch** example:

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

> **Flax and Keras** — Replace `scm torch` with `scm flax` or `scm keras`. Both use channel-last shapes (e.g., `'image: [224, 224, 3]'`). Flax additionally accepts `--training-mode-kwargs` to override keyword arguments for [`nnx.view`](https://flax.readthedocs.io/en/stable/api_reference/flax.nnx/transforms.html). Keras compilation uses [`keras.Model.compile`](https://keras.io/api/models/model_training_apis/#compile-method). When using the Keras JAX backend on GPU, you may need to set `LD_LIBRARY_PATH` to include NVIDIA shared libraries from your virtual environment.

### 6. Train a Generated Model

Below is the complete training command from the included ConvNeXtV2 example.

```bash
scm torch train \
    'model: [_obj_, {_addr_: model.Model, _file_: model.py}, _call_]' \
    -s 'image: [3, 224, 224]' \
    -d cuda \
    -L '[_obj_, {_addr_: learner.Learner, _file_: learner.py}]' \
    -c cfg/torch/others/compile_default.yaml \
    -e 5 \
    --training-dataset dataset_train.yaml \
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
    --logger mlflow \
    -E Test \
    -A model.py \
    -A learner.py \
    -A cfg/torch/others/compile_default.yaml \
    -A dataset_train.yaml \
    -A dataset_valid.yaml
```

Key arguments:

- positional model patterns: one or more named model definitions
- `-s/--shape`: dummy input shapes used for model initialization
- `-d/--device`: `cpu` or `cuda`
- `-L/--learner`: StructCast pattern for the learner class; it is called with the instantiated models as keyword arguments
- `-LO/--learner-outputs`: criterion names to track, when the learner exposes no `outputs` attribute
- `-c/--compile`: boolean, YAML file, or inline dict for `torch.compile`
- `--training-dataset`: training dataset pattern or rendered dataset YAML
- `-V/--validation-dataset`: validation dataset pattern or rendered dataset YAML; omit it to skip validation
- `-f/--validation-frequency`: run validation every N epochs
- `-LC/--lower-criterion`: criteria where lower is better
- `-HC/--higher-criterion`: criteria where higher is better
- `-SC/--save-criterion`: criteria that should trigger best-model saving
- `--logger`: experiment tracking service, `mlflow` (default) or `wandb`
- `-E/--experiment`: experiment name passed to the logger
- `-A/--log-artifacts`: files to store as run artifacts
- `--trainer`: StructCast pattern for a `TorchTrainer` replacement, when the default loop is not enough
- `--strategy`: StructCast pattern for the `DistributedStrategy`; it is called with the resolved `device` and `local_rank`. Defaults to `DistributedDataParallelStrategy` when a distributed environment is detected, and `SingleDeviceStrategy` otherwise
- `--resume`: training state to restore before the loop starts; the reference is resolved by the active `--logger`, so a local path always works, a `runs:/<run_id>/<artifact>` MLflow URI requires `--logger mlflow`, and a `wandb://<entity>/<project>/<run_id>/<file>` reference requires `--logger wandb` — resuming across services is not supported. Models, optimizers, and gradient scalers are restored and training continues from the saved epoch (`--start-epoch` is overridden, with a warning)

What the train command does internally:

1. Instantiates the datasets and composes them into a `SimpleDataProvider`, which reports `steps_per_epoch` and `validation_steps`. The trainer scans the provider datasets for event protocols, so a dataset implementing one receives the lifecycle events it defines.
2. Builds the models from their patterns on the training device, initializes them with optional dummy-input forward passes, applies the initializers on rank 0 and broadcasts the result (`sync_initial_weights`), then compiles each model where the strategy places the units and hands it to the strategy, which wraps it. The learner is built from the already-wrapped models.
3. Builds a `TorchTracker` from the learner's output names, still inside the device scope so its buffers live on the training device.
4. Compiles the learner's generated `_flow_*` functions — the pure-compute part of each step — on a single device only. `train()`/`eval()`, backward, optimizer steps, and `zero_grad()` stay eager.
5. Creates the `TorchTrainer` with the learner, the tracker, and the data provider.
6. Collects the callbacks from the trainer's prefixes: a `ProgressBar` (or a `Printer` under `--ci`) and the logger on rank 0 only, plus a training-state saver and one `TorchBestCriterion` per monitored criterion on every rank — producing their states is a collective, and off rank 0 they hold a `NullLogger` and write nothing. They join the trainer's events on first use, and the resulting routing is printed.
7. Runs `fit()` inside the logger's run context, recording metrics, arguments, model states, optimizer states, gradient scaler states, and best checkpoints.

#### Distributed Training with `torchrun`

`scm torch train` supports multi-GPU and multi-node [distributed data parallel (DDP)](https://docs.pytorch.org/tutorials/beginner/dist_overview.html) training out of the box via [`torchrun`](https://docs.pytorch.org/docs/stable/elastic/run.html). No changes to your generated code, YAML templates, or dataset configurations are required — the same `scm torch train` command works for both single-GPU and distributed training.

> **⚠️ SyncBatchNorm**
>
> Standard `BatchNorm` computes statistics per-GPU, which can cause inconsistent behavior across ranks — especially with small per-GPU batch sizes. `scm torch train` therefore converts every `BatchNorm` layer to [`SyncBatchNorm`](https://docs.pytorch.org/docs/stable/generated/torch.nn.SyncBatchNorm.html) for you, inside the distributed strategy's `wrap()` and before DDP wrapping or FSDP2 sharding. The converter is [`timm.layers.convert_sync_batchnorm`](https://github.com/huggingface/pytorch-image-models/blob/main/timm/layers/norm_act.py), not torch's: timm's fused `BatchNormAct2d` — the default norm layer of the efficientnet/mobilenet/regnet families — becomes a `SyncBatchNormAct` that keeps its activation, where torch's stock converter would replace it with a plain `SyncBatchNorm` and silently drop that activation. Do not call [`torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)`](https://docs.pytorch.org/docs/stable/generated/torch.nn.SyncBatchNorm.html#torch.nn.SyncBatchNorm.convert_sync_batchnorm) yourself in the model definition.
>
> - **On by default** for `DistributedDataParallelStrategy` and `FullyShardedDataParallelStrategy`; `SingleDeviceStrategy` never converts. The only off-switch is the strategy pattern's YAML — `_bind_: {sync_batchnorm: false}`; there is no CLI flag for it.
> - **Skipped on CPU devices.** `SyncBatchNorm`'s training forward rejects CPU input whenever `torch.distributed` is initialized, even with a single rank, so a CPU run keeps its plain `BatchNorm` layers.
> - **The rank-0 weight broadcast survives.** The conversion carries parameters and buffers over by reference, so the values synchronized before wrapping stay authoritative.
> - **`torch.compile` graph-breaks on `SyncBatchNorm`** ([pytorch#161302](https://github.com/pytorch/pytorch/issues/161302)): a converted model under `--compile` pays that break.
> - **Layers that already are `SyncBatchNorm` are left alone**, `SyncBatchNormAct` and a hand-built `process_group` included. The conversion is idempotent, so a model that converts itself keeps working exactly as it did — nothing is re-created, nothing is reset.
> - **A replaced layer is a new object, so anything attached to the old one is dropped.** Hooks you registered on a `BatchNorm` layer before the strategy wraps the models do not survive its replacement — that holds for every converted layer, not just the model root.
> - **A `BatchNorm` layer that is also a compile unit loses its compilation.** `--compile` runs *before* `wrap()` and compiles in place — the model root by default, the matched `shard_modules` blocks under FSDP2 — so a model whose root *is* a `BatchNorm` layer, or a `shard_modules` pattern matching one, is compiled and then replaced.

##### How It Works

When launched through `torchrun`, the environment variables `RANK`, `LOCAL_RANK`, `WORLD_SIZE`, `MASTER_ADDR`, and `MASTER_PORT` are set automatically. `scm torch train` detects these and enables distributed mode:

1. **Process group initialization** — The NCCL backend is initialized via [`torch.distributed.init_process_group`](https://docs.pytorch.org/docs/stable/distributed.html#torch.distributed.init_process_group).
2. **Per-rank device assignment** — Each process is assigned to `cuda:<LOCAL_RANK>`.
3. **Strategy model wrapping** — Every model is wrapped by the selected `DistributedStrategy` before the learner is built. The default in a distributed environment is [`DistributedDataParallel`](https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html); `SingleDeviceStrategy`, `FullyShardedDataParallelStrategy` ([FSDP2](https://docs.pytorch.org/docs/stable/distributed.fsdp.fully_shard.html), requires `torch>=2.6`), `TensorParallelStrategy` and the FSDP2+TP combination ([`cfg/torch/strategies/tp.yaml`](cfg/torch/strategies/tp.yaml), [`fsdp_tp.yaml`](cfg/torch/strategies/fsdp_tp.yaml), requires `torch>=2.4`; docs/adr/0022) are selectable through `--strategy`.
4. **Distributed data loading** — The example [`TimmDataLoaderWrapper`](examples/torch/data.py) automatically creates a [`DistributedSampler`](https://docs.pytorch.org/docs/stable/data.html#torch.utils.data.distributed.DistributedSampler) when a distributed environment is detected. Per-epoch reshuffling additionally needs the sampler's `set_epoch()`, which the wrapper issues from its own `on_epoch_begin`; the trainer scans the provider datasets for event protocols on every rank, so the hook runs everywhere it must.
5. **Metric synchronization** — `TorchTracker` uses [`all_reduce`](https://docs.pytorch.org/docs/stable/distributed.html#torch.distributed.all_reduce) to average loss and metric values across all ranks.
6. **Rank-0 logging** — Experiment logging and progress bars run only on rank 0. Checkpoint states are produced on **every** rank, because the strategy's state dict is a collective, and written only by rank 0.
7. **Gradient sync gating** — Generated learners precede every model call with a `sync_gate(model, armed)` statement. Gradients synchronize only on the last call of a model owned by the running optimizer segment, on steps that update; every other call runs without synchronization, which covers gradient accumulation.
8. **Cleanup** — `torch.distributed.destroy_process_group()` is called when training finishes.

##### Single-Node Multi-GPU

To train on all GPUs of a single machine, prefix your `scm torch train` command with `torchrun`:

```bash
# Use all available GPUs on the current machine
torchrun --nproc_per_node=gpu \
    -m structcast_model.commands.main \
    torch train \
    'model: [_obj_, {_addr_: model.Model, _file_: model.py}, _call_]' \
    -s 'image: [3, 224, 224]' \
    -d cuda \
    -L '[_obj_, {_addr_: learner.Learner, _file_: learner.py}]' \
    -c cfg/torch/others/compile_default.yaml \
    -e 5 \
    --training-dataset dataset_train.yaml \
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

##### Multi-Node Training

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

| Option           | Description                                                                              |
| ---------------- | ---------------------------------------------------------------------------------------- |
| `--dist-backend` | Distributed backend (`nccl`, `gloo`). Auto-selected if omitted. Env var: `DIST_BACKEND`. |
| `--dist-url`     | URL for distributed setup. Defaults to `env://`. Env var: `DIST_URL`.                    |
| `--ci`           | Disables `tqdm` progress bars — useful in cluster job logs.                              |

##### Dataset Configuration

Dataset YAML files do **not** need per-rank customization. A single `device: cuda` value in the dataset configuration works for all ranks — the example `TimmDataLoaderWrapper` internally resolves it to the correct `cuda:<LOCAL_RANK>` device for each process.

```bash
# The same dataset YAML works for single-GPU and distributed training
scm format cfg/torch/others/default_timm.yaml \
    -o dataset_train.yaml \
    -p 'DEFAULT: {training: true, epochs: 5, batch_size: 32, dataset: torch/cifar100, num_classes: 100, label_smoothing: 0.1, input_size: [3, 224, 224], image_dtype: bfloat16, download: true}'
```

> **Tip:** The `batch_size` in the dataset template is the **per-GPU** batch size. With 4 GPUs and `batch_size: 32`, the effective global batch size is 128.

##### Distributed Training Notes

- **Seed reproducibility** — Each rank's random seed is offset by `global_rank` to ensure different data augmentation across processes while remaining reproducible.
- **Learning rate scaling** — When scaling to multiple GPUs, consider adjusting the learning rate. A common practice is [linear scaling](https://arxiv.org/abs/1706.02677): multiply the base learning rate by the number of GPUs. This must be configured in the learner template or optimizer settings — `scm torch train` does not scale the learning rate automatically.
- **SyncBatchNorm** — under DDP and FSDP2, `scm torch train` converts `BatchNorm` layers to [`SyncBatchNorm`](https://docs.pytorch.org/docs/stable/generated/torch.nn.SyncBatchNorm.html) automatically before the models are wrapped, on non-CPU devices. Turn it off with `_bind_: {sync_batchnorm: false}` on the strategy pattern; see the [SyncBatchNorm note](#distributed-training-with-torchrun) for the limitations, including the `torch.compile` graph break and the compilation a replaced layer loses when it is the model root or a `shard_modules` match.
- **`torch.compile` and the strategy** — with `--compile`, the strategy decides where its compile units sit: the model root in place by default, the matched `shard_modules` blocks under per-block FSDP2 — always **before** wrapping, so the strategy wrapper stays outermost. The learner's generated `_flow_*` functions compile on a single device only (distributed wrappers graph-break inside them); the eager step methods are never compiled.
- **Checkpoint saving** — State dicts are produced through [`torch.distributed.checkpoint.state_dict`](https://docs.pytorch.org/docs/stable/distributed.checkpoint.html), so the keys are wrapper-free for raw, compiled, DDP, and FSDP2 models alike. Producing them is a collective that runs on every rank; only rank 0 writes them to the experiment tracking service. `--resume` loads the same training state on all ranks.

### 7. Train a Flax Model

`scm flax train` is the Flax (JAX) counterpart of `scm torch train`. It reuses the same trainer, callbacks, and loggers, and differs where JAX differs: one process drives every device of the host, so there is no launcher and no rank — `--strategy` names the device mesh instead.

```bash
# 1. Generate the model and the learner classes
scm flax create model cfg/flax/models/ConvNeXtV2.yaml -p 'DEFAULT: {backbone: femto}' -o model.py
scm flax create learner cfg/flax/learners/ConvNeXtV2.yaml -o learner.py

# 2. Train
scm flax train \
    'model: [_obj_, {_addr_: model.Model, _file_: model.py}]' \
    -s 'image: [224, 224, 3]' \
    -L '[_obj_, {_addr_: learner.Learner, _file_: learner.py}]' \
    -e 5 \
    --training-dataset '[_obj_, {_addr_: batches, _file_: my_data.py}, {_call_: {split: train}}]' \
    -V '[_obj_, {_addr_: batches, _file_: my_data.py}, {_call_: {split: validation}}]' \
    -f 1 \
    -LC ce_loss \
    -LC val_ce_loss \
    -HC acc1 \
    -HC val_acc1 \
    -SC val_acc1 \
    --strategy cfg/flax/strategies/dp.yaml \
    --logger mlflow \
    -E Test
```

The repository ships no Flax dataset template — `my_data.py` above stands for your own code. Any iterable of `{input_name: array}` batches works, as long as the names match the learner's `INPUTS` (`image` and `label` for the template above).

Where it differs from `scm torch train`:

- positional model patterns resolve to the model **class, not an instance**: the command calls each one with the run's `nnx.Rngs` as `rngs=...`, built from `--seed`, so the pattern carries no `_call_` entry
- `-s/--shape`: channel-last, and nothing is allocated from it — a Flax module builds its parameters in its constructor, so the shapes only identify the run's configuration and default to the models' `INPUT_SHAPES`
- `-c/--compile`: wraps the learner's generated step functions in [`nnx.jit`](https://flax.readthedocs.io/en/stable/api_reference/flax.nnx/transforms.html) and is **on by default**, unlike `scm torch train`'s `--compile`, which is off unless given. Pass `--compile none` to run them eagerly, or a dict of extra `nnx.jit` keyword arguments; what is static and what is donated is the generated step's contract and cannot be overridden
- `--strategy`: the preset name `single`, `dp`, `fsdp`, `tp`, or `fsdp_tp`, or an object pattern — the templates under [`cfg/flax/strategies/`](cfg/flax/strategies/) bind the remaining knobs. Every batch is placed across the mesh before it reaches the learner, so each entry needs a leading dimension the mesh size divides
- `-d/--device`: names the device of the `single` preset only (`cpu:0`, `gpu:0`, …); the multi-device presets span the devices themselves
- `--matmul-precision`: sets `jax_default_matmul_precision` and defaults to `high`
- there is no `--dist-backend`, `--dist-url`, or `-I/--initializer`, and no gradient-scaler options: `FlaxDistributedStrategy` refuses to build a scaler
- training states are saved as `training_state.tar.gz` — an [orbax](https://orbax.readthedocs.io/) checkpoint packed into one archive — instead of the torch `.pt` file, and `--resume` reads that format back

What the command does internally:

1. Builds the strategy **first**: constructing it activates its device mesh process-wide, so every array allocated afterwards lands on it.
2. Builds the run's `nnx.Rngs` from `--seed`, calls each model factory with it, then hands the models to `strategy.wrap(...)`, which places every parameter on the sharding its rule asks for. The learner is built from the placed models, so its optimizers inherit those shardings and its inference views share their arrays.
3. Compiles the learner's `flow_functions` unless `--compile none`: `_training_step` takes the contract arguments (the models and the optimizers donated — gradient accumulation lives inside the optimizer state, so it travels with them and needs no static gate), every other flow takes only the extra arguments given.
4. Instantiates the datasets, wraps each one so every batch is placed across the mesh on the way out, and composes them into a `SimpleDataProvider`.
5. Builds the logger with a `FlaxStateBackend`, and restores `--resume` through it before the loop starts, continuing at the saved epoch plus one.
6. Creates the `FlaxTrainer` with a `FlaxTracker` over the criterion names, then appends a `ProgressBar` (a `Printer` under `--ci`), the logger, a `FlaxTrainingStateSaver`, and one `FlaxBestCriterion` per monitored criterion — and prints the resulting routing.
7. Runs `fit()` inside the logger's run context, recording the metrics, the arguments, the per-epoch training state, and the best checkpoints.

### 8. Train a Keras Model

`scm keras train` is the Keras counterpart of `scm torch train`. It reuses the same trainer, callbacks, and loggers, and differs where Keras differs: the run names the backend it executes on, and what a strategy can do follows from that choice.

```bash
# 1. Generate the model and the learner classes
scm keras create model cfg/keras/models/ConvNeXtV2.yaml -p 'DEFAULT: {backbone: femto}' -o model.py
scm keras create learner learner.yaml -o learner.py

# 2. Train
scm keras train \
    'model: [_obj_, {_addr_: Model, _file_: model.py}, _call_]' \
    --backend jax \
    -s 'image: [224, 224, 3]' \
    -L '[_obj_, {_addr_: Learner, _file_: learner.py}]' \
    -e 5 \
    --training-dataset '[_obj_, {_addr_: batches, _file_: my_data.py}, {_call_: {split: train}}]' \
    -V '[_obj_, {_addr_: batches, _file_: my_data.py}, {_call_: {split: validation}}]' \
    -f 1 \
    -LC ce_loss \
    -LC val_ce_loss \
    -HC acc1 \
    -HC val_acc1 \
    -SC val_acc1 \
    --strategy cfg/keras/strategies/dp.yaml \
    --logger mlflow \
    -E Test
```

As with Flax, the repository ships no Keras dataset template — `my_data.py` stands for your own code, and any iterable of `{input_name: array}` batches works.

Where it differs from `scm torch train`:

- `--backend`: required, `tensorflow`, `jax`, or `torch`. Keras resolves its backend once, while it is first imported, so the command sets `KERAS_BACKEND` before importing anything that would; if Keras is already running on another backend it refuses rather than pretending to switch
- `-s/--shape`: channel-last, and used to trace each model into existence; when omitted, every model is traced with the `INPUT_SHAPES` it declares itself
- `--strategy`: the preset name `single`, `dp`, `fsdp`, or `tp`, or an object pattern — the templates under [`cfg/keras/strategies/`](cfg/keras/strategies/) bind the remaining knobs. `dp` runs on each backend's own data parallelism (`keras.distribution` on JAX, `tf.distribute.MirroredStrategy` on TensorFlow, `DistributedDataParallel` under `torchrun` on torch), while `fsdp` is refused anywhere but JAX instead of silently replicating
- `-d/--device`: named as `keras.distribution.list_devices()` spells it (`cpu:0`, `gpu:0`, …), it places nothing — which devices a backend computes on is the backend's own choice (restrict it with `CUDA_VISIBLE_DEVICES`) — so the name is validated and recorded with the run
- training states are saved as `training_state.npz`, tagged with the backend that wrote them: `--resume` continues at the saved epoch plus one and refuses a state written on another backend, since normalization statistics and RNG trajectories are not verified equivalent across backends

> **TensorFlow backend on GPU** — as with the JAX backend, you may need to set `LD_LIBRARY_PATH` to include the NVIDIA shared libraries from your virtual environment (`.venv/lib/python3.*/site-packages/nvidia/*/lib`). Without it TensorFlow logs `Cannot dlopen some GPU libraries`, falls back to the CPU and the run still reports success — at CPU speed. The tell is the device list: `keras.distribution.list_devices()` names `cpu:0` alone, and `-d gpu:0` aborts with that list instead of training.

## Training Loop Anatomy

Whether it is built by the CLI or by hand, a training run is the same five objects handed to a trainer at construction:

| Object           | Responsibility                                                                                     | Ready-made pieces                                            |
| ---------------- | -------------------------------------------------------------------------------------------------- | ------------------------------------------------------------ |
| **Learner**      | Owns the models; decides when to update and how a training and an inference step run                | `scm torch create learner`                                    |
| **Tracker**      | Turns the criteria of each step into the values recorded for the epoch                             | `TorchTracker` (averages, and reduces across ranks)           |
| **DataProvider** | Supplies the datasets and their step counts (`steps_per_epoch`, `validation_steps`) for the run    | `SimpleDataProvider`                                          |
| **Callbacks**    | React to lifecycle events                                                                          | `ProgressBar`, `Printer`, `BestCriterion`                     |
| **Logger**       | Owns the run on an experiment tracking service and logs the epoch metrics                          | `MLflowLogger`, `WandbLogger`                                 |

```python
trainer = TorchTrainer(
    device="cpu",
    learner=learner,
    tracker=tracker,
    data=SimpleDataProvider(training_dataset=training_dataset, validation_dataset=validation_dataset),
    callbacks=[Printer(), BestCriterion(target="val_loss", mode="min")],
)
trainer.fit(epochs=3)
```

There is no registration call and no global registry. Every participant — the learner, the learner's `optimizers`, the tracker, the data provider and its datasets, then the `callbacks` in the order given — is scanned once on first use — the first dispatched event; `describe()` only previews the routing — and is routed into each lifecycle event whose protocol it implements:

`on_update`, `on_training_begin`, `on_training_end`, `on_training_step_begin`, `on_training_step_end`, `on_validation_begin`, `on_validation_end`, `on_validation_step_begin`, `on_validation_step_end`, `on_epoch_begin`, `on_epoch_end`.

An object joins an event simply by defining the matching method; `trainer.describe()` shows the resulting routing. This is how an optimizer + scheduler composition steps its schedule, how `TorchTracker` resets its averages between training and validation, and how a logger records epoch metrics — all through the same mechanism.

Datasets arrive at construction through the data provider, so `fit(epochs, start_epoch, validation_frequency)` takes loop parameters only. `train(dataset)` and `evaluate(dataset)` remain available for a single pass over a dataset.

For a complete, commented program built from these pieces, see [`examples/`](examples/).

## Configuration Examples

The `cfg/` directory contains working YAML templates that demonstrate each part of the workflow. Templates are organized by framework under `cfg/torch/`, `cfg/flax/`, and `cfg/keras/`. For schema details on every key used below, see [REFERENCE.md](REFERENCE.md).

### PyTorch

**[`cfg/torch/models/ConvNeXtV2.yaml`](cfg/torch/models/ConvNeXtV2.yaml)** — Demonstrates the model-building style used throughout the project. The root model defines the top-level execution flow, and sublayer keys (`Backbone`, `Block`, etc.) define reusable nested modules:

```yaml
# Root model: routes tensors through backbone → pooling → classifier
INPUTS: [image]
OUTPUTS: [cls]
FLOW:
  - [image, {feature: feat4}, backbone, {TYPE: Backbone}]
  - [feature, _, [_obj_, {_addr_: torch.nn.AdaptiveAvgPool2d}, {_call_: {output_size: 1}}]]
  - [_, _, [_obj_, {_addr_: torch.nn.Flatten}, _call_]]
  - # ... LayerNorm (Jinja-expanded from backbone dims) ...
  - [_, cls, head, [_obj_, {_addr_: torch.nn.LazyLinear}, {_call_: {out_features: 1000}}]]
```

Parameter groups define multiple backbone sizes, and Jinja rendering expands blocks based on `depths` and `dims`:

```yaml
PARAMETERS:
  DEFAULT:
    backbone: atto
  SHARED:
    stem_kernel_size: 4
    kernel_size: 7
    drop_path_rate: 0.0
    num_classes: 1000
  atto:
    dims: [40, 80, 160, 320]
    depths: [2, 2, 6, 2]
  femto:
    dims: [48, 96, 192, 384]
    depths: [2, 2, 6, 2]
  # ... tiny, small, base, large, huge ...
```

The `Block` sublayer shows how a single convolutional block is defined with depthwise convolution, normalization, MLP expansion, GRN, and residual addition:

```yaml
Block:
  OUTPUTS: [out]
  _jinja_yaml_: |-
    FLOW:
      - INPUTS: inp
        OUTPUTS: _
        LAYER:
          - _obj_
          - _addr_: torch.nn.LazyConv2d
          - _call_: {out_channels: {{fout}}, kernel_size: {{kernel_size}}, groups: {{fout}}, padding: "eval: {{kernel_size}} // 2"}
      - [_, _, [_obj_, {_addr_: structcast_model.torch.layers.ToChannelLast}, _call_]]
      - [_, _, [_obj_, {_addr_: timm.layers.LayerNorm}, {_call_: {num_channels: {{fout}}, eps: {{norm_eps}}}}]]
      - [_, _, [_obj_, {_addr_: torch.nn.LazyLinear}, {_call_: {out_features: "eval: {{fout}} * {{mlp_ratio}}"}}]]
      - [_, _, [_obj_, {_addr_: "timm.layers.{{activation}}"}, {_call_: {inplace: true}}]]
      - [_, _, [_obj_, {_addr_: timm.layers.grn.GlobalResponseNorm}, {_call_: {dim: "eval: {{fout}} * {{mlp_ratio}}"}}]]
      - [_, _, [_obj_, {_addr_: torch.nn.LazyLinear}, {_call_: {out_features: {{fout}}}}]]
      - [_, _, [_obj_, {_addr_: structcast_model.torch.layers.ToChannelFirst}, _call_]]
      - [_, feat, {TYPE: DropPath, PARAM: {DEFAULT: {drop_prob: {{drop_path}}}}}]
      - ["eval: inp + feat", out, null]
```

**[`cfg/torch/learners/ConvNeXtV2.yaml`](cfg/torch/learners/ConvNeXtV2.yaml)** — Demonstrates a single-optimizer learner with mixed precision, gradient accumulation, cosine LR scheduling, and inline loss/metric definitions in the flow. The optimizer is a file-addressed composition from [`examples/torch/optimizers.py`](examples/torch/optimizers.py): the package builds optimizers (`create_opt`), while optimizer + scheduler combinations are example code you can copy and adapt:

```yaml
MIXED_PRECISION:
  init_scale: "eval: 2.0**16"
  growth_factor: 2.0
  backoff_factor: 0.5
  growth_interval: 2000
  enabled: True
MIXED_PRECISION_TYPE: bfloat16
OUTPUTS: [ce_loss, acc1, acc5]
LEARNERS:
  - LOSS: ce_loss
    TRAINABLE_LAYERS: [model]
    NAME: optimizer
    OPTIMIZER:
      - _obj_
      - _addr_: AdamWWithCosine
        _file_: examples/torch/optimizers.py
      - _bind_:
          optimizer_kwargs: {opt: adamw, lr: 4.0e-3, weight_decay: 0.001}
          scheduler_kwargs: {sched: cosine, num_epochs: 300, criterion: ce_loss}
    FLOW:
      - [image, cls, model]
      - [{target: label, input: cls}, ce_loss, cross_entropy_loss, [_obj_, _addr_: torch.nn.CrossEntropyLoss, _call_]]
      - [{y_true: label, y_pred: cls}, acc1, accuracy, [_obj_, _addr_: torch.no_grad, _call_, _call_: [[_obj_, {_addr_: structcast_model.torch.layers.sparse_categorical_accuracy}]]]]
      - [{y_true: label, y_pred: cls, k: 5}, acc5, top_5_accuracy, [_obj_, _addr_: torch.no_grad, _call_, _call_: [[_obj_, {_addr_: structcast_model.torch.layers.sparse_top_k_categorical_accuracy}]]]]
```

**[`cfg/torch/learners/CycleGAN.yaml`](cfg/torch/learners/CycleGAN.yaml)** — Demonstrates a multi-optimizer learner for GAN-style training with three `LEARNERS` entries (generator pair + two discriminators), each with its own flow, optimizer, and trainable layers.

**[`cfg/torch/learners/ImageClassifierShowcase.yaml`](cfg/torch/learners/ImageClassifierShowcase.yaml)** — Turns gradient checkpointing, gradient accumulation, mixed precision and the EMA on at once over the VisionTransformer template (see REFERENCE.md, *Putting it together*).

**`cfg/torch/models/CycleGAN_generator.yaml` and `CycleGAN_discriminator.yaml`** — Pair of model templates for the CycleGAN architecture:

- **Generator** — uses `ResidualBlock`, `DownBlock`, and `UpBlock` sublayers with reflection padding, instance normalization, and Jinja-driven residual block expansion (`n_residual_blocks` parameter)
- **Discriminator** — uses a `DiscriminatorBlock` sublayer with conditional instance normalization controlled by a `normalize` parameter
- both templates use `LazyConv2d` for automatic input channel inference

**[`cfg/torch/others/default_timm.yaml`](cfg/torch/others/default_timm.yaml)** — Formats directly into a `TimmDataLoaderWrapper.model_validate(...)` pattern, loading the wrapper from the example file [`examples/torch/data.py`](examples/torch/data.py) by path. The template covers timm dataset and dataloader construction, device and prefetch settings, mixup/cutmix configuration, and train/validation split generation — all from a single parameterized template:

```yaml
_obj_:
  - _addr_: TimmDataLoaderWrapper
    _file_: examples/torch/data.py
  - _attr_: model_validate
  - - _call_
    - spec: {image: "0", label: "1"}
      dataset:
        input_img_mode: RGB
        _jinja_yaml_: |-
          batch_size: {{batch_size}}
          name: {{dataset}}
          root: {{dataset_dir}}
          is_training: {{training}}
          split: {{"train" if training else "validation"}}
          # ...
      use_prefetcher: true
      mixup_alpha: 0.0
      cutmix_alpha: 0.0
      # ...
```

### Flax

**[`cfg/flax/models/ConvNeXtV2.yaml`](cfg/flax/models/ConvNeXtV2.yaml)** — Generates a [Flax `nnx.Module`](https://flax.readthedocs.io/en/stable/api_reference/flax.nnx/module.html) equivalent of the PyTorch ConvNeXtV2 model. The template mirrors the same parameter groups (`atto` through `huge`) and uses [`GlobalResponseNorm`](src/structcast_model/flax/layers/grn.py) as a custom Flax layer. Key differences from the PyTorch variant:

- uses channel-last tensor layout (*H × W × C*)
- constructor accepts a `rngs: flax.nnx.Rngs` argument for parameter initialization
- `__call__` propagates a `training` flag to sub-modules
- layer APIs differ (e.g., `flax.nnx.Conv` instead of `torch.nn.LazyConv2d`)

**[`cfg/flax/learners/ConvNeXtV2.yaml`](cfg/flax/learners/ConvNeXtV2.yaml)** — The learner for that model. Its single `LEARNERS` entry builds an `nnx.Optimizer` over an `optax.chain` of `clip_by_global_norm` and `adamw`, masked by [`no_weight_decay_mask`](src/structcast_model/flax/optimizers.py) so biases and normalization scales are exempt from weight decay. There is no `CLIP` and no `MIXED_PRECISION` key — clipping is a stage of the chain — and the criteria are `"eval: ..."` expressions over the model output.

**[`cfg/flax/models/`](cfg/flax/models/)** also ships [`VisionTransformer.yaml`](cfg/flax/models/VisionTransformer.yaml), [`SmallLanguageModel.yaml`](cfg/flax/models/SmallLanguageModel.yaml) and the [`CycleGAN_generator.yaml`](cfg/flax/models/CycleGAN_generator.yaml) / [`CycleGAN_discriminator.yaml`](cfg/flax/models/CycleGAN_discriminator.yaml) pair — NHWC, every convolution declaring `in_features` (`flax.nnx.Conv` has no lazy form), and the layers with no nnx twin folded into their neighbours (`ReflectionPad2d` into `padding: REFLECT`, `Upsample` into a row/column repeat), each fold documented in the template header.

**[`cfg/flax/learners/`](cfg/flax/learners/)** — [`ImageClassifier.yaml`](cfg/flax/learners/ImageClassifier.yaml) trains both image models, [`SmallLanguageModel.yaml`](cfg/flax/learners/SmallLanguageModel.yaml) does next-token prediction, and [`CycleGAN.yaml`](cfg/flax/learners/CycleGAN.yaml) drives three optimizer segments. [`ImageClassifierShowcase.yaml`](cfg/flax/learners/ImageClassifierShowcase.yaml) turns checkpointing, accumulation and the EMA on at once (see REFERENCE.md, *Putting it together*). The accumulation window is an `optax.MultiSteps` that must be the **outermost** transformation — the generated step reads its applied count off the outermost `opt_state`, so a window buried inside `optax.chain` accumulates identically and still reports an update on every step. `clip_grad_norm` here means the clipping bound (`optax.clip_by_global_norm`), and optax schedules count optimizer applies, which is why the CycleGAN learner takes a `steps_per_epoch` parameter its torch twin does not.

**[`cfg/flax/others/`](cfg/flax/others/)** — [`compile_default.yaml`](cfg/flax/others/compile_default.yaml) for `--compile` (only `backend`, `keep_unused` and `inline`: the CLI already fixes the donation contract), and [`default_tfdata.yaml`](cfg/flax/others/default_tfdata.yaml), a `tf.data` pipeline pattern over [`examples/flax/data.py`](examples/flax/data.py) whose `split` follows `training` unless overridden. The dataset name is required — the template refuses to render without one — and loading it needs the `tensorflow-datasets` package, which structcast-model does not depend on.

**[`cfg/flax/strategies/`](cfg/flax/strategies/)** — Object patterns for `--strategy`, binding a `FlaxDistributedStrategy` preset: [`dp.yaml`](cfg/flax/strategies/dp.yaml) replicates the parameters and splits each batch across the devices, [`fsdp.yaml`](cfg/flax/strategies/fsdp.yaml) additionally shards parameters along their leading dimension, leaving the ones below `min_size` bytes replicated, and [`tp.yaml`](cfg/flax/strategies/tp.yaml) / [`fsdp_tp.yaml`](cfg/flax/strategies/fsdp_tp.yaml) add a model axis whose `column`/`row` rules split the matmuls themselves (docs/adr/0022).

### Keras

**[`cfg/keras/models/ConvNeXtV2.yaml`](cfg/keras/models/ConvNeXtV2.yaml)** — Generates a [Keras `Layer`](https://keras.io/api/layers/base_layer/) equivalent of the ConvNeXtV2 model. Shares the same backbone parameter groups and uses [`GlobalResponseNormalization`](src/structcast_model/keras/layers/grn.py) as a custom Keras layer. Key differences:

- uses channel-last tensor layout (*H × W × C*)
- follows the Keras `call(self, ..., *, training=None, **kwargs)` convention
- runs on any [Keras backend](https://keras.io/getting_started/#configuring-your-backend) (JAX, PyTorch, or TensorFlow)
- uses `keras.layers.Add` for residual connections instead of `"eval: inp + feat"` expressions

**[`cfg/keras/models/`](cfg/keras/models/)** also ships [`VisionTransformer.yaml`](cfg/keras/models/VisionTransformer.yaml), [`SmallLanguageModel.yaml`](cfg/keras/models/SmallLanguageModel.yaml) and the [`CycleGAN_generator.yaml`](cfg/keras/models/CycleGAN_generator.yaml) / [`CycleGAN_discriminator.yaml`](cfg/keras/models/CycleGAN_discriminator.yaml) pair, the Keras twins of the PyTorch templates. The two transformers use `keras.layers.MultiHeadAttention`, which owns its query/key/value projections, so neither writes an attention section of its own — and the language model therefore carries a learned position table instead of the rotary embedding of its torch twin, which needs those projections spelled out.

**[`cfg/keras/learners/`](cfg/keras/learners/)** — [`ConvNeXtV2.yaml`](cfg/keras/learners/ConvNeXtV2.yaml) and [`ImageClassifier.yaml`](cfg/keras/learners/ImageClassifier.yaml) train the two image models, [`SmallLanguageModel.yaml`](cfg/keras/learners/SmallLanguageModel.yaml) does next-token prediction, and [`CycleGAN.yaml`](cfg/keras/learners/CycleGAN.yaml) drives three optimizer segments over four models. [`ImageClassifierShowcase.yaml`](cfg/keras/learners/ImageClassifierShowcase.yaml) turns checkpointing, accumulation, mixed precision and the optimizer's EMA on at once (see REFERENCE.md, *Putting it together*). All four put the schedule, the clipping (`global_clipnorm`) and the gradient accumulation (`gradient_accumulation_steps`) inside the `OPTIMIZER` pattern, because on this backend the Keras optimizer owns all three — which is why the Keras learner schema has no `CLIP` field and no `ACCUMULATE_GRADIENTS` field, and rejects both by name. Weight-decay exemptions are the one optimizer knob no object pattern can express (Keras configures them through a method call), so the templates route them through [`create_optimizer`](examples/keras/optimizers.py) with `_file_`.

The schema is the shared one, written against Keras objects. A minimal single-segment learner over a model taking `x` and returning `y`, against a label `target`, is:

```yaml
INPUTS: [x, target]
OUTPUTS: [loss]
LEARNERS:
  - NAME: optimizer
    LOSS: loss
    TRAINABLE_LAYERS: [model]
    OPTIMIZER: [_obj_, {_addr_: keras.optimizers.SGD}, {_call_: {learning_rate: 0.1}}]
    FLOW:
      - INPUTS: x
        OUTPUTS: {prediction: y}
        NAME: model
      - INPUTS: {y_true: target, y_pred: prediction}
        OUTPUTS: errors
        NAME: mse
        LAYER: [_obj_, {_addr_: keras.losses.mean_squared_error}]
      - ["eval: keras.ops.mean(errors)", loss, null]
    INFERENCE_FLOW:
      - INPUTS: x
        OUTPUTS: {prediction: y}
        NAME: model
      - [{y_true: target, y_pred: prediction}, errors, mse]
      - ["eval: keras.ops.mean(errors)", loss, null]
```

The model step is written in the mapping form, as in the Flax template above, because `scm keras create model` returns the outputs as a dict by default: `OUTPUTS: {prediction: y}` binds the model's `y` output to `prediction`. A model generated with `--no-structured-output` returns the value positionally and takes the short form `- [x, prediction, model]` instead.

The `OPTIMIZER` pattern builds a [`keras.optimizers.Optimizer`](https://keras.io/api/optimizers/) — gradient clipping is one of its keyword arguments, which is why there is no `CLIP` key — and the criteria are `"eval: ..."` expressions over the model output, written with [`keras.ops`](https://keras.io/api/ops/) so the same template runs on every backend.

**[`cfg/keras/strategies/`](cfg/keras/strategies/)** — Object patterns for `--strategy`, binding a `KerasDistributedStrategy` preset: [`dp.yaml`](cfg/keras/strategies/dp.yaml) replicates the variables and splits each batch across the replicas, [`fsdp.yaml`](cfg/keras/strategies/fsdp.yaml) additionally shards the variables along their leading dimension, leaving the ones the device count cannot divide replicated — and is available on the JAX backend alone, as is [`tp.yaml`](cfg/keras/strategies/tp.yaml), whose `column`/`row` rules split the matmuls across a model axis.

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

## Migration Notes

### Upgrading to v2.x

The training loop was redesigned around protocol-routed callbacks. The rationale is recorded in [`docs/adr/0002-protocol-routed-training-loop.md`](docs/adr/0002-protocol-routed-training-loop.md); the vocabulary in [`CONTEXT.md`](CONTEXT.md). There are no compatibility aliases:

- **`Backward` is now `Learner`** — The rename cascades through the runtime, the CLI (`scm torch create learner`, `--learner/-L`), the builder and schema names (`LEARNERS`, `LearnerBehavior`, `UserDefinedLearner`), and the template directory (`cfg/torch/learners/`).
- **Callbacks are routed by protocol** — The `GLOBAL_CALLBACKS` registry, the `callbacks_session` context manager, and `NamedCallbackList.register()` are gone. Pass participants to the trainer as `callbacks=[...]`; each one joins the events whose `on_*` method it defines. Ad-hoc lambdas become small callback classes — `ProgressBar` and `Printer` ship with the package.
- **Datasets are given at construction** — `fit()` no longer takes datasets. Build a `DataProvider` (`SimpleDataProvider`, or your own object with `training_dataset`, `validation_dataset`, `steps_per_epoch`, and `validation_steps` — the dataset properties must return the same object on every read, since the trainer reads them for the event scan and again in `fit()`) and pass it as `data=`. The trainer also scans the provider datasets for event protocols, so a dataset with an `on_*` hook (e.g. a distributed sampler wrapper) takes part in the loop without being passed as a callback. `fit()` keeps `epochs`, `start_epoch`, and `validation_frequency`; `train(dataset)` and `evaluate(dataset)` are unchanged.
- **`create_with_scheduler` is removed** — The package keeps `create_opt` (regex weight-decay and layer-decay grouping over `torch.optim` and timm engines). Optimizer + scheduler combinations move to example code referenced by file path; `AdamWWithCosine` (timm schedules) and `OptimizerWithNativeScheduler` (per-epoch native schedules) in [`examples/torch/optimizers.py`](examples/torch/optimizers.py) cover the cosine and per-epoch native cases and also keep the schedule in their `state_dict`; metric-driven (`ReduceLROnPlateau`), per-update, and composite schedules need a wrapper of their own modeled on these.
- **Loggers own the run** — `MLflowLogger` (`structcast_model.loggers.mlflow`) and `WandbLogger` (`structcast_model.loggers.wandb`) are context managers that start and end the run and log epoch metrics; both follow the `Logger` protocol in `structcast_model.loggers.base`. Select the backend with `--logger mlflow|wandb`.
- **Trackers reset themselves** — `TorchTracker` clears its averages from `on_training_begin` and `on_validation_begin`; the explicit `reset()` call in the loop is gone.

### Upgrading from v1.x

The following breaking changes were introduced by the learner-template restructure for multi-optimizer GAN training support:

- **EMA support removed** — `TimmEmaWrapper`, the `cfg/torch/others/ema.yaml` configuration, and all `InferenceWrapper`-based EMA integration in `cmd_torch.py` and `torch/trainer.py` have been removed. If your training workflow relied on built-in EMA, you will need to manage EMA externally.
- **Learner template schema restructured** — The `LEARNERS` key expects a list of `LearnerBehavior` entries (each with its own `NAME`, `LOSS`, `TRAINABLE_LAYERS`, `OPTIMIZER`, `FLOW`, and optional `INFERENCE_FLOW`). Previous single-optimizer configurations must be wrapped in a single-entry list.
- **Separate loss and metric templates removed** — Losses and metrics are declared inline in the learner's flow, so `scm torch train` no longer takes `--loss` or `--metric`.

## Roadmap

- [x] PyTorch model construction from YAML configuration files
- [x] PyTorch training workflow generation from YAML configuration files
- [x] JAX (Flax) model construction from YAML configuration files
- [x] JAX (Flax) training workflow generation from YAML configuration files
- [x] Keras model construction from YAML configuration files
- [x] Keras training workflow generation from YAML configuration files
