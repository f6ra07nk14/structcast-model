# StructCast-Model — AI Agent Reference

> This document is written for AI coding agents. For human-oriented usage, see [README.md](README.md).
> For the upstream library, see the [StructCast repository](https://github.com/f6ra07nk14/structcast).

## What This Project Does

StructCast-Model turns YAML templates into executable models and training systems across multiple frameworks. It has four responsibilities:

1. **Code generation**: Generate model classes from declarative YAML templates — PyTorch `nn.Module`, Flax `nnx.Module`, and Keras `Layer`. PyTorch also supports learner class generation (the object owning the models, the optimizers, and the training and inference steps), including multi-optimizer setups (e.g., GAN training with separate generator and discriminator optimizers).
2. **Template rendering**: Format parameterized YAML templates into concrete runtime configurations.
3. **Inference benchmarking**: Measure model inference time via `scm [torch/flax/keras] time`.
4. **Training execution**: Instantiate generated artifacts through [StructCast](https://github.com/f6ra07nk14/structcast) object patterns and run them via `scm torch train` (PyTorch only; Flax and Keras training is planned).

## Repository Map

```text
cfg/torch/
├── learners/                  # Learner templates (losses, metrics, optimizers, flows)
├── models/                    # Model architecture templates
└── others/                    # Other templates (timm datasets, compile settings)
cfg/flax/
└── models/                    # Flax model architecture templates
cfg/keras/
└── models/                    # Keras model architecture templates

examples/torch/
├── simple_training.py         # Runnable programmatic training tutorial
└── optimizers.py              # Optimizer + scheduler compositions referenced by _file_

src/structcast_model/
├── base_trainer.py            # Generic trainer, event protocols, best-criterion handling
├── builders/
│   ├── base_builder.py        # Generic template -> intermediate -> script pipeline
│   ├── schema.py              # Pydantic schemas for layer/learner templates
│   ├── torch_builder.py       # PyTorch-specific code generation
│   ├── flax_builder.py        # Flax-specific code generation
│   └── keras_builder.py       # Keras-specific code generation
├── commands/
│   ├── main.py                # Top-level scm CLI
│   ├── cmd_torch.py           # PyTorch CLI commands
│   ├── cmd_flax.py            # Flax CLI commands
│   ├── cmd_keras.py           # Keras CLI commands
│   └── utils.py               # CLI argument parsers and reducers
├── torch/
│   ├── trainer.py             # Trainer, tracker, learner factory, loggers, timm wrappers
│   ├── optimizers.py          # create_opt: regex parameter grouping over optimizer engines
│   ├── layers/                # Reusable torch layers referenced by templates
│   └── types.py               # Tensor aliases and related typing
├── flax/
│   ├── trainer.py             # Flax inference time measurement
│   └── layers/                # Reusable Flax layers (e.g. GlobalResponseNorm)
├── keras/
│   ├── trainer.py             # Keras inference time measurement
│   └── layers/                # Reusable Keras layers (e.g. GlobalResponseNormalization)
└── utils/                     # YAML and helper utilities used by builders/commands

tests/
├── builders/                  # Builder and schema tests
├── commands/                  # CLI tests
├── torch/                     # Trainer and torch layer tests
├── flax/                      # Flax builder and layer tests
├── keras/                     # Keras builder and layer tests
├── fixtures/                  # YAML and data fixtures
└── test_base_trainer.py       # Generic trainer tests
```

## Data Flow

The following diagram shows how data moves through the system. Use this to understand which module to inspect when debugging or modifying a specific stage.

```text
YAML template in cfg/[torch/flax/keras]/
  |  TemplateLayer / TemplateLearner validation      <- builders/schema.py
  v
Builder intermediate objects
  |  BaseModelBuilder / BaseLearnerBuilder           <- builders/base_builder.py
  |  TorchBuilder / FlaxBuilder / KerasBuilder       <- builders/{torch,flax,keras}_builder.py
  |  TorchLearnerBuilder                             <- builders/torch_builder.py
  v
Generated Python source files
  |  scm [torch/flax/keras] create model
  |  scm torch create learner (PyTorch only)
  v
StructCast object patterns
  |  _obj_ + _addr_ + _file_ + _call_               <- commands/cmd_{torch,flax,keras}.py
  v
Live model objects
  |  Inference benchmarking                          <- scm [torch/flax/keras] time
  |  Training (PyTorch only)                         <- scm torch train
  v
TorchTrainer.fit(...)  (PyTorch training path)
  |  train/evaluate loop + routed callbacks          <- base_trainer.py + torch/trainer.py
  v
MLflow / wandb logs + model states + best checkpoints
```

The repository's signature workflow (the "generate-then-reimport" loop) is:

1. Render specialized YAML from templates with `scm format`.
2. Generate model Python modules with `scm [torch/flax/keras] create model` (plus `create learner` for PyTorch).
3. Re-import those modules through StructCast `_file_` patterns at runtime.
4. Benchmark with `scm [torch/flax/keras] time`, or train through `scm torch train` (PyTorch only).

## CLI Surface

The CLI entry point is defined in `pyproject.toml` as `scm = "structcast_model.commands.main:app"` (a [Typer](https://typer.tiangolo.com/) application).

### Top-level commands

- `scm format`
- `scm torch create model`
- `scm torch create learner`
- `scm torch ptflops`
- `scm torch calflops`
- `scm torch time`
- `scm torch train` (also supports distributed training via `torchrun`)
- `scm flax create model`
- `scm flax time`
- `scm keras create model`
- `scm keras time`

### `scm format`

Defined in `commands/main.py`, not `cmd_torch.py`.

Purpose:

- Render a template file with `-p/--parameter` overrides.
- Print YAML to stdout or write it with `-o/--output`.

Key implementation note:

- Repeated `-p` options are merged through `reduce_dict()` in `commands/utils.py`.

### `scm torch create model`

Purpose:

- Load a YAML layer template.
- Build a `TorchLayerIntermediate`.
- Optionally write generated Python to disk.

Key options:

- `-p/--parameter`
- `-c/--classname`
- `--structured-output/--no-structured-output`
- `-s/--sublayer`
- `-o/--output`

### `scm torch create learner`

Purpose:

- Load a learner template from `cfg/torch/learners/`.
- Build a `TorchLearnerIntermediate`.
- Optionally write generated Python to disk.

Key options: `-p/--parameter`, `-c/--classname` (default `Learner`), `-o/--output`.

The generated learner class supports:

- Per-entry execution graphs (`FLOW` / `INFERENCE_FLOW`) with inline layer instantiation
- Multiple `LEARNERS` entries, each with its own optimizer and trainable layers (multi-optimizer training, e.g., GAN)
- Automatic train/eval mode switching per entry
- Gradient accumulation, AMP scaler logic, and gradient clipping

It implements the `Learner` protocol (`models`, `update`, `training_step`, `inference_step`) and exposes `optimizers`, `grad_scalers`, `learning_rates`, `param_group_names`, `inputs`, and `outputs`.

### `scm torch ptflops` and `scm torch calflops`

Purpose:

- Instantiate a model from a StructCast object pattern.
- Materialize dummy inputs from `-s/--shape`.
- Run `initial_model(...)` once.
- Compute complexity metrics.

### `scm [torch/flax/keras] time`

Purpose:

- Instantiate a model from a StructCast object pattern.
- Create dummy inputs (PyTorch tensors, JAX arrays, or NumPy arrays).
- Optionally compile the model (`torch.compile`, `nnx.jit`, or `keras.Model.compile`).
- Execute warmup runs, then time averaged inference iterations.

Key differences per framework:

- PyTorch: `--matmul-precision` option; channel-first shapes (*C × H × W*).
- Flax: `--training-mode-kwargs` option; channel-last shapes (*H × W × C*); uses `nnx.jit` for compilation.
- Keras: channel-last shapes (*H × W × C*); may require `LD_LIBRARY_PATH` for JAX+NVIDIA GPU.

### `scm flax create model` / `scm keras create model`

Defined in `commands/cmd_flax.py` and `commands/cmd_keras.py` respectively.

Purpose:

- Load a framework-specific YAML layer template from `cfg/flax/` or `cfg/keras/`.
- Build a `FlaxLayerIntermediate` or `KerasLayerIntermediate`.
- Generate Python source implementing a `flax.nnx.Module` or `keras.layers.Layer`.

Key options are the same as `scm torch create model`: `-p`, `-c`, `--structured-output/--no-structured-output`, `-s`, `-o`.

### `scm torch train`

Purpose:

- Instantiate the models and the learner (via `TorchLearnerFactory`), the datasets, and the compile settings.
- Run a training loop via `TorchTrainer`.
- Log metrics and states through the selected logger.

Key runtime behavior:

- `configure_security()` is called because commands frequently import generated local files via `_file_` patterns.
- `torch.compile` is optional and configured via `-c/--compile`; it is applied to the models and to the learner's step functions.
- Mixed precision is owned by the learner (its `MIXED_PRECISION` template keys), not by a CLI flag.
- The two dataset options are composed into a `SimpleDataProvider` passed as `data=`; `fit()` receives only loop parameters.
- Callbacks passed to the trainer: first each dataset that implements at least one event protocol (on every rank), then — on rank 0 only — `ProgressBar` (or `Printer` under `--ci`), the logger, `TrainingStateSaver`, and one `TorchBestCriterion` per `-LC`/`-HC` criterion.
- `--logger mlflow|wandb` selects the backend; the logger is entered as a context manager around `fit()`, and a `KeyboardInterrupt` saves the current training state before leaving it.
- `trainer.describe()` is printed before fitting, showing which object handles which event.

Distributed training behavior (when launched through `torchrun`):

- `initial_distributed_env()` detects `RANK`/`LOCAL_RANK`/`WORLD_SIZE` env vars and initializes the NCCL process group.
- Each process is assigned to `cuda:<LOCAL_RANK>`.
- All models are wrapped with `DistributedDataParallel`.
- The example `TimmDataLoaderWrapper` creates `DistributedSampler` automatically and calls `set_epoch()` from its own `on_epoch_begin`; the CLI adds event-protocol datasets to the callbacks of every rank, so the sampler epoch advances on all of them.
- `TorchTracker` uses `all_reduce(ReduceOp.AVG)` to synchronize metrics across ranks.
- Experiment logging, checkpoints, and progress bars are gated to rank 0 only.
- DDP gradient synchronization is skipped during gradient accumulation steps via `TorchTrainer.no_sync()`.
- CLI options `--dist-backend` and `--dist-url` (also settable via `DIST_BACKEND` / `DIST_URL` env vars) control the distributed backend.

Launch command for distributed training:

```bash
torchrun --nproc_per_node=gpu -m structcast_model.commands.main torch train ...
```

## Builder Architecture

### Generic builder layer

`builders/base_builder.py` is the generic code generation engine.

Key responsibilities:

- Resolve StructCast object patterns into Python expressions and import tables.
- Walk nested user-defined layers and cross-file template references.
- Build framework-agnostic intermediate representations.
- Render those intermediates to `.py` files.

Key APIs:

- `resolve_object(imports, pattern)`
- `resolve_getter(imports, spec, variable=None)`
- `_Intermediate`
- `LayerIntermediate`
- `BaseModelBuilder`
- `LearnerIntermediate`
- `BaseLearnerBuilder`

### PyTorch builder layer

`builders/torch_builder.py` specializes the generic intermediates into concrete PyTorch code.

Important classes:

- `TorchLayerIntermediate`: renders `torch.nn.Module` classes with `forward()`.
- `TorchBuilder`: main entry point for model generation.
- `TorchLearnerIntermediate`: renders the generated learner class.
- `TorchLearnerBuilder`: main entry point for learner generation.

Important generation details:

- Model code uses `self.<layer_name>` submodules.
- Inference flow is rendered separately when `INFERENCE_FLOW` is present.
- Learner code supports multiple `LEARNERS` entries, each with its own `FLOW`, `INFERENCE_FLOW`, `OPTIMIZER`, `TRAINABLE_LAYERS`, and `CLIP`.
- Each entry's trainable layers are set to training mode before its flow executes and set back to eval mode after the optimizer step.
- Learner code can include gradient accumulation, AMP scaler logic, clipping, optimizer stepping, and optimizer metadata properties.
- The optimizer pattern receives the named parameters of the entry's trainable layers, so it works with `create_opt` and with file-addressed optimizer compositions alike.

### Flax builder layer

`builders/flax_builder.py` specializes the generic intermediates into Flax (JAX) code.

Important classes:

- `FlaxLayerIntermediate`: renders `flax.nnx.Module` classes with `__call__()`.
- `FlaxBuilder`: main entry point for Flax model generation.

Important generation details:

- Model code uses `self.<layer_name>` submodules, same as PyTorch.
- Uses channel-last tensor layout (*H × W × C*).
- `rngs` argument is passed through the constructor for Flax RNG handling.

### Keras builder layer

`builders/keras_builder.py` specializes the generic intermediates into Keras code.

Important classes:

- `KerasLayerIntermediate`: renders `keras.layers.Layer` classes with `call()`.
- `KerasBuilder`: main entry point for Keras model generation.

Important generation details:

- Model code uses `self.<layer_name>` sub-layers.
- Uses channel-last tensor layout (*H × W × C*).
- Keras models are backend-agnostic (JAX, PyTorch, or TensorFlow).

## Runtime Architecture

### Generic trainer layer

`base_trainer.py` provides the framework-independent trainer skeleton:

- `BaseInfo`: epoch/step/update/history state
- `Learner`, `DataProvider`: the two protocols a trainer is built around
- `EVENTS` / `EVENT_PROTOCOLS`: the eleven lifecycle events and the protocol gating each one
- `BaseTrainer`: `train()`, `evaluate()`, `fit()` loop, plus `describe()` for the routing table
- `SimpleDataProvider`: dataclass holding a training dataset and an optional validation dataset
- `BestCriterion`: criterion monitor for best-value callbacks
- `ProgressBar`, `Printer`: the two built-in reporting callbacks

Routing rule: at construction the trainer scans the learner, the learner's `optimizers` values, the tracker, the data provider, and then `callbacks` in order, registering each object for every event whose `runtime_checkable` protocol it satisfies, never twice for the same event. There is no registry and no `register()` call.

### PyTorch runtime layer

`torch/trainer.py` provides the runtime objects actually used by the CLI.

Utility functions:

- `create_torch_inputs(shape, batch_size=1)`
- `get_torch_device(device=None)` / `get_torch_device_type(device=None)`
- `initial_distributed_env(device, dist_backend, dist_url)` — detects torchrun env vars, initializes process group, returns per-rank device
- `initial_model(model, shapes=None)` — returns `(inputs, outputs)`
- `autocast_inputs(inputs, device_type)`

Training/evaluation helpers:

- `TorchLearnerFactory` — builds models and learner from object patterns; excludes the tracker and DDP wrapping
- `TorchTracker` — averaging tracker that resets itself on training/validation begin
- `TorchTrainer` — adds `device`, `sync()`, and `no_sync()` to `BaseTrainer`
- `TorchBestCriterion`

Loggers (run-owning context managers that also implement `on_epoch_end`):

- `MLflowLogger`
- `WandbLogger`

timm data integrations — example code in `examples/torch/data.py`, not package API; a configuration loads them by file path (`_addr_` plus `_file_`):

- `TimmDatasetWrapper`
- `TimmDataLoaderWrapper` — implements `on_epoch_begin` (sampler reshuffling) and `on_training_begin` (mixup cutoff), so the CLI routes it into those events like any other callback
- `TimmDataProvider` — the programmatic `DataProvider` forwarding both events to the training wrapper

### Flax runtime layer

`flax/trainer.py` provides inference timing utilities for Flax models.

Utility functions:

- `create_jax_inputs(shape)` — creates JAX arrays from shape specs
- `get_jax_device(device=None)` — resolves JAX device (cpu, gpu:N)
- `measure_inference_time(...)` — benchmarks Flax model inference with optional `nnx.jit` compilation

### Keras runtime layer

`keras/trainer.py` provides inference timing utilities for Keras models.

Utility functions:

- `create_numpy_inputs(shape)` — creates NumPy arrays from shape specs
- `get_keras_device(device=None)` — resolves Keras/JAX device
- `measure_inference_time(...)` — benchmarks Keras model inference with optional compilation

### Training flow in practice

1. Datasets are instantiated from YAML or inline StructCast patterns and composed into a `SimpleDataProvider`; those implementing an event protocol are also collected as callbacks.
2. `TorchLearnerFactory` instantiates the models, initializes them with dummy inputs, applies initializers, and builds the learner with those models.
3. Models are DDP-wrapped and compiled where requested.
4. `TorchTracker` is built from the learner's `outputs` (or `--learner-outputs`).
5. Callbacks are assembled: progress reporting, logger, state saver, best criteria.
6. `TorchTrainer` is constructed — routing every participant into its events — and `fit()` runs inside the logger's run context.
7. The logger receives arguments, metrics, artifacts, training state, and best-checkpoint snapshots.

When running under `torchrun`, the flow gains additional distributed steps:

8. Process group is initialized and per-rank device is assigned.
9. Models are wrapped with `DistributedDataParallel`.
10. Metrics are synchronized across ranks via `all_reduce`.
11. Only rank 0 writes experiment logs, checkpoints, and progress output.
12. `destroy_process_group()` is called during cleanup.

## Pattern Alias Quick Reference

### Instantiator patterns used in this repository

These are [StructCast](https://github.com/f6ra07nk14/structcast) object pattern aliases. See the StructCast README for full pattern documentation.

| Alias | Meaning | Typical use here |
| --- | --- | --- |
| `_addr_` | Import by dotted address | Import timm layers, torch layers, helper functions |
| `_file_` | Load from a local Python file | Import generated `model.py` and `learner.py`, and example optimizer compositions |
| `_call_` | Call the imported symbol | Instantiate generated classes |
| `_bind_` | Partially apply arguments | Optimizer and scheduler factory configuration |
| `_attr_` | Resolve an attribute on the current object | `model_validate`, helper method access |
| `_obj_` | Chain all of the above | Main object construction mechanism |

### Template features used in config YAML files

| Alias or syntax | Meaning |
| --- | --- |
| `_jinja_yaml_` | Render Jinja and parse as YAML |
| `_jinja_group_` | Select a named parameter group |
| `eval: ...` | Inject a raw expression into generated Python |
| `DEFAULT` / `SHARED` groups | Parameter groups used by template rendering |

### Spec usage in the dataset template

`cfg/torch/others/default_timm.yaml` uses a `FlexSpec`-compatible mapping so dataloader batches can be transformed from positional `(input, target)` tuples into structured dictionaries such as `{image: ..., label: ...}`.

## Dynamic Import and Security Notes

- CLI commands call `configure_security()` to install the StructCast security settings before importing anything through `_file_` patterns.
- Generated modules are loaded with `_file_` patterns pointing at local files.
- When debugging command failures, verify that generated files exist at the exact paths referenced in `_file_`.
- Code used outside the CLI must explicitly configure StructCast security settings to permit local imports.

## Development Commands

```bash
uv sync --group dev                 # Install lint, test, and type-check tooling
pytest                              # Run tests and doctests with coverage
ruff check src tests                # Lint
ruff format src tests               # Format
mypy src && mypy tests              # Type check
tox                                 # Full automation from tox.ini
```

For training-related CLI work, the environment often also needs:

```bash
uv sync --extra torch-cu130 --extra mlflow --extra flops
```

## Code Conventions

- Python target is `>=3.11` (set in `pyproject.toml`). Use modern union syntax `X | Y` instead of `Union[X, Y]`.
- Pydantic v2 is used throughout builders and schemas.
- Google-style docstrings are expected.
- Dataclasses use `@dataclass(kw_only=True, slots=True)`.
- Lazy import wrappers are used broadly:
  - `LazyModuleImporter` for optional heavy dependencies (torch, timm, mlflow).
  - `LazySelectedImporter` for module export surfaces (`__all__`).
- Generated code should stay minimal and preserve current public APIs.
- The `outputs` attribute on a generated learner is significant — the CLI reads it to determine which keys `TorchTracker` should track, falling back to `--learner-outputs`.
- A method named after a lifecycle event (`on_epoch_end`, `on_update`, …) on any participant is live code: the trainer will call it. Do not add such names for unrelated purposes.

## Testing Notes

- Tests mirror the source layout: builders, commands, trainer, and torch layers each have dedicated test modules.
- CLI tests patch command callback globals directly because lazy import wrappers make normal monkeypatching less reliable.
- Trainer tests often patch function globals instead of module attributes for the same reason.
- The pytest configuration runs doctests in `src/` as well as tests under `tests/`.

## Common Failure Modes

| Error | Cause | Fix |
| --- | --- | --- |
| `ValueError: Each model pattern should contain exactly one model definition` | A positional model argument included multiple names in one YAML dict. | Split into separate positional arguments, one model name per dict. |
| `Module "learner" does not have an "outputs" attribute` | The generated or custom learner does not expose `outputs`, and no CLI default was provided. | Define `outputs` on the learner or pass `-LO/--learner-outputs`. |
| `ValueError: Invalid tensor shape` | `-s/--shape` was not a tuple/list/dict of integers. | Use shapes like `'image: [3, 224, 224]'`. |
| `ValueError: Mixup is not active` | Code accessed `TimmDataLoaderWrapper.mixup` (`examples/torch/data.py`) while mixup/cutmix settings were disabled. | Enable `mixup_alpha`, `cutmix_alpha`, or `cutmix_minmax` first. |
| CUDA requested but CPU used | `get_torch_device("cuda")` falls back when CUDA is unavailable. | Verify PyTorch CUDA installation. |
| Flax/Keras GPU not detected | JAX cannot find NVIDIA libraries. | Set `LD_LIBRARY_PATH` to include CUDA/cuDNN paths. |
| `RuntimeError: Address already in use` during distributed training | Another process is using the `MASTER_PORT`. | Change `--master_port` or kill the conflicting process. |
| All ranks log to the tracking service / print progress bars | Rank gating is not working correctly. | Verify `initial_distributed_env()` is called before logging setup. |
| `ValueError: No data provider was given to the trainer` | `fit()` was called on a trainer constructed without `data=`. | Pass a `DataProvider`, or call `train(dataset)` / `evaluate(dataset)` directly. |
| A callback never fires | Its method name does not match an event, or the object was not passed to the trainer. | Check `trainer.describe()` and the spelling against `EVENTS`. |

## Key Integration Example

The ConvNeXtV2 example demonstrates the full end-to-end workflow:

1. Generate `model.py` from `cfg/torch/models/ConvNeXtV2.yaml` (or from `cfg/flax/models/` / `cfg/keras/models/` for other frameworks).
2. Generate `learner.py` from `cfg/torch/learners/ConvNeXtV2.yaml`, which also brings in the losses, metrics, and the optimizer composition it references by `_file_`.
3. Format `dataset_train.yaml` and `dataset_valid.yaml` from `cfg/torch/others/default_timm.yaml`.
4. Benchmark with `scm [torch/flax/keras] time`, or train through `scm torch train` using `_file_`-based StructCast object patterns.

This **generate-then-reimport** loop is the core mental model for the entire repository: YAML templates become Python modules through the builders (generation phase), then those modules are re-imported through StructCast patterns and executed by the CLI (inference benchmarking or training).

Model generation is available for all three frameworks. Training is currently PyTorch-only; for distributed training, the execution phase is launched through `torchrun` instead of a direct `scm` invocation. The generation phase is identical — the same generated files and dataset YAML work for both single-GPU and multi-GPU training.
