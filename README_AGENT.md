# StructCast-Model — AI Agent Reference

> This document is written for AI coding agents. For human-oriented usage, see [README.md](README.md).
> For the upstream library, see the [StructCast repository](https://github.com/f6ra07nk14/structcast).

## What This Project Does

StructCast-Model turns YAML templates into executable models and training systems across multiple frameworks. It has four responsibilities:

1. **Code generation**: Generate model classes from declarative YAML templates — PyTorch `nn.Module`, Flax `nnx.Module`, and Keras `Layer`. PyTorch also supports backward/optimizer orchestration class generation.
2. **Template rendering**: Format parameterized YAML templates into concrete runtime configurations.
3. **Inference benchmarking**: Measure model inference time via `scm [torch/flax/keras] time`.
4. **Training execution**: Instantiate generated artifacts through [StructCast](https://github.com/f6ra07nk14/structcast) object patterns and run them via `scm torch train` (PyTorch only; Flax and Keras training is planned).

## Repository Map

```text
cfg/torch/
├── backwards/                 # Backward, optimizer, scheduler templates
├── datasets/                  # Reusable timm dataset/dataloader templates
├── losses/                    # Loss layer templates
├── metrics/                   # Metric layer templates
├── models/                    # Model architecture templates
└── others/                    # Other templates (e.g. compile settings)
cfg/flax/
└── models/                    # Flax model architecture templates
cfg/keras/
└── models/                    # Keras model architecture templates

src/structcast_model/
├── base_trainer.py            # Generic trainer, callback system, best-criterion handling
├── builders/
│   ├── base_builder.py        # Generic template -> intermediate -> script pipeline
│   ├── schema.py              # Pydantic schemas for layer/backward templates
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
│   ├── trainer.py             # Training steps, tracker, timm wrappers, trainer
│   ├── optimizers.py          # Optimizer/scheduler helpers used by backward templates
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
  |  TemplateLayer / TemplateBackward validation     <- builders/schema.py
  v
Builder intermediate objects
  |  BaseModelBuilder / BaseBackwardBuilder          <- builders/base_builder.py
  |  TorchBuilder / FlaxBuilder / KerasBuilder       <- builders/{torch,flax,keras}_builder.py
  |  TorchBackwardBuilder                            <- builders/torch_builder.py
  v
Generated Python source files
  |  scm [torch/flax/keras] create model
  |  scm torch create backward (PyTorch only)
  v
StructCast object patterns
  |  _obj_ + _addr_ + _file_ + _call_               <- commands/cmd_{torch,flax,keras}.py
  v
Live model objects
  |  Inference benchmarking                          <- scm [torch/flax/keras] time
  |  Training (PyTorch only)                         <- scm torch train
  v
TorchTrainer.fit(...)  (PyTorch training path)
  |  train/evaluate loop + callbacks                 <- base_trainer.py + torch/trainer.py
  v
MLflow logs + model states + best checkpoints
```

The repository's signature workflow (the "generate-then-reimport" loop) is:

1. Render specialized YAML from templates with `scm format`.
2. Generate model Python modules with `scm [torch/flax/keras] create model` (plus loss/metric/backward for PyTorch).
3. Re-import those modules through StructCast `_file_` patterns at runtime.
4. Benchmark with `scm [torch/flax/keras] time`, or train through `scm torch train` (PyTorch only).

## CLI Surface

The CLI entry point is defined in `pyproject.toml` as `scm = "structcast_model.commands.main:app"` (a [Typer](https://typer.tiangolo.com/) application).

### Top-level commands

- `scm format`
- `scm torch create model`
- `scm torch create backward`
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

### `scm torch create backward`

Purpose:

- Load a backward template.
- Build a `TorchBackwardIntermediate`.
- Optionally write generated Python to disk.

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

- Instantiate models, losses, metrics, backward logic, datasets, and compile settings.
- Run a training loop via `TorchTrainer`.
- Log metrics and states to MLflow.

Key runtime behavior:

- `configure_security(allowed_modules_check=False)` is called because commands frequently import generated local files via `_file_` patterns.
- `torch.compile` is optional and configured via `-c/--compile`.
- Mixed precision dtype is taken from the backward object when available; otherwise from the `--mixed-precision` CLI flag.
- Validation progress uses `tqdm` unless `--ci` is enabled.

Distributed training behavior (when launched through `torchrun`):

- `initial_distributed_env()` detects `RANK`/`LOCAL_RANK`/`WORLD_SIZE` env vars and initializes the NCCL process group.
- Each process is assigned to `cuda:<LOCAL_RANK>`.
- All models are wrapped with `DistributedDataParallel`.
- `TimmDataLoaderWrapper` creates `DistributedSampler` automatically; `set_epoch()` is called each epoch.
- `TorchTracker` uses `all_reduce(ReduceOp.AVG)` to synchronize metrics across ranks.
- MLflow logging, checkpoints, and progress bars are gated to rank 0 only.
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
- `BackwardIntermediate`
- `BaseBackwardBuilder`

### PyTorch builder layer

`builders/torch_builder.py` specializes the generic intermediates into concrete PyTorch code.

Important classes:

- `TorchLayerIntermediate`: renders `torch.nn.Module` classes with `forward()`.
- `TorchBuilder`: main entry point for model generation.
- `TorchBackwardIntermediate`: renders the generated backward class.
- `TorchBackwardBuilder`: main entry point for backward generation.

Important generation details:

- Model code uses `self.<layer_name>` submodules.
- Inference flow is rendered separately when `INFERENCE_FLOW` is present.
- Backward code can include gradient accumulation, AMP scaler logic, clipping, optimizer stepping, and optimizer metadata properties.

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
- `Callbacks`: hook container
- `GLOBAL_CALLBACKS`: shared callback registry
- `BaseTrainer`: `train()`, `evaluate()`, `fit()` loop
- `BestCriterion`: criterion monitor for best-value callbacks

### PyTorch runtime layer

`torch/trainer.py` provides the runtime objects actually used by the CLI.

Utility functions:

- `create_torch_inputs(shape)`
- `get_torch_device(device=None)`
- `initial_distributed_env(device, dist_backend, dist_url)` — detects torchrun env vars, initializes process group, returns per-rank device
- `initial_model(model, shapes=None, compile_fn=None)`
- `get_autocast(mixed_precision_type, device)`

Training/evaluation helpers:

- `TrainingStep`
- `ValidationStep`
- `TorchTracker`
- `TorchTrainer`

timm integrations:

- `TimmDatasetWrapper`
- `TimmDataLoaderWrapper`

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

1. Datasets are instantiated from YAML or inline StructCast patterns.
2. Models are instantiated and optionally initialized with dummy inputs.
3. Loss and metric modules are instantiated.
4. The backward class is instantiated with the created models.
5. `TorchTracker` is built from declared output names.
6. `TorchTrainer` runs the loop and invokes callbacks.
7. MLflow receives arguments, metrics, artifacts, training state, and best-checkpoint snapshots.

When running under `torchrun`, the flow gains additional distributed steps:

8. Process group is initialized and per-rank device is assigned.
9. Models are wrapped with `DistributedDataParallel`.
10. Metrics are synchronized across ranks via `all_reduce`.
11. Only rank 0 writes MLflow logs, checkpoints, and progress output.
12. `destroy_process_group()` is called during cleanup.

## Pattern Alias Quick Reference

### Instantiator patterns used in this repository

These are [StructCast](https://github.com/f6ra07nk14/structcast) object pattern aliases. See the StructCast README for full pattern documentation.

| Alias | Meaning | Typical use here |
| --- | --- | --- |
| `_addr_` | Import by dotted address | Import timm layers, torch layers, helper functions |
| `_file_` | Load from a local Python file | Import generated `model.py`, `loss.py`, `metric.py`, `backward.py` |
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

`cfg/torch/datasets/default_timm.yaml` uses a `FlexSpec`-compatible mapping so dataloader batches can be transformed from positional `(input, target)` tuples into structured dictionaries such as `{image: ..., label: ...}`.

## Dynamic Import and Security Notes

- CLI commands call `configure_security(allowed_modules_check=False)` to disable the StructCast module allowlist, because generated local files (loaded via `_file_`) would otherwise be blocked.
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
- `outputs` attributes on generated loss and metric modules are significant — the CLI reads them to determine which keys `TorchTracker` should track.

## Testing Notes

- Tests mirror the source layout: builders, commands, trainer, and torch layers each have dedicated test modules.
- CLI tests patch command callback globals directly because lazy import wrappers make normal monkeypatching less reliable.
- Trainer tests often patch function globals instead of module attributes for the same reason.
- The pytest configuration runs doctests in `src/` as well as tests under `tests/`.

## Common Failure Modes

| Error | Cause | Fix |
| --- | --- | --- |
| `ValueError: Each model pattern should contain exactly one model definition` | A positional model argument included multiple names in one YAML dict. | Split into separate positional arguments, one model name per dict. |
| `Module "loss" does not have an "outputs" attribute` | Generated or custom loss/metric modules do not expose `outputs`, and CLI defaults were not provided. | Define `outputs` on the module or pass `--loss-outputs` / `--metric-outputs`. |
| `ValueError: Invalid tensor shape` | `-s/--shape` was not a tuple/list/dict of integers. | Use shapes like `'image: [3, 224, 224]'`. |
| `ValueError: Mixup is not active` | Code accessed `TimmDataLoaderWrapper.mixup` while mixup/cutmix settings were disabled. | Enable `mixup_alpha`, `cutmix_alpha`, or `cutmix_minmax` first. |
| CUDA requested but CPU used | `get_torch_device("cuda")` falls back when CUDA is unavailable. | Verify PyTorch CUDA installation. |
| Flax/Keras GPU not detected | JAX cannot find NVIDIA libraries. | Set `LD_LIBRARY_PATH` to include CUDA/cuDNN paths. |
| `RuntimeError: Address already in use` during distributed training | Another process is using the `MASTER_PORT`. | Change `--master_port` or kill the conflicting process. |
| All ranks log to MLflow / print progress bars | Rank gating is not working correctly. | Verify `initial_distributed_env()` is called before logging setup. |

## Key Integration Example

The ConvNeXtV2 example demonstrates the full end-to-end workflow:

1. Generate `model.py` from `cfg/torch/models/ConvNeXtV2.yaml` (or from `cfg/flax/models/` / `cfg/keras/models/` for other frameworks).
2. Generate `loss.py`, `metric.py`, and `backward.py` from their respective PyTorch templates.
3. Format `dataset_train.yaml` and `dataset_valid.yaml` from `cfg/torch/datasets/default_timm.yaml`.
4. Benchmark with `scm [torch/flax/keras] time`, or train through `scm torch train` using `_file_`-based StructCast object patterns.

This **generate-then-reimport** loop is the core mental model for the entire repository: YAML templates become Python modules through the builders (generation phase), then those modules are re-imported through StructCast patterns and executed by the CLI (inference benchmarking or training).

Model generation is available for all three frameworks. Training is currently PyTorch-only; for distributed training, the execution phase is launched through `torchrun` instead of a direct `scm` invocation. The generation phase is identical — the same generated files and dataset YAML work for both single-GPU and multi-GPU training.
