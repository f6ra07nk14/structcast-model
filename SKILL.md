---
name: structcast-model
description: StructCast-Model generates PyTorch, Flax (JAX), and Keras models — plus PyTorch training workflows — from YAML templates built on StructCast. Use this skill when working with scm CLI commands (format, torch/flax/keras create, torch/flax/keras time, torch train, torch ptflops, torch calflops), StructCast object patterns (_obj_, _addr_, _file_, _call_, _bind_, _attr_), YAML template formatting, code generation through TorchBuilder, FlaxBuilder, KerasBuilder, or TorchBackwardBuilder, PyTorch training orchestration through TrainingStep, ValidationStep, TorchTracker, TorchTrainer, timm dataset wrappers, MLflow-integrated training runs, or distributed multi-GPU training with torchrun and DistributedDataParallel (DDP).
---

# StructCast-Model

Capability reference for the repository, organized by workflow and module entry point.

Upstream library: [StructCast](https://github.com/f6ra07nk14/structcast)

## Quick Reference

**Install runtime extras**: `uv sync --extra torch-cu130 --extra mlflow --extra flops` (PyTorch) or `uv sync --extra all-cpu` (all frameworks)

**Format config**: `scm format cfg/torch/datasets/default_timm.yaml -o dataset.yaml -p 'DEFAULT: {...}'`

**Generate PyTorch model**: `scm torch create model cfg/torch/models/ConvNeXtV2.yaml -o model.py`

**Generate Flax model**: `scm flax create model cfg/flax/models/ConvNeXtV2.yaml -o model.py`

**Generate Keras model**: `scm keras create model cfg/keras/models/ConvNeXtV2.yaml -o model.py`

**Generate backward**: `scm torch create backward cfg/torch/backwards/ConvNeXtV2.yaml -o backward.py`

**Inspect FLOPs**: `scm torch ptflops '[_obj_, {_addr_: model.Model, _file_: model.py}, _call_]' -s 'image: [3, 224, 224]'`

**Measure inference time**: `scm [torch/flax/keras] time '[_obj_, ...]' -s 'image: [3, 224, 224]' -d cuda`

**Train**: `scm torch train 'model: [_obj_, {_addr_: model.Model, _file_: model.py}, _call_]' ...`

**Distributed train**: `torchrun --nproc_per_node=gpu -m structcast_model.commands.main torch train ...`

## Common Workflows

### Workflow 1: Generate a Model from YAML

Each framework has its own `create model` command. The pipeline is the same — only the builder and output class differ.

```bash
# PyTorch → torch.nn.Module
scm torch create model cfg/torch/models/ConvNeXtV2.yaml \
  -p 'DEFAULT: {backbone: femto}' -o torch_model.py

# Flax → flax.nnx.Module
scm flax create model cfg/flax/models/ConvNeXtV2.yaml \
  -p 'DEFAULT: {backbone: femto}' -o flax_model.py

# Keras → keras.layers.Layer
scm keras create model cfg/keras/models/ConvNeXtV2.yaml \
  -p 'DEFAULT: {backbone: femto}' -o keras_model.py
```

What happens:

1. `[Torch/Flax/Keras]Builder.from_path(...)` loads and validates the YAML template.
2. `BaseModelBuilder` resolves user-defined layers, imports, inputs, outputs, and flow.
3. Framework-specific intermediate renders the corresponding module implementation.
4. The intermediate writes the generated source file.

### Workflow 2: Generate Loss, Metric, and Backward Code

```bash
scm torch create model cfg/torch/losses/cls.yaml -c Loss -o loss.py
scm torch create model cfg/torch/metrics/topk.yaml -c Metric -o metric.py
scm torch create backward cfg/torch/backwards/ConvNeXtV2.yaml -p 'DEFAULT: {epochs: 5}' -o backward.py
```

Use this when the training workflow should remain fully declarative.

### Workflow 3: Format a Reusable Dataset Template

```bash
scm format cfg/torch/datasets/default_timm.yaml \
  -o dataset_train.yaml \
  -p 'DEFAULT: {training: true, dataset: torch/cifar100, num_classes: 100, input_size: [3, 224, 224], download: true}'
```

What happens:

1. `commands/main.py` loads the template through `schema.Template.from_path(...)`.
2. Parameter groups are merged.
3. The rendered YAML becomes a StructCast object pattern that instantiates `TimmDataLoaderWrapper.model_validate(...)`.

### Workflow 4: Run FLOPs Inspection on a Generated Model

```bash
scm torch ptflops '[_obj_, {_addr_: model.Model, _file_: model.py}, _call_]' \
  -s 'image: [3, 224, 224]' \
  --backend pytorch
```

What happens:

1. The model is instantiated from a StructCast `_obj_` pattern.
2. Dummy inputs are built with `create_torch_inputs(...)`.
3. `initial_model(...)` performs an initialization forward pass.
4. The model is passed to `ptflops` or `calflops`.

### Workflow 5: Train End-to-End

```bash
scm torch train \
  'model: [_obj_, {_addr_: model.Model, _file_: model.py}, _call_]' \
  -s 'image: [3, 224, 224]' \
  -d cuda \
  --ema cfg/torch/others/ema.yaml \
  -L '[_obj_, {_addr_: loss.Loss, _file_: loss.py}, _call_]' \
  -M '[_obj_, {_addr_: metric.Metric, _file_: metric.py}, _call_]' \
  -B '[_obj_, {_addr_: backward.Backward, _file_: backward.py}]' \
  -c cfg/torch/others/compile_default.yaml \
  -T dataset_train.yaml \
  -V dataset_valid.yaml \
  -LC ce_loss -LC val_ce_loss \
  -HC acc1 -HC val_acc1
```

What happens:

1. Datasets are instantiated and counted.
2. Models are initialized and optionally compiled.
3. Loss, metric, backward, and EMA objects are instantiated.
4. `TorchTracker` is built from output names.
5. `TorchTrainer` runs the loop and MLflow logging is attached.

### Workflow 6: Measure Inference Time

All three frameworks support inference benchmarking via `scm [torch/flax/keras] time`:

```bash
# PyTorch
scm torch time \
  '[_obj_, {_addr_: model.Model, _file_: torch_model.py}, _call_]' \
  -s 'image: [3, 224, 224]' -c cfg/torch/others/compile_default.yaml -d cuda

# Flax (channel-last layout: H×W×C)
scm flax time \
  '[_obj_, {_addr_: model.Model, _file_: flax_model.py}, {_call_: {rngs: [_obj_, _addr_: flax.nnx.Rngs, _call_: {params: 0, dropout: 1}]}}]' \
  -s 'image: [224, 224, 3]' -c true -d gpu:0

# Keras (channel-last layout: H×W×C)
scm keras time \
  '[_obj_, {_addr_: model.Model, _file_: keras_model.py}, _call_]' \
  -s 'image: [224, 224, 3]' -c true -d gpu:0
```

What happens:

1. The model is instantiated from the StructCast pattern.
2. Dummy inputs are created (`create_torch_inputs`, `create_jax_inputs`, or `create_numpy_inputs`).
3. Optional compilation is applied (`torch.compile`, `nnx.jit`, or `keras.Model.compile`).
4. Warmup runs are executed, then timed iterations are averaged.

### Workflow 7: Distributed Training with `torchrun`

The same `scm torch train` command supports multi-GPU and multi-node distributed training when launched through `torchrun`:

```bash
# Single-node, all GPUs
torchrun --nproc_per_node=gpu \
  -m structcast_model.commands.main \
  torch train \
  'model: [_obj_, {_addr_: model.Model, _file_: model.py}, _call_]' \
  -s 'image: [3, 224, 224]' \
  -d cuda \
  --ema cfg/torch/others/ema.yaml \
  -L '[_obj_, {_addr_: loss.Loss, _file_: loss.py}, _call_]' \
  -M '[_obj_, {_addr_: metric.Metric, _file_: metric.py}, _call_]' \
  -B '[_obj_, {_addr_: backward.Backward, _file_: backward.py}]' \
  -c cfg/torch/others/compile_default.yaml \
  -T dataset_train.yaml \
  -V dataset_valid.yaml \
  -LC ce_loss -LC val_ce_loss \
  -HC acc1 -HC val_acc1

# Multi-node (2 nodes × 4 GPUs)
torchrun --nproc_per_node=4 --nnodes=2 --node_rank=0 \
  --master_addr=192.168.1.100 --master_port=29500 \
  -m structcast_model.commands.main \
  torch train ...
```

What happens:

1. `torchrun` sets `RANK`, `LOCAL_RANK`, `WORLD_SIZE`, `MASTER_ADDR`, `MASTER_PORT` environment variables.
2. `initial_distributed_env()` detects the distributed environment and initializes the NCCL process group.
3. Each model is wrapped with `DistributedDataParallel`.
4. `TimmDataLoaderWrapper` creates a `DistributedSampler` and calls `set_epoch()` each epoch.
5. `TorchTracker` uses `all_reduce` to average metrics across ranks.
6. MLflow logging and checkpoints are gated to rank 0 only.
7. DDP gradient sync is skipped during gradient accumulation steps.

## CLI Surface

| Command | Module | Primary entry point |
| -- | -- | -- |
| `scm format` | `commands.main` | `format_template()` |
| `scm torch create model` | `commands.cmd_torch` | `create_model()` |
| `scm torch create backward` | `commands.cmd_torch` | `create_backward()` |
| `scm torch ptflops` | `commands.cmd_torch` | `call_ptflops()` |
| `scm torch calflops` | `commands.cmd_torch` | `call_calflops()` |
| `scm torch time` | `commands.cmd_torch` | `measure_inference_time()` |
| `scm torch train` | `commands.cmd_torch` | `train()` |
| `scm flax create model` | `commands.cmd_flax` | `create_model()` |
| `scm flax time` | `commands.cmd_flax` | `measure_inference_time()` |
| `scm keras create model` | `commands.cmd_keras` | `create_model()` |
| `scm keras time` | `commands.cmd_keras` | `measure_inference_time()` |

### Important CLI conventions

- Model arguments for `ptflops`, `calflops`, `time`, and `train` are [StructCast](https://github.com/f6ra07nk14/structcast) object patterns, not plain import strings.
- Dataset arguments can be rendered YAML files or inline StructCast patterns.
- `configure_security(allowed_modules_check=False)` is called in CLI paths because generated local modules are imported via `_file_`.
- Flax and Keras use channel-last tensor layout (*H × W × C*); PyTorch uses channel-first (*C × H × W*).

## Builder APIs

**Modules**: `structcast_model.builders.base_builder`, `structcast_model.builders.torch_builder`, `structcast_model.builders.flax_builder`, `structcast_model.builders.keras_builder`

### Generic generation layer

| Capability | Entry point | Purpose |
| -- | -- | -- |
| Resolve object pattern to code | `resolve_object(imports, pattern)` | Build Python expression strings and collect imports |
| Resolve spec to getter code | `resolve_getter(imports, spec, variable=None)` | Convert StructCast specs into Python expressions |
| Write generated module | `_Intermediate.__call__(module_path)` | Serialize imports + scripts to disk |
| Build layer graph intermediate | `BaseModelBuilder(...)` | Parse template and create flow graph |
| Build backward intermediate | `BaseBackwardBuilder(...)` | Parse optimizer/loss/backward config |

### PyTorch generation layer

| Capability | Entry point | Purpose |
| -- | -- | -- |
| Generate model intermediate | `TorchBuilder.from_path(path)(...)` | Build `TorchLayerIntermediate` |
| Generate backward intermediate | `TorchBackwardBuilder.from_path(path)(...)` | Build `TorchBackwardIntermediate` |
| Render `torch.nn.Module` code | `TorchLayerIntermediate._get_layer_script(...)` | Emit model class source |
| Render backward runtime code | `TorchBackwardIntermediate._get_scripts()` | Emit backward/optimizer class source |

### Flax generation layer

| Capability | Entry point | Purpose |
| -- | -- | -- |
| Generate model intermediate | `FlaxBuilder.from_path(path)(...)` | Build `FlaxLayerIntermediate` |
| Render `flax.nnx.Module` code | `FlaxLayerIntermediate._get_layer_script(...)` | Emit Flax module class source |

### Keras generation layer

| Capability | Entry point | Purpose |
| -- | -- | -- |
| Generate model intermediate | `KerasBuilder.from_path(path)(...)` | Build `KerasLayerIntermediate` |
| Render `keras.layers.Layer` code | `KerasLayerIntermediate._get_layer_script(...)` | Emit Keras layer class source |

### Builder usage pattern

```python
from structcast_model.builders.torch_builder import TorchBuilder

built = TorchBuilder.from_path("cfg/torch/models/ConvNeXtV2.yaml")(
    parameters={"DEFAULT": {"backbone": "femto"}},
    classname="Model",
    forced_structured_output=True,
)

print(built.scripts[0])
built("model.py")
```

The same `.from_path(...)(...)(output_path)` pattern applies to `FlaxBuilder` and `KerasBuilder`.

## Training Runtime APIs

**Module**: `structcast_model.torch.trainer`

### Utility layer

| Capability | Entry point | Purpose |
| -- | -- | -- |
| Dummy inputs | `create_torch_inputs(shape)` | Build tensors from tuple/list/dict shape specs |
| Device selection | `get_torch_device(device=None)` | Resolve `cpu` vs `cuda` with fallback |
| Initialize model | `initial_model(model, shapes=None, compile_fn=None)` | Run warm-up forward pass and optional compile |
| Build AMP context | `get_autocast(mixed_precision_type, device)` | Return `torch.autocast` partial or `suppress` |

### Step and tracker layer

| Capability | Entry point | Purpose |
| -- | -- | -- |
| Training step | `TrainingStep(...)` | Sequential forward pass + loss/metric computation |
| Validation step | `ValidationStep(...)` | Evaluation-time forward pass under `torch.no_grad()` |
| Criteria tracking | `TorchTracker.from_criteria(...)` | Build loss/metric trackers and reset callbacks |
| Device-aware trainer | `TorchTrainer(...)` | Specialize `BaseTrainer` with CUDA synchronization |
| Distributed env init | `initial_distributed_env(...)` | Detect torchrun env, init process group, resolve per-rank device |

### timm integration layer

| Capability | Entry point | Purpose |
| -- | -- | -- |
| Dataset wrapper | `TimmDatasetWrapper` | Lazily call `timm.data.create_dataset(...)` |
| Dataloader wrapper | `TimmDataLoaderWrapper` | Lazily call `timm.data.create_loader(...)` |
| EMA wrapper | `TimmEmaWrapper.from_models(...)` | Manage `ModelEmaV3` instances and update callbacks |

### Distributed training layer

| Capability | Entry point | Purpose |
| -- | -- | -- |
| Distributed environment detection | `initial_distributed_env(device, ...)` | Read `RANK`/`LOCAL_RANK`/`WORLD_SIZE` env vars, init process group |
| DDP model wrapping | `DistributedDataParallel` (via `cmd_torch.py`) | Wrap models for multi-GPU gradient synchronization |
| Cross-rank metric averaging | `TorchTracker.__call__()` | `all_reduce` with `ReduceOp.AVG` when distributed |
| Gradient sync optimization | `TorchTrainer.no_sync()` | Skip DDP gradient sync during accumulation steps |
| EMA DDP unwrapping | `TimmEmaWrapper.update()` | Unwrap DDP module before EMA weight update |

## Config and Pattern Vocabulary

### StructCast object patterns used in this repository

See the [StructCast README](https://github.com/f6ra07nk14/structcast) for full pattern documentation.

| Alias | Meaning | Example |
| -- | -- | -- |
| `_obj_` | Chain object-building operations | `[_obj_, {_addr_: model.Model, _file_: model.py}, _call_]` |
| `_addr_` | Import by dotted path | `{_addr_: torch.nn.Identity}` |
| `_file_` | Resolve symbol from a local file | `{_addr_: model.Model, _file_: model.py}` |
| `_call_` | Invoke current callable | `_call_` or `{_call_: {out_features: 1000}}` |
| `_bind_` | Partially apply arguments | `{_bind_: {optimizer_kwargs: {...}}}` |
| `_attr_` | Access attribute or method | `{_attr_: model_validate}` |

### Template features used in YAML

| Syntax | Meaning |
| -- | -- |
| `_jinja_yaml_` | Render Jinja to YAML and parse it |
| `_jinja_group_` | Select parameter group such as `backbone` |
| `DEFAULT` | Default parameter group |
| `SHARED` | Shared parameters across groups |
| `eval: ...` | Inject raw Python expression into generated code |

### Signature config examples in this repo

- `cfg/torch/models/ConvNeXtV2.yaml` uses nested user-defined layers and Jinja-expanded blocks (PyTorch channel-first).
- `cfg/flax/models/ConvNeXtV2.yaml` mirrors the PyTorch model for Flax `nnx.Module` (channel-last, `rngs` constructor arg).
- `cfg/keras/models/ConvNeXtV2.yaml` mirrors the PyTorch model for Keras `Layer` (channel-last, multi-backend).
- `cfg/torch/backwards/ConvNeXtV2.yaml` uses optimizer factories, scheduler settings, optional clipping, and gradient accumulation.
- `cfg/torch/datasets/default_timm.yaml` formats into a `TimmDataLoaderWrapper.model_validate(...)` object pattern.

## Base Trainer and Callback System

**Module**: `structcast_model.base_trainer`

| Capability | Entry point |
| -- | -- |
| Dataset normalization | `get_dataset(dataset)` |
| Dataset size detection | `get_dataset_size(dataset)` |
| Callback invocation | `invoke_callback(callbacks, info, ...)` |
| Shared callback container | `Callbacks` |
| Global callback registry | `GLOBAL_CALLBACKS` |
| Generic train/eval loop | `BaseTrainer` |
| Best-criterion monitor | `BestCriterion` |

Use this layer when the task is about callback ordering, history storage, epoch/step/update semantics, or best-model tracking independent of the torch-specific wrapper.

## Development Commands

```bash
uv sync --group dev
pytest
ruff check src tests
ruff format src tests
mypy src && mypy tests
tox
```

For runtime CLI workflows, also install the necessary extras:

```bash
uv sync --extra torch-cu130 --extra mlflow --extra flops
```

## Troubleshooting

### Common Errors

**`ValueError: Each model pattern should contain exactly one model definition`**
- Cause: a positional model argument passed multiple names in one dictionary.
- Solution: split them into separate positional arguments, one named model per object.

**`Module "loss" does not have an "outputs" attribute`**
- Cause: the loss or metric module does not expose `outputs` and no CLI fallback outputs were provided.
- Solution: define `outputs` on the module or pass `--loss-outputs` / `--metric-outputs`.

**`ValueError: Invalid tensor shape`**
- Cause: a shape spec was not a nested tuple/list/dict of integers.
- Solution: pass shapes like `'image: [3, 224, 224]'`.

**`ValueError: Mixup is not active`**
- Cause: `TimmDataLoaderWrapper.mixup` was accessed with all mixup/cutmix settings disabled.
- Solution: enable `mixup_alpha`, `cutmix_alpha`, or `cutmix_minmax` before using `mixup`.

**CUDA requested but training runs on CPU**
- Cause: `get_torch_device("cuda")` falls back when CUDA is unavailable.
- Solution: verify PyTorch CUDA installation and runtime availability.

**Generated file import fails**
- Cause: `_file_` path in the StructCast pattern does not point to an existing generated module.
- Solution: regenerate the file and verify the exact path used in the pattern.

## Mental Model

The repository operates as a two-phase system:

1. **Generation phase**: YAML templates under `cfg/[torch/flax/keras]/` are transformed into Python modules through framework-specific builders (`TorchBuilder`, `FlaxBuilder`, `KerasBuilder`).
2. **Execution phase**: Generated modules are re-imported through StructCast `_file_` patterns and executed by `scm [torch/flax/keras] time` (inference benchmarking) or `scm torch train` (training, PyTorch only).

Model code generation is available for all three frameworks. Training workflow generation and `scm torch train` are currently PyTorch-only; Flax and Keras training support is planned.

If a task relates to YAML templates, import resolution, generated source code, optimizer orchestration, inference benchmarking, or the training command, this skill is the correct reference.