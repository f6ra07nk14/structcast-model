---
name: structcast-model
description: StructCast-Model generates PyTorch models and training workflows from YAML templates built on StructCast. Use this skill when working with scm CLI commands (format, torch create, torch train, torch ptflops, torch calflops), StructCast object patterns (_obj_, _addr_, _file_, _call_, _bind_, _attr_), YAML template formatting, code generation through TorchBuilder or TorchBackwardBuilder, PyTorch training orchestration through TrainingStep, ValidationStep, TorchTracker, TorchTrainer, timm dataset wrappers, or MLflow-integrated training runs.
---

# StructCast-Model

Capability reference for the repository, organized by workflow and module entry point.

Upstream library: [StructCast](https://github.com/f6ra07nk14/structcast)

## Quick Reference

**Install runtime extras**: `uv sync --extra torch-cu130 --extra mlflow --extra flops`

**Format config**: `scm format cfg/datasets/default_timm.yaml -o dataset.yaml -p 'DEFAULT: {...}'`

**Generate model**: `scm torch create model cfg/models/ConvNeXtV2.yaml -o model.py`

**Generate backward**: `scm torch create backward cfg/backwards/ConvNeXtV2.yaml -o backward.py`

**Inspect FLOPs**: `scm torch ptflops '[_obj_, {_addr_: model.Model, _file_: model.py}, _call_]' -s 'image: [3, 224, 224]'`

**Train**: `scm torch train 'model: [_obj_, {_addr_: model.Model, _file_: model.py}, _call_]' ...`

## Common Workflows

### Workflow 1: Generate a Model from YAML

```bash
scm torch create model cfg/models/ConvNeXtV2.yaml \
  -p 'DEFAULT: {backbone: femto}' \
  -c Model \
  -o model.py
```

What happens:

1. `TorchBuilder.from_path(...)` loads and validates the YAML template.
2. `BaseModelBuilder` resolves user-defined layers, imports, inputs, outputs, and flow.
3. `TorchLayerIntermediate` renders a `torch.nn.Module` implementation.
4. The intermediate writes the generated source file.

### Workflow 2: Generate Loss, Metric, and Backward Code

```bash
scm torch create model cfg/losses/cls.yaml -c Loss -o loss.py
scm torch create model cfg/metrics/topk.yaml -c Metric -o metric.py
scm torch create backward cfg/backwards/ConvNeXtV2.yaml -p 'DEFAULT: {epochs: 5}' -o backward.py
```

Use this when the training workflow should remain fully declarative.

### Workflow 3: Format a Reusable Dataset Template

```bash
scm format cfg/datasets/default_timm.yaml \
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
  --ema cfg/others/ema.yaml \
  -l '[_obj_, {_addr_: loss.Loss, _file_: loss.py}, _call_]' \
  -m '[_obj_, {_addr_: metric.Metric, _file_: metric.py}, _call_]' \
  -b '[_obj_, {_addr_: backward.Backward, _file_: backward.py}]' \
  -c cfg/others/compile_default.yaml \
  -t dataset_train.yaml \
  -v dataset_valid.yaml \
  -LC ce_loss -LC val_ce_loss \
  -HC acc1 -HC val_acc1
```

What happens:

1. Datasets are instantiated and counted.
2. Models are initialized and optionally compiled.
3. Loss, metric, backward, and EMA objects are instantiated.
4. `TorchTracker` is built from output names.
5. `TorchTrainer` runs the loop and MLflow logging is attached.

## CLI Surface

| Command | Module | Primary entry point |
| -- | -- | -- |
| `scm format` | `commands.main` | `format_template()` |
| `scm torch create model` | `commands.cmd_torch` | `create_model()` |
| `scm torch create backward` | `commands.cmd_torch` | `create_backward()` |
| `scm torch ptflops` | `commands.cmd_torch` | `call_ptflops()` |
| `scm torch calflops` | `commands.cmd_torch` | `call_calflops()` |
| `scm torch train` | `commands.cmd_torch` | `train()` |

### Important CLI conventions

- Model arguments for `ptflops`, `calflops`, and `train` are [StructCast](https://github.com/f6ra07nk14/structcast) object patterns, not plain import strings.
- Dataset arguments can be rendered YAML files or inline StructCast patterns.
- `configure_security(allowed_modules_check=False)` is called in CLI paths because generated local modules are imported via `_file_`.

## Builder APIs

**Modules**: `structcast_model.builders.base_builder`, `structcast_model.builders.torch_builder`

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

### Builder usage pattern

```python
from structcast_model.builders.torch_builder import TorchBuilder

built = TorchBuilder.from_path("cfg/models/ConvNeXtV2.yaml")(
    parameters={"DEFAULT": {"backbone": "femto"}},
    classname="Model",
    forced_structured_output=True,
)

print(built.scripts[0])
built("model.py")
```

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

### timm integration layer

| Capability | Entry point | Purpose |
| -- | -- | -- |
| Dataset wrapper | `TimmDatasetWrapper` | Lazily call `timm.data.create_dataset(...)` |
| Dataloader wrapper | `TimmDataLoaderWrapper` | Lazily call `timm.data.create_loader(...)` |
| EMA wrapper | `TimmEmaWrapper.from_models(...)` | Manage `ModelEmaV3` instances and update callbacks |

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

- `cfg/models/ConvNeXtV2.yaml` uses nested user-defined layers and Jinja-expanded blocks.
- `cfg/backwards/ConvNeXtV2.yaml` uses optimizer factories, scheduler settings, optional clipping, and gradient accumulation.
- `cfg/datasets/default_timm.yaml` formats into a `TimmDataLoaderWrapper.model_validate(...)` object pattern.

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

1. **Generation phase**: YAML templates are transformed into Python modules through the builders.
2. **Execution phase**: Generated modules are re-imported through StructCast `_file_` patterns and executed by the training CLI.

If a task relates to YAML templates, import resolution, generated source code, optimizer orchestration, or the training command, this skill is the correct reference.