# StructCast-Model -- AI Agent Reference

> This document is designed for AI coding agents. For human-oriented usage, see [README.md](README.md).

## What This Project Does

StructCast-Model turns YAML templates into executable PyTorch training systems. The repository has three main responsibilities:

1. Generate PyTorch model code from declarative templates.
2. Generate backward, optimizer, AMP, and scheduler orchestration code from declarative templates.
3. Run training workflows by instantiating generated artifacts through StructCast object patterns.

The active implementation is PyTorch-first. JAX and TensorFlow appear in extras and roadmap items, but the CLI, tests, and runtime integration in this repository are centered on PyTorch.

## Repository Map

```text
cfg/
├── backwards/                 # Backward, optimizer, scheduler templates
├── datasets/                  # Reusable timm dataset/dataloader templates
├── losses/                    # Loss layer templates
├── metrics/                   # Metric layer templates
├── models/                    # Model architecture templates
└── others/                    # Runtime presets such as compile and EMA

src/structcast_model/
├── base_trainer.py            # Generic trainer, callback system, best-criterion handling
├── builders/
│   ├── base_builder.py        # Generic template -> intermediate -> script pipeline
│   ├── schema.py              # Pydantic schemas for layer/backward templates
│   └── torch_builder.py       # PyTorch-specific code generation
├── commands/
│   ├── main.py                # Top-level scm CLI
│   ├── cmd_torch.py           # PyTorch CLI commands
│   └── utils.py               # CLI argument parsers and reducers
├── torch/
│   ├── trainer.py             # Training steps, tracker, EMA, timm wrappers, trainer
│   ├── optimizers.py          # Optimizer/scheduler helpers used by backward templates
│   ├── layers/                # Reusable torch layers referenced by templates
│   └── types.py               # Tensor aliases and related typing
└── utils/                     # YAML and helper utilities used by builders/commands

tests/
├── builders/                  # Builder and schema tests
├── commands/                  # CLI tests
├── torch/                     # Trainer and torch layer tests
├── fixtures/                  # YAML and data fixtures
└── test_base_trainer.py       # Generic trainer tests
```

## Data Flow -- How the Pieces Connect

```text
YAML template in cfg/
  |  TemplateLayer / TemplateBackward validation     <- builders/schema.py
  v
Builder intermediate objects
  |  BaseModelBuilder / BaseBackwardBuilder          <- builders/base_builder.py
  |  TorchBuilder / TorchBackwardBuilder             <- builders/torch_builder.py
  v
Generated Python source files
  |  scm torch create model / scm torch create backward
  v
StructCast object patterns
  |  _obj_ + _addr_ + _file_ + _call_               <- commands/cmd_torch.py
  v
Live models / losses / metrics / backward objects
  |  initial_model(), get_autocast(), trackers       <- torch/trainer.py
  v
TorchTrainer.fit(...)
  |  train/evaluate loop + callbacks                <- base_trainer.py + torch/trainer.py
  v
MLflow logs + model states + best checkpoints
```

The repository's signature workflow is:

1. Render specialized YAML from templates with `scm format`.
2. Generate model, loss, metric, and backward Python modules.
3. Re-import those modules through StructCast `_file_` patterns.
4. Train them through `scm torch train`.

## CLI Surface

The CLI entry point is `scm = structcast_model.commands.main:app`.

### Top-level commands

- `scm format`
- `scm torch create model`
- `scm torch create backward`
- `scm torch ptflops`
- `scm torch calflops`
- `scm torch train`

### `scm format`

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

### `scm torch train`

Purpose:

- Instantiate models, losses, metrics, backward logic, datasets, compile settings, and EMA.
- Run a training loop via `TorchTrainer`.
- Log metrics and states to MLflow.

Key runtime behavior:

- `configure_security(allowed_modules_check=False)` is used because commands frequently import generated local files.
- `torch.compile` is optional and configured via `-c/--compile`.
- Mixed precision dtype comes from the backward object when available, otherwise from the CLI flag.
- Validation progress uses `tqdm` unless `--ci` is enabled.

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
- `TimmEmaWrapper`

### Training flow in practice

1. Datasets are instantiated from YAML or inline StructCast patterns.
2. Models are instantiated and optionally initialized with dummy inputs.
3. Loss and metric modules are instantiated.
4. The backward class is instantiated with the created models.
5. `TorchTracker` is built from declared output names.
6. `TorchTrainer` runs the loop and invokes callbacks.
7. MLflow receives arguments, metrics, artifacts, training state, and best-checkpoint snapshots.

## Pattern Alias Quick Reference

### Instantiator patterns used heavily in this repository

| Alias | Meaning | Typical use here |
| --- | --- | --- |
| `_addr_` | Import by dotted address | Import timm layers, torch layers, helper functions |
| `_file_` | Load from a local Python file | Import generated `model.py`, `loss.py`, `metric.py`, `backward.py` |
| `_call_` | Call the imported symbol | Instantiate generated classes |
| `_bind_` | Partially apply arguments | Optimizer and scheduler factory configuration |
| `_attr_` | Resolve an attribute on the current object | `model_validate`, helper method access |
| `_obj_` | Chain all of the above | Main object construction mechanism |

### Template features used heavily in config

| Alias or syntax | Meaning |
| --- | --- |
| `_jinja_yaml_` | Render Jinja and parse as YAML |
| `_jinja_group_` | Select a named parameter group |
| `eval: ...` | Inject a raw expression into generated Python |
| `DEFAULT` / `SHARED` groups | Parameter groups used by template rendering |

### Spec usage in the dataset template

`cfg/datasets/default_timm.yaml` uses a `FlexSpec`-compatible mapping so dataloader batches can be transformed from positional `(input, target)` tuples into structured dictionaries such as `{image: ..., label: ...}`.

## Dynamic Import and Security Notes

- CLI commands intentionally disable the StructCast allowlist check when instantiating user-provided or generated runtime modules.
- Generated modules are commonly loaded with `_file_` patterns pointing at local files.
- When debugging command failures, confirm that generated files exist at the paths referenced in `_file_`.
- If code is being used outside the CLI, remember that StructCast security settings may block imports unless configured appropriately.

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

- Python target is 3.11. Prefer modern union syntax such as `X | Y`.
- Pydantic v2 is used throughout the builders and schemas.
- Google-style docstrings are expected.
- Dataclasses typically use `@dataclass(kw_only=True, slots=True)`.
- Lazy import wrappers are used broadly:
  - `LazyModuleImporter` for optional dependencies
  - `LazySelectedImporter` for module export surfaces
- Generated code should stay minimal and preserve current public APIs.
- `outputs` attributes on loss and metric modules matter because the CLI uses them to infer tracked keys.

## Testing Notes

- Tests mirror the source layout: builders, commands, trainer, and torch layers each have dedicated test modules.
- CLI tests patch command callback globals directly because lazy import wrappers make normal monkeypatching less reliable.
- Trainer tests often patch function globals instead of module attributes for the same reason.
- The pytest configuration runs doctests in `src/` as well as tests under `tests/`.

## Common Failure Modes

- `ValueError: Each model pattern should contain exactly one model definition`
  - Cause: a positional model argument passed multiple names in one YAML object.
- `Module "loss" does not have an "outputs" attribute`
  - Cause: generated or custom loss/metric modules do not expose `outputs`, and CLI defaults were not provided.
- `ValueError: Invalid tensor shape`
  - Cause: `-s/--shape` was not a tuple/list/dict shape structure.
- `ValueError: Mixup is not active`
  - Cause: code accessed `TimmDataLoaderWrapper.mixup` while mixup/cutmix settings were disabled.
- CUDA requested but CPU used
  - Cause: `get_torch_device("cuda")` falls back when CUDA is unavailable.

## Key Integration Example

The repository's core end-to-end workflow is the ConvNeXtV2 example:

1. Generate `model.py` from `cfg/models/ConvNeXtV2.yaml`.
2. Generate `loss.py`, `metric.py`, and `backward.py`.
3. Format `dataset_train.yaml` and `dataset_valid.yaml` from `cfg/datasets/default_timm.yaml`.
4. Train through `scm torch train` using `_file_`-based StructCast object patterns.

This generated-code-plus-pattern-reimport loop is the main mental model for the entire repository.
