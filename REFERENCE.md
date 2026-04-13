# StructCast-Model Reference

> **This document contains the schema and API reference extracted from the main [README](README.md).** For installation, quick start, and CLI usage, see the README first.

## Table of Contents

- [StructCast-Model Reference](#structcast-model-reference)
  - [Table of Contents](#table-of-contents)
  - [Schema Reference](#schema-reference)
    - [Template Parameters](#template-parameters)
    - [Model Template Schema](#model-template-schema)
      - [`FLOW` and `INFERENCE_FLOW`](#flow-and-inference_flow)
      - [`FLOW` Entry Format](#flow-entry-format)
      - [`TYPE`, `PARAM`, and `CFG`](#type-param-and-cfg)
    - [Backward Template Schema](#backward-template-schema)
      - [`BACKWARDS`](#backwards)
      - [`BACKWARDS` Entry Keys](#backwards-entry-keys)
  - [API Reference: `base_trainer.py`](#api-reference-base_trainerpy)
    - [Utility functions](#utility-functions)
    - [Protocols](#protocols)
    - [State and callbacks](#state-and-callbacks)
    - [Core classes](#core-classes)
  - [API Reference: `trainer.py`](#api-reference-trainerpy)
    - [Utility functions](#utility-functions-1)
    - [Tracking and orchestration](#tracking-and-orchestration)
    - [timm integrations](#timm-integrations)

---

## Schema Reference

All configuration templates under `cfg/` follow a shared schema that controls how YAML files are parsed, rendered, and validated by the code generators. This section explains every top-level key and sub-key that appears in these templates.

### Template Parameters

Every YAML template may begin with an optional top-level `PARAMETERS` block that declares named sets of values consumed by the Jinja rendering engine.

**`PARAMETERS`** — The top-level container for all template variable groups. Any key nested inside `PARAMETERS` (other than `DEFAULT` and `SHARED`) is treated as a named group that can be selected at render time.

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

**`DEFAULT`** — Defines the default template variables. These values are active when no named group is selected and can be overridden at the command line with `-p 'DEFAULT: {key: value}'`.

```yaml
DEFAULT:
  backbone: atto
  epochs: 300
  lr: 4.0e-3
```

**`SHARED`** — Defines variables that are merged into **every** named group (including `DEFAULT`). Use `SHARED` for constants that apply to all backbone or variant choices.

```yaml
SHARED:
  stem_kernel_size: 4
  kernel_size: 7
  norm_eps: 1.0e-6
```

**Named groups** — Any key in `PARAMETERS` that is not `DEFAULT` or `SHARED` is a named parameter group — for example `atto`, `femto`, `tiny`, or `base`. A named group is activated via `_jinja_group_` and its variables (merged with `SHARED`) replace the template variables for that rendering scope.

```yaml
atto:
  dims: [40, 80, 160, 320]
  depths: [2, 2, 6, 2]
femto:
  dims: [48, 96, 192, 384]
  depths: [2, 2, 6, 2]
```

**`_jinja_yaml_`** — Embeds an inline Jinja template that is rendered and merged back into the surrounding YAML. The rendered result must itself be valid YAML. `_jinja_yaml_` blocks are evaluated with the currently active template variables and can emit any number of sibling YAML keys or list entries.

```yaml
_jinja_yaml_: |-
  {% if accumulate_gradients is none %}
  ACCUMULATE_GRADIENTS: null
  {% else %}
  ACCUMULATE_GRADIENTS: {{accumulate_gradients}}
  {% endif %}
```

Inside a `_jinja_yaml_` block you can also use standard Jinja control structures (`{% for %}`, `{% if %}`, `{% set %}`, etc.) as well as the custom filter `cumsum` (provided by `structcast_model.builders.jinja_filters`).

**`_jinja_group_`** — Selects a named parameter group from `PARAMETERS`, merging its values (together with `SHARED`) into the template variable scope for the enclosing block. `_jinja_group_` must appear alongside a `_jinja_yaml_` sibling that consumes the newly activated variables.

```yaml
- _jinja_group_: {{backbone}}
  _jinja_yaml_: |-
    - [_, cls, head, [_obj_, {_addr_: torch.nn.LazyLinear}, {_call_: {out_features: {{num_classes}}}}]]
```

When `backbone` resolves to `atto`, the `atto` group from `PARAMETERS` (merged with `SHARED`) becomes the local variable scope for the inner `_jinja_yaml_` block.

---

### Model Template Schema

The following keys appear in model configuration files such as [`cfg/torch/models/ConvNeXtV2.yaml`](cfg/torch/models/ConvNeXtV2.yaml). Each top-level key that is not `PARAMETERS` or a Jinja directive defines either the **root model** (using the reserved keys below) or a **named sublayer** (an arbitrary key whose value follows the same schema).

**`IMPORTS`** — Additional Python imports to inject at the top of the generated file. Accepts a dict mapping module names to lists of names to import, or an empty dict `{}` when no extra imports are needed.

```yaml
IMPORTS: {}
# or
IMPORTS:
  torch.nn: [Module, Linear]
  my_package.utils: null  # imports the entire module
```

**`INPUTS`** — Ordered list of tensor names that the generated `forward()` method accepts as keyword arguments. These names correspond to the first element of each `FLOW` entry and to the keys in the `inputs` dict passed at runtime.

```yaml
INPUTS: [image]
```

**`OUTPUTS`** — Ordered list of tensor names produced by the generated `forward()` method. When `STRUCTURED_OUTPUT` is `true`, these names become the keys of the returned dict; otherwise, they determine the order of the returned tuple.

```yaml
OUTPUTS: [cls]
# or, for a multi-output model:
OUTPUTS: [feat1, feat2, feat3, feat4]
```

**`STRUCTURED_OUTPUT`** — Controls the return type of the generated `forward()` method.

| Value             | Behavior                                                                 |
| ----------------- | ------------------------------------------------------------------------ |
| `true`            | Returns `{"cls": tensor, ...}` — a dict keyed by the names in `OUTPUTS`. |
| `false` (default) | Returns a plain tuple in the order of `OUTPUTS`.                         |

```yaml
STRUCTURED_OUTPUT: true
```

#### `FLOW` and `INFERENCE_FLOW`

`FLOW` is the training-time execution graph: an ordered list of `LayerBehavior` entries (see [`FLOW` Entry Format](#flow-entry-format) below) that describes how tensors are routed through the model's submodules.

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

#### `FLOW` Entry Format

Each entry in `FLOW` or `INFERENCE_FLOW` is a `LayerBehavior` — a list of 2 to 4 elements:

```
[INPUTS, OUTPUTS]
[INPUTS, OUTPUTS, NAME_or_LAYER]
[INPUTS, OUTPUTS, NAME, LAYER]
```

- **Position 0 — `INPUTS`**: Input variable name(s) for this step. A plain string (`image`, `feat1`) reads a named tensor from the current scope. Use `_` to pass the previous step's output forward. A nested list `[[a, b]]` collects tensors from multiple sources (e.g., for residual additions).
- **Position 1 — `OUTPUTS`**: Output variable name(s) produced by this step. Use `_` for intermediate values that need not be named. A dict `{alias: real_name}` renames the output in the current scope.
- **Position 2 — `NAME`**: (optional) A unique identifier for the generated submodule attribute. Auto-generated when omitted. Must be a valid Python identifier.
- **Position 2 or 3 — `LAYER`**: (optional) The layer definition — either a StructCast `ObjectPattern` (e.g., `[_obj_, {_addr_: torch.nn.ReLU}, _call_]`) or a `UserLayer` dict (see [`TYPE`, `PARAM`, and `CFG`](#type-param-and-cfg)).

```yaml
FLOW:
  - [image, {feature: feat4}, backbone, {TYPE: Backbone}]
  - [feature, _, [_obj_, {_addr_: torch.nn.AdaptiveAvgPool2d}, {_call_: {output_size: 1}}]]
  - [_, _, [_obj_, {_addr_: torch.nn.Flatten}, _call_]]
  - [_, cls, head, [_obj_, {_addr_: torch.nn.LazyLinear}, {_call_: {out_features: 1000}}]]
```

**`NAME`** — `NAME` appears in two contexts:

1. **As the third element of a `FLOW` entry** — sets the Python attribute name of the generated submodule (e.g., `"block0"`, `"head"`). Must be a valid Python identifier.
2. **As a key in a `BACKWARDS` entry** — sets the generated attribute name for that backward pass and its optimizer.

```yaml
# In FLOW:
- [feat1, feat1, "block0", {TYPE: Block, PARAM: {DEFAULT: {fout: 40}}}]

# In BACKWARDS:
BACKWARDS:
  - NAME: optimizer
    LOSS: ce_loss
    TRAINABLE_LAYERS: [model]
    OPTIMIZER: [_obj_, ...]
```

**`LAYER`** — The fourth (or third) element of a `FLOW` entry. Defines how the submodule for this step is constructed. Two forms are accepted:

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

- **`TYPE`** (`str`): Name of a sublayer defined as a top-level key in the same YAML file (e.g., `Backbone`, `Block`, `Stem`). The code generator expands it into a nested `nn.Module` subclass.
- **`PARAM`** (`PARAMETERS` dict): Template variable overrides passed when rendering the sublayer. Uses the same `DEFAULT` / `SHARED` / named-group structure as the top-level `PARAMETERS` block.
- **`CFG`** (file path): Path to an external YAML file that defines the sublayer. Allows sublayer reuse across multiple model templates. When `CFG` is set, `TYPE` selects the sublayer name within that file.

```yaml
# References Backbone sublayer defined in the same file, no parameter overrides
- [image, {feature: feat4}, backbone, {TYPE: Backbone}]

# References Block sublayer with per-instance parameter overrides
- [feat1, feat1, "block0", {TYPE: Block, PARAM: {DEFAULT: {fout: 40, drop_path: 0.0}}}]
```

---

### Backward Template Schema

The following keys appear in backward configuration files such as [`cfg/torch/backwards/ConvNeXtV2.yaml`](cfg/torch/backwards/ConvNeXtV2.yaml) and [`cfg/torch/backwards/CycleGAN.yaml`](cfg/torch/backwards/CycleGAN.yaml).

**`IMPORTS`** — Same format as in the model schema. Injects additional Python imports into the generated backward file.

```yaml
IMPORTS: {}
```

**`INPUTS` and `OUTPUTS`** — `INPUTS` lists the tensor names the generated backward class expects as keyword arguments during training and inference. `OUTPUTS` lists the tensor names produced by the backward flow. Both default to `[]`, which instructs the code generator to infer them automatically from the `BACKWARDS` entries' `FLOW` definitions.

```yaml
INPUTS: []                # auto-inferred from BACKWARDS[*].FLOW
OUTPUTS: [loss_G, loss_GAN, loss_cycle, loss_identity, loss_D_A, loss_D_B, fake_A, fake_B]
```

**`MIXED_PRECISION`** — Controls `torch.amp.GradScaler` for automatic mixed-precision training.

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

**`MIXED_PRECISION_TYPE`** — The dtype forwarded to `torch.autocast` when mixed precision is enabled. Accepts `"bfloat16"` or `"float16"`. Has no effect when `MIXED_PRECISION` is `false`.

```yaml
MIXED_PRECISION_TYPE: bfloat16
```

**`ACCUMULATE_GRADIENTS`** — The number of forward–backward steps to accumulate before calling the optimizer. Set to `null` to disable accumulation (optimizer steps every batch). When set to a positive integer `n`, `optimizer.step()` and `optimizer.zero_grad()` are called once every `n` batches.

```yaml
ACCUMULATE_GRADIENTS: null   # disabled
ACCUMULATE_GRADIENTS: 4      # accumulate over 4 steps
```

#### `BACKWARDS`

An ordered list of `BackwardBehavior` entries. Each entry defines one backward pass — i.e., one loss to differentiate, one optimizer to update, and its own execution graph. Multiple entries enable multi-optimizer training (e.g., GAN-style training where generator and discriminator optimizers are stepped independently).

During code generation, each entry's trainable layers are automatically set to training mode before its flow executes, and set back to eval mode after the optimizer step.

```yaml
# Single-optimizer example (classification)
BACKWARDS:
  - NAME: optimizer
    LOSS: ce_loss
    TRAINABLE_LAYERS: [model]
    OPTIMIZER: [_obj_, ...]
    CLIP: null
    EXTRA: {}
    FLOW:
      - [image, cls, model]
      - [{target: label, input: cls}, ce_loss, cross_entropy_loss, [_obj_, ...]]
    INFERENCE_FLOW:
      - [image, cls, model]
      - [{target: label, input: cls}, ce_loss, cross_entropy_loss]

# Multi-optimizer example (GAN)
BACKWARDS:
  - NAME: optimizer_G
    LOSS: loss_G
    TRAINABLE_LAYERS: [G_AB, G_BA]
    OPTIMIZER: [_obj_, ...]
    FLOW: [...]          # generator forward + loss computation
    INFERENCE_FLOW: [...] # inference-only flow
  - NAME: optimizer_D_A
    LOSS: loss_D_A
    TRAINABLE_LAYERS: [D_A]
    OPTIMIZER: [_obj_, ...]
    FLOW: [...]          # discriminator A forward + loss computation
  - NAME: optimizer_D_B
    LOSS: loss_D_B
    TRAINABLE_LAYERS: [D_B]
    OPTIMIZER: [_obj_, ...]
    FLOW: [...]          # discriminator B forward + loss computation
```

**`LOSSES` and `TRAINABLE_LAYERS`** — Both fields default to `[]`, which instructs the code generator to infer their values automatically from the `BACKWARDS` entries.

- **`LOSSES`** (`list[str]`): Explicit list of loss key names that the generated backward class tracks. Auto-inferred from `BACKWARDS[*].LOSS` when left as `[]`.
- **`TRAINABLE_LAYERS`** (`list[str]`): Explicit list of trainable model names the generated backward class expects as constructor arguments. Auto-inferred from `BACKWARDS[*].TRAINABLE_LAYERS` when left as `[]`.

```yaml
LOSSES: []           # auto-inferred
TRAINABLE_LAYERS: [] # auto-inferred
```

#### `BACKWARDS` Entry Keys

Each entry in `BACKWARDS` is a `BackwardBehavior` with the following fields:

- **`NAME`** (`str`): Optional identifier for this backward pass. Used as the generated attribute name for the optimizer. Must be a valid Python identifier.
- **`LOSS`** (`str`): The loss key (produced by the `FLOW`) that this backward pass differentiates.
- **`TRAINABLE_LAYERS`** (`list[str]`): Model names whose parameters this optimizer manages. Each value must match a model passed to the backward class constructor.
- **`FLOW`** (`list`): Training-time execution graph for this backward entry. Uses the same entry format as model `FLOW` (see [`FLOW` Entry Format](#flow-entry-format)), plus support for `"eval: ..."` expressions and inline layer instantiation via StructCast patterns.
- **`INFERENCE_FLOW`** (`list`): Optional inference-time execution graph. When absent, `FLOW` is used for inference as well.
- **`OPTIMIZER`** (StructCast pattern): A StructCast `ObjectPattern` that constructs the optimizer (and optionally its learning-rate scheduler). Commonly uses `structcast_model.torch.optimizers.create_with_scheduler` with `_bind_` to pass `optimizer_kwargs` and `scheduler_kwargs`.
- **`CLIP`** (StructCast pattern or `null`): Optional gradient-clipping callable. When non-null, the pattern is bound once and called before each optimizer step with the parameters identified by `TRAINABLE_LAYERS`. Set to `null` to disable gradient clipping.
- **`EXTRA`** (`dict`): Extra keyword arguments forwarded to the backward/optimizer logic. Default is `{}`.

```yaml
BACKWARDS:
  - NAME: optimizer
    LOSS: ce_loss
    TRAINABLE_LAYERS: [model]
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
    CLIP:
      - _obj_
      - _addr_: timm.utils.clip_grad.dispatch_clip_grad
      - _bind_: {value: 1.0, mode: norm, norm_type: 2.0}
    EXTRA: {}
    FLOW:
      - [image, cls, model]
      - [{target: label, input: cls}, ce_loss, cross_entropy_loss, [_obj_, _addr_: torch.nn.CrossEntropyLoss, _call_]]
    INFERENCE_FLOW:
      - [image, cls, model]
      - [{target: label, input: cls}, ce_loss, cross_entropy_loss]
```

---

## API Reference: `base_trainer.py`

[`src/structcast_model/base_trainer.py`](src/structcast_model/base_trainer.py) provides the framework-agnostic training loop, state management, and callback system. Concrete trainers such as `TorchTrainer` build on top of these abstractions.

### Utility functions

**`get_dataset(dataset)`** — Resolves a `DatasetLike` or a zero-argument callable into an actual iterable. This allows lazy dataset construction.

**`get_dataset_size(dataset)`** — Returns the number of batches. Uses `__len__` when available, otherwise iterates to count.

**`invoke_callback(callbacks, info, *args, **models)`** — Iterates over a callback list and calls each entry with `info` and keyword model arguments.

### Protocols

**`Forward`** — Called once per batch during training or validation. Accepts an `inputs` dictionary and keyword model arguments; returns a `dict[str, Any]` of named outputs and criteria.

**`Backward`** — Called once per training step. Receives the step index and criterion keyword arguments; returns `True` when the optimizer has stepped, `False` when gradients are being accumulated.

**`Callback` and `BestCallback`** — Lifecycle hooks called with `(info: BaseInfo, **models)`. `BestCallback` additionally receives `target: str` and `best: float` arguments.

### State and callbacks

**`BaseInfo`** — Dataclass holding mutable training state:

- `step` — total training steps taken
- `update` — optimizer update count
- `epoch` — current epoch number
- `history` — per-epoch log dictionaries
- `logs(epoch=None)` — returns the log dict for the current (or given) epoch

**`Callbacks`** — Dataclass holding callback lists for each lifecycle hook:

- `on_update` — after each optimizer update
- `on_training_begin` / `on_training_end`
- `on_training_step_begin` / `on_training_step_end`
- `on_validation_begin` / `on_validation_end`
- `on_validation_step_begin` / `on_validation_step_end`
- `on_epoch_begin` / `on_epoch_end`

When `add_global_callbacks=True` (the default), entries from `GLOBAL_CALLBACKS` are copied into each list at construction time.

**`GLOBAL_CALLBACKS`** — A shared `Callbacks[Any]` instance. Callbacks registered here are automatically picked up by every newly created trainer.

### Core classes

**`BaseTrainer`** — The main training loop driver. Inherits both `BaseInfo` and `Callbacks`.

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

**`BestCriterion`** — A callable that monitors a log key and fires `on_best` callbacks whenever a new best is found. Attach it to `on_epoch_end` or `on_validation_end`.

```python
checkpoint = BestCriterion(
    target="val_acc1",
    mode="max",
    on_best=[save_checkpoint],
)
trainer.on_epoch_end.append(checkpoint)
```

Fields: `target` (str), `mode` (`"min"` or `"max"`, default `"min"`), `on_best` (list of `BestCallback`).

---

## API Reference: `trainer.py`

[`src/structcast_model/torch/trainer.py`](src/structcast_model/torch/trainer.py) contains the PyTorch-specific runtime layer.

### Utility functions

**`create_torch_inputs(shape)`** — Creates dummy `float32` tensors from tuple, list, or dict shape descriptions. Used for model initialization and FLOPs inspection.

**`get_torch_device(device=None)`** — Returns the runtime device. Selects `cuda` when available and requested, otherwise falls back to `cpu`.

**`initial_model(model, shapes=None, compile_fn=None)`** — Walks a module or nested module structure, optionally builds dummy inputs, runs a forward pass, and applies a compile function to each module. Returns:

```python
(initialized_model, inputs, outputs)
```

**`get_autocast(mixed_precision_type, device)`** — Returns a context manager for automatic mixed precision:

- `contextlib.suppress` when AMP is disabled.
- A configured `torch.autocast(...)` partial when AMP is enabled.

### Tracking and orchestration

**`TorchTracker`** — Wraps `CriteriaTracker` instances for losses and metrics, resets them through global callbacks, and returns float-valued logs suitable for history storage and MLflow logging.

```python
tracker = TorchTracker.from_criteria(["ce_loss"], ["acc1", "acc5"])
logs = tracker(ce_loss=loss_tensor, acc1=acc1_tensor, acc5=acc5_tensor)
```

**`TorchTrainer`** — `TorchTrainer` extends the generic `BaseTrainer` with PyTorch-specific synchronization.

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

**`TimmDatasetWrapper`** — Holds validated dataset configuration and lazily calls `timm.data.create_dataset(...)`.

**`TimmDataLoaderWrapper`** — Builds a timm dataloader with support for:

- Prefetching
- Channels-last memory format conversion
- Mixup and cutmix data augmentation
- Train/validation-specific augmentation settings
- Distributed device initialization
- Optional `FlexSpec` output remapping

The dataset template at [`cfg/torch/others/default_timm.yaml`](cfg/torch/others/default_timm.yaml) formats into this wrapper.
