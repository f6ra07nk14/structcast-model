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
    - [Learner Template Schema](#learner-template-schema)
      - [`LEARNERS`](#learners)
      - [`LEARNERS` Entry Keys](#learners-entry-keys)
  - [API Reference: `base_trainer.py`](#api-reference-base_trainerpy)
    - [Utility functions](#utility-functions)
    - [Protocols](#protocols)
    - [Event protocols](#event-protocols)
    - [State and callbacks](#state-and-callbacks)
    - [Core classes](#core-classes)
  - [API Reference: `trainer.py`](#api-reference-trainerpy)
    - [Utility functions](#utility-functions-1)
    - [Tracking and orchestration](#tracking-and-orchestration)
  - [API Reference: `optimizers.py`](#api-reference-optimizerspy)
    - [Loggers](#loggers)
    - [State backends](#state-backends)
    - [timm integrations](#timm-integrations)
  - [API Reference: `flax/optimizers.py`](#api-reference-flaxoptimizerspy)
  - [API Reference: `flax/trainer.py`](#api-reference-flaxtrainerpy)
  - [API Reference: `flax/distributed.py`](#api-reference-flaxdistributedpy)

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
# From a torch learner template — ACCUMULATE_GRADIENTS is a torch-only key (see the learner schema).
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
| `false` (default) | Returns the bare value for a single output, else a tuple in `OUTPUTS` order. |

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
2. **As a key in a `LEARNERS` entry** — sets the generated attribute name for that entry's optimizer.

```yaml
# In FLOW:
- [feat1, feat1, "block0", {TYPE: Block, PARAM: {DEFAULT: {fout: 40}}}]

# In LEARNERS:
LEARNERS:
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

- **`UserLayer` dict** — references a sublayer defined in the same file (via `TYPE`), the root template of an external file (via `CFG`), or a sublayer defined inside an external file (via both):

  ```yaml
  {TYPE: Backbone}
  {TYPE: Block, PARAM: {DEFAULT: {fout: 40, drop_path: 0.0}}}
  {CFG: cfg/torch/models/my_model.yaml}
  {CFG: cfg/torch/models/my_model.yaml, TYPE: MySublayer}
  ```

#### `TYPE`, `PARAM`, and `CFG`

These three keys form the `UserLayer` dict that activates a sublayer. At least one of `TYPE` or `CFG` must be
present:

- **`TYPE`** (`str`): Name of a sublayer defined as a top-level key — resolved in the same YAML file, or inside the `CFG` file when `CFG` is also set (e.g., `Backbone`, `Block`, `Stem`). A dotted path (e.g., `Block.Norm`) selects a sublayer nested inside another sublayer. The code generator expands it into a nested `nn.Module` subclass.
- **`PARAM`** (`PARAMETERS` dict): Template variable overrides passed when rendering the sublayer. Uses the same `DEFAULT` / `SHARED` / named-group structure as the top-level `PARAMETERS` block.
- **`CFG`** (file path): Path to an external YAML file. Allows reuse across multiple model templates. On its own it embeds that file's root template as the sublayer; combined with `TYPE` it selects a sublayer defined inside that file instead.

```yaml
# References Backbone sublayer defined in the same file, no parameter overrides
- [image, {feature: feat4}, backbone, {TYPE: Backbone}]

# References Block sublayer with per-instance parameter overrides
- [feat1, feat1, "block0", {TYPE: Block, PARAM: {DEFAULT: {fout: 40, drop_path: 0.0}}}]

# Embeds the root template of an external file as the sublayer, no TYPE needed
- [image, feature, backbone, {CFG: cfg/torch/models/my_model.yaml}]
```

---

### Learner Template Schema

The following keys appear in learner configuration files such as [`cfg/torch/learners/ConvNeXtV2.yaml`](cfg/torch/learners/ConvNeXtV2.yaml) and [`cfg/torch/learners/CycleGAN.yaml`](cfg/torch/learners/CycleGAN.yaml). `scm torch create learner` turns them into a class implementing the `Learner` protocol.

The schema splits in two (see [`docs/adr/0012`](docs/adr/0012-framework-neutral-learner-schema-with-torch-extensions.md)): [`builders/schema.py`](src/structcast_model/builders/schema.py) owns the framework-neutral `UserDefinedLearner` and `LearnerBehavior`, and [`builders/torch.py`](src/structcast_model/builders/torch.py) adds the torch-only keys as `TorchUserDefinedLearner` and `TorchLearnerBehavior`, reached through the `TorchTemplateLearner` the torch builder validates against. `scm flax create learner` validates the same file against the neutral classes, which forbid unknown fields, so the keys marked *(torch only)* below are rejected there.

**`IMPORTS`** — Same format as in the model schema. Injects additional Python imports into the generated learner file.

```yaml
IMPORTS: {}
```

**`INPUTS` and `OUTPUTS`** — `INPUTS` lists the tensor names the generated learner expects as keyword arguments during training and inference. `OUTPUTS` lists the criterion names produced by the flow; they become the learner's `outputs` attribute, which `scm torch train` reads to build the tracker. Both default to `[]`, which instructs the code generator to infer them automatically from the `LEARNERS` entries' `FLOW` definitions.

```yaml
INPUTS: []                # auto-inferred from LEARNERS[*].FLOW
OUTPUTS: [loss_G, loss_GAN, loss_cycle, loss_identity, loss_D_A, loss_D_B, fake_A, fake_B]
```

**`MIXED_PRECISION`** *(torch only, `TorchUserDefinedLearner`)* — Controls gradient scaling, which only counteracts float16 underflow. The generated learner constructs a `torch.amp.GradScaler` directly, on the training `device`.

| Value             | Behavior                                                                                                                                     |
| ----------------- | -------------------------------------------------------------------------------------------------------------------------------------------- |
| `false` (default) | No `GradScaler` is created. Autocast still runs when `MIXED_PRECISION_TYPE` is set — this is the bfloat16 path, which needs no scaler.        |
| `true`            | Gradient scaling enabled with default `GradScaler` settings; requires `MIXED_PRECISION_TYPE: float16`.                                        |
| `dict`            | Gradient scaling enabled; the dict is forwarded as keyword arguments to `torch.amp.GradScaler`. Requires `MIXED_PRECISION_TYPE: float16`.    |

```yaml
MIXED_PRECISION:
  init_scale: "eval: 2.0**16"
  growth_factor: 2.0
  backoff_factor: 0.5
  growth_interval: 2000
  enabled: True
```

**`MIXED_PRECISION_TYPE`** *(torch only, `TorchUserDefinedLearner`)* — The dtype forwarded to `torch.autocast` when mixed precision is enabled. Accepts `"bfloat16"` or `"float16"`. It is valid on its own — `MIXED_PRECISION: false` with `MIXED_PRECISION_TYPE: bfloat16` is autocast without a scaler. Enabling `MIXED_PRECISION` with anything other than `float16` raises a `SpecError` at build time, because gradient scaling only applies to float16.

```yaml
MIXED_PRECISION_TYPE: bfloat16
```

**`ACCUMULATE_GRADIENTS`** *(torch only, `TorchUserDefinedLearner`)* — The number of forward–backward steps to accumulate before calling the optimizer. Set to `null` to disable accumulation (optimizer steps every batch). When set to a positive integer `n`, each loss is divided by `n` before `backward()`, `optimizer.step()` and `optimizer.zero_grad()` are called once every `n` batches, and the generated `update(step)` returns `True` only on those steps — which is what makes the trainer fire `on_update` once per update rather than once per step.

```yaml
ACCUMULATE_GRADIENTS: null   # disabled
ACCUMULATE_GRADIENTS: 4      # accumulate over 4 steps
```

The Keras and Flax schemas carry no such field: each backend declares the accumulation window through the mechanism its framework already owns, inside the `OPTIMIZER` pattern, and the generated `update()` gates on it — so the trainer's update counter tracks real optimizer applies (see [`docs/adr/0017`](docs/adr/0017-accumulation-gating-follows-each-backends-native-mechanism.md)).

In Keras the window is the optimizer's own `gradient_accumulation_steps` keyword argument. The generated `update()` reads the optimizer's step counter to predict whether the step about to run lands an apply, which keeps the gate in phase even when a float16 `LossScaleOptimizer` skips a step. All optimizers of a learner must share the same window — the generated `__init__` raises a `ValueError` otherwise.

```yaml
OPTIMIZER:
  - _obj_
  - _addr_: keras.optimizers.SGD
  - _call_:
      learning_rate: 0.1
      gradient_accumulation_steps: 4
```

In Flax the window is an [`optax.MultiSteps`](https://optax.readthedocs.io/) wrapping the entry's `tx`. The builder statically parses the wrapper and bakes its window into `update()` as a compile-time constant, so `every_k_schedule` must be an int literal (a callable is a `SpecError`), `should_skip_update_fn` is rejected (`SpecError`), and every entry of a learner must declare the same window — an entry without `MultiSteps` counts as 1 (`SpecError` at build time). `MultiSteps` accumulates in float32 by default, which doubles accumulator memory against bfloat16 parameters; pass `accumulator_dtype` when that matters.

```yaml
OPTIMIZER:
  - _obj_
  - _addr_: flax.nnx.Optimizer
  - _bind_:
      tx:
        - _obj_
        - _addr_: optax.MultiSteps
        - _call_:
            opt: [_obj_, {_addr_: optax.sgd}, {_call_: {learning_rate: 0.1}}]
            every_k_schedule: 4
```

#### `LEARNERS`

An ordered list of `TorchLearnerBehavior` entries (`LearnerBehavior` for the other frameworks). Each entry defines one loss to differentiate, one optimizer to update, and its own execution graph. Multiple entries enable multi-optimizer training (e.g., GAN-style training where generator and discriminator optimizers are stepped independently).

During code generation, a mode preset runs before each entry's flow segment: the entry's trainable layers are set to training mode and every other model to eval mode. Nothing restores modes after the optimizer step — a later entry's preset (or nothing, for the last entry) is what changes them next.

```yaml
# Single-optimizer example (classification)
LEARNERS:
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
LEARNERS:
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

**`LOSSES` and `TRAINABLE_LAYERS`** — Both fields default to `[]`, which instructs the code generator to infer their values automatically from the `LEARNERS` entries.

- **`LOSSES`** (`list[str]`): Explicit list of loss key names that the generated learner tracks. Auto-inferred from `LEARNERS[*].LOSS` when left as `[]`.
- **`TRAINABLE_LAYERS`** (`list[str]`): Explicit list of trainable model names the generated learner expects as constructor arguments. Auto-inferred from `LEARNERS[*].TRAINABLE_LAYERS` when left as `[]`.

```yaml
LOSSES: []           # auto-inferred
TRAINABLE_LAYERS: [] # auto-inferred
```

#### `LEARNERS` Entry Keys

Each entry in `LEARNERS` is a `TorchLearnerBehavior` with the following fields:

- **`NAME`** (`str`): Optional identifier for this entry. Used as the generated attribute name for the optimizer, and as its key in the learner's `optimizers` property. Must be a valid Python identifier.
- **`LOSS`** (`str`): The loss key (produced by the `FLOW`) that this entry differentiates.
- **`TRAINABLE_LAYERS`** (`list[str]`): Model names whose parameters this optimizer manages. Each value must match a model passed to the learner constructor.
- **`FLOW`** (`list`): Training-time execution graph for this entry. Uses the same entry format as model `FLOW` (see [`FLOW` Entry Format](#flow-entry-format)), plus support for `"eval: ..."` expressions and inline layer instantiation via StructCast patterns.
- **`INFERENCE_FLOW`** (`list`): Optional inference-time execution graph. When absent, `FLOW` is used for inference as well.
- **`OPTIMIZER`** (StructCast pattern): A StructCast `ObjectPattern` that constructs the optimizer, called with the named parameters of `TRAINABLE_LAYERS`. It may address `structcast_model.torch.optimizers.create_opt` directly, or an optimizer + scheduler composition loaded from a file with `_file_` — see [`examples/torch/optimizers.py`](examples/torch/optimizers.py). Such a composition implements the event protocols itself, so the trainer steps its schedule when it scans the learner's `optimizers`.
- **`CLIP`** (StructCast pattern or `null`) *(torch only, `TorchLearnerBehavior`)*: Optional gradient-clipping callable. When non-null, the pattern is bound once and called before each optimizer step with the parameters identified by `TRAINABLE_LAYERS`. Set to `null` to disable gradient clipping.
- **`EXTRA`** (`dict`): Extra keyword arguments forwarded to the optimizer or the update logic in general. Default is `{}`.

```yaml
LEARNERS:
  - NAME: optimizer
    LOSS: ce_loss
    TRAINABLE_LAYERS: [model]
    OPTIMIZER:
      - _obj_
      - _addr_: AdamWWithCosine
        _file_: examples/torch/optimizers.py
      - _bind_:
          optimizer_kwargs:
            opt: adamw
            lr: 4.0e-3
            weight_decay: 0.001
          scheduler_kwargs:
            sched: cosine
            num_epochs: 300
            criterion: ce_loss
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

[`src/structcast_model/base_trainer.py`](src/structcast_model/base_trainer.py) provides the framework-agnostic training loop, state management, and the protocol-routed callback system. Concrete trainers such as `TorchTrainer` build on top of these abstractions.

### Utility functions

**`get_dataset(dataset)`** — Resolves a `DatasetLike` or a zero-argument callable into an actual iterable. This allows lazy dataset construction.

**`get_dataset_size(dataset)`** — Returns the number of batches. Uses `__len__` when available, otherwise iterates to count.

### Protocols

**`Learner`** — The object that owns the models and defines how they learn. Members:

- `models` (property) — `dict[str, ModelT]` of the models to train, where `ModelT` is the model type the trainer is specialized to (`torch.nn.Module` for `TorchTrainer`); the trainer exposes them to every event as `info.models`.
- `optimizer_models` (property) — `dict[str, list[str]]` naming the models each optimizer updates (optimizer name -> model names); checkpointing uses it to pair sharded optimizer state with its modules, and an empty mapping means the pairing is not declared.
- `update(step) -> bool` — whether the given training step applied the optimizers. `False` means gradients are still accumulating.
- `training_step(**inputs) -> dict[str, Any]` — runs one training batch and returns its criteria.
- `inference_step(**inputs) -> dict[str, Any]` — runs one validation batch and returns its criteria.

Three required members are also read elsewhere in the toolkit: `optimizers` (a mapping, additionally scanned for event protocols by the trainer), `optimizer_models` (read whenever checkpointing saves or restores optimizer state, as described above), and `learning_rates` (shown by `ProgressBar` / `Printer` and logged by the loggers). Optional members: `grad_scalers` and `param_group_names` (saved and logged by the CLI), and `weight_decays` (per-group decay metrics merged into the logged epoch metrics; generated learners flatten it from `create_opt`'s parameter groups via `get_decays`).

**`DataProvider`** — Supplies the datasets of a whole run and their step counts: a `training_dataset` property, a `validation_dataset` property that may be `None` to skip validation, and `steps_per_epoch` / `validation_steps` properties (`validation_steps` is `0` without a validation dataset). Each dataset may be a dataset or a zero-argument callable returning one, and the dataset properties must return the same object on every read: the trainer reads them for the event-protocol scan and again in `fit()`. Both datasets are scanned against every event protocol, so a validation dataset implementing training-phase hooks receives those events too — guard on the split inside the hook, as `TimmDataLoaderWrapper` does.

**`OnBest`** — Protocol for the participants of `BestCriterion.on_best`, mirroring how the trainer routes events: an object with an `on_best(info: BaseInfo, best: BestCriterion)` method.

### Event protocols

`EVENTS` lists the eleven lifecycle events, and `EVENT_PROTOCOLS` maps each event name to the `runtime_checkable` protocol an object must implement to receive it:

| Event                                              | Protocol                                        | Fired                                        |
| -------------------------------------------------- | ----------------------------------------------- | -------------------------------------------- |
| `on_update`                                        | `OnUpdate`                                      | after a step in which the learner updated     |
| `on_training_begin` / `on_training_end`            | `OnTrainingBegin` / `OnTrainingEnd`             | around the training pass of an epoch          |
| `on_training_step_begin` / `on_training_step_end`  | `OnTrainingStepBegin` / `OnTrainingStepEnd`     | around each training step                     |
| `on_validation_begin` / `on_validation_end`        | `OnValidationBegin` / `OnValidationEnd`         | around the validation pass                    |
| `on_validation_step_begin` / `on_validation_step_end` | `OnValidationStepBegin` / `OnValidationStepEnd` | around each validation step                |
| `on_epoch_begin` / `on_epoch_end`                  | `OnEpochBegin` / `OnEpochEnd`                   | around a whole epoch, validation included     |

Every handler has the signature `(info: BaseInfo) -> None`, where `info` is the trainer itself, so the models are read from `info.models`. An object joins an event by defining the matching method — there is no registration call and no global registry. Because the protocols are `runtime_checkable`, only the method name is checked, not its signature.

### State and callbacks

**`BaseInfo`** — Dataclass holding mutable training state:

- `step` — total training steps taken
- `update` — optimizer update count
- `epoch` — current epoch number
- `history` — per-epoch log dictionaries
- `logs(epoch=None)` — returns the log dict for the current (or given) epoch
- `models` (property) — the models by name; empty on a bare info, delegated to the learner by `BaseTrainer`

**`SimpleDataProvider`** — Dataclass implementing `DataProvider` over an already-built training dataset and an optional validation dataset.

```python
provider = SimpleDataProvider(training_dataset=train_loader, validation_dataset=valid_loader)
```

**`ProgressBar`** — Callback showing training and validation progress on a `tqdm` bar (`tqdm` must be installed). Constructor (keyword-only): `ProgressBar(steps_per_epoch=..., validation_steps=0, training_criteria=(), validation_criteria=())`, where the two criteria sequences name the log keys shown next to the bar. It writes the criteria of each finished epoch above the bar.

**`Printer`** — Callback printing the criteria of each finished epoch, for environments without a terminal. Both callbacks prepend the learner's `learning_rates` when it reports them.

### Core classes

**`BaseTrainer`** — The main training loop driver, and itself the `BaseInfo` passed to every event.

Required fields: `learner` (`Learner`), `tracker` (callable returning `dict[str, float]`, called once per step with the criteria of that step), `data` (`DataProvider`).

Optional fields: `callbacks` (sequence of participants, default `()`), `training_prefix` (default `""`), `validation_prefix` (default `"val_"`).

On first use (the first dispatched event) the trainer scans, in this order, the learner, the values of the learner's `optimizers` mapping, the tracker, the data provider, its `training_dataset` and `validation_dataset`, and then the `callbacks` in the order given. Each object is routed into every event whose protocol it implements, and is never registered twice for the same event. Because the scan is deferred, callbacks appended to the given sequence after construction — the CLI builds its display callbacks from the constructed trainer's prefixes — still take part.

Key methods:

- `describe()` — returns `dict[event, list[str]]` of the registered display names, omitting empty events
- `train(dataset)` — runs one training pass, returns the final step logs
- `evaluate(dataset)` — runs one validation pass, returns the final step logs
- `fit(epochs, start_epoch=1, validation_frequency=1)` — runs the full loop over the data provider's datasets and returns the complete history dict
- `update_models(inputs)` — performs one training step, returning `(updated, criteria)`; gradient synchronization is gated inside the generated training step, not here
- `sync()` — optional synchronization hook, no-op by default (overridden in `TorchTrainer`)

```python
trainer = BaseTrainer(
    learner=my_learner,
    tracker=my_tracker,
    data=SimpleDataProvider(training_dataset=train_loader, validation_dataset=valid_loader),
    callbacks=[Printer()],
)
history = trainer.fit(epochs=10)
```

**`BestCriterion`** — Callback monitoring one criterion for its best value. It implements `on_epoch_end`, so passing it in `callbacks` is all that is needed.

```python
class SaveCheckpoint:
    def on_best(self, info: BaseInfo, best: BestCriterion) -> None: ...  # log or save by best.value / best.step


checkpoint = BestCriterion(target="val_acc1", mode="max", callbacks=[SaveCheckpoint()])
trainer = TorchTrainer(device="cuda", learner=learner, tracker=tracker, data=data, callbacks=[checkpoint])
```

Fields: `target` (str), `mode` (`"min"` or `"max"`, default `"min"`), `callbacks` (list of `OnBest` participants, notified after every epoch in which the target appeared, whether or not it improved; named `callbacks` so the field cannot shadow the protocol method in an isinstance check). Properties: `value` (the best value so far) and `step` (the step at which it was reached).

---

## API Reference: `trainer.py`

[`src/structcast_model/torch/trainer.py`](src/structcast_model/torch/trainer.py) contains the PyTorch-specific runtime layer. The `DistributedStrategy` implementations it saves states through live in [`src/structcast_model/torch/distributed.py`](src/structcast_model/torch/distributed.py) — `SingleDeviceStrategy`, `DistributedDataParallelStrategy`, and `FullyShardedDataParallelStrategy` (FSDP2, `torch>=2.6`) — together with `sync_gate(model, armed)`, the per-call gradient-synchronization gate the generated learners use.

Both multi-rank strategies convert every `BatchNorm` layer to `SyncBatchNorm` inside `wrap()`, before DDP construction or FSDP2 sharding; `SingleDeviceStrategy` never converts, and the conversion is skipped on CPU devices, where `SyncBatchNorm`'s training forward rejects the input. It runs through timm's `convert_sync_batchnorm`, so a fused `BatchNormAct2d` becomes a `SyncBatchNormAct` — a `torch.nn.SyncBatchNorm` subclass — keeping the activation that a plain replacement would drop. The conversion is idempotent — a layer that already is a `torch.nn.SyncBatchNorm` is left untouched, `process_group` included — so pre-converted models keep working; calling `torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)` in the model definition is no longer needed. Bind `sync_batchnorm: false` on the strategy pattern to opt out — there is no CLI flag. Limitations: a non-timm third-party `_BatchNorm` subclass is flattened to a plain `SyncBatchNorm` (opt out and convert it yourself), `torch.compile` graph-breaks on `SyncBatchNorm`, and every replaced layer is a new object, so hooks registered on it are dropped — as is an in-place `--compile` of a model root or `shard_modules` match that is itself a `BatchNorm` layer.

### Utility functions

**`create_torch_inputs(shape, *, batch_size=1)`** — Creates dummy tensors from a tensor specification, or from a dict or list nesting more of them, using each specification's dtype and initializer. Used for model initialization and FLOPs inspection.

**`get_torch_device(device=None)`** — Returns the runtime device. Selects `cuda` when available and requested, otherwise falls back to `cpu`. `get_torch_device_type(device=None)` strips the rank suffix.

**`resolve_input_shapes(model, shapes=None)`** — Returns the input shapes to initialize with: explicit *shapes* win, otherwise the model's own `INPUT_SHAPES` attribute (or the merged attributes of a model mapping), otherwise `None`.

**`initial_model(model, shapes=None)`** — Walks a module or nested module structure, builds dummy inputs from the resolved shapes, and runs one forward pass to materialize lazy layers. Returns `(inputs, outputs)`; when no shapes are available, `inputs` is `None` and no forward pass is run.

**`autocast_inputs(inputs, device_type)`** — Returns the autocast context matching the element type of the dummy inputs, or a null context when they contain no low-precision floating point tensor.

**`initial_distributed_env(device=None, dist_backend=None, dist_url=None, *, return_dict=True)`** — Initializes the distributed environment and reports the resolved device, ranks, world size, and whether the run is distributed.

### Tracking and orchestration

**`TorchTracker`** — Wraps a `CriteriaTracker`, averaging the criteria of the current pass and returning float-valued logs. It implements `on_training_begin` and `on_validation_begin`, which reset the averages, so training and validation values never mix. In a distributed run it reduces each value across ranks with `all_reduce`.

```python
tracker = TorchTracker.from_criteria(["ce_loss", "acc1", "acc5"])
logs = tracker(ce_loss=loss_tensor, acc1=acc1_tensor, acc5=acc5_tensor)
```

**`TorchTrainer`** — Extends `BaseTrainer` with a `device` field and CUDA synchronization. It adds no gradient-synchronization context: generated learners gate synchronization per model call with `sync_gate`, so the trainer runs the step as-is.

```python
data = SimpleDataProvider(training_dataset=train_loader, validation_dataset=valid_loader)
trainer = TorchTrainer(
    device="cuda",
    learner=learner,
    tracker=tracker,
    data=data,
    callbacks=[
        ProgressBar(steps_per_epoch=data.steps_per_epoch, validation_steps=data.validation_steps),
        TorchBestCriterion(target="val_acc1", mode="max"),
    ],
)
history = trainer.fit(epochs=5)
```

**`TorchBestCriterion`** — `BestCriterion` specialized to `torch.nn.Module` models. `TorchBestCriterion.from_criteria(higher_criteria, lower_criteria, save_criteria, logger, strategy)` builds one monitor per criterion — `"max"` mode for the higher list, `"min"` for the lower — each logging its best value through *logger* and, for criteria named in *save_criteria*, saving the model states that reached it, produced through *strategy*. *logger* is a `NullLogger` on the ranks that write nothing, while the state dicts are still produced on every rank; the CLI appends the returned monitors to its callbacks.

**`TrainingStateSaver`** — Callback saving the full training state of each finished epoch through a logger, so a run can be resumed from it with `--resume`: the model and optimizer state dicts produced by the strategy through `torch.distributed.checkpoint.state_dict` (wrapper-free keys for raw, compiled, DDP, and FSDP2 models alike), the learner's gradient scaler state dicts, and the epoch/step/update counters, written as the `training_state` artifact. Producing the states is a collective that runs on every rank; only the ranks holding a logger write them.

```python
saver = TrainingStateSaver(logger=logger, strategy=strategy)
trainer = TorchTrainer(device="cuda", learner=learner, tracker=tracker, data=data, callbacks=[logger, saver])
```

The models and the learner of a CLI run are assembled inline by `scm torch train`, not by a factory class: the models are instantiated on the training device, initialized with dummy inputs, given their initializers on the main rank, broadcast via `sync_initial_weights`, compiled and wrapped by the strategy, and handed to the learner by name.

---

## API Reference: `optimizers.py`

**`create_opt(params, *, opt, layer_decay=None, layer_group_regexes=None, weight_decay=0.0, weight_decay_regexes=None, no_weight_decay_regexes=None, **kwargs)`** — Creates an optimizer over regex-grouped named parameters. Grouping runs first and is engine-agnostic; the engine is then chosen from *opt* across three paths: a callable is instantiated directly, an explicit `torch.optim.X` (or `torch.X`) name instantiates that class natively, and every bare name goes to `timm.optim.create_optimizer_v2` with timm's defaults. Layer decay emits an `lr_scale` per group: the callable and native engines bake it into the learning rate immediately, while the timm engine keeps it for its schedulers.

**`get_decays(optimizers)`** — Flattens the per-group `weight_decay` and `lr_scale` of every optimizer in the mapping into loggable metrics, keyed `{optimizer}_group{index}_weight_decay` / `{optimizer}_group{index}_lr_scale`. Generated learners expose the result as `weight_decays`, which the loggers merge into the epoch metrics.

**`set_lr_scale(optimizer, delete_lr_scale=False)`** — Bakes the `lr_scale` of every parameter group into its learning rate; groups without the key are untouched. Pass `delete_lr_scale=True` to drop the key so a later call cannot apply the same scale twice.

### Loggers

**`Logger`** (`structcast_model.loggers.base`) — The runtime-checkable protocol both backends implement: `log_params`, `log_dict`, `log_artifact`, `log_metric`, `log_metrics`, `log_state_dict`, `on_epoch_end`, and the `__enter__` / `__exit__` pair that owns the run.

**`MLflowLogger`** (`structcast_model.loggers.mlflow`) and **`WandbLogger`** (`structcast_model.loggers.wandb`) — Record a run to MLflow or to Weights & Biases through that protocol. Each is a context manager owning the run: entering it starts the run, leaving it ends the run. Both also implement `on_epoch_end`, which logs the criteria of the finished epoch together with the learner's learning rates and its optional `weight_decays` — so passing a logger in `callbacks` is enough to get per-epoch metrics, including weight/layer-decay dynamics.

```python
from structcast_model.loggers.mlflow import MLflowLogger

with MLflowLogger(experiment="my-experiment") as logger:
    trainer = TorchTrainer(device="cuda", learner=learner, tracker=tracker, data=data, callbacks=[logger])
    logger.log_params({"epochs": 5})
    trainer.fit(epochs=5)
```

`MLflowLogger` needs the `mlflow` extra, `WandbLogger` the `wandb` extra; each module imports its backend at import time and raises a descriptive `ImportError` from the constructor when it is missing.

### State backends

**`StateBackend`** (`structcast_model.loggers.state_backends`) — The protocol deciding what a saved training state looks like on disk: a `suffix`, `save(states, directory, name)` returning the written path, and `load(path)`. Both loggers take one as a `state_backend` field, defaulting to `TorchStateBackend`; `scm flax train` passes a `FlaxStateBackend` instead. A backend returns state in host memory only — numpy arrays or CPU tensors, never device-resident state — because placing it is the distributed strategy's job, which is what keeps a checkpoint independent of the topology that wrote it ([`docs/adr/0015`](docs/adr/0015-logger-state-backends-and-single-file-archives.md)).

**`TorchStateBackend`** — One `torch.save` pickle, suffix `.pt`, the format every torch run has used. `load` reads it with `map_location="cpu"` and `weights_only=True`, because the reference is user input and an unpickled checkpoint executes code.

**`FlaxStateBackend`** — An [orbax](https://orbax.readthedocs.io/) composite checkpoint packed into one gzipped tar, suffix `.tar.gz`, because orbax writes a directory while every transport around the loggers carries a file. Each top-level entry becomes one orbax item: array trees through the standard handler, JSON-serializable entries (`meta`, and the always-empty `grad_scalers`) as plain JSON. `load` extracts with the stdlib `filter="data"` guard — and refuses outright on an interpreter older than Python 3.11.4, which has no extraction filters — then restores without naming a target or a sharding, so a state saved on four devices comes back on any topology.

`MLflowLogger.log_state_dict` writes the artifact as `<name><suffix>`, a single file, where `mlflow.pytorch.log_state_dict` used to write a directory. `fetch_training_state` still reads that older layout: a downloaded artifact directory is searched for the active backend's suffix first, then for a legacy `*.pth`, which is read as a torch pickle whatever the logger's backend is.

### timm integrations

These are example code in [`examples/torch/data.py`](examples/torch/data.py), not package API: the CLI knows nothing about timm and loads them from a configuration by file path (`_addr_` plus `_file_`), the same way the optimizer compositions are loaded.

**`TimmDatasetWrapper`** — Holds validated dataset configuration and lazily calls `timm.data.create_dataset(...)`.

**`TimmDataLoaderWrapper`** — Builds a timm dataloader with support for:

- Prefetching
- Channels-last memory format conversion
- Mixup and cutmix data augmentation
- Train/validation-specific augmentation settings
- Distributed device initialization
- Optional `FlexSpec` output remapping

It implements `on_epoch_begin` and `on_training_begin`, which forward the new epoch to its dataset or `DistributedSampler` (`set_epoch`) and turn mixup off once `mixup_off_epoch` is reached. The trainer scans the provider datasets for event protocols — on every rank — so these hooks run without any further wiring.

The dataset template at [`cfg/torch/others/default_timm.yaml`](cfg/torch/others/default_timm.yaml) formats into this wrapper.

**`TimmDataProvider`** — `DataProvider` over a `training` wrapper and an optional `validation` dataset. The trainer scans both datasets directly, and the wrapper hooks no-op on validation splits, so the provider forwards nothing. The CLI always composes a `SimpleDataProvider`; use this one when wiring a trainer programmatically.

```python
provider = TimmDataProvider(training=training_wrapper, validation=validation_wrapper)
```

---

## API Reference: `flax/optimizers.py`

[`src/structcast_model/flax/optimizers.py`](src/structcast_model/flax/optimizers.py) holds the optimizer helpers a Flax learner template and its generated class use. All three are re-exported from `structcast_model.flax`.

**`get_learning_rate(optimizer)`** — Returns the learning rate an `nnx.Optimizer`'s state currently reports, as a float32 scalar. Optax stores no rate of its own — a constant lives in the update closure, and a schedule leaves only its step count behind — so the rate is readable only when the transformation was built through [`optax.inject_hyperparams`](https://optax.readthedocs.io/en/latest/api/utilities.html#optax.inject_hyperparams), which is why `FlaxLearnerBuilder` wraps the factory carrying `learning_rate` in it ([`docs/adr/0013`](docs/adr/0013-flax-optimizers-are-dsl-built-nnx-optimizers.md)). The walk is a pure pytree traversal, so calling it inside a traced training step compiles to a reference to the state array rather than to a host read. The result is NaN when the chain injects no rate at all, or several of them, since neither case names a single rate to report.

**`unwrap_variables(tree)`** — Returns the pytree with every `flax.nnx.Variable` leaf replaced by the value it holds, so a state can be filtered or serialized by value. It is `nnx.as_pure` written out by hand: the supported flax floor, 0.12.6, still calls that function `nnx.pure`, and one walk here keeps a single code path across the whole supported range.

**`no_weight_decay_mask(*regexes)`** — Returns the mask callable `optax.adamw(mask=...)` and `optax.masked` consume: it maps a parameter tree to a same-structure tree of booleans, `False` where the leaf's dotted path (`"encoder.bias"`) matches any of the regexes. The match is a search, not an anchor, so a plain `"bias"` exempts every bias in the tree.

```python
no_weight_decay_mask(r"\.bias$")({"layer": {"kernel": 1.0, "bias": 2.0}})
# {'layer': {'bias': False, 'kernel': True}}
```

---

## API Reference: `flax/trainer.py`

[`src/structcast_model/flax/trainer.py`](src/structcast_model/flax/trainer.py) is the Flax runtime layer: the dummy-input helpers `scm flax time` uses, plus the tracker, the trainer, and the checkpointing callbacks of a `scm flax train` run. The strategy those callbacks produce their states through is [`FlaxDistributedStrategy`](#api-reference-flaxdistributedpy).

**`create_jax_inputs(shape, *, batch_size=1)`** — Creates dummy JAX arrays from a tensor specification, or from a dict or list nesting more of them, using each specification's dtype and initializer. A floating point specification defaults to a uniform random initializer, an integer one to `jax.numpy.zeros`.

**`get_jax_devices()`** and **`get_jax_device(device=None)`** (in [`flax/utils.py`](src/structcast_model/flax/utils.py)) — The available devices as an ordered `{"cpu:0": Device}` mapping, and one looked up by that name; `None` returns the first. An unknown name raises with the available ones listed.

**`resolve_input_shapes(model, shapes=None)`** — The shared helper from `structcast_model.utils.base`, re-exported here: explicit *shapes* win, otherwise the model's own `INPUT_SHAPES` attribute (merged across a model mapping), otherwise `None`.

**`ShardedDataset(dataset, strategy)`** — Frozen dataclass wrapping a dataset -- an iterable of batches, or a callable returning one -- so that every batch it yields is placed across *strategy*'s mesh as it is read; `len()` counts an epoch through the wrapped dataset and places nothing. The wrapped dataset's event methods are copied onto the instance in `__post_init__` rather than forwarded from `__getattr__`, because the trainer picks an event's participants with `isinstance` against a runtime-checkable protocol, and that check looks attributes up statically. `scm flax train` builds one per split and hands both to its data provider.

**`FlaxTracker`** — Running mean of the criteria of one training or validation split. It sums on device as plain JAX arrays and returns Python floats — the contract `BaseTrainer.tracker`, the epoch history, the `BestCriterion` comparison, and `log_metric` all consume — reading the host once per step with a single `jax.device_get`. It implements `on_training_begin` and `on_validation_begin`, which reset the sums, so training and validation values never mix. Unlike `TorchTracker` there is no all-reduce: JAX is single-controller, so a criterion computed from a sharded batch is already the global value.

```python
tracker = FlaxTracker.from_criteria(["ce_loss", "acc1"])
logs = tracker(ce_loss=loss, acc1=acc1)
```

**`FlaxTrainer`** — `BaseTrainer` specialized to `nnx.Module` models, with nothing added: `sync()` stays the inherited no-op, because `FlaxTracker.logs` already waits for each step's program, and the devices a run uses are the strategy's mesh rather than a trainer field.

**`FlaxBestCriterion`** — `BestCriterion` specialized to `nnx.Module` models, the twin of `TorchBestCriterion`, duplicated rather than shared because importing either module imports its framework. `FlaxBestCriterion.from_criteria(higher_criteria, lower_criteria, save_criteria, logger, strategy)` builds one monitor per criterion — `"max"` mode for the higher list, `"min"` for the lower — each logging its best value through *logger* and, for criteria named in *save_criteria*, saving the model states that reached it as a `best_<criterion>` artifact produced through *strategy*.

**`FlaxTrainingStateSaver`** — Callback saving the training state of each finished epoch through a logger as the `training_state` artifact, so a run can be resumed from it with `--resume`: the model and optimizer states the strategy produces, an always-empty `grad_scalers` slot so both frameworks resume from the same payload shape, and a `meta` mapping of the epoch, step, and update counters plus whatever `extra_meta` the caller adds — the CLI adds the seed, a configuration hash, and the optimizer hashes.

```python
saver = FlaxTrainingStateSaver(logger=logger, strategy=strategy, extra_meta={"seed": 42})
trainer = FlaxTrainer(learner=learner, tracker=tracker, data=data, callbacks=[logger, saver])
```

**`restore_training_state(*, resume, strategy, models, learner, start_epoch, logger, optimizer_hashes=None, config_hash=None, is_main=True)`** — Fetches the state through *logger*, loads it into the live models and optimizers through *strategy*, and returns the epoch to continue at: the saved one plus one, which overrides *start_epoch* with a logged message ([`docs/adr/0005`](docs/adr/0005-checkpoints-through-dcp-state-dict-and-epoch-boundary-resume.md)). Optax rebuilds its transformation from configuration and the restore cannot see it, so a changed optimizer or a changed configuration warns rather than refuses — extending a schedule or lowering the rate of a fine-tune is legitimate.

---

## API Reference: `flax/distributed.py`

[`src/structcast_model/flax/distributed.py`](src/structcast_model/flax/distributed.py) holds the single strategy class a Flax run trains through. JAX expresses single-device, data-parallel, and fully-sharded execution with the same mechanism — a device mesh plus a `PartitionSpec` per array — so there is one class here instead of the three torch has, and what distinguishes the modes is a preset naming the mesh to build and the rules deciding each parameter's spec ([`docs/adr/0014`](docs/adr/0014-flax-strategies-are-spec-presets-on-an-explicit-mesh.md)).

**`FlaxDistributedStrategy`** — Satisfies the torch `DistributedStrategy` protocol structurally, so the trainer and the checkpointing callbacks treat both backends alike. Constructing it activates its mesh process-wide (`jax.set_mesh` takes effect at `__init__`), which is what makes models built afterwards land on it — so `scm flax train` builds it before anything else.

Fields: `preset` (`"single"`, `"dp"`, or `"fsdp"`, default `"single"`), `device` (the device the `single` preset runs on, e.g. `"cpu:0"`; the first available one by default), `devices` (how many devices `dp` and `fsdp` span; every available one by default), `rules` (ordered `(parameter-path regex, tactic)` pairs replacing the preset's table), and `min_size` (parameters smaller than this many bytes stay replicated, default 1 MiB).

Members:

- `mesh` (property) — the activated mesh, for placing a batch or reading its size
- `wrap(models)` — places every parameter on the sharding its first matching rule asks for and returns the same models. Nothing is wrapped: sharding is a property of the arrays, so the module objects and the step closures capturing them survive, and an optimizer built afterwards inherits the shardings for its own state
- `sync_initial_weights(models)` — a no-op: JAX is single-controller, so one process initializes every device
- `compile(module, compile_kw)` — `nnx.jit(module, **compile_kw)`, or *module* unchanged when *compile_kw* is `None`; the caller owns which arguments are static and which are donated. `scm flax train` compiles the learner's steps through this seam by default, which is where the Flax CLI differs from `scm torch train`, whose `--compile` is off unless given
- `shard_batch(batch)` — splits a batch across the mesh along its leading dimension and commits it to the devices, raising when an entry has no leading dimension or one the mesh size does not divide
- `state_dict(models, optimizers=None, optimizer_models=None)` and `load_state_dict(models, optimizers, optimizer_models, state)` — the full state (parameters, batch statistics, and RNG state) to and from host memory, keyed by model and by optimizer name. A typed RNG key travels as its raw key data and is rewrapped on the way back, and a restored leaf takes the dtype and the sharding of the live array it replaces — which is what makes a state saved on four devices load onto one. *optimizer_models* is accepted for protocol compatibility and unused: nnx optimizer state is already keyed by parameter path

Module constants: `AXIS` (`"data"`, the single mesh axis every preset builds), `TACTICS` (`"replicate"` and `"fsdp"`, the tactics a rule may name), and `PRESET_RULES` (each preset's ordered rule table). The `fsdp` tactic only ever splits a parameter's leading dimension — sharding any other one puts the parameter's own axis on the axis the batch is already split along — and falls back to replication for a parameter with fewer than two dimensions, one below `min_size`, or one whose leading dimension the mesh size does not divide.
