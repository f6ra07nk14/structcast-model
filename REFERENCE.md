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
    - [Loggers](#loggers)
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

**`IMPORTS`** — Same format as in the model schema. Injects additional Python imports into the generated learner file.

```yaml
IMPORTS: {}
```

**`INPUTS` and `OUTPUTS`** — `INPUTS` lists the tensor names the generated learner expects as keyword arguments during training and inference. `OUTPUTS` lists the criterion names produced by the flow; they become the learner's `outputs` attribute, which `scm torch train` reads to build the tracker. Both default to `[]`, which instructs the code generator to infer them automatically from the `LEARNERS` entries' `FLOW` definitions.

```yaml
INPUTS: []                # auto-inferred from LEARNERS[*].FLOW
OUTPUTS: [loss_G, loss_GAN, loss_cycle, loss_identity, loss_D_A, loss_D_B, fake_A, fake_B]
```

**`MIXED_PRECISION`** — Controls gradient scaling, which only counteracts float16 underflow. The scaler is built through the learner's injectable `__grad_scaler_creator__` argument, which defaults to `torch.amp.GradScaler` and is called with the training `device`.

| Value             | Behavior                                                                                                                                     |
| ----------------- | -------------------------------------------------------------------------------------------------------------------------------------------- |
| `false` (default) | No `GradScaler` is created. Autocast still runs when `MIXED_PRECISION_TYPE` is set — this is the bfloat16 path, which needs no scaler.        |
| `true`            | Gradient scaling enabled with default `GradScaler` settings; requires `MIXED_PRECISION_TYPE: float16`.                                        |
| `dict`            | Gradient scaling enabled; the dict is forwarded as keyword arguments to the scaler creator. Requires `MIXED_PRECISION_TYPE: float16`.         |

```yaml
MIXED_PRECISION:
  init_scale: "eval: 2.0**16"
  growth_factor: 2.0
  backoff_factor: 0.5
  growth_interval: 2000
  enabled: True
```

**`MIXED_PRECISION_TYPE`** — The dtype forwarded to `torch.autocast` when mixed precision is enabled. Accepts `"bfloat16"` or `"float16"`. It is valid on its own — `MIXED_PRECISION: false` with `MIXED_PRECISION_TYPE: bfloat16` is autocast without a scaler. Enabling `MIXED_PRECISION` with anything other than `float16` raises a `SpecError` at build time, because gradient scaling only applies to float16.

```yaml
MIXED_PRECISION_TYPE: bfloat16
```

**`ACCUMULATE_GRADIENTS`** — The number of forward–backward steps to accumulate before calling the optimizer. Set to `null` to disable accumulation (optimizer steps every batch). When set to a positive integer `n`, `optimizer.step()` and `optimizer.zero_grad()` are called once every `n` batches, and the generated `update(step)` returns `True` only on those steps — which is what makes the trainer fire `on_update` once per update rather than once per step.

```yaml
ACCUMULATE_GRADIENTS: null   # disabled
ACCUMULATE_GRADIENTS: 4      # accumulate over 4 steps
```

#### `LEARNERS`

An ordered list of `LearnerBehavior` entries. Each entry defines one loss to differentiate, one optimizer to update, and its own execution graph. Multiple entries enable multi-optimizer training (e.g., GAN-style training where generator and discriminator optimizers are stepped independently).

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

Each entry in `LEARNERS` is a `LearnerBehavior` with the following fields:

- **`NAME`** (`str`): Optional identifier for this entry. Used as the generated attribute name for the optimizer, and as its key in the learner's `optimizers` property. Must be a valid Python identifier.
- **`LOSS`** (`str`): The loss key (produced by the `FLOW`) that this entry differentiates.
- **`TRAINABLE_LAYERS`** (`list[str]`): Model names whose parameters this optimizer manages. Each value must match a model passed to the learner constructor.
- **`FLOW`** (`list`): Training-time execution graph for this entry. Uses the same entry format as model `FLOW` (see [`FLOW` Entry Format](#flow-entry-format)), plus support for `"eval: ..."` expressions and inline layer instantiation via StructCast patterns.
- **`INFERENCE_FLOW`** (`list`): Optional inference-time execution graph. When absent, `FLOW` is used for inference as well.
- **`OPTIMIZER`** (StructCast pattern): A StructCast `ObjectPattern` that constructs the optimizer, called with the named parameters of `TRAINABLE_LAYERS`. It may address `structcast_model.torch.optimizers.create_opt` directly, or an optimizer + scheduler composition loaded from a file with `_file_` — see [`examples/torch/optimizers.py`](examples/torch/optimizers.py). Such a composition implements the event protocols itself, so the trainer steps its schedule when it scans the learner's `optimizers`.
- **`CLIP`** (StructCast pattern or `null`): Optional gradient-clipping callable. When non-null, the pattern is bound once and called before each optimizer step with the parameters identified by `TRAINABLE_LAYERS`. Set to `null` to disable gradient clipping.
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
- `update(step) -> bool` — whether the given training step applied the optimizers. `False` means gradients are still accumulating.
- `training_step(**inputs) -> dict[str, Any]` — runs one training batch and returns its criteria.
- `inference_step(**inputs) -> dict[str, Any]` — runs one validation batch and returns its criteria.

Two required members are also read elsewhere in the toolkit: `optimizers` (a mapping, additionally scanned for event protocols by the trainer) and `learning_rates` (shown by `ProgressBar` / `Printer` and logged by the loggers). Optional members: `grad_scalers` and `param_group_names` (saved and logged by the CLI), and `weight_decays` (per-group decay metrics merged into the logged epoch metrics; generated learners flatten it from `create_opt`'s parameter groups via `get_decays`).

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
    def on_best(self, info: BaseInfo, best: BestCriterion) -> None:
        ...  # log or save by best.value / best.step

checkpoint = BestCriterion(target="val_acc1", mode="max", callbacks=[SaveCheckpoint()])
trainer = TorchTrainer(device="cuda", learner=learner, tracker=tracker, data=data, callbacks=[checkpoint])
```

Fields: `target` (str), `mode` (`"min"` or `"max"`, default `"min"`), `callbacks` (list of `OnBest` participants, notified after every epoch in which the target appeared, whether or not it improved; named `callbacks` so the field cannot shadow the protocol method in an isinstance check). Properties: `value` (the best value so far) and `step` (the step at which it was reached).

---

## API Reference: `trainer.py`

[`src/structcast_model/torch/trainer.py`](src/structcast_model/torch/trainer.py) contains the PyTorch-specific runtime layer. The `DistributedStrategy` implementations it saves states through live in [`src/structcast_model/torch/distributed.py`](src/structcast_model/torch/distributed.py) — `SingleDeviceStrategy`, `DistributedDataParallelStrategy`, and `FullyShardedDataParallelStrategy` (FSDP2, `torch>=2.6`) — together with `sync_gate(model, armed)`, the per-call gradient-synchronization gate the generated learners use.

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

**`Logger`** (`structcast_model.torch.logger`) — The runtime-checkable protocol both backends implement: `log_params`, `log_dict`, `log_artifact`, `log_metric`, `log_metrics`, `log_state_dict`, `on_epoch_end`, and the `__enter__` / `__exit__` pair that owns the run.

**`MLflowLogger`** (`structcast_model.torch.mlflow_logger`) and **`WandbLogger`** (`structcast_model.torch.wandb_logger`) — Record a run to MLflow or to Weights & Biases through that protocol. Each is a context manager owning the run: entering it starts the run, leaving it ends the run. Both also implement `on_epoch_end`, which logs the criteria of the finished epoch together with the learner's learning rates and its optional `weight_decays` — so passing a logger in `callbacks` is enough to get per-epoch metrics, including weight/layer-decay dynamics.

```python
from structcast_model.torch.mlflow_logger import MLflowLogger

with MLflowLogger(experiment="my-experiment") as logger:
    trainer = TorchTrainer(device="cuda", learner=learner, tracker=tracker, data=data, callbacks=[logger])
    logger.log_params({"epochs": 5})
    trainer.fit(epochs=5)
```

`MLflowLogger` needs the `mlflow` extra, `WandbLogger` the `wandb` extra; each module imports its backend at import time and raises a descriptive `ImportError` from the constructor when it is missing.

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
