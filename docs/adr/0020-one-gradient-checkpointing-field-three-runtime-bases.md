# One GRADIENT_CHECKPOINTING field, three runtime base classes

All three frameworks now ship activation checkpointing (torch `torch.utils.checkpoint`, flax
`nnx.remat`, keras `keras.remat` since 3.9), so the field lives on the shared `UserDefinedLayer` —
`GRADIENT_CHECKPOINTING: bool | dict[str, Any] = False` — rather than behind the per-framework
template seam issue #20 designed when the feature looked torch-only. `true` enables the framework's
defaults; a mapping is the constructor keywords of that framework's mechanism, each value resolved
through `resolve_getter` so callables and object patterns work, exactly as `MIXED_PRECISION`
keywords already do. The mapping is validated by the framework builder at build time — keras
rejects any non-empty mapping outright, because `keras.remat` takes no options.

The enabling logic is not emitted per layer: each framework package ships a runtime base class
(`structcast_model.torch.layers.GradientCheckpointingLayer`, after the HuggingFace
`GradientCheckpointingLayer` shape; `structcast_model.flax.layers.GradientCheckpointingModule`;
the keras layer wraps its call implementation in `keras.remat`), and a generated layer with the
field enabled inherits that base instead of the plain framework module. No wrapper module is
introduced, so `named_modules()` paths, parameter names and state-dict keys are unchanged — which
FSDP2 `shard_modules` globs and in-place compilation depend on. The torch predicate is
`self.training and torch.is_grad_enabled()`, so checkpointing follows training mode: generated
learners run frozen models in `eval()` while gradients still flow through them, and those models are
deliberately left uncheckpointed. Recomputing a forward pass that is defined to behave differently
in the two modes is the anomaly, so the memory saving is declined rather than bought under other
semantics; `torch.is_grad_enabled()` decides nothing beyond keeping a `no_grad` inference forward
from paying for a checkpoint no backward pass will use. The flax base routes through a closure
taking the module and arrays positionally —
`nnx.remat` cannot resolve keyword-only parameters or `functools.partial`, verified against
flax 0.12.8 with identical gradients eager, jitted and under a checkpoint policy.

Recomputation runs the body twice, so a checkpointed layer must hold no state the second pass would
advance. Keras refuses at build time to checkpoint a layer whose own flow builds a seed-consuming or
variable-updating Keras layer — `keras.remat` re-draws `SeedGenerator` state and re-runs
normalization updates, which is silently different gradients on TensorFlow and PyTorch and a tracer
error on JAX. The blocklist is by name, so an equivalent user-defined layer is documented in
`REFERENCE.md` rather than caught; torch's own recomputation caveat for buffer-mutating sub-layers
is documented there too rather than refused, since ADR-0009's batch-norm-bearing backbones are the
expected workload.

The resolved configuration is a field of the layer intermediate, so the content-hash deduplication
distinguishes otherwise identical layers with different checkpoint behavior. The recomputation cost
caveat lives in `REFERENCE.md`, never in generated comments (ADR-0019). Option availability across
torch versions is documented there too; the floors do not move (ADR-0003's guard-not-floor stance).

## Considered options

- A `TorchUserDefinedLayer`/`TorchTemplateLayer` seam with per-framework same-named fields
  (issue #20's design, extended threefold). Rejected once the field became genuinely shared: the
  seam exists to host framework-specific fields, and this field is framework-neutral at the bool
  level while the mapping is validated per framework anyway — ADR-0017's lesson ("one field, three
  half-agreements") is avoided by validating, not by splitting the field.
