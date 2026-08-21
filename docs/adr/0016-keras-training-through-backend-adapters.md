# Keras training drives Keras-native APIs behind per-backend adapters

> **Superseded in part by ADR-0017.** The gradient-accumulation consequence below — `update()`
> returning true every step and the accepted update-counter divergence — is replaced by the
> private-counter gate of ADR-0017. The rest stands.

Keras 3 runs on three backends whose training mechanics disagree: TensorFlow allows only stateful
`optimizer.apply` under a `GradientTape`, JAX requires the stateless `stateless_call` /
`stateless_apply` path under `jax.jit`, and torch uses autograd with path-keyed parameters. We
generate one backend-neutral Keras learner and route every backend-specific concern — gradient
computation, optimizer application, variable state handling, step compilation — through a backend
adapter selected exactly once from `keras.backend.backend()` at startup. Distribution likewise uses
each backend's Keras-native path: `keras.distribution` on JAX (`dp` and `fsdp`),
`tf.distribute.MirroredStrategy` on TensorFlow (`dp`), and DistributedDataParallel wrapping on torch
(`dp`); unsupported cells (`fsdp` on TensorFlow or torch) are rejected at validation with the reason.

## Considered options

Reusing the existing flax/nnx training loop for the JAX backend (the keras.io NNX guide shows
exactly that) and the existing torch loop including FSDP2 for the torch backend. CPU probes rejected
both:

- Keras variables are not `nnx.Param`, so the flax builder's `wrt=Param` loop silently trains
  nothing and the `fsdp` preset silently replicates every parameter.
- Keras `SeedGenerator` state is unreachable from the model's nnx graph, so `nnx.jit` freezes
  dropout masks and leaks tracers; random layers break under a sharded batch on the explicit mesh.
- FSDP2's `fully_shard` swaps module parameters to DTensors while `keras.Variable.value` keeps
  returning the stale cached tensor — the loss repeats bit-identically with no error.
- The upstream NNX integration is documented but untested: keras CI runs one basic-flow MLP under
  `KERAS_NNX_ENABLED`, has no `nnx.Optimizer` training test, and no third-party usage exists.

A training path whose dominant failure mode is a silent no-op is disqualified regardless of code
reuse savings.

## Consequences

- `MIXED_PRECISION` maps to `keras.mixed_precision` global policies; `float16` auto-wraps the
  optimizer in `LossScaleOptimizer` (a dict supplies its kwargs), `bfloat16` does not. The optimizer
  pattern stays a keras-optimizer pattern on every backend, and gradient clipping lives inside it
  (`clipnorm` / `global_clipnorm`), so the keras learner schema has no `CLIP` field.
- The keras state backend serializes every model and optimizer variable into a single archive, each
  nested under the segments of its `variable.path` — backend-portable, and restored by assigning the
  matching paths back rather than through `keras.Model.set_state_tree`, whose category-keyed shape
  (`trainable_variables`, `optimizer_variables`, …) does not apply to a learner whose optimizers are
  not attached to its models. Resume still refuses a backend mismatch: normalization statistics and
  RNG trajectories are not verified equivalent across backends, and a silently different
  continuation is worse than a clear error.
- The CLI resolves `KERAS_BACKEND` before keras imports, with no default backend; a conflict with an
  already-initialized backend fails loudly. `scm keras time` is the one exception: it trains nothing
  and takes no `--backend`, inheriting the ambient one (`~/.keras/keras.json`) and printing which
  one produced the timing.
- Gradient accumulation is the keras optimizer's own (`gradient_accumulation_steps`), which owns the
  gate deciding when an update lands, so the generated learner's `update()` returns true every step.
  Under `ACCUMULATE_GRADIENTS` the keras update counter therefore advances per step, not per
  optimizer application, unlike its torch and flax twins; re-deriving the gate in the learner would
  be a second implementation free to drift from the optimizer's real apply phase.
- Criteria returned by a training or inference step are reduced across replicas by the backend
  adapter (or the distributed strategy driving it) before they reach the tracker — the tracker
  itself never all-reduces, unlike its torch twin, and a distributed cell that skips this reduction
  must fail its strategy tests rather than silently log per-replica values.
- The `dp` preset applies the *mean* of the per-replica gradients on every backend, so a run's step
  size does not depend on how many devices it was given. JAX gets that from the sharded step and
  torch from `DistributedDataParallel`; the Keras TensorFlow optimizer all-reduces with
  `ReduceOp.SUM`, so the strategy divides each segment's loss by the replica count before the
  adapter differentiates it.
