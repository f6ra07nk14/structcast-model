# Keras training drives Keras-native APIs behind per-backend adapters

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
- The keras state backend serializes `model.get_state_tree(value_format="numpy_array")` into a
  single archive — path-keyed and backend-portable — yet resume refuses a backend mismatch:
  normalization statistics and RNG trajectories are not verified equivalent across backends, and a
  silently different continuation is worse than a clear error.
- The CLI resolves `KERAS_BACKEND` before keras imports, with no default backend; a conflict with an
  already-initialized backend fails loudly.
- Criteria returned by a training or inference step are reduced across replicas by the backend
  adapter (or the distributed strategy driving it) before they reach the tracker — the tracker
  itself never all-reduces, unlike its torch twin, and a distributed cell that skips this reduction
  must fail its strategy tests rather than silently log per-replica values.
