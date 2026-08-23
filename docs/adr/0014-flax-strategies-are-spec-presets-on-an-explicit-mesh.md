# Flax distributed strategies are sharding-rule presets on an always-on Explicit mesh

> **Superseded in part by ADR-0022.** Tensor parallelism is no longer out of scope: `tp` and
> `fsdp_tp` presets add a model axis, and ADR-0022's H200 amendment retires the "Explicit axis types
> everywhere" rule outright — every axis, the data one included, is Auto unless
> `model_axis_mode: explicit` types the whole mesh. The accumulation drift experiment below did not
> reproduce on the current stack (the zeros-inheritance hazard is fixed upstream), and the remaining
> rationale for Explicit — loud errors over silent compiler choices — turned out to reject
> legitimate models: the errors land inside `nnx.Embed`'s gather and a class-token concatenate,
> library code no model template can annotate.

JAX has no strategy zoo: one mechanism — a device mesh plus a `PartitionSpec` per tensor — expresses
single-device, data-parallel, and FSDP execution, with XLA inserting every collective. The Flax
`DistributedStrategy` implementation is therefore **one class**, not a mirror of the three torch classes: on
this side `wrap()` returns the models untouched (sharding is Variable metadata applied eagerly at
construction), `sync_initial_weights()` is a no-op (single controller, one globally-addressable init),
the protocol carries no gradient-scaler seam at all (bf16 needs none, and there is no fp16 scaler here), and `compile()` is the `nnx.jit` seam. What
distinguishes strategies is only a **preset**: the mesh to build and an ordered `(parameter-path regex, tactic)`
rule table deciding each parameter's spec — the shape used by big_vision/scalax, first match wins, with a
minimum-size cutoff so biases and norm scales stay replicated.

v1 ships `single`, `dp`, and `fsdp`. Presets are YAML/CLI-selectable; a custom rule table
remains expressible through the strategy object pattern.

## Explicit axis types everywhere, including a size-1 mesh

Experiments (4 simulated CPU devices, flax 0.12.8 / jax 0.11) settled a documentation contradiction:

- The reported "FSDP fails under Explicit" reproduces only with a `P(None, 'data')` annotation — which shards
  `out_features` on the batch axis and is not FSDP. The correct dim-0 layout `P('data', None)` traces, keeps
  every param spec bit-stable, and needs `cache_size == 1`.
- The reported gradient-accumulation sharding drift (`P(None,'data')` → `P()` after the reset branch, one
  retrace per flip) is an Auto-mode GSPMD artifact. Under Explicit the accumulator spec is stable across
  micro-steps with no `with_sharding_constraint` at all.

So the strategy always runs under `jax.set_mesh(mesh)` with Explicit axis types — `single` is
`make_mesh((1,), ('data',))`, not a branch — and model construction must happen inside the mesh scope (eager
sharding raises otherwise; note `jax.set_mesh` takes effect at `__init__`, not `__enter__`). Batch sharding
lives at the loader seam via `jax.device_put(batch, NamedSharding(mesh, P('data', None)))`; divisibility of the
per-microbatch size by the mesh is checked there, as each batch is placed, and a batch entry the mesh does not
divide is rejected by name -- a configuration-time check is impossible, since a dataset is an object pattern
whose batch shapes only exist at run time. A parameter whose dimension does not divide falls back to replicate
rather than erroring inside a trace.

## What is deliberately not shipped

- **ZeRO-2**: no JAX/Flax API expresses it and it is not reachable by composition — under jit, per-step
  gradients are fused-program intermediates, and forcing sharded-gradients-with-replicated-params back costs
  the full ZeRO-3 all-gather for ~1.5Ψ less saving than ZeRO-3 itself.
- **ZeRO-1** (`optimizer_sharding` metadata): deferred. It is not a sharding fixed point (params drift to `P('data', None)`
  after `update`, and the repair reshard makes its communication identical to FSDP for less than half the
  memory saving); and the checkpoint round-trip silently drops the optimizer-state sharding.
- **Tensor parallelism**: verified feasible (the `dot_general=` constructor hook reaches row-parallel
  `out_sharding` from YAML today) but deferred to a coordinated torch/Flax/Keras strategy issue (#29). Its one
  silent-wrong-answer trap — the row-parallel bias must stay replicated — is recorded there.

## Trade-offs

- Explicit mode makes wrong sharding a trace-time `ShardingTypeError` instead of a silent compiler choice, at
  the price of ruling out row-parallel TP through plain layers — accepted, TP is out of scope here.
- An always-on size-1 Explicit mesh has no official precedent; it carries its own regression test rather than a
  citation.
- Loss parity across mesh sizes is `allclose(1e-6)` on the first step (reduction order), bitwise afterwards —
  parity tests must not assert bitwise equality on step one.
