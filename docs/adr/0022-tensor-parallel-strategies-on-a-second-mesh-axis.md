# Tensor-parallel strategies on a second mesh axis

All three backends gain tensor parallelism behind their existing strategy surfaces, including the
two-dimensional data×model combinations, with the declaration living in the strategy pattern —
never in a model template, which must keep running unchanged under `single`/`dp`/`tp`.

- **torch** — a `TensorParallelStrategy` (1-D) and an FSDP2+TP combination on a 2-D
  `init_device_mesh`, the mesh construction lifted out of the FSDP2 class; TP applies first,
  `fully_shard` wraps the result on the data submesh (torchtitan's order). The plan is a
  `parallel_modules` table of house segment-globs to styles — a short vocabulary (`column`, `row`,
  `sequence`, `column_heads` for the `use_local_output=False` attention shape) with an
  object-pattern escape hatch for any `ParallelStyle` the vocabulary lacks. Globs compile through
  the existing `shard_modules` machinery: unmatched patterns error, the tied-parameter check runs.
  The protocol gains `data_rank`/`data_world_size` — the demonstrated defect ADR-0003 demands:
  ranks of one TP group must consume the identical batch and identical dropout seed, and both the
  dataset patterns and the CLI's `seed + rank` derivation previously only knew the global rank.
  `loss_parallel` is out of scope (cross-entropy-only, mean-reduction constraints).
- **flax** — `tp` and `fsdp_tp` presets add a `"model"` axis and the tactics `column` and `row`.
  The `row` tactic shards the kernel only and pins the bias to `P()` — the one silent-wrong-answer
  TP mistake (a sharded or per-shard-added bias is multiply-counted by the all-reduce) is encoded
  in the tactic, unreachable from a rule table. The data axis stays Explicit; the model axis
  defaults to `AxisType.Auto`, so plain layers row-parallelize with no model-template change. The
  drift argument that once justified Explicit-everywhere for accumulation did not survive
  re-verification: the zeros-inheritance hazard is fixed upstream (`shard_like`), the oscillation
  did not reproduce on the current stack, and a sharding flip costs at most one extra cache entry
  plus resharding, not perpetual retraces. A strategy option `model_axis_mode: explicit` restores
  typed shardings for templates that carry their own annotations — `dot_general:
  "eval: dot_general_out(...)"` (the shipped helper; `functools.partial` silently loses to flax's
  explicit `out_sharding=` pass-through) or `nnx.with_partitioning` initializers — and only in
  that mode does `wrap` verify every row-matched layer carries the hook, erroring with the exact
  YAML fix otherwise. Precedence under the tp presets: a rule match wins; an unmatched parameter
  keeps its construction sharding, so template annotations survive. `dp`/`fsdp` semantics are
  untouched.
- **keras** — `tp` on the JAX backend only, through `RuleModelParallel.get_variable_layout`
  learning non-leading axes on a 2-D mesh with the batch dimension named; the TensorFlow and torch
  backends get `REJECTED` rows naming the reason, because both would otherwise fail silently
  (ADR-0016's matrix exists for exactly this).

## Consequences

- Supersedes ADR-0014's "TP is out of scope" clause and narrows its "Explicit axis types
  everywhere": the model axis of the flax tp presets is Auto by default, by choice, with the
  explicit mode one option away. Amends ADR-0003's three-strategies enumeration and its protocol
  member list.
- CONTEXT.md's Strategy-preset glossary gains `tp` and the combined presets.
- H200 validation must include numerics parity (tp degree 2 and 4 against single-device loss
  curves), the (2,2) combination, and throughput — a TP mistake is a silent numeric one.
