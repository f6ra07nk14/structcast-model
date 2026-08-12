# A distributed strategy owns the distributed lifecycle

This ADR supersedes the "excluding … DDP wrapping" boundary and its recorded limitation in ADR-0002:
distributed wrapping becomes part of learner assembly, ordered before learner construction, and every
distributed concern lives behind one replaceable object.

## Why the wrap moves before learner construction

Generated learner step closures and optimizers capture the exact module objects handed to `__init__`
(`torch_builder.py` emits the step functions as closures over the constructor arguments). Wrapping after
construction can therefore never reach the training path: with DDP the reducer never arms and gradient
synchronization silently does not happen; with FSDP2 (`fully_shard` swaps parameters to DTensors in place)
the optimizer built inside `__init__` holds orphaned parameters and silently trains nothing. The only fix is
ordering: instantiate → shape resolution/dummy forward (DDP does not forward attribute lookups, so shape
resolution must see raw modules) → initializers → initial-weight sync → wrap → learner construction.

## One strategy object instead of scattered branches

The four places that must behave differently per distribution mode — wrapping, initial-weight
synchronization, checkpoint state production, and checkpoint loading — used to be (or would have become)
`isinstance` branches spread over the CLI, the trainer, and the saver callbacks. They are collapsed into a
`DistributedStrategy` protocol (`torch/distributed.py`) with three implementations, named in full:
`SingleDeviceStrategy`, `DistributedDataParallelStrategy`, `FullyShardedDataParallelStrategy`. The
interface is deliberately minimal: `wrap`, `sync_initial_weights`, `state_dict`, `load_state_dict`, and a
`grad_scaler_creator` attribute. Every member exists because a verified defect required strategy-specific
behavior; nothing speculative is included. PyTorch Lightning's `Strategy` hierarchy and MMEngine's
`_strategy` package are the same shape, independently converged on.

A reusable per-model wrapper facade (a module wrapping DDP/FSDP2 modules) was rejected: torch internals
dispatch on the real wrapper types, so a facade breaks `isinstance` semantics, hook registration, and
parameter naming at once. Reintroducing a learner factory was also rejected — ADR-0002's revert stands; the
fix needs only ordering, not a class.

## Initial weights are synchronized by explicit broadcast

Rank agreement used to be a side effect of `DistributedDataParallel.__init__`'s `_sync_module_states`
broadcast — owned by nobody, absent under FSDP2, and silently defeated by the per-rank seeding
(`seed + global_rank`) that runs before model construction. `sync_initial_weights` now broadcasts rank 0's
parameters and buffers explicitly, before wrapping, so the tensors are plain and one implementation serves
DDP and FSDP2 alike. The alternative — same-seed construction on every rank — was rejected because it turns
correctness into a convention (every initializer deterministic, no rank-dependent RNG consumption before a
reseed point) whose failure mode is silent divergence, undetectable without cross-rank comparison.

## Strategy selection is an object pattern

The CLI takes `--strategy`, an object pattern like every other configurable in this CLI, and calls
the instantiated factory with the runtime arguments (`device`, `local_rank`). Without a pattern, a detected
distributed environment defaults to `DistributedDataParallelStrategy` (preserving the existing torchrun UX)
and a single device to `SingleDeviceStrategy`. FSDP2's wrap-time knobs (`reshard_after_forward`,
`mp_policy`) are constructor arguments expressed in the pattern; the device mesh derives from the default
process group. Dedicated CLI flags per knob were rejected as surface-area growth that breaks the
"everything is a pattern" convention.

## torch floor stays; FSDP2 is import-guarded

`fully_shard`'s stable API needs torch >= 2.6, and the DTensor-aware gradient scaler needs >= 2.5, but the
`torch-cpu`/`torch-cu118` extras keep their `>=2.0.0` floor: `FullyShardedDataParallelStrategy` raises an
actionable `ImportError` when `torch.distributed.fsdp.fully_shard` is missing, and DDP/single-device keep
working on old torch. Raising the floors per extra was the alternative; the guard was chosen to avoid
forcing upgrades on users who never touch FSDP2.

## Gradient scalers are created through the strategy, and only for float16

`MIXED_PRECISION` now means "gradient scaling", which only applies to `float16`: bfloat16 shares float32's
exponent range, so scaling it is pure overhead (the shipped bf16-plus-scaler default config was exactly this
mistake, invisible only because the scaler also silently self-disabled on CPU by defaulting to
`device="cuda"`). Generated fp16 learners take `__grad_scaler_creator__` as an explicit constructor
parameter defaulting to `torch.amp.GradScaler`, and the CLI passes `strategy.grad_scaler_creator` — the
scaler is captured by step closures at construction, so its class must be right before the learner exists.
All three strategies currently return the plain `torch.amp.GradScaler`: since torch 2.5 (pytorch/pytorch
PR #132816) the DTensor dispatcher all-reduces `found_inf` inside `unscale_`, so FSDP2 needs no sharded
scaler class; `ShardedGradScaler` is an FSDP1 artifact that double-reduces and is soft-deprecated upstream.
