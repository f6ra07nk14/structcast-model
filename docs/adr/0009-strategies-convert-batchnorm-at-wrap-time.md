# The multi-rank strategies convert BatchNorm at wrap time

Cross-rank `BatchNorm` used to be the user's problem: the README told model authors to call
`torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)` in the model definition. The conversion now happens
inside `DistributedDataParallelStrategy.wrap()` and `FullyShardedDataParallelStrategy.wrap()`, on by
default, controlled by a `sync_batchnorm: bool = True` field on both.

## Why the strategy owns it, and why at the top of `wrap()`

Per-GPU `BatchNorm` statistics silently skew training at the small per-GPU batch sizes distributed runs
produce, and the fix is mechanical — exactly the kind of distributed concern ADR-0003 put behind the
strategy instead of leaving it to convention in user code. Documentation asking every model author to
remember one call is the failure mode that ADR-0003's initial-weight broadcast already rejected: correctness
depending on a habit, failing silently when the habit lapses.

`wrap()` is the only place where the conversion is both possible and safe. It runs after
`sync_initial_weights()`, and the conversion carries parameters and buffers over by reference, so the
rank-0 broadcast stays authoritative. It runs before the wrapper exists, which the module tree must be final
for: DDP's reducer is built from the modules it is handed, `fully_shard` replaces parameters in place, and
per-block sharding matches `named_modules()` paths — matching a tree whose layers are about to be replaced
would shard the wrong modules. Hence the conversion sits at the very top of `wrap()`, above FSDP2's mesh
initialization and pattern matching, and the strategies wrap the conversion's *return value*: containers are
converted in place, but a model that is itself a `BatchNorm` layer comes back as a new object.

`SingleDeviceStrategy` does not convert: there is nothing to synchronize across, and `SyncBatchNorm` on one
device is pure overhead.

## CPU devices are skipped

`SyncBatchNorm`'s training forward rejects CPU input whenever `torch.distributed` is initialized — the check
precedes the world-size check, so even a single-rank gloo run raises. A CPU run therefore keeps its plain
`BatchNorm` layers.

The skip is forced by torch, not free. Multi-rank CPU training over gloo is reachable — `torchrun
--nproc_per_node=N ... torch train ... --device cpu` — and those runs keep per-rank `BatchNorm` statistics,
the very skew the conversion exists to remove. torch offers no CPU implementation to convert to, so the
alternative to skipping is raising at the first training step; the statistics gap is accepted and recorded
here rather than papered over.

## The converter is timm's, not torch's

The conversion calls `timm.layers.convert_sync_batchnorm`, already a hard dependency of every torch extra.
torch's stock `torch.nn.SyncBatchNorm.convert_sync_batchnorm` was rejected: it matches on `_BatchNorm` and
replaces the match with a plain `SyncBatchNorm`, which for timm's `BatchNormAct2d` — the default norm layer
across timm's efficientnet/mobilenet/regnet families, all reachable from a model pattern — silently strips
the fused activation, leaving the `state_dict` keys unchanged and the failure invisible. timm's converter
maps `BatchNormAct2d` to `SyncBatchNormAct`, a `torch.nn.SyncBatchNorm` subclass that keeps running the
activation, and plain `BatchNorm` to `torch.nn.SyncBatchNorm` exactly as torch does. Norm layers from other
third-party libraries are still flattened to plain `SyncBatchNorm`; the off-switch is their escape.

## The walk skips layers that are already synchronized

timm's converter is not idempotent, verified against timm 1.0.28: `SyncBatchNormAct` subclasses
`torch.nn.SyncBatchNorm` but not `BatchNormAct2d`, so a second pass matches it as an ordinary `_BatchNorm`
and rebuilds it as a plain `torch.nn.SyncBatchNorm` — the fused activation is dropped, the `state_dict` keys
are unchanged, and nothing raises. A plain `torch.nn.SyncBatchNorm` likewise comes back re-created with its
`process_group` reset to `None`. Both are reachable in practice: a model definition that still calls a
converter itself, or any second `wrap()` of the same models.

The conversion therefore walks the tree itself and hands only not-yet-synchronized `_BatchNorm` layers to
timm's converter; anything that already is a `torch.nn.SyncBatchNorm` is returned as-is. The walk is a few
lines against a defect verified in the pinned dependency, not a speculative reimplementation of it.

## The off-switch is YAML only

`sync_batchnorm` is a constructor field on the strategy pattern, disabled with
`_bind_: {sync_batchnorm: false}` and nothing else. ADR-0003 rejected dedicated CLI flags per knob, and this
knob is no more special than `reshard_after_forward`.

## Accepted consequences

The conversion is idempotent, and pre-converted models keep working: layers that already are
`torch.nn.SyncBatchNorm` — timm's fused `SyncBatchNormAct` included — pass through untouched, keeping their
identity and their `process_group`, so a hand-built subgroup survives and running `wrap()` twice is safe.

Every layer the conversion *does* replace is a new object, and whatever was attached to the old one is
dropped. That covers user-registered forward/backward hooks on any converted `BatchNorm`, not only roots.
`--compile` makes it visible at the top level too: the CLI compiles before `wrap()` and compiles in place —
the model root, or the matched `shard_modules` blocks under FSDP2 — so a model whose root *is* a `BatchNorm`,
or a matched block that is one, loses that compilation when it is replaced.

Separately, `torch.compile` graph-breaks on `SyncBatchNorm` (pytorch/pytorch#161302); the break is the price
of correct statistics and is documented rather than worked around.
