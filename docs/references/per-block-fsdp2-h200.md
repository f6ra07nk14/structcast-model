# Per-block fully_shard: validation on DGX H200

Evidence backing `FullyShardedDataParallelStrategy.shard_modules` (ADR-0004): per-block sharding
of the generated Transformer example, on 4×H200 (driver 595.71.05, container
`docker/train.dockerfile`, torch 2.11.0+cu130).

## Setup

- Model: generated `cfg/torch/models/Transformer.yaml` `small` (dim 512, 8 heads, 8 blocks,
  vocab 65, sequence 256) — `named_modules()` paths `backbone.block0..7`.
- Patterns: `shard_modules: ["backbone.block*"]` (segment glob: matches the 8 blocks and nothing
  inside them). Both arms `reshard_after_forward: true`, batch 16×256 per rank, 4 ranks, AdamW.

## Results (2026-08-13)

Peak memory, 5 training steps, `max_memory_allocated` all-reduced MAX over ranks:

| Arm         | child FSDP groups | peak (MiB) | last loss |
| ----------- | ----------------- | ---------- | --------- |
| single-root | 0                 | 1392.7     | 3.422     |
| per-block   | 8 (`backbone.block0..7`) | **1250.3** (−10.2%) | 3.422 |

Identical losses: the arms are numerically equivalent, only the communication grouping differs.
The saving is bounded by this model's size (~25M parameters; activations dominate) — it grows with
the parameter share of peak memory, since single-root holds the whole model unsharded through
forward while per-block holds one block plus the sharded rest.

End-to-end CLI (`torchrun --nproc-per-node=4 … --strategy fsdp2(shard_modules) --compile true`,
Tiny Shakespeare via `TinyShakespeareLoader`): epochs 1→2 val_ce_loss 2.490 → 2.437; resume from
the epoch-2 DCP training state continued at epoch 3 with 2.290. Per-block in-place compile
(`nn.Module.compile` on each matched block) ran in the same commands.

## Incidental findings (both fixed)

- A bare `DataLoader` feeds every rank the same CPU batches: the example gained
  `TinyShakespeareLoader`, which owns device placement and `DistributedSampler` sharding — the
  training loop deliberately does neither (same convention as the timm wrapper in
  `examples/torch/data.py`).
- FSDP2's checkpoint path refuses optimizer proxies (`AdamWWithCosine`), by design: the
  Transformer learner switched to a native `torch.optim.AdamW` built by
  `structcast_model.torch.optimizers.create_opt`.
