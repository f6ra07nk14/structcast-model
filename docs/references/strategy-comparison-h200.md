# Strategy comparison: full training runs on DGX H200

Does the choice of `SingleDeviceStrategy`, `DistributedDataParallelStrategy`, or
`FullyShardedDataParallelStrategy` (per-block) change what a model learns? Full-length training
runs on 8×H200 (driver 595.71.05, container `docker/train.dockerfile`, torch 2.11.0+cu130).
Global batch size is held constant across strategies; the single-device arm reaches it with
`accumulate_gradients`.

MLflow store: `/Coretronic3610/2/frankkang/structcast-validation/mlflow.db`, experiments
`StratCmp-*`.

## Language model — WikiText-103 (final)

`cfg/torch/models/SmallLanguageModel.yaml` `tiny`, context 4096, global batch 768, 10 epochs,
native AdamW at a constant lr 1e-3, `weight_decay` 0.1 (0.0 for the no-decay group).
Distributed arms use 4 ranks × 192; the single arm uses 192 × `accumulate_gradients: 4`.

Final-epoch `val_ce_loss`, three seeds per strategy:

| strategy | seed 42 | seed 43 | seed 44 | mean | spread |
| --- | --- | --- | --- | --- | --- |
| single (accumulate 4) | 1.1379 | 1.1117 | 1.1199 | 1.1232 | 0.0262 |
| DDP ×4 | 1.1605 | 1.1403 | 1.1196 | 1.1401 | 0.0409 |
| FSDP2 per-block ×4 | 1.1597 | 1.1073 | 1.1107 | 1.1259 | 0.0524 |

The spread within a single strategy (up to 0.052) exceeds the spread between strategy means
(0.017). **No strategy imposes a systematic training penalty**; the differences are seed noise.

`optimizer` (learning rate) and both `optimizer_group*_weight_decay` series are identical across
all nine runs, which is the direct check that sharding does not perturb optimizer state.

## Vision — ImageNet-1K (in progress)

Supervised classification (full image, no masking, CE head), global batch 512, 90 epochs,
constant lr 1e-3, `weight_decay` 0.05 / 0.0. Distributed arms use 4 ranks × 128; single-device
arms use 512 directly. Data staged on local NVMe (`/raid/frankkang/imagenet-1k`, 172 G).

| model | strategy | epochs | val_ce_loss | val_acc1 | val_acc5 |
| --- | --- | --- | --- | --- | --- |
| ViT-B | DDP ×4 | 90/90 | 1.5064 | 0.6497 | 0.8573 |
| ViT-B | FSDP2 per-block ×4 | 90/90 | — | 0.6454 | 0.8559 |
| ViT-B | single | running | — | — | — |
| ConvNeXt V2-B | DDP ×4 | running | — | — | — |
| ConvNeXt V2-B | FSDP2 per-block ×4 | queued | — | — | — |
| ConvNeXt V2-B | single | running | — | — | — |

Two extra single-device ViT-B seeds (43, 44) are running to give the vision side the same error
bar the language model has: the 0.43 pp DDP-vs-FSDP2 gap is not interpretable without one.

There is no scheduler in `cfg/torch/learners/ImageClassifier.yaml`, so ~65% top-1 is the expected
level for a constant-lr run, well below the ~81% a full timm recipe reaches. The comparison
target is cross-strategy behaviour, not absolute accuracy.

## Implementation error found by these runs

The single-device language-model arms reported `ce_loss` 0.2866 against 1.1649 for the 4-GPU arms
at the same global batch size — exactly the factor `accumulate_gradients: 4`. The generated
training step rebound the tracked loss to the accumulation-scaled value before `backward()`, so
every reported loss metric was divided by the accumulation count. Fixed in
`builders/torch_builder.py` by scaling inside the backward expression only; gradients were never
affected, and `val_ce_loss` (no accumulation in validation) needs no correction.

## Per-rank data verification

Both loaders were probed under `torchrun --nproc-per-node=4` before the matrix ran:

- `examples/torch/data.py` (timm wrapper): 2502 steps per rank, disjoint index sets across ranks,
  `DistributedSampler.set_epoch` called from the epoch-begin callback.
- `examples/torch/corpus.py` (`TinyShakespeareLoader`): shards through `DistributedSampler` but
  never calls `set_epoch`, so the shuffle order repeats every epoch. Harmless for a fixed-length
  language-model corpus, but it is a real difference from the timm wrapper.

## Operational notes

- `PYTHONUNBUFFERED=1` is mandatory. Without it Python block-buffers stdout and a healthy run's
  log looks frozen for hours; judge progress from the MLflow step counter, not the log tail.
- Peak dataloader throughput on this host is reached around 32 workers per rank. A first sweep
  reported 128 workers at 12.5k img/s, which was the prefetch queue draining, not steady state;
  re-measured over a 600-batch window the advantage disappears.
- Containers need `--ipc=host --ulimit memlock=-1 --ulimit stack=67108864`, and root-level
  `torch.compile` needs the model module importable, so `PYTHONPATH` must include the directory
  holding the generated modules.
