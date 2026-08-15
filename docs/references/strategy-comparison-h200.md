# Strategy comparison: full training runs on DGX H200

Does the choice of `SingleDeviceStrategy`, `DistributedDataParallelStrategy`, or
`FullyShardedDataParallelStrategy` (per-block) change what a model learns? Full-length training
runs on 8×H200 (driver 595.71.05, container `docker/train.dockerfile`, torch 2.11.0+cu130).
Global batch size is held constant across strategies: the language-model single-device arm reaches
it with `accumulate_gradients: 4`, the vision single-device arms take the full batch directly.

**Answer: no.** Across two vision architectures and one language model, every cross-strategy
spread is smaller than the spread between two runs of the same strategy. The strategies differ in
throughput and memory, not in what the model learns.

MLflow store: `<data-root>/structcast-validation/mlflow.db`, experiments `StratCmp-*` (uncompiled
vision pass and the language model) and `StratCmpC-*` (the vision matrix).

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

## Vision — ImageNet-1K (final)

Supervised classification (full image, no masking, CE head), global batch 512, 90 epochs,
constant lr 1e-3, `weight_decay` 0.05 / 0.0, `torch.compile` on. Distributed arms use 4 ranks ×
128; single-device arms use 512 directly. Data staged on local NVMe
(`<nvme-root>/imagenet-1k`, 172 G). Experiments are prefixed `StratCmpC-`.

Both models, all three strategies, 90 epochs:

| model | strategy | ce_loss | val_ce_loss | val_acc1 | val_acc5 | min/epoch |
| --- | --- | --- | --- | --- | --- | --- |
| ViT-B | single | 1.3845 | 1.5516 | 0.6404 | 0.8509 | 7.30 |
| ViT-B | DDP ×4 | 1.3687 | 1.5352 | 0.6431 | 0.8533 | 2.77 |
| ViT-B | FSDP2 per-block ×4 | 1.3496 | 1.5432 | 0.6414 | 0.8534 | 3.74 |
| ConvNeXt V2-B | single | 0.9292 | 1.1511 | 0.7284 | 0.9093 | 14.02 |
| ConvNeXt V2-B | DDP ×4 | 0.9350 | 1.1433 | 0.7279 | 0.9102 | 4.22 |
| ConvNeXt V2-B | FSDP2 per-block ×4 | 0.9340 | 1.1576 | 0.7269 | 0.9082 | 7.75 |

Top-1 spans **0.27 pp** across the ViT-B arms and **0.15 pp** across the ConvNeXt V2-B arms;
val_ce_loss spans 0.016 and 0.014. Both are smaller than the spread between two runs of one
strategy (next section), so **no strategy costs measurable accuracy** at a fixed global batch of
512, on either architecture. The ordering inside those spans carries no signal and should not be
read — the single-device arm happens to lead on ConvNeXt and trail on ViT.

What does differ is cost, and it is architecture-dependent:

| | ViT-B | ConvNeXt V2-B |
| --- | --- | --- |
| DDP speedup over single | 2.64× | 3.32× |
| FSDP2 penalty over DDP | +35% | +84% |

DDP falls short of the ideal 4× on four GPUs. Measured separately on this host for ViT-B, the
smaller per-GPU batch accounts for 3.7% (3474 img/s at 128 versus 3606 at 512), gradient
all-reduce for about 10% at two ranks (3133 img/s per rank against 3474 at the same per-rank
batch), and the rest is host-side JPEG decode: the four-rank run demands 7910 img/s against the
single run's 2925, while sharing the host with other jobs. ConvNeXt V2-B scales better precisely
because it is slower per image, so the same data pipeline covers a larger share of its step.

The FSDP2 penalty tracks block count rather than parameter count: ConvNeXt V2-B has 36 sharded
blocks against ViT-B's 12, so it pays three times as many all-gather / reduce-scatter rounds per
step. Removing the other jobs from the host barely moved it (7.75 average, 7.06 best), which
places the cost in communication rather than contention. Neither model needs the memory that buys
— ViT-B is 86M parameters — so per-block sharding only pays once a model no longer fits.

An earlier pass of the same matrix ran without compilation; those `StratCmp-` runs are kept as an
uncompiled reference (ViT-B DDP 0.6497, FSDP2 0.6454).

The vision matrix runs one seed per configuration, so unlike the language model it carries no
error bar: a small cross-strategy gap such as the 0.43 pp above cannot be separated from
run-to-run noise from the vision runs alone.

### Run-to-run spread is large early and small late

Two single-device ViT-B runs with identical code, configuration, seed, and data — verified by
`md5sum` on the generated model, learner, and loader, and by identical logged parameters — reached
**35.05% and 43.65% top-1 at epoch 10**. The recipe has no warmup and holds lr at 1e-3, so early
training is chaotic and bf16 with non-deterministic reductions is enough to separate two runs.
The same pair of runs agreed to four decimals on the epoch-1 training loss in one comparison and
differed by 0.038 in another, so the spread itself is not stable.

The practical consequences:

- Mid-training cross-strategy comparisons are meaningless here. An epoch-10 gap of several points
  says nothing about the strategy.
- The spread shrinks as training proceeds: the uncompiled arms sat at 59.58 / 59.03 / 59.74 by
  epoch 30 and 64.97 / 64.54 at epoch 90.
- Only final-epoch numbers are usable, and even those carry noise this matrix cannot quantify
  without replicates.

This also settles a false lead. An apparent 9 pp single-device regression under `torch.compile`
did not reproduce: with the flow-function compile at `cmd_torch.py:401` disabled the run reached
43.96%, and with it enabled 43.65%, both matching the uncompiled baseline. ConvNeXt V2-B is
numerically unaffected by compilation as well (epoch-10 top-1 0.668063 uncompiled vs 0.668058
compiled). Compilation changes speed, not what these models learn.

### Recipe

The commands set only `input_size`, `is_training`, `batch_size`, `num_workers`, and
`image_dtype`, so the augmentation is whatever `examples/torch/data.py` defaults to: random
resized crop (scale 0.08–1.0, ratio 3/4–4/3), horizontal flip at 0.5, colour jitter 0.4, bicubic
interpolation, ImageNet mean/std. Everything else is off — no AutoAugment or RandAugment, no
random erasing, no mixup or cutmix, no label smoothing, no augmentation repeats. Validation is
resize plus centre crop at `crop_pct` 0.875.

There is also no scheduler in `cfg/torch/learners/ImageClassifier.yaml`: lr is held at 1e-3 for
all 90 epochs with no warmup.

That is a 2016-era baseline recipe, not the ConvNeXt or DeiT one (RandAugment, mixup 0.8,
cutmix 1.0, label smoothing 0.1, random erasing 0.25, stochastic depth, cosine schedule with
warmup). It is why top-1 lands near 64% rather than the ~81–84% those papers report, and it is
also why early training is unstable enough to produce the spread documented below. None of this
affects the comparison — all arms share one loader and one learner — but the numbers are not
model-quality figures.

### Why ConvNeXt V2-B costs 4× ViT-B per epoch

Uncompiled, ConvNeXt V2-B ran 34.0 min/epoch against ViT-B's 8.6 at the same global batch. It is
not memory format — forcing `channels_last` changes nothing (240.9 → 239.7 ms/step at batch 64),
because cuDNN already picks NHWC internally (`dgrad2d_c1_k1_nhwc_specialized` in the profile).
It is not FLOPs either: ConvNeXt V2-B has fewer than ViT-B (15.4 vs 17.6 GFLOPs).

`torch.profiler` attributes ConvNeXt V2-B's CUDA time roughly as 67% elementwise kernels, 7%
reduce, 6.5% layer_norm, 6% conv dgrad/wgrad, 2.5% tensor-core GEMM. The model is
memory-bandwidth bound: each of its 36 blocks walks the whole activation tensor about ten times
(depthwise 7×7 conv → permute → LayerNorm → Linear → GELU → GRN → Linear → permute → DropPath →
residual), on activations larger than ViT's token tensors, while ViT-B has 12 blocks whose cost
sits in four large GEMMs.

Compilation fuses exactly those chains, which is why it pays off so asymmetrically:

| model | `compile=False` | `compile=True` | speedup |
| --- | --- | --- | --- |
| ConvNeXtV2-B | 241.4 ms/step | 90.7 ms/step | 2.66× |
| ViT-B | 55.0 ms/step | 43.0 ms/step | 1.28× |

Measured at batch 64 on a GPU concurrently running another job, so absolute values carry
contention; both arms saw the same contention, so the ratios hold. Compilation is the reason the
matrix was relaunched — a convolutional model built from these blocks should not be trained
uncompiled.

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
