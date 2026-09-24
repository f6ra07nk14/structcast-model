# Framework parity and training features: full training runs on DGX H200

Do generated flax and keras learners train a model to the same place as generated torch learners, and
do gradient checkpointing, gradient accumulation, mixed precision, weight averaging (EMA) and tensor
parallelism leave what a model learns unchanged? A two-tier campaign on a shared 8×H200 host
answered this for PR https://github.com/f6ra07nk14/structcast-model/pull/32: nine short verification
experiments first, then 39 full trainings.

**Answer: yes, with three limits.** Of the 39 cells, **35 pass, none fail, and 4 were skipped**.
Every framework reaches its torch twin inside the grading band, or outside it only where an input
pipeline difference is documented. The same holds for every feature: each one reproduces the
model's own baseline. The limits are:

- keras/tensorflow cannot accumulate gradients under `MirroredStrategy`.
- keras refuses gradient checkpointing over a `Dropout` with rate > 0.
- torch refuses an EMA over a sharded model: under FSDP2
  (https://github.com/f6ra07nk14/structcast-model/issues/33) and under tensor parallelism.

The per-cell record, with a verdict comment for every row, is the
[canonical inventory](https://github.com/f6ra07nk14/structcast-model/pull/32#issuecomment-5483211067).
This page is the summary.

## Setup

- **Hardware.** One 8×H200 host (143,160 MiB addressable per card), shared with other tenants for the
  whole campaign. Throughput figures below are therefore reference only and carry no claim.
- **Images.** Built from `docker/train.dockerfile` per framework, plus one image with all extras
  (`all-cuda`) for the flax and keras runs, because their image examples import TensorFlow for data.
  Stack at the end of the campaign: keras 3.15.1, TensorFlow 2.21.0.
- **Data.** ImageNet-1k class-per-folder tree, streamed by the example loaders. Tiny Shakespeare for
  the language model and horse↔zebra for CycleGAN.
- **Vision recipe.** Global batch **512** in every vision cell: per-device batch and accumulation vary
  to hold it. 100 epochs, constant lr 1e-3, weight decay 0.05, no warmup and no schedule, seed 42.
  - This minimal recipe matches
    [strategy-comparison-h200.md](strategy-comparison-h200.md). The absolute ViT-B/16 top-1
    (~0.65) is therefore far below a tuned ViT-B ImageNet number and should be read only against
    its own control.
- **Precision.** bf16 unless the cell varies it: torch autocast, a keras mixed-precision policy, and a flax
  model-level `dtype`.
- **Code under test.** Tier 1 closed at `d195d7f`. Tier 2 ran on staged trees that advanced with the
  fixes it drove, the last at `243e877`. The review-round commits `13e1c0f` → `99297eb`
  (2026-09-18 → 09-20) landed after the last run and were not re-run on hardware; CPU CI covers
  them.

## How cells were graded

- **Primary metric.** Validation top-1 against the cell's named control. Later verdicts read it at
  epoch 90, where the published reference was also taken; earlier verdicts read epoch 100. Each
  table below says which. The bands are:
  - under 1.0 pt: **PASS**
  - 1.0–2.0 pt: **PASS with annotation**
  - over 2.0 pt: **FAIL**
- **Pipeline carve-out.** In the framework family, a gap with a documented input-pipeline attribution
  does not fail a row; grading falls back to correct execution. The carve-out covers residual
  differences between two faithful pipelines, never a pipeline defect: a defect is fixed and the
  row re-run.
- **fp16 rows** are judged on the loss-scale trajectory and on whether the loss moves, never on
  step counts. Tier 1 showed why: a run that overflows every step skips every apply while its
  counter still advances.
- **Health-only rows.** Rows 13, 18 and 39 (keras/tensorflow) are judged on execution health only:
  finite training, exact counters, sane convergence. They carry no accuracy verdict.
- **Counters** (`epoch`, `step`, `update`) must land exactly on the value predicted before the run.

## Tier 1 — verification (9/9 PASS at `d195d7f`)

Short runs on a 10-class ImageNet subset and Tiny Shakespeare, gating tier 2.

| # | experiment | result |
| --- | --- | --- |
| 1 | torch regression baseline (ConvNeXtV2 + ViT, single / DDP / resume) | counters exact; resumed epochs within 0.004 of uninterrupted |
| 2 | flax / keras parity sweep | epoch-3 validation spread 0.052 nats across torch, flax and keras on jax / tensorflow / torch |
| 3 | gradient checkpointing on / off | peak VRAM −46.8 % … −66.0 %; torch loss bit-identical, others ≤ 7e-4 |
| 4 | accumulation k=4 vs k=1 at equal global batch | `update == step / 4` exactly on all three mechanisms |
| 5 | mixed precision | forced fp16 overflow skips 40/40 applies with the loss frozen |
| 6 | EMA (torch / flax) | resume continues the blend within 0.002 |
| 7 | tensor parallel tp=2 / tp=4 vs single | parity 0.0007 (flax) … 0.043 (torch) |
| 8 | 2-D `fsdp_tp` (2,2) | per-GPU memory −14.1 % (torch) / −18.7 % (flax) against pure TP at equal global batch |
| 9 | distributed regressions and cross-strategy resume | a DDP×4 checkpoint restores under FSDP2×4 and single device |

Seven defects surfaced here and were fixed before tier 2; see [Defects found](#defects-found). Five of
them were invisible to the CPU suite: each needs a real multi-device placement, a real fused kernel
or a real sharded batch to appear.

## Tier 2 — full training (35 PASS · 0 FAIL · 4 SKIPPED)

### Baselines — torch

| row | model | setup | result | reference |
| --- | --- | --- | --- | --- |
| 1 | ConvNeXt V2-Base, ImageNet-1k | single GPU, bf16 | top-1 0.726784 @90, **0.731282 @100** | 0.7284 @90 in [strategy-comparison-h200.md](strategy-comparison-h200.md): −0.16 pt |
| 2 | ViT-B/16, ImageNet-1k | single GPU, bf16 | top-1 0.644815 @90, **0.645313 @100** | 0.6404 @90, same reference: +0.44 pt |
| 3 | SmallLanguageModel (25.3 M), Tiny Shakespeare | single GPU, fp32 | best val loss **1.4921** @7 | historical numbers |
| 4 | CycleGAN, horse↔zebra | single GPU, fp32, 200 epochs | generator loss **1.8838**, loss composition exact to 2e-4 | historical numbers |

### Framework — same recipe, different framework

| row | framework | model | result | control | Δ | verdict |
| --- | --- | --- | --- | --- | --- | --- |
| 5 | flax | SLM | best val loss 1.4837 | row 3: 1.4921 | 0.56 % better | PASS |
| 6 | keras / jax | SLM | best val loss 1.5033 | row 3: 1.4921 | 0.75 % worse | PASS |
| 7 | flax | CycleGAN | generator loss 1.7773 | row 4: 1.8838 | composition exact to 6e-4 | PASS |
| 8 | keras / jax | CycleGAN | generator loss 1.5559 | row 4 (health only) | composition exact | PASS |
| 9 | flax | ViT-B/16 | 0.661947 @90, 0.666730 @100 | row 2: 0.644815 @90 | +1.71 pt | PASS (pipeline carve-out) |
| 10 | keras / jax | ViT-B/16 | 0.662059 @90, 0.664631 @100 | row 2: 0.644815 @90 | +1.72 pt | PASS with annotation (pipeline carve-out) |
| 11 | flax | ConvNeXt V2-B | 0.729936 @100 | row 1: 0.731282 @100 | −0.135 pt | PASS |
| 12 | keras / jax | ConvNeXt V2-B | 0.731875 @100 | row 1: 0.731282 @100 | +0.059 pt | PASS |
| 13 | keras / tensorflow | ViT-B/16 | 0.659719 @100 (record only) | reference row | — | PASS (health) |

The three language models land within 0.0196 nats of each other. On ConvNeXt V2-B, flax and keras
land within 0.19 pt of torch and of each other. On ViT-B/16, flax and keras both sit about 1.7 pt
*above* torch, and within 0.1 pt of each other. The offset has four named input-pipeline
candidates and passes under the carve-out
([candidates](https://github.com/f6ra07nk14/structcast-model/pull/32#issuecomment-5481754315), [ruling](https://github.com/f6ra07nk14/structcast-model/pull/32#issuecomment-5481980796)). The control that would have split it among
them was deleted, so the attribution is unmeasured (see [Open items](#open-items)).

### Feature stack — checkpointing + accumulation + mixed precision on 4 GPUs

| row | framework | setup | result | control | Δ | verdict |
| --- | --- | --- | --- | --- | --- | --- |
| 14 | torch | DDP ×4, 32/GPU × accumulate 4, checkpointing, EMA | raw train top-1 0.668911 @100 | row 2: 0.668933 | −0.0022 pt | PASS |
| 15 | torch | FSDP2 per-block ×4, same bundle, no EMA | raw train top-1 0.670899 @100 | row 14: 0.668911 | +0.199 pt | PASS with annotations |
| 16 | flax | FSDP ×4, same bundle | 0.667874 @100 | row 9: 0.666730 | +0.114 pt | PASS |
| 17 | keras / jax | dp ×4, same bundle, stochastic depth off | 0.652059 @90 | row 10: 0.662059 | −1.000 pt | PASS with annotation |
| 18 | keras / tensorflow | MirroredStrategy ×4, checkpointing, **no accumulation**, stochastic depth off | 0.650049 @100 (record only) | row 13 (health only) | — | PASS (health) |

- **Counters.** The torch rows land exactly on the predicted
  `{epoch 100, step 1000900, update 250225}`: four micro-steps per update.
- **Rows 14 and 15 are compared on training metrics,** because row 14 validates its EMA shadow
  rather than the raw model.
- **Rows 17 and 18 run with stochastic depth off**, while their controls keep the ViT 0.1 ramp. The
  keras builder refuses checkpointing over a `Dropout` at rate > 0, so row 17's −1.000 pt carries
  two deltas and is annotated for it.
- **Row 18 has no accumulation.** It was dropped because accumulation under `MirroredStrategy`
  raises `merge_call` on keras 3.15.1 / TF 2.21.0. The tensorflow bundle is therefore
  checkpointing only.

### Tensor parallel — one model split across GPUs

| row | framework | strategy | result | control | Δ | cost |
| --- | --- | --- | --- | --- | --- | --- |
| 19 | torch | `tp` 2 | 0.650572 @100 | row 2: 0.645313 | +0.53 pt | 1.81× wall clock for ~13 % less memory per GPU |
| 20 | flax | `tp` 2 | 0.661493 @90 | row 9: 0.661947 | −0.045 pt | 1.79× GPU-hours |
| 21 | keras / jax | `tp` 2 | 0.663465 @90 | row 10: 0.662059 | +0.141 pt | 2.71× s/step |
| 22 | torch | `fsdp_tp` (2,2) ×4 | 0.648338 @100 | row 2: 0.645313 | +0.30 pt | — |
| 23 | flax | `fsdp_tp` (2,2) ×4 | 0.668270 @100 | row 9: 0.666730 | +0.154 pt | — |

All five pass. Splitting a model does not change what it learns in any framework. keras has no 2-D
combination, and tensor parallelism on keras/tensorflow is refused by design. At ViT-B/16 size,
which fits one card, the split costs far more time than it saves memory.

### Precision — does the number format change the answer?

| row | framework | setup | result | control | Δ | final loss scale |
| --- | --- | --- | --- | --- | --- | --- |
| 24 | torch | fp16 + GradScaler, single | 0.648336 @100 | row 2 (bf16): 0.645313 | +0.30 pt | 2²⁰, held all run |
| 25 | flax | fp32, single | 0.664972 @100 | row 9 (bf16): 0.666730 | deltas flip sign, within ±0.55 pt | — |
| 26 | flax | fp16 + `DynamicScale`, single | 0.664170 @90 | row 9 (bf16): 0.661947 | +0.222 pt | 2¹⁹ |
| 27 | torch | fp16, DDP ×4 | 0.646542 @100 | row 24: 0.648336 | −0.179 pt | 2²⁰ → 2¹⁸ |
| 28 | torch | fp16, FSDP2 ×4 | 0.651275 @100 | row 24: 0.648336 | +0.294 pt | 2¹⁸ |
| 29 | flax | fp16, FSDP ×4 | 0.666870 @90 | row 26: 0.664170 | +0.270 pt | 2¹⁹ |
| 30 | keras / jax | fp16 + `LossScaleOptimizer`, single | 0.663998 @90 | row 10 (bf16): 0.662059 | +0.194 pt | 2²⁰; 114 of 250,200 updates skipped |
| 31 | keras / jax | fp16, dp ×4 | 0.666046 @90 | row 10 (bf16): 0.662059 | +0.399 pt | 2¹⁹; 110 of 250,200 updates skipped |

All eight pass. fp16 with dynamic loss scaling converges like bf16 on every framework, on one device
and on four. On torch and keras the four-device arm settles one to two halvings below the
single-device arm. On flax, single device and FSDP ×4 settle on the same 2¹⁹.

- **Skipped updates.** Only keras reports them. There `update` is read back from the optimizer's
  own counter.
- **torch and flax counters record intent.** An update whose apply was skipped is still counted,
  so the scale trajectory is the evidence there.
- **fp32 costs time, not accuracy.** On flax it takes 1.67× the wall clock of bf16 for no accuracy
  change.

### Weight averaging (EMA) — averaging as the only variable

| row | framework | strategy | EMA val top-1 | control (raw, no averaging) | Δ | verdict |
| --- | --- | --- | --- | --- | --- | --- |
| 32–35 | torch / flax / keras-jax / keras-tf | single GPU | — | — | — | SKIPPED |
| 36 | torch | DDP ×4 | 0.727077 @100 | row 2: 0.645313 | +8.18 pt | PASS |
| 37 | flax | FSDP ×4 | 0.727857 @100 | row 9: 0.666730 | +6.11 pt | PASS |
| 38 | keras / jax | dp ×4 | 0.723929 @90, 0.728595 @100 | row 10: 0.662059 @90 | +6.187 pt | PASS |
| 39 | keras / tensorflow | MirroredStrategy ×4 | 0.722994 @100 (record only) | row 13 (health only) | — | PASS (health) |

- **Same place, three implementations.** Averaging at momentum 0.999 lands the three frameworks'
  EMA models within 0.15 pt of each other at epoch 100 (0.7271, 0.7279, 0.7286), though the torch
  and flax raw baselines are 2.1 pt apart. On row 36 the EMA lags the raw model for one epoch and
  overtakes it by epoch 2.
- **Row 37 is the only cell that averages over sharded parameters.** torch refuses an EMA under
  FSDP2 and under tensor parallelism (see [Limits](#limits)), so its FSDP2 + EMA cell was removed.
- **Why rows 32–35 were skipped.** They would have repeated the check on a single device, after
  rows 36–39 had already covered every framework
  ([skip record](https://github.com/f6ra07nk14/structcast-model/pull/32#issuecomment-5679198514)).
- **What `val_*` measures.** In these rows `val_*` measures the averaged model, while the training
  metrics measure the raw model.

## Defects found

### Tier 1 — fixed before tier 2

| # | defect | fix |
| --- | --- | --- |
| 1 | flax `nnx.Embed` gather fails under an Explicit data axis; SLM unrunnable | `0dff549` |
| 2 | flax ViT class-token concat fails under an Explicit data axis; ViT unrunnable | `0dff549` |
| 3 | flax templates used tanh-approximate GELU where torch and keras use exact erf | `dbf6404` |
| 4 | keras SLM was a different architecture (MHA + learned positions, no RoPE) | `f6860cb` |
| 5 | torch TP crashed on a param group mixing DTensor and plain tensors under the foreach optimizer | `ee6ece2` |
| 6 | keras/tensorflow `dp` scaled the loss **and** the gradients by 1/N | `d195d7f` |
| 7 | keras/jax checkpointing: the flash-attention kernel raised inside `keras.remat` | `43d1f32` |

### Tier 2 — fixed during the campaign

| # | defect | fix |
| --- | --- | --- |
| 1 | the flax / keras directory loaders shuffled only inside a 1024-item buffer over a class-sorted listing, so a batch held 2–5 classes; flax ConvNeXt went NaN from epoch 1 | `b888232`: path-level shuffle, 116–123 classes per batch |
| 2 | classifier precision and the showcase `EMA` block were literals outside `PARAMETERS`, so the fp16 cells and FSDP2-without-EMA could not be expressed | `84c77a6` |
| 3 | the flax ViT `dtype` override reached 50 of 55 sites, and flax ConvNeXt V2 had no `dtype` knob | `c4cc509`: `-p "base: {dtype: ...}"` reaches every layer |
| 4 | the keras image pipeline only rescaled: no mean/std normalization, no crop, an aspect-distorting resize, and no parallelism knob | `e340d9e` |
| 5 | flax ConvNeXt V2 `DropPath` broadcast over the wrong axes for NHWC, raising at `drop_path_rate > 0` | `42e0d95` |
| 6 | the flax / keras GRN gradient was NaN on a zero-norm channel; first read as "flax needs gradient clipping", refuted when a probe NaN'd with clipping | `9cb6c82` ([diagnosis](https://github.com/f6ra07nk14/structcast-model/pull/32#issuecomment-5579292644)) |
| 7 | the flax / keras GRN scale was one-initialized where timm zero-initializes it; flax sat at chance | `243e877` |
| 8 | the flax showcase rendered a null accumulation window as the string `'None'` into `optax.MultiSteps` (latent) | `ca19a2f` |
| 9 | keras bf16 logits were rejected by `in_top_k` in the criteria | `cf26bdc`, `2f56391` |
| 10 | torch `clip_grad_norm` selected the p of the norm with the bound fixed at 1.0, where flax and keras read it as the L2 threshold | `1d55e6a` (breaking) |
| 11 | `--compile` meant something different on each command | ADR-0024, `381ded6` … `13895a8` (breaking) |

## Limits

- **torch refuses an EMA over a sharded model.** The generated `__init__` raises a `ValueError`
  instead of degrading silently, for a different reason under each strategy:
  - Under FSDP2, `fully_shard` forbids the copy an `AveragedModel` takes. Tracked in
    https://github.com/f6ra07nk14/structcast-model/issues/33.
  - Under tensor parallelism, the copy succeeds, but `torch._foreach_lerp_` refuses the mixed
    DTensor / plain parameter list (`b02c1ef`).
  - flax averages under FSDP without restriction (row 37).
- **keras refuses gradient checkpointing over a `Dropout` at rate > 0.** The refusal is correct:
  - on jax the combination raises `UnexpectedTracerError`;
  - on the tensorflow and torch backends it returns silently wrong gradients;
  - torch and flax are unaffected
    ([ruling](https://github.com/f6ra07nk14/structcast-model/pull/32#issuecomment-5564617988)).
- **keras/tensorflow cannot accumulate under `MirroredStrategy`.** The accumulation `ops.cond` makes
  `merge_call` illegal on keras 3.15.1 / TF 2.21.0.
- **keras tensor parallelism is jax-only**, and it shards attention only, leaving the MLP
  replicated. keras has no 2-D `fsdp_tp`.
- **keras/tensorflow `--compile` maps to `tf.function` without XLA.** It runs 5.7× slower per step
  than keras/jax on the same model: 1.2482 against 0.2178 s/step
  ([diagnosis](https://github.com/f6ra07nk14/structcast-model/pull/32#issuecomment-5628527269)).

## Performance and memory (reference only)

The host was shared throughout, so none of these figures is a throughput claim.

- **ViT-B/16 memory, bf16, batch 512, one card** (reserved):
  - torch 41,508 MiB
  - flax 70,270 MiB
  - keras / jax 70,276 MiB
- **ViT-B/16 step time.** flax and keras / jax run 1.61× and 1.53× torch on a freed host (1.88× and
  1.86× under contention). The attribution was never redone after the host freed up.
- **torch `tp` 2.** Peak allocated memory per rank is about 13 % lower (29.66 GiB, extrapolated from a
  probe, against 34.10 GiB) at 1.81× the wall clock. An earlier "TP costs memory" reading was retracted: it read reserved memory
  ([probe](https://github.com/f6ra07nk14/structcast-model/pull/32#issuecomment-5396763601)).
- **flax fp32.** It takes 1.67× the wall clock of bf16 and at least 1.29× the memory.
- **flax ConvNeXt V2-B, batch 512, one card.** The compiled training step peaks at 99,729 MiB,
  leaving about 43 GiB spare. Buffer donation saves 4.92 GiB.
- **Gradient checkpointing** was not re-measured at full scale. Tier 1 measured −46.8 % to −66.0 %
  peak VRAM.

## Open items

- **Row 36's raw-model offset.** Row 36's raw-model training top-1 sits 1.03 pt above row 2, where
  row 14 matches row 2 to 2.2e-5. Row 38 does not reproduce it (−0.123 pt at epoch 90). Unexplained
  ([verdict](https://github.com/f6ra07nk14/structcast-model/pull/32#issuecomment-5535749060)).
- **The ViT framework offset.** flax is +1.71 pt and keras / jax +1.72 pt over torch at epoch 90,
  while each framework's own rows cluster tightly. The offset passes under the pipeline carve-out,
  but it was never measured, because the interpolation control was deleted.
- **Row 30's train/validation split.** keras fp16 trails its bf16 control by 0.625 pt in *training*
  top-1 while leading in validation. Row 31 shows the same sign, smaller (−0.237 pt). Unexplained.
- **flax inference step time.** A flax inference step measured 3.17× torch's; the cause was never
  attributed.

## Operational notes

- **MLflow file store.** On current MLflow a file-store tracking URI needs
  `MLFLOW_ALLOW_FILE_STORE=true` (documented in `examples/README.md`). If rank 0 fails on it before
  `fit`, the other ranks surface a 10-minute NCCL collective timeout, and the traceback is lost
  ([defect](https://github.com/f6ra07nk14/structcast-model/pull/32#issuecomment-5493614620)). Launchers should also export `MLFLOW_TRACKING_URI` explicitly.
- **Bound TensorFlow's threads in the flax and keras input pipelines.** TensorFlow sizes its pools
  to the host:
  - One unbounded flax run spawned 1,737 threads and slowed co-resident runs by 31–34 %.
  - A CPU quota alone throttled it 2.83×.
  - `TF_NUM_INTRAOP_THREADS=16` and `TF_NUM_INTEROP_THREADS=4` with a 32-CPU budget removed the
    throttling ([fix](https://github.com/f6ra07nk14/structcast-model/pull/32#issuecomment-5415983036)).
- **Size-group overrides.** Use `-p "base: {...}"` for size-group parameters such as `dtype` and
  `drop_path_rate`; `SHARED:` does not reach the size group.
- **keras dataset configurations need `crop_pct`.** Without it they take the small-image path, and
  the first epoch lands near chance ([staging record](https://github.com/f6ra07nk14/structcast-model/pull/32#issuecomment-5567216862)).
- **Read memory from the allocator, not `nvidia-smi`.** `nvidia-smi` shows reserved memory. XLA
  with `XLA_PYTHON_CLIENT_PREALLOCATE=false` still grows greedily, and the `XLA_PYTHON_CLIENT_*`
  variables do not cap TensorFlow.
