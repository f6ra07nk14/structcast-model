# Flow-function compilation: steady-state step time on DGX H200

Measurement backing the decision in ADR-0004's follow-up: generated `_flow_*` functions compile
only on a single device; under distributed execution they stay eager while the models themselves
remain compiled.

## Setup

- Hardware: DGX H200, driver 595.71.05; container `docker/train.dockerfile` (torch 2.11.0+cu130).
- Model: generated ConvNeXtV2 `atto` (`cfg/torch/models/ConvNeXtV2.yaml`, `num_classes=10`,
  unstructured output), generated learner from `cfg/torch/learners/ConvNeXtV2.yaml`
  (bf16 autocast, `AdamWWithCosine`).
- Data: synthetic `16×3×64×64` batches, fixed seed.
- Both arms compile the model itself (`torch.compile(model)` inside the strategy wrap); the only
  variable is whether the generated flow functions are additionally compiled.
- Metric: mean wall time of `learner.training_step` / `learner.inference_step` over 100 calls
  after 30 warmup calls (compilation excluded), `torch.cuda.synchronize` around the timed loop.
  DDP arms ran 2 ranks over NCCL via `torchrun --standalone`; rank 0 reported.

## Results (2026-08-13)

| Arm                       | train_step (ms) | infer_step (ms) |
| ------------------------- | --------------- | --------------- |
| single GPU, flow eager    | 15.950          | 3.632           |
| single GPU, flow compiled | **13.582** (−14.8%) | **3.337** (−8.1%) |
| DDP 2-GPU, flow eager     | **20.169**      | **3.823**       |
| DDP 2-GPU, flow compiled  | 21.970 (+8.9%)  | 6.130 (+60.3%)  |

## Reading

- Single device: compiling the flow fuses the loss/metric glue into one graph — a real win, kept.
- DDP: `DistributedDataParallel.forward` is a dynamo graph break, so the compiled flow shatters
  into fragments whose re-entry and guard overhead exceeds the glue-fusion gain on both the
  training and (worse) the inference path. Distributed flow therefore stays eager.
- Caveat: measured on a small model at 64px, where glue is a large share of the step. Larger
  models shift time toward the (still compiled) model body, which shrinks both the single-device
  win and the distributed loss — the decision's direction does not flip.

## Incidental findings (both fixed)

- Generated flow functions previously imported `sync_gate` through the package's lazy-import
  shim, which `torch.compile`'s tracer cannot introspect (`InternalTorchDynamoError` on
  `__class__`): flow compilation was broken everywhere until the gate became an inline helper in
  the generated script.
- `torch.compile` needs a host C/C++ toolchain at run time (triton builds its driver shim with
  `cc`); the training image now installs `gcc`/`g++`.
