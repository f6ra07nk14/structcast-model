# Generated code owns gradient-sync gating; compile boundaries are the flow functions

This ADR records how generated learner code cooperates with distributed wrappers (ADR-0003) and where
`torch.compile` applies. It changes what `scm torch create learner` emits and deletes two mechanisms:
`TorchTrainer.no_sync` and the CLI's step-closure compilation.

## Sync gating lives in the generated step, nowhere else

Whether a backward participates in gradient synchronization needs per-forward granularity: a
multi-optimizer learner forwards non-owned models inside another optimizer's segment (a discriminator
scored during the generator's loss) and forwards the same model twice before one backward (real and fake
batches) — both violate DDP's "one forward, one backward, every parameter reduced exactly once" contract if
the reducer arms on every call. A trainer-level context (`TorchTrainer.no_sync`) sees only whole steps and
cannot express this, so it is deleted rather than kept alongside the finer mechanism: two owners of the
same DDP flag, split across layers, is the exact pathology that made the original no_sync dead code.

Generated steps wrap every model invocation in `sync_gate(model, armed)` (from
`structcast_model.torch.distributed`, which generated code may import — precedent: generated learners
already import `get_decays`). The arming rule is computed at code-generation time:
`armed = model owned by the current optimizer segment AND last invocation of that model in the segment`,
multiplied at runtime by `__need_update__`, which also subsumes gradient-accumulation gating — StyleGAN2-ADA's
`ddp_sync` discipline, generated instead of hand-written. Single-device models get `nullcontext`, keeping
the gate free and traceable.

## Optimizer segments freeze the models they do not own

Segment presets emit `requires_grad_(model in segment's trainable layers)` next to the existing
`train()`/`eval()` toggles. `.eval()` does not stop autograd, so before this, an earlier segment's backward
deposited gradients into later segments' models — under CycleGAN, the generator's adversarial gradients
(pushing discriminators to score fakes as real) accumulated into the discriminator updates every iteration,
single-device included (issue #15). Freezing fixes that while gradients still flow *through* frozen models
to their inputs, preserving adversarial semantics; under DDP it also keeps frozen models' reducers from
waiting for gradients that never come.

## FLOW and INFERENCE_FLOW are extracted into compilable functions

Each optimizer segment's pure computation (model calls, criteria, arithmetic) becomes a `_flow_<optimizer>`
function; the inference flow becomes `_flow_inference`. `model.train()/.eval()`, `requires_grad_`,
`loss.backward()`, `optimizer.step()`, and `zero_grad()` stay outside in the eager step — every one of them
is a graph break or a guard churn source inside `torch.compile`, which is why compiling whole step closures
performed poorly. Backward needs no special treatment: AOTAutograd compiles the backward of a compiled
forward region even though `.backward()` is called outside it. The functions are `self`-assigned so the CLI
can rebind them compiled (`learner.flow_functions` lists them); cross-segment values thread through
parameters and returns, derived from the flow's dataflow at generation time.

The CLI composes compilation as `strategy.wrap(strategy.compile(model))`: the wrapper stays outermost, so
`isinstance` checks and the sync gate keep seeing real DDP/FSDP2 types, and checkpoint keys are handled by
the DCP state-dict API, which strips both `module.` and `_orig_mod.` prefixes. The previous model-level
compile (applied outside the wrapper, into a discarded local) and the step-closure compile are deleted.
Under distributed execution the flow-function graphs break at wrapper boundaries, and measurement settled
the follow-up (`docs/references/flow-compile-step-time-h200.md`): compiling the fragments is a net loss
(+8.9% training, +60% inference step time on DDP 2×H200) while the single-device fusion win is real
(−14.8% training), so flow functions compile only on a single device. Generated steps import the sync gate
from `structcast_model.torch.distributed`, which for that reason is the one module NOT self-replaced by the
package's lazy-import shim — the shim raises on dunder lookups and breaks dynamo's tracing of anything that
calls into the module from a compiled region.
Per-block compilation and per-block `fully_shard` build on a structural seam the templates already have:
configs that nest layers via `TYPE`/`CFG` generate real per-block submodule classes. The FSDP2 strategy's
`shard_modules` names the units — glob patterns over `named_modules()` paths whose `*` never crosses a
`.`, because fnmatch semantics would match a block's whole subtree and shard every leaf as its own group —
and its wrap shards matched modules descendants-first with the root last, so the root group holds exactly
the leftovers. Where the compile units sit is the strategy's decision, not the CLI's: `strategy.compile`
compiles the model root in place by default, and the FSDP2 strategy overrides it to compile the matched
submodules instead, so compile-unit boundaries follow the shard boundaries and FSDP2's per-block hooks stay
out of the compiled graphs. The CLI only invokes it. Unset, each model stays one group and one compile
unit, as before.
