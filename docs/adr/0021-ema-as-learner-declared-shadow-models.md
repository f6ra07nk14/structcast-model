# EMA instances are learner-declared shadow models

> **Amended by torch 2.13 verification: the DTensor refusal is right, its recorded reason is only
> half of it.** "An `AveragedModel` cannot copy a DTensor-parameter module" is not what happens: a
> `DTensor` deep-copies fine, and the two strategies fail for two different reasons. Under FSDP2 the
> copy is what fails — `fully_shard` swaps the module's class for a synthesized `FSDP<Cls>` carrying
> a `__deepcopy__` that raises ("FSDP does not support deepcopy",
> `torch/distributed/fsdp/_fully_shard/_fully_shard.py:299-302`, bound onto the class at
> `_fsdp_init.py:425`). Under tensor parallelism the copy succeeds and the *blend* fails:
> `parallelize_module` replaces only the parameters of the modules its plan matched, and
> `get_ema_multi_avg_fn`'s `torch._foreach_lerp_` raises `RuntimeError: got mixed torch.Tensor and
> DTensor` on the mixed list every realistic plan leaves — at the second Update, not the first, which
> only seeds. The parameter-type check below therefore stays exactly as it is; only the explanation
> changes. Enabling either (a DTensor-safe `multi_avg_fn` for the tensor-parallel case, a
> clone-before-`wrap` assembly order for FSDP2) is tracked in
> [issue #33](https://github.com/f6ra07nk14/structcast-model/issues/33); everything else below is
> unchanged.

Exponential moving averages are declared at the top level of a torch or flax learner template —
`EMA: dict[str, bool | dict[str, Any]]`, keyed by model name — and emitted as named learner
attributes `ema_<model>`: `torch.optim.swa_utils.AveragedModel` over the (DDP-unwrapped) model on
torch, `flax.nnx.EMA` on flax with the `apply_to` view bound as the attribute so the averaged
weights are directly callable. The mapping values are the EMA constructor's keywords, resolved
through `resolve_getter` (a `multi_avg_fn` object pattern works). Keras declares nothing: its
optimizer already owns EMA (`use_ema` and friends in the `OPTIMIZER` pattern), with the recorded
limitations that it updates on every apply call including accumulation no-ops, must be configured
on the inner optimizer under a loss-scale wrapper, and offers no separate averaged copy to
validate against.

The update lands at the end of `training_step`, host-gated on the learner's own detection:
`if self._has_updated:` — once per real Update, never on an accumulation micro-step, after every
optimizer segment of the step has finished. On flax the update is a deliberate eager call rather
than part of the traced step: `nnx.EMA` captured by a step closure raises `TraceContextError`
under `nnx.jit` (verified), and threading it through the donated signature would change the step
contract for a per-update host call that costs nothing on the step path. Torch fp16 keeps the
intent semantics ADR-0018 records: a scaler-skipped apply still advances EMA.

`ema_<model>` is a first-class flow name: `INFERENCE_FLOW` may run it (validation over averaged
weights), the training `FLOW` may not (`SpecError`), and the reserved-name guard covers the
generated attributes. The EMA modules ride the generated `models` property, so the existing
strategy `state_dict`/`load_state_dict`, resume and best-checkpoint paths carry them with no
change to the training-state payload contract (ADR-0015) — `n_averaged` and the averaged weights
restore like any model's state.

FSDP2 is explicitly rejected for now: the models reach the learner already sharded, an
`AveragedModel` cannot copy a DTensor-parameter module, and the workable shape (clone before
`wrap`, shard both under one plan) belongs to the CLI assembly order — a follow-up issue, not a
silent degradation. The generated `__init__` detects DTensor parameters on an EMA'd model and
raises with that explanation.
