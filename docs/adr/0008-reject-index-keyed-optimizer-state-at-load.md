# Index-keyed optimizer state is rejected on the name-keyed load path

A training state saved without a pairing keys optimizer state by parameter position (`0`, `1`, …).
When such an artifact is resumed by a learner that does declare a pairing, the load routes into
`set_optimizer_state_dict` (the load path ADR-0005 defines), which matches entries by parameter
name and cannot resolve positions.
On torch 2.13 the outcome depends on the optimizer: state containing a 0-dim tensor (Adam's `step`)
crashes with an opaque `KeyError: 'state.0.step'`; state without one (SGD momentum) is silently
discarded — training resumes with fresh moments and the next save crashes. Under FSDP2 the same
crossing installs unsharded tensors beside DTensor parameters and corrupts state without an error
(issue #24).

## Decision: detect the layout and refuse it

`_set_optimizer_state` now raises `ValueError` when every `state` key is an `int` — the same
predicate torch itself uses to classify the layout — naming the cause (saved without a pairing) and
the remedy (resume from a state saved with the pairing, or restart training). The check lives in the shared
state-dict mixin, so every strategy fails the same way, including FSDP2 where the silent-corruption
case was worst. Digit-string keys (a layout no `torch.save` round-trip produces) fall outside the
predicate; the default strict load still rejects them loudly. A pre-pairing save of a stateless
optimizer carries an empty `state`, passes the predicate, and restores hyperparameters positionally
— accepted as a residual: it is the existing no-pairing behavior and has no moments to corrupt.

## Rejected: restoring the state anyway

Two working alternatives were prototyped and verified before being rejected:

- **Plain positional fallback** — route the index-keyed dict through `optimizer.load_state_dict`
  (bit-identical restore on single-device and DDP; must still refuse FSDP2, where positional load
  cannot reshard).
- **Index→name conversion** — rekey the dict against the pairing's container and proceed through
  DCP (correct on all strategies including 2-rank FSDP2; PyTorch Lightning ships this shape via
  `FSDP.rekey_optim_state_dict`).

Both were rejected on policy, not feasibility: training states are not supported across a pairing
change — every learner this package generates declares `optimizer_models`, so index-keyed state can
only reach this path from a pre-pairing artifact or a hand-edited learner. Either restore would
also inherit the layout's positional blindness: a reordered-but-same-shape parameter set restores
the wrong moments with no error, which is worse than refusing. If pre-pairing artifacts ever need
a migration path, the conversion prototype is the one to revive.

## Related: partial optimizer state is accepted on request

The name-keyed path passed torch's default `strict=True`, which rejects any training state whose
optimizer entries do not cover every trainable parameter — yet a parameter that has not been
stepped yet (frozen phase, warmup branch) legitimately has no state, so a normal mid-run save of
such a learner could never be resumed (upstream: pytorch/pytorch#164257, #192202). Strategies now
expose `strict_optimizer_load: bool = True`: the default keeps the loud failure; setting it to
`False` accepts any coverage — a name-keyed state matching nothing loads as a silent no-op, so the
flag is strictly an opt-in for deliberately partial states — and uncovered parameters keep the
zeroed state torch materializes
before loading (for Adam-like optimizers that includes a step counter already at 1, so their first
real update is bias-corrected as a second step). Missing state is never synthesized by this
package — zero-filled placeholders are the documented failure mode of that idea (VeOmni).
