# The Learner owns the training counters

`update(step)` was a pre-step prediction: the trainer incremented its own `step` field, asked the
learner "will the step about to run land an apply", ran `training_step`, and incremented its own
`update` field when the answer was yes (`update_models` in `base_trainer.py` returned the pair left
to right). The counters were trainer-owned `BaseInfo` fields, written into checkpoint meta on save
but never restored — all three resume paths read back only the epoch, so `step` and `update`
restarted at zero on every resume. ADR-0017 kept that shape, rejecting a post-hoc read-back partly
because it "forces a per-step device sync on the jax paths". That cost does not exist: the trainer
already calls the tracker after every training step, and the tracker converts the step's criteria
to host NumPy on every backend — the `KerasTrainer` docstring states outright that this "already
waits for the step's computation on every backend". The sync the rejection priced in is one the
loop has always paid. With that objection gone, the counters move to the object that actually
knows the answer: the learner.

- **The learner owns three counters**, exposed as read-only properties: `steps` (completed Steps in
  the glossary sense — batch iterations), `updates` (completed Updates — optimizer applies), and
  `has_updated` (whether the just-finished step landed an Update — retrospective, the
  `optax.MultiSteps.has_updated` sense). The name `updated` was rejected: it sits one character
  from `updates`, and because `bool` subclasses `int` the typo survives type checking.
- **`update(step)` is deleted.** `update_models` becomes: run `training_step`, then read
  `has_updated`. Every count is a retrospective "completed" count. One contract change follows:
  during `on_training_step_begin` for step N, `info.step` now reads N-1 rather than N. No in-repo
  callback reads `step` at step-begin, so nothing in the package observes the shift; external
  callbacks that did are on notice.
- **`BaseInfo.step` / `BaseInfo.update` stay**, as read-only properties delegating to the learner.
  The old names are kept deliberately: checkpoint meta keys, `on_update` consumers and save gates
  all keep working with no migration.
- **`restore_counters(steps, updates)` joins the protocol**, closing the saved-but-never-restored
  hole: the three resume paths call it after loading state.

## Where each counter comes from

`steps` is a learner-maintained host counter on every backend. It cannot be derived from any
optimizer counter: when a keras `LossScaleOptimizer` skips on non-finite float16 gradients, the
inner optimizer's raw counter freezes — but the batch iteration still happened, and a glossary Step
must count it. Only the learner sees every `training_step` call.

`updates` and `has_updated` come from wherever the apply decision actually lives:

- **torch** — the learner's own gate arithmetic, moved inside `training_step`: increment
  `self._steps`, gate on `(self._steps + 1) % k == 0`. Under the trainer's old 1-based clock this is
  cadence-identical to the previous `(step + 1) % k`, so the historically short first window is
  preserved. `GradScaler` skips keep the intent semantics ADR-0017 accepted for torch: the gate
  reports that an apply was attempted, not that the scaler let it land.
- **keras** — a post-step read of the inner optimizer's private counter: `updates = raw // k`, and
  `has_updated` by delta against the previous read. This turns ADR-0017's prediction into genuine
  detection: the accepted edge where the pre-step read "cannot foresee the NaN" and misreports for
  one skip step disappears, because the question is now asked after the answer exists.
- **flax** — a post-step read of `MultiStepsState.gradient_step`, again by delta. Without
  `MultiSteps` there is no window: `updates` equals `steps` and `has_updated` is always true.
  `accumulation_window()` survives only for the init-time uniform-window validation of ADR-0017;
  its role in the gate formula is gone.

Torch's DDP synchronization gate needs no `arm()` / `update()` protocol split, the shape ADR-0017's
rejection assumed: `__need_update__` was always a plain per-call argument into the compiled flow
functions, so the compiled code is untouched — only the eager bind that fed it from
`self.need_update` moves inside `training_step`, next to the arithmetic that produces it.

## Resume seeds the learner

The three resume paths (`commands/cmd_torch.py`, `keras/trainer.py`, `flax/trainer.py` — which
today restore only the epoch) call `restore_counters(steps, updates)` with the counts from
checkpoint meta. Torch applies both, since both live only on the host. Keras and flax apply
`steps` only: their `updates` self-restores through the optimizer state that checkpointing already
round-trips (the inner keras counter, the `MultiStepsState`), and seeding it separately would
invite the two sources to disagree. Keras additionally re-baselines its cached last-read `updates`
at restore, so the first post-resume delta does not misfire.

## Considered options

- Keeping the pre-step `update(step)` protocol — ADR-0017's own stance. Rejected: its load-bearing
  cost argument (the per-step device sync) is a sync the tracker already performs every step on
  every backend, and the post-step read is strictly more truthful under float16, where prediction
  provably misreports on skip steps.
- Deriving `steps` from the optimizer counters, keeping the learner stateless. Rejected: the
  float16 loss-scale skip freezes the keras raw counter while the Step still happened, so the
  optimizer can never yield the glossary Step count.
- The name triple `steps` / `updates` / `updated`. Rejected: `updated` is one character from
  `updates` while `bool` subclasses `int`, so swapping them type-checks clean and miscounts
  silently.

## Consequences

- Breaking `Learner` protocol change, with no shim: every implementer migrates — generated
  learners, hand-written user learners, and test fakes — from `update(step)` to the three
  properties plus `restore_counters`.
- Tests that assigned `trainer.step` or `info.step` directly must seed the learner's counters
  instead; the `BaseInfo` fields are now read-only views.
- ADR-0017 is superseded in part: its "`Learner` protocol and `base_trainer` are untouched —
  `update()` keeps its pre-step position" clause and its rejection of the post-hoc read-back
  considered option no longer hold. The rest of ADR-0017 stands: the per-backend native window
  mechanisms, the choice of counter clocks, and the uniform-window validation are unchanged.
- Counters now survive resume; previously they were saved into meta and silently discarded.
