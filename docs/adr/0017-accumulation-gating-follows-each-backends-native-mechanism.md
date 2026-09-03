# Accumulation gating follows each backend's native mechanism

> **Superseded in part by ADR-0018.** The "`update()` keeps its pre-step position" clause below and
> the rejection of the post-hoc read-back considered option are replaced by the learner-owned
> retrospective counters of ADR-0018. The per-backend window mechanisms stand.

> **Superseded in part by ADR-0019.** The keras private `_iterations` read (its phase was only
> needed by the pre-step prediction), the flax `accumulation_window` read-back in `__init__`, and
> the flax half of the "one learner, one window" validation are replaced by in-step /
> public-property detection on the first optimizer.

`ACCUMULATE_GRADIENTS` was a shared schema field on `UserDefinedLearner`, on the theory that gradient
accumulation is portable (ADR-0012). Practice disagreed per backend: torch derives a host-side
`(step + 1) % k` gate; keras mapped the field onto the optimizer's own `gradient_accumulation_steps`
but left `update()` returning constant true, accepting that the update counter advances per step
rather than per apply (ADR-0016); flax hand-rolled a params-shaped accumulator behind a static
`need_update` jit argument after rejecting `optax.MultiSteps` (ADR-0013). Three backends, three
half-agreements with one field. The field now lives only where the framework has no native window
mechanism — `TorchUserDefinedLearner` — and the keras and flax schemas carry no such field: each
backend declares the window through the mechanism its framework already owns, and gates `update()`
from it.

- **torch** — generated code unchanged: `update(step)` computes `(step + 1) % k` on the host, and its
  `need_update` side effect gates DDP gradient synchronization, clipping, `optimizer.step()` and
  `zero_grad()` inside `training_step`.
- **keras** — the window is `gradient_accumulation_steps` declared as a kwarg of the `OPTIMIZER`
  pattern, exactly where clipping already lives (ADR-0016); the learner reads the optimizer's own
  step counter to predict when the apply lands.
- **flax** — the window is `optax.MultiSteps` as the outermost transformation of the `OPTIMIZER`
  pattern; the generated `__init__` reads it back from each built optimizer and gates on it.

## The keras clock is the optimizer's private step counter

The generated `__init__` unwraps a `LossScaleOptimizer` (`inner_optimizer`) and captures the inner
optimizer's private `_iterations` variable together with `k`; `update(step)` returns
`(int(counter) + 1) % k == 0`. The `+ 1` is because the read is pre-step: the trainer evaluates
`update()` before `training_step` (`update_models` returns the pair left to right), so the question
answered is "will the step about to run land an apply", predicted from how many steps the optimizer
has already consumed.

The optimizer's counter, not the trainer step, is the clock — and that is what keeps
float16 × accumulation legal. When the `LossScaleOptimizer` sees non-finite gradients it skips the
inner optimizer entirely, `_iterations` freezes, and the next pre-step read is back in phase:
the prediction self-heals. Trainer-step arithmetic would drift out of phase permanently after the
first skip. The read is live on all three keras backends: the jax adapter assigns the optimizer
variable pytree back onto the variables every step, so the host `int()` sees the current count.

The private attribute is forced, not preferred: under accumulation the public `iterations` property
floor-divides the raw counter by `k`, so the intra-window phase is unrecoverable from any public
API, and `LossScaleOptimizer.step_counter` counts steps toward loss-scale growth, not optimizer
calls. Runtime tests pin the `_iterations` dependency so a keras upgrade that renames it fails
loudly. Accepted edges: on a skip step itself the pre-step read cannot foresee the NaN, so `update()`
misreports for that one step; and under multi-segment float16, segments whose scalers skip different
steps can de-phase — the first optimizer's counter is the learner's clock, documented rather than
reconciled.

## Flax accumulation is `optax.MultiSteps`, read back after construction

The generated `__init__` reads the window from each optimizer it just built —
`accumulation_window` in `structcast_model.flax.optimizers` — instead of the builder parsing the
pattern: `optimizer.tx` must be the `MultiSteps` instance itself (`isinstance`, which no address
spelling can fake), so the wrapper must be outermost — a `MultiSteps` nested inside `optax.chain`
leaves only its `MultiStepsState` reachable, and walking the state for that shell is what turns the
silent window-of-one misread into a `ValueError`. The same walk runs under an outermost
`MultiSteps`: a second `MultiStepsState` nested inside it means the device applies at the product
of the windows, and that misread is likewise a `ValueError`. Optax has no public accessor for
either argument,
so two normalized privates are read, both pinned by tests across the supported optax range: an int
`every_k_schedule` is wrapped by `MultiSteps.__init__` in a local lambda whose qualname identifies
it (calling it returns the int), and a `None` `should_skip_update_fn` becomes a local default
likewise identified by qualname. A user callable keeps its own qualname in either slot and is
rejected (`ValueError`) — either would break the identity that `mini_step` is a pure function of
the update-call count, which is what lets `update()` return `step % self._accumulate == 0` with no
device read — the trainer's step counter is 1-based and increments before it asks, so the step *is*
the call count, and the apply lands on the k-th call. The cadence therefore follows the native
mechanism (windows of exactly `k`), not the torch learners' historically short first window, whose
`(step + 1)` gate is self-consistent only because it also drives the apply.
No `MultiSteps` means a window of one, and the same formula holds. This reverses ADR-0013 because
the constraints dissolve its objections: with no skip predicate the device counter can never disagree
with the host formula, the `use_grad_mean` default replaces the previously emitted `loss / k`
scaling, `has_updated` is never called, and the `inject_learning_rate` rewrite recurses to the innermost
factory (wrapping it in `inject_hyperparams`) so an LR schedule inside `MultiSteps` still advances
per real update. The manual accumulator, the
static `need_update` jit argument and the donated `acc_grads` are deleted. One footgun is documented
rather than papered over: `MultiSteps` accumulates in float32 by default, doubling accumulator
memory against bf16 params — users who care pass `accumulator_dtype`.

## One learner, one window

All optimizers of a learner must share the same `k`: the trainer's update counter is learner-scoped
and `update()` answers for the whole learner. Both backends validate in the generated `__init__` —
a `ValueError`, the runtime convention, since generated scripts do not import builder errors: keras
compares the `gradient_accumulation_steps` of the prepared optimizers, flax the windows
`accumulation_window` reads back. The `Learner` protocol and `base_trainer` are
untouched — `update()` keeps its pre-step position and signature.

Migration is documentation, not validators: in-repo configs and test fixtures move to the new form,
`REFERENCE.md` and the README teach it, and an external keras or flax YAML still carrying
`ACCUMULATE_GRADIENTS` fails with pydantic's `extra="forbid"` field-not-permitted error — the
mechanism ADR-0012 already relies on — rather than a bespoke rejection validator.

## Considered options

- A post-hoc read-back behind an `arm()` / `update()` protocol split, reporting after the step
  whether an apply actually landed. Rejected: it redesigns the trainer protocol every backend
  implements, needs per-backend "did it apply" detection code, and forces a per-step device sync on
  the jax paths.
- Trainer-step arithmetic on keras with a build-time ban on float16 × accumulation. Rejected in
  favor of the counter clock, which tolerates loss-scale skips without banning the combination.

## Consequences

- Breaking config change for keras and flax `ACCUMULATE_GRADIENTS` users; torch configs are
  unaffected.
- ADR-0016's accepted counter divergence is gone: the keras update counter now tracks real optimizer
  applies, modulo the transient skip-step misreport above.
- ADR-0013's `MultiSteps` rejection is reversed under the constraints above; ADR-0012's
  "`ACCUMULATE_GRADIENTS` stays in the base: portable" rationale is narrowed to torch.
