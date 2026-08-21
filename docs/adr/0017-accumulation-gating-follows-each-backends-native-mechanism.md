# Accumulation gating follows each backend's native mechanism

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
- **flax** — the window is `optax.MultiSteps` in the user's optax chain inside the `OPTIMIZER`
  pattern; the builder statically parses it and bakes the gate as a compile-time constant.

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

## Flax accumulation is `optax.MultiSteps`, statically parsed

The builder parses the `OPTIMIZER` pattern for a `MultiSteps` wrapper: `every_k_schedule` must be an
int literal (a callable is a `SpecError`) and `should_skip_update_fn` is rejected (`SpecError`) —
either would break the identity that `mini_step` is a pure function of the update-call count, which
is what lets `update()` bake `(step + 1) % k == 0` as a compile-time constant with no device read.
No `MultiSteps` in the pattern means `update()` returns true. This reverses ADR-0013 because the
constraints dissolve its objections: with no skip predicate the device counter can never disagree
with the host constant, the `use_grad_mean` default replaces the previously emitted `loss / k`
scaling, `has_updated` is never called, and the `inject_learning_rate` rewrite recurses to the innermost
factory (wrapping it in `inject_hyperparams`) so an LR schedule inside `MultiSteps` still advances
per real update. The manual accumulator, the
static `need_update` jit argument and the donated `acc_grads` are deleted. One footgun is documented
rather than papered over: `MultiSteps` accumulates in float32 by default, doubling accumulator
memory against bf16 params — users who care pass `accumulator_dtype`.

## One learner, one window

All optimizers of a learner must share the same `k`: the trainer's update counter is learner-scoped
and `update()` answers for the whole learner. Keras validates in the generated `__init__` — a
`ValueError`, the runtime convention of the adapters, since generated scripts do not import builder
errors — and flax at build time, a `SpecError`. The `Learner` protocol and `base_trainer` are
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
