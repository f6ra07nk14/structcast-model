# Flax loss scaling drives DynamicScale from outside its own value_and_grad

A float16 Flax run had no counterpart to the torch `GradScaler` path. The element type is already a
model property there — `dtype: float16` threads float16 through the matmuls while `param_dtype`
keeps the weights, the gradients and the optax moments in float32 (ADR-0012's split of what belongs
to the model and what to the learner) — but the backward pass then underflows with nothing to scale
it. `MIXED_PRECISION: bool | dict` on `FlaxUserDefinedLearner` supplies the missing half: one
`flax.training.dynamic_scale.DynamicScale` per optimizer segment, named `<optimizer>_dynamic_scale`
as the torch scaler is named `<optimizer>_grad_scaler`. It declares no `MIXED_PRECISION_TYPE`
beside it, because a learner cannot pick a compute type the model's layers own, and it refuses no
pairing, because it cannot see that type either.

`DynamicScale` is used for the scale it carries and for the growth, backoff and floor dynamics it
implements — but not through its `value_and_grad`, which is what makes this worth recording.
That method wraps `jax.value_and_grad`, which differentiates a plain pytree. A generated Flax
segment differentiates with `flax.nnx.value_and_grad`, which follows the module graph: an nnx module
is a pytree in flax 0.12, so `jax.value_and_grad` would accept one and then differentiate every
float leaf, raise on a typed RNG key, and — the fatal part — drop the mutations the forward wrote,
so dropout would redraw the same key on every step and batch statistics would never move.

The flow therefore takes the scale as a trailing parameter and returns `loss * _loss_scale` as the
value it is differentiated on, reporting the plain loss as its criterion, and
`update_with_loss_scale` divides the scale back out in float32. To advance the scale it hands
`DynamicScale.value_and_grad` a scalar function `x -> x * carrier`, where `carrier` is `0.0` when
every gradient is finite and `NaN` when one is not: the gradient that method inspects is then finite
exactly when the real gradients are, so the interval, the backoff factor and the minimum scale stay
flax's own rather than a second copy of them here. The skip is the official pattern applied to the
live objects: the state the update may write is snapshotted as a pure dict beforehand
(`nnx.Not(nnx.RngState)`, so the RNG the forward advanced is left alone) and `jax.numpy.where`
chooses between the two afterwards.

Under an `optax.MultiSteps` window that rollback covers the accumulator and the window counter, so
a non-finite micro-step is dropped rather than the window it landed in — torch drops the window,
because its accumulator is each parameter's own gradient buffer, which a scaler can only discard
whole. The window counter is read after the update and before the rollback, which keeps the
counters on ADR-0018's intent semantics: a skipped apply still reports the update it attempted.
The scale's own clock is the step, not the window: it is advanced on every micro-step, where the
torch scaler updates once per window.

Considered and rejected: differentiating a split-out `nnx.State` so `DynamicScale.value_and_grad`
could produce the gradients itself (fifteen generated lines per segment reimplementing what
`nnx.value_and_grad` already does, and the mutation propagation has to be rebuilt by hand);
re-implementing the eight lines of growth and backoff (duplicates an installed dependency and
drifts from it silently); `optax.apply_if_finite` (expressible in a template today, scales no loss,
applies the poisoned update after `max_consecutive_errors`, and leaves the accumulator advanced);
and masking the gradients to zero instead of skipping (AdamW would still decay the weights and
advance its count, which is not what the torch scaler does).

## Consequences

- Amends ADR-0015: the payload's `grad_scalers` slot is no longer always empty on the Flax side. A
  scaled learner fills it with the two numbers each scale carries, `{"scale", "fin_steps"}`, which
  keeps the item on the state backend's JSON path; a learner without the field leaves it empty, so
  the payload contract itself is unchanged.
- Amends ADR-0019: the step signature gains the scales,
  `_training_step(<models...>, <optimizers...>, <scales...>, *, <inputs...>, **kwargs)`, and the
  step returns one more value per scale. The donation contract is unchanged — they are
  positional-or-keyword, so `donate_argnames` picks them up — and an unscaled learner emits the
  signature ADR-0019 describes, byte for byte.
- The carry is pinned to float32/int32 at construction rather than left on the dataclass's Python
  scalars, without which a compiled run (`--compile`) retraces the whole step on the second call and
  the first call's donation has nothing to donate.
- A `DynamicScale` is immutable, so the resume path rebinds it by name: the key of `grad_scalers`
  names the learner attribute holding it, where the torch loader mutates its scaler in place.
- The rollback holds a second copy of the parameters and the optimizer state live across the
  update. That is transient memory on the order of the state itself, which matters to exactly the
  runs this field exists for.
