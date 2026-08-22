# Generated learners pass state by name and keep module scope empty

A generated learner module used to carry user-named flow layers, `OPTIMIZER_HASHES` and the two
step functions at module scope, and the flax steps took the models and the optimizers as
dictionaries. Everything a learner template names comes from the user's configuration — and the
model builder imports whatever those patterns reference into the same module scope — so every
generated module-level name is a collision waiting for the right config. The templates now emit
one shape across all three frameworks: module scope holds imports and the learner class alone,
constants (`OPTIMIZER_HASHES`, and on keras `MIXED_PRECISION`/`MIXED_PRECISION_TYPE`) are class
attributes read off the class by the CLI (not off `__init__.__globals__`), state travels through
named parameters, and the learner keeps no anonymous collection attributes — each model, segment
and view is a named attribute, and `models`/`optimizers`/`flow_functions` are properties
assembling literal dictionaries, as the torch template always did. Generated comments describe
behavior and caveats only; they never cite repository documents the reader of a generated file
cannot see.

## Flax: the step signature is the donation contract

`_training_step(<models...>, <optimizers...>, *, <inputs...>, **kwargs)` — every model and
optimizer is its own positional-or-keyword parameter, the batch is keyword-only. The CLI derives
`donate_argnames` from the signature: positional-or-keyword parameters are donated state,
keyword-only parameters are the batch. This replaces the fixed `("models", "optimizers")` pair
and extends donation to hand-written learners that follow the convention; a hand-written step
that takes non-state arguments positionally will see them donated (harmless for per-step batches,
documented rather than guarded).

Flow functions are siblings of the steps inside `__init__`, not nested in `_training_step`:
everything a flow reads — owned models first (they are the `argnums`), then read-only models,
then batch entries and values stored by earlier segments — is a parameter. Differentiated flows
cannot use keyword-only parameters (`flax.nnx` resolves keywords to positions and raises
otherwise), so flow batch parameters are positional-or-keyword while the enclosing step keeps the
batch keyword-only.

A segment owning several modules passes them as a plain tuple — `nnx.Optimizer((a, b), ...)`,
`value_and_grad(..., argnums=(0, 1))`, `update((a, b), grads)` — verified against flax 0.12.8
under `nnx.jit` with donation. The `nnx.List` container of ADR-0013, its generated attribute and
its name-collision guard are gone.

## Counting is in-step detection on the first optimizer

The flax step reads the first optimizer's `MultiStepsState.gradient_step` before and after its
update and returns the comparison with the outputs; the learner does `_updates += has_updated`
and `restore_counters` seeds both counters from checkpoint meta. The host-side
`accumulation_window` read-back, the stored window, and the uniform-window `ValueError` are gone:
the first optimizer is the learner's clock, the stance ADR-0017 already documents for keras
de-phased segments. The outermost-`MultiSteps` wellformedness walk moves into the
`gradient_steps` helper the step calls, so a misconfigured chain now fails on the first traced
step instead of in `__init__`.

Keras keeps post-step detection but reads the public `iterations` property (`raw // k`) of the
first segment's unwrapped optimizer. The private `_iterations` phase read existed for the
pre-step *prediction* ADR-0018 abolished; detection only needs completed windows, which is
exactly what the public property returns. Keras flows and adapter-built steps take the batch as
named keyword arguments end-to-end (`Flow`/`InferenceFlow` contract change through the three
backend adapters and the MirroredStrategy wrapper), and segments are named attributes
(`_segment_<optimizer>`) with call sites assembling literal lists.

## Consequences

- Supersedes in part: ADR-0013 (the `nnx.List` container clause), ADR-0017 (the flax window
  read-back and uniform-window validation; the keras private-counter rationale), ADR-0018 (the
  flax host-side delta read and its "updates self-restore through optimizer state" resume rule;
  the keras `_iterations` read).
- Hand-written flax learners gain donation by following the signature convention; hand-written
  keras `InferenceFlow` implementations must accept named keyword arguments.
- A non-outermost `MultiSteps` misconfiguration surfaces at first step trace rather than at
  learner construction.
