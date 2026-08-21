# Flax optimizer segments build nnx.Optimizer directly from the DSL

The torch `OPTIMIZER` pattern resolves to a callable invoked with named parameters
(`opt_inst(get_named_parameters([...]))`), and the package supplies `create_opt` because torch schedulers are
separate stateful objects that must be married to the optimizer behind one proxy. Optax has no such split: an
optimizer, its clipping, its weight-decay masking, and its learning-rate schedule are all stages of one
`GradientTransformation` chain, and `flax.nnx.Optimizer(module, tx, wrt=...)` merely binds that chain to
Variables. The Flax side therefore ships **no** `create_opt`. The YAML `OPTIMIZER` pattern is a `_bind_` over
`flax.nnx.Optimizer` whose `tx` kwarg is a nested optax pattern, rendered to code by the existing recursive
pattern resolver:

```python
self.optimizer_g = (
    lambda *_arg0, **_kw0: Optimizer(
        *_arg0, tx=chain(clip_by_global_norm(max_norm=2.0), inject_hyperparams(adamw)(learning_rate=0.0002)), **_kw0
    )
)(List([G_AB, G_BA]), wrt=Param)
```

`FlaxLearnerBuilder._get_optimizer` mirrors the torch emission point: it appends the owned-module container
(single model verbatim; several wrapped in `nnx.List`) and appends `wrt=Param` only when the pattern did not
bind `wrt` itself. Any optax factory — or a user factory referenced by `_addr_`/`_file_` — is valid, because the
contract is only "a callable returning `nnx.Optimizer` when applied to the owned modules".

## Learning-rate reporting: auto-inject at generation time, `tree_get` at run time

Optax stores no learning rate: a constant lives in the update closure (state slot `EmptyState()`), a schedule
leaves only its `count` in state. The only sanctioned readable location is the `hyperparams` dict that
`optax.inject_hyperparams` materializes. The builder therefore rewrites the optimizer pattern before emission:
the factory call carrying a `learning_rate` kwarg is wrapped in
`inject_hyperparams(factory, static_args=(<every other kwarg>,))`. `static_args` is the safety valve — without
it inject arrayifies every numeric kwarg, and `bool` is an `int` subclass, so flags like `nesterov=True` would
reach the factory as `Array(1)`. Patterns already containing `inject_hyperparams` are left alone; patterns where
no `learning_rate` kwarg is identifiable (positional, renamed) are left alone and reported as `NaN`, with a
generation-time warning naming the fix.

At run time `structcast_model.flax.optimizers.get_learning_rate` reads
`optax.tree_utils.tree_get(unwrap_variables(optimizer.opt_state), "learning_rate", filtering=<numeric leaves>)` —
the filter is required because a scheduled inject state carries a second `learning_rate` entry under
`hyperparams_states`, and the nnx `OptArray` wrappers defeat type-based filtering unless the state is made pure
first; `unwrap_variables` is the local stand-in for `nnx.as_pure`, whose whole body is that one tree walk. The
generated `_training_step` calls it inside the trace (the walk runs at trace time, compiles to an
array reference), returns the values through an auxiliary output, and the `learning_rates` property converts the
stashed arrays lazily so the host sync lands on the epoch-end read, not on the step path.

## Gradient accumulation stays manual, not `optax.MultiSteps`

> **Superseded by ADR-0017.** Accumulation now goes through `optax.MultiSteps` in the user's chain,
> under the static-parse constraints recorded there; the rest of this ADR stands.

`Learner.update(step)` is the host-side single source of truth for "does this step apply" — the trainer counts
updates and dispatches the Update event from it. `MultiSteps` duplicates that counter in device state
(`mini_step`/`gradient_step`), returns all-zero update trees on non-apply steps, defaults to `use_grad_mean`
(which double-scales against the emitted `loss / k`), moves LR-schedule cadence depending on nesting, and its
`has_updated` helper raises on nnx-wrapped state. The generated learner instead keeps a params-shaped
`nnx.State` accumulator (the only tree shape `nnx.Optimizer.update` accepts — it filters the incoming gradients
through `wrt`, and a plain dict filters to empty): add gradients every micro-step, feed the accumulated tree to
`optimizer.update` and zero it under
the host-computed `__need_update__` flag. Clipping placed first in the tx chain then applies once per update, to
the accumulated gradients — the same semantics as the emitted torch `if __need_update__:` block.

## Trade-offs

- The auto-inject rewrite means emitted code differs from the YAML by one visible wrapper; fidelity is kept by
  the generated file itself, and the rewrite is a pure pattern-to-pattern function with its own unit tests.
- Constant learning rates pay one inject wrapper they do not strictly need; in exchange constants, schedules,
  and future runtime LR overrides all read through one mechanism, and the checkpoint layout does not depend on
  which kind the YAML used.
- `nnx.Optimizer.update()` return values and the `graph=` kwarg are never used, and generated code always
  passes `deterministic`/`use_running_average` explicitly — the emitted source stays valid across the whole
  supported flax range without version branches. The range is floored at 0.12.6: `nnx.as_pure` only appears in
  0.12.7 (0.12.6 calls it `nnx.pure`), so its body — one `jax.tree.map` replacing `Variable` leaves with
  `get_raw_value()` — is vendored as `structcast_model.flax.optimizers.unwrap_variables` instead of branching on
  the flax version. `tox -e flax-floor` runs the Flax suite against 0.12.6 to keep the floor honest.
