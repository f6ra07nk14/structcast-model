# `--compile` is graph compilation of what the command runs, on every command

One flag name carried four meanings. `scm torch time|train` and `scm flax time` compiled the model
graph; `scm flax train` compiled the generated steps, did so by default, and read an extra `"none"`
spelling of off that no other command had (14915c2, never recorded in ADR-0014); `scm keras train`
had no `--compile` at all while its backend adapter compiled every step unconditionally; and
`scm keras time --compile` called `keras.Model.compile(optimizer=None, **kw)`, Keras run
configuration on a command that runs no `fit`. No sentence about the flag survived the trip from one
command to the next, and on the two keras commands it named something other than graph compilation:
run configuration under `time`, nothing at all under `train`.

The six now share one definition: **`--compile` hands what the command runs to the framework's own
graph compiler.** Under `time` that is the timed forward call. Under `train` it is the learner's
compile units — on torch the models, compiled in place before `strategy.wrap` where the strategy
decides the unit boundaries exactly as ADR-0004 places them, the criteria tracker, and the flow
functions on a single device only; on flax the generated steps through the `nnx.jit` seam of
ADR-0014; on keras the generated steps through the backend adapter of ADR-0016, `tf.function` on
tensorflow and `jax.jit` on jax. Off unless given, on all six.

The value grammar is one YAML string everywhere: omitted, `null`, `~` or `false` is eager; `true` is
the compiler's own defaults (`{}`); a mapping — or a path to a YAML/JSON file holding one — is that
compiler's keyword arguments. `"none"` was a local hack in `scm flax train`, YAML reading the bare
word as a string a path load then fails on; YAML null is the honest spelling of off, and it raised a
`ValidationError` on every command until the shared parser's union gained `None`.

The keys that are the step's own contract are stripped, not honored: `static_argnames`,
`static_argnums`, `donate_argnames` and `donate_argnums` on `nnx.jit` and on the keras JAX adapter,
`input_signature` on the keras TensorFlow adapter. One mapping is splatted into both a training step
and an inference step whose positional signatures differ, and a `donate_argnums` meant for the first
would tell the second to donate live weights that are never assigned back. A backend with no
compiler of its own refuses a given mapping rather than ignoring it — keras on torch — at the top
of `train`, where `--backend` is known, and again in the adapter for a learner that reaches it
without the CLI; under `time` the ambient backend decides. Keras `dp` on tensorflow forwards the
mapping into the outer `tf.function` it wraps the replicated call in.

## `scm keras time --compile` was measuring nothing

The old spelling was a no-op on the path being timed. `Model.compile` stores `jit_compile` for the
trainer to consume in `make_train_function` and its test and predict twins, while the timed loop
calls `model(x)` directly — `Layer.__call__` and `Operation.__call__` in keras 3.15.1 contain no
reference to it. On a ConvNeXtV2-atto CPU run the medians with and without the flag differ by 3–5%,
inside run-to-run noise, and `model.predict_function` is still `None` after the loop, while the
compiled predict path is 4.3× (tensorflow) / 1.9× (jax) faster on that same model
(`docs/references/keras-time-compile-cpu.md`). So `time` hands the forward to the ambient backend
adapter's `build_inference_step` — the seam `train` uses — when the flag is given, and keeps the
direct call when it is not; no command calls `keras.Model.compile`.

All six declarations come from one factory, `scm_args.compile_option(api, tail)`, over the one
shared parser; both local `_compile_parser` twins are deleted. This is ADR-0010's second rule
executed ("near-identical → a factory with the differences as explicit arguments"), and it retires
the keras example of its third: the declaration never turned on run configuration versus graph
compilation, only on which API to name, a factory argument; `--device` and `--strategy` remain that
rule's examples.

Considered and rejected: dropping `scm keras time --compile` (leaves one `time` command without the
flag its siblings have); keeping `keras.Model.compile` behind it (never on the timed path); a
separate `--jit` flag (a second spelling of one intent); keeping `"none"` (not a YAML value, and a
second spelling of off with no prior art); and refusing the contract keys instead of stripping them
(flax strips them today, so refusing would mean changing flax as well).

## Consequences

- Two breaking default changes: `scm flax train` compiled by default and now runs eager unless
  `--compile true` is passed, and the keras adapters compiled every step they built and now compile
  only what `compile_kw` asks for. Launchers relying on either run eager until they pass the flag.
- The generated-code contract is untouched — `select_backend_adapter()`, `adapter.prepare`,
  `build_train_step`, `build_inference_step(..., models=[...])` and `flow_functions` are called
  exactly as before — so byte-pinned learners keep working without regeneration.
- `cfg/keras/others/compile_default.yaml` is rewritten for the new grammar: backend-compiler keyword
  arguments, not `keras.Model.compile` ones.
- Amends ADR-0010 (the keras example leaves its third rule), ADR-0014 (flax's default flips to off
  and `"none"` stops being a spelling) and ADR-0016 (adapter compilation becomes opt-in).
- If TensorFlow rejects a forwarded keyword argument at the `dp` path's outer `tf.function`, that
  strategy refuses non-empty mappings there instead of forwarding them.
