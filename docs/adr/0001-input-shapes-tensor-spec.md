# Input shapes are validated as `TensorSpec`

`INPUT_SHAPES` entries are validated as a `TensorSpec` tree (`src/structcast_model/builders/schema.py`), which accepts
both an implicit compact form (`image: [3, 224, 224]`) and an explicit form
(`tokens: {_SHAPE_: [512], _DTYPE_: int64, _INIT_: torch.zeros}`). Both forms exist because nearly every input needs
only a shape, and requiring the explicit form everywhere would make the common configuration markedly noisier for no
benefit; the explicit form serves the minority of inputs that need a non-default element type or a specific
initializer. A `TensorSpec` serializes back to a bare shape tuple whenever `_DTYPE_` and `_INIT_` are at their
defaults, so configurations that never used the explicit form round-trip unchanged.

## Default dtype is `bfloat16`, not `float32`

`create_torch_inputs` (`src/structcast_model/torch/trainer.py`) hardcoded `dtype=torch.float32` for every dummy input.
The default `_DTYPE_` is `bfloat16`, which is a deliberate behaviour change: an input carrying no dtype annotation now
gets a `bfloat16` tensor where it previously got `float32`. These models are trained and served in `bfloat16`, so the
dummy inputs used for shape inference, tracing and smoke tests should match the real thing rather than quietly differ
from it. Configurations that need the old behaviour must now say `_DTYPE_: float32` explicitly.

## Integer inputs fall back to `zeros`, with a warning

The default initializer is `rand`, which is not usable for integer dtypes. Two options were considered:

- **Strict**: always `rand`. An integer input with no `_INIT_` fails loudly, and every such input must name an
  initializer.
- **Smart default (chosen)**: an integer dtype with no `_INIT_` gets `zeros`, plus a `logging.warning` naming the
  input's dtype and the fallback that was applied.

The smart default won because token-id and label inputs are the common case, and failing on them would force an
`_INIT_` into every configuration that has an embedding input — punishing the majority for something the code can
resolve on its own. `zeros` is also a valid index into any embedding table, which arbitrary random values would not
be. The warning keeps the fallback honest: silently fabricating all-zero token ids is a real failure mode, so it is
reported, just not made fatal.

## `TensorInitializer` cannot check the signature

An `_INIT_` address is imported through structcast's `import_from_address` (which applies the security checks) and
then gated with `isinstance(fn, TensorInitializer)`, a `runtime_checkable` Protocol. `runtime_checkable` only verifies
that `__call__` exists — the check is effectively `callable(fn)` and does not inspect the signature. An address
pointing at a callable with the wrong signature therefore passes the gate and fails later, at call time, with whatever
that callable raises. This is accepted: the gate rejects obvious non-callables and states the expected contract in the
type system; it does not guarantee it.

## No `_LOW_` / `_HIGH_` fields

Value-range fields were deliberately not added. Dummy inputs exist to have the right shape and element type; nothing
today depends on their values being meaningful. Add them when a real use case appears — for example token ids that
must stay below a vocabulary size — and not before.
