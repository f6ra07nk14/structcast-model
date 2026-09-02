# `scm keras time --compile`: `keras.Model.compile` never reached the timed call

Measurement backing ADR-0024's redefinition of `scm keras time --compile`: with the flag spelled as
`keras.Model.compile(optimizer=None, **kw)`, the number the command reports does not move, because
the timed loop calls `model(inputs, training=...)` directly and Keras 3 consumes `jit_compile` only
in the trainer's `make_train_function` / `make_test_function` / `make_predict_function`.

## Setup

- Hardware: 12-core CPU devcontainer, no GPU; keras 3.15.1, tensorflow 2.21.0, jax 0.11.0,
  Python 3.12.3. Measured 2026-09-02 on `dev` at `1ea2739` with the keras `train --compile`
  working tree applied (`measure_inference_time` unchanged by it).
- Model: generated ConvNeXtV2 `atto` (`cfg/keras/models/ConvNeXtV2.yaml`, `backbone: atto`),
  input `image: [224, 224, 3]`, batch 1.
- Command, three repeats per cell, `-w 3 -t 20 -d cpu:0`:

```bash
KERAS_BACKEND=<backend> uv run scm keras time \
  '[_obj_, {_addr_: kmodel.Model, _file_: /tmp/kmodel.py}, _call_]' \
  -s 'image: [224, 224, 3]' -w 3 -t 20 -d cpu:0 [--compile '{jit_compile: true|false}']
```

## CLI numbers (median of three, seconds per call)

| backend | `--compile` | median | vs. omitted |
| --- | --- | --- | --- |
| tensorflow | omitted | 0.149963 | — |
| tensorflow | `{jit_compile: true}` | 0.155442 | +3.7% |
| tensorflow | `{jit_compile: false}` | 0.145858 | −2.7% |
| jax | omitted | 0.078905 | — |
| jax | `{jit_compile: true}` | 0.076127 | −3.5% |
| jax | `{jit_compile: false}` | 0.074798 | −5.2% |

Within-cell spread (tensorflow omitted: 0.142–0.157, plus two outliers at 0.287 and 0.208) is the
same size as the between-cell differences: the flag has no measurable effect on what the command
reports. All 18 runs printed `Compiling the model...` on the flagged cells, so the branch was taken.

## Control: same model, same process, by call path (median of 20)

| backend | compile | call | median |
| --- | --- | --- | --- |
| tensorflow | none | `model(x, training=False)` | 0.143143 |
| tensorflow | `jit_compile=False` | `model(x, training=False)` | 0.144250 |
| tensorflow | `jit_compile=True` | `model(x, training=False)` | 0.143657 |
| tensorflow | `jit_compile=False` | `predict_on_batch(x)` | 0.033698 |
| tensorflow | `jit_compile=True` | `predict_on_batch(x)` | 0.053235 |
| jax | none | `model(x, training=False)` | 0.083234 |
| jax | `jit_compile=False` | `model(x, training=False)` | 0.071652 |
| jax | `jit_compile=True` | `model(x, training=False)` | 0.074859 |
| jax | `jit_compile=False` | `predict_on_batch(x)` | 0.079102 |
| jax | `jit_compile=True` | `predict_on_batch(x)` | 0.039229 |

The direct call is flat across compile states (tensorflow 0.1431 / 0.1443 / 0.1437); the trainer's
predict path is 4.3× faster on tensorflow (graph mode either way) and 1.9× faster on jax under
`jit_compile=True`, where the same switch is a 2.0× difference on the predict path and none on the
direct call.

## Introspection

After `compile(optimizer=None, jit_compile=True)`, `model.jit_compile` is `True` and
`model.compiled` is `True`, but `model.predict_function` is `None`; it is still `None` after the
3 warmup and 20 timed direct calls. One `predict_on_batch(x)` builds it immediately (a
`tf.function` polymorphic function on tensorflow, `JAXTrainer.make_predict_function.<locals>.step_fun`
on jax). The compiled step was never built, so it could not have been timed.
