r"""Example optax schedule compositions for Flax (nnx) training runs.

This file is much shorter than `examples/torch/optimizers.py`, and deliberately so: optax already
ships every piece a learner template composes -- `optax.chain`, `optax.adamw`,
`optax.clip_by_global_norm`, `optax.MultiSteps` -- and `flax.nnx.Optimizer` only binds the resulting
transformation to a module, so a Flax `OPTIMIZER` pattern needs no wrapper class at all. What optax
does not do is count epochs: its schedules are functions of the update count, while a training
recipe is written in epochs. The two helpers below are that conversion, referenced from a
configuration by file path:

```yaml
learning_rate:
  - _obj_
  - _addr_: linear_decay_after
    _file_: examples/flax/optimizers.py
  - _call_: {init_value: 2.0e-4, epochs: 200, decay_epoch: 100, steps_per_epoch: 1000}
```

Both return an `optax.Schedule` -- a callable of the update count -- which is what
`optax.adam(learning_rate=...)` accepts. `FlaxLearnerBuilder` wraps the factory carrying
`learning_rate` in `optax.inject_hyperparams`, so the rate a schedule produces stays readable at
run time.

`steps_per_epoch` is the number of *optimizer applies* per epoch, not the number of batches: under
`optax.MultiSteps` an epoch of N batches advances the schedule by N // every_k_schedule.
"""

import optax


def linear_decay_after(
    *,
    init_value: float,
    epochs: int,
    decay_epoch: int,
    steps_per_epoch: int,
    offset: int = 0,
    end_value: float = 0.0,
) -> optax.Schedule:
    """Hold the rate for *decay_epoch* epochs, then ramp it linearly down to *end_value*.

    The CycleGAN recipe's schedule: the optax twin of the `torch.optim.lr_scheduler.LambdaLR` of
    `cfg/torch/learners/CycleGAN.yaml`, whose lambda is
    `1 - max(0, epoch + offset - decay_epoch) / (epochs - decay_epoch)`.

    Not the same curve, only the same envelope: the torch schedule steps once per epoch and holds
    one rate for all of it, while an optax schedule is read on every update, so this one falls
    continuously. The two agree exactly at epoch boundaries and differ inside an epoch by at most
    one epoch's worth of the ramp.

    Args:
        init_value: The rate held until the decay begins.
        epochs: Total number of epochs of the run, i.e. where the ramp reaches *end_value*.
        decay_epoch: The epoch the ramp starts at.
        steps_per_epoch: Optimizer applies per epoch, which is what converts the two into steps.
        offset: Epochs already trained, so a resumed run starts further down the ramp.
        end_value: The rate the ramp ends at.

    Returns:
        The schedule, callable with the update count.

    Raises:
        ValueError: If the ramp has no length, i.e. *epochs* is not past *decay_epoch*.

    Example:
        >>> schedule = linear_decay_after(init_value=1.0, epochs=4, decay_epoch=2, steps_per_epoch=1)
        >>> [float(schedule(step)) for step in range(5)]
        [1.0, 1.0, 1.0, 0.5, 0.0]
    """
    if epochs <= decay_epoch:
        raise ValueError(f"epochs ({epochs}) must be greater than decay_epoch ({decay_epoch}) for a ramp to exist.")
    return optax.linear_schedule(
        init_value=init_value,
        end_value=end_value,
        transition_steps=(epochs - decay_epoch) * steps_per_epoch,
        transition_begin=(decay_epoch - offset) * steps_per_epoch,
    )


def warmup_cosine(
    *,
    peak_value: float,
    epochs: int,
    steps_per_epoch: int,
    warmup_epochs: int = 1,
    init_value: float = 0.0,
    end_value: float = 1e-6,
) -> optax.Schedule:
    """Ramp the rate up over *warmup_epochs*, then anneal it to *end_value* by *epochs*.

    The optax twin of the timm `cosine` schedule `examples/torch/optimizers.py` builds through
    `AdamWWithCosine`, written in epochs for the same reason.

    Args:
        peak_value: The rate reached at the end of the warmup.
        epochs: Total number of epochs of the run, i.e. where the cosine reaches *end_value*.
        steps_per_epoch: Optimizer applies per epoch, which is what converts the two into steps.
        warmup_epochs: Epochs spent ramping from *init_value* up to *peak_value*.
        init_value: The rate the warmup starts at.
        end_value: The rate the cosine ends at.

    Returns:
        The schedule, callable with the update count.

    Raises:
        ValueError: If the warmup does not fit inside the run.

    Example:
        >>> schedule = warmup_cosine(peak_value=1.0, epochs=2, steps_per_epoch=2, warmup_epochs=1)
        >>> round(float(schedule(2)), 6)
        1.0
    """
    if warmup_epochs >= epochs:
        raise ValueError(f"warmup_epochs ({warmup_epochs}) must be smaller than epochs ({epochs}).")
    return optax.warmup_cosine_decay_schedule(
        init_value=init_value,
        peak_value=peak_value,
        warmup_steps=warmup_epochs * steps_per_epoch,
        decay_steps=epochs * steps_per_epoch,
        end_value=end_value,
    )
