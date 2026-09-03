"""Example optimizer factory for Keras training runs.

A Keras optimizer already carries its own learning-rate schedule, its clipping
(`clipnorm` / `global_clipnorm`) and its gradient accumulation
(`gradient_accumulation_steps`), so the learner templates name `keras.optimizers.*` directly and the
package ships no `create_opt` twin. One knob has no constructor keyword: weight-decay exemptions are
configured by calling `optimizer.exclude_from_weight_decay(...)` after construction and before the
optimizer is built, which an object pattern cannot express. That call is what this factory adds, and
a learner template references it by file path:

```yaml
OPTIMIZER:
  - _obj_
  - _addr_: create_optimizer
    _file_: examples/keras/optimizers.py
  - _call_:
      name: AdamW
      no_weight_decay_names: [bias, gamma, beta, embeddings]
      learning_rate: 0.001
      weight_decay: 0.05
```

The backend adapter builds the returned optimizer against the variables of its segment, which is
after this function returns, so the exemption is always configured in time.
"""

from collections.abc import Sequence
from typing import Any

import keras


def create_optimizer(name: str, no_weight_decay_names: Sequence[str] = (), **kwargs: Any) -> keras.optimizers.Optimizer:
    """Build a `keras.optimizers` optimizer and exempt the named variables from weight decay.

    Args:
        name (str): Name of the class in `keras.optimizers`, e.g. `AdamW` or `SGD`.
        no_weight_decay_names (Sequence[str]): Anchor words matched against each variable's name;
            a variable whose name contains one of them is not decayed. Keras names the parameters
            of its layers `kernel`, `bias`, `gamma`, `beta` and `embeddings`, so
            `[bias, gamma, beta, embeddings]` is the usual "no decay on biases, normalization
            scales and lookup tables" rule. Ignored by an optimizer without weight decay.
        **kwargs (Any): Keyword arguments of the optimizer class, e.g. `learning_rate`,
            `weight_decay`, `global_clipnorm` or `gradient_accumulation_steps`.

    Returns:
        keras.optimizers.Optimizer: The optimizer, not yet built.

    Raises:
        ValueError: If `keras.optimizers` has no class of that name.
    """
    optimizer_type = getattr(keras.optimizers, name, None)
    if optimizer_type is None:
        raise ValueError(f"keras.optimizers has no optimizer named {name!r}.")
    optimizer = optimizer_type(**kwargs)
    if no_weight_decay_names:
        optimizer.exclude_from_weight_decay(var_names=list(no_weight_decay_names))
    return optimizer
