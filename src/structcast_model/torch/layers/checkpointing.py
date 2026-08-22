"""Activation checkpointing base class for generated PyTorch layers."""

from functools import partial
from typing import Any, ClassVar

import torch


class GradientCheckpointingLayer(torch.nn.Module):
    """A module that recomputes its own forward pass during the backward pass.

    A generated layer with `GRADIENT_CHECKPOINTING` enabled inherits this instead of
    `torch.nn.Module` and overrides the two class attributes below. Interception happens in
    `__call__`, so no wrapper module is introduced: `named_modules()` paths, parameter names and
    state-dict keys stay the ones the layer would have without checkpointing, which sharding globs
    and in-place compilation depend on.

    Checkpointing follows training mode: a model a learner freezes runs in `eval()` while gradients
    still flow through it, and it is deliberately not checkpointed there. Recomputing a layer whose
    forward pass is defined to behave differently in the two modes is the anomaly, so the memory
    saving is declined rather than bought with a second pass under other semantics.
    `torch.is_grad_enabled()` adds nothing to that decision; it only keeps a `torch.no_grad()`
    forward -- an inference step -- from paying for a checkpoint no backward pass will use
    (`docs/adr/0020`).
    """

    gradient_checkpointing: ClassVar[bool] = False
    """Whether the layer recomputes its forward pass; the generated subclass sets it to `True`."""

    _checkpoint_kwargs: ClassVar[dict[str, Any]] = {}
    """The keyword arguments handed to `torch.utils.checkpoint.checkpoint`."""

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Run the forward pass, recomputing it in the backward pass while gradients are recorded."""
        if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
            # Only the positional arguments are tracked for recomputation, so the keyword arguments
            # are bound into the callable, as `transformers.GradientCheckpointingLayer` does.
            return torch.utils.checkpoint.checkpoint(
                partial(super().__call__, **kwargs), *args, **self._checkpoint_kwargs
            )
        return super().__call__(*args, **kwargs)
