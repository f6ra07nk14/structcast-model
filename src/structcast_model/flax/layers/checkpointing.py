"""Activation checkpointing base class for generated Flax (nnx) modules."""

from typing import Any, ClassVar

from flax import nnx


class GradientCheckpointingModule(nnx.Module):
    """An `nnx.Module` that recomputes its own forward pass during the backward pass.

    A generated module with `GRADIENT_CHECKPOINTING` enabled emits its body as `_forward` and
    inherits this instead of `nnx.Module`, so no wrapper module is introduced and the variable paths
    of the module are unchanged.

    `nnx.remat` resolves neither keyword-only parameters nor a `functools.partial`, so the
    rematerialized callable takes the module and the arrays positionally and reads the flags off the
    enclosing closure -- verified against flax 0.12.8 eager, jitted and under a checkpoint policy.
    """

    gradient_checkpointing: ClassVar[bool] = False
    """Whether the module recomputes its forward pass; the generated subclass sets it to `True`."""

    _remat_kwargs: ClassVar[dict[str, Any]] = {}
    """The keyword arguments handed to `flax.nnx.remat`."""

    inputs: list[str]
    """The input names of the module, bound by the generated `__init__`."""

    training: bool
    """The mode the module runs in when a call does not say; bound by the generated `__init__`."""

    def __call__(self, *args: Any, training: bool | None = None, **kwargs: Any) -> Any:
        """Run the forward pass, recomputing it in the backward pass while training."""
        training = self.training if training is None else training
        if not (self.gradient_checkpointing and training):
            return self._forward(*args, training=training, **kwargs)
        # A caller may pass the batch by name -- the CLI initializing a model does -- while the
        # rematerialized callable takes the arrays positionally, so the declared inputs are moved.
        arrays = (*args, *(kwargs.pop(name) for name in self.inputs[len(args) :] if name in kwargs))

        def _ckpt(module: "GradientCheckpointingModule", *arrays: Any) -> Any:
            return type(module)._forward(module, *arrays, training=training, **kwargs)

        return nnx.remat(_ckpt, **self._remat_kwargs)(self, *arrays)

    def _forward(self, *args: Any, training: bool | None = None, **kwargs: Any) -> Any:
        """Compute the forward pass; the generated subclass emits its body here."""
        raise NotImplementedError("The _forward method is emitted by the generated subclass.")
