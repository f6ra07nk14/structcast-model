"""Backend adapters running the training and inference steps of a Keras learner.

One generated learner drives every Keras backend, so every mechanic the three backends disagree
about -- gradient computation, optimizer application, variable state handling, step compilation --
lives behind the `BackendAdapter` selected once from `keras.backend.backend()` (`docs/adr/0016`).

Gradient accumulation is deliberately absent: `keras.optimizers.Optimizer` takes
`gradient_accumulation_steps`, which accumulates and gates the update inside the optimizer on all
three backends, so it belongs in the optimizer pattern exactly as `clipnorm` does, not in a step
callable that would need its own buffers, its own gate and its own per-backend carrying.

Each backend's framework is bound lazily at module level: only the active backend's framework is
guaranteed to be installed, so the binding resolves on first attribute access inside an adapter,
never at import time.
"""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import cache
from typing import TYPE_CHECKING, Any

# Protocol and runtime_checkable come from typing_extensions so that isinstance checks use
# inspect.getattr_static on Python 3.11 as well (backported from 3.12), as in base_trainer.
from typing_extensions import Protocol, runtime_checkable

import keras

if TYPE_CHECKING:
    import jax
    import tensorflow as tf

    import torch
else:
    from structcast.utils.lazy_import import LazyModuleImporter

    # An inactive backend's framework may not be installed, so each one is bound lazily and only
    # resolved by the adapter the active backend selects, as in `loggers.state_backends`.
    jax = LazyModuleImporter("jax")
    tf = LazyModuleImporter("tensorflow")
    torch = LazyModuleImporter("torch")

Flow = Callable[[Mapping[str, Any]], tuple[Any, dict[str, Any]]]
"""A training flow: it takes the batch and returns the loss to differentiate and the criteria.

The flow closes over its models and is written in `keras.ops`, so the same closure runs on every
backend; the adapter decides how the loss is differentiated and how the update is applied.
"""

InferenceFlow = Callable[[Mapping[str, Any]], dict[str, Any]]
"""An inference flow: it takes the batch and returns the criteria, differentiating nothing."""


@dataclass(kw_only=True, slots=True)
class AdapterSegment:
    """One optimizer segment of a learner, as an adapter sees it at run time."""

    name: str
    """Name of the optimizer, used to report which segment is misconfigured."""

    flow: Flow
    """The training flow of this segment."""

    optimizer: Any
    """The Keras optimizer of this segment; `prepare` may replace it with a wrapped one."""

    variables: list[Any]
    """The Keras variables this optimizer updates, the trainable variables of its layers."""

    models: list[Any]
    """The Keras models the flow runs, whose remaining state (moving statistics, seeds) it may touch."""


@runtime_checkable
class BackendAdapter(Protocol):
    """Runs the backend-specific half of one Keras training loop."""

    @property
    def name(self) -> str:
        """The `keras.backend.backend()` value this adapter implements."""

    def prepare(
        self,
        segments: Sequence[AdapterSegment],
        *,
        mixed_precision: bool | dict[str, Any] = False,
        mixed_precision_type: str | None = None,
    ) -> None:
        """Build every segment's optimizer against its variables, wrapping it for loss scaling.

        A `float16` policy needs the gradients scaled, which `keras.optimizers.LossScaleOptimizer`
        does; a `dict` mixed precision supplies its keyword arguments. `bfloat16` shares float32's
        exponent range, so its gradients cannot underflow and it stays unwrapped -- the same pairing
        `Model.compile(auto_scale_loss=True)` applies. Setting the global
        `keras.mixed_precision` policy is the caller's job and must happen before the models are
        built; an adapter only honors it.

        Args:
            segments: The optimizer segments of the learner, mutated in place when wrapped.
            mixed_precision: False, True, or the keyword arguments of the `LossScaleOptimizer`.
            mixed_precision_type: The element type of the global policy, if any.
        """

    def build_train_step(self, segments: Sequence[AdapterSegment]) -> InferenceFlow:
        """Compile the training step running every segment: its flow, its gradients, its update.

        The returned step takes the batch and returns the merged criteria of every segment. A
        single-process adapter reduces nothing: `docs/adr/0016` obliges whoever runs a distributed
        cell to reduce the criteria across replicas before they reach the tracker, since the Keras
        tracker -- unlike its torch twin -- never all-reduces, and a strategy that skips this must
        fail its own tests rather than log per-replica values.

        The criteria are merged last-wins, so keeping the names of two segments distinct is the
        caller's job: an adapter only sees the names at run time and cannot detect a clash here.
        `prepare` must have run first -- only the JAX backend refuses an unbuilt optimizer, while
        TensorFlow and torch would build it inside the compiled step.

        Args:
            segments: The optimizer segments, applied in order, each against its own variables.

        Returns:
            The compiled training step.
        """

    def build_inference_step(self, flow: InferenceFlow, *, models: Sequence[Any] = ()) -> InferenceFlow:
        """Compile the inference step, which updates no variable.

        Args:
            flow: The inference flow.
            models: The Keras models the flow runs, which JAX needs to thread their variables
                through the jitted step; the other backends read them from the closure and ignore
                this argument.

        Returns:
            The compiled inference step.
        """


class _Adapter:
    """The optimizer preparation the three adapters share."""

    name: str

    def prepare(
        self,
        segments: Sequence[AdapterSegment],
        *,
        mixed_precision: bool | dict[str, Any] = False,
        mixed_precision_type: str | None = None,
    ) -> None:
        """Build every segment's optimizer, wrapping it in a `LossScaleOptimizer` under float16."""
        # Any mapping enables loss scaling, an empty one included: it carries the wrapper's keyword
        # arguments, and `builders/torch.py` reads the same field the same way.
        enabled = mixed_precision if isinstance(mixed_precision, bool) else True
        for segment in segments:
            if not segment.variables:
                # An optimizer built against no variable trains nothing and reports no error, the
                # exact silent no-op docs/adr/0016 rejects the alternatives for.
                raise ValueError(f"Optimizer segment {segment.name!r} has no trainable variables to update.")
            if enabled and mixed_precision_type == "float16":
                kwargs = mixed_precision if isinstance(mixed_precision, Mapping) else {}
                segment.optimizer = keras.optimizers.LossScaleOptimizer(segment.optimizer, **kwargs)
            # Building here rather than on the first update keeps every slot variable out of the
            # compiled step, where TensorFlow forbids creating variables and JAX would trace them.
            segment.optimizer.build(segment.variables)


class TensorFlowAdapter(_Adapter):
    """Adapter differentiating with `tf.GradientTape` and applying the optimizer statefully.

    `optimizer.stateless_apply` refuses to run on TensorFlow by design, so the stateful `apply` is
    the only path; `tf.function` wraps the whole step once, so the tape, the gradients and the
    update land in one graph.
    """

    name = "tensorflow"

    def build_train_step(self, segments: Sequence[AdapterSegment]) -> InferenceFlow:
        """Trace one `tf.function` running every segment's tape, gradients and update."""

        def step(batch: Mapping[str, Any]) -> dict[str, Any]:
            criteria: dict[str, Any] = {}
            for segment in segments:
                with tf.GradientTape() as tape:
                    loss, values = segment.flow(batch)
                    # Outside `fit()` nobody scales the loss for us. `scale_loss` returns the loss
                    # untouched on an optimizer without a loss scale, so no branch is needed.
                    scaled = segment.optimizer.scale_loss(loss)
                gradients = tape.gradient(scaled, segment.variables)
                segment.optimizer.apply(gradients, segment.variables)
                criteria.update(values)
            return criteria

        return tf.function(step)

    def build_inference_step(self, flow: InferenceFlow, *, models: Sequence[Any] = ()) -> InferenceFlow:
        """Trace the inference flow into one `tf.function`, reading its variables from the closure."""
        return tf.function(flow)


class JaxAdapter(_Adapter):
    """Adapter differentiating with `jax.value_and_grad` and applying the optimizer statelessly.

    The stateful path is unavailable: `LossScaleOptimizer.apply` raises an `UnexpectedTracerError`
    under a trace. Every variable the step reads is therefore threaded through the jitted function
    as an argument and bound with a `keras.StatelessScope`, the idiom of Keras' own JAX trainer --
    a variable read from the closure instead would be traced as a constant and freeze at its first
    value. That includes the state the flow updates without owning: the moving statistics of a
    normalization layer and the state of every `SeedGenerator`, which `Layer.variables` lists, and
    which a dropped thread would silently freeze into one repeated mask.
    """

    name = "jax"

    def build_train_step(self, segments: Sequence[AdapterSegment]) -> InferenceFlow:
        """Compile one `jax.jit` step threading trainable, state and optimizer values through."""
        segments = list(segments)
        owned = {id(variable) for segment in segments for variable in segment.variables}
        state_variables = _state_variables(segments, owned)

        def mapping(
            trainables: list[list[Any]], states: list[Any], optimizers: list[list[Any]]
        ) -> list[tuple[Any, Any]]:
            pairs = list(zip(state_variables, states, strict=True))
            for segment, values, optimizer_values in zip(segments, trainables, optimizers, strict=True):
                pairs += zip(segment.variables, values, strict=True)
                # The optimizer variables carry the loss scale `scale_loss` reads.
                pairs += zip(segment.optimizer.variables, optimizer_values, strict=True)
            return pairs

        def gradient_function(index: int) -> Any:
            segment = segments[index]

            def loss_function(
                own: list[Any],
                trainables: list[list[Any]],
                states: list[Any],
                optimizers: list[list[Any]],
                batch: Mapping[str, Any],
            ) -> tuple[Any, tuple[dict[str, Any], list[Any]]]:
                trainables = [*trainables[:index], own, *trainables[index + 1 :]]
                with keras.StatelessScope(state_mapping=mapping(trainables, states, optimizers)) as scope:
                    loss, values = segment.flow(batch)
                    scaled = segment.optimizer.scale_loss(loss)
                # `mapping` seeds the scope with every state variable, so each one has a current
                # value: the one the flow wrote, or the one threaded in.
                updated = [scope.get_current_value(variable) for variable in state_variables]
                return scaled, (values, updated)

            return jax.value_and_grad(loss_function, has_aux=True)

        gradients = [gradient_function(index) for index in range(len(segments))]

        def step(
            trainables: list[list[Any]], states: list[Any], optimizers: list[list[Any]], batch: Mapping[str, Any]
        ) -> tuple[dict[str, Any], list[list[Any]], list[Any], list[list[Any]]]:
            criteria: dict[str, Any] = {}
            for index, segment in enumerate(segments):
                (_, (values, states)), grads = gradients[index](
                    trainables[index], trainables, states, optimizers, batch
                )
                # `stateless_apply` opens its own scope, so it runs outside the flow's.
                own, optimizer_values = segment.optimizer.stateless_apply(optimizers[index], grads, trainables[index])
                trainables = [*trainables[:index], own, *trainables[index + 1 :]]
                optimizers = [*optimizers[:index], optimizer_values, *optimizers[index + 1 :]]
                criteria.update(values)
            return criteria, trainables, states, optimizers

        jitted = jax.jit(step)

        def train_step(batch: Mapping[str, Any]) -> dict[str, Any]:
            criteria, trainables, states, optimizers = jitted(
                [[variable.value for variable in segment.variables] for segment in segments],
                [variable.value for variable in state_variables],
                [[variable.value for variable in segment.optimizer.variables] for segment in segments],
                batch,
            )
            # Assigning every step keeps the variables the single source of truth, so the tracker,
            # the checkpoints and the next step all read what this step computed.
            _assign(state_variables, states)
            for segment, values, optimizer_values in zip(segments, trainables, optimizers, strict=True):
                _assign(segment.variables, values)
                _assign(segment.optimizer.variables, optimizer_values)
            return criteria

        return train_step

    def build_inference_step(self, flow: InferenceFlow, *, models: Sequence[Any] = ()) -> InferenceFlow:
        """Compile one `jax.jit` step threading the models' variables through, as the training step does.

        A jitted closure reading its variables directly would trace them as constants and answer
        with the weights of the step it was first traced on, so every variable is threaded as an
        argument and bound with a `keras.StatelessScope`. The scope's writes are dropped instead of
        assigned back, which is exactly inference: whatever the flow touches -- moving statistics,
        seeds -- stays as the training step left it. Without the models there is nothing to thread,
        so the flow runs eagerly rather than freezing at its first weights.
        """
        variables = [variable for model in models for variable in model.variables]
        if not variables:
            return flow

        def inference(values: list[Any], batch: Mapping[str, Any]) -> dict[str, Any]:
            with keras.StatelessScope(state_mapping=list(zip(variables, values, strict=True))):
                return flow(batch)

        jitted = jax.jit(inference)

        def inference_step(batch: Mapping[str, Any]) -> dict[str, Any]:
            return jitted([variable.value for variable in variables], batch)

        return inference_step


class TorchAdapter(_Adapter):
    """Adapter differentiating with autograd and applying the Keras optimizer under `no_grad`.

    A Keras variable *is* the `torch.nn.Parameter` it wraps, so the gradient of a segment is read
    off `variable.value.grad` per variable. Zipping `module.parameters()` against the segment's
    variables instead would pair the wrong tensors: `parameters()` walks a path-alphabetical
    `ParameterDict` while the variables keep creation order.
    """

    name = "torch"

    def build_train_step(self, segments: Sequence[AdapterSegment]) -> InferenceFlow:
        """Build the step reading each segment's gradients off its own variables."""

        def step(batch: Mapping[str, Any]) -> dict[str, Any]:
            criteria: dict[str, Any] = {}
            for segment in segments:
                # Autograd accumulates into `.grad`, so a step that did not clear it would apply
                # the sum of every step so far -- including what another segment's backward pass
                # left behind on a shared variable.
                for variable in segment.variables:
                    variable.value.grad = None
                loss, values = segment.flow(batch)
                segment.optimizer.scale_loss(loss).backward()
                gradients = [variable.value.grad for variable in segment.variables]
                with torch.no_grad():
                    segment.optimizer.apply(gradients, segment.variables)
                criteria.update(values)
            return criteria

        return step

    def build_inference_step(self, flow: InferenceFlow, *, models: Sequence[Any] = ()) -> InferenceFlow:
        """Run the inference flow with autograd disabled, reading its variables from the closure."""
        inference: InferenceFlow = torch.no_grad()(flow)
        return inference


def _state_variables(segments: Sequence[AdapterSegment], owned: set[int]) -> list[Any]:
    """List every variable of every model that no optimizer owns, deduplicated, in model order.

    `Layer.variables` covers the moving statistics, the frozen weights and the `SeedGenerator`
    states, all of which the JAX step has to thread to stay unfrozen and unstale.
    """
    seen: set[int] = set()
    state = []
    for segment in segments:
        for model in segment.models:
            for variable in model.variables:
                if id(variable) in owned or id(variable) in seen:
                    continue
                seen.add(id(variable))
                state.append(variable)
    return state


def _assign(variables: Sequence[Any], values: Sequence[Any]) -> None:
    """Write the values a step computed back into their variables."""
    for variable, value in zip(variables, values, strict=True):
        variable.assign(value)


# Typed as constructors rather than as classes so that a type checker verifies each one against the
# protocol here, where the mismatch is one line, instead of at the call site of a missing method.
_ADAPTERS: dict[str, Callable[[], BackendAdapter]] = {
    "tensorflow": TensorFlowAdapter,
    "jax": JaxAdapter,
    "torch": TorchAdapter,
}


@cache
def select_backend_adapter() -> BackendAdapter:
    """Return the adapter of the active Keras backend, resolved once and reused.

    Returns:
        The adapter, the same instance on every call.

    Raises:
        ValueError: If the active backend has no adapter.
    """
    backend = keras.backend.backend()
    if backend not in _ADAPTERS:
        supported = ", ".join(repr(name) for name in _ADAPTERS)
        raise ValueError(f"Keras backend {backend!r} has no training adapter. Supported backends: {supported}.")
    return _ADAPTERS[backend]()


__all__ = [
    "AdapterSegment",
    "BackendAdapter",
    "Flow",
    "InferenceFlow",
    "JaxAdapter",
    "TensorFlowAdapter",
    "TorchAdapter",
    "select_backend_adapter",
]


if not TYPE_CHECKING:
    import sys

    from structcast.utils.lazy_import import LazySelectedImporter

    sys.modules[__name__] = LazySelectedImporter(__name__, globals())
