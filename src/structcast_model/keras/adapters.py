"""Backend adapters running the training and inference steps of a Keras learner.

One generated learner drives every Keras backend, so every mechanic the three backends disagree
about -- gradient computation, optimizer application, variable state handling, step compilation --
lives behind the `BackendAdapter` selected once from `keras.backend.backend()`.

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

Flow = Callable[..., tuple[Any, dict[str, Any]]]
"""A training flow: it returns the loss to differentiate and the criteria.

The batch reaches it as keyword arguments, one per input name, which is also how
the steps built here take it: a positional batch mapping would bind the entries by declaration
order instead. `Callable[..., ...]` because the names are the learner's, so no signature can spell
them.

The flow closes over its models and is written in `keras.ops`, so the same closure runs on every
backend; the adapter decides how the loss is differentiated and how the update is applied.
"""

InferenceFlow = Callable[..., dict[str, Any]]
"""An inference flow: it takes the batch by name and returns the criteria, differentiating nothing."""


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

    compile_kw: Mapping[str, Any] | None
    """How the steps this adapter builds are compiled: the compiler's keyword arguments, or None.

    None leaves them eager, an empty mapping compiles with the backend compiler's own defaults, and
    a non-empty one is passed to it. Read while a step is built, so it must be set before the
    learner is built; `select_backend_adapter` documents how `scm keras train --compile` sets it.

    The protocol's one data member, which the CLI needs -- it writes the choice through this typed
    return -- and which costs `issubclass`: a runtime-checkable protocol with a non-method member
    still answers `isinstance`, and refuses `issubclass` by design.
    """

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
        """Build the training step running every segment: its flow, its gradients, its update.

        Compiled only when `compile_kw` asks for it; eager otherwise, on every backend.

        The returned step takes the batch by name and returns the merged criteria of every segment. A
        single-process adapter reduces nothing: whoever runs a distributed cell must reduce the
        criteria across replicas before they reach the tracker, since the Keras tracker -- unlike its
        torch twin -- never all-reduces, and a strategy that skips this must fail its own tests rather
        than log per-replica values.

        The criteria are merged last-wins, so keeping the names of two segments distinct is the
        caller's job: an adapter only sees the names at run time and cannot detect a clash here.
        `prepare` must have run first -- only the JAX backend refuses an unbuilt optimizer, while
        TensorFlow and torch would build it inside the step, inside its graph when compiled.

        Args:
            segments: The optimizer segments, applied in order, each against its own variables.

        Returns:
            The training step.
        """

    def build_inference_step(self, flow: InferenceFlow, *, models: Sequence[Any] = ()) -> InferenceFlow:
        """Build the inference step, which updates no variable; compiled only when `compile_kw` asks.

        Args:
            flow: The inference flow.
            models: The Keras models the flow runs, which JAX needs to thread their variables
                through the jitted step; the other backends read them from the closure and ignore
                this argument.

        Returns:
            The inference step.
        """


class _Adapter:
    """The optimizer preparation the three adapters share, and the step compilation torch has none of."""

    name: str

    compile_kw: Mapping[str, Any] | None = None
    """How this adapter compiles its steps, None -- eager -- until something asks for more.

    An attribute rather than a constructor argument because nobody constructs an adapter:
    `select_backend_adapter` is `@cache`d and takes none, and a generated learner calls it that way
    inside its own constructor, which is where its steps are built. Whoever chooses -- `scm keras
    train --compile` -- therefore writes on that one cached instance before building the learner,
    and clears it again once the steps exist.
    """

    def _compile_step(self, step: Callable[..., Any]) -> Callable[..., Any]:
        """Return *step* as it is, which is every step the torch backend runs.

        Keras' torch backend compiles nothing here, deliberately and not by omission: these steps
        assign to Keras variables and run a Keras optimizer, which `torch.compile` would graph-break
        on rather than accelerate, so there is no compiler to hand keyword arguments to. Asking for
        one is refused instead of dropped -- a run that silently ignored `--compile` would report a
        compiled run it never had.

        Args:
            step: The step to compile.

        Returns:
            The step, unchanged.

        Raises:
            ValueError: If any compilation was asked for.
        """
        if self.compile_kw is None:
            return step
        raise ValueError(
            f"The {self.name!r} Keras backend builds no compiled step, so it cannot be compiled with "
            f"{dict(self.compile_kw)}. Drop --compile, or run the same learner on the tensorflow or jax backend."
        )

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
            # Building here rather than on the first update keeps every slot variable out of a
            # compiled step, where TensorFlow forbids creating variables and JAX would trace them.
            segment.optimizer.build(segment.variables)


class TensorFlowAdapter(_Adapter):
    """Adapter differentiating with `tf.GradientTape` and applying the optimizer statefully.

    `optimizer.stateless_apply` refuses to run on TensorFlow by design, so the stateful `apply` is
    the only path; a compiled step is one `tf.function` around the whole of it, so the tape, the
    gradients and the update land in one graph.
    """

    name = "tensorflow"

    @staticmethod
    def _tf_function_kw(compile_kw: Mapping[str, Any] | None) -> dict[str, Any]:
        """The `tf.function` arguments of a `--compile` mapping, minus the one no step here can take.

        `input_signature` replaces the traced function's parameters with the signature it is given,
        and every step built here takes its batch by name, so a forwarded one would trace a step
        unable to bind its own batch: it is the step's contract, and is dropped rather than honored.
        `keras/distributed.py` reads it through this class for the same reason -- the call its `dp`
        preset traces takes the batch by name too.

        Args:
            compile_kw: The run's `--compile` mapping, or None for an eager run.

        Returns:
            The keyword arguments to trace with, empty when nothing was asked for.
        """
        return {name: value for name, value in (compile_kw or {}).items() if name != "input_signature"}

    def _compile_step(self, step: Callable[..., Any]) -> Callable[..., Any]:
        """Trace *step* into a `tf.function`, or leave it eager when nothing asked for a graph."""
        if self.compile_kw is None:
            return step
        return tf.function(step, **self._tf_function_kw(self.compile_kw))

    def build_train_step(self, segments: Sequence[AdapterSegment]) -> InferenceFlow:
        """Build the step running every segment's tape, gradients and update, traced when asked."""

        def step(**batch: Any) -> dict[str, Any]:
            criteria: dict[str, Any] = {}
            for segment in segments:
                with tf.GradientTape() as tape:
                    loss, values = segment.flow(**batch)
                    # Outside `fit()` nobody scales the loss for us. `scale_loss` returns the loss
                    # untouched on an optimizer without a loss scale, so no branch is needed.
                    scaled = segment.optimizer.scale_loss(loss)
                gradients = tape.gradient(scaled, segment.variables)
                segment.optimizer.apply(gradients, segment.variables)
                criteria.update(values)
            return criteria

        return self._compile_step(step)

    def build_inference_step(self, flow: InferenceFlow, *, models: Sequence[Any] = ()) -> InferenceFlow:
        """Run the inference flow, traced when asked, reading its variables from the closure."""
        return self._compile_step(flow)


class JaxAdapter(_Adapter):
    """Adapter differentiating with `jax.value_and_grad` and applying the optimizer statelessly.

    The stateful path is unavailable: `LossScaleOptimizer.apply` raises an `UnexpectedTracerError`
    under a trace. Every variable the step reads is therefore threaded through the step as an
    argument and bound with a `keras.StatelessScope`, the idiom of Keras' own JAX trainer, which is
    what lets a jitted step stay correct: a variable read from the closure instead would be traced
    as a constant and freeze at its first value. That includes the state the flow updates without
    owning: the moving statistics of a normalization layer and the state of every `SeedGenerator`,
    which `Layer.variables` lists, and which a dropped thread would silently freeze into one
    repeated mask. The threading is unconditional, so the eager step runs the identical arithmetic.
    """

    name = "jax"

    def _compile_step(self, step: Callable[..., Any]) -> Callable[..., Any]:
        """Jit *step*, or leave it eager when nothing asked for a compiled one.

        Only the pure inner function is ever handed here: the variable reads and writes around it
        stay Python either way, which is what lets the same threading run compiled and eager.
        """
        if self.compile_kw is None:
            return step
        # Both spellings of both contract arguments go, as `cmd_flax` drops them for `nnx.jit`: one
        # mapping is splatted into a training step and an inference step whose positional signatures
        # differ, so a `donate_argnums` meant for the first would donate the second's live weights.
        fixed = {"static_argnames", "static_argnums", "donate_argnames", "donate_argnums"}
        return jax.jit(step, **{name: value for name, value in self.compile_kw.items() if name not in fixed})

    def build_train_step(self, segments: Sequence[AdapterSegment]) -> InferenceFlow:
        """Build one step threading trainable, state and optimizer values through, jitted when asked."""
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
                    loss, values = segment.flow(**batch)
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

        jitted = self._compile_step(step)

        # The batch is gathered back into one mapping for the jitted call: the state lists are the
        # positional arguments the trace is built around, and a batch spread over keywords there
        # would move with every learner's input names.
        def train_step(**batch: Any) -> dict[str, Any]:
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
        """Build one step threading the models' variables through, as the training step does.

        Jitted when `compile_kw` asks for it. A jitted closure reading its variables directly would
        trace them as constants and answer with the weights of the step it was first traced on, so
        every variable is threaded as an argument and bound with a `keras.StatelessScope`. The
        scope's writes are dropped instead of assigned back, which is exactly inference: whatever
        the flow touches -- moving statistics, seeds -- stays as the training step left it. Without
        the models there is nothing to thread, so the flow runs eagerly rather than freezing at its
        first weights, and a compilation asked for there is refused rather than dropped.

        Raises:
            ValueError: If compilation was asked for and there is no model to thread.
        """
        variables = [variable for model in models for variable in model.variables]
        if not variables:
            if self.compile_kw is not None:
                raise ValueError(
                    "A JAX inference step with no model to thread its variables through cannot be jitted: the "
                    "jitted closure would answer with the weights of its first trace forever. Got: "
                    f"{dict(self.compile_kw)}. Pass the models the flow runs, or drop --compile."
                )
            return flow

        def inference(values: list[Any], batch: Mapping[str, Any]) -> dict[str, Any]:
            with keras.StatelessScope(state_mapping=list(zip(variables, values, strict=True))):
                return flow(**batch)

        jitted = self._compile_step(inference)

        def inference_step(**batch: Any) -> dict[str, Any]:
            return jitted([variable.value for variable in variables], batch)

        return inference_step


class TorchAdapter(_Adapter):
    """Adapter differentiating with autograd and applying the Keras optimizer under `no_grad`.

    A Keras variable *is* the `torch.nn.Parameter` it wraps, so the gradient of a segment is read
    off `variable.value.grad` per variable. Zipping `module.parameters()` against the segment's
    variables instead would pair the wrong tensors: `parameters()` walks a path-alphabetical
    `ParameterDict` while the variables keep creation order.

    Its steps are the eager ones: it inherits `_Adapter._compile_step`, which compiles nothing and
    refuses to be asked to -- see there for why this backend has no compiler to pass arguments to.
    """

    name = "torch"

    def build_train_step(self, segments: Sequence[AdapterSegment]) -> InferenceFlow:
        """Build the eager step reading each segment's gradients off its own variables."""

        def step(**batch: Any) -> dict[str, Any]:
            criteria: dict[str, Any] = {}
            for segment in segments:
                # Autograd accumulates into `.grad`, so a step that did not clear it would apply
                # the sum of every step so far -- including what another segment's backward pass
                # left behind on a shared variable.
                for variable in segment.variables:
                    variable.value.grad = None
                loss, values = segment.flow(**batch)
                segment.optimizer.scale_loss(loss).backward()
                gradients = [variable.value.grad for variable in segment.variables]
                with torch.no_grad():
                    segment.optimizer.apply(gradients, segment.variables)
                criteria.update(values)
            return criteria

        return self._compile_step(step)

    def build_inference_step(self, flow: InferenceFlow, *, models: Sequence[Any] = ()) -> InferenceFlow:
        """Run the inference flow with autograd disabled, reading its variables from the closure.

        The refusal comes first, so a request this backend cannot honor is raised about the flow it
        was made for rather than about the `no_grad` wrapper put around it.

        Raises:
            ValueError: If compilation was asked for -- `_Adapter._compile_step` says why.
        """
        inference: InferenceFlow = self._compile_step(flow)
        return torch.no_grad()(inference)


def _exchange(variable: Any, average: Any) -> None:
    """Trade one variable's value for its average, through a copy so the trade is exact.

    Not the add-and-subtract dance of `keras.callbacks.SwapEMAWeights._tf_swap_variables`: that one
    spares the temporary at the cost of rounding both values, and a run has to resume training from
    the weights it paused on. It needs no `tf.distribute` branch either -- the swap runs on the host
    under a `MirroredStrategy` (`keras/distributed.py`, `wrap_steps`, replicates the inner flow
    alone), and assigning a `MirroredVariable` from there updates every replica.
    """
    held = keras.ops.copy(variable)
    variable.assign(average)
    average.assign(held)


def _ema_pairs(optimizers: Sequence[Any]) -> list[tuple[Any, Any]]:
    """List the (variable, average) pairs one swap should trade, in the order it trades them.

    Three things are skipped, each for a reason a swap cannot recover from:

    - An optimizer that has never applied. `_model_variables_moving_average` is zero-initialized and
      first written by an `apply`, so evaluating before a run has trained -- `Trainer.evaluate()` on
      a fresh model -- would measure an all-zero model. Keras seeds the average from the weights on
      every apply while `iterations` is still 0, so a zero count also means "the average is the
      weights", and skipping is the same answer either way.
    - A variable whose gradient overwrites it. `add_optimizer_variables` puts `None` in place of the
      average of an `overwrite_with_gradient` variable, so there is nothing to trade.
    - A variable a previous optimizer already claimed. Two optimizers averaging one model would
      otherwise trade it twice, which puts the *second* average in the model on the way in and
      leaves the *first* one in it on the way out -- a corruption, not a wrong reading. The first
      optimizer in the sequence wins; the learner builder refuses the configuration that gets here.
    """
    pairs: list[tuple[Any, Any]] = []
    claimed: set[int] = set()
    for optimizer in optimizers:
        # The same host read `training_step` makes of this counter, on the same variable.
        if not int(keras.ops.convert_to_numpy(optimizer.iterations)):
            continue
        # Paired positionally against the optimizer's own list, as `keras.callbacks.SwapEMAWeights`
        # does: both are built from the variables `build` was given.
        for variable, average in zip(
            optimizer._trainable_variables, optimizer._model_variables_moving_average, strict=True
        ):
            if average is None or id(variable) in claimed:
                continue
            claimed.add(id(variable))
            pairs.append((variable, average))
    return pairs


def swap_ema_weights(optimizers: Sequence[Any]) -> None:
    """Exchange the trainable variables of each optimizer with the moving averages it keeps.

    A Keras optimizer blends its EMA into `_model_variables_moving_average` on every `apply` and
    only writes it back into the weights at `finalize_variable_values`, so a flow reading the
    variables as they stand reports the raw weights. Called before an inference flow and again after
    it -- which is what a generated learner's `inference_step` does, under a `try`/`finally` -- this
    runs the flow on the average and leaves the weights exactly as it found them.

    It is all-or-nothing: a swap that fails partway -- an allocation failure inside the copy, above
    all -- puts back what it had already traded before re-raising, because the caller's `finally`
    only knows how to undo a swap that finished. What is not traded is listed by `_ema_pairs`.

    Args:
        optimizers: The built optimizers whose average to swap in, each already unwrapped from any
            `keras.optimizers.LossScaleOptimizer` -- that wrapper refuses `use_ema` and keeps no
            average, so only the inner optimizer has one. An optimizer without an EMA has no
            averages to pair against and raises rather than passing silently.
    """
    pairs = _ema_pairs(optimizers)
    traded = 0
    try:
        for variable, average in pairs:
            _exchange(variable, average)
            traded += 1
    except BaseException:
        for variable, average in pairs[:traded]:
            _exchange(variable, average)
        raise


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

    Cached, so it is one adapter per process, and that is what `compile_kw` travels on: a generated
    learner builds its steps in its own constructor, calling this with no arguments, so a caller
    choosing how those steps are compiled -- `scm keras train --compile` -- sets `compile_kw` on the
    adapter this returns *before* it builds the learner. Setting it afterwards changes nothing: the
    steps are already built, and `cache_clear()` drops the choice with the adapter that carried it.

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
    "swap_ema_weights",
]


if not TYPE_CHECKING:
    import sys

    from structcast.utils.lazy_import import LazySelectedImporter

    sys.modules[__name__] = LazySelectedImporter(__name__, globals())
