# The learner schema splits into a framework-neutral base and torch extensions

`LearnerBehavior.CLIP`, `UserDefinedLearner.MIXED_PRECISION`, and `MIXED_PRECISION_TYPE` encode torch-only
machinery: `CLIP` resolves to a callable applied to `.grad` buffers between backward and step, and the mixed
precision pair configures `torch.amp` autocast plus a `GradScaler`. Neither concept exists in Flax/optax —
clipping is a stage of the optax transformation chain expressed inside the `OPTIMIZER` pattern, and mixed
precision is a model-construction property (`dtype`/`param_dtype` layer kwargs), not a learner behavior. Keeping
those fields in the shared schema would force every non-torch builder to reject them with bespoke validators.

Instead the shared schema keeps only portable concepts, and torch re-adds its own:

- `builders/schema.py` keeps `LearnerBehavior` (without `CLIP`) and a generic
  `UserDefinedLearner(Serializable, Generic[LearnerBehaviorT])` whose `LEARNERS` is `list[LearnerBehaviorT]`
  (without the mixed precision fields and their float16-only validator). `ACCUMULATE_GRADIENTS` stays in the
  base: gradient accumulation is portable.
- `builders/torch.py` defines `TorchLearnerBehavior(LearnerBehavior)` adding `CLIP`,
  `TorchUserDefinedLearner(UserDefinedLearner[TorchLearnerBehavior])` adding the mixed precision fields and
  validator, and `TorchTemplateLearner(Template[TorchUserDefinedLearner])`.
- Unknown fields fail by construction: the base schema inherits `extra="forbid"`, so a Flax learner YAML using
  `CLIP` or `MIXED_PRECISION` is rejected by pydantic with a field-not-permitted error, without any
  framework-aware validator.

## The template type is part of the seam, not a convenience

`Template.raw` partitions YAML keys by `target_type.model_fields` (`schema.py`): keys that are not schema fields
fall through to `others` and reach the layer builder. If the torch builder kept the base `TemplateLearner`,
`MIXED_PRECISION` would silently land in `others` and be treated as a layer. Every learner builder therefore
declares its template class through a `template_type` class variable next to the existing
`user_defined_learner_layer_type` / `layer_builder_type` bindings.

## Optimizer segments become a dataclass, not a wider tuple

`LearnerIntermediate.flow` carried optimizer segments as positional 6-tuples
`(loss, backward_kw, optimizer, clip, scaler, trainable_layers)` — two of the six slots are torch-only. The
shared intermediate now carries an `OptimizerSegment` dataclass (loss, backward kwargs, optimizer, trainable
layers) with `TorchOptimizerSegment` adding `clip` and `scaler`, and `LearnerIntermediate` is generic over the
segment type. The segment construction moves behind a `_build_segment` hook on `BaseLearnerBuilder`, so the base
`__call__` never reads torch fields and the `len(unit) == 6` checks become `isinstance(unit, OptimizerSegment)`.
`mixed_precision_type` moves from the shared intermediate onto `TorchLearnerIntermediate` the same way.

## Trade-offs

- Breaking for anyone importing `UserDefinedLearner` to validate torch YAML or introspecting `flow` tuples; the
  torch classes keep the old behavior under new names, and the flat routers re-export both sets.
- Generic pydantic models bind the house `ClassVar` + `cast` pattern one level deeper (the template's
  `target_type` is a parametrized class object); the pydantic mypy plugin handles this, plain `mypy --strict`
  users of the schema module would not notice a difference.
