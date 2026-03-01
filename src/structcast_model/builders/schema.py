"""Core Schema for StructCast-Model builders."""

from collections.abc import Sequence
from functools import cached_property
from logging import getLogger
from typing import TYPE_CHECKING, Any, ClassVar, Generic, Literal, Self, TypeVar

from pydantic import (
    Field,
    FilePath,
    PositiveInt,
    SerializerFunctionWrapHandler,
    TypeAdapter,
    ValidationError,
    field_validator,
    model_serializer,
    model_validator,
)
from structcast.core.base import Serializable, WithExtra
from structcast.core.constants import SPEC_SOURCE
from structcast.core.exceptions import SpecError
from structcast.core.instantiator import ObjectPattern
from structcast.core.specifier import SPEC_CONSTANT, FlexSpec, SpecIntermediate, register_resolver
from structcast.core.template import ALIAS_ALL, Parameters as BaseParameters, configure_jinja, extend_structure
from structcast.utils.base import check_elements
from structcast.utils.security import split_attribute, validate_attribute

from structcast_model.builders import jinja_filters
from structcast_model.utils.base import unique

logger = getLogger(__name__)

configure_jinja(filters={k: getattr(jinja_filters, k) for k in jinja_filters.__all__})

SPEC_EVAL = register_resolver("eval", lambda x: x)


class Parameters(BaseParameters):
    """Parameters for template formatting."""

    shared: dict[str, Any] = Field(default_factory=dict, alias="SHARED")
    """Shared parameters for all groups."""

    default: dict[str, Any] = Field(default_factory=dict, alias="DEFAULT")
    """Default parameters for the default group."""


class UserLayer(Serializable):
    """User layer configuration."""

    CFG: FilePath | None = None
    """Path to the layer configuration file."""

    TYPE: str | None = None
    """Type of the layer."""

    PARAM: Parameters = Field(default_factory=Parameters)
    """Parameters for template formatting."""


class LayerBehavior(Serializable):
    """Layer behavior configuration."""

    INPUTS: FlexSpec | None = None
    """Inputs of the layer."""

    OUTPUTS: FlexSpec | None = None
    """Outputs of the layer."""

    NAME: str | None = None
    """The name of the layer class or an instance of the layer."""

    LAYER: UserLayer | ObjectPattern | None = None
    """The name of the layer class or an instance of the layer."""

    @model_validator(mode="before")
    @classmethod
    def _validate_raw(cls, raw: Any) -> Any:
        """Validate the object data."""
        if isinstance(raw, LayerBehavior):
            return raw
        if isinstance(raw, (list, tuple)):
            if len(raw) == 2:
                inp, out = raw
                name, layer = None, None
            elif len(raw) == 3:
                inp, out, name_or_layer = raw
                name, layer = (name_or_layer, None) if isinstance(name_or_layer, str) else (None, name_or_layer)
            elif len(raw) == 4:
                inp, out, name, layer = raw
            else:
                raise SpecError(f"LayerBehavior tuple/list must have 2, 3, or 4 elements but got: {raw}.")
            return {"INPUTS": inp, "OUTPUTS": out, "NAME": name, "LAYER": layer}
        return raw

    @model_serializer(mode="wrap")
    def _serialize_model(self, handler: SerializerFunctionWrapHandler) -> list[Any]:
        """Serialize the model."""
        res = [handler(self.INPUTS), handler(self.OUTPUTS)]
        if self.NAME:
            res.append(self.NAME)
        if self.LAYER is not None:
            res.append(handler(self.LAYER))
        return res


def resolve_inputs(inputs: FlexSpec) -> list[str]:
    """Resolve the input layer names from the given inputs specification.

    The inputs specification can be in the following forms:
    - A dictionary mapping input names to specifications,
      where the input names are resolved from the specifications in the dictionary values.
    - A list of specifications, where the input names are resolved from the specifications in the list.
    - A source identifier specification, where the input name is resolved from the value of the source identifier.
      The source identifier must have a single string index as its value, which is used as the input name.

    Args:
        inputs (FlexSpec): The inputs specification to resolve.

    Returns:
        list[str]: A list of resolved input layer names.
    """

    def _resolve(spec: Any) -> list[str]:
        if isinstance(spec, SpecIntermediate):
            if spec.identifier == SPEC_SOURCE:
                if spec.value and isinstance(spec.value[0], str):
                    return [spec.value[0]]
                msg = f"First element of source identifier must be a string index but got: {inputs.model_dump()}"
                raise SpecError(msg)
            if spec.identifier in (SPEC_EVAL, SPEC_CONSTANT):
                return []
            raise SpecError(f"Unsupported spec identifier: {spec.identifier}")
        if isinstance(spec, dict):
            return [n for v in spec.values() for n in _resolve(v)]
        if isinstance(spec, list):
            return [n for item in spec for n in _resolve(item)]
        raise SpecError(f"Unsupported spec type: {type(spec)}")

    return _resolve(inputs.spec)


def resolve_outputs(outputs: FlexSpec) -> list[str]:
    """Resolve the output layer names from the given outputs specification.

    The outputs specification can be in the following forms:
    - A dictionary mapping output names to specifications, where the output names are the keys of the dictionary.
    - A list of specifications, where the output names are resolved from the specifications in the list.
    - A source identifier specification, where the output name is resolved from the value of the source identifier.
      The source identifier must have a single string index as its value, which is used as the output name.

    Args:
        outputs (FlexSpec): The outputs specification to resolve.

    Returns:
        list[str]: A list of resolved output layer names.
    """
    if isinstance(outputs.spec, dict):
        return list(outputs.spec)

    def _resolve(spec: Any) -> list[str]:
        if isinstance(spec, dict):
            raise SpecError(f"Outputs cannot be a dictionary in list form but got: {outputs.model_dump()}")
        if isinstance(spec, list):
            return [n for v in spec for n in _resolve(v)]
        if isinstance(spec, SpecIntermediate):
            if spec.identifier != SPEC_SOURCE:
                raise SpecError(f"Outputs must be consist of source identifier but got: {outputs.model_dump()}")
            if spec.value and len(spec.value) == 1 and isinstance(spec.value[0], str):
                return [spec.value[0]]
            msg = f"Outputs must be a source identifier with a single string index but got: {outputs.model_dump()}"
            raise SpecError(msg)
        msg = f"Outputs must be a dictionary or consist of a source identifier but got: {outputs.model_dump()}"
        raise SpecError(msg)

    return _resolve(outputs.spec)


def resolve_flow(flow: list[LayerBehavior]) -> tuple[list[str], list[str]]:
    """Resolve the input and output layer names from the given flow.

    Args:
        flow (list[LayerBehavior]): The flow to resolve.

    Returns:
        tuple[list[str], list[str]]: A tuple containing the list of resolved input/output names.
    """
    inputs, outputs = [], []
    for unit in flow:
        if unit.INPUTS is not None:
            inputs += [n for n in resolve_inputs(unit.INPUTS) if n not in outputs]
        outputs += [] if unit.OUTPUTS is None else resolve_outputs(unit.OUTPUTS)
    return unique(inputs), unique(outputs)


def _validate_imports(data: Any) -> Any:
    """Validate the imports."""
    if isinstance(data, dict):
        return {k: set(check_elements(v)) for k, v in data.items()}
    try:
        return {k: {None} for k in check_elements(TypeAdapter(str | set[str] | Sequence[str]).validate_python(data))}
    except ValidationError:
        pass
    return data


class UserDefinedLayer(Serializable):
    """User defined layer configuration."""

    IMPORTS: dict[str, set[str | None]] = Field(default_factory=dict)
    """Imports required for the layer,
    where the keys are module names and the values are sets of imported names from the corresponding modules.

    The imported names can be `None`, which indicates that the entire module is imported.
    """

    INPUTS: list[str] = Field(default_factory=list)
    """Inputs of the layer."""

    OUTPUTS: list[str] = Field(default_factory=list)
    """Outputs of the layer."""

    FLOW: list[LayerBehavior] = Field(default_factory=list)
    """Flow of the layer."""

    INFERENCE_FLOW: list[LayerBehavior] = Field(default_factory=list)
    """Inference flow of the layer. If not specified, the inference flow will be the same as the flow."""

    STRUCTURED_OUTPUT: bool = False
    """Whether the output is structured."""

    @field_validator("IMPORTS", mode="before")
    @classmethod
    def _validate_imports(cls, data: Any) -> Any:
        """Validate the imports."""
        return _validate_imports(data)

    @field_validator("INPUTS", "OUTPUTS", mode="before")
    @classmethod
    def _validate_inout_format(cls, data: Any) -> Any:
        """Validate the data."""
        return check_elements(data)

    @model_validator(mode="after")
    def _validate_user_defined_layer(self) -> Self:
        """Validate the user-defined layer."""
        train_inputs, train_outputs = resolve_flow(self.FLOW)
        if not self.INPUTS:
            self.INPUTS.extend(train_inputs)
        if not self.OUTPUTS:
            self.OUTPUTS.extend(train_outputs)
        if unknown := set(self.INPUTS) - set(train_inputs):
            raise SpecError(f"Unknown inputs found: {unknown}.")
        if missing := set(train_inputs) - set(self.INPUTS):
            raise SpecError(f"Missing inputs found: {missing}.")
        if unknown := set(self.OUTPUTS) - set(train_outputs):
            raise SpecError(f"Unknown outputs found: {unknown}.")
        if self.INFERENCE_FLOW:
            infer_inputs, infer_outputs = resolve_flow(self.INFERENCE_FLOW)
            if unknown := set(self.INPUTS) - set(infer_inputs):
                raise SpecError(f"Unknown inputs found in INFERENCE_FLOW: {unknown}.")
            if missing := set(infer_inputs) - set(self.INPUTS):
                raise SpecError(f"Missing inputs found in INFERENCE_FLOW: {missing}.")
            if unknown := set(self.OUTPUTS) - set(infer_outputs):
                raise SpecError(f"Unknown outputs found in INFERENCE_FLOW: {unknown}.")
        return self


class OptimizerBehavior(Serializable):
    """Optimizer behavior configuration."""

    NAME: str | None = None
    """The name of the optimizer class or an instance of the optimizer."""

    OPTIMIZER: ObjectPattern
    """The name of the optimizer class or an instance of the optimizer."""

    LAYERS: list[str] = Field(default_factory=list, min_length=1)
    """The trainable layers to apply the optimizer to."""

    CLIP: ObjectPattern | None = None
    """Gradient clipping configuration, which can be an instance of the gradient clipping configuration
    or a pattern to instantiate the gradient clipping configuration."""

    @model_validator(mode="before")
    @classmethod
    def _validate_raw(cls, raw: Any) -> Any:
        """Validate the object data."""
        if isinstance(raw, OptimizerBehavior):
            return raw
        if isinstance(raw, (list, tuple)):
            if len(raw) == 4:
                name, opt, layers, clip = raw
            elif len(raw) == 3:
                if isinstance(raw[0], str) or raw[0] is None:
                    name, opt, layers = raw
                    clip = None
                else:
                    opt, layers, clip = raw
                    name = None
            elif len(raw) == 2:
                opt, layers = raw
                name, clip = None, None
            else:
                raise SpecError(f"The tuple/list for OptimizerBehavior must have 2, 3, or 4 elements but got: {raw}.")
            return {"NAME": name, "OPTIMIZER": opt, "LAYERS": layers, "CLIP": clip}
        return raw

    @field_validator("LAYERS", mode="before")
    @classmethod
    def _validate_trainable_layers(cls, data: Any) -> Any:
        """Validate the trainable layers."""
        return check_elements(data)

    @field_validator("LAYERS", mode="after")
    @classmethod
    def _validate_trainable_layers_legal(cls, data: list[str]) -> list[str]:
        """Validate that the trainable layers are legal."""
        for layer in data:
            validate_attribute(layer)
        return data

    @model_serializer(mode="wrap")
    def _serialize_model(self, handler: SerializerFunctionWrapHandler) -> list[Any]:
        """Serialize the model."""
        res = [handler(self.OPTIMIZER), handler(self.LAYERS)]
        if self.CLIP:
            res.append(handler(self.CLIP))
        if self.NAME:
            res = [self.NAME] + res
        return res

    @cached_property
    def models(self) -> set[str]:
        """Get the models to apply the optimizer behavior to."""
        return {split_attribute(L)[0] for L in self.LAYERS}


class BackwardBehavior(WithExtra):
    """Backward behavior configuration."""

    NAME: str | None = None
    """The name of the backward layer class or an instance of the backward layer."""

    LOSS: str
    """The target loss to optimize."""

    OPTIMIZERS: list[OptimizerBehavior] = Field(default_factory=list, min_length=1)
    """The ordered list of optimizer behaviors to apply during backward pass."""

    @model_validator(mode="before")
    @classmethod
    def _validate_raw(cls, raw: Any) -> Any:
        """Validate the object data."""
        if isinstance(raw, BackwardBehavior):
            return raw
        if isinstance(raw, (list, tuple)):
            if len(raw) == 4:
                name, loss, opts, others = raw
            elif len(raw) == 3:
                if isinstance(raw[0], str) or raw[0] is None:
                    name, loss, opts = raw
                    others = {}
                else:
                    loss, opts, others = raw
                    name = None
            elif len(raw) == 2:
                loss, opts = raw
                name, others = None, {}
            else:
                raise SpecError(f"The tuple/list for BackwardBehavior must have 2, 3, or 4 elements but got: {raw}.")
            return {"NAME": name, "LOSS": loss, "OPTIMIZERS": opts, **others}
        return raw

    @model_serializer(mode="wrap")
    def _serialize_model(self, handler: SerializerFunctionWrapHandler) -> list[Any]:
        """Serialize the model."""
        res = [handler(self.LOSS), handler(self.OPTIMIZERS)]
        if self.NAME:
            res = [self.NAME] + res
        res.extend(handler(self.model_extra))
        return res

    @cached_property
    def models(self) -> set[str]:
        """Get the models to apply the backward behavior to."""
        return {m for b in self.OPTIMIZERS for m in b.models}


class UserDefinedBackward(Serializable):
    """User defined backward configuration."""

    IMPORTS: dict[str, set[str | None]] = Field(default_factory=dict)
    """Imports required for the backward behavior,
    where the keys are module names and the values are sets of imported names from the corresponding modules.

    The imported names can be `None`, which indicates that the entire module is imported.
    """

    BACKWARDS: list[BackwardBehavior] = Field(default_factory=list, min_length=1)
    """Backward behavior configuration."""

    LOSSES: list[str] = Field(default_factory=list)
    """The losses to optimize. If not specified, the losses will be inferred from the BACKWARDS field."""

    MODELS: list[str] = Field(default_factory=list)
    """The models to apply the backward behavior to.

    If not specified, the backward behavior will be applied to all models involved in the flows of the layers
    defined in the same user-defined layer configuration."""

    MIXED_PRECISION: bool | dict[str, Any] = False
    """Whether to use mixed precision during backward pass.

    If the value is a dictionary, it will be used as the keyword arguments for configuring mixed precision context.
    """

    MIXED_PRECISION_TYPE: Literal["bfloat16", "float16"] | None = None
    """The mixed precision type to use during backward pass when mixed precision is enabled."""

    ACCUMULATE_GRADIENTS: PositiveInt | None = None
    """Whether to accumulate gradients for multiple steps before updating the parameters,
    and the number of steps to accumulate for."""

    @field_validator("IMPORTS", mode="before")
    @classmethod
    def _validate_imports(cls, data: Any) -> Any:
        """Validate the imports."""
        return _validate_imports(data)

    @field_validator("LOSSES", "MODELS", mode="before")
    @classmethod
    def _validate_losses_models(cls, data: Any) -> Any:
        """Validate the losses and models."""
        return check_elements(data)

    @model_validator(mode="after")
    def _validate_user_defined_backward(self) -> Self:
        """Validate the user-defined backward configuration."""
        losses = {b.LOSS for b in self.BACKWARDS}
        if not self.LOSSES:
            self.LOSSES.extend(list(losses))
        if unknown := set(self.LOSSES) - losses:
            raise SpecError(f"Unknown losses found in LOSSES: {unknown}.")
        if missing := losses - set(self.LOSSES):
            raise SpecError(f"Missing losses found in LOSSES: {missing}.")
        models = {m for b in self.BACKWARDS for m in b.models}
        if not self.MODELS:
            self.MODELS.extend(list(models))
        if unknown := set(self.MODELS) - models:
            raise SpecError(f"Unknown models found in MODELS: {unknown}.")
        if missing := models - set(self.MODELS):
            raise SpecError(f"Missing models found in MODELS: {missing}.")
        if (isinstance(self.MIXED_PRECISION, dict) or self.MIXED_PRECISION) and self.MIXED_PRECISION_TYPE is None:
            raise SpecError("MIXED_PRECISION_TYPE must be specified when MIXED_PRECISION is enabled.")
        return self


SerializableT = TypeVar("SerializableT", bound=Serializable)


class _Template(WithExtra, Generic[SerializableT]):
    PARAMETERS: Parameters = Field(default_factory=Parameters)
    """Parameters for template formatting."""

    target_type: ClassVar[type[SerializableT]] = Serializable

    @cached_property
    def _raw_and_others(self) -> tuple[dict[str, Any], dict[str, Any]]:
        target_fields = list(self.target_type.model_fields) + ALIAS_ALL
        raw: dict[str, Any] = {}
        others: dict[str, Any] = {}
        for key, value in self.model_extra.items():
            (raw if key in target_fields else others)[key] = value
        return raw, others

    @property
    def raw(self) -> dict[str, Any]:
        """Get the raw fields for the target type."""
        return self._raw_and_others[0]

    @property
    def others(self) -> dict[str, Any]:
        """Get the other fields that are not in the target type."""
        return self._raw_and_others[1]

    def __call__(
        self,
        parameters: dict[str, dict[str, Any]] | Parameters | None = None,
        *,
        merged: bool = True,
    ) -> SerializableT:
        """Format the template with the given parameters.

        Args:
            parameters (dict[str, dict[str, Any]] | Parameters | None):
                The template keyword arguments to format the template with,
                or a `Parameters` instance containing the template keyword arguments.
            merged (bool, optional): Whether to merge the given parameters with the template's own `PARAMETERS`
                before formatting. Defaults to `True`.

        Returns:
            An instance of the target type created from the formatted template.
        """
        if merged:
            parameters = Parameters.create(self.PARAMETERS, parameters)
        elif parameters is None:
            parameters = self.PARAMETERS
        else:
            parameters = Parameters.create(parameters)
        return self.target_type.model_validate(extend_structure(self.raw, template_kwargs=parameters))


class TemplateLayer(_Template[UserDefinedLayer]):
    """Template for user-defined layers."""

    target_type: ClassVar[type[UserDefinedLayer]] = UserDefinedLayer


class TemplateBackward(_Template[UserDefinedBackward]):
    """Template for user-defined backwards."""

    target_type: ClassVar[type[UserDefinedBackward]] = UserDefinedBackward


__all__ = [
    "SPEC_EVAL",
    "BackwardBehavior",
    "LayerBehavior",
    "OptimizerBehavior",
    "Parameters",
    "TemplateBackward",
    "TemplateLayer",
    "UserDefinedBackward",
    "UserDefinedLayer",
    "UserLayer",
    "resolve_flow",
    "resolve_inputs",
    "resolve_outputs",
]


if not TYPE_CHECKING:
    import sys

    from structcast.utils.lazy_import import LazySelectedImporter

    sys.modules[__name__] = LazySelectedImporter(__name__, globals())
