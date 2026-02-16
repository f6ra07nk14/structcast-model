"""Core functionality for StructCast-Model builders."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from logging import getLogger
from typing import Any, ClassVar, Generic, Self, TypeVar, cast

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    FilePath,
    SerializerFunctionWrapHandler,
    field_validator,
    model_serializer,
    model_validator,
)
from pydantic.alias_generators import to_pascal, to_snake
from structcast.core.constants import SPEC_SOURCE
from structcast.core.exceptions import SpecError
from structcast.core.instantiator import AddressPattern, ObjectPattern
from structcast.core.specifier import SPEC_CONSTANT, FlexSpec, SpecIntermediate, register_resolver
from structcast.core.template import extend_structure
from structcast.utils.base import check_elements
from structcast.utils.security import split_attribute

from structcast_model.builders.auto_name import AutoName
from structcast_model.utils.base import load_any, unique

logger = getLogger(__name__)

SPEC_EVAL = register_resolver("eval", lambda x: x)


class Serializable(BaseModel):
    """Base configuration."""

    model_config = ConfigDict(frozen=True, validate_default=True, extra="forbid", serialize_by_alias=True)


class WithExtra(Serializable):
    """Base class for configurations with extra fields allowed."""

    model_config = ConfigDict(extra="allow")

    @property
    def model_extra(self) -> dict[str, Any]:
        """Get extra fields set during validation.

        Returns:
            A dictionary of extra fields, or `None` if `config.extra` is not set to `"allow"`.
        """
        return cast(dict[str, Any], self.__pydantic_extra__)


_TEMPLATE_ALIASES = ["_jinja_", "_jinja_pipe_", "_jinja_yaml_", "_jinja_json_", "_jinja_group_"]


class Parameters(WithExtra):
    """Parameters for template formatting."""

    SHARED: dict[str, Any] = Field(default_factory=dict)
    """Shared parameters for all groups."""

    DEFAULT: dict[str, Any] = Field(default_factory=dict)
    """Default parameters for the default group."""

    @model_validator(mode="after")
    def _validate_parameters(self) -> Self:
        if "default" in self.model_extra:
            # logger.warning(
            #     'The default parameters should be defined in the "DEFAULT" field, '
            #     'not in the "default" key of extra fields. '
            #     'The "default" key in extra fields will be merged into the "DEFAULT" field.'
            # )
            self.DEFAULT.update(self.model_extra.pop("default"))
        for key, value in self.model_extra.items():
            if key in _TEMPLATE_ALIASES:
                raise SpecError(f'Key "{key}" is reserved for template aliases and cannot be used in parameters.')
            if not isinstance(value, dict):
                raise SpecError(f'Parameters for group "{key}" must be a dictionary but got: {value}')
        if duplicate_keys := set(self.DEFAULT) & set(self.SHARED):
            raise ValueError(f"Duplicate keys found between DEFAULT and SHARED parameters: {duplicate_keys}")
        return self

    @cached_property
    def template_kwargs(self) -> dict[str, dict[str, Any]]:
        """Get the template keyword arguments for formatting."""
        res = {k: {**v, **self.SHARED} for k, v in self.model_extra.items()}
        res["default"] = {**self.DEFAULT, **self.SHARED}
        return res

    def merge(self, other: dict[str, Any] | Parameters) -> Parameters:
        """Merge the given template keyword arguments with the parameters.

        Args:
            other (dict[str, Any] | Parameters): The template keyword arguments to merge, or another
                `Parameters` instance to merge with.

        Returns:
            A new `Parameters` instance with the merged template keyword arguments.
        """

        def _get(group: dict[str, dict[str, Any]], name: str) -> dict[str, Any]:
            return group.get(name, None) or {}

        if isinstance(other, Parameters):
            other = other.template_kwargs
        owner = self.template_kwargs
        return Parameters.model_validate({k: {**_get(owner, k), **_get(other, k)} for k in set(owner) | set(other)})


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
            if len(raw) == 3:
                inp, out, name_or_layer = raw
                name, layer = (name_or_layer, None) if isinstance(name_or_layer, str) else (None, name_or_layer)
            elif len(raw) == 4:
                inp, out, name, layer = raw
            else:
                raise SpecError("LayerBehavior tuple/list must have 3 or 4 elements.")
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


class UserDefinedLayer(Serializable):
    """User defined layer configuration."""

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


class TemplateLayer(WithExtra):
    """Template-like configuration."""

    PARAMETERS: Parameters = Field(default_factory=Parameters)
    """Parameters for template formatting."""

    @cached_property
    def _raw_and_user_defined_layers(self) -> tuple[dict[str, Any], dict[str, Any]]:
        target_fields = list(UserDefinedLayer.model_fields) + _TEMPLATE_ALIASES
        raw: dict[str, Any] = {}
        user_defined_layers: dict[str, Any] = {}
        for key, value in self.model_extra.items():
            (raw if key in target_fields else user_defined_layers)[key] = value
        return raw, user_defined_layers

    @property
    def raw(self) -> dict[str, Any]:
        """Get the raw fields for the `UserDefinedLayer`."""
        return self._raw_and_user_defined_layers[0]

    @property
    def user_defined_layers(self) -> dict[str, Any]:
        """Get the user-defined layers defined in the template."""
        return self._raw_and_user_defined_layers[1]

    def format(self, parameters: dict[str, dict[str, Any]] | Parameters) -> UserDefinedLayer:
        """Format the template with the given parameters.

        Args:
            parameters (dict[str, dict[str, Any]] | Parameters):
                The template keyword arguments to format the template with,
                or a `Parameters` instance containing the template keyword arguments.

        Returns:
            A `UserDefinedLayer` instance created from the formatted template.
        """
        template_kwargs = self.PARAMETERS.merge(parameters).template_kwargs
        raw = extend_structure(self.raw, template_kwargs=template_kwargs, default="default")
        return UserDefinedLayer.model_validate(raw)


class LayerIntermediate(Serializable):
    """Intermediate representation of a layer during the building process."""

    classname: str
    """The name of the layer class."""

    inputs: list[str]
    """The names of the input layers."""

    outputs: list[str]
    """The names of the output layers."""

    layers: dict[str, ObjectPattern | LayerIntermediate]
    """The sub-layers of the layer."""

    flow: list[tuple[FlexSpec, FlexSpec, str | None]]
    """The flow of the layer."""

    inference_flow: list[tuple[FlexSpec, FlexSpec, str | None]]
    """The inference flow of the layer."""

    structured_output: bool
    """Whether the output is structured."""

    layer_call_name: ClassVar[str | None] = None
    """The name of the method to call the layer, if applicable."""

    def _get_script(self) -> str:
        """Implement the method to get the script for the layer."""
        raise NotImplementedError("The _get_script method must be implemented in the subclass.")

    @cached_property
    def script(self) -> str:
        """Get the script for the layer."""
        return self._get_script()


LayerIntermediateT = TypeVar("LayerIntermediateT", bound=LayerIntermediate)


@dataclass(kw_only=True, slots=True)
class BaseBuilder(Generic[LayerIntermediateT]):
    """Base builder for building layers from templates."""

    user_defined_layer_type: ClassVar[type[LayerIntermediateT]] = LayerIntermediate

    raw: Any
    predefined_user_defined_layers: dict[str, Any] = field(default_factory=dict)
    current_path: str = ""
    from_references: dict[str, list[str]] = field(default_factory=dict)

    template: TemplateLayer = field(init=False)
    user_defined_layers: dict[str, Any] = field(init=False)

    def __post_init__(self) -> None:
        """Post-initialization to set up the template."""
        self.template = TemplateLayer.model_validate(self.raw)
        self.user_defined_layers = {**self.predefined_user_defined_layers, **self.template.user_defined_layers}

    def get_user_defined_layer(
        self,
        parts: list[str],
        parameters: dict[str, dict[str, Any]] | Parameters,
        classname: str,
    ) -> LayerIntermediateT:
        """Get the user-defined layer with the given parts and parameters.

        Args:
            parts (list[str]): The parts of the user-defined layer reference to resolve.
            parameters (dict[str, dict[str, Any]] | Parameters):
                The template keyword arguments to format the user-defined layer with,
                or a `Parameters` instance containing the template keyword arguments.
            classname (str): The name of the layer class to use for the user-defined layer.

        Returns:
            LayerIntermediateT: The user-defined layer as a `LayerIntermediateT` instance.
        """
        if not parts:
            return self(parameters, classname)
        first, *parts = parts
        if first not in self.user_defined_layers:
            raise SpecError(f'User-defined layer with key "{first}" not found in the template.')
        current_parts = self.from_references.get(self.current_path, None) or []
        circular_detected = first in current_parts
        current_parts = current_parts + [first]
        if circular_detected:
            raise SpecError(f"Circular reference detected for user-defined layer: {'.'.join(current_parts)}")
        return type(self)(
            raw=self.user_defined_layers[first],
            predefined_user_defined_layers=self.user_defined_layers,
            current_path=self.current_path,
            from_references={**self.from_references, self.current_path: current_parts},
        ).get_user_defined_layer(parts, parameters, classname)

    def _get_sublayer(self, parameters: Parameters, unit: UserLayer) -> tuple[str, LayerIntermediateT]:
        if unit.LAYER.CFG is not None:
            current_path = str(unit.LAYER.CFG)
            current_parts = self.from_references.get(current_path, None) or []
            subclassname = to_pascal(unit.LAYER.CFG.stem)
            if unit.LAYER.TYPE:
                subclassname, parts = f"{subclassname}{to_pascal(unit.LAYER.TYPE)}", split_attribute(unit.LAYER.TYPE)
            else:
                if "__root__" in current_parts:
                    raise SpecError(f"Circular reference detected for layer configuration: {self.from_references}")
                current_parts, parts = (current_parts + ["__root__"]), []
            builder = type(self)(
                raw=load_any(unit.LAYER.CFG),
                predefined_user_defined_layers=self.user_defined_layers,
                current_path=current_path,
                from_references={**self.from_references, current_path: current_parts},
            )
        elif unit.LAYER.TYPE is not None:
            subclassname, parts, builder = to_pascal(unit.LAYER.TYPE), split_attribute(unit.LAYER.TYPE), self
        else:
            raise SpecError(f"LAYER must have either CFG or TYPE specified but got: {unit.model_dump()}")
        return subclassname, builder.get_user_defined_layer(parts, parameters.merge(unit.LAYER.PARAM), subclassname)

    def __call__(self, parameters: dict[str, dict[str, Any]] | Parameters, classname: str) -> LayerIntermediateT:
        """Build the layer from the template with the given parameters and class name.

        Args:
            parameters (dict[str, dict[str, Any]] | Parameters):
                The template keyword arguments to format the template with,
                or a `Parameters` instance containing the template keyword arguments.
            classname (str): The name of the layer class to use for the built layer.

        Returns:
            LayerIntermediateT: The built layer as a `LayerIntermediateT` instance.
        """
        if not isinstance(parameters, Parameters):
            parameters = Parameters.model_validate(parameters)
        layer = self.template.format(parameters)
        sublayers: dict[str, ObjectPattern | LayerIntermediate] = {}
        naming = AutoName("_")

        def _create_flow(units: list[LayerBehavior]) -> list[tuple[FlexSpec, FlexSpec, str | None]]:
            flow: list[tuple[FlexSpec, FlexSpec, str | None]] = []
            for unit in units:
                if unit.LAYER is None:
                    if unit.NAME and unit.NAME not in sublayers:
                        raise SpecError(f'Layer with name "{unit.NAME}" not defined in the flow.')
                    name = unit.NAME
                else:
                    if isinstance(unit.LAYER, ObjectPattern):
                        if not isinstance((ptn := unit.LAYER.patterns[0]), AddressPattern):
                            raise SpecError(
                                "If LAYER is an ObjectPattern, the first pattern must be an AddressPattern "
                                f"but got: {unit.LAYER.model_dump()}"
                            )
                        subclassname, subinst = split_attribute(ptn.address)[-1], unit.LAYER
                    else:
                        subclassname, subinst = self._get_sublayer(parameters, unit)  # type: ignore[arg-type]
                    if (name := unit.NAME or naming(to_snake(subclassname))) in sublayers:
                        raise SpecError(f'Duplicate layer name "{name}" found in the flow.')
                    sublayers[name] = subinst
                if (has_inputs := unit.INPUTS is not None) and (has_outputs := unit.OUTPUTS is not None):
                    flow.append((unit.INPUTS, unit.OUTPUTS, name))
                elif has_inputs or has_outputs:
                    raise SpecError(
                        f"Both INPUTS and OUTPUTS must be specified together in the training/inference flow "
                        f"but got: {unit.model_dump()}"
                    )
            return flow

        return self.user_defined_layer_type(
            classname=classname,
            inputs=layer.INPUTS,
            outputs=layer.OUTPUTS,
            layers=sublayers,
            flow=_create_flow(layer.FLOW),
            inference_flow=_create_flow(layer.INFERENCE_FLOW),
            structured_output=layer.STRUCTURED_OUTPUT,
        )


# todo: custom script for object pattern
