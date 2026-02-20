"""Core functionality for StructCast-Model builders."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from functools import cached_property
from hashlib import sha256
from json import dumps as json_dumps
from logging import getLogger
from typing import Any, ClassVar, Generic, Self, TypeVar, cast

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    FilePath,
    SerializerFunctionWrapHandler,
    TypeAdapter,
    ValidationError,
    field_validator,
    model_serializer,
    model_validator,
)
from pydantic.alias_generators import to_pascal, to_snake
from pydantic_core import to_jsonable_python
from structcast.core.constants import SPEC_SOURCE
from structcast.core.exceptions import SpecError
from structcast.core.instantiator import AddressPattern, AttributePattern, BindPattern, CallPattern, ObjectPattern
from structcast.core.specifier import SPEC_CONSTANT, FlexSpec, SpecIntermediate, register_resolver
from structcast.core.template import Sequence, extend_structure
from structcast.utils.base import check_elements
from structcast.utils.security import check_path, resolve_address, split_attribute
from structcast.utils.types import PathLike

from structcast_model.builders.auto_name import AutoName, defaultdict
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

    IMPORTS: dict[str, set[str | None]] = Field(default_factory=dict)
    """Imports required for the layer, where the keys are module names and the values are sets of imported names
    from the corresponding modules.

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
        if isinstance(data, dict):
            return {k: set(check_elements(v)) for k, v in data.items()}
        try:
            data = check_elements(TypeAdapter(str | set[str] | Sequence[str]).validate_python(data))
            return {k: {None} for k in data}
        except ValidationError:
            pass
        return data

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


def resolve_object(pattern: ObjectPattern, imports: defaultdict[str, set]) -> tuple[str, str]:
    """Resolve the object pattern to a string representation and collect the required imports.

    The object pattern is resolved by processing its patterns in order.
    The first pattern must be an `AddressPattern` or an `ObjectPattern`, which is resolved to a string representation.
    The subsequent patterns are applied to the resolved string representation in order,
    where an `AttributePattern` is resolved to an attribute access, a `CallPattern` is resolved to a function call,
    and a `BindPattern` is resolved to a function call with the bound arguments.
    The required imports are collected from the `AddressPattern`s encountered during the resolution process.

    Args:
        pattern (ObjectPattern): The pattern to resolve.
        imports (defaultdict[str, set]): A dictionary to collect the required imports,
            where the keys are module names and the values are sets of imported names from the corresponding modules.

    Returns:
        tuple[str, str]: A tuple containing the resolved string representation of the object pattern and
            the name of the top-level class or function.
    """
    classes: list[str] = []

    def _repr(raw: Any) -> str:
        if isinstance(raw, (int, float, bool, bytes, type(None))):
            return repr(raw)
        try:
            return _resolve(ObjectPattern.model_validate(raw))
        except ValidationError:
            pass
        if isinstance(raw, str):
            if raw.startswith("eval:"):
                return raw[5:].strip()
            return repr(raw)
        if isinstance(raw, dict):
            return f"{{{', '.join(f'{_repr(k)}: {_repr(v)}' for k, v in raw.items())}}}"
        if isinstance(raw, (list, tuple)):
            return f"[{', '.join(_repr(item) for item in raw)}]"
        raise SpecError(f"Unsupported type for validation: {type(raw)}")

    def _args(raw: Any) -> str:
        if isinstance(raw, dict):
            return ", ".join(f"{k}={_repr(v)}" for k, v in raw.items())
        if isinstance(raw, (list, tuple)):
            return ", ".join(_repr(v) for v in raw)
        return _repr(raw)

    def _resolve(obj: ObjectPattern) -> str:
        first, rest = obj.patterns[0], obj.patterns[1:]
        if isinstance(first, AddressPattern):
            module, res = resolve_address(first.address)
            classes.append(res)
            if module:
                imports[module].add(res)
        elif isinstance(first, ObjectPattern):
            res = _resolve(first)
        else:
            raise SpecError(
                "First pattern of an ObjectPattern must be an AddressPattern or ObjectPattern "
                f"but got: {to_jsonable_python(pattern)}"
            )
        for ptn in rest:
            if isinstance(ptn, (AddressPattern, ObjectPattern)):
                raise SpecError(
                    "Only the first pattern of an ObjectPattern can be an AddressPattern or ObjectPattern "
                    f"but got: {to_jsonable_python(pattern)}"
                )
            if isinstance(ptn, AttributePattern):
                res = f"{res}.{ptn.attribute}"
            elif isinstance(ptn, CallPattern):
                res = f"{res}({_args(ptn.call)})"
            elif isinstance(ptn, BindPattern):
                pid = str(id(ptn))[1:4]
                aname, kwname = f"_arg{pid}", f"_kw{pid}"
                args = _args(ptn.bind)
                if isinstance(ptn.bind, dict):
                    res = f"(lambda *{aname}, **{kwname}: {res}(*{aname}, {args}, **{kwname}))"
                else:
                    res = f"(lambda *{aname}, **{kwname}: {res}({args}, *{aname}, **{kwname}))"
            else:
                raise SpecError(
                    "Patterns after the first pattern of an ObjectPattern must be AttributePattern, CallPattern, "
                    f"or BindPattern but got: {to_jsonable_python(pattern)}"
                )
        return res

    return _resolve(pattern), (classes[0] if classes else "_Class")


class LayerIntermediate(Serializable):
    """Intermediate representation of a layer during the building process."""

    imports: dict[str, set[str | None]]
    """The imports required for the layer."""

    classname: str
    """The name of the layer class."""

    inputs: list[str]
    """The names of the input layers."""

    outputs: list[str]
    """The names of the output layers."""

    layers: dict[str, LayerIntermediate | str]
    """The sub-layers of the layer."""

    flow: list[tuple[str, str, str | None]]
    """The flow of the layer."""

    inference_flow: list[tuple[str, str, str | None]]
    """The inference flow of the layer."""

    structured_output: bool
    """Whether the output is structured."""

    layer_call_name: ClassVar[str | None] = None
    """The name of the method to call the layer, if applicable."""

    @cached_property
    def collected_imports(self) -> dict[str, set[str | None]]:
        """Collect the required imports from the layer and its sub-layers."""
        imports: dict[str, set[str | None]] = defaultdict(set)
        imports.update(self.imports)
        for sub in self.layers.values():
            if isinstance(sub, LayerIntermediate):
                for module, names in sub.collected_imports.items():
                    imports[module].update(names)
        return imports

    @cached_property
    def _forward_inputs(self) -> str:
        """Get the input arguments for calling the layer in the forward method."""
        return ", ".join(self.inputs)

    @cached_property
    def _forward_outputs(self) -> str:
        """Get the output arguments for calling the layer in the forward method."""
        if self.structured_output:
            return f"{{{','.join(f'{repr(k)}: {k}' for k in self.OUTPUTS)}}}"
        return ", ".join(self.OUTPUTS)

    def _get_layer(self, layername: str) -> str:
        """Get the sub-layer with the given name."""
        return layername

    def _forward_flow(self, flow: list[tuple[str, str, str | None]]) -> list[str]:
        """Get the code for the flow in the forward method."""
        return [f"{o} = {self._get_layer(L)}({i})" if L else f"{o} = {i}" for i, o, L in flow]

    @cached_property
    def _forward_training_flow(self) -> list[str]:
        """Get the code for the training flow in the forward method."""
        return self._forward_flow(self.flow)

    @cached_property
    def _forward_inference_flow(self) -> list[str]:
        """Get the code for the inference flow in the forward method."""
        return self._forward_flow(self.inference_flow)

    def _get_script(self, class_name: str, initialized_layers: list[str]) -> str:
        """Implement the method to get the script for the layer."""
        raise NotImplementedError("The _get_script method must be implemented in the subclass.")

    @classmethod
    def _get_scripts(cls, cfg: LayerIntermediate) -> list[str]:
        naming = AutoName("")
        classnames: OrderedDict[str, str] = OrderedDict()
        scripts: list[str] = []

        def _hash(raw: Any) -> str:
            return sha256(json_dumps(to_jsonable_python(raw), sort_keys=True).encode()).hexdigest()

        def _scripts(sub: LayerIntermediate) -> str:
            if (hash_id := _hash(sub)) in classnames:
                return f"{classnames[hash_id]}()"
            classnames[hash_id] = (classname := naming(sub.classname))
            layers: list[str] = [f"{k} = {v if isinstance(v, str) else _scripts(v)}" for k, v in sub.layers.items()]
            scripts.append(sub._get_script(classname, layers))
            return f"{classname}()"

        _scripts(cfg)
        return scripts

    @cached_property
    def scripts(self) -> list[str]:
        """Get the scripts for the layer and its sub-layers."""
        return self._get_scripts(self)

    def __call__(self, module_path: PathLike) -> None:
        """Save the script for the layer to the given path."""
        imports = "\n".join(
            [f"from {p} import {', '.join([m for m in i if m])}" for p, i in self.collected_imports.items()]
            + [f"import {p}" for p, i in self.collected_imports.items() if None in i]
        )
        code = "\n\n".join([s for s in [imports, *self.scripts] if s])
        module_path = check_path(module_path)
        module_path.parent.mkdir(parents=True, exist_ok=True)
        module_path.write_text(code, encoding="utf-8")


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
        imports: defaultdict[str, set[str | None]] = defaultdict(set)
        imports.update(layer.IMPORTS)
        sublayers: dict[str, LayerIntermediate | str] = {}
        naming = AutoName("_")

        def _getter(raw: Any, var_name: str | None = None) -> str:
            try:
                return resolve_object(ObjectPattern.model_validate(raw), imports)[0]
            except ValidationError:
                pass
            if isinstance(raw, dict):
                return f"{{{', '.join(f'{repr(k)}: {_getter(s, var_name)}' for k, s in raw.items())}}}"
            if isinstance(raw, list):
                return f"[{', '.join(_getter(s, var_name) for s in raw)}]"
            if isinstance(raw, tuple):
                return f"({', '.join(_getter(s, var_name) for s in raw)})"
            if not isinstance(raw, str):
                return repr(raw)
            spec = SpecIntermediate.convert_spec(raw)
            if spec.identifier == SPEC_SOURCE:
                var_name, attr = (var_name, spec.value) if var_name else (spec.value[0], spec.value[1:])
                return f"{var_name}{''.join(f'[{repr(s)}]' for s in attr)}"
            if spec.identifier in SPEC_EVAL:
                return spec.value
            if spec.identifier == SPEC_CONSTANT:
                return repr(spec.value)
            raise SpecError(f"Unsupported spec identifier: {spec.identifier}")

        def _inputs(raw: Any) -> str:
            if isinstance(raw, dict):
                return ", ".join(f"{k}={_getter(v)}" for k, v in raw.items())
            if isinstance(raw, (list, tuple)):
                return ", ".join(_getter(v) for v in raw)
            return _getter(raw)

        def _outputs(raw: SpecIntermediate | list[SpecIntermediate]) -> str:
            return raw.value[0] if isinstance(raw, SpecIntermediate) else f"({', '.join(_outputs(r) for r in raw)})"

        def _create_flow(units: list[LayerBehavior]) -> list[tuple[str, str, str | None]]:
            flow: list[tuple[str, str, str | None]] = []
            for unit in units:
                if unit.LAYER is None:
                    if unit.NAME and unit.NAME not in sublayers:
                        raise SpecError(f'Layer with name "{unit.NAME}" not defined in the flow.')
                    name = unit.NAME
                else:
                    if isinstance(unit.LAYER, ObjectPattern):
                        subinst, subclassname = resolve_object(unit.LAYER, imports)
                    else:
                        subclassname, subinst = self._get_sublayer(parameters, unit)  # type: ignore[arg-type]
                    if (name := unit.NAME or naming(to_snake(subclassname))) in sublayers:
                        raise SpecError(f'Duplicate layer name "{name}" found in the flow.')
                    sublayers[name] = subinst
                if (has_inputs := unit.INPUTS is not None) and (has_outputs := unit.OUTPUTS is not None):
                    inp = _inputs(unit.INPUTS.model_dump())
                    if isinstance(unit.OUTPUTS.spec, dict):
                        flow.append((inp, (tmpname := f"{name or naming('tmp')}_output"), name))
                        for key, value in unit.OUTPUTS.model_dump().items():
                            flow.append((key, _getter(value, tmpname), None))
                    else:
                        flow.append((inp, _outputs(unit.OUTPUTS.spec), name))
                elif has_inputs or has_outputs:
                    raise SpecError(
                        f"Both INPUTS and OUTPUTS must be specified together in the training/inference flow "
                        f"but got: {unit.model_dump()}"
                    )
            return flow

        return self.user_defined_layer_type(
            imports=imports,
            classname=classname,
            inputs=layer.INPUTS,
            outputs=layer.OUTPUTS,
            layers=sublayers,
            flow=_create_flow(layer.FLOW),
            inference_flow=_create_flow(layer.INFERENCE_FLOW),
            structured_output=layer.STRUCTURED_OUTPUT,
        )
