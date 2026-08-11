"""Base builder for building layers or learners from templates."""

from collections import defaultdict
from dataclasses import dataclass, field
from functools import cached_property
from hashlib import sha256
from json import dumps as json_dumps
from logging import getLogger
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Generic, TypeVar, Union, cast

from pydantic import Field, TypeAdapter, ValidationError
from pydantic_core import to_jsonable_python
from structcast.core.base import Serializable
from structcast.core.constants import SPEC_SOURCE
from structcast.core.exceptions import SpecError
from structcast.core.instantiator import AddressPattern, AttributePattern, BindPattern, CallPattern, ObjectPattern
from structcast.core.specifier import SPEC_CONSTANT, SpecIntermediate
from structcast.utils.base import resolve_address, split_attribute
from structcast.utils.types import PathLike

from structcast_model.builders.auto_name import AutoName
from structcast_model.builders.schema import (
    SPEC_EVAL,
    LayerBehavior,
    Parameters,
    TemplateLayer,
    TemplateLearner,
    TensorSpecTree,
    UserLayer,
)
from structcast_model.utils.base import load_any, to_pascal, to_snake, unique

logger = getLogger(__name__)

FILE_IMPORT_PREFIX = "__file__:"
"""Prefix marking a collected import that refers to a file path instead of a module name."""


def resolve_object(imports: defaultdict[str, set[str | None]], pattern: ObjectPattern) -> tuple[str, str]:
    """Resolve the object pattern to a string representation and collect the required imports.

    The object pattern is resolved by processing its patterns in order.
    The first pattern must be an `AddressPattern` or an `ObjectPattern`, which is resolved to a string representation.
    The subsequent patterns are applied to the resolved string representation in order,
    where an `AttributePattern` is resolved to an attribute access, a `CallPattern` is resolved to a function call,
    and a `BindPattern` is resolved to a function call with the bound arguments.
    The required imports are collected from the `AddressPattern`s encountered during the resolution process.

    Args:
        imports (defaultdict[str, set[str | None]]): A dictionary to collect the required imports,
            where the keys are module names and the values are sets of imported names from the corresponding modules.
        pattern (ObjectPattern): The pattern to resolve.


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
            if first.file:
                # File-addressed objects cannot be imported by module name: record the file under a
                # special key so the script renderer emits an import_from_address binding instead.
                imports[f"{FILE_IMPORT_PREFIX}{first.file}"].add(first.address)
            elif module:
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


def resolve_getter(imports: defaultdict[str, set[str | None]], spec: Any, variable: str | None = None) -> str:
    """Resolve the given specification to a string representation and collect the required imports if applicable.

    Args:
        imports (defaultdict[str, set[str | None]]): A dictionary to collect the required imports,
            where the keys are module names and the values are sets of imported names from the corresponding modules.
        spec (Any): The specification to resolve.
        variable (str | None): The variable name to use for source identifier specifications
            if the source identifier does not have a single string index as its value.
            If not provided, the variable name will be resolved from the value of the source identifier.

    Returns:
        str: The resolved string representation of the specification.
    """

    def _getter(raw: Any, var_name: str | None = None) -> str:
        try:
            return resolve_object(imports, ObjectPattern.model_validate(raw))[0]
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

    return _getter(spec, variable)


def _merge_imports(*imports_list: dict[str, set[str | None]]) -> dict[str, set[str | None]]:
    merged: dict[str, set[str | None]] = defaultdict(set)
    for imports in imports_list:
        for module, names in imports.items():
            merged[module].update(names)
    return merged


class _Intermediate(Serializable):
    """Intermediate representation of an operator during the building process."""

    classname: str
    """The name of the class."""

    imports: dict[str, set[str | None]]
    """The imports required for the operator and its sub-operators,
    where the keys are module names and the values are sets of imported names from the corresponding modules."""

    default_imports: ClassVar[dict[str, set[str | None]]] = {}
    """Default imports that are always included for all operators."""

    @cached_property
    def collected_imports(self) -> dict[str, set[str | None]]:
        """Collect the required imports from the layer and its sub-layers."""
        return _merge_imports(self.default_imports, self.imports)

    def _get_scripts(self) -> list[str]:
        """Implement the method to get the scripts for the layer."""
        raise NotImplementedError("The _get_scripts method must be implemented in the subclass.")

    @cached_property
    def scripts(self) -> list[str]:
        """Get the scripts for the layer and its sub-layers."""
        return self._get_scripts()

    def __call__(self, module_path: PathLike | None = None) -> None:
        """Save the script for the layer to the given path."""
        module_imports = {p: i for p, i in self.collected_imports.items() if not p.startswith(FILE_IMPORT_PREFIX)}
        file_imports = {
            p.removeprefix(FILE_IMPORT_PREFIX): i
            for p, i in self.collected_imports.items()
            if p.startswith(FILE_IMPORT_PREFIX)
        }
        from_imports = {p: {m for m in i if m} for p, i in module_imports.items()}
        imported_code = "\n".join(
            [f"from {p} import {', '.join([m for m in i if m])}" for p, i in from_imports.items() if i]
            + [f"import {p}" for p, i in module_imports.items() if None in i]
            + (["from structcast.utils.base import import_from_address"] if file_imports else [])
        ).strip()
        file_bindings = "\n".join(
            f'{resolve_address(address)[1]} = import_from_address("{address}", module_file="{file}")'
            for file, addresses in sorted(file_imports.items())
            for address in sorted(a for a in addresses if a)
        )
        code = "\n\n".join([s for s in [(imported_code + "\n"), file_bindings, *self.scripts] if s])
        if module_path is None:
            module_path = Path(f"{to_snake(self.classname)}.py")
        elif not isinstance(module_path, Path):
            module_path = Path(module_path)
        module_path.parent.mkdir(parents=True, exist_ok=True)
        module_path.write_text(code, encoding="utf-8")


def _hash(raw: Any) -> str:
    return sha256(json_dumps(to_jsonable_python(raw), sort_keys=True).encode()).hexdigest()


_input_shapes_adapter = TypeAdapter(dict[str, TensorSpecTree])
"""Adapter dumping `INPUT_SHAPES` back to the plain nested data emitted in the generated script."""


class LayerIntermediate(_Intermediate):
    """Intermediate representation of a layer during the building process."""

    inputs: list[str]
    """The names of the input layers."""

    input_shapes: dict[str, Any] = Field(default_factory=dict)
    """The shapes of the input layers in their serialized compact form, where the keys are the input names.

    Each value is a plain shape, a mapping with the `_SHAPE_` key, or a dictionary or list nesting more of them,
    and is emitted as a literal in the generated script."""

    outputs: list[str]
    """The names of the output layers."""

    layers: dict[str, Union["LayerIntermediate", str]]
    """The layers used in the layer, where the keys are the layer names and the values are either the layer
    as a `LayerIntermediate` instance or a string representation of the layer to be used directly in the script."""

    flow: list[tuple[str, str, str | None]]
    """The flow of the layer during training, where each element is a tuple of the form (input, output, layer),
    where `input` is the input expression for the layer,
    `output` is the output variable name to assign the result of the layer to,
    and `layer` is the name of the layer to call for this step in the flow
    (or `None` if this step does not involve calling a layer)."""

    inference_flow: list[tuple[str, str, str | None]]
    """The flow of the layer during inference, where each element is a tuple of the form (input, output, layer),
    where `input` is the input expression for the layer,
    `output` is the output variable name to assign the result of the layer to,
    and `layer` is the name of the layer to call for this step in the flow
    (or `None` if this step does not involve calling a layer)."""

    structured_output: bool
    """Whether the output is structured."""

    @cached_property
    def collected_imports(self) -> dict[str, set[str | None]]:
        """Collect the required imports from the layer and its sub-layers."""
        sub_imports = (s.collected_imports for s in self.layers.values() if isinstance(s, LayerIntermediate))
        return _merge_imports(super().collected_imports, *sub_imports)

    @cached_property
    def _forward_inputs(self) -> str:
        """Get the input arguments for calling the layer in the forward method."""
        return ", ".join(self.inputs)

    @cached_property
    def _forward_outputs(self) -> str:
        """Get the output arguments for calling the layer in the forward method."""
        if self.structured_output:
            return f"{{{','.join(f'{repr(k)}: {k}' for k in self.outputs)}}}"
        return ", ".join(self.outputs)

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

    def _get_layer_script(self, class_name: str, initialized_layers: list[str]) -> str:
        """Implement the method to get the script for the layer."""
        raise NotImplementedError("The _get_layer_script method must be implemented in the subclass.")

    @classmethod
    def _get_class_instance(cls, classname: str) -> str:
        return f"{classname}()"

    @classmethod
    def _get_layer_scripts(cls, cfg: "LayerIntermediate") -> list[str]:
        naming = AutoName("")
        classnames: dict[str, str] = {}
        scripts: list[str] = []
        for name in [n for v in cfg.collected_imports.values() for n in v if n]:
            naming(name)

        def _scripts(sub: LayerIntermediate) -> str:
            if (hash_id := _hash(sub)) in classnames:
                return cls._get_class_instance(classnames[hash_id])
            classnames[hash_id] = (classname := naming(sub.classname))
            layers: list[str] = [f"{k} = {v if isinstance(v, str) else _scripts(v)}" for k, v in sub.layers.items()]
            scripts.append(sub._get_layer_script(classname, layers))
            return cls._get_class_instance(classname)

        _scripts(cfg)
        return scripts

    def _get_scripts(self) -> list[str]:
        """Get the scripts for the layer and its sub-layers."""
        return self._get_layer_scripts(self)


LayerIntermediateT = TypeVar("LayerIntermediateT", bound=LayerIntermediate)


@dataclass(kw_only=True, slots=True)
class BaseModelBuilder(Generic[LayerIntermediateT]):
    """Base model builder for building layers from templates."""

    user_defined_layer_type: ClassVar[type[LayerIntermediateT]] = LayerIntermediate

    raw: Any
    predefined_user_defined_layers: dict[str, Any] = field(default_factory=dict)
    current_path: str = ""
    from_references: dict[str, list[str]] = field(default_factory=dict)

    template: TemplateLayer = field(init=False)
    user_defined_layers: dict[str, Any] = field(init=False)

    @classmethod
    def from_path(cls, path: PathLike) -> "BaseModelBuilder[LayerIntermediateT]":
        """Create a model builder from the given configuration file path."""
        curr_path = str(path)
        return cls(raw=load_any(path), current_path=curr_path, from_references={curr_path: ["__root__"]})

    def __post_init__(self) -> None:
        """Post-initialization to set up the template."""
        self.template = TemplateLayer.model_validate(self.raw)
        self.user_defined_layers = {**self.predefined_user_defined_layers, **self.template.others}

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

    def _get_layer(self, parameters: Parameters, unit: UserLayer) -> tuple[str, LayerIntermediateT]:
        if unit.CFG is not None:
            current_path = str(unit.CFG)
            current_parts = self.from_references.get(current_path, None) or []
            subclassname = to_pascal(unit.CFG.stem)
            if unit.TYPE:
                subclassname, parts = f"{subclassname}{to_pascal(unit.TYPE)}", split_attribute(unit.TYPE)
            else:
                if "__root__" in current_parts:
                    raise SpecError(f"Circular reference detected for layer configuration: {self.from_references}")
                current_parts, parts = (current_parts + ["__root__"]), []
            builder = type(self)(
                raw=load_any(unit.CFG),
                predefined_user_defined_layers=self.user_defined_layers,
                current_path=current_path,
                from_references={**self.from_references, current_path: current_parts},
            )
        elif unit.TYPE is not None:
            subclassname, parts, builder = to_pascal(unit.TYPE), split_attribute(unit.TYPE), self
        else:
            raise SpecError(f"LAYER must have either CFG or TYPE specified but got: {unit.model_dump()}")
        return subclassname, builder.get_user_defined_layer(parts, parameters.merge(unit.PARAM), subclassname)

    def __call__(
        self,
        parameters: dict[str, dict[str, Any]] | Parameters | None = None,
        classname: str = "Model",
        forced_structured_output: bool | None = None,
        user_defined_layer: str | None = None,
    ) -> LayerIntermediateT:
        """Build the layer from the template with the given parameters and class name.

        Args:
            parameters (dict[str, dict[str, Any]] | Parameters | None):
                The template keyword arguments to format the template with,
                or a `Parameters` instance containing the template keyword arguments.
            classname (str): The name of the layer class to use for the built layer. Default is "Model".
            forced_structured_output (bool | None): Whether to force the output to be structured
                regardless of the template specification.
            user_defined_layer (str | None): The reference to a user-defined layer to build instead of the root layer
                defined in the template. If specified, the reference should be in the format of "key1.key2...keyN",
                where each key is a key defined in the user-defined layers.

        Returns:
            LayerIntermediateT: The built layer as a `LayerIntermediateT` instance.
        """
        parameters = cast(Parameters, Parameters.create(self.template.PARAMETERS, parameters))
        if user_defined_layer:
            return self.get_user_defined_layer(split_attribute(user_defined_layer), parameters, classname)
        module = self.template(parameters, merged=False)
        imports: defaultdict[str, set[str | None]] = defaultdict(set)
        imports.update(module.IMPORTS)
        layers: dict[str, LayerIntermediate | str] = {}
        naming = AutoName("_")

        def _inputs(raw: Any) -> str:
            if isinstance(raw, dict):
                return ", ".join(f"{k}={resolve_getter(imports, v)}" for k, v in raw.items())
            if isinstance(raw, (list, tuple)):
                return ", ".join(resolve_getter(imports, v) for v in raw)
            return resolve_getter(imports, raw)

        def _outputs(raw: SpecIntermediate | list[SpecIntermediate]) -> str:
            return raw.value[0] if isinstance(raw, SpecIntermediate) else f"({', '.join(_outputs(r) for r in raw)})"

        def _create_flow(units: list[LayerBehavior]) -> list[tuple[str, str, str | None]]:
            flow: list[tuple[str, str, str | None]] = []
            for unit in units:
                if unit.LAYER is None:
                    if unit.NAME and unit.NAME not in layers:
                        raise SpecError(f'Layer with name "{unit.NAME}" not defined in the flow.')
                    name = unit.NAME
                else:
                    if isinstance(unit.LAYER, ObjectPattern):
                        subinst, subclassname = resolve_object(imports, unit.LAYER)
                    else:
                        subclassname, subinst = self._get_layer(parameters, unit.LAYER)
                    if (name := unit.NAME or naming(to_snake(subclassname))) in layers:
                        raise SpecError(f'Duplicate layer name "{name}" found in the flow.')
                    layers[name] = subinst
                if unit.INPUTS is not None and unit.OUTPUTS is not None:
                    inp = _inputs(unit.INPUTS.model_dump())
                    if isinstance(unit.OUTPUTS.spec, dict):
                        flow.append((inp, (tmpname := f"{name or naming('tmp')}_output"), name))
                        for key, value in unit.OUTPUTS.model_dump().items():
                            flow.append((resolve_getter(imports, value, tmpname), key, None))
                    else:
                        flow.append((inp, _outputs(unit.OUTPUTS.spec), name))
                elif unit.INPUTS is not None or unit.OUTPUTS is not None:
                    raise SpecError(
                        f"Both INPUTS and OUTPUTS must be specified together in the training/inference flow "
                        f"but got: {unit.model_dump()}"
                    )
            return flow

        structured_output = module.STRUCTURED_OUTPUT if forced_structured_output is None else forced_structured_output
        return self.user_defined_layer_type(
            imports=imports,
            classname=classname,
            inputs=module.INPUTS,
            input_shapes=_input_shapes_adapter.dump_python(module.INPUT_SHAPES),
            outputs=module.OUTPUTS,
            layers=layers,
            flow=_create_flow(module.FLOW),
            inference_flow=_create_flow(module.INFERENCE_FLOW),
            structured_output=structured_output,
        )


class LearnerIntermediate(_Intermediate):
    """Intermediate representation of a learner during the building process."""

    imports: dict[str, set[str | None]]
    """The imports required for the learner and its sub-layers,
    where the keys are module names and the values are sets of imported names from the corresponding modules."""

    classname: str
    """The name of the learner class."""

    mixed_precision_type: str | None
    """The mixed precision type for the learner, or `None` if mixed precision is not used."""

    accumulate_gradients: int | None
    """The number of steps to accumulate gradients for before performing an optimizer step,
    or `None` if not applicable."""

    inputs: list[str]
    """The names of the input layers."""

    outputs: list[str]
    """The names of the output layers."""

    layers: dict[str, Union["LayerIntermediate", str]]
    """The layers used in the learner, where the keys are the layer names and the values are either the layer
    as a `LayerIntermediate` instance or a string representation of the layer to be used directly in the script."""

    others: dict[str, str]
    """Other instances used in the learner that are not layers, where the keys are the instance names and
    the values are the string representations of the instances to be used directly in the script."""

    flow: list[tuple[str, str, str | None] | tuple[str, str, str, str | None, str | None, list[str]]]
    """The forward flow during training, where each element is either a tuple of the form (input, output, layer)
    for regular steps, or a tuple of the form (loss, output, optimizer, clip, mixed_precision_scale, trainable_models)
    for optimizer steps."""

    inference_flow: list[tuple[str, str, str | None]]
    """The forward flow during inference, where each element is a tuple of the form (input, output, layer)."""

    @cached_property
    def collected_imports(self) -> dict[str, set[str | None]]:
        """Collect the required imports from the layer and its sub-layers."""
        sub_imports = (s.collected_imports for s in self.layers.values() if isinstance(s, LayerIntermediate))
        return _merge_imports(super().collected_imports, *sub_imports)

    @cached_property
    def models(self) -> list[str]:
        """Get the models used in the layer."""
        return unique([m for u in self.flow if len(u) == 6 for m in u[-1]])

    @cached_property
    def optimizers(self) -> list[str]:
        """Get the optimizers used in the layer."""
        return unique([u[2] for u in self.flow if len(u) == 6])

    @cached_property
    def mixed_precision_scales(self) -> list[str]:
        """Get the mixed precision scales used in the layer."""
        return unique([u[4] for u in self.flow if len(u) == 6 and u[4]])

    @cached_property
    def _learner_models(self) -> str:
        """Get the models used in the learner."""
        return ", ".join(self.models)

    @cached_property
    def _forward_inputs(self) -> str:
        """Get the input arguments for calling the layer in the forward method."""
        return ", ".join(self.inputs)

    @cached_property
    def _forward_outputs(self) -> str:
        """Get the output arguments for calling the layer in the forward method."""
        return f"{{{','.join(f'{repr(k)}: {k}' for k in self.outputs)}}}"

    def _get_regular_step(self, inputs: str, output: str, layer: str | None) -> str:
        return f"{output} = {layer}({inputs})" if layer else f"{output} = {inputs}"

    def _get_forward_training_flow(self) -> list[str]:
        """Get the code for the training flow in the forward method."""
        raise NotImplementedError("The _forward_training_flow method must be implemented in the subclass.")

    @cached_property
    def _forward_training_flow(self) -> list[str]:
        """Get the code for the training flow in the forward method."""
        return self._get_forward_training_flow()

    def _get_forward_inference_flow(self) -> list[str]:
        """Get the code for the inference flow in the forward method."""
        return [self._get_regular_step(i, o, L) for i, o, L in self.inference_flow]

    @cached_property
    def _forward_inference_flow(self) -> list[str]:
        """Get the code for the inference flow in the forward method."""
        return self._get_forward_inference_flow()

    def _get_learner_script(self, initialized_layers: dict[str, str]) -> str:
        """Get the script for the learner."""
        raise NotImplementedError("The _get_learner_script method must be implemented in the subclass.")

    def _get_scripts(self) -> list[str]:
        """Get the scripts for the layer and its sub-layers."""
        naming = AutoName("")
        classnames: dict[str, str] = {}
        scripts: list[str] = []
        naming(self.classname)
        for name in [n for v in self.collected_imports.values() for n in v if n]:
            naming(name)

        def _scripts(sub: LayerIntermediate) -> str:
            if (hash_id := _hash(sub)) in classnames:
                return sub._get_class_instance(classnames[hash_id])
            classnames[hash_id] = (classname := naming(sub.classname))
            layers: list[str] = [f"{k} = {v if isinstance(v, str) else _scripts(v)}" for k, v in sub.layers.items()]
            scripts.append(sub._get_layer_script(classname, layers))
            return sub._get_class_instance(classname)

        init_layers = {k: v if isinstance(v, str) else _scripts(v) for k, v in self.layers.items()}
        scripts.append(self._get_learner_script(init_layers))
        return scripts


LearnerIntermediateT = TypeVar("LearnerIntermediateT", bound=LearnerIntermediate)


@dataclass(kw_only=True, slots=True)
class BaseLearnerBuilder(Generic[LearnerIntermediateT]):
    """Base learner builder for building learners from templates."""

    user_defined_learner_layer_type: ClassVar[type[LearnerIntermediateT]] = LearnerIntermediate
    layer_builder_type: ClassVar[type[BaseModelBuilder]] = BaseModelBuilder

    raw: Any
    current_path: str = ""
    template: TemplateLearner = field(init=False)
    layer_builder: BaseModelBuilder = field(init=False)

    @classmethod
    def from_path(cls, path: PathLike) -> "BaseLearnerBuilder[LearnerIntermediateT]":
        """Create a learner builder from the given configuration file path."""
        return cls(raw=load_any(path), current_path=str(path))

    def __post_init__(self) -> None:
        """Post-initialization to set up the template."""
        self.template = TemplateLearner.model_validate(self.raw)
        from_references = {self.current_path: ["__root__"]} if self.current_path else {}
        self.layer_builder = self.layer_builder_type(
            raw=self.template.others, current_path=self.current_path, from_references=from_references
        )

    def _get_mixed_precision(
        self,
        imports: defaultdict[str, set[str | None]],
        mixed_precision: bool | dict[str, Any],
    ) -> tuple[str, str | None]:
        logger.warning(
            "Mixed precision is not implemented in the base learner builder. Returning None for mixed precision."
        )
        return "", None

    def _get_optimizer(
        self,
        imports: defaultdict[str, set[str | None]],
        optimizer: ObjectPattern,
        trainable_layers: list[str],
    ) -> tuple[str, str]:
        return resolve_object(imports, optimizer)

    def __call__(  # noqa: PLR0915
        self,
        parameters: dict[str, dict[str, Any]] | Parameters | None = None,
        classname: str = "Learner",
    ) -> LearnerIntermediateT:
        """Build the learner class from the template with the given parameters and class name.

        Args:
            parameters (dict[str, dict[str, Any]] | Parameters | None):
                The template keyword arguments to format the template with,
                or a `Parameters` instance containing the template keyword arguments.
            classname (str): The name of the learner class to use for the built learner.
                Default is "Learner".

        Returns:
            LearnerIntermediateT: The built learner class as a `LearnerIntermediateT` instance.
        """
        parameters = cast(Parameters, Parameters.create(self.template.PARAMETERS, parameters))
        module = self.template(parameters, merged=False)
        imports: defaultdict[str, set[str | None]] = defaultdict(set)
        imports.update(module.IMPORTS)
        layers: dict[str, LayerIntermediate | str] = {}
        others: dict[str, str] = {}
        naming = AutoName("_")
        for layer in module.TRAINABLE_LAYERS:
            layer = naming(layer)
            others[layer] = layer

        def _inputs(raw: Any) -> str:
            if isinstance(raw, dict):
                return ", ".join(f"{k}={resolve_getter(imports, v)}" for k, v in raw.items())
            if isinstance(raw, (list, tuple)):
                return ", ".join(resolve_getter(imports, v) for v in raw)
            return resolve_getter(imports, raw)

        def _outputs(raw: SpecIntermediate | list[SpecIntermediate]) -> str:
            return raw.value[0] if isinstance(raw, SpecIntermediate) else f"({', '.join(_outputs(r) for r in raw)})"

        def _create_flow(units: list[LayerBehavior]) -> list[tuple[str, str, str | None]]:
            flow: list[tuple[str, str, str | None]] = []
            for unit in units:
                if unit.LAYER is None:
                    if unit.NAME and not (unit.NAME in layers or unit.NAME in others):
                        raise SpecError(f'Layer with name "{unit.NAME}" not defined in the flow.')
                    name = unit.NAME
                else:
                    if isinstance(unit.LAYER, ObjectPattern):
                        subinst, subclassname = resolve_object(imports, unit.LAYER)
                    else:
                        subclassname, subinst = self.layer_builder._get_layer(parameters, unit.LAYER)
                    if (name := unit.NAME or naming(to_snake(subclassname))) in layers or name in others:
                        raise SpecError(f'Duplicate layer name "{name}" found in the flow.')
                    layers[name] = subinst
                if unit.INPUTS is not None and unit.OUTPUTS is not None:
                    inp = _inputs(unit.INPUTS.model_dump())
                    if isinstance(unit.OUTPUTS.spec, dict):
                        flow.append((inp, (tmpname := f"{name or naming('tmp')}_output"), name))
                        for key, value in unit.OUTPUTS.model_dump().items():
                            flow.append((resolve_getter(imports, value, tmpname), key, None))
                    else:
                        flow.append((inp, _outputs(unit.OUTPUTS.spec), name))
                elif unit.INPUTS is not None or unit.OUTPUTS is not None:
                    raise SpecError(
                        f"Both INPUTS and OUTPUTS must be specified together in the training/inference flow "
                        f"but got: {unit.model_dump()}"
                    )
            return flow

        learner_flow: list[tuple[str, str, str | None] | tuple[str, str, str, str | None, str | None, list[str]]] = []
        inference_flow: list[tuple[str, str, str | None]] = []
        amp_inst, amp_cls = self._get_mixed_precision(imports, module.MIXED_PRECISION)
        for learner in module.LEARNERS:
            opt_inst, opt_cls = self._get_optimizer(imports, learner.OPTIMIZER, learner.TRAINABLE_LAYERS)
            if (opt_name := learner.NAME or naming(to_snake(opt_cls))) in layers or opt_name in others:
                raise SpecError(f'Duplicate variable name "{opt_name}" for optimizer found in the learner flow.')
            others[opt_name] = opt_inst
            if learner.CLIP:
                clip_inst, clip_cls = resolve_object(imports, learner.CLIP)
                if (clip_name := naming(f"{opt_name}_{to_snake(clip_cls)}")) in layers or clip_name in others:
                    raise SpecError(f'Duplicate variable name "{clip_name}" for clip found in the learner flow.')
                others[clip_name] = clip_inst
            else:
                clip_inst, clip_name = None, None
            if amp_cls is None:
                amp_name = None
            else:
                if (amp_name := naming(f"{opt_name}_{to_snake(amp_cls)}")) in layers or amp_name in others:
                    raise SpecError(
                        f'Duplicate variable name "{amp_name}" for mixed precision instance found in the learner flow.'
                    )
                others[amp_name] = amp_inst
            learner_flow += _create_flow(learner.FLOW)
            inference_flow += _create_flow(learner.INFERENCE_FLOW or learner.FLOW)
            backward_kw = ", ".join(f"{k}={resolve_getter(imports, v)}" for k, v in learner.EXTRA.items())
            learner_flow.append((learner.LOSS, backward_kw, opt_name, clip_name, amp_name, learner.TRAINABLE_LAYERS))
        return self.user_defined_learner_layer_type(
            imports=imports,
            classname=classname,
            mixed_precision_type=module.MIXED_PRECISION_TYPE,
            accumulate_gradients=module.ACCUMULATE_GRADIENTS,
            inputs=module.INPUTS,
            outputs=module.OUTPUTS,
            layers=layers,
            others=others,
            flow=learner_flow,
            inference_flow=inference_flow,
        )


__all__ = [
    "BaseLearnerBuilder",
    "BaseModelBuilder",
    "LayerIntermediate",
    "LearnerIntermediate",
    "resolve_getter",
    "resolve_object",
]

if not TYPE_CHECKING:
    import sys

    from structcast.utils.lazy_import import LazySelectedImporter

    sys.modules[__name__] = LazySelectedImporter(__name__, globals())
