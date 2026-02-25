"""Base builder for building layers or backward operators from templates."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from functools import cached_property
from hashlib import sha256
from json import dumps as json_dumps
from pathlib import Path
from typing import Any, ClassVar, Generic, TypeVar, cast

from pydantic import (
    ValidationError,
)
from pydantic.alias_generators import to_pascal, to_snake
from pydantic_core import to_jsonable_python
from structcast.core.base import Serializable
from structcast.core.constants import SPEC_SOURCE
from structcast.core.exceptions import SpecError
from structcast.core.instantiator import AddressPattern, AttributePattern, BindPattern, CallPattern, ObjectPattern
from structcast.core.specifier import SPEC_CONSTANT, SpecIntermediate
from structcast.utils.security import get_default_dir, resolve_address, split_attribute
from structcast.utils.types import PathLike

from structcast_model.builders.auto_name import AutoName
from structcast_model.builders.schema import (
    SPEC_EVAL,
    LayerBehavior,
    Parameters,
    TemplateBackward,
    TemplateLayer,
    UserLayer,
)
from structcast_model.utils.base import load_any


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
        from_imports = {p: {m for m in i if m} for p, i in self.collected_imports.items()}
        imported_code = "\n".join(
            [f"from {p} import {', '.join([m for m in i if m])}" for p, i in from_imports.items() if i]
            + [f"import {p}" for p, i in self.collected_imports.items() if None in i]
        ).strip()
        code = "\n\n".join([s for s in [(imported_code + "\n"), *self.scripts] if s])
        if module_path is None:
            module_path = Path(f"{to_snake(self.classname)}.py")
        elif not isinstance(module_path, Path):
            module_path = Path(module_path)
        module_path.parent.mkdir(parents=True, exist_ok=True)
        module_path.write_text(code, encoding="utf-8")


class LayerIntermediate(_Intermediate):
    """Intermediate representation of a layer during the building process."""

    inputs: list[str]
    """The names of the input layers."""

    outputs: list[str]
    """The names of the output layers."""

    layers: dict[str, LayerIntermediate | str]
    """The sub-layers used in the layer, where the keys are the layer names and the values are either the sub-layer
    as a `LayerIntermediate` instance or a string representation of the sub-layer to be used directly in the script."""

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
    def _get_layer_scripts(cls, cfg: LayerIntermediate) -> list[str]:
        naming = AutoName("")
        classnames: dict[str, str] = {}
        scripts: list[str] = []
        for name in [n for v in cfg.collected_imports.values() for n in v if n]:
            naming(name)

        def _hash(raw: Any) -> str:
            return sha256(json_dumps(to_jsonable_python(raw), sort_keys=True).encode()).hexdigest()

        def _scripts(sub: LayerIntermediate) -> str:
            if (hash_id := _hash(sub)) in classnames:
                return f"{classnames[hash_id]}()"
            classnames[hash_id] = (classname := naming(sub.classname))
            layers: list[str] = [f"{k} = {v if isinstance(v, str) else _scripts(v)}" for k, v in sub.layers.items()]
            scripts.append(sub._get_layer_script(classname, layers))
            return f"{classname}()"

        _scripts(cfg)
        return scripts

    def _get_scripts(self) -> list[str]:
        """Get the scripts for the layer and its sub-layers."""
        return self._get_layer_scripts(self)


LayerIntermediateT = TypeVar("LayerIntermediateT", bound=LayerIntermediate)


def _to_pascal(val: str) -> str:
    return to_pascal(to_snake(val))


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
    def from_path(cls, path: PathLike) -> BaseModelBuilder:
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

    def _get_sublayer(self, parameters: Parameters, unit: UserLayer) -> tuple[str, LayerIntermediateT]:
        if unit.CFG is not None:
            current_path = str(unit.CFG)
            current_parts = self.from_references.get(current_path, None) or []
            subclassname = _to_pascal(unit.CFG.stem)
            if unit.TYPE:
                subclassname, parts = f"{subclassname}{_to_pascal(unit.TYPE)}", split_attribute(unit.TYPE)
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
            subclassname, parts, builder = _to_pascal(unit.TYPE), split_attribute(unit.TYPE), self
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
        layer = self.template(parameters, merged=False)
        imports: defaultdict[str, set[str | None]] = defaultdict(set)
        imports.update(layer.IMPORTS)
        sublayers: dict[str, LayerIntermediate | str] = {}
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
                    if unit.NAME and unit.NAME not in sublayers:
                        raise SpecError(f'Layer with name "{unit.NAME}" not defined in the flow.')
                    name = unit.NAME
                else:
                    if isinstance(unit.LAYER, ObjectPattern):
                        subinst, subclassname = resolve_object(imports, unit.LAYER)
                    else:
                        subclassname, subinst = self._get_sublayer(parameters, unit.LAYER)
                    if (name := unit.NAME or naming(to_snake(subclassname))) in sublayers:
                        raise SpecError(f'Duplicate layer name "{name}" found in the flow.')
                    sublayers[name] = subinst
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

        return self.user_defined_layer_type(
            imports=imports,
            classname=classname,
            inputs=layer.INPUTS,
            outputs=layer.OUTPUTS,
            layers=sublayers,
            flow=_create_flow(layer.FLOW),
            inference_flow=_create_flow(layer.INFERENCE_FLOW),
            structured_output=layer.STRUCTURED_OUTPUT if forced_structured_output is None else forced_structured_output,
        )


class BackwardIntermediate(_Intermediate):
    """Intermediate representation of a backward layer during the building process."""

    imports: dict[str, set[str | None]]
    """The imports required for the backward layer and its sub-layers,
    where the keys are module names and the values are sets of imported names from the corresponding modules."""

    classname: str
    """The name of the backward layer class."""

    mixed_precision: str | None
    """The mixed precision configuration for the backward layer, or `None` if mixed precision is not used."""

    accumulate_gradients: int | None
    """The number of steps to accumulate gradients for before performing an optimizer step,
    or `None` if not applicable."""

    losses: list[str]
    """The loss expressions for the backward layer."""

    models: list[str]
    """The models used in the backward layer."""

    optimizers: dict[str, tuple[str, list[str], str | None]]
    """The optimizers defined in the backward layer, where the keys are the optimizer names and the values are
    tuples of the form (optimizer, trainable_layers, clip_grad)."""

    backwards: list[tuple[str, str, list[str]]]
    """The backward steps in the backward layer,
    where each element is a tuple of the form (loss, backward_kwargs, optimizers)."""

    @cached_property
    def _backward_losses(self) -> str:
        """Get the loss expressions for the backward method."""
        return ", ".join(self.losses)

    @cached_property
    def _backward_models(self) -> str:
        """Get the models used in the backward method."""
        return ", ".join(self.models)


BackwardIntermediateT = TypeVar("BackwardIntermediateT", bound=BackwardIntermediate)


@dataclass(kw_only=True, slots=True)
class BaseBackwardBuilder(Generic[BackwardIntermediateT]):
    """Base backward builder for building backward layers from templates."""

    user_defined_backward_layer_type: ClassVar[type[BackwardIntermediateT]] = BackwardIntermediate

    raw: Any
    template: TemplateBackward = field(init=False)

    @classmethod
    def from_path(cls, path: PathLike) -> BaseBackwardBuilder:
        """Create a backward builder from the given configuration file path."""
        return cls(raw=load_any(path))

    def __post_init__(self) -> None:
        """Post-initialization to set up the template."""
        self.template = TemplateBackward.model_validate(self.raw)

    def _get_mixed_precision(
        self,
        imports: defaultdict[str, set[str | None]],
        mixed_precision: bool | dict[str, Any],
    ) -> str | None:
        raise NotImplementedError("The _get_mixed_precision method must be implemented in the subclass.")

    def __call__(
        self,
        parameters: dict[str, dict[str, Any]] | Parameters | None = None,
        classname: str = "Backward",
    ) -> BackwardIntermediateT:
        """Build the backward class from the template with the given parameters and class name.

        Args:
            parameters (dict[str, dict[str, Any]] | Parameters | None):
                The template keyword arguments to format the template with,
                or a `Parameters` instance containing the template keyword arguments.
            classname (str): The name of the backward class to use for the built backward operator.
                Default is "Backward".

        Returns:
            BackwardIntermediateT: The built backward class as a `BackwardIntermediateT` instance.
        """
        backward = self.template(parameters or Parameters())
        imports: defaultdict[str, set[str | None]] = defaultdict(set)
        imports.update(backward.IMPORTS)
        naming = AutoName("_")
        opts: dict[str, tuple[str, list[str], str | None]] = {}
        backward_names = set()
        backwards: list[tuple[str, str, list[str]]] = []
        for unit in backward.BACKWARDS:
            if (backward_name := unit.NAME or naming("backward")) in backward_names:
                raise SpecError(f'Duplicate backward name "{backward_name}" found in the backwards.')
            backward_names.add(backward_name)
            repr_backward_kw = ", ".join(f"{k}={resolve_getter(imports, v)}" for k, v in unit.model_extra.items())
            backwards.append((unit.LOSS, repr_backward_kw, []))
            for opt in unit.OPTIMIZERS:
                optinst, optclassname = resolve_object(imports, opt.OPTIMIZER)
                optname = opt.NAME or naming(optclassname)
                if optname in opts:
                    raise SpecError(f'Duplicate optimizer name "{optname}" found in the backwards.')
                opt_clip = resolve_object(imports, opt.OPTIMIZER)[0] if opt.CLIP else None
                opts[optname] = (optinst, opt.LAYERS, opt_clip)
                backwards[-1][-1].append(optname)
        return self.user_defined_backward_layer_type(
            imports=imports,
            classname=classname,
            mixed_precision=self._get_mixed_precision(imports, backward.MIXED_PRECISION),
            accumulate_gradients=backward.ACCUMULATE_GRADIENTS,
            losses=backward.LOSSES,
            models=backward.MODELS,
            optimizers=opts,
            backwards=backwards,
        )


__all__ = [
    "BackwardIntermediate",
    "BaseBackwardBuilder",
    "BaseModelBuilder",
    "LayerIntermediate",
    "resolve_getter",
    "resolve_object",
]


def __dir__() -> list[str]:
    return get_default_dir(globals())
