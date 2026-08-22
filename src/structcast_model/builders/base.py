"""Base builder for building layers or learners from templates."""

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import cached_property
from hashlib import sha256
from json import dumps as json_dumps
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Generic, TypeVar, Union, cast

from pydantic import Field, TypeAdapter
from pydantic_core import to_jsonable_python
from structcast.core.base import Serializable
from structcast.core.exceptions import SpecError
from structcast.core.instantiator import ObjectPattern
from structcast.core.specifier import SpecIntermediate
from structcast.utils.base import resolve_address, split_attribute
from structcast.utils.types import PathLike

from structcast_model.builders.auto_name import AutoName
from structcast_model.builders.constants import FILE_IMPORT_PREFIX
from structcast_model.builders.schema import (
    LayerBehavior,
    LearnerBehavior,
    Parameters,
    Template,
    TemplateLayer,
    TemplateLearner,
    TensorSpecTree,
    UserDefinedLearner,
    UserLayer,
)
from structcast_model.builders.utils import resolve_getter, resolve_object
from structcast_model.utils.base import load_any, to_pascal, to_snake, unique


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
            [f"from {p} import {', '.join(sorted(m for m in i if m))}" for p, i in from_imports.items() if i]
            + [f"import {p}" for p, i in module_imports.items() if None in i]
            + (["from structcast.utils.base import import_from_address"] if file_imports else [])
        ).strip()
        module_names = {name for names in from_imports.values() for name in names}
        bound: dict[str, str] = {}
        binding_lines: list[str] = []
        for file, addresses in sorted(file_imports.items()):
            for address in sorted(a for a in addresses if a):
                leaf = resolve_address(address)[1]
                if bound.get(leaf, file) != file:
                    raise SpecError(
                        f'File-addressed name "{leaf}" is bound from both "{bound[leaf]}" and "{file}": '
                        "the generated script would silently shadow one of them. Rename one symbol."
                    )
                if leaf in module_names:
                    raise SpecError(
                        f'File-addressed name "{leaf}" collides with an imported module member of the same name: '
                        "the generated script would silently shadow the import. Rename one symbol."
                    )
                bound[leaf] = file
                # Resolve the config-relative path while it is resolvable, so the generated script
                # imports the same file regardless of the directory it is later run from.
                rendered_file = str(Path(file).resolve()) if Path(file).exists() else file
                binding_lines.append(f"{leaf} = import_from_address({address!r}, module_file={rendered_file!r})")
        file_bindings = "\n".join(binding_lines)
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

    gradient_checkpointing: dict[str, str] | None = None
    """The activation-checkpointing keyword arguments, already resolved to the code emitted for them,
    or `None` when the layer is not checkpointed; an empty mapping is the framework's own defaults.

    A field of the intermediate, so two otherwise identical layers configured differently hash apart
    and stay separate generated classes (`docs/adr/0020`)."""

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

    # Subclasses bind the type to the concrete intermediate they parametrize the builder with,
    # which a `ClassVar` cannot express, so the default is cast to the type variable. Dropping
    # `ClassVar` instead would turn the attribute into a per-instance dataclass field.
    user_defined_layer_type: ClassVar[type[LayerIntermediateT]] = cast(type[LayerIntermediateT], LayerIntermediate)

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
        parts: Sequence[str | int],
        parameters: dict[str, dict[str, Any]] | Parameters,
        classname: str,
    ) -> LayerIntermediateT:
        """Get the user-defined layer with the given parts and parameters.

        Args:
            parts (Sequence[str | int]): The parts of the user-defined layer reference to resolve,
                as returned by `split_attribute`. Numeric index parts never name a user-defined layer.
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
        if not isinstance(first, str) or first not in self.user_defined_layers:
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

    def _resolve_gradient_checkpointing(
        self,
        imports: defaultdict[str, set[str | None]],
        config: bool | dict[str, Any],
    ) -> dict[str, str] | None:
        """Validate one layer's `GRADIENT_CHECKPOINTING` and resolve its keywords to emitted code.

        Subclasses check the mapping against the keywords of their own mechanism and add the imports
        the enabled layer needs -- the runtime base class above all -- to `imports`, which is the
        per-instance import table of the layer being built rather than the shared `default_imports`,
        so a module holding no checkpointed layer imports nothing new.

        Returns:
            dict[str, str] | None: The resolved keyword arguments, empty for the framework's own
                defaults, or `None` when the layer is not checkpointed.
        """
        if config is False:
            return None
        raise SpecError(
            "GRADIENT_CHECKPOINTING names a framework mechanism, and the base builder emits no framework "
            "module: build the layer with the torch, flax or keras builder."
        )

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
                current_parts, parts = (current_parts + ["__root__"]), ()
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
        # `merge` is annotated to return the base `Parameters` while it instantiates `type(self)` at runtime.
        merged = cast(Parameters, parameters.merge(unit.PARAM))
        return subclassname, builder.get_user_defined_layer(parts, merged, subclassname)

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
            subinst: LayerIntermediate | str
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
            # Resolved after the flows: `collected_imports` keeps insertion order, so an earlier
            # resolution would reorder the emitted import header of every checkpointed layer.
            gradient_checkpointing=self._resolve_gradient_checkpointing(imports, module.GRADIENT_CHECKPOINTING),
        )


@dataclass(kw_only=True, slots=True)
class OptimizerSegment:
    """One optimizer step of a learner flow."""

    loss: str
    """The name of the loss variable the optimizer minimizes."""

    backward_kwargs: str = ""
    """The rendered keyword arguments of the backward call, rendered by the caller after the flow."""

    optimizer: str
    """The variable name of the optimizer instance."""

    trainable_layers: list[str]
    """The names of the models the optimizer updates."""


OptimizerSegmentT = TypeVar("OptimizerSegmentT", bound=OptimizerSegment)


class LearnerIntermediate(_Intermediate, Generic[OptimizerSegmentT]):
    """Intermediate representation of a learner during the building process."""

    imports: dict[str, set[str | None]]
    """The imports required for the learner and its sub-layers,
    where the keys are module names and the values are sets of imported names from the corresponding modules."""

    classname: str
    """The name of the learner class."""

    accumulate_gradients: int | None = None
    """The number of steps to accumulate gradients for before performing an optimizer step,
    or `None` if not applicable.

    Populated only by the torch builder through `_intermediate_fields`: the other backends declare
    the accumulation window through their optimizer (`docs/adr/0017`)."""

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

    flow: list[tuple[str, str, str | None] | OptimizerSegmentT]
    """The forward flow during training, where each element is either a tuple of the form (input, output, layer)
    for regular steps, or an `OptimizerSegment` for optimizer steps."""

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
        return unique([m for u in self.flow if isinstance(u, OptimizerSegment) for m in u.trainable_layers])

    @cached_property
    def optimizers(self) -> list[str]:
        """Get the optimizers used in the layer."""
        return unique([u.optimizer for u in self.flow if isinstance(u, OptimizerSegment)])

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

    # Subclasses bind the type to the concrete intermediate they parametrize the builder with,
    # which a `ClassVar` cannot express, so the default is cast to the type variable. Dropping
    # `ClassVar` instead would turn the attribute into a per-instance dataclass field.
    user_defined_learner_layer_type: ClassVar[type[LearnerIntermediateT]] = cast(
        type[LearnerIntermediateT], LearnerIntermediate
    )
    layer_builder_type: ClassVar[type[BaseModelBuilder]] = BaseModelBuilder
    # The template class decides which keys count as learner fields and which fall through to the
    # layer builder, so a framework extending the learner schema must bind its own template here.
    template_type: ClassVar[type[Template[Any]]] = TemplateLearner

    raw: Any
    current_path: str = ""
    template: Template[Any] = field(init=False)
    layer_builder: BaseModelBuilder = field(init=False)

    @classmethod
    def from_path(cls, path: PathLike) -> "BaseLearnerBuilder[LearnerIntermediateT]":
        """Create a learner builder from the given configuration file path."""
        return cls(raw=load_any(path), current_path=str(path))

    def __post_init__(self) -> None:
        """Post-initialization to set up the template."""
        self.template = self.template_type.model_validate(self.raw)
        from_references = {self.current_path: ["__root__"]} if self.current_path else {}
        self.layer_builder = self.layer_builder_type(
            raw=self.template.others, current_path=self.current_path, from_references=from_references
        )

    def _build_segment(
        self,
        imports: defaultdict[str, set[str | None]],
        module: Any,
        learner: LearnerBehavior,
        opt_name: str,
        naming: AutoName,
        layers: dict[str, LayerIntermediate | str],
        others: dict[str, str],
    ) -> OptimizerSegment:
        """Build the optimizer segment of one learner behavior.

        Subclasses extending the learner schema narrow `module` to their own schema type, return their
        own `OptimizerSegment` subclass, and register the extra instances they need (under a name from
        `naming`, rejecting collisions with `layers` and `others`) in `others`, so those instances are
        constructed by the generated learner.
        """
        return OptimizerSegment(
            loss=learner.LOSS,
            optimizer=opt_name,
            trainable_layers=learner.TRAINABLE_LAYERS,
        )

    def _register_shadow_models(
        self,
        imports: defaultdict[str, set[str | None]],
        module: Any,
        naming: AutoName,
        others: dict[str, str],
    ) -> None:
        """Register the learner-level instances a framework declares beside its trainable models.

        Called before any flow is resolved, so each name it adds to `others` is a name the flows may
        reference and an instance the generated learner constructs. The torch and flax builders
        register the `EMA` shadow of each model here (`docs/adr/0021`).
        """

    def _intermediate_fields(self, module: Any) -> dict[str, Any]:
        """Get the framework-specific fields of the built learner intermediate."""
        return {}

    def _get_optimizer(
        self,
        imports: defaultdict[str, set[str | None]],
        optimizer: ObjectPattern,
        trainable_layers: list[str],
    ) -> tuple[str, str]:
        return resolve_object(imports, optimizer)

    def __call__(
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
        module: UserDefinedLearner[LearnerBehavior] = self.template(parameters, merged=False)
        imports: defaultdict[str, set[str | None]] = defaultdict(set)
        imports.update(module.IMPORTS)
        layers: dict[str, LayerIntermediate | str] = {}
        others: dict[str, str] = {}
        naming = AutoName("_")
        for layer in module.TRAINABLE_LAYERS:
            layer = naming(layer)
            others[layer] = layer
        self._register_shadow_models(imports, module, naming, others)

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

        learner_flow: list[tuple[str, str, str | None] | OptimizerSegment] = []
        inference_flow: list[tuple[str, str, str | None]] = []
        for learner in module.LEARNERS:
            opt_inst, opt_cls = self._get_optimizer(imports, learner.OPTIMIZER, learner.TRAINABLE_LAYERS)
            if (opt_name := learner.NAME or naming(to_snake(opt_cls))) in layers or opt_name in others:
                raise SpecError(f'Duplicate variable name "{opt_name}" for optimizer found in the learner flow.')
            others[opt_name] = opt_inst
            segment = self._build_segment(imports, module, learner, opt_name, naming, layers, others)
            learner_flow += _create_flow(learner.FLOW)
            inference_flow += _create_flow(learner.INFERENCE_FLOW or learner.FLOW)
            # Rendered after the flow: `collected_imports` keeps insertion order, so resolving EXTRA any
            # earlier reorders the emitted import header for every config that uses it.
            segment.backward_kwargs = ", ".join(f"{k}={resolve_getter(imports, v)}" for k, v in learner.EXTRA.items())
            learner_flow.append(segment)
        return self.user_defined_learner_layer_type(
            imports=imports,
            classname=classname,
            inputs=module.INPUTS,
            outputs=module.OUTPUTS,
            layers=layers,
            others=others,
            flow=learner_flow,
            inference_flow=inference_flow,
            **self._intermediate_fields(module),
        )


__all__ = [
    "BaseLearnerBuilder",
    "BaseModelBuilder",
    "LayerIntermediate",
    "LearnerIntermediate",
    "OptimizerSegment",
]

if not TYPE_CHECKING:
    import sys

    from structcast.utils.lazy_import import LazySelectedImporter

    sys.modules[__name__] = LazySelectedImporter(__name__, globals())
