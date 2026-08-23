"""Utility functions for the commands package."""

from collections.abc import Mapping
from hashlib import sha256
from json import dumps
from typing import TYPE_CHECKING, Any

from structcast.utils.base import load_yaml_from_string

if TYPE_CHECKING:
    import pydantic
    from structcast.core import instantiator

    from structcast_model.builders import schema
    from structcast_model.utils import base
else:
    from structcast.utils.lazy_import import LazyModuleImporter

    pydantic = LazyModuleImporter("pydantic")
    instantiator = LazyModuleImporter("structcast.core.instantiator")
    schema = LazyModuleImporter("structcast_model.builders.schema")
    base = LazyModuleImporter("structcast_model.utils.base")


def reduce_dict(params: list[dict[str, Any]] | None) -> dict[str, Any]:
    """Reduce a list of dictionaries into a single dictionary.

    Args:
        params (list[dict[str, Any]] | None): A list of dictionaries to reduce.
            If None, an empty dictionary is returned.

    Returns:
        dict[str, Any]: The reduced dictionary.
    """
    return {k: v for p in params for k, v in p.items()} if params else {}


def dict_parser(value: str) -> dict[str, Any]:
    """Parse a YAML string into a dictionary.

    Args:
        value (str): The YAML string to parse.

    Returns:
        dict[str, Any]: The parsed dictionary.
    """
    return pydantic.TypeAdapter(dict[str, Any]).validate_python(load_yaml_from_string(value)) if value else {}


def path_or_any_parser(value: str) -> dict[str, Any] | None:
    """Parse a YAML string into a boolean, a path, or a dictionary.

    If the string is a valid path to a file, the content of the file will be loaded and returned.
    Otherwise, the string will be parsed as a YAML string and returned as a dictionary.

    Args:
        value (str): The string to parse.

    Returns:
        dict[str, Any] | None: The parsed dictionary, or None if the input is empty.
    """
    if not value:
        return None
    data = load_yaml_from_string(value)
    if isinstance(data, str):
        return base.load_any(data) if data else None
    return data


def bool_or_path_or_dict_parser(value: str) -> dict[str, Any] | None:
    """Parse a YAML string into a boolean, a path, or a dictionary.

    If the string is a valid path to a file, the content of the file will be loaded and returned.
    If the string is "true" or "false" (case-insensitive), it will be parsed as a boolean.
    Otherwise, the string will be parsed as a YAML string and returned as a dictionary.

    Args:
        value (str): The string to parse.

    Returns:
        dict[str, Any] | None: The parsed dictionary, or None if the input is empty or "false".
    """
    if not value:
        return None
    data: bool | str | dict[str, Any] = pydantic.TypeAdapter(bool | str | dict[str, Any]).validate_python(
        load_yaml_from_string(value)
    )
    if isinstance(data, bool):
        return {} if data else None
    if isinstance(data, str):
        return base.load_any(data) if data else None
    return data


def tensor_shape_parser(value: str) -> dict[str, Any]:
    """Parse a YAML string into a dictionary of tensor shapes.

    The input string is a YAML representation of a dictionary mapping input names to tensor specifications,
    validated through `structcast_model.builders.schema.TensorSpecTree`. A specification is either the compact
    form, which is a plain shape, or the explicit form, which is a mapping with the `_SHAPE_` key and the
    optional `_DTYPE_` and `_INIT_` keys. Specifications can be nested in dictionaries and lists.
    For example: `{image: [224, 224, 3], tokens: {_SHAPE_: [512], _DTYPE_: int64},
    metadata: {feature1: [10], feature2: [5]}}`.

    Args:
        value (str): The YAML string to parse.

    Returns:
        dict[str, Any]: The parsed dictionary of tensor shapes, dumped back to plain Python data.
            A specification with default dtype and no initializer collapses to a shape tuple,
            while any other specification stays a mapping keyed by `_SHAPE_`, `_DTYPE_` and `_INIT_`.
    """
    if not value:
        return {}
    adapter = pydantic.TypeAdapter(dict[str, schema.TensorSpecTree])
    return adapter.dump_python(adapter.validate_python(load_yaml_from_string(value)))


def get_module_outputs(module: Any, default: list[str] | None, name: str) -> list[str]:
    """Return output names from a module attribute or the provided default, raising if neither is available.

    Args:
        module (Any): The module whose ``outputs`` attribute is read when no default is given.
        default (list[str] | None): Output names given on the command line, which win over the attribute.
        name (str): How the module is named on the command line, used to name the option in the error.

    Returns:
        list[str]: The output names.
    """
    if default:
        return default
    if hasattr(module, "outputs"):
        return module.outputs
    raise ValueError(
        f'Module "{name}" does not have an "outputs" attribute. '
        f'Please provide default outputs using the "--{name}-outputs" option.'
    )


def config_hash(model_patterns: list[dict], learner_pattern: Any, shapes: Mapping[str, Any]) -> str:
    """Return the digest of what a run trains: its model patterns, its learner pattern and its shapes.

    Recorded in the saved training state so a resumed run can be told apart from the configuration it
    was saved from. The optimizers are not part of it: they are hashed separately, by the builder
    that emits them, and reported per segment as `optimizer_hashes`.

    Args:
        model_patterns (list[dict]): The patterns the run's models are built from.
        learner_pattern (Any): The pattern the run's learner is built from.
        shapes (Mapping[str, Any]): The input shapes the models are traced with.

    Returns:
        str: The hexadecimal digest of the three together.
    """
    payload = {"models": model_patterns, "learner": learner_pattern, "shapes": shapes}
    return sha256(dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def check_gpu_memory_fraction(fraction: float | None) -> None:
    """Reject a `--gpu-memory-fraction` outside (0, 1], where it would cap nothing while reading as a cap.

    Each framework applies the fraction through whatever mechanism it actually has, but they all
    reject the same values with the same sentence, so what an operator reads does not depend on which
    `train` they ran.

    Args:
        fraction (float | None): The option's value, None when it was omitted.

    Raises:
        ValueError: If the fraction is given and is not greater than 0 and at most 1.
    """
    if fraction is not None and not 0 < fraction <= 1:
        raise ValueError(f"--gpu-memory-fraction must be in (0, 1]. Got: {fraction}.")


def strategy_parser(value: str) -> Any:
    """Parse `--strategy`: a bare name is a preset, anything else an object pattern or a path to one.

    Args:
        value (str): The option's raw value.

    Returns:
        Any: The preset name, or the parsed pattern.
    """
    return value if value.isidentifier() else path_or_any_parser(value)


def instantiate_object(raw: Any) -> Any:
    """Instantiate an object from a raw pattern using the structcast instantiator.

    Args:
        raw (Any): The raw pattern to instantiate.

    Returns:
        Any: The instantiated object.
    """
    return instantiator.ObjectPattern.model_validate(raw).build().runs[0]


__all__ = [
    "bool_or_path_or_dict_parser",
    "check_gpu_memory_fraction",
    "config_hash",
    "dict_parser",
    "get_module_outputs",
    "instantiate_object",
    "path_or_any_parser",
    "reduce_dict",
    "strategy_parser",
    "tensor_shape_parser",
]


if not TYPE_CHECKING:
    import sys

    from structcast.utils.lazy_import import LazySelectedImporter

    sys.modules[__name__] = LazySelectedImporter(__name__, globals())
