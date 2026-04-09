"""Utility functions for the commands package."""

from typing import TYPE_CHECKING, Any

from structcast.utils.base import load_yaml_from_string

if TYPE_CHECKING:
    import pydantic
    from structcast.core import instantiator

    from structcast_model.utils import base
else:
    from structcast.utils.lazy_import import LazyModuleImporter

    pydantic = LazyModuleImporter("pydantic")
    instantiator = LazyModuleImporter("structcast.core.instantiator")
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
    data = pydantic.TypeAdapter(bool | str | dict[str, Any]).validate_python(load_yaml_from_string(value))
    if isinstance(data, bool):
        return {} if data else None
    if isinstance(data, str):
        return base.load_any(data) if data else None
    return data


def tensor_shape_parser(value: str) -> dict[str, Any]:
    """Parse a YAML string into a dictionary of tensor shapes.

    The input string can be a YAML representation of a dictionary, where the values can be tuples, lists,
    or dictionaries representing tensor shapes.
    For example: `{image: [224, 224, 3], metadata: {feature1: 10, feature2: 5}}`.

    Args:
        value (str): The YAML string to parse.

    Returns:
        dict[str, Any]: The parsed dictionary of tensor shapes.
    """

    def _check(shape: Any) -> Any:
        try:
            return pydantic.TypeAdapter(tuple[int, ...]).validate_python(shape)
        except pydantic.ValidationError:
            pass
        if isinstance(shape, dict):
            return {k: _check(v) for k, v in shape.items()}
        if isinstance(shape, (list, tuple)):
            return [_check(v) for v in shape]
        raise ValueError(f"Invalid tensor shape: {shape}")

    return _check(pydantic.TypeAdapter(dict[str, Any]).validate_python(load_yaml_from_string(value))) if value else {}


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
    "dict_parser",
    "instantiate_object",
    "path_or_any_parser",
    "reduce_dict",
    "tensor_shape_parser",
]


if not TYPE_CHECKING:
    import sys

    from structcast.utils.lazy_import import LazySelectedImporter

    sys.modules[__name__] = LazySelectedImporter(__name__, globals())
