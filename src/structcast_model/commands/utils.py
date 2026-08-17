"""Utility functions for the commands package."""

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


TEMPLATE_PARAM_HELP = (
    "Parameters to format the template configuration file with. "
    'Each parameter is "group: {...}", where the group name selects the parameter group '
    '("SHARED" applies to every group, "DEFAULT" to the default one) and the value is a dictionary of '
    "keyword arguments for the template. Repeat the option for more groups; a group named twice keeps "
    "only the last occurrence, so pass all of a group's keys together."
)


def object_pattern_help(subject: str, symbol: str, *, keyed: bool = False, call: bool = True) -> str:
    """Build the help text documenting the object pattern accepted for `subject`.

    Args:
        subject (str): What the pattern instantiates, written with its article so that it reads both as
            "used to instantiate <subject>" and as "if <subject> is defined as ...", e.g. "the model".
        symbol (str): The symbol used in the example, e.g. "MyModel".
        keyed (bool): Whether the pattern is keyed by a name, e.g. "model_name: [_obj_, ...]".
        call (bool): Whether the example object is called, adding the "_call_" entry to the pattern.

    Returns:
        str: The help text documenting both accepted spellings of the object pattern.
    """
    prefix = "model_name: " if keyed else ""
    call_part = ", {_call_: {...}}" if call else ""
    definition = "(...)" if call else ""
    return (
        f"The object pattern used to instantiate {subject}. "
        f"For example, if {subject} is defined as `my_package.{symbol}{definition}`, then the pattern should be "
        f'"{prefix}[_obj_, {{_addr_: my_package.{symbol}, _file_: my_package.py}}{call_part}]" or '
        f'"{prefix}[_obj_, [_addr_, my_package.{symbol}, my_package.py]{call_part}]".'
    )


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


def instantiate_object(raw: Any) -> Any:
    """Instantiate an object from a raw pattern using the structcast instantiator.

    Args:
        raw (Any): The raw pattern to instantiate.

    Returns:
        Any: The instantiated object.
    """
    return instantiator.ObjectPattern.model_validate(raw).build().runs[0]


__all__ = [
    "TEMPLATE_PARAM_HELP",
    "bool_or_path_or_dict_parser",
    "dict_parser",
    "instantiate_object",
    "object_pattern_help",
    "path_or_any_parser",
    "reduce_dict",
    "tensor_shape_parser",
]


if not TYPE_CHECKING:
    import sys

    from structcast.utils.lazy_import import LazySelectedImporter

    sys.modules[__name__] = LazySelectedImporter(__name__, globals())
