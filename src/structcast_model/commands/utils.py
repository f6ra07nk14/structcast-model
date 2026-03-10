"""Utility functions for the commands package."""

from typing import TYPE_CHECKING, Any

from structcast.utils.base import load_yaml_from_string

if TYPE_CHECKING:
    import pydantic

    from structcast_model.utils import base
else:
    from structcast.utils.lazy_import import LazyModuleImporter

    pydantic = LazyModuleImporter("pydantic")
    base = LazyModuleImporter("structcast_model.utils.base")


def reduce_dict(params: list[dict[str, Any]] | None) -> dict[str, Any]:
    """Reduce a list of dictionaries into a single dictionary."""
    return {k: v for p in params for k, v in p.items()} if params else {}


def dict_parser(value: str) -> dict[str, Any]:
    """Parse a YAML string into a dictionary."""
    return pydantic.TypeAdapter(dict[str, Any]).validate_python(load_yaml_from_string(value)) if value else {}


def bool_or_path_or_dict_parser(value: str) -> dict[str, Any] | None:
    """Parse a YAML string into a boolean, a path, or a dictionary."""
    if not value:
        return None
    data = pydantic.TypeAdapter(bool | str | dict[str, Any]).validate_python(load_yaml_from_string(value))
    if isinstance(data, bool):
        return {} if data else None
    if isinstance(data, str):
        return base.load_any(data) if data else None
    return data


def tensor_shape_parser(value: str) -> dict[str, Any]:
    """Parse a YAML string into a dictionary of tensor shapes."""

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


__all__ = ["bool_or_path_or_dict_parser", "dict_parser", "reduce_dict", "tensor_shape_parser"]


if not TYPE_CHECKING:
    import sys

    from structcast.utils.lazy_import import LazySelectedImporter

    sys.modules[__name__] = LazySelectedImporter(__name__, globals())
