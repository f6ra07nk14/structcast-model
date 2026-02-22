"""Utility functions for the commands package."""

from typing import Any

from pydantic import TypeAdapter, ValidationError
from structcast.utils.base import load_yaml_from_string
from structcast.utils.security import get_default_dir


def reduce_dict(params: list[dict[str, Any]] | None) -> dict[str, Any]:
    """Reduce a list of dictionaries into a single dictionary."""
    return {k: v for p in params for k, v in p.items()} if params else {}


def dict_parser(value: str) -> dict[str, Any]:
    """Parse a YAML string into a dictionary."""
    return TypeAdapter(dict[str, Any]).validate_python(load_yaml_from_string(value)) if value else {}


def tensor_shape_parser(value: str) -> dict[str, Any]:
    """Parse a YAML string into a dictionary of tensor shapes."""

    def _check(shape: Any) -> Any:
        try:
            return TypeAdapter(tuple[int, ...]).validate_python(shape)
        except ValidationError:
            pass
        if isinstance(shape, dict):
            return {k: _check(v) for k, v in shape.items()}
        if isinstance(shape, (list, tuple)):
            return [_check(v) for v in shape]
        raise ValueError(f"Invalid tensor shape: {shape}")

    return _check(TypeAdapter(dict[str, Any]).validate_python(load_yaml_from_string(value))) if value else {}


__all__ = ["dict_parser", "reduce_dict", "tensor_shape_parser"]


def __dir__() -> list[str]:
    return get_default_dir(globals())
