"""Core functionality for StructCast-Model builders."""

from dataclasses import dataclass, field
from functools import cached_property
from typing import Any, Self, cast

from pydantic import BaseModel, ConfigDict, Field, FilePath, model_validator
from structcast.core.specifier import FlexSpec
from structcast.core.template import extend_structure

from structcast_model.builders.auto_name import AutoName


class BaseConfig(BaseModel):
    """Base configuration."""

    model_config = ConfigDict(frozen=True, validate_default=True, extra="forbid", serialize_by_alias=True)


class UserLayer(BaseConfig):
    """User layer configuration."""

    CFG: FilePath | None = None
    """Path to the layer configuration file."""

    TYPE: str | None = None
    """Type of the layer."""

    PARAM: dict[str, Any] = Field(default_factory=dict)
    """Parameters for the layer."""


class LayerBehavior(BaseConfig):
    """Layer behavior configuration."""

    INPUTS: FlexSpec | None = None
    """Inputs of the layer."""

    OUTPUTS: FlexSpec | None = None
    """Outputs of the layer."""

    LAYER: UserLayer
    """The name of the layer class or an instance of the layer."""

    NAME: str | None = None
    """The name of the layer class or an instance of the layer."""


class UserDefinedLayer(BaseConfig):
    """User defined layer configuration."""

    INPUTS: list[str] = Field(default_factory=list)
    """Inputs of the layer."""

    OUTPUTS: list[str] = Field(default_factory=list)
    """Outputs of the layer."""

    FLOW: list[LayerBehavior] = Field(default_factory=list)
    """Flow of the layer."""

    STRUCTURED_OUTPUT: bool = False
    """Whether the output is structured."""


_TEMPLATE_ALIASES = ["_jinja_", "_jinja_pipe_", "_jinja_yaml_", "_jinja_json_"]


class TemplateLayer(BaseConfig):
    """Template-like configuration."""

    model_config = ConfigDict(extra="allow")

    DEFAULT_PARAMETERS: dict[str, Any] = Field(default_factory=dict)
    """Default parameters for the template."""

    SHARED_VARIABLES: dict[str, Any] = Field(default_factory=dict)
    """Shared variables."""

    @property
    def model_extra(self) -> dict[str, Any]:
        """Get extra fields set during validation.

        Returns:
            A dictionary of extra fields, or `None` if `config.extra` is not set to `"allow"`.
        """
        return cast(dict[str, Any], self.__pydantic_extra__)

    @model_validator(mode="after")
    def _validate_template(self) -> Self:
        if duplicate_keys := set(self.DEFAULT_PARAMETERS) & set(self.SHARED_VARIABLES):
            raise ValueError(f"Duplicate keys found between DEFAULT_PARAMETERS and SHARED_VARIABLES: {duplicate_keys}")
        return self

    @cached_property
    def _raw_and_others(self) -> tuple[dict[str, Any], dict[str, Any]]:
        target_fields = list(UserDefinedLayer.model_fields) + _TEMPLATE_ALIASES
        raw, others = {}, {}
        for key, value in self.model_extra.items():
            (raw if key in target_fields else others)[key] = value
        return raw, others

    @property
    def raw(self) -> dict[str, Any]:
        """Get the raw fields for the `UserDefinedLayer`."""
        return self._raw_and_others[0]

    @property
    def others(self) -> dict[str, Any]:
        """Get the other fields that are not part of the `UserDefinedLayer`."""
        return self._raw_and_others[1]

    def format(self, **kwargs: dict[str, Any]) -> UserDefinedLayer:
        """Format the template with the given parameters.

        Args:
            **kwargs: Runtime parameters for formatting the template.

        Returns:
            A `UserDefinedLayer` instance created from the formatted template.
        """
        if duplicate_keys := set(kwargs) & set(self.SHARED_VARIABLES):
            raise ValueError(f"Duplicate keys found between runtime parameters and SHARED_VARIABLES: {duplicate_keys}")
        template_kwargs = {"default": {**self.DEFAULT_PARAMETERS, **self.SHARED_VARIABLES, **kwargs}}
        user_defined_layer = extend_structure(self.raw, template_kwargs=template_kwargs, default="default")
        return UserDefinedLayer.model_validate(user_defined_layer)


@dataclass(kw_only=True, slots=True)
class LayerIntermediate:
    """Intermediate representation of a layer during the building process."""

    root: TemplateLayer
    from_paths: list[str] = field(default_factory=list)
    from_layers: list[str] = field(default_factory=list)
    auto_name: AutoName = field(default_factory=AutoName)


def build(
    raw: dict[str, Any],
    training: bool,
    *,
    template_kwargs: dict[str, Any] | None = None,
) -> Any:
    pass
