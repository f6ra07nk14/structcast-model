"""Builder for Flax (nnx) models."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

from structcast_model.builders.base_builder import BaseModelBuilder, LayerIntermediate


class FlaxLayerIntermediate(LayerIntermediate):
    """Intermediate representation of a Flax nnx module.

    Generates a ``flax.nnx.Module`` subclass whose ``__init__`` accepts a ``rngs: flax.nnx.Rngs`` argument
    (passed down to sub-module constructors via ``eval: rngs`` in the YAML template) and
    whose ``__call__`` accepts a ``training: bool`` keyword argument for toggling training vs. inference behaviour.

    Example:
        >>> from structcast_model.builders.flax_builder import FlaxLayerIntermediate
        >>> script = FlaxLayerIntermediate(
        ...     classname="Unit",
        ...     imports={},
        ...     inputs=["x"],
        ...     outputs=["y"],
        ...     layers={},
        ...     flow=[("x", "y", None)],
        ...     inference_flow=[],
        ...     structured_output=False,
        ... )._get_layer_script("Unit", [])
        >>> "class Unit(flax.nnx.Module):" in script
        True
    """

    default_imports: ClassVar[dict[str, set[str | None]]] = {"flax": {None}}
    """Default imports for Flax nnx modules."""

    def _get_layer(self, layername: str) -> str:
        """Get the sub-module with the given name."""
        return f"self.{layername}"

    def _get_layer_script(self, class_name: str, initialized_layers: list[str]) -> str:
        """Return the Python class script for a Flax nnx module."""
        indent = " " * 4
        sep = "\n" + indent * 2
        if self._forward_inference_flow:
            codes = [
                "if training:",
                *[indent + c for c in self._forward_training_flow],
                "else:",
                *[indent + c for c in self._forward_inference_flow],
            ]
        else:
            codes = self._forward_training_flow
        inputs = self._forward_inputs
        inputs += ", " if inputs else ""
        init_body = sep.join([f"self.{v}" for v in initialized_layers]) if initialized_layers else "pass"
        return f"""\
class {class_name}(flax.nnx.Module):

    def __init__(self, *, rngs: flax.nnx.Rngs, training: bool = True):
        self.inputs = {self.inputs}
        self.outputs = {self.outputs}
        self.training = training
        {init_body}

    def __call__(self, {inputs}*, training = None, **kwargs):
        training = self.training if training is None else training
        {sep.join(codes)}
        return {self._forward_outputs}

    def set_view(self, training = None):
        if training is not None:
            self.training = training
"""


@dataclass(kw_only=True, slots=True)
class FlaxBuilder(BaseModelBuilder[FlaxLayerIntermediate]):
    """Builder for Flax nnx models.

    Generates Python scripts containing ``flax.nnx.Module`` subclasses from a YAML template,
    following the same template-to-code pipeline as :class:`~structcast_model.builders.torch_builder.TorchBuilder`.

    Sub-modules that require a random-number generator should receive ``rngs: "eval: rngs"`` in
    their ``_call_`` arguments so that the builder emits ``rngs=rngs`` in the generated ``__init__`` body.

    Example:
        >>> from structcast_model.builders.flax_builder import FlaxBuilder
        >>> layer_spec = {"_obj_": [["_addr_", "flax.nnx.Linear"], {"_call_": {"in_features": 8, "out_features": 4}}]}
        >>> raw = {"INPUTS": ["x"], "OUTPUTS": ["y"], "FLOW": [["x", "y", layer_spec]]}
        >>> built = FlaxBuilder(raw=raw)(classname="TinyNet")
        >>> built.classname
        'TinyNet'
    """

    user_defined_layer_type: ClassVar[type[FlaxLayerIntermediate]] = FlaxLayerIntermediate


__all__ = ["FlaxBuilder", "FlaxLayerIntermediate"]


if not TYPE_CHECKING:
    import sys

    from structcast.utils.lazy_import import LazySelectedImporter

    sys.modules[__name__] = LazySelectedImporter(__name__, globals())
