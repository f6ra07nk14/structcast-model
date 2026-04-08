"""Builder for Keras models."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

from structcast_model.builders.base_builder import BaseModelBuilder, LayerIntermediate


class KerasLayerIntermediate(LayerIntermediate):
    """Intermediate representation of a Keras layer.

    Generates a ``keras.Layer`` subclass with a ``call`` method that accepts a ``training`` keyword argument,
    propagating it to every sub-layer call to support Keras's standard training/inference mode.

    Example:
        >>> from structcast_model.builders.keras_builder import KerasLayerIntermediate
        >>> script = KerasLayerIntermediate(
        ...     classname="Unit",
        ...     imports={},
        ...     inputs=["x"],
        ...     outputs=["y"],
        ...     layers={},
        ...     flow=[("x", "y", None)],
        ...     inference_flow=[],
        ...     structured_output=False,
        ... )._get_layer_script("Unit", [])
        >>> "class Unit(keras.layers.Layer):" in script
        True
    """

    default_imports: ClassVar[dict[str, set[str | None]]] = {"keras": {None}}
    """Default imports for Keras layers."""

    def _get_layer(self, layername: str) -> str:
        """Get the sub-layer with the given name."""
        return f"self.{layername}"

    def _forward_flow(self, flow: list[tuple[str, str, str | None]]) -> list[str]:
        """Generate call expressions that forward the ``training`` flag to sub-layers."""
        return [f"{o} = {self._get_layer(L)}({i}, training=training)" if L else f"{o} = {i}" for i, o, L in flow]

    def _get_layer_script(self, class_name: str, initialized_layers: list[str]) -> str:
        """Return the Python class script for a Keras layer."""
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
class {class_name}(keras.layers.Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.input_names = {self.inputs}
        self.output_names = {self.outputs}
        {init_body}

    def call(self, {inputs}*, training = None, **kwargs):
        {sep.join(codes)}
        return {self._forward_outputs}
"""


@dataclass(kw_only=True, slots=True)
class KerasBuilder(BaseModelBuilder[KerasLayerIntermediate]):
    """Builder for Keras models.

    Generates Python scripts containing ``keras.Layer`` subclasses from a YAML template,
    following the same template-to-code pipeline as :class:`~structcast_model.builders.torch_builder.TorchBuilder`.

    Example:
        >>> from structcast_model.builders.keras_builder import KerasBuilder
        >>> raw = {
        ...     "INPUTS": ["x"],
        ...     "OUTPUTS": ["y"],
        ...     "FLOW": [["x", "y", {"_obj_": [["_addr_", "keras.layers.Dense"], ["_call_", {"units": 4}]]}]],
        ... }
        >>> built = KerasBuilder(raw=raw)(classname="TinyNet")
        >>> built.classname
        'TinyNet'
    """

    user_defined_layer_type: ClassVar[type[KerasLayerIntermediate]] = KerasLayerIntermediate


__all__ = ["KerasBuilder", "KerasLayerIntermediate"]


if not TYPE_CHECKING:
    import sys

    from structcast.utils.lazy_import import LazySelectedImporter

    sys.modules[__name__] = LazySelectedImporter(__name__, globals())
