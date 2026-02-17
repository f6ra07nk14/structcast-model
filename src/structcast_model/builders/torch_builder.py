"""Builder for PyTorch models."""

from dataclasses import dataclass
from typing import ClassVar, Literal

from structcast_model.builders.core import BaseBuilder, LayerIntermediate


class TorchLayerIntermediate(LayerIntermediate):
    """Intermediate representation of a PyTorch layer."""

    layer_call_name: ClassVar[Literal["forward"]] = "forward"
    """The name of the method to call the layer, if applicable."""

    def _get_layer(self, layername: str) -> str:
        """Get the sub-layer with the given name."""
        return f"self.{layername}"

    def _get_script(self, class_name: str, initialized_layers: list[str]) -> str:
        """Implement the method to get the script for the layer."""
        indent = " " * 4
        sep = "\n" + indent * 2
        if self._forward_inference_flow:
            codes = [
                "if self.training:",
                *[indent + c for c in self._forward_training_flow],
                "else:",
                *[indent + c for c in self._forward_inference_flow],
            ]
        else:
            codes = self._forward_training_flow
        return f"""\
class {class_name}(torch.nn.Module):

    def __init__(self):
        super().__init__()
        {sep.join([f"self.{v}" for v in initialized_layers])}

    def forward(self, {self._forward_inputs}, **kwargs):
        {sep.join(codes)}
        return {self._forward_outputs}
"""


@dataclass(kw_only=True, slots=True)
class TorchBuilder(BaseBuilder[TorchLayerIntermediate]):
    """Builder for PyTorch models."""

    user_defined_layer_type: ClassVar[type[TorchLayerIntermediate]] = TorchLayerIntermediate


# todo: implement backward
