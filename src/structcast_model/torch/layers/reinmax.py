"""ReinMax.

References:
    - `Bridging Discrete and Backpropagation: Straight-Through and Beyond <https://arxiv.org/abs/2304.08612>`_
    - `ReinMax GitHub <https://github.com/microsoft/ReinMax>`_
"""

from typing import Any

from structcast.utils.security import get_default_dir
from torch.autograd import Function
from torch.jit import unused

from structcast_model.torch.types import Tensor
import torch


class ReinMaxCore(Function):
    """ReinMax gradient estimator."""

    @staticmethod
    def forward(ctx: Any, logits: Tensor, tau: Tensor) -> tuple[Tensor, Tensor]:
        """Forward method."""
        y_soft = logits.softmax(dim=-1)
        sample = torch.multinomial(y_soft, num_samples=1, replacement=True)
        one_hot = torch.zeros_like(y_soft, memory_format=torch.legacy_contiguous_format).scatter_(-1, sample, 1.0)
        ctx.save_for_backward(one_hot, logits, y_soft, tau)
        return one_hot, y_soft

    @staticmethod
    def backward(ctx: Any, grad_at_sample: Tensor, grad_at_p: Tensor) -> Any:  # type: ignore[override]
        """Backward method."""
        one_hot_sample, logits, y_soft, tau = ctx.saved_tensors

        shifted_y_soft = 0.5 * ((logits / tau).softmax(dim=-1) + one_hot_sample)
        grad_at_input_1 = (2 * grad_at_sample) * shifted_y_soft
        grad_at_input_1 = grad_at_input_1 - shifted_y_soft * grad_at_input_1.sum(dim=-1, keepdim=True)

        grad_at_input_0 = (-0.5 * grad_at_sample + grad_at_p) * y_soft
        grad_at_input_0 = grad_at_input_0 - y_soft * grad_at_input_0.sum(dim=-1, keepdim=True)

        grad_at_input = grad_at_input_0 + grad_at_input_1
        return grad_at_input - grad_at_input.mean(dim=-1, keepdim=True), None


@unused
def reinmax(logits: Tensor, tau: float = 1.0) -> tuple[Tensor, Tensor]:
    """ReinMax.

    Example:
        ```python
        >>> data = torch.randn((3, 4))
        >>> y_hard, y_soft = reinmax(data, tau=1.0)
        >>> y_hard.shape == data.shape
        True
        >>> y_soft.shape == data.shape
        True

        ```
    """
    if tau < 1:
        raise ValueError("ReinMax prefers to set the temperature (tau) larger or equal to 1.")
    shape = logits.size()
    logits = logits.view(-1, shape[-1])
    grad_sample, y_soft = ReinMaxCore.apply(logits, logits.new_empty(1).fill_(tau))
    return grad_sample.view(shape), y_soft.view(shape)


__all__ = ["reinmax"]


def __dir__() -> list[str]:
    return get_default_dir(globals())
