"""A tracker for multiple criteria."""

from torch.nn import Module

from structcast_model.torch.types import Tensor
from torch import autocast, float32, no_grad, zeros


class CriteriaTracker(Module):
    """A tracker for multiple criteria."""

    total: Tensor
    """Number of updates, registered as a buffer; annotated so `Module.__getattr__` does not widen it."""

    def __init__(self, criteria: list[str]) -> None:
        """Initialize the criteria tracker."""
        super().__init__()
        self.criteria = criteria
        self.register_buffer("total", zeros(1, dtype=float32))
        for criterion in criteria:
            self.register_buffer(f"{criterion}", zeros(1, dtype=float32))

    @no_grad()
    def forward(self, values: dict[str, Tensor]) -> dict[str, Tensor]:
        """Update the total and count for each criterion."""
        with autocast(device_type=self.total.device.type, enabled=False):
            self.total.add_(self.total.new_ones(1, dtype=float32))
            return {c: self.get_buffer(c).add_(values[c].to(float32)).div(self.total) for c in self.criteria}

    @no_grad()
    def reset(self) -> None:
        """Reset all trackers."""
        for criterion in self.criteria:
            self.get_buffer(criterion).zero_()
        self.total.zero_()


__all__ = ["CriteriaTracker"]
