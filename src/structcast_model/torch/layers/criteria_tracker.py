"""A tracker for multiple criteria."""

from torch.nn import Module

from structcast_model.torch.types import Tensor
from torch import autocast, float32, no_grad, zeros


class CriteriaTracker(Module):
    """A tracker for multiple criteria."""

    def __init__(self, criteria: list[str]) -> None:
        """Initialize the criteria tracker."""
        super().__init__()
        self.criteria = criteria
        self.register_buffer("total", zeros(1, dtype=float32))
        self.trackers = {}
        for criterion in criteria:
            self.register_buffer(f"{criterion}", zeros(1, dtype=float32))
            self.trackers[criterion] = getattr(self, criterion)

    @no_grad()
    def forward(self, values: dict[str, Tensor]) -> dict[str, Tensor]:
        """Update the total and count for each criterion."""
        first = next(iter(values.values()))
        with autocast(device_type=first.device.type, enabled=False):
            self.total.add_(first.new_ones(1, dtype=float32))
            return {c: v.add_(values[c].to(float32)).div(self.total) for c, v in self.trackers.items()}

    @no_grad()
    def reset(self) -> None:
        """Reset all trackers."""
        for tracker in self.trackers.values():
            tracker.zero_()
        self.total.zero_()


__all__ = ["CriteriaTracker"]
