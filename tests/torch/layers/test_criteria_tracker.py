"""Tests for criteria tracker layer."""

from structcast_model.torch.layers.criteria_tracker import CriteriaTracker
import torch


def test_criteria_tracker_accumulates_running_means() -> None:
    """Update trackers with running means across calls."""
    tracker = CriteriaTracker(["loss", "acc"])
    result1 = tracker({"loss": torch.tensor(2.0), "acc": torch.tensor(0.5)})
    result2 = tracker({"loss": torch.tensor(4.0), "acc": torch.tensor(1.0)})
    assert torch.isclose(result1["loss"], torch.tensor([2.0])).all()
    assert torch.isclose(result1["acc"], torch.tensor([0.5])).all()
    assert torch.isclose(result2["loss"], torch.tensor([3.0])).all()
    assert torch.isclose(result2["acc"], torch.tensor([0.75])).all()
    assert torch.equal(tracker.total, torch.tensor([2.0]))


def test_criteria_tracker_reset() -> None:
    """Reset all trackers and total counter."""
    tracker = CriteriaTracker(["loss"])
    tracker({"loss": torch.tensor(1.0)})
    tracker.reset()
    assert torch.equal(tracker.total, torch.tensor([0.0]))
    assert torch.equal(tracker.trackers["loss"], torch.tensor([0.0]))
