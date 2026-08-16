"""Tests for criteria tracker layer."""

from torch._subclasses.fake_tensor import FakeTensorMode

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


def test_criteria_tracker_updates_own_buffers_after_dtype_move() -> None:
    """Accumulate into the module's own buffers, which is what a device/dtype move and a checkpoint see.

    `Module._apply` (`.to()`, `.cuda()`, `.double()`) replaces the tensors in `_buffers` instead of mutating them, so
    tracking into tensors captured at construction time would leave `state_dict` frozen at zero after any move and mix
    the pre-move dtype into the accumulation.
    """
    tracker = CriteriaTracker(["loss"]).double()
    tracker({"loss": torch.tensor(2.0)})
    assert tracker.get_buffer("loss").dtype == torch.float64
    assert torch.equal(tracker.get_buffer("loss"), torch.tensor([2.0], dtype=torch.float64))
    assert torch.equal(tracker.state_dict()["loss"], torch.tensor([2.0], dtype=torch.float64))


def test_criteria_tracker_accepts_host_side_values_after_device_move() -> None:
    """Pull incoming values onto the buffers' device, so values follow the module as its buffers already do.

    `Module._apply` (`.cuda()`, `.to(device)`) moves the checkpoint-facing buffers but not the caller's values, so a
    host-resident value would otherwise hit a cross-device in-place add. `FakeTensorMode` reproduces that mismatch on a
    CPU-only build; torch exempts 0-dim CPU tensors from the same-device rule, hence the 1-element value.
    """
    with FakeTensorMode():
        tracker = CriteriaTracker(["loss"]).to("cuda:0")
        result = tracker({"loss": torch.zeros(1, device="cpu")})
        assert result["loss"].device == torch.device("cuda:0")


def test_criteria_tracker_reset() -> None:
    """Reset all trackers and total counter."""
    tracker = CriteriaTracker(["loss"])
    tracker({"loss": torch.tensor(1.0)})
    tracker.reset()
    assert torch.equal(tracker.total, torch.tensor([0.0]))
    assert torch.equal(tracker.get_buffer("loss"), torch.tensor([0.0]))
