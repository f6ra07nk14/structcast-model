"""Example optimizer and scheduler compositions for PyTorch training runs.

The package ships `structcast_model.torch.optimizers.create_opt`, which builds an optimizer and
nothing else. A learning-rate schedule is use-case specific, so combinations live here as example
code and are referenced from a configuration by file path -- this is where the old
`structcast_model.torch.optimizers.create_with_scheduler` patterns move to:

```yaml
OPTIMIZER:
  - _obj_
  - _addr_: AdamWWithCosine
    _file_: examples/torch/optimizers.py
  - _bind_:
      optimizer_kwargs: {opt: adamw, lr: 0.001, weight_decay: 0.05}
      scheduler_kwargs: {sched: cosine, num_epochs: 100, criterion: loss}
```

The trainer scans the learner's optimizers for event protocols, so `on_update` and `on_epoch_end`
below step the schedule without any registration.
"""

from typing import Any

from timm.scheduler.scheduler_factory import create_scheduler_v2

from structcast_model.torch.optimizers import create_opt
import torch


class AdamWWithCosine:
    """An optimizer stepping a timm scheduler, usable anywhere a `torch.optim.Optimizer` is."""

    def __init__(self, params: Any, optimizer_kwargs: dict[str, Any], scheduler_kwargs: dict[str, Any]) -> None:
        """Create the optimizer and its schedule.

        Args:
            params (Any): The named model parameters to optimize.
            optimizer_kwargs (dict[str, Any]): Keyword arguments for `create_opt`, e.g. `opt` and `lr`.
            scheduler_kwargs (dict[str, Any]): Keyword arguments for timm's `create_scheduler_v2`,
                plus the required `criterion` naming the tracked value the schedule reacts to.

        Raises:
            ValueError: If `criterion` is missing from *scheduler_kwargs*.
        """
        scheduler_kwargs = dict(scheduler_kwargs)
        if "criterion" not in scheduler_kwargs:
            raise ValueError('"criterion" is required in scheduler_kwargs: it names the criterion the schedule reads.')
        self.criterion: str = scheduler_kwargs.pop("criterion")
        self.optimizer = create_opt(params, **optimizer_kwargs)
        self.scheduler, _ = create_scheduler_v2(self.optimizer, **scheduler_kwargs)

    def __getattr__(self, name: str) -> Any:
        """Delegate the `Optimizer` interface to the wrapped optimizer.

        This keeps generated learner code (`step`, `zero_grad`, `param_groups`) and
        `torch.amp.GradScaler` working on this object unchanged.
        """
        return getattr(self.optimizer, name)

    def on_update(self, info: Any, **models: Any) -> None:
        """Advance a per-update schedule."""
        self.scheduler.step_update(info.update, info.logs().get(self.criterion))

    def on_epoch_end(self, info: Any, **models: Any) -> None:
        """Advance a per-epoch schedule."""
        self.scheduler.step(info.epoch, info.logs().get(self.criterion))

    def state_dict(self) -> dict[str, Any]:
        """Return the optimizer and schedule state, so a resumed run keeps its schedule."""
        return {"optimizer": self.optimizer.state_dict(), "scheduler": self.scheduler.state_dict()}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore the optimizer and schedule state."""
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])


class OptimizerWithNativeScheduler:
    """An optimizer stepping a native `torch.optim.lr_scheduler` schedule, usable anywhere an optimizer is."""

    def __init__(self, params: Any, optimizer_kwargs: dict[str, Any], scheduler_kwargs: dict[str, Any]) -> None:
        """Create the optimizer and its schedule.

        Args:
            params (Any): The named model parameters to optimize.
            optimizer_kwargs (dict[str, Any]): Keyword arguments for `create_opt`, e.g. `opt` and `lr`.
            scheduler_kwargs (dict[str, Any]): Keyword arguments for the scheduler, plus the required
                `name` of the `torch.optim.lr_scheduler` class to build.
        """
        scheduler_kwargs = dict(scheduler_kwargs)
        scheduler_type = getattr(torch.optim.lr_scheduler, scheduler_kwargs.pop("name"))
        self.optimizer = create_opt(params, **optimizer_kwargs)
        self.scheduler = scheduler_type(optimizer=self.optimizer, **scheduler_kwargs)

    def __getattr__(self, name: str) -> Any:
        """Delegate the `Optimizer` interface to the wrapped optimizer.

        This keeps generated learner code (`step`, `zero_grad`, `param_groups`) and
        `torch.amp.GradScaler` working on this object unchanged.
        """
        return getattr(self.optimizer, name)

    def on_epoch_end(self, info: Any, **models: Any) -> None:
        """Advance the schedule, which native schedulers count in epochs."""
        self.scheduler.step()

    def state_dict(self) -> dict[str, Any]:
        """Return the optimizer and schedule state, so a resumed run keeps its schedule."""
        return {"optimizer": self.optimizer.state_dict(), "scheduler": self.scheduler.state_dict()}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore the optimizer and schedule state."""
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])
