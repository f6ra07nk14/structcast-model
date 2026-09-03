"""Collaborators shared by the tests: the Learner boilerplate and an info driven without a trainer.

Every test that runs a loop needs a full `Learner`, and most of its members are not what the test is
about. The two classes here carry that part -- empty collaborators plus the host-owned counters of
`docs/adr/0018` -- so each test's fake is only the behaviour it asserts on.
"""

from dataclasses import dataclass, field
from typing import Any

from structcast_model.base_trainer import BaseInfo


class CountingLearner:
    """A `Learner` owning nothing, whose counters advance the way `docs/adr/0018` has a learner move them.

    Subclasses override the members their test is about: the mappings are properties, and a step
    returning criteria overrides `training_step`/`inference_step`. A subclass whose training step
    does its own work calls `count_step` from it, so the counters the trainer reads stay honest.
    """

    should_update: bool = True
    """Whether a training step lands an Update; False stands in for a step accumulating gradients."""

    def __init__(self) -> None:
        """Start the counters at zero, as a learner that has not stepped yet."""
        self.steps = 0
        self.updates = 0
        self.has_updated = False

    @property
    def models(self) -> dict[str, Any]:
        """No models: a test needing them overrides this."""
        return {}

    @property
    def optimizers(self) -> dict[str, Any]:
        """No optimizers; the trainer scan must handle an empty mapping."""
        return {}

    @property
    def optimizer_models(self) -> dict[str, list[str]]:
        """No pairing, there being no optimizer."""
        return {}

    @property
    def flow_functions(self) -> dict[str, Any]:
        """No separable flows: nothing here compiles or replicates a step."""
        return {}

    @property
    def learning_rates(self) -> dict[str, float]:
        """No rates, there being no optimizer."""
        return {}

    def restore_counters(self, steps: int, updates: int) -> None:
        """Seed the counters, the way a resume path would."""
        self.steps = steps
        self.updates = updates

    def count_step(self) -> None:
        """Count the Step that just ran and report whether it landed an Update."""
        self.steps += 1
        self.has_updated = self.should_update
        if self.has_updated:
            self.updates += 1

    def training_step(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """Count one Step and report no criteria."""
        self.count_step()
        return {}

    def inference_step(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """Report no criteria, and touch no counter: inference is not a training Step."""
        return {}


@dataclass(kw_only=True)
class SteppedInfo(BaseInfo[Any]):
    """Info carrying models and a drivable step count, for callbacks tested outside a training loop.

    `BaseInfo.models` and `BaseInfo.step` are read-only views of the trainer's learner
    (`docs/adr/0018`), so a test running a callback without a trainer drives them from here.
    """

    named_models: dict[str, Any] = field(default_factory=dict)
    """The models the `models` property hands out."""

    current_step: int = 0
    """The step count the `step` property reports."""

    @property
    def models(self) -> dict[str, Any]:
        """Return the models this info was built with."""
        return self.named_models

    @property
    def step(self) -> int:
        """Report the driven step, standing in for a trainer's learner-backed count."""
        return self.current_step
