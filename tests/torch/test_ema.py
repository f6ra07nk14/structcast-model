"""Runtime tests for the `EMA` shadow models a torch learner declares (`docs/adr/0021`).

The learner is generated from the linear fixture and driven with a real `torch.nn.Module`: what an
average is worth depends on when it is blended, by how much, and whether it survives a checkpoint,
and none of that is decided until the emitted code runs.
"""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from structcast.core.exceptions import SpecError

from structcast_model.builders.torch import TorchBuilder, TorchLearnerBuilder
from structcast_model.torch.distributed import SingleDeviceStrategy
from structcast_model.torch.trainer import initial_model
from structcast_model.utils.base import load_any
from tests import FIXTURES_DIR
import torch

CFG_DIR = FIXTURES_DIR / "cfg" / "torch"
MODEL_YAML = CFG_DIR / "Linear.yaml"
LEARNER_YAML = CFG_DIR / "LinearLearner.yaml"

BATCH = {
    "x": torch.tensor([[1.0, 0.5, -0.5, 2.0], [0.0, 1.0, 1.0, -1.0]]),
    "y": torch.tensor([[1.0, -1.0], [0.5, 0.25]]),
}
"""One fixed batch, so a weight that moves can only be the optimizer's doing."""

AVERAGED_INFERENCE = [
    ["x", "prediction", "ema_model"],
    [{"input": "prediction", "target": "y"}, "loss", "mse"],
]
"""An inference flow validating over the average instead of over the trained weights."""


def _load(path: Path, name: str) -> ModuleType:
    """Load a generated module by file path, the way a configuration does: not by import name."""
    spec = spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _model(tmp_path: Path) -> Any:
    """Build the seeded fixture model, materializing its lazy layer before anything reads it."""
    TorchBuilder.from_path(MODEL_YAML)()(tmp_path / "model.py")
    torch.manual_seed(0)
    model = _load(tmp_path / "model.py", "generated_model").Model()
    initial_model(model, {"x": (4,)})
    return model


def _ema_raw(**overrides: Any) -> dict[str, Any]:
    """Load the linear learner fixture with an EMA over its single model."""
    return {**load_any(LEARNER_YAML), "EMA": {"model": True}, **overrides}


def _learner(tmp_path: Path, raw: dict[str, Any], name: str = "learner", **kwargs: Any) -> Any:
    """Generate a learner from a mutated fixture configuration and build it on a fresh model."""
    TorchLearnerBuilder(raw=raw, current_path=str(LEARNER_YAML))(**kwargs)(tmp_path / f"{name}.py")
    return _load(tmp_path / f"{name}.py", f"generated_{name}").Learner(_model(tmp_path))


def _weight(learner: Any, name: str = "ema_model") -> torch.Tensor:
    """Read the first weight of one of the learner's models, detached from the graph."""
    module = learner.models[name]
    return getattr(module, "module", module).fc.weight.detach().clone()


@pytest.mark.parametrize(
    ("window", "gates"),
    [(None, [True, True, True, True]), (2, [True, False, True, False])],
    ids=["every-step", "window-of-two"],
)
def test_the_average_moves_on_update_steps_and_on_no_other(
    tmp_path: Path, window: int | None, gates: list[bool]
) -> None:
    """The blend follows Updates, not steps: an accumulation micro-step must leave the average alone.

    An average blended on every call would advance `1/k` as fast as its decay says on an accumulating
    run -- and against gradients no optimizer has applied yet. That is invisible in the criteria and
    only surfaces as an evaluation curve that never catches up with the training one. The cadence
    itself is the learner's own gate, which keeps the historically short first torch window.
    """
    learner = _learner(tmp_path, _ema_raw(), "cadence", parameters={"DEFAULT": {"accumulate_gradients": window}})
    previous = _weight(learner)

    moved, reported = [], []
    for _ in gates:
        learner.training_step(**BATCH)
        current = _weight(learner)
        moved.append(not torch.equal(previous, current))
        reported.append(learner.has_updated)
        previous = current

    assert reported == gates
    assert moved == gates


def test_one_update_blends_the_average_by_its_decay(tmp_path: Path) -> None:
    """The rule is exact arithmetic, so it is asserted exactly rather than within a tolerance.

    The first Update seeds the average from the current weights -- torch copies rather than blending
    while nothing has been averaged yet -- and every Update after it is
    `decay * average + (1 - decay) * weights`, which is what `lerp` computes.
    """
    learner = _learner(tmp_path, _ema_raw(), "math")

    learner.training_step(**BATCH)
    seeded = _weight(learner)
    assert torch.equal(seeded, _weight(learner, "model"))

    learner.training_step(**BATCH)

    assert torch.equal(_weight(learner), torch.lerp(seeded, _weight(learner, "model"), 1 - 0.999))
    assert not torch.equal(_weight(learner), _weight(learner, "model"))


def test_the_inference_flow_runs_the_average_and_not_the_trained_weights(tmp_path: Path) -> None:
    """Validating over the average is the point of keeping one, so the criteria have to come from it.

    The two are far apart here -- two Updates against a decay of 0.999 -- so a flow that quietly ran
    the trained model instead would report the training loss, which is what the second half pins.
    """
    raw = _ema_raw()
    raw["LEARNERS"][0]["INFERENCE_FLOW"] = AVERAGED_INFERENCE
    learner = _learner(tmp_path, raw, "inference")
    plain = _learner(tmp_path, load_any(LEARNER_YAML), "plain")
    for _ in range(2):
        learner.training_step(**BATCH)
        plain.training_step(**BATCH)

    averaged = learner.inference_step(**BATCH)["loss"]

    assert torch.isfinite(averaged)
    assert float(averaged) != pytest.approx(float(plain.inference_step(**BATCH)["loss"]))


def test_a_training_flow_that_reads_the_average_is_rejected() -> None:
    """An average is a copy no optimizer owns: differentiating it computes gradients for nothing."""
    raw = _ema_raw()
    raw["LEARNERS"][0]["FLOW"][0][2] = "ema_model"

    with pytest.raises(SpecError, match='The training FLOW reads "ema_model"'):
        # `scripts` is a cached property: binding it is what runs the check being tested here.
        _ = TorchLearnerBuilder(raw=raw, current_path=str(LEARNER_YAML))().scripts


def test_the_models_property_carries_the_average_through_a_state_round_trip(tmp_path: Path) -> None:
    """The `models` property is the whole persistence path: a resume restores what it names.

    The averaged weights and the count of blends behind them both have to survive, or a resumed run
    would restart the average from the weights it happens to hold and blend from an `n_averaged` of
    zero -- copying the next Update's weights over the average it was resumed with.
    """
    learner = _learner(tmp_path, _ema_raw(), "state")
    for _ in range(3):
        learner.training_step(**BATCH)
    strategy = SingleDeviceStrategy(device="cpu")
    saved = strategy.state_dict(dict(learner.models))["models"]
    assert set(saved) == {"model", "ema_model"}

    restored = _learner(tmp_path, _ema_raw(), "state")
    strategy.load_state_dict(dict(restored.models), {}, None, {"models": saved})

    assert torch.equal(_weight(restored), _weight(learner))
    assert int(restored.models["ema_model"].n_averaged) == int(learner.models["ema_model"].n_averaged) == 3


def test_an_ema_key_that_names_no_model_is_rejected() -> None:
    """A typo in the key would otherwise average nothing at all, silently."""
    raw = {**load_any(LEARNER_YAML), "EMA": {"modle": True}}

    with pytest.raises(SpecError, match='EMA names "modle"'):
        TorchLearnerBuilder(raw=raw, current_path=str(LEARNER_YAML))()


def test_a_mapping_carries_the_averaging_keywords_into_the_run(tmp_path: Path) -> None:
    """The keywords are DSL values, so a `multi_avg_fn` object pattern has to reach the constructor.

    A decay of one half makes the blend visible in a single Update, which no default could produce:
    the assertion fails if the pattern was dropped, resolved to something else, or never called.
    """
    fn = {"_obj_": [{"_addr_": "torch.optim.swa_utils.get_ema_multi_avg_fn"}, {"_call_": {"decay": 0.5}}]}
    learner = _learner(tmp_path, _ema_raw(EMA={"model": {"multi_avg_fn": fn}}), "keywords")

    learner.training_step(**BATCH)
    seeded = _weight(learner)
    learner.training_step(**BATCH)

    assert torch.equal(_weight(learner), torch.lerp(seeded, _weight(learner, "model"), 0.5))


def test_a_name_the_average_needs_and_the_learner_already_uses_is_rejected() -> None:
    """`ema_<model>` is an attribute of the learner and a name its flows resolve: it must be free."""
    raw = _ema_raw()
    raw["LEARNERS"][0]["FLOW"][1]["INPUTS"]["target"] = "ema_model"
    raw["LEARNERS"][0]["INFERENCE_FLOW"][1][0]["target"] = "ema_model"

    with pytest.raises(SpecError, match='The EMA of "model" is emitted as "ema_model"'):
        TorchLearnerBuilder(raw=raw, current_path=str(LEARNER_YAML))()


class DTensor(torch.nn.Parameter):
    """Stands in for a sharded FSDP2 parameter, which the generated guard recognizes by type name.

    Naming is all the guard reads, deliberately: the generated learner cannot import
    `torch.distributed.tensor` to run an `isinstance` check that would only ever matter in a run that
    has already sharded, and a real `DTensor` needs a device mesh this test has no process group for.
    """


def test_a_model_reaching_the_learner_sharded_is_refused_rather_than_averaged(tmp_path: Path) -> None:
    """An `AveragedModel` copies the module it averages, and a DTensor-parameter module has no copy.

    The models arrive already wrapped, so the learner is the first place this is knowable; refusing
    it here is what keeps a half-copied shard out of a checkpoint nobody can tell apart from a good
    one.
    """
    TorchLearnerBuilder(raw=_ema_raw(), current_path=str(LEARNER_YAML))()(tmp_path / "sharded.py")
    model = _model(tmp_path)
    model.fc.weight = DTensor(model.fc.weight.detach())

    with pytest.raises(ValueError, match="EMA works with neither FSDP2 nor tensor parallel"):
        _load(tmp_path / "sharded.py", "sharded_learner").Learner(model)


def test_a_mapping_keeps_the_averaging_defaults_it_does_not_mention(tmp_path: Path) -> None:
    """A mapping declares keywords, not a different mechanism.

    `AveragedModel` averages every Update equally without a `multi_avg_fn`, so a mapping that only
    sets `use_buffers` would silently be SWA -- the second Update would blend by one half instead of
    by the declared decay, which is what this pins.
    """
    learner = _learner(tmp_path, _ema_raw(EMA={"model": {"use_buffers": True}}), "defaults")

    learner.training_step(**BATCH)
    seeded = _weight(learner)
    learner.training_step(**BATCH)

    assert torch.equal(_weight(learner), torch.lerp(seeded, _weight(learner, "model"), 1 - 0.999))


def test_the_average_is_taken_over_the_module_a_ddp_wrapper_holds(tmp_path: Path, single_process_gloo: None) -> None:
    """The models reach the learner already wrapped, and a wrapper is neither copyable nor weights.

    Averaging the wrapper would deep-copy a process group, and the copy's state-dict keys would carry
    a second `module.` that no unwrapped run could load. The unwrap is by wrapper type, so a model
    owning a submodule of its own called `module` keeps its whole self averaged.
    """
    TorchLearnerBuilder(raw=_ema_raw(), current_path=str(LEARNER_YAML))()(tmp_path / "wrapped.py")
    model = _model(tmp_path)
    wrapped = torch.nn.parallel.DistributedDataParallel(model)
    learner = _load(tmp_path / "wrapped.py", "wrapped_learner").Learner(wrapped)
    average = learner.models["ema_model"]

    learner.training_step(**BATCH)

    assert isinstance(average.module, type(model))
    assert set(average.state_dict()) == {"n_averaged", "module.fc.weight", "module.fc.bias"}
    # The seeding Update copied the trained weights, so the blend read the wrapped module, not the wrapper.
    assert torch.equal(average.module.fc.weight, model.fc.weight)


def test_resuming_a_checkpoint_written_before_the_average_was_declared_says_so(tmp_path: Path) -> None:
    """Adding an `EMA` to a learner outdates every checkpoint of it, and the resume has to say which.

    Torch reports a model handed an empty state as a process-group failure, and a wrapped one accepts
    it without a word and keeps its construction weights with nothing averaged -- so the run would
    either die pointing at the wrong thing or quietly validate against noise for the rest of its life.
    """
    plain = _learner(tmp_path, load_any(LEARNER_YAML), "plain")
    strategy = SingleDeviceStrategy(device="cpu")
    saved = strategy.state_dict(dict(plain.models))
    averaged = _learner(tmp_path, _ema_raw(), "averaged")

    with pytest.raises(ValueError, match='carries no state for model "ema_model"'):
        strategy.load_state_dict(dict(averaged.models), {}, None, saved)


def test_a_checkpoint_carrying_an_average_the_learner_dropped_still_resumes(tmp_path: Path) -> None:
    """The other direction is not a failure: state for a model the learner no longer has is ignored.

    Dropping an `EMA` declaration leaves the weights, the optimizer and the counters resumable, and
    nothing in the run depends on the average that was.
    """
    averaged = _learner(tmp_path, _ema_raw(), "averaged")
    for _ in range(2):
        averaged.training_step(**BATCH)
    strategy = SingleDeviceStrategy(device="cpu")
    saved = strategy.state_dict(dict(averaged.models))
    plain = _learner(tmp_path, load_any(LEARNER_YAML), "plain")

    strategy.load_state_dict(dict(plain.models), {}, None, saved)

    assert set(plain.models) == {"model"}
    assert torch.equal(_weight(plain, "model"), _weight(averaged, "model"))


def test_the_old_torch_fallback_saves_and_loads_the_average_symmetrically(tmp_path: Path) -> None:
    """On a torch without the state-dict API the two halves must agree on the keys they name.

    The fallback used to strip every `module.` prefix on the way out, which is a wrapper's on a
    wrapped model and the average's own on an `AveragedModel`: the keys came back one level short and
    the load failed on all of them.
    """
    learner = _learner(tmp_path, _ema_raw(), "fallback")
    for _ in range(2):
        learner.training_step(**BATCH)
    strategy = SingleDeviceStrategy(device="cpu")
    # The state-dict API is resolved into a field at construction, so an old torch is one assignment
    # on this instance rather than a patch of the module global the strategy used to read.
    strategy._api = None
    saved = strategy.state_dict(dict(learner.models))
    assert set(saved["models"]["ema_model"]) == {"n_averaged", "module.fc.weight", "module.fc.bias"}

    restored = _learner(tmp_path, _ema_raw(), "fallback")
    strategy.load_state_dict(dict(restored.models), {}, None, saved)

    assert torch.equal(_weight(restored), _weight(learner))
    assert int(restored.models["ema_model"].n_averaged) == 2


def test_a_model_owning_a_submodule_called_module_is_averaged_whole(tmp_path: Path) -> None:
    """The unwrap reads the wrapper type, never the attribute name.

    A model whose own sublayer is called `module` looks exactly like a wrapped one to a name-based
    unwrap, which would average that sublayer alone -- a fragment, silently, with the rest of the
    weights never following training at all.
    """
    raw = load_any(MODEL_YAML)
    raw["FLOW"][0][2] = "module"
    TorchBuilder(raw=raw)()(tmp_path / "named.py")
    torch.manual_seed(0)
    model = _load(tmp_path / "named.py", "named_model").Model()
    initial_model(model, {"x": (4,)})
    TorchLearnerBuilder(raw=_ema_raw(), current_path=str(LEARNER_YAML))()(tmp_path / "whole.py")
    learner = _load(tmp_path / "whole.py", "whole_learner").Learner(model)

    learner.training_step(**BATCH)

    assert set(learner.models["ema_model"].state_dict()) == {"n_averaged", "module.module.weight", "module.module.bias"}
