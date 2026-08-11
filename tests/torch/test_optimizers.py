"""Unit tests for structcast_model.torch.optimizers."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import pytest
from torch.nn import Linear

from structcast_model.base_trainer import BaseInfo
from structcast_model.torch.optimizers import create_opt, set_lr_scale
import torch

# Access private helpers via the exported function's __globals__
# (the module uses LazySelectedImporter, blocking direct private imports)
_g = create_opt.__globals__
_match_no_weight_decay = _g["_match_no_weight_decay"]
_get_layer_group_id = _g["_get_layer_group_id"]
_param_groups_layer_decay = _g["_param_groups_layer_decay"]
_param_groups_weight_decay = _g["_param_groups_weight_decay"]
re_compile = _g["re_compile"]

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _named_params() -> list[tuple[str, torch.nn.Parameter]]:
    """Return a minimal set of named parameters from a Linear layer."""
    return list(Linear(4, 2).named_parameters())


# ---------------------------------------------------------------------------
# _match_no_weight_decay
# ---------------------------------------------------------------------------


def test_match_no_weight_decay_1d_param_is_no_decay_by_default() -> None:
    """1-D parameters (e.g., bias) are no-decay when no overriding regexes are set."""
    bias = torch.nn.Parameter(torch.zeros(4))
    assert _match_no_weight_decay("bias", bias, [], []) is True


def test_match_no_weight_decay_2d_param_is_decay_by_default() -> None:
    """2-D parameters (e.g., weight matrix) decay when no overriding regexes are set."""
    weight = torch.nn.Parameter(torch.zeros(4, 4))
    assert _match_no_weight_decay("weight", weight, [], []) is False


def test_match_no_weight_decay_wd_regex_overrides_1d() -> None:
    """A weight_decay regex forces decay even on a 1-D parameter."""
    bias = torch.nn.Parameter(torch.zeros(4))
    wd = [re_compile(r"bias")]
    assert _match_no_weight_decay("bias", bias, wd, []) is False


def test_match_no_weight_decay_nwd_regex_forces_no_decay_on_2d() -> None:
    """A no_weight_decay regex forces no-decay even on a 2-D parameter."""
    weight = torch.nn.Parameter(torch.zeros(4, 4))
    nwd = [re_compile(r"emb.*")]
    assert _match_no_weight_decay("emb.weight", weight, [], nwd) is True


def test_match_no_weight_decay_wd_takes_precedence_over_nwd() -> None:
    """weight_decay regex takes precedence over no_weight_decay regex."""
    param = torch.nn.Parameter(torch.zeros(4))
    wd = [re_compile(r"emb.*")]
    nwd = [re_compile(r"emb.*")]
    assert _match_no_weight_decay("emb.weight", param, wd, nwd) is False


# ---------------------------------------------------------------------------
# _get_layer_group_id
# ---------------------------------------------------------------------------


def test_get_layer_group_id_returns_first_matching_index() -> None:
    """Returns the index of the first pattern that matches the name."""
    patterns = [re_compile(r"layer1.*"), re_compile(r"layer2.*")]
    assert _get_layer_group_id("layer1.weight", patterns) == 0
    assert _get_layer_group_id("layer2.bias", patterns) == 1


def test_get_layer_group_id_returns_minus_one_on_no_match() -> None:
    """Returns -1 when no pattern matches."""
    patterns = [re_compile(r"layer1.*")]
    assert _get_layer_group_id("head.weight", patterns) == -1


def test_get_layer_group_id_empty_patterns_always_returns_minus_one() -> None:
    """With an empty pattern list, always returns -1."""
    assert _get_layer_group_id("anything.weight", []) == -1


# ---------------------------------------------------------------------------
# _param_groups_layer_decay
# ---------------------------------------------------------------------------


def test_param_groups_layer_decay_produces_groups_with_lr_scale() -> None:
    """All groups produced by layer decay have an lr_scale key."""
    params = _named_params()
    groups = _param_groups_layer_decay(
        params,
        layer_decay=0.8,
        layer_group_regexes=[],
        weight_decay=0.01,
        weight_decay_regexes=[],
        no_weight_decay_regexes=[],
    )
    assert len(groups) > 0
    assert all("lr_scale" in g for g in groups)


def test_param_groups_layer_decay_excludes_frozen_params() -> None:
    """Frozen parameters (requires_grad=False) are excluded."""
    model = Linear(4, 2)
    model.weight.requires_grad_(False)
    params = list(model.named_parameters())
    groups = _param_groups_layer_decay(
        params,
        layer_decay=0.9,
        layer_group_regexes=[],
        weight_decay=0.0,
        weight_decay_regexes=[],
        no_weight_decay_regexes=[],
    )
    all_params = [p for g in groups for p in g["params"]]
    assert not any(p is model.weight for p in all_params)
    assert any(p is model.bias for p in all_params)


def test_param_groups_layer_decay_with_named_groups() -> None:
    """Parameters are bucketed into the correct named group by layer regex."""
    model = Linear(4, 2)
    params = list(model.named_parameters())
    groups = _param_groups_layer_decay(
        params,
        layer_decay=0.8,
        layer_group_regexes=[re_compile(r"weight")],
        weight_decay=0.01,
        weight_decay_regexes=[],
        no_weight_decay_regexes=[],
    )
    group_names = [g.get("param_names", []) for g in groups]
    # "weight" should land in group 0, not in the default -1 group
    has_weight_group = any("weight" in names for names in group_names)
    assert has_weight_group


# ---------------------------------------------------------------------------
# _param_groups_weight_decay
# ---------------------------------------------------------------------------


def test_param_groups_weight_decay_always_returns_two_groups() -> None:
    """Returns exactly two groups: decay and no-decay."""
    params = _named_params()
    groups = _param_groups_weight_decay(
        params,
        weight_decay=0.01,
        weight_decay_regexes=[],
        no_weight_decay_regexes=[],
    )
    assert len(groups) == 2


def test_param_groups_weight_decay_1d_params_go_to_no_decay() -> None:
    """1-D parameters (bias) land in the no-decay group."""
    model = Linear(4, 2)
    params = list(model.named_parameters())
    groups = _param_groups_weight_decay(
        params,
        weight_decay=0.01,
        weight_decay_regexes=[],
        no_weight_decay_regexes=[],
    )
    no_decay = next(g for g in groups if g["weight_decay"] == 0.0)
    assert any(p is model.bias for p in no_decay["params"])


def test_param_groups_weight_decay_excludes_frozen_params() -> None:
    """Frozen parameters are excluded from both groups."""
    model = Linear(4, 2)
    model.weight.requires_grad_(False)
    params = list(model.named_parameters())
    groups = _param_groups_weight_decay(
        params,
        weight_decay=0.01,
        weight_decay_regexes=[],
        no_weight_decay_regexes=[],
    )
    all_params = [p for g in groups for p in g["params"]]
    assert not any(p is model.weight for p in all_params)


# ---------------------------------------------------------------------------
# set_lr_scale
# ---------------------------------------------------------------------------


def test_set_lr_scale_multiplies_lr_by_lr_scale() -> None:
    """Multiplies the learning rate of each group that has lr_scale."""
    model = Linear(4, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
    optimizer.param_groups[0]["lr_scale"] = 0.5
    set_lr_scale(optimizer)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.5)


def test_set_lr_scale_removes_key_when_delete_true() -> None:
    """lr_scale key is deleted when delete_lr_scale=True."""
    model = Linear(4, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
    optimizer.param_groups[0]["lr_scale"] = 0.5
    set_lr_scale(optimizer, delete_lr_scale=True)
    assert "lr_scale" not in optimizer.param_groups[0]


def test_set_lr_scale_skips_groups_without_lr_scale() -> None:
    """Groups without lr_scale are left unchanged."""
    model = Linear(4, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    original_lr = optimizer.param_groups[0]["lr"]
    set_lr_scale(optimizer)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(original_lr)


def test_set_lr_scale_with_tensor_lr() -> None:
    """A tensor learning rate is scaled in place, since optimizers may hold lr as a tensor."""
    optimizer = torch.optim.SGD(Linear(4, 2).parameters(), lr=1.0)
    optimizer.param_groups[0]["lr"] = torch.tensor(1.0)
    optimizer.param_groups[0]["lr_scale"] = 0.5
    set_lr_scale(optimizer)
    assert optimizer.param_groups[0]["lr"].item() == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# create_opt - engine selection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("opt", ["torch.optim.AdamW", "torch.AdamW", "torch.optim.SGD"])
def test_create_opt_builds_a_native_optimizer_by_explicit_name(opt: str) -> None:
    """Only an explicit torch.optim.X name selects the native engine: bare names keep timm's defaults."""
    optimizer = create_opt(_named_params(), opt=opt, lr=0.01)
    assert type(optimizer) is getattr(torch.optim, opt.rsplit(".", 1)[-1])


def test_create_opt_sends_bare_names_to_timm() -> None:
    """Bare names like "sgd" must keep timm's defaults (nesterov momentum), or migrated cfgs change behavior."""
    optimizer = create_opt(_named_params(), opt="sgd", lr=0.01, momentum=0.9)
    assert optimizer.param_groups[0]["nesterov"] is True


def test_create_opt_rejects_a_bad_explicit_native_name() -> None:
    """An explicit torch.optim.X name that is not an optimizer must fail loudly, not fall back to timm."""
    with pytest.raises(ValueError, match="does not name a torch.optim optimizer class"):
        create_opt(_named_params(), opt="torch.optim.NoSuchOptimizer", lr=0.01)


def test_create_opt_builds_the_given_optimizer_class() -> None:
    """A callable is used as the engine directly, so any optimizer class works without a name lookup."""
    optimizer = create_opt(_named_params(), opt=torch.optim.Adagrad, lr=0.01)
    assert type(optimizer) is torch.optim.Adagrad


def test_create_opt_falls_back_to_timm_for_unknown_names() -> None:
    """Names torch does not know reach timm, which is what makes its extra optimizers available."""
    optimizer = create_opt(_named_params(), opt="lamb", lr=0.01)
    assert type(optimizer).__name__ == "Lamb"


def test_create_opt_rejects_unknown_optimizer_names() -> None:
    """An unknown name must fail loudly rather than silently fall back to a default optimizer."""
    with pytest.raises(ValueError, match="not found in registry"):
        create_opt(_named_params(), opt="definitely-not-an-optimizer", lr=0.01)


# ---------------------------------------------------------------------------
# create_opt - parameter grouping
# ---------------------------------------------------------------------------


def test_create_opt_groups_by_weight_decay_regexes() -> None:
    """Regex grouping is what lets a run decay weights but not biases, whatever the engine is."""
    optimizer = create_opt(_named_params(), opt="sgd", lr=0.01, weight_decay=0.05, no_weight_decay_regexes=["bias"])
    decays = {tuple(group["param_names"]): group["weight_decay"] for group in optimizer.param_groups}
    assert decays == {("bias",): 0.0, ("weight",): 0.05}


def test_create_opt_passes_weight_decay_through_when_no_group_consumed_it() -> None:
    """Without grouping regexes the weight decay must still reach the engine."""
    optimizer = create_opt(_named_params(), opt="sgd", lr=0.01, weight_decay=0.05)
    assert all(group["weight_decay"] == pytest.approx(0.05) for group in optimizer.param_groups)


def test_create_opt_bakes_layer_scales_into_lr_on_the_native_engine() -> None:
    """Native schedulers ignore lr_scale, so the scale has to be applied to lr before training."""
    optimizer = create_opt(
        _named_params(),
        opt="torch.optim.SGD",
        lr=1.0,
        layer_decay=0.5,
        layer_group_regexes=["weight"],
        weight_decay=0.05,
    )
    assert not any("lr_scale" in group for group in optimizer.param_groups)
    assert sorted(group["lr"] for group in optimizer.param_groups) == [pytest.approx(0.5), pytest.approx(1.0)]


def test_create_opt_keeps_lr_scale_on_the_timm_engine() -> None:
    """Timm schedulers read lr_scale themselves, so removing it would double-apply the decay."""
    optimizer = create_opt(
        _named_params(), opt="lamb", lr=1.0, layer_decay=0.5, layer_group_regexes=["weight"], weight_decay=0.05
    )
    assert sorted(group["lr_scale"] for group in optimizer.param_groups) == [pytest.approx(0.5), pytest.approx(1.0)]
    assert all(group["lr"] == pytest.approx(1.0) for group in optimizer.param_groups)


# ---------------------------------------------------------------------------
# examples/torch/optimizers.py - AdamWWithCosine
# ---------------------------------------------------------------------------


def _load_example_module() -> Any:
    """Load the example module the way a configuration does: by file path, not by import name."""
    path = Path(__file__).resolve().parents[2] / "examples" / "torch" / "optimizers.py"
    spec = importlib.util.spec_from_file_location("example_optimizers", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_example_adamw_with_cosine_requires_a_criterion() -> None:
    """The schedule reads a tracked criterion; leaving it out is a configuration error, not a default."""
    module = _load_example_module()
    with pytest.raises(ValueError, match="criterion"):
        module.AdamWWithCosine(
            _named_params(), optimizer_kwargs={"opt": "adamw", "lr": 0.1}, scheduler_kwargs={"num_epochs": 2}
        )


def test_example_adamw_with_cosine_proxies_the_optimizer_interface() -> None:
    """Generated learner code calls step/zero_grad/param_groups on it, so the proxy must be transparent."""
    module = _load_example_module()
    model = Linear(4, 2)
    optimizer = module.AdamWWithCosine(
        list(model.named_parameters()),
        optimizer_kwargs={"opt": "adamw", "lr": 0.1},
        scheduler_kwargs={"sched": "cosine", "num_epochs": 2, "criterion": "loss"},
    )
    model(torch.ones(1, 4)).sum().backward()
    optimizer.step()
    optimizer.zero_grad()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.1)


def test_example_adamw_with_cosine_steps_the_schedule_on_epoch_end() -> None:
    """Routed as a callback, the wrapper is what advances the schedule -- nothing else does."""
    module = _load_example_module()
    optimizer = module.AdamWWithCosine(
        _named_params(),
        optimizer_kwargs={"opt": "adamw", "lr": 0.1},
        scheduler_kwargs={"sched": "cosine", "num_epochs": 4, "min_lr": 0.0, "criterion": "loss"},
    )
    info = BaseInfo(epoch=3)
    info.logs()["loss"] = 0.5
    optimizer.on_epoch_end(info)
    assert optimizer.param_groups[0]["lr"] < 0.1


def test_example_adamw_with_cosine_state_dict_covers_both_halves() -> None:
    """Schedule state was lost on resume before; the merged state dict is the fix."""
    module = _load_example_module()
    optimizer = module.AdamWWithCosine(
        _named_params(),
        optimizer_kwargs={"opt": "adamw", "lr": 0.1},
        scheduler_kwargs={"sched": "cosine", "num_epochs": 2, "criterion": "loss"},
    )
    state = optimizer.state_dict()
    assert set(state) == {"optimizer", "scheduler"}
    optimizer.load_state_dict(state)


# ---------------------------------------------------------------------------
# examples/torch/optimizers.py - OptimizerWithNativeScheduler
# ---------------------------------------------------------------------------


def test_example_native_scheduler_steps_the_schedule_on_epoch_end() -> None:
    """A native scheduler counts epochs on its own, so the wrapper only has to step it once per epoch."""
    module = _load_example_module()
    optimizer = module.OptimizerWithNativeScheduler(
        _named_params(),
        optimizer_kwargs={"opt": "adam", "lr": 0.1},
        scheduler_kwargs={"name": "LambdaLR", "lr_lambda": lambda epoch: 1.0 - 0.5 * epoch},
    )
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.1)
    optimizer.on_epoch_end(BaseInfo(epoch=1))
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.05)


def test_example_native_scheduler_state_dict_covers_both_halves() -> None:
    """Schedule state must survive a resume, exactly as for the timm-scheduled combination."""
    module = _load_example_module()
    optimizer = module.OptimizerWithNativeScheduler(
        _named_params(),
        optimizer_kwargs={"opt": "adam", "lr": 0.1},
        scheduler_kwargs={"name": "LambdaLR", "lr_lambda": lambda epoch: 1.0},
    )
    state = optimizer.state_dict()
    assert set(state) == {"optimizer", "scheduler"}
    optimizer.load_state_dict(state)
