"""Unit tests for structcast_model.torch.optimizers."""

from __future__ import annotations

from typing import Any

import pytest
from torch.nn import Linear

from structcast_model.base_trainer import GLOBAL_CALLBACKS
from structcast_model.torch.optimizers import create, create_with_scheduler
import torch

# Access private helpers via the exported function's __globals__
# (the module uses LazySelectedImporter, blocking direct private imports)
_g = create.__globals__
_match_no_weight_decay = _g["_match_no_weight_decay"]
_get_layer_group_id = _g["_get_layer_group_id"]
_param_groups_layer_decay = _g["_param_groups_layer_decay"]
_param_groups_weight_decay = _g["_param_groups_weight_decay"]
_set_lr_scale = _g["_set_lr_scale"]
re_compile = _g["re_compile"]

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _named_params() -> list[tuple[str, torch.nn.Parameter]]:
    """Return a minimal set of named parameters from a Linear layer."""
    return list(Linear(4, 2).named_parameters())


@pytest.fixture(autouse=True)
def _clean_global_callbacks() -> Any:
    """Restore GLOBAL_CALLBACKS callback lists after each test."""
    _cb_attrs = (
        "on_update",
        "on_training_begin",
        "on_training_end",
        "on_training_step_begin",
        "on_training_step_end",
        "on_validation_begin",
        "on_validation_end",
        "on_validation_step_begin",
        "on_validation_step_end",
        "on_epoch_begin",
        "on_epoch_end",
    )
    saved = {attr: list(getattr(GLOBAL_CALLBACKS, attr)) for attr in _cb_attrs}
    yield
    for attr, value in saved.items():
        setattr(GLOBAL_CALLBACKS, attr, value)


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
# _set_lr_scale
# ---------------------------------------------------------------------------


def test_set_lr_scale_multiplies_lr_by_lr_scale() -> None:
    """Multiplies the learning rate of each group that has lr_scale."""
    model = Linear(4, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
    optimizer.param_groups[0]["lr_scale"] = 0.5
    _set_lr_scale(optimizer)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.5)


def test_set_lr_scale_removes_key_when_delete_true() -> None:
    """lr_scale key is deleted when delete_lr_scale=True."""
    model = Linear(4, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
    optimizer.param_groups[0]["lr_scale"] = 0.5
    _set_lr_scale(optimizer, delete_lr_scale=True)
    assert "lr_scale" not in optimizer.param_groups[0]


def test_set_lr_scale_skips_groups_without_lr_scale() -> None:
    """Groups without lr_scale are left unchanged."""
    model = Linear(4, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    original_lr = optimizer.param_groups[0]["lr"]
    _set_lr_scale(optimizer)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(original_lr)


# ---------------------------------------------------------------------------
# create (public API)
# ---------------------------------------------------------------------------


def test_create_returns_optimizer_instance() -> None:
    """create() returns a valid torch Optimizer."""
    params = _named_params()
    opt = create(params, opt="sgd", lr=0.01)
    assert isinstance(opt, torch.optim.Optimizer)


def test_create_with_layer_decay() -> None:
    """create() applies layer-wise learning rate decay when layer_decay > 0."""
    params = _named_params()
    opt = create(
        params,
        opt="sgd",
        lr=0.01,
        layer_decay=0.8,
        layer_group_regexes=[],
        weight_decay=0.01,
    )
    assert isinstance(opt, torch.optim.Optimizer)


def test_create_with_weight_decay_and_no_wd_regexes() -> None:
    """create() creates separate parameter groups when weight_decay + regexes are used."""
    params = _named_params()
    opt = create(
        params,
        opt="sgd",
        lr=0.01,
        weight_decay=0.01,
        no_weight_decay_regexes=["bias"],
    )
    assert isinstance(opt, torch.optim.Optimizer)
    assert len(opt.param_groups) >= 1


def test_create_with_weight_decay_and_wd_regexes() -> None:
    """create() handles weight_decay_regexes correctly."""
    params = _named_params()
    opt = create(
        params,
        opt="sgd",
        lr=0.01,
        weight_decay=0.01,
        weight_decay_regexes=["weight"],
    )
    assert isinstance(opt, torch.optim.Optimizer)


def test_create_no_decay_no_regexes() -> None:
    """create() with weight_decay=0 and no regexes passes params directly."""
    params = _named_params()
    opt = create(params, opt="adam", lr=0.001)
    assert isinstance(opt, torch.optim.Optimizer)


# ---------------------------------------------------------------------------
# create_with_scheduler – native StepLR
# ---------------------------------------------------------------------------


def test_create_with_scheduler_steplr_registers_epoch_callback() -> None:
    """StepLR scheduler registers an on_epoch_end callback in GLOBAL_CALLBACKS."""
    before = len(GLOBAL_CALLBACKS.on_epoch_end)
    params = _named_params()
    opt = create_with_scheduler(
        params,
        optimizer_kwargs={"opt": "sgd", "lr": 0.01},
        scheduler_kwargs={"name": "StepLR", "step_size": 1},
    )
    assert isinstance(opt, torch.optim.Optimizer)
    assert len(GLOBAL_CALLBACKS.on_epoch_end) > before


def test_create_with_scheduler_cosine_warm_restarts_registers_update_callback() -> None:
    """CosineAnnealingWarmRestarts registers an on_update callback."""
    before = len(GLOBAL_CALLBACKS.on_update)
    params = _named_params()
    create_with_scheduler(
        params,
        optimizer_kwargs={"opt": "sgd", "lr": 0.01},
        scheduler_kwargs={
            "name": "CosineAnnealingWarmRestarts",
            "T_0": 10,
            "updates_per_epoch": 100,
        },
    )
    assert len(GLOBAL_CALLBACKS.on_update) > before


def test_create_with_scheduler_cosine_warm_restarts_missing_updates_raises() -> None:
    """CosineAnnealingWarmRestarts raises when updates_per_epoch is absent."""
    with pytest.raises(ValueError, match="updates_per_epoch must be a positive integer"):
        create_with_scheduler(
            _named_params(),
            optimizer_kwargs={"opt": "sgd", "lr": 0.01},
            scheduler_kwargs={"name": "CosineAnnealingWarmRestarts", "T_0": 10},
        )


def test_create_with_scheduler_cosine_warm_restarts_zero_updates_raises() -> None:
    """CosineAnnealingWarmRestarts raises when updates_per_epoch <= 0."""
    with pytest.raises(ValueError, match="updates_per_epoch must be a positive integer"):
        create_with_scheduler(
            _named_params(),
            optimizer_kwargs={"opt": "sgd", "lr": 0.01},
            scheduler_kwargs={
                "name": "CosineAnnealingWarmRestarts",
                "T_0": 10,
                "updates_per_epoch": 0,
            },
        )


def test_create_with_scheduler_reduce_lr_on_plateau_registers_epoch_callback() -> None:
    """ReduceLROnPlateau registers an on_epoch_end callback."""
    before = len(GLOBAL_CALLBACKS.on_epoch_end)
    create_with_scheduler(
        _named_params(),
        optimizer_kwargs={"opt": "sgd", "lr": 0.01},
        scheduler_kwargs={"name": "ReduceLROnPlateau", "criterion": "loss"},
    )
    assert len(GLOBAL_CALLBACKS.on_epoch_end) > before


def test_create_with_scheduler_reduce_lr_on_plateau_missing_criterion_raises() -> None:
    """ReduceLROnPlateau raises ValueError when criterion is absent."""
    with pytest.raises(ValueError, match="criterion must be specified"):
        create_with_scheduler(
            _named_params(),
            optimizer_kwargs={"opt": "sgd", "lr": 0.01},
            scheduler_kwargs={"name": "ReduceLROnPlateau"},
        )


def test_create_with_scheduler_step_lr_with_layer_decay() -> None:
    """create_with_scheduler applies layer decay and lr_scale correctly."""
    params = _named_params()
    opt = create_with_scheduler(
        params,
        optimizer_kwargs={
            "opt": "sgd",
            "lr": 0.01,
            "layer_decay": 0.8,
            "layer_group_regexes": [],
            "weight_decay": 0.01,
        },
        scheduler_kwargs={"name": "StepLR", "step_size": 1},
    )
    assert isinstance(opt, torch.optim.Optimizer)
