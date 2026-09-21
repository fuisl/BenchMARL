"""T-A1 measures the true cross-agent effect that every Task A claim rests on.

These tests pin the four properties that decide whether a reported off-diagonal
number means what the note says it means: the intervention stays applied for the
whole horizon, terminated steps are excluded, uncertainty is clustered by root
episode rather than by anchor, and an identical pair of branches reads as
exactly zero response.
"""

import json

import pytest
import torch

from examples.world_model.interaction_jacobian import (
    bootstrap_by_episode,
    constant_plan,
    cumulative_valid,
    resolve_task,
    witness_response,
)


def test_constant_plan_holds_the_intervened_coordinate_for_every_step():
    """A Jacobian column needs the action fixed; a transient is a different quantity."""
    action = torch.tensor([[[0.3, -0.2], [0.7, 0.1]]])  # (1, 2 agents, 2 axes)
    plan = constant_plan(action, steps=25)
    assert plan.shape == (1, 25, 2, 2)
    assert torch.equal(plan[:, 0], plan[:, -1])
    assert torch.equal(plan[0, 13], action[0])


def test_identical_branches_report_exactly_zero_response():
    """The floor of the measurement is zero, not an epsilon."""
    data = {
        "next_agent_state": torch.randn(4, 3, 2, 6),
        "next_package_state": torch.randn(4, 3, 1, 6),
        "valid": torch.ones(4, 3, dtype=torch.bool),
    }
    agent, shared, valid = witness_response(
        data, data, torch.ones(4), torch.ones(4)
    )
    assert torch.count_nonzero(agent) == 0
    assert torch.count_nonzero(shared) == 0
    assert valid.all()


def test_response_is_scaled_per_dimension():
    """Y is unitless, so halving a scale must double that dimension's response."""
    reference = {
        "next_agent_state": torch.zeros(1, 1, 1, 6),
        "next_package_state": torch.zeros(1, 1, 1, 6),
        "valid": torch.ones(1, 1, dtype=torch.bool),
    }
    branch = {key: value.clone() for key, value in reference.items()}
    branch["next_agent_state"][0, 0, 0, 0] = 1.0
    unit, _, _ = witness_response(
        reference, branch, torch.ones(4), torch.ones(4)
    )
    halved, _, _ = witness_response(
        reference, branch, torch.tensor([0.5, 1.0, 1.0, 1.0]), torch.ones(4)
    )
    assert unit[0, 0, 0] == pytest.approx(1.0)
    assert halved[0, 0, 0] == pytest.approx(2.0)


def test_cumulative_valid_never_revives_a_terminated_anchor():
    """Stored rows are zeroed after termination; their difference is not physics."""
    valid = torch.tensor([[True, True, False, True], [True, False, False, False]])
    live = cumulative_valid(valid)
    assert live.tolist() == [
        [True, True, False, False],
        [True, False, False, False],
    ]


def test_bootstrap_clusters_by_episode_not_by_anchor():
    """Anchors branch from shared episodes, so they are not independent samples.

    Twenty anchors drawn from two episodes must not produce a tighter interval
    than the two episodes themselves support. Audit section 5.3.
    """
    values = torch.cat([torch.zeros(10), torch.ones(10)])
    two_episodes = torch.cat([torch.zeros(10), torch.ones(10)]).long()
    twenty_episodes = torch.arange(20)

    clustered = bootstrap_by_episode(values, two_episodes)
    unclustered = bootstrap_by_episode(values, twenty_episodes)

    assert clustered["episodes"] == 2
    assert unclustered["episodes"] == 20
    clustered_width = clustered["high"] - clustered["low"]
    unclustered_width = unclustered["high"] - unclustered["low"]
    assert clustered_width > unclustered_width
    assert clustered["mean"] == pytest.approx(unclustered["mean"])


def test_resolve_task_rejects_a_manifest_that_disagrees_with_the_config():
    """A drifted task config would silently describe a different environment."""
    manifest = {
        "task_name": "vmas/buzz_wire",
        "task": {"collision_reward": -999.0},
    }
    with pytest.raises(ValueError, match="disagrees with the bank manifest"):
        resolve_task(manifest)


def test_resolve_task_accepts_the_recorded_buzz_wire_configuration():
    manifest = json.loads(
        '{"task_name": "vmas/buzz_wire", "task": {"collision_reward": -10.0}}'
    )
    task, recorded = resolve_task(manifest)
    assert recorded["collision_reward"] == -10.0
    assert task.config["collision_reward"] == -10.0


# --- T-A2 stage 2: effect-normalized fidelity -------------------------------

from examples.world_model.counterfactual_fidelity import (  # noqa: E402
    RESOLUTION_RATIO,
    column_groups,
    is_usable,
    verify_against_jacobian,
)


def test_column_groups_separate_the_diagonal_from_the_off_diagonal():
    """Pooling self with cross would hide the only block that distinguishes arms.

    Self-dynamics dominate the response and every arm, including `independent`,
    can represent them. An off-diagonal claim computed on a pooled vector would
    be a self-dynamics claim wearing its name.
    """
    groups = column_groups(agents=2, shared_bodies=3, intervened=0)
    assert groups["self"].tolist() == [0, 1, 2, 3]
    assert groups["cross"].tolist() == [4, 5, 6, 7]
    assert groups["shared"].tolist() == list(range(8, 20))

    mirrored = column_groups(agents=2, shared_bodies=3, intervened=1)
    assert mirrored["self"].tolist() == [4, 5, 6, 7]
    assert mirrored["cross"].tolist() == [0, 1, 2, 3]


def test_column_groups_cover_every_y_column_exactly_once():
    groups = column_groups(agents=3, shared_bodies=2, intervened=2)
    covered = sorted(c for group in groups.values() for c in group.tolist())
    assert covered == list(range(3 * 4 + 2 * 4))


def test_registered_resolution_rule_is_the_inverse_probe_floor():
    """E_CF and the floor are both divided by ||dY_true||, so the reported
    floor is the inverse resolution and the 3x rule is `floor <= 1/3`.

    This is the rule that retired K8: Balance's true response sat far below
    probe error and the ordering reported on it was noise.
    """
    assert RESOLUTION_RATIO == 3.0
    assert is_usable(0.0)           # a perfect probe
    assert is_usable(1.0 / 3.0)     # exactly at the registered boundary
    assert not is_usable(0.34)      # just inside the floor
    assert not is_usable(1.0)       # predicting nothing beats the instrument
    assert not is_usable(70.0)      # the Balance case that had to be retired


def _branch(value, steps=5, anchors=2, agents=2, bodies=1):
    low = {
        "next_agent_state": torch.zeros(anchors, steps, agents, 6),
        "next_package_state": torch.zeros(anchors, steps, bodies, 6),
        "valid": torch.ones(anchors, steps, dtype=torch.bool),
        "action": torch.zeros(anchors, steps, agents, 2),
    }
    high = {k: v.clone() for k, v in low.items()}
    high["next_agent_state"][..., 0] = value
    return {"low": low, "high": high}


def test_verify_against_jacobian_rejects_a_drifted_intervention():
    """Scoring models against a counterfactual other than the registered one
    would silently detach T-A2 from the floor T-A1 established."""
    branches = {(0, 0, 0): _branch(1.0)}
    scale = torch.ones(4)
    recorded = {
        "a0_axis0__agent_0__h1": {"mean": 1.0},
        "a0_axis0__agent_1__h1": {"mean": 1.0},
    }
    assert verify_against_jacobian(branches, 2, 5, scale, recorded, 1) == 2

    drifted = {"a0_axis0__agent_0__h1": {"mean": 0.25}}
    with pytest.raises(ValueError, match="do not reproduce T-A1"):
        verify_against_jacobian(branches, 2, 5, scale, drifted, 1)


def test_bootstrap_reports_an_empty_cell_instead_of_crashing_the_report():
    """Deep-horizon cells can have every anchor terminated.

    Job 1497 computed and wrote the whole Jacobian and then died formatting a
    horizon-4 cell whose anchors had all terminated.
    """
    empty = bootstrap_by_episode(torch.empty(0), torch.empty(0, dtype=torch.long))
    assert empty["episodes"] == 0
    assert empty["anchors"] == 0
    assert empty["mean"] != empty["mean"]  # NaN, not a silent zero


def test_oracle_per_anchor_gain_bound_matches_the_closed_form():
    """min over r of ||r*u - v|| / ||v|| is sqrt(1 - cos^2), at a per-anchor r.

    The minimising r depends on the TRUE response, so this is an oracle bound.
    It separates "the model has no cross-agent information" from "it has the
    information and the wrong gain"; it does not claim one global scalar would
    do as well.
    """
    torch.manual_seed(0)
    true = torch.randn(64, 4).double()
    predicted = torch.randn(64, 4).double()
    cosine = torch.nn.functional.cosine_similarity(predicted, true, dim=1)
    closed_form = (1.0 - cosine**2).clamp_min(0).sqrt()

    grid = torch.linspace(-3.0, 3.0, 2001).double()
    for row in range(8):
        errors = (
            grid[:, None] * predicted[row] - true[row]
        ).norm(dim=1) / true[row].norm()
        assert float(errors.min()) == pytest.approx(float(closed_form[row]), abs=2e-3)


def test_zero_response_scores_exactly_one_and_cannot_be_rescaled_below_it():
    """`independent` is the structural floor, not a competitor.

    Its predicted cross response is exactly zero, so E_CF is exactly 1 and no
    rescaling helps -- which is why a conditioned arm scoring above 1 is a real
    finding rather than a tie.
    """
    true = torch.randn(32, 4).double()
    zero = torch.zeros_like(true)
    e_cf = (zero - true).norm(dim=1) / true.norm(dim=1)
    cosine = torch.nn.functional.cosine_similarity(zero, true, dim=1, eps=1e-12)
    assert torch.allclose(e_cf, torch.ones_like(e_cf))
    assert torch.allclose(cosine, torch.zeros_like(cosine))
    assert torch.allclose((1.0 - cosine**2).sqrt(), torch.ones_like(cosine))


# --- T-A2b: counterfactual information localization -------------------------

from examples.world_model.counterfactual_localization import (  # noqa: E402
    build_rows,
)


def _localization_branch(anchors=6, steps=5, agents=2, bodies=1, obs=6):
    """Mirrors what `collect.rollout_actions` records, including `observation`."""
    return {
        "observation": torch.randn(anchors, steps, agents, obs),
        "next_agent_state": torch.randn(anchors, steps, agents, 6),
        "next_package_state": torch.randn(anchors, steps, bodies, 6),
        "agent_state": torch.randn(anchors, steps, agents, 6),
        "package_state": torch.randn(anchors, steps, bodies, 6),
        "valid": torch.ones(anchors, steps, dtype=torch.bool),
        "action": torch.randn(anchors, steps, agents, 2),
    }


def test_build_rows_keeps_the_action_block_width_across_cells():
    """The per-cell column loop must not rebind the action-block width.

    It did, which passed the string "self" into `blocked()` on the second cell.
    Four cells is the smallest case that exposes it; one cell would pass.
    """
    branches = {
        (0, intervened, axis): {
            "low": _localization_branch(), "high": _localization_branch()
        }
        for intervened in range(2)
        for axis in range(2)
    }
    scale = torch.ones(2 * 4 + 1 * 4).double()
    rows, target, columns, episodes = build_rows(
        branches, 5, 4, 2, scale, torch.arange(6), None
    )
    assert rows["actions_only"].shape == (24, 40)
    # 2 agents x 6 observation dims, plus the 40 action columns.
    assert rows["observation_raw"].shape == (24, 12 + 40)
    assert target.shape == (24, 12)
    assert columns["cross"].shape == columns["self"].shape == (24, 4)
    assert episodes.shape[0] == 24


def test_build_rows_self_and_cross_columns_are_disjoint_per_row():
    """A row's self and cross blocks must never name the same column."""
    branches = {
        (0, intervened, 0): {
            "low": _localization_branch(), "high": _localization_branch()
        }
        for intervened in range(2)
    }
    scale = torch.ones(2 * 4 + 1 * 4).double()
    _, _, columns, _ = build_rows(branches, 5, 4, 2, scale, torch.arange(6), None)
    overlap = (columns["cross"].unsqueeze(2) == columns["self"].unsqueeze(1)).any()
    assert not bool(overlap)


# --- G0: history window alignment -------------------------------------------

from examples.world_model.counterfactual_localization import (  # noqa: E402
    SOURCE_REGIMES,
    history_window,
)


def _fake_bank(tmp_path, episodes=4, steps=10, agents=2, obs=6, act=2):
    """Two regime trajectory files whose contents are distinguishable."""
    for index, name in enumerate(SOURCE_REGIMES):
        torch.save(
            {
                "observation": torch.arange(
                    episodes * steps * agents * obs, dtype=torch.float32
                ).reshape(episodes, steps, agents, obs)
                + index * 1000.0,
                "action": torch.arange(
                    episodes * steps * agents * act, dtype=torch.float32
                ).reshape(episodes, steps, agents, act)
                + index * 1000.0,
            },
            tmp_path / f"trajectories_{name}.pt",
        )


def test_history_window_rejects_a_misaligned_lookup(tmp_path):
    """A window aligned to the wrong episode makes every later number meaningless.

    The regime index order was validated bit-exactly against real data; this
    pins that a disagreement is raised rather than absorbed.
    """
    _fake_bank(tmp_path)
    anchors = {
        "episode_id": torch.tensor([0, 1]),
        "source_step": torch.tensor([5, 6]),
        "source_regime": torch.tensor([0, 1]),
    }
    rows = torch.tensor([0, 1])
    wrong = torch.zeros(2, 2, 6)
    with pytest.raises(ValueError, match="misaligned"):
        history_window(anchors, rows, tmp_path, 3, wrong)


def test_history_window_shape_and_episode_start_clamping(tmp_path):
    """Frames clamp at 0 at an episode start, repeating the earliest frame.

    That is the convention `model_input.PlanningContext` already registers for a
    decision with no past, so the diagnostic and the planner agree.
    """
    _fake_bank(tmp_path)
    source = torch.load(
        tmp_path / f"trajectories_{SOURCE_REGIMES[0]}.pt", weights_only=True
    )
    anchors = {
        "episode_id": torch.tensor([0]),
        "source_step": torch.tensor([0]),  # episode start: nothing precedes it
        "source_regime": torch.tensor([0]),
    }
    rows = torch.tensor([0])
    observed = source["observation"][0, 0].unsqueeze(0)
    window = history_window(anchors, rows, tmp_path, 3, observed)

    agents, obs, act = 2, 6, 2
    assert window.shape == (1, 3 * agents * obs + 2 * agents * act)
    frames = window[0, : 3 * agents * obs].reshape(3, agents, obs)
    # All three frames are the same clamped frame, and it is the observed one.
    assert torch.equal(frames[0], frames[2])
    assert torch.allclose(frames[2], observed[0].double())


# --- F13: authentic reference context ---------------------------------------

from examples.world_model.counterfactual_fidelity import (  # noqa: E402
    SOURCE_REGIMES as CF_SOURCE_REGIMES,
    reference_context,
)


def _cf_bank(tmp_path, episodes=4, steps=40, agents=2, obs=6, act=2):
    for index, name in enumerate(CF_SOURCE_REGIMES):
        torch.save(
            {
                "observation": torch.arange(
                    episodes * steps * agents * obs, dtype=torch.float32
                ).reshape(episodes, steps, agents, obs)
                + index * 100000.0,
                "action": torch.arange(
                    episodes * steps * agents * act, dtype=torch.float32
                ).reshape(episodes, steps, agents, act)
                + index * 100000.0,
            },
            tmp_path / f"trajectories_{name}.pt",
        )


def test_reference_context_uses_the_block_stride_not_consecutive_steps(tmp_path):
    """A reference frame is a BLOCK BOUNDARY.

    `dataset.py` builds the sequence as [observation[0]] +
    next_observation[block-1::block], so a 3-frame context spans 2*block
    primitive steps. Using the stride-1 diagnostic window here would reproduce
    F13 in a new place.
    """
    _cf_bank(tmp_path)
    block, history = 5, 3
    source = torch.load(
        tmp_path / f"trajectories_{CF_SOURCE_REGIMES[0]}.pt", weights_only=True
    )
    anchors = {
        "episode_id": torch.tensor([0]),
        "source_step": torch.tensor([20]),
        "source_regime": torch.tensor([0]),
    }
    observed = source["observation"][0, 20].unsqueeze(0)
    frames, actions = reference_context(
        anchors, torch.tensor([0]), tmp_path, history, block, observed
    )
    assert frames.shape == (1, history, 2, 6)
    assert actions.shape == (1, history - 1, 2, block * 2)
    # Frames at steps 10, 15, 20 -- stride `block`, not 18, 19, 20.
    for position, step in enumerate((10, 15, 20)):
        assert torch.equal(frames[0, position], source["observation"][0, step])


def test_reference_context_rejects_a_misaligned_anchor(tmp_path):
    _cf_bank(tmp_path)
    anchors = {
        "episode_id": torch.tensor([0]),
        "source_step": torch.tensor([20]),
        "source_regime": torch.tensor([0]),
    }
    with pytest.raises(ValueError, match="misaligned"):
        reference_context(
            anchors, torch.tensor([0]), tmp_path, 3, 5, torch.zeros(1, 2, 6)
        )


def test_reference_context_clamps_at_an_episode_start(tmp_path):
    """At step 0 every frame clamps to the first, which is the only case where
    the synthetic-start convention is legitimate."""
    _cf_bank(tmp_path)
    source = torch.load(
        tmp_path / f"trajectories_{CF_SOURCE_REGIMES[0]}.pt", weights_only=True
    )
    anchors = {
        "episode_id": torch.tensor([0]),
        "source_step": torch.tensor([0]),
        "source_regime": torch.tensor([0]),
    }
    frames, _ = reference_context(
        anchors, torch.tensor([0]), tmp_path, 3, 5,
        source["observation"][0, 0].unsqueeze(0),
    )
    assert torch.equal(frames[0, 0], frames[0, 2])
