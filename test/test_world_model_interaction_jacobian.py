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


def test_rescaling_bound_matches_the_closed_form():
    """min over r of ||r*u - v|| / ||v|| is sqrt(1 - cos^2), attained at r = cos.

    This is what separates "the model has no cross-agent information" from
    "it has the information and the wrong gain".
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
