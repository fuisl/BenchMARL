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
