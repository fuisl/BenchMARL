import pytest
import torch

from examples.world_model.decision_information import (
    ball_clearance,
    diagnostic_targets,
    rank_metrics,
)


def test_ball_clearance_is_signed_distance_to_buzz_wire_boundary():
    position = torch.tensor([[0.0, 0.0], [0.1, 0.0], [0.0, 0.98]])
    torch.testing.assert_close(
        ball_clearance(position), torch.tensor([0.095, -0.005, -0.010])
    )


def test_diagnostic_targets_separate_progress_and_collision_penalty():
    agent_state = torch.zeros(1, 3, 2, 6)
    observation = torch.zeros(1, 3, 2, 6)
    observation[..., 4] = -1.0  # agent position - goal position
    package_state = torch.zeros(1, 3, 1, 6)
    package_state[0, :, 0, 0] = torch.tensor([0.0, 0.1, 0.3])
    primitive_reward = torch.tensor([[[[[0.1], [0.1]]], [[[-9.8], [-9.8]]]]])
    regression, collision = diagnostic_targets(
        {
            "agent_state": agent_state,
            "observation": observation,
            "package_state": package_state,
            "primitive_reward": primitive_reward,
            "primitive_valid": torch.ones(1, 2, 1, dtype=torch.bool),
        }
    )
    torch.testing.assert_close(regression[0, :, 0], torch.tensor([0.1, 0.2]))
    torch.testing.assert_close(regression[0, :, 4], torch.tensor([0.0, 20.0]))
    torch.testing.assert_close(collision, torch.tensor([[0.0, 1.0]]))


def test_rank_metrics_only_name_oracle_for_the_fixed_bank():
    trace = {
        "return": torch.tensor([[4.0, 0.0, 1.0, 2.0]]),
        "distance": torch.tensor([[[0.1], [1.0], [0.8], [0.5]]]),
        "collision": torch.zeros(1, 4, 1, dtype=torch.bool),
    }
    cost = -trace["return"]
    fixed = rank_metrics(cost, trace, topk=2, named_candidates=True)
    assert fixed["spearman"] == pytest.approx(1.0)
    assert fixed["topk_recall"] == 1.0
    assert fixed["oracle_percentile"] == 0.0
    assert fixed["selected_regret"] == 0.0

    stage = rank_metrics(cost, trace, topk=2)
    assert "oracle_percentile" not in stage
    assert stage["candidate0_percentile"] == 0.0
