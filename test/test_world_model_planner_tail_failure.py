import pytest
import torch

from examples.world_model.planner_tail_failure import (
    information_contract,
    rank_and_tail_metrics,
    registered_branch_result,
    surrogate_trajectories,
)
from examples.world_model.mpc import unpack_actions
from examples.world_model.structured_surrogate import (
    StructuredSurrogate,
    surrogate_cost,
)


class DriftModel:
    """A locally deterministic model whose recursion visibly accumulates."""

    def __call__(self, state, action):
        next_state = state.clone()
        next_state[:, :12] += 1
        batch = state.shape[0]
        return (
            next_state,
            torch.zeros(batch),
            torch.full((batch,), -4.0),
            torch.zeros(batch),
        )


def test_teacher_forcing_and_recursion_match_first_step_then_separate():
    roots, candidates, horizon = 1, 2, 3
    truth = {"state": torch.zeros(roots, candidates, horizon + 1, 14)}
    actions = torch.zeros(roots, candidates, horizon, 20)
    paths = surrogate_trajectories(
        DriftModel(), truth, actions, action_block=5, device="cpu"
    )
    teacher = paths["teacher_forced"]["next_state"]
    recursive = paths["recursive"]["next_state"]
    assert torch.equal(teacher[:, :, 0], recursive[:, :, 0])
    assert torch.equal(teacher[:, :, 1, :12], torch.ones(1, 2, 12))
    assert torch.equal(recursive[:, :, 1, :12], torch.full((1, 2, 12), 2.0))


def test_tail_collision_metric_is_conditioned_on_predicted_topk():
    predicted = torch.tensor([[0.0, 1.0, 2.0, 3.0]])
    truth = torch.tensor([[3.0, 0.0, 1.0, 2.0]])
    collision = torch.tensor([[True, False, False, True]])
    probability = torch.tensor([[0.1, 0.2, 0.3, 0.4]])
    oracle = torch.tensor([-1.0])
    metrics = rank_and_tail_metrics(
        predicted, truth, collision, probability, oracle
    )
    assert metrics["true_collision_given_predicted_top1"] == 1.0
    assert metrics["predicted_collision_given_predicted_top1"] == pytest.approx(0.1)
    assert metrics["top1_recall"] == 0.0
    assert metrics["oracle_percentile"] == 0.0
    assert metrics["selected_regret"] == 3.0


def test_information_contract_marks_legacy_state_as_privileged():
    contract = information_contract()
    assert contract["candidate_selection_inputs"]["initial_structured_state"] == "P"
    assert contract["future_truth_used_by_candidate_selection"] is False
    assert contract["deployment_contract_satisfied"] is False
    assert contract["diagnostic_only"] is True


def test_recorded_recursive_cost_exactly_matches_planner_cost():
    torch.manual_seed(4)
    model = StructuredSurrogate(
        torch.zeros(14),
        torch.ones(14),
        torch.zeros(20),
        torch.ones(20),
        torch.zeros(12),
        torch.ones(12),
        torch.zeros(1),
        torch.ones(1),
        torch.zeros(1),
        torch.ones(1),
        hidden=8,
    ).eval()
    candidates = torch.rand(1, 3, 2, 20) * 2 - 1
    state = torch.rand(1, 14)
    truth = {"state": state[:, None, None].expand(1, 3, 3, 14).clone()}
    paths = surrogate_trajectories(
        model, truth, candidates, action_block=5, device="cpu"
    )
    expected = surrogate_cost(
        model,
        state,
        unpack_actions(candidates, 5),
        action_block=5,
        objective="probability",
    )
    assert torch.equal(paths["recursive"]["cost"][:, :, -1], expected)


def test_registered_branch_prioritizes_upstream_ood():
    rows = []
    for seed in (1, 2, 3):
        for iteration in (1, 30):
            for horizon in range(1, 6):
                scale = 2.0 if iteration == 30 and seed in (1, 2) else 1.0
                rows.append(
                    {
                        "seed": seed,
                        "iteration": iteration,
                        "horizon": horizon,
                        "teacher_forced": {
                            "ball_position_rmse": scale,
                            "clearance_rmse": scale,
                            "cost_rmse": scale,
                        },
                        "recursive": {
                            "ball_position_rmse": 3.0,
                            "true_collision_given_predicted_top10": 0.5,
                            "predicted_collision_given_predicted_top10": 0.1,
                        },
                        "recursive_vs_teacher_forced": {
                            "ball_position_rmse": 0.2,
                        },
                    }
                )
    result = registered_branch_result(rows)
    assert result["all_rules"]["G6a_planner_aware_aggregation"] is True
    assert result["primary_next_intervention"] == "G6a_planner_aware_aggregation"
