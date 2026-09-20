import pytest
import torch

from examples.world_model.planner_tail_failure import (
    information_contract,
    load_surrogate,
    rank_and_tail_metrics,
    registered_branch_result,
    state_slices,
    surrogate_trajectories,
)
from examples.world_model.mpc import unpack_actions
from examples.world_model.structured_surrogate import (
    StructuredSurrogate,
    state_spec,
    surrogate_cost,
)


class DriftModel:
    """A locally deterministic model whose recursion visibly accumulates."""

    state_profile = "legacy14"
    state_dim = 14

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


@pytest.mark.parametrize("profile", ["legacy14", "full32"])
def test_information_contract_marks_structured_state_as_privileged(profile):
    contract = information_contract(state_spec(profile))
    assert contract["candidate_selection_inputs"]["initial_structured_state"] == "P"
    assert contract["future_truth_used_by_candidate_selection"] is False
    assert contract["deployment_contract_satisfied"] is False
    assert contract["diagnostic_only"] is True
    # The successor state is privileged for exactly the same reason, so the
    # contract must not quietly become deployable when the width changes.
    assert contract["state_profile"] == profile


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


def test_frozen_gate5_rejects_full32_checkpoint(tmp_path):
    model = StructuredSurrogate(
        torch.zeros(32),
        torch.ones(32),
        torch.zeros(20),
        torch.ones(20),
        torch.zeros(30),
        torch.ones(30),
        torch.zeros(1),
        torch.ones(1),
        torch.zeros(1),
        torch.ones(1),
        hidden=8,
    )
    checkpoint = tmp_path / "full32.pt"
    torch.save(
        {"state_dict": model.state_dict(), "state_profile": "full32"},
        checkpoint,
    )
    # The frozen job-1331 protocol must keep refusing the successor state
    # under its default and when named explicitly.
    for call in (lambda: load_surrogate(checkpoint, "cpu"),
                 lambda: load_surrogate(checkpoint, "cpu", "gate5_legacy14")):
        with pytest.raises(ValueError, match="never silently accept a successor"):
            call()
    # A1.2's own protocol is the registered place where full32 is measured.
    model, _checkpoint = load_surrogate(checkpoint, "cpu", "a1_paired")
    assert model.state_profile == "full32"


def test_frozen_gate5_loads_historical_checkpoint_without_profile(tmp_path):
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
    )
    checkpoint = tmp_path / "legacy.pt"
    torch.save({"state_dict": model.state_dict()}, checkpoint)
    restored, _payload = load_surrogate(checkpoint, "cpu")
    assert restored.state_profile == "legacy14"


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


@pytest.mark.parametrize(
    "profile,expected",
    [
        (
            "legacy14",
            {
                "agents": (0, 8),
                "ball_position": (8, 10),
                "ball_velocity": (10, 12),
                "link_bodies": (12, 12),
                "goal": (12, 14),
            },
        ),
        (
            "full32",
            {
                "agents": (0, 12),
                "ball_position": (12, 14),
                "ball_velocity": (14, 16),
                "link_bodies": (18, 30),
                "goal": (30, 32),
            },
        ),
    ],
)
def test_state_slices_match_each_registered_profile(profile, expected):
    """A1.2 reports ball and linkage error apart, so the indices must follow
    the profile rather than legacy14's layout. legacy14's linkage group is
    empty, which is the omission A1.1 demonstrated."""
    spec = state_spec(profile)
    slices = state_slices(spec)
    assert {
        name: (item.start, item.stop) for name, item in slices.items()
    } == expected
    # The ball slice the surrogate itself uses must agree with the report's.
    assert (slices["ball_position"].start, slices["ball_position"].stop) == (
        spec.ball_position.start,
        spec.ball_position.stop,
    )


def test_full32_rollout_uses_its_own_ball_and_goal_coordinates():
    """A full32 rollout must read the ball and goal at 12:14 and 30:32.

    Reusing legacy14's indices would silently score the progress term on the
    wrong coordinates, which no shape check would catch.
    """
    torch.manual_seed(11)
    model = StructuredSurrogate(
        torch.zeros(32),
        torch.ones(32),
        torch.zeros(20),
        torch.ones(20),
        torch.zeros(30),
        torch.ones(30),
        torch.zeros(1),
        torch.ones(1),
        torch.zeros(1),
        torch.ones(1),
        hidden=16,
    )
    roots, candidates, horizon = 1, 3, 2
    truth = {"state": torch.randn(roots, candidates, horizon + 1, 32)}
    actions = torch.randn(roots, candidates, horizon, 20)
    paths = surrogate_trajectories(
        model, truth, actions, action_block=5, device="cpu"
    )
    assert paths["recursive"]["state"].shape[-1] == 32
    assert paths["teacher_forced"]["next_state"].shape[-1] == 32
    # The goal coordinates are carried, never predicted, by either path.
    torch.testing.assert_close(
        paths["recursive"]["state"][..., 30:32],
        truth["state"][..., :1, 30:32].expand(roots, candidates, horizon + 1, 2),
    )
