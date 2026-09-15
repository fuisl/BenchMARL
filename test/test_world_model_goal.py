#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#
"""The goal-cost prototype must match LeWM, not merely resemble it.

Each test pins one semantic of the reference (revision 8edfeb33, `jepa.py`) that
is easy to get subtly wrong and impossible to notice afterwards: terminal-frame
only, a detached goal, a summed rather than averaged reduction, and a predictor
context truncated to `history_size`.
"""

import torch
import torch.nn.functional as F

from examples.world_model.goal_planning import (
    lewm_rollout,
    terminal_goal_cost,
    true_goal_costs,
)
from test.test_world_model_models import build, sample


def reference_criterion(pred_emb, goal_emb):
    """Literal transcription of LeWM `JEPA.criterion`, for comparison."""
    goal_emb = goal_emb[..., -1:, :].expand_as(pred_emb)
    return F.mse_loss(
        pred_emb[..., -1:, :],
        goal_emb[..., -1:, :].detach(),
        reduction="none",
    ).sum(dim=tuple(range(2, pred_emb.ndim)))


def test_matches_the_reference_criterion_for_one_agent():
    """With a single agent our cost must reduce to LeWM's exactly."""
    torch.manual_seed(0)
    predicted = torch.randn(3, 5, 4, 1, 8)  # (B,K,H,N=1,D)
    goal = torch.randn(3, 1, 1, 1, 8)
    ours = terminal_goal_cost(predicted, goal)
    theirs = reference_criterion(
        predicted.squeeze(3), goal.squeeze(3).expand(3, 5, 1, 8)
    )
    torch.testing.assert_close(ours, theirs, atol=1e-6, rtol=1e-6)


def test_cost_uses_only_the_terminal_frame():
    """LeWM scores the last predicted frame; earlier frames must not matter."""
    torch.manual_seed(0)
    predicted = torch.randn(2, 4, 5, 3, 6)
    goal = torch.randn(2, 1, 1, 3, 6)
    before = terminal_goal_cost(predicted, goal)
    predicted[:, :, :-1] = torch.randn_like(predicted[:, :, :-1])
    torch.testing.assert_close(before, terminal_goal_cost(predicted, goal))


def test_goal_is_detached():
    """The reference detaches the goal, so no gradient may reach it."""
    predicted = torch.randn(2, 3, 4, 2, 5, requires_grad=True)
    goal = torch.randn(2, 1, 1, 2, 5, requires_grad=True)
    terminal_goal_cost(predicted, goal).sum().backward()
    assert goal.grad is None or torch.count_nonzero(goal.grad) == 0
    assert predicted.grad is not None


def test_reduction_is_a_sum_not_a_mean():
    """A mean would make the cost independent of latent width; a sum does not."""
    torch.manual_seed(0)
    narrow = terminal_goal_cost(torch.ones(1, 1, 2, 1, 4), torch.zeros(1, 1, 1, 1, 4))
    wide = terminal_goal_cost(torch.ones(1, 1, 2, 1, 8), torch.zeros(1, 1, 1, 1, 8))
    assert float(narrow) == 4.0 and float(wide) == 8.0


def test_cost_is_zero_at_the_goal_and_grows_with_distance():
    goal = torch.zeros(1, 1, 1, 2, 3)
    at_goal = torch.zeros(1, 1, 2, 2, 3)
    away = torch.full((1, 1, 2, 2, 3), 2.0)
    assert float(terminal_goal_cost(at_goal, goal)) == 0.0
    assert float(terminal_goal_cost(away, goal)) > 0.0


def test_rollout_truncates_context_to_history_size():
    """LeWM keeps only the last `history_size` frames; ours otherwise grows."""
    model = build("relational", wake=True)
    observation, action = sample()
    with torch.no_grad():
        latent = model.encode(observation)[:, :1]
        truncated = lewm_rollout(model, latent, action, history_size=2)
        untruncated = model.rollout(latent, action)
    assert truncated.shape == untruncated.shape
    # The first two steps cannot differ; later ones must, or no truncation happened.
    torch.testing.assert_close(truncated[:, :2], untruncated[:, :2], atol=1e-5, rtol=1e-5)
    assert not torch.allclose(truncated[:, -1], untruncated[:, -1], atol=1e-6)


def test_rollout_matches_the_unrestricted_one_when_history_is_large():
    """With a window wider than the horizon the two must coincide exactly."""
    model = build("relational", wake=True)
    observation, action = sample()
    with torch.no_grad():
        latent = model.encode(observation)[:, :1]
        wide = lewm_rollout(model, latent, action, history_size=99)
        base = model.rollout(latent, action)
    torch.testing.assert_close(wide, base, atol=1e-6, rtol=1e-6)


def test_true_goal_cost_is_zero_for_a_candidate_that_reaches_the_goal():
    """Endpoints now come from the rollout, so the cost takes them directly."""
    endpoint = torch.randn(2, 3, 2, 5)
    goal = endpoint[:, 0]
    cost = true_goal_costs(endpoint, goal)
    assert float(cost[0, 0]) < 1e-10 and float(cost[1, 0]) < 1e-10
    assert float(cost[0, 1]) > 0.0


def test_block_boundary_endpoints_are_refused():
    """The replaced helper mis-located endpoints; it must not be used silently."""
    import pytest

    from examples.world_model.goal_planning import terminal_observations

    with pytest.raises(NotImplementedError):
        terminal_observations(torch.zeros(1, 1, 3, 1, 1), torch.ones(1, 1, 2, dtype=torch.bool))
