# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the license in the LICENSE file in the root directory.
"""Audit Gate A0 step-2 temporal context and SIGReg contracts."""

from types import SimpleNamespace

import pytest
import torch

from examples.world_model.closed_loop import goal_costs, reward_costs
from examples.world_model.model_input import PlanningContext, ReferenceHistory
from examples.world_model.models import ReferenceMultiAgentWorldModel, SIGReg
from examples.world_model.train import sigreg_loss
from omegaconf import OmegaConf


class _ActionSpec:
    def __getitem__(self, key):
        assert key == ("agents", "action")
        return SimpleNamespace(shape=torch.Size([2]))


class _Env:
    full_action_spec_unbatched = _ActionSpec()


class _Frames:
    state_input = "observation"

    def __init__(self, frames):
        self.frames = frames
        self.index = 0

    def reset(self):
        self.index = 0

    def __call__(self, _env):
        value = self.frames[self.index]
        self.index += 1
        return value


def _model(agents=2):
    torch.manual_seed(4)
    return ReferenceMultiAgentWorldModel(
        "relational",
        obs_dim=3,
        action_dim=4,
        agents=agents,
        dim=16,
        hidden_dim=24,
        history_size=3,
        depth=2,
        heads=2,
        dim_head=8,
        mlp_dim=32,
        dropout=0.0,
        projector_hidden_dim=32,
    ).eval()


def test_episode_start_and_real_history_are_exact():
    frames = [torch.full((1, 2, 3), float(i)) for i in range(3)]
    history = ReferenceHistory(_Frames(frames), history_size=3, action_block=2)
    env = _Env()
    first = history(env)
    assert first.observations[:, :, 0, 0].tolist() == [[0.0, 0.0, 0.0]]
    assert not first.past_actions.any()

    a0 = torch.arange(8, dtype=torch.float32).reshape(1, 2, 4)
    history.record_action(a0)
    second = history(env)
    assert second.observations[:, :, 0, 0].tolist() == [[0.0, 0.0, 1.0]]
    assert torch.equal(second.past_actions[:, 0], torch.zeros_like(a0))
    assert torch.equal(second.past_actions[:, 1], a0)

    a1 = a0 + 10
    history.record_action(a1)
    third = history(env)
    assert third.observations[:, :, 0, 0].tolist() == [[0.0, 1.0, 2.0]]
    assert torch.equal(third.past_actions[:, 0], a0)
    assert torch.equal(third.past_actions[:, 1], a1)


def test_history_refuses_missing_or_double_actions():
    frames = [torch.zeros(1, 2, 3), torch.ones(1, 2, 3)]
    history = ReferenceHistory(_Frames(frames), action_block=2)
    history(_Env())
    with pytest.raises(RuntimeError, match="recorded before the next frame"):
        history(_Env())
    action = torch.zeros(1, 2, 4)
    history.record_action(action)
    with pytest.raises(RuntimeError, match="previous action"):
        history.record_action(action)


def test_rollout_matches_pinned_context_update_bit_for_bit():
    model = _model(agents=1)
    latent = torch.randn(2, 3, 1, 16)
    past = torch.randn(2, 2, 1, 4)
    future = torch.randn(2, 5, 1, 4)
    with torch.no_grad():
        ours = model.rollout_from_context(latent, past, future)
        history, actions, expected = latent, past, []
        for current in future.split(1, dim=1):
            action_context = torch.cat([actions, current], dim=1)
            predicted = model.predict(history[:, -3:], action_context[:, -3:])[:, -1:]
            expected.append(predicted)
            history = torch.cat([history, predicted], dim=1)
            actions = torch.cat([actions, current], dim=1)
        expected = torch.cat(expected, dim=1)
    assert torch.equal(ours, expected)


def test_rollout_action_windows_are_past_then_candidate():
    model = _model(agents=1)
    seen = []

    def predict(latent, action):
        seen.append((latent.clone(), action.clone()))
        return latent

    model.predict = predict
    latent = torch.tensor([[[[0.0]], [[1.0]], [[2.0]]]]).expand(-1, -1, -1, 16)
    past = torch.tensor([[[[10.0]], [[11.0]]]]).expand(-1, -1, -1, 4)
    future = torch.tensor([[[[12.0]], [[13.0]], [[14.0]]]]).expand(-1, -1, -1, 4)
    model.rollout_from_context(latent, past, future)
    assert [value[1][0, :, 0, 0].tolist() for value in seen] == [
        [10.0, 11.0, 12.0],
        [11.0, 12.0, 13.0],
        [12.0, 13.0, 14.0],
    ]
    assert seen[0][0][0, :, 0, 0].tolist() == [0.0, 1.0, 2.0]


class _ShapeSIGReg(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.shapes = []

    def forward(self, value):
        self.shapes.append(value.shape)
        return value.square().mean()


def _sigreg_cfg(mode):
    return OmegaConf.create({"train": {"sigreg_population": mode}})


def test_joint_sigreg_preserves_time_and_uses_agents_as_population():
    latent = torch.randn(5, 4, 3, 7)
    valid = torch.ones(5, 4, dtype=torch.bool)
    valid[-1, -1] = False
    sigreg = _ShapeSIGReg()
    sigreg_loss(_model(agents=3), sigreg, latent, valid, _sigreg_cfg("joint"))
    assert sigreg.shapes == [torch.Size([4, 4 * 3, 7])]


def test_per_agent_sigreg_keeps_batch_population_and_averages_agents():
    latent = torch.randn(5, 4, 3, 7)
    valid = torch.ones(5, 4, dtype=torch.bool)
    sigreg = _ShapeSIGReg()
    sigreg_loss(
        _model(agents=3), sigreg, latent, valid, _sigreg_cfg("per_agent")
    )
    assert sigreg.shapes == [torch.Size([4, 5, 7])] * 3


def test_reference_sigreg_accepts_singleton_validation_batch():
    latent = torch.randn(1, 4, 3, 7)
    valid = torch.ones(1, 4, dtype=torch.bool)
    sigreg = _ShapeSIGReg()
    sigreg_loss(_model(agents=3), sigreg, latent, valid, _sigreg_cfg("joint"))
    assert sigreg.shapes == [torch.Size([4, 3, 7])]


def test_single_agent_sigreg_matches_pinned_transpose_exactly():
    latent = torch.randn(8, 4, 1, 12)
    valid = torch.ones(8, 4, dtype=torch.bool)
    sigreg = SIGReg(knots=17, num_proj=64)
    model = _model(agents=1)
    torch.manual_seed(91)
    ours = sigreg_loss(
        model, sigreg, latent, valid, _sigreg_cfg("joint")
    )
    torch.manual_seed(91)
    pinned = sigreg(latent[:, :, 0].transpose(0, 1))
    torch.testing.assert_close(ours, pinned, atol=0, rtol=0)


@pytest.mark.parametrize(
    "device", ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
)
def test_reference_plan_costs_require_and_accept_real_context(device):
    model = _model(agents=2).to(device)
    batch, candidates, horizon, action_block = 2, 4, 3, 2
    observation = torch.randn(batch, 3, 2, 3)
    past = torch.randn(batch, 2, 2, 4)
    context = PlanningContext(observation, past)
    plans = torch.randn(batch, candidates, horizon * action_block, 4)
    goal = torch.randn(batch, 2, 3)
    for cost in (
        reward_costs(model, action_block=action_block, device=device),
        goal_costs(model, goal, action_block=action_block, device=device),
    ):
        result = cost(None, context, plans)
        assert result.shape == (batch, candidates)
        assert torch.isfinite(result).all()
        with pytest.raises(ValueError, match="real PlanningContext"):
            cost(None, context.current, plans)
