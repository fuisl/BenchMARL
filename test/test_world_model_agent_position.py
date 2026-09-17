# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the license in the LICENSE file in the root directory.
"""Direct per-agent physical-position validation."""

import pytest
import torch

from examples.world_model.agent_position import error_summary
from examples.world_model.models import SharedAgentPositionModel


AGENTS, OBS, ACT = 3, 7, 4


def build(kind, seed=0):
    torch.manual_seed(seed)
    return SharedAgentPositionModel(
        kind,
        OBS,
        ACT,
        AGENTS,
        hidden_dim=16,
        conditioner_budget=2000,
        delta_mean=torch.tensor([0.1, -0.2]),
        delta_std=torch.tensor([0.5, 2.0]),
    ).eval()


def sample(batch=5, steps=2):
    generator = torch.Generator().manual_seed(7)
    observation = torch.randn(
        batch, steps, AGENTS, OBS, generator=generator
    )
    action = torch.randn(batch, steps, AGENTS, ACT, generator=generator)
    return observation, action


@pytest.mark.parametrize("kind", ("independent", "joint", "relational"))
def test_shared_position_model_shape_and_finite_output(kind):
    prediction = build(kind)(*sample())
    assert prediction.shape == (5, 2, AGENTS, 2)
    assert torch.isfinite(prediction).all()


def test_independent_position_model_has_no_cross_agent_path():
    model = build("independent")
    observation, action = sample()
    changed_observation = observation.clone()
    changed_action = action.clone()
    changed_observation[..., 1, :] += 10
    changed_action[..., 1, :] -= 10
    with torch.no_grad():
        original = model(observation, action)
        changed = model(changed_observation, changed_action)
    torch.testing.assert_close(original[..., 0, :], changed[..., 0, :])
    assert not torch.equal(original[..., 1, :], changed[..., 1, :])


@pytest.mark.parametrize("kind", ("independent", "relational"))
def test_shared_position_model_is_permutation_equivariant(kind):
    model = build(kind)
    observation, action = sample()
    order = [2, 0, 1]
    with torch.no_grad():
        original = model(observation, action)
        permuted = model(observation[..., order, :], action[..., order, :])
    torch.testing.assert_close(permuted, original[..., order, :])


def test_joint_position_model_keeps_fixed_agent_order():
    model = build("joint")
    observation, action = sample()
    order = [2, 0, 1]
    with torch.no_grad():
        original = model(observation, action)
        permuted = model(observation[..., order, :], action[..., order, :])
    assert not torch.allclose(permuted, original[..., order, :])


def test_position_conditioners_are_capacity_matched():
    counts = {
        kind: sum(
            parameter.numel()
            for parameter in build(kind).conditioner.parameters()
        )
        for kind in ("independent", "joint", "relational")
    }
    assert max(counts.values()) - min(counts.values()) < 100


def test_position_metrics_compare_against_no_motion():
    target = torch.tensor([[0.03, 0.04], [0.0, 0.02]])
    perfect = error_summary(torch.zeros_like(target), target)
    persistence = error_summary(-target, target)
    assert perfect["coordinate_rmse"] == 0.0
    assert perfect["relative_mse_to_persistence"] == 0.0
    assert perfect["improvement_over_persistence_percent"] == 100.0
    assert persistence["relative_mse_to_persistence"] == 1.0
