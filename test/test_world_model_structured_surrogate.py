from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict

from benchmarl.environments import VmasTask
from examples.world_model.snapshot_restore import restore_state, snapshot_state
from examples.world_model.structured_surrogate import (
    StructuredSurrogate,
    blockify,
    live_structured_state,
    structured_state,
    surrogate_cost,
    train_surrogate,
    training_sampling_weights,
    wire_clearance,
)


def test_structured_state_contains_agents_ball_and_goal_in_declared_order():
    agents = torch.zeros(2, 2, 6)
    agents[..., :4] = torch.arange(16).reshape(2, 2, 4)
    package = torch.zeros(2, 1, 6)
    package[..., :4] = torch.tensor([[[20.0, 21.0, 22.0, 23.0]]])
    goal = torch.tensor([[30.0, 31.0], [32.0, 33.0]])
    state = structured_state(agents, package, goal=goal)
    assert state.shape == (2, 14)
    torch.testing.assert_close(state[:, :8], agents[..., :4].flatten(1))
    torch.testing.assert_close(state[:, 8:12], package[:, 0, :4])
    torch.testing.assert_close(state[:, 12:14], goal)


def test_blockify_preserves_action_order_and_reward_decomposition():
    count, steps, agents, action_dim = 1, 2, 2, 2
    agent_state = torch.zeros(count, steps, agents, 6)
    next_agent_state = agent_state.clone()
    package_state = torch.zeros(count, steps, 1, 6)
    next_package_state = package_state.clone()
    package_state[0, :, 0, 1] = torch.tensor([0.0, 0.1])
    next_package_state[0, :, 0, 1] = torch.tensor([0.1, 0.2])
    observation = torch.zeros(count, steps, agents, 6)
    observation[..., 5] = -1.0
    action = torch.arange(steps * agents * action_dim).reshape(
        count, steps, agents, action_dim
    ).float()
    reward = torch.tensor([[[[0.05], [0.05]], [[-9.95], [-9.95]]]])
    data = {
        "observation": observation,
        "action": action,
        "reward": reward,
        "valid": torch.ones(count, steps, dtype=torch.bool),
        "agent_state": agent_state,
        "next_agent_state": next_agent_state,
        "package_state": package_state,
        "next_package_state": next_package_state,
    }
    result = blockify(data, 0, 0, 0, 7, -1, action_block=2)
    assert result["state"].shape == (1, 14)
    torch.testing.assert_close(result["action"][0, 0], action[0, :, 0].flatten())
    torch.testing.assert_close(result["action"][0, 1], action[0, :, 1].flatten())
    torch.testing.assert_close(
        2 * result["progress"] - result["collision_cost"], result["team_reward"]
    )
    assert result["collision"].item()
    assert result["split"].item() == 0
    assert result["root_id"].item() == 7


def test_structured_surrogate_rollout_cost_is_finite_and_batched():
    model = StructuredSurrogate(
        torch.zeros(14),
        torch.ones(14),
        torch.zeros(20),
        torch.ones(20),
        torch.zeros(12),
        torch.ones(12),
        torch.tensor(0.1),
        torch.tensor(1.0),
        torch.tensor(0.0),
        torch.tensor(1.0),
        hidden=16,
    )
    state = torch.zeros(2, 14)
    state[:, 13] = 1.0
    candidates = torch.zeros(2, 3, 25, 4)
    for objective in ("probability", "penalty", "clearance"):
        cost = surrogate_cost(model, state, candidates, 5, objective)
        assert cost.shape == (2, 3)
        assert torch.isfinite(cost).all()


def test_clearance_is_negative_outside_the_wire():
    clearance = wire_clearance(torch.tensor([[0.0, 0.0], [0.1, 0.0]]))
    torch.testing.assert_close(clearance, torch.tensor([0.095, -0.005]))


@pytest.mark.parametrize("augmentation_count", [2, 20])
def test_base_augmentation_sampler_assigns_equal_expected_mass(
    augmentation_count,
):
    family = torch.tensor([0] * 7 + [1] * 3 + [2] * 5 + [3] * augmentation_count)
    data = {"family": family}
    indices = torch.arange(family.numel())
    weights, mass = training_sampling_weights(
        data,
        indices,
        scheme="base_augmentation",
        augmentation_family=3,
    )
    augmentation = family == 3
    expected = torch.tensor(0.5, dtype=weights.dtype)
    torch.testing.assert_close(weights[~augmentation].sum(), expected)
    torch.testing.assert_close(weights[augmentation].sum(), expected)
    assert mass == {"base": 0.5, "augmentation": 0.5}


def test_matched_training_uses_fixed_epoch_and_optimizer_step_budget():
    torch.manual_seed(7)
    train_rows, validation_rows = 12, 4
    rows = train_rows + validation_rows
    state = torch.randn(rows, 14)
    action = torch.randn(rows, 2, 4)
    data = {
        "state": state,
        "action": action,
        "next_state": state + 0.05 * torch.randn_like(state),
        "dynamics_valid": torch.ones(rows, dtype=torch.bool),
        "minimum_clearance": torch.randn(rows),
        "collision": torch.arange(rows) % 2 == 0,
        "collision_cost": torch.rand(rows),
        "family": torch.tensor([0] * 6 + [3] * 6 + [0] * validation_rows),
        "split": torch.tensor([0] * train_rows + [1] * validation_rows),
    }
    args = SimpleNamespace(
        hidden=8,
        device="cpu",
        batch_size=4,
        learning_rate=3e-4,
        weight_decay=1e-4,
        epochs=3,
        patience=1,
    )
    _model, fit = train_surrogate(
        data,
        "full",
        11,
        args,
        sampling_scheme="base_augmentation",
        augmentation_family=3,
        samples_per_epoch=8,
        fixed_epochs=True,
    )
    assert fit["epochs"] == 3
    assert fit["optimizer_steps"] == 6
    assert fit["samples_per_epoch"] == 8
    assert fit["expected_sampling_mass"] == {"base": 0.5, "augmentation": 0.5}


def test_legacy_14d_state_is_provably_non_markov_when_link_state_differs():
    """Same 14-D state and action diverge when an omitted joint body differs."""
    task = VmasTask.BUZZ_WIRE.get_from_yaml()
    source = task.get_env_fun(1, True, 0, "cpu")()
    paired = task.get_env_fun(2, True, 0, "cpu")()
    try:
        source.set_seed(1)
        source.reset()
        snapshot = snapshot_state(source)

        def duplicate(value):
            if isinstance(value, dict):
                return {key: duplicate(item) for key, item in value.items()}
            return value.repeat_interleave(2, dim=0)

        counterexample = duplicate(snapshot)
        link = counterexample["entities"]["joint agent_0 ball"]["state"]
        link["_vel"][1, 0] += 0.5

        paired.reset()
        restore_state(paired, counterexample)
        before = live_structured_state(paired)
        torch.testing.assert_close(before[0], before[1], atol=0, rtol=0)

        action = torch.zeros(2, 2, 2)
        td = TensorDict({("agents", "action"): action}, batch_size=[2])
        paired.step(td)
        after = live_structured_state(paired)
        assert torch.linalg.vector_norm(after[0] - after[1]) > 1e-3
    finally:
        source.close()
        paired.close()
