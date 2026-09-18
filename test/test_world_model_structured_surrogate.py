import torch

from examples.world_model.structured_surrogate import (
    StructuredSurrogate,
    blockify,
    structured_state,
    surrogate_cost,
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
