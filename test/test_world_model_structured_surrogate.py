from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict

from benchmarl.environments import VmasTask
from examples.world_model.snapshot_restore import restore_state, snapshot_state
from examples.world_model.structured_surrogate import (
    FULL_SPEC,
    LEGACY_SPEC,
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


def test_full_structured_state_contains_linkage_pose_and_velocity():
    agents = torch.arange(2 * 2 * 6, dtype=torch.float32).reshape(2, 2, 6)
    entities = torch.arange(2 * 3 * 6, dtype=torch.float32).reshape(2, 3, 6) + 100
    goal = torch.tensor([[200.0, 201.0], [202.0, 203.0]])
    state = structured_state(agents, entities, goal=goal, profile="full32")
    assert state.shape == (2, FULL_SPEC.state_dim)
    torch.testing.assert_close(state[:, :12], agents.flatten(1))
    torch.testing.assert_close(state[:, 12:30], entities.flatten(1))
    torch.testing.assert_close(state[:, 30:32], goal)


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


def test_blockify_builds_full32_transition_without_dropping_links():
    count, steps, agents, action_dim = 1, 2, 2, 2
    agent_state = torch.randn(count, steps, agents, 6)
    package_state = torch.randn(count, steps, 3, 6)
    data = {
        "observation": torch.zeros(count, steps, agents, 6),
        "action": torch.zeros(count, steps, agents, action_dim),
        "reward": torch.zeros(count, steps, agents, 1),
        "valid": torch.ones(count, steps, dtype=torch.bool),
        "agent_state": agent_state,
        "next_agent_state": agent_state + 0.1,
        "package_state": package_state,
        "next_package_state": package_state + 0.1,
    }
    result = blockify(
        data, 0, 0, 0, 7, -1, action_block=2, state_profile="full32"
    )
    assert result["state"].shape == (1, FULL_SPEC.state_dim)
    assert result["next_state"].shape == (1, FULL_SPEC.state_dim)
    torch.testing.assert_close(result["state"][0, 12:30], package_state[0, 0].flatten())


@pytest.mark.parametrize("spec", [LEGACY_SPEC, FULL_SPEC], ids=lambda spec: spec.name)
def test_structured_surrogate_rollout_cost_is_finite_and_batched(spec):
    model = StructuredSurrogate(
        torch.zeros(spec.state_dim),
        torch.ones(spec.state_dim),
        torch.zeros(20),
        torch.ones(20),
        torch.zeros(spec.dynamic_dim),
        torch.ones(spec.dynamic_dim),
        torch.tensor(0.1),
        torch.tensor(1.0),
        torch.tensor(0.0),
        torch.tensor(1.0),
        hidden=16,
    )
    state = torch.zeros(2, spec.state_dim)
    state[:, -1] = 1.0
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


@pytest.mark.parametrize("spec", [LEGACY_SPEC, FULL_SPEC], ids=lambda spec: spec.name)
def test_matched_training_uses_fixed_epoch_and_optimizer_step_budget(spec):
    torch.manual_seed(7)
    train_rows, validation_rows = 12, 4
    rows = train_rows + validation_rows
    state = torch.randn(rows, spec.state_dim)
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
        full_before = live_structured_state(paired, "full32")
        assert torch.linalg.vector_norm(full_before[0] - full_before[1]) > 0

        action = torch.zeros(2, 2, 2)
        td = TensorDict({("agents", "action"): action}, batch_size=[2])
        paired.step(td)
        after = live_structured_state(paired)
        assert torch.linalg.vector_norm(after[0] - after[1]) > 1e-3
    finally:
        source.close()
        paired.close()


def test_paired_blockify_differs_only_in_the_state_representation():
    """A1.2 compares legacy14 against full32 on identical transitions.

    Nothing upstream of blockify depends on the state profile, so one raw
    trajectory must yield two banks whose every non-state column is bit
    identical. If this ever drifts, the A1.2 comparison stops isolating the
    Markov repair and starts confounding it with different data.
    """
    count, steps, agents, action_dim = 3, 4, 2, 2
    torch.manual_seed(5)
    agent_state = torch.randn(count, steps, agents, 6)
    package_state = torch.randn(count, steps, 3, 6)
    data = {
        "observation": torch.randn(count, steps, agents, 6),
        "action": torch.randn(count, steps, agents, action_dim),
        "reward": torch.randn(count, steps, agents, 1),
        "valid": torch.ones(count, steps, dtype=torch.bool),
        "agent_state": agent_state,
        "next_agent_state": agent_state + 0.1,
        "package_state": package_state,
        "next_package_state": package_state + 0.1,
    }
    split = torch.tensor([0, 1, 2])
    root_id = torch.tensor([11, 12, 13])
    common = dict(family=0, source=1, action_block=2)
    legacy = blockify(
        data, split, root_id=root_id, stage=-1, state_profile="legacy14", **common
    )
    full = blockify(
        data, split, root_id=root_id, stage=-1, state_profile="full32", **common
    )
    assert legacy["state"].shape[-1] == LEGACY_SPEC.state_dim
    assert full["state"].shape[-1] == FULL_SPEC.state_dim
    shared = [key for key in legacy if key not in ("state", "next_state")]
    assert set(shared) >= {
        "action",
        "collision",
        "minimum_clearance",
        "team_reward",
        "progress",
        "split",
        "root_id",
        "stage",
    }
    for key in shared:
        assert torch.equal(legacy[key], full[key]), key
    # The shared physical quantities must also agree across representations.
    torch.testing.assert_close(
        legacy["state"][:, LEGACY_SPEC.ball_position],
        full["state"][:, FULL_SPEC.ball_position],
    )
    torch.testing.assert_close(legacy["state"][:, 12:14], full["state"][:, 30:32])
