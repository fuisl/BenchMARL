# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the license in the LICENSE file in the root directory.
"""Contract checks for the three defects the 2026-09-16 review reproduced.

Each one produced a number that looked like a result. The existing suite passed
throughout, because every test named Transport and Buzz Wire explicitly and
every defect was in what happens to a task that is neither:

1. the collector built its heuristic from `vmas.scenarios.transport` whatever
   scenario it was collecting, so job 1233's Balance "competent" trajectories
   are Transport's policy driving Balance agents;
2. the closed loop fell back to `buzz_wire_outcome` for every non-Transport
   task, which reads `scenario.ball` -- an attribute Balance does not have;
3. the horizon evaluator sized its context from the positional embedding rather
   than from the positions training actually supervises, which put a never-
   trained slot in the rollout at h=6 and produced the reported cliff.

These check the behaviour, not the fix: a future task added without an outcome
contract, or a second predictor with different supervision, fails here.
"""

import importlib

import pytest
import torch

from benchmarl.environments import VmasTask
from examples.world_model.collect import rollout_actions
from examples.world_model.horizon_rollout import trained_frames
from examples.world_model.models import MultiAgentWorldModel
from examples.world_model.mpc import (
    balance_outcome,
    buzz_wire_outcome,
    task_outcome,
    transport_outcome,
)
from examples.world_model.snapshot_restore import snapshot_state


@pytest.fixture
def make_env():
    envs = []

    def make(task_enum, count=3, max_steps=10):
        task = task_enum.get_from_yaml()
        task.config["max_steps"] = max_steps
        env = task.get_env_fun(count, True, 0, "cpu")()
        env.reset()
        envs.append(env)
        return env

    yield make
    for env in envs:
        env.close()


@pytest.mark.parametrize(
    "task_enum,scenario_name",
    [(VmasTask.TRANSPORT, "transport"), (VmasTask.BALANCE, "balance")],
)
def test_collected_heuristic_actions_come_from_the_collected_scenario(
    make_env, task_enum, scenario_name
):
    """The stored actions must match *this* scenario's policy, not another's.

    Recomputed over every stored action, which is how the review detected the
    original defect: Transport's policy reproduced Balance's saved actions to
    1.46e-9 while Balance's own differed by 1.0795 on average.
    """
    env = make_env(task_enum, count=2, max_steps=4)
    module = importlib.import_module(f"vmas.scenarios.{scenario_name}")
    heuristic = module.HeuristicPolicy(continuous_action=True)

    def policy(observation):
        count, agents, features = observation.shape
        return heuristic.compute_action(
            observation.reshape(count * agents, features), 1.0
        ).reshape(count, agents, -1)

    agents = len(env._env.world.agents)
    action_dim = env.full_action_spec_unbatched["agents", "action"].shape[-1]
    data, _ = rollout_actions(
        env,
        snapshot_state(env),
        torch.zeros(2, 4, agents, action_dim),
        policy=policy,
    )
    for step in range(4):
        torch.testing.assert_close(
            data["action"][:, step], policy(data["observation"][:, step])
        )
    # A different scenario's policy must be visibly wrong here, or this test
    # would pass for the bank that was actually collected.
    other = "balance" if scenario_name == "transport" else "transport"
    other_policy = importlib.import_module(
        f"vmas.scenarios.{other}"
    ).HeuristicPolicy(continuous_action=True)
    count, _, features = data["observation"][:, 0].shape
    mismatch = other_policy.compute_action(
        data["observation"][:, 0].reshape(count * agents, features), 1.0
    ).reshape(count, agents, -1)
    assert not torch.allclose(data["action"][:, 0], mismatch, atol=1e-5)


def test_balance_outcome_separates_completion_from_falling(make_env):
    """`done()` is the union of both, so it cannot stand in for success."""
    env = make_env(VmasTask.BALANCE, count=3)
    scenario = env._env.scenario
    # Slot 0 completes, slot 1 drops the package, slot 2 does neither.
    scenario.package.state.pos[0] = scenario.package.goal.state.pos[0]
    scenario.package.state.pos[1, 1] = scenario.floor.state.pos[1, 1]
    reached, grounded, distance = balance_outcome(env)
    assert reached.tolist() == [True, False, False]
    assert grounded.tolist() == [False, True, False]
    assert distance[0] < distance[2]
    # Whatever the scenario terminates on must be classified by one of the two,
    # or `EpisodeStats.update` raises "Unclassified task termination".
    assert ((reached | grounded) == env._env.done()).all()
    # Both at once is a fall, not a completion: precedence is declared, not
    # left to whichever branch is evaluated first.
    scenario.package.state.pos[0] = scenario.package.goal.state.pos[0]
    scenario.package.state.pos[0, 1] = scenario.floor.state.pos[0, 1]
    reached, grounded, _ = balance_outcome(env)
    assert not reached[0] and grounded[0]


def test_every_task_names_its_own_outcome_contract():
    """A task with no contract must fail rather than inherit another task's."""
    assert task_outcome("vmas/balance") is balance_outcome
    assert task_outcome("vmas/transport") is transport_outcome
    assert task_outcome("vmas/buzz_wire") is buzz_wire_outcome
    with pytest.raises(ValueError, match="No outcome contract"):
        task_outcome("vmas/wheel")


def test_rollout_context_covers_only_supervised_positions():
    """`trained_frames` must equal the positions a training step reaches.

    Derived from a real backward pass rather than asserted from the constant:
    the last allocated position receives no prediction gradient because
    `dynamics_losses` predicts from `latent[:, :-1]`.
    """
    frames = 4
    model = MultiAgentWorldModel(
        kind="joint",
        agents=2,
        obs_dim=6,
        action_dim=2,
        dim=8,
        hidden_dim=16,
        frames=frames,
        depth=1,
        heads=2,
        dim_head=4,
        mlp_dim=16,
    )
    latent = model.encode(torch.randn(3, frames, 2, 6))
    predicted = model.predict(latent[:, :-1], torch.randn(3, frames - 1, 2, 2))
    predicted.square().sum().backward()
    reached = [
        position
        for position, grad in enumerate(model.predictor.pos_embedding.grad[0])
        if grad.abs().sum() > 0
    ]
    assert reached == list(range(frames - 1))
    assert trained_frames(model) == frames - 1
