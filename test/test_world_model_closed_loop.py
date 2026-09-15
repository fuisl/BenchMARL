#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#
"""Closed-loop MPC with injected dynamics.

The point of the change under test is that the planner's dynamics become
swappable, so the two things worth pinning are that the oracle path still
behaves exactly as it did, and that an injected cost really is what drives the
loop -- a cost that is silently ignored would produce a plausible-looking
control result that means nothing.
"""

import pytest
import torch

from benchmarl.environments import VmasTask
from examples.world_model.cem import CEMConfig
from examples.world_model.closed_loop import (
    achieved_goals,
    goal_costs,
    reward_costs,
    summarize,
)
from examples.world_model.mpc import (
    agent_observations,
    evaluate_policy,
    MPCConfig,
    oracle_costs,
)
from examples.world_model.snapshot_restore import snapshot_state

STATES, SAMPLES = 2, 6
CEM = CEMConfig(horizon=2, num_samples=SAMPLES, num_elites=2, num_iters=2)
MPC = MPCConfig(receding_horizon=2, action_block=5)


def build():
    task = VmasTask.BUZZ_WIRE.get_from_yaml()
    task.config["max_steps"] = 20
    env = task.get_env_fun(STATES, True, 0, "cpu")()
    scratch = task.get_env_fun(STATES * SAMPLES, True, 0, "cpu")()
    env.set_seed(0)
    env.reset()
    scratch.reset()
    # Captured once: evaluate_policy leaves the env at terminal states, so a
    # snapshot taken per call would restore the end of the previous episode.
    return env, scratch, snapshot_state(env)


def run(env, scratch, initial, plan_costs=None):
    return evaluate_policy(
        env,
        initial,
        policy="mpc",
        generator=torch.Generator().manual_seed(3),
        cem_config=CEM,
        mpc_config=MPC,
        scratch_env=scratch,
        plan_costs=plan_costs,
    )


def test_explicit_oracle_cost_matches_the_default_path():
    """The refactor must not have changed what the oracle planner does."""
    env, scratch, initial = build()
    try:
        default, _ = run(env, scratch, initial)
        explicit, _ = run(env, scratch, initial, oracle_costs(scratch))
    finally:
        env.close()
        scratch.close()
    assert default == explicit


def test_injected_cost_is_used_and_receives_planner_shapes():
    """A cost that ranks plans by a fixed key must change the chosen actions."""
    env, scratch, initial = build()
    seen = {}

    def costs(snapshot, observation, candidates):
        seen["observation"] = observation.shape
        seen["candidates"] = candidates.shape
        # Prefer whichever candidate pushes agent 0's x action most negative,
        # a rule the oracle has no reason to agree with.
        return candidates[..., 0].sum(dim=-1)

    try:
        oracle, _ = run(env, scratch, initial)
        injected, _ = run(env, scratch, initial, costs)
    finally:
        env.close()
        scratch.close()

    assert seen, "the injected cost was never called"
    assert seen["observation"][0] == STATES
    assert seen["candidates"][:2] == torch.Size([STATES, SAMPLES])
    # Primitive steps, not blocks: the planner unpacks before scoring.
    assert seen["candidates"][2] == CEM.horizon * MPC.action_block
    assert [row["return"] for row in oracle] != [row["return"] for row in injected]


def test_agent_observations_match_the_scenario():
    env, _scratch, _initial = build()
    try:
        observation = agent_observations(env)
        expected = torch.stack(
            [
                env._env.scenario.observation(agent)
                for agent in env._env.world.agents
            ],
            dim=1,
        )
        assert torch.equal(observation, expected)
        assert observation.shape[0] == STATES
    finally:
        env.close()
        _scratch.close()


def test_achieved_goal_is_a_state_the_plan_actually_reached():
    """The LeWM goal must be reachable, so it has to come from a real rollout."""
    env, scratch, initial = build()
    try:
        plans = torch.rand(STATES, SAMPLES, 10, 4) * 2 - 1
        goals = achieved_goals(scratch, initial, plans, MPC.action_block)
        assert goals.shape[0] == STATES
        assert goals.shape[-1] == agent_observations(env).shape[-1]
        assert torch.isfinite(goals).all()
    finally:
        env.close()
        scratch.close()


def test_summary_groups_every_policy_separately():
    rows = [
        {
            "policy": name,
            "return": value,
            "team_return": 2 * value,
            "success": value > 0,
            "collision": False,
            "timeout": True,
            "length": 10,
            "final_goal_distance": 0.5,
            "goal_observation_distance": 1.5,
        }
        for name, value in (("a", 1.0), ("a", -1.0), ("b", 3.0))
    ]
    summary = summarize(rows)
    assert set(summary) == {"a", "b"}
    assert summary["a"]["episodes"] == 2 and summary["b"]["episodes"] == 1
    assert summary["a"]["success"]["count"] == 1
    assert summary["b"]["success"]["rate"] == 1.0


@pytest.mark.parametrize(
    "device",
    ["cpu"] + (["cuda"] if torch.cuda.is_available() else []),
)
def test_learned_costs_return_on_the_planner_device(device):
    """CEM gathers from candidates with topk indices taken from the cost.

    Both underlying helpers return on CPU because their own callers compare on
    CPU. Left alone, that hands a CUDA planner CPU indices -- the same class of
    device mismatch that killed job 1202, and invisible on a CPU-only run.
    """
    from test.test_world_model_models import ACT, AGENTS, OBS, build

    model = build("relational", wake=True).to(device)
    # The test model's action width is one block, so score with block size 1;
    # the device contract under test does not depend on the blocking.
    block, blocks = 1, 3
    observation = torch.randn(STATES, AGENTS, OBS, device=device)
    candidates = torch.rand(STATES, SAMPLES, blocks, AGENTS * ACT, device=device)
    goal = torch.randn(STATES, AGENTS, OBS, device=device)

    for cost_fn in (
        reward_costs(model, block, device),
        goal_costs(model, goal, block, device),
    ):
        cost = cost_fn(None, observation, candidates)
        assert cost.device.type == candidates.device.type
        assert cost.shape == (STATES, SAMPLES)
        assert torch.isfinite(cost).all()
