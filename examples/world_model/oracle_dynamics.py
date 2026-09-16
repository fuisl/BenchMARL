#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#
"""Simulator-backed dynamics for oracle CEM-MPC (M2).

The oracle plays the role the learned world model will later play: given B
evaluation states and K candidate joint-action sequences, return one scalar cost
per candidate. Here the "model" is VMAS itself, so the costs are exact -- which
is what makes it the reference every learned model is measured against
(`experiment_plan.md` M2, and the oracle gap in the impact notes).

Dynamics and objective are kept as separate pieces, mirroring
``stable_worldmodel``'s ``ShootingCostEvaluator(model, objective)`` seam. The
point is not tidiness: the paper's claim is "same planner, same cost, different
world model", so the cost must be a thing that can be held fixed while the
dynamics underneath it are swapped for a learned model.

* dynamics: :func:`oracle_rollout` -- candidates in, rollout dict out.
* objective: :class:`NegativeTaskReward` or :class:`GoalDistance` -- rollout
  dict in, per-candidate cost out.
* composed: :func:`oracle_plan_costs` -- what a solver's ``cost_fn`` calls.

Cost convention (decided 2026-09-11): ``J = -sum_h r_h``, the true task reward,
for oracle *and* learned models alike, so the oracle gap compares planners
rather than two different objectives. Reward accumulation stops at episode
termination so a candidate cannot bank reward past a collision or a goal.

Run directly for the self-contained checks:
``python -m examples.world_model.oracle_dynamics``
"""

import torch

from benchmarl.environments import VmasTask

from examples.world_model.snapshot_restore import (
    agent_observations,
    broadcast_state,
    restore_state,
    snapshot_state,
)
from tensordict import TensorDict


@torch.no_grad()
def oracle_rollout(scratch_env, snapshot, candidates) -> dict:
    """Score B independent states with K plans each, without resetting live envs.

    The caller initializes scratch_env once. Its batch is B*K, with K adjacent
    copies of each source state. Candidates are (B,K,T,N*d_a), where T counts
    primitive VMAS steps, not action blocks. Outputs reward/live are (B,K,T).
    A terminal transition contributes reward; every later transition is masked.

    ``endpoint`` (B,K,N,O) is the state each candidate actually reached: the
    frame at its own first termination, or the final frame if it never
    terminated. The last frame of the batch is a different state -- finished
    slots keep being stepped while their neighbours run -- so an objective that
    scores where a plan ends must read this rather than the environment after
    the loop.
    """
    if candidates.ndim != 4:
        raise ValueError("Oracle candidates must have shape (B,K,T,N*d_a)")
    batch_size, n_candidates, horizon, action_dim = candidates.shape
    count = batch_size * n_candidates
    n_agents = len(scratch_env._env.world.agents)
    primitive_dim = scratch_env.full_action_spec_unbatched["agents", "action"].shape[-1]
    if (
        min(batch_size, n_candidates, horizon) < 1
        or action_dim != n_agents * primitive_dim
    ):
        raise ValueError("Invalid candidate horizon, batch or joint action dimension")
    if scratch_env.batch_size[0] != count or snapshot["steps"].shape != (batch_size,):
        raise ValueError("Oracle requires B source states and B*K scratch environments")

    indices = torch.arange(batch_size, device=candidates.device).repeat_interleave(
        n_candidates
    )
    broadcast_state(scratch_env, snapshot, source_indices=indices)
    live = ~scratch_env._env.done()
    actions = candidates.reshape(count, horizon, n_agents, primitive_dim)
    td = TensorDict({}, batch_size=[count], device=candidates.device)
    rewards, lives = [], []
    ended = ~live
    endpoint = agent_observations(scratch_env)
    for h in range(horizon):
        td.set(("agents", "action"), actions[:, h])
        td = scratch_env.step(td)["next"]
        rewards.append(td["agents", "reward"].sum(dim=1).squeeze(-1))
        lives.append(live)
        # Overwrite while the candidate has not yet terminated, so the last
        # write a candidate receives is the frame at its own termination.
        endpoint = torch.where(
            (~ended).reshape(-1, 1, 1), agent_observations(scratch_env), endpoint
        )
        ended = ended | td["done"].squeeze(-1)
        live = live & ~td["done"].squeeze(-1)

    return {
        "reward": torch.stack(rewards, dim=-1).reshape(
            batch_size, n_candidates, horizon
        ),
        "live": torch.stack(lives, dim=-1).reshape(batch_size, n_candidates, horizon),
        "endpoint": endpoint.reshape(batch_size, n_candidates, *endpoint.shape[-2:]),
    }


class NegativeTaskReward:
    """Cost = negative team reward summed over the horizon, masked after termination."""

    def __call__(self, rollout: dict) -> torch.Tensor:
        return -rollout["reward"].masked_fill(~rollout["live"], 0).sum(dim=-1)


class GoalDistance:
    """Cost = L2 from where the plan ends to a fixed goal observation.

    The objective the goal planners are actually optimizing, given the true
    simulator instead of a learned model. Job 1218 showed the reward oracle is
    not a ceiling for it: on Buzz Wire the reward oracle ended 0.839 from the
    generated goal against random's 0.256, because maximizing task reward moves
    away from a goal drawn from an arbitrary trajectory. Without this, the goal
    results have a random floor and no ceiling.

    ``goal_observation`` is (B,N,O) -- one goal per evaluation state, the same
    targets every learned policy is scored against.
    """

    def __init__(self, goal_observation: torch.Tensor, weight=None):
        if goal_observation.ndim != 3:
            raise ValueError("Goal observation must be (B,N,O)")
        self.goal_observation = goal_observation
        # (O,) of 0/1, or None. Must be the same weight EpisodeStats scores, or
        # this oracle bounds a different quantity than the one reported.
        self.weight = weight

    def __call__(self, rollout: dict) -> torch.Tensor:
        endpoint = rollout["endpoint"]
        if endpoint.shape[0] != self.goal_observation.shape[0]:
            raise ValueError(
                f"{endpoint.shape[0]} rollout states against "
                f"{self.goal_observation.shape[0]} goals"
            )
        # (B,K,N,O) against (B,1,N,O), flattened over agents and observation
        # so the distance matches EpisodeStats' own goal metric exactly.
        difference = endpoint - self.goal_observation.unsqueeze(1)
        if self.weight is not None:
            difference = difference * self.weight
        return difference.flatten(2).norm(dim=-1)


def oracle_plan_costs(
    scratch_env, snapshot, candidates, objective=None
) -> torch.Tensor:
    """Compose simulator dynamics and objective; return (B,K) costs."""
    objective = NegativeTaskReward() if objective is None else objective
    return objective(oracle_rollout(scratch_env, snapshot, candidates))


if __name__ == "__main__":
    torch.manual_seed(0)
    n_candidates, horizon = 64, 5

    task = VmasTask.BUZZ_WIRE.get_from_yaml()
    live_env = task.get_env_fun(
        num_envs=1, continuous_actions=True, seed=0, device="cpu"
    )()
    scratch_env = task.get_env_fun(
        num_envs=n_candidates, continuous_actions=True, seed=0, device="cpu"
    )()

    scratch_env.reset()
    td = live_env.reset()
    for _ in range(5):
        td = live_env.rand_action(td)
        td = live_env.step(td)
        td = td["next"]

    snapshot = snapshot_state(live_env)
    n_agents = len(live_env._env.world.agents)
    action_dim = live_env.full_action_spec_unbatched["agents", "action"].shape[-1]
    candidates = torch.rand(1, n_candidates, horizon, n_agents * action_dim) * 2 - 1

    costs = oracle_plan_costs(scratch_env, snapshot, candidates)
    assert costs.shape == (1, n_candidates), costs.shape
    assert (
        costs.std() > 0
    ), "all candidates scored identically -- branching is not working"
    print(
        f"PASS: {n_candidates} candidates scored, cost range "
        f"[{costs.min():.4f}, {costs.max():.4f}], std {costs.std():.4f}."
    )

    # Determinism: same snapshot + same candidates must reproduce the same costs.
    assert torch.equal(
        costs, oracle_plan_costs(scratch_env, snapshot, candidates)
    ), "oracle costs are not reproducible"
    print("PASS: identical costs when the same branch is re-scored.")

    # The objective is swappable without touching the dynamics: scoring only the
    # first step must disagree with scoring the whole horizon, or the seam is a
    # no-op and the cost is still fused into the rollout.
    class FirstStepRewardOnly(NegativeTaskReward):
        def __call__(self, rollout):
            return -(rollout["reward"][..., :1] * rollout["live"][..., :1]).sum(dim=-1)

    first_step = oracle_plan_costs(
        scratch_env, snapshot, candidates, objective=FirstStepRewardOnly()
    )
    assert not torch.equal(first_step, costs), "swapping the objective changed nothing"
    print("PASS: objective is swappable independently of the dynamics.")

    # Faithfulness: the oracle's cost for a candidate must equal what the *live*
    # env actually produces when that same action sequence is executed from the
    # same state. This is the property that makes it an oracle at all.
    best = int(costs.argmin())
    td_live = live_env.reset()
    restore_state(live_env, snapshot)
    live_reward, live_alive = torch.zeros(()), True
    for h in range(horizon):
        action = candidates[0, best, h].reshape(1, n_agents, action_dim)
        td_live.set(
            ("agents", "action"),
            action.expand(live_env.batch_size[0], n_agents, action_dim).clone(),
        )
        td_live = live_env.step(td_live)
        if live_alive:
            live_reward = live_reward + td_live["next", "agents", "reward"][0].sum()
            live_alive = not bool(td_live["next", "done"][0].item())
        td_live = td_live["next"]

    gap = (costs[0, best] - (-live_reward)).abs().item()
    assert gap < 1e-5, f"oracle disagrees with the live simulator by {gap}"
    print(
        f"PASS: oracle cost matches the live simulator for the selected plan "
        f"({costs[0, best]:.6f} vs {-live_reward:.6f})."
    )

    scratch_env.close()
    live_env.close()
