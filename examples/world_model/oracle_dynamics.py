#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#
"""Simulator-backed dynamics for oracle CEM-MPC (M2).

The oracle plays the role the learned world model will later play: given one
evaluation state and K candidate joint-action sequences, return one scalar cost
per candidate. Here the "model" is VMAS itself, so the costs are exact -- which
is what makes it the reference every learned model is measured against
(`experiment_plan.md` M2, and the oracle gap in the impact notes).

Dynamics and objective are kept as separate pieces, mirroring
``stable_worldmodel``'s ``ShootingCostEvaluator(model, objective)`` seam. The
point is not tidiness: the paper's claim is "same planner, same cost, different
world model", so the cost must be a thing that can be held fixed while the
dynamics underneath it are swapped for a learned model.

* dynamics: :func:`oracle_rollout` -- candidates in, rollout dict out.
* objective: :class:`NegativeTaskReward` -- rollout dict in, per-candidate cost out.
* composed: :func:`oracle_plan_costs` -- what a solver's ``cost_fn`` calls.

Cost convention (decided 2026-09-11): ``J = -sum_h r_h``, the true task reward,
for oracle *and* learned models alike, so the oracle gap compares planners
rather than two different objectives. Reward accumulation stops at episode
termination so a candidate cannot bank reward past a collision or a goal.

Run directly for the self-contained checks:
``python examples/world_model/oracle_dynamics.py``
"""

import torch

from benchmarl.environments import VmasTask
from snapshot_restore import broadcast_state, restore_state, snapshot_state


def oracle_rollout(scratch_env, snapshot, candidates, env_index: int = 0) -> dict:
    """Roll candidate joint-action plans through the real simulator.

    Args:
        scratch_env: a VMAS env whose batch size equals the candidate count K.
            Must not be the live evaluation env -- its state gets overwritten.
        snapshot: from `snapshot_state`, the state every candidate branches from.
        candidates: ``(K, H, N * d_a)`` flattened joint actions, matching the
            layout `cem.cem_plan` optimises.
        env_index: which slot of `snapshot` to branch from.

    Returns:
        Rollout dict with ``reward`` ``(K, H)`` (team reward per step) and
        ``live`` ``(K, H)`` (False once that candidate's episode has ended).
        Objectives needing more than reward add their keys here.
    """
    n_candidates, horizon, _ = candidates.shape
    n_agents = len(scratch_env._env.world.agents)
    if scratch_env.batch_size[0] != n_candidates:
        raise ValueError(
            f"scratch_env batch {scratch_env.batch_size[0]} != candidates {n_candidates}"
        )

    td = scratch_env.reset()
    broadcast_state(scratch_env, snapshot, env_index)

    rewards, lives = [], []
    live = torch.ones(n_candidates, dtype=torch.bool, device=candidates.device)

    for h in range(horizon):
        td.set(
            ("agents", "action"), candidates[:, h].reshape(n_candidates, n_agents, -1)
        )
        td = scratch_env.step(td)
        rewards.append(td["next", "agents", "reward"].sum(dim=1).squeeze(-1))
        lives.append(live.clone())
        live = live & ~td["next", "done"].squeeze(-1)
        td = td["next"]

    return {"reward": torch.stack(rewards, dim=1), "live": torch.stack(lives, dim=1)}


class NegativeTaskReward:
    """Cost = negative team reward summed over the horizon, masked after termination."""

    def __call__(self, rollout: dict) -> torch.Tensor:
        return -(rollout["reward"] * rollout["live"]).sum(dim=1)


def oracle_plan_costs(
    scratch_env,
    snapshot,
    candidates,
    objective=None,
    env_index: int = 0,
) -> torch.Tensor:
    """Score candidate plans with the simulator: dynamics then objective.

    Returns ``(K,)`` costs, lower is better -- the signature a solver's
    ``cost_fn`` expects once the batch axis is added.
    """
    objective = NegativeTaskReward() if objective is None else objective
    return objective(oracle_rollout(scratch_env, snapshot, candidates, env_index))


if __name__ == "__main__":
    torch.manual_seed(0)
    n_candidates, horizon = 64, 5

    task = VmasTask.GIVE_WAY.get_from_yaml()
    live_env = task.get_env_fun(
        num_envs=4, continuous_actions=True, seed=0, device="cpu"
    )()
    scratch_env = task.get_env_fun(
        num_envs=n_candidates, continuous_actions=True, seed=0, device="cpu"
    )()

    td = live_env.reset()
    for _ in range(5):
        td = live_env.rand_action(td)
        td = live_env.step(td)
        td = td["next"]

    snapshot = snapshot_state(live_env)
    n_agents = len(live_env._env.world.agents)
    action_dim = live_env.full_action_spec_unbatched["agents", "action"].shape[-1]
    candidates = torch.rand(n_candidates, horizon, n_agents * action_dim) * 2 - 1

    costs = oracle_plan_costs(scratch_env, snapshot, candidates)
    assert costs.shape == (n_candidates,), costs.shape
    assert costs.std() > 0, "all candidates scored identically -- branching is not working"
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
            return -(rollout["reward"][:, :1] * rollout["live"][:, :1]).sum(dim=1)

    first_step = oracle_plan_costs(
        scratch_env, snapshot, candidates, objective=FirstStepRewardOnly()
    )
    assert not torch.equal(first_step, costs), "swapping the objective changed nothing"
    print("PASS: objective is swappable independently of the dynamics.")

    # Faithfulness: the oracle's cost for a candidate must equal what the *live*
    # env actually produces when that same action sequence is executed from the
    # same state. This is the property that makes it an oracle at all.
    best = int(costs.argmin())
    restore_state(live_env, snapshot)
    td_live = live_env.reset()
    restore_state(live_env, snapshot)
    live_reward, live_alive = torch.zeros(()), True
    for h in range(horizon):
        action = candidates[best, h].reshape(1, n_agents, action_dim)
        td_live.set(
            ("agents", "action"),
            action.expand(live_env.batch_size[0], n_agents, action_dim).clone(),
        )
        td_live = live_env.step(td_live)
        if live_alive:
            live_reward = live_reward + td_live["next", "agents", "reward"][0].sum()
            live_alive = not bool(td_live["next", "done"][0].item())
        td_live = td_live["next"]

    gap = (costs[best] - (-live_reward)).abs().item()
    assert gap < 1e-5, f"oracle disagrees with the live simulator by {gap}"
    print(
        f"PASS: oracle cost matches the live simulator for the selected plan "
        f"({costs[best]:.6f} vs {-live_reward:.6f})."
    )
