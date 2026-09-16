# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the license in the LICENSE file in the root directory.
"""Batched VMAS oracle MPC, in primitive action units."""

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import torch

from examples.world_model.cem import cem_plan, CEMConfig
from examples.world_model.oracle_dynamics import GoalDistance, oracle_plan_costs
from examples.world_model.snapshot_restore import (
    agent_observations,
    restore_state,
    snapshot_state,
)
from tensordict import TensorDict


@dataclass
class MPCConfig:
    receding_horizon: int = 5
    action_block: int = 5
    warm_start: bool = True

    def validate(self, horizon):
        if self.action_block < 1 or not 1 <= self.receding_horizon <= horizon:
            raise ValueError(
                "Require action_block >= 1 and 1 <= receding_horizon <= horizon"
            )


def unpack_actions(blocks, action_block):
    """(...,H,block*N*d_a) -> (...,H*block,N*d_a), preserving time order."""
    if action_block < 1 or blocks.shape[-1] % action_block:
        raise ValueError("Blocked action dimension must be divisible by action_block")
    return blocks.reshape(
        *blocks.shape[:-2],
        blocks.shape[-2] * action_block,
        blocks.shape[-1] // action_block,
    )


def shift_plan(plan, executed_blocks):
    """Retain unused blocks and append zeros, as in LeWM's warm start."""
    shifted = torch.zeros_like(plan)
    remaining = plan.shape[1] - executed_blocks
    if remaining > 0:
        shifted[:, :remaining] = plan[:, executed_blocks:]
    return shifted


def action_bounds(env):
    spec = env.full_action_spec_unbatched["agents", "action"]
    low, high = spec.space.low, spec.space.high
    if not (low == low.flatten()[0]).all() or not (high == high.flatten()[0]).all():
        raise ValueError("This pilot requires uniform joint-action bounds")
    return float(low.flatten()[0]), float(high.flatten()[0])


def scenario_heuristic(task_name, u_range):
    """VMAS's own hand-written policy for a scenario, or None if it ships none.

    ``random`` is uniform noise resampled every primitive step, which makes it a
    floor rather than a comparator: beating it shows a planner does something,
    not that it does something useful. Where VMAS ships a heuristic, that is the
    baseline a learned model has to be worth more than.

    Only Transport and Wheel have one. Buzz Wire and Dropout do not, so a
    heuristic column is simply absent there rather than faked.
    """
    import importlib

    module = importlib.import_module(f"vmas.scenarios.{task_name.split('/')[-1]}")
    policy_class = getattr(module, "HeuristicPolicy", None)
    if policy_class is None:
        return None
    # compute_action takes a u_range and clamps to it, so asymmetric or
    # non-unit bounds would silently rescale the policy rather than fail.
    if u_range <= 0:
        raise ValueError(f"Heuristic needs a positive action range, got {u_range}")
    policy = policy_class(continuous_action=True)

    def actions(env):
        """(B, 1, N*d_a) -- reactive, recomputed from the current observation."""
        observation = agent_observations(env)
        count, agents, features = observation.shape
        joint = policy.compute_action(
            observation.reshape(count * agents, features), u_range
        )
        return joint.reshape(count, 1, -1)

    return actions


def buzz_wire_outcome(env):
    scenario = env._env.scenario
    distance = torch.linalg.vector_norm(
        scenario.ball.state.pos - scenario.goal.state.pos, dim=-1
    )
    collided = scenario.collided.bool()
    return (distance <= 0.01) & ~collided, collided, distance


def goal_dimension_weight(task_name, obs_dim, device):
    """Which observation components a goal distance should score: (O,) of 0/1.

    Job 1229 measured why this is needed. Given a goal taken from the reward
    oracle -- the best controller Buzz Wire has -- the goal oracle reached it
    12/20 and ended at task distance 0.9536 with a 0.90 collision rate, against
    random's 1.0852 and 0.70. Pursuing the goal was worse than acting randomly.

    The reason is velocity. The goal is a full observation, so reaching it means
    arriving at that position *carrying that speed*, and on Buzz Wire that means
    driving hard through a narrow corridor. The agents observe their own
    position, velocity and offset to their goal; they never observe the ball or
    the wire, so nothing in the objective penalises the crash it causes. The
    planner follows the objective faithfully and destroys the task.

    Dropping velocity leaves the part of the goal that says *where to be* and
    removes the part that says *how fast to be going when you get there*, which
    the task never asked for.
    """
    if task_name == "vmas/buzz_wire":
        # [pos(2), vel(2), pos - goal_pos(2)]
        velocity = [2, 3]
    elif task_name == "vmas/transport":
        # [pos(2), vel(2)] then, per package, [pos - goal(2), pos - agent(2),
        # vel(2), on_goal(1)]
        packages = (obs_dim - 4) // 7
        velocity = [2, 3] + [4 + 7 * p + 4 + i for p in range(packages) for i in (0, 1)]
    elif task_name == "vmas/balance":
        # [pos(2), vel(2), pos - package(2), pos - line(2), package - goal(2),
        # package.vel(2), line.vel(2), line.ang_vel(1), line.rot(1)]
        velocity = [2, 3, 10, 11, 12, 13, 14]
    else:
        raise ValueError(f"No goal dimension layout recorded for {task_name}")
    weight = torch.ones(obs_dim, device=device)
    weight[velocity] = 0.0
    return weight


def transport_outcome(env):
    """All packages must overlap their goal; contacts are not terminal failures."""
    scenario = env._env.scenario
    goal = scenario.done()
    distance = (
        torch.stack(
            [
                torch.linalg.vector_norm(
                    package.state.pos - package.goal.state.pos, dim=-1
                )
                for package in scenario.packages
            ],
            dim=-1,
        )
        .max(dim=-1)
        .values
    )
    return goal, torch.zeros_like(goal), distance


def balance_outcome(env):
    """Package on its goal is success; line or package on the floor is failure.

    The scenario's ``done()`` is the union of the two, so reading it as success
    would score every fall as a completion. ``compute_on_the_ground`` is the
    scenario's own definition of the failure and is what ``reward`` calls each
    step, so calling it here reads the current state rather than a stale flag.
    """
    scenario = env._env.scenario
    distance = torch.linalg.vector_norm(
        scenario.package.state.pos - scenario.package.goal.state.pos, dim=-1
    )
    scenario.compute_on_the_ground()
    grounded = scenario.on_the_ground.bool()
    reached = scenario.world.is_overlapping(scenario.package, scenario.package.goal)
    # Failure takes precedence where both land on the same frame, as in Buzz
    # Wire: a package that arrives on the floor has not completed the task.
    return reached & ~grounded, grounded, distance


TASK_OUTCOMES = {
    "vmas/buzz_wire": buzz_wire_outcome,
    "vmas/transport": transport_outcome,
    "vmas/balance": balance_outcome,
}


def task_outcome(task_name):
    """The (success, failure, progress) reading a task is scored by.

    Every caller used to pick with `transport if ... else buzz_wire`, which gave
    Balance Buzz Wire's reading and an AttributeError on `scenario.ball`. A task
    with no recorded contract must fail here rather than inherit another's.
    """
    if task_name not in TASK_OUTCOMES:
        raise ValueError(
            f"No outcome contract recorded for {task_name}; "
            f"have {', '.join(sorted(TASK_OUTCOMES))}"
        )
    return TASK_OUTCOMES[task_name]


class EpisodeStats:
    """Latch first terminal outcomes; never count post-terminal rewards or goals."""

    def __init__(self, env, outcome_fn, goal_observation=None,
                 goal_threshold=None, goal_weight=None):
        """``goal_observation`` (B,N,O) scores a goal-reaching objective.

        The native ``success`` flag is the scenario's own ``done()``; goal
        reaching is a separate outcome with its own declared threshold, latched
        at the same terminal frame. Reading either from the environment after the
        loop would measure whichever state a finished slot drifted to while its
        neighbours kept running.
        """
        self.outcome_fn = outcome_fn
        self.goal_observation = goal_observation
        self.goal_threshold = goal_threshold
        # (O,) of 0/1, or None to score every component. The goal oracle must
        # optimize whatever this scores or it bounds nothing, so the same weight
        # goes to `GoalDistance`.
        self.goal_weight = goal_weight
        self.alive = ~env._env.done()
        if not self.alive.all():
            raise ValueError("Evaluation states must be nonterminal")
        self.team_return = torch.zeros_like(env._env.steps)
        self.length = torch.zeros_like(env._env.steps, dtype=torch.long)
        self.success = torch.zeros_like(self.alive)
        self.collision = torch.zeros_like(self.alive)
        self.timeout = torch.zeros_like(self.alive)
        self.final_distance = torch.zeros_like(env._env.steps)
        # The observation at the frame where this episode came closest to the
        # task goal, and the distance there. A goal drawn from a competent policy
        # needs this rather than the terminal frame: the Buzz Wire reward oracle
        # collides in 35% of episodes, so its terminal frames are the states it
        # crashed in, and aiming a planner at one would aim it at a crash. The
        # best frame is reachable for the same reason the terminal one is -- the
        # policy was there -- and it is the best that trajectory ever achieved.
        _reached, _collided, distance = outcome_fn(env)
        self.best_observation = agent_observations(env)
        self.best_distance = distance.clone()
        self.goal_distance = torch.full_like(env._env.steps, float("nan"))
        self.goal_reached = torch.zeros_like(self.alive)

    def _goal_distance(self, env):
        difference = agent_observations(env) - self.goal_observation
        if self.goal_weight is not None:
            difference = difference * self.goal_weight
        return difference.flatten(1).norm(dim=-1)

    def update(self, env, td):
        reward = td["agents", "reward"].sum(dim=1).squeeze(-1)
        if not torch.isfinite(reward[self.alive]).all():
            raise ValueError("Non-finite live episode reward")
        self.team_return += reward.masked_fill(~self.alive, 0)
        self.length += self.alive.long()
        ended = self.alive & td["done"].squeeze(-1)
        goal, collided, distance = self.outcome_fn(env)
        timed_out = env._env.steps >= env._env.max_steps
        if (ended & ~(collided | goal | timed_out)).any():
            raise ValueError("Unclassified task termination")
        self.success |= ended & goal
        self.collision |= ended & collided
        self.timeout |= ended & ~collided & ~goal & timed_out
        self.final_distance = torch.where(ended, distance, self.final_distance)
        # Only live frames are candidates: a finished slot keeps stepping beside
        # its neighbours and must not contribute the state it drifted to.
        improved = self.alive & (distance < self.best_distance)
        self.best_distance = torch.where(improved, distance, self.best_distance)
        self.best_observation = torch.where(
            improved.reshape(-1, 1, 1),
            agent_observations(env),
            self.best_observation,
        )
        if self.goal_observation is not None:
            # Latched at the terminal frame, and refreshed while still running
            # so an episode that never terminates is scored at its last state.
            current = self._goal_distance(env)
            live_or_ending = self.alive
            self.goal_distance = torch.where(
                live_or_ending, current, self.goal_distance
            )
            if self.goal_threshold is not None:
                self.goal_reached |= live_or_ending & (current <= self.goal_threshold)
        self.alive &= ~ended

    def rows(self, policy, n_agents):
        return [
            {
                "policy": policy,
                "episode": i,
                "return": float(self.team_return[i]) / n_agents,
                "team_return": float(self.team_return[i]),
                "success": bool(self.success[i]),
                "collision": bool(self.collision[i]),
                "timeout": bool(self.timeout[i]),
                "length": int(self.length[i]),
                "final_goal_distance": float(self.final_distance[i]),
                # The closest this episode ever came, which is what a goal is
                # drawn from and is worth reporting beside where it ended.
                "best_goal_distance": float(self.best_distance[i]),
                # Only when a goal was actually supplied. A NaN placeholder
                # would make identical rows compare unequal and would fail the
                # results write, which forbids non-finite JSON.
                **(
                    {
                        "goal_observation_distance": float(self.goal_distance[i]),
                        "goal_reached": bool(self.goal_reached[i]),
                    }
                    if self.goal_observation is not None
                    else {}
                ),
            }
            for i in range(len(self.alive))
        ]


def oracle_costs(scratch_env):
    """The default plan cost: roll the true simulator from the snapshot."""

    def costs(snapshot, observation, candidates):
        return oracle_plan_costs(scratch_env, snapshot, candidates)

    return costs


def goal_oracle_costs(scratch_env, goal_observation, goal_weight=None):
    """Plan cost for the goal objective, with the true simulator as dynamics.

    Same planner and same goals as the learned goal policies; only the world
    model differs. This is the ceiling the goal results were missing -- the
    reward oracle optimizes a different objective and is not one.
    """

    objective = GoalDistance(goal_observation, goal_weight)

    def costs(snapshot, observation, candidates):
        return oracle_plan_costs(
            scratch_env, snapshot, candidates, objective=objective
        )

    return costs


def synchronize(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


@torch.no_grad()
def evaluate_policy(
    env,
    initial_state,
    *,
    policy: str,
    generator: torch.Generator,
    cem_config: CEMConfig,
    mpc_config: MPCConfig,
    scratch_env=None,
    diagnostics_path: Path | None = None,
    outcome_fn,
    plan_costs=None,
    goal_observation=None,
    goal_threshold=None,
    goal_weight=None,
):
    """Evaluate exactly one episode per slot; return episode rows and timing.

    Finished slots receive zero actions and remain masked; there is no reset or
    replacement episode. VMAS batch slots are independent. The horizon counts
    blocks, while reward, timeout and termination always count primitive steps.

    ``policy`` is "random", "mpc", or a callable ``env -> (B,1,N*d_a)`` for a
    reactive baseline such as a VMAS scenario heuristic.

    ``plan_costs(snapshot, observation, candidates) -> (B,K)`` scores candidate
    plans. It defaults to the simulator oracle, which needs the snapshot and
    ignores the observation; a learned model is the reverse, planning from what
    the agents can actually see. Passing it is what makes this a test of the
    world model rather than of CEM.
    """
    mpc_config.validate(cem_config.horizon)
    if policy not in ("random", "mpc") and not callable(policy):
        raise ValueError(f"Unknown policy: {policy}")
    if env._env.max_steps is None:
        raise ValueError("Evaluation requires a finite task max_steps")
    if policy == "mpc" and plan_costs is None:
        if scratch_env is None or scratch_env is env:
            raise ValueError("Oracle MPC requires a separate scratch environment")
        plan_costs = oracle_costs(scratch_env)
    restore_state(env, initial_state)
    stats = EpisodeStats(
        env, outcome_fn, goal_observation, goal_threshold, goal_weight
    )
    batch_size = env.batch_size[0]
    n_agents = len(env._env.world.agents)
    action_dim = env.full_action_spec_unbatched["agents", "action"].shape[-1]
    joint_dim = n_agents * action_dim
    low, high = action_bounds(env)
    td = TensorDict({}, batch_size=[batch_size], device=env.device)
    mean = None
    decisions = []
    trajectory = []
    synchronize(env.device)
    start = perf_counter()

    while stats.alive.any():
        if policy == "mpc":
            snapshot = snapshot_state(env)

            observation = agent_observations(env)

            def cost_fn(candidates, snapshot=snapshot, observation=observation):
                return plan_costs(
                    snapshot,
                    observation,
                    unpack_actions(candidates, mpc_config.action_block),
                )

            synchronize(env.device)
            plan_start = perf_counter()
            result = cem_plan(
                cost_fn,
                action_dim=joint_dim * mpc_config.action_block,
                action_low=low,
                action_high=high,
                config=cem_config,
                batch_size=batch_size,
                init_mean=mean,
                device=env.device,
                generator=generator,
            )
            synchronize(env.device)
            decisions.append(
                {
                    "primitive_step": len(trajectory),
                    "active_episodes": int(stats.alive.sum()),
                    "seconds": perf_counter() - plan_start,
                    "best_seen_cost_mean": float(
                        result.best_cost_history[-1, stats.alive].mean()
                    ),
                    "elite_cost_mean": float(
                        result.elite_cost_history[-1, stats.alive].mean()
                    ),
                }
            )
            # Save a fixed-state candidate bank once, for subsequent rescoring.
            if diagnostics_path is not None and len(decisions) == 1:
                torch.save(
                    {
                        "snapshot": snapshot,
                        "candidates": unpack_actions(
                            result.candidates, mpc_config.action_block
                        ),
                        "costs": result.costs,
                        "final_mean": unpack_actions(
                            result.plan, mpc_config.action_block
                        ),
                        "elite_idx": result.elite_idx,
                        "elite_cost_history": result.elite_cost_history,
                        "best_cost_history": result.best_cost_history,
                    },
                    diagnostics_path / "candidate_bank.pt",
                )
            actions = unpack_actions(
                result.plan[:, : mpc_config.receding_horizon], mpc_config.action_block
            )
            mean = (
                shift_plan(result.plan, mpc_config.receding_horizon)
                if mpc_config.warm_start
                else None
            )
        elif policy == "random":
            actions = (
                torch.rand(
                    batch_size, 1, joint_dim, device=env.device, generator=generator
                )
                * (high - low)
                + low
            )
        else:
            # A reactive baseline: one step from the current observation, not a
            # plan. No snapshot, no search, so it costs nothing to run.
            actions = policy(env)
            if actions.shape != (batch_size, 1, joint_dim):
                raise ValueError(
                    f"Policy returned {tuple(actions.shape)}, expected "
                    f"{(batch_size, 1, joint_dim)}"
                )

        for action in actions.unbind(dim=1):
            alive = stats.alive.clone()
            action = action.masked_fill(~alive[:, None], 0).reshape(
                batch_size, n_agents, action_dim
            )
            td.set(("agents", "action"), action)
            td = env.step(td)["next"]
            stats.update(env, td)
            trajectory.append(
                {
                    "action": action.cpu(),
                    "live": alive.cpu(),
                    "reward": td["agents", "reward"].cpu(),
                    "done": td["done"].cpu(),
                }
            )
            if not stats.alive.any():
                break

    synchronize(env.device)
    timing = {"seconds": perf_counter() - start, "decisions": decisions}
    if diagnostics_path is not None:
        torch.save(trajectory, diagnostics_path / f"{policy}_trajectory.pt")
    return stats.rows(policy, n_agents), timing, stats.best_observation
