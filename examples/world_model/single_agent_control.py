# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the license in the LICENSE file in the root directory.
"""Single-agent control: can the learned world model plan navigation to its goal?

The closed-loop counterpart of ``self_rollout``. VMAS ``navigation`` with one
agent and collisions off; success is the scenario's own ``done()`` -- the agent
within its radius (0.1) of the goal -- inside the task's default 100 steps.

Planning follows LeWorldModel's shipped protocol as closely as this stack allows:

* CEM with LeWM's solver config (300 samples, 30 elites, 30 iterations, init
  std 1), horizon 5 action blocks of 5 primitive steps;
* cost = squared L2 between the FINAL imagined latent and the encoded goal
  observation (``JEPA.criterion``). The goal observation is the agent at its
  goal and at rest, ``[g_x, g_y, 0, 0, 0, 0]`` -- the navigation analogue of a
  goal image;
* real three-frame context. The episode starts at rest, so LeWM's
  start-of-episode convention (repeat the first frame, zero past actions) is
  physically exact here, not a fabrication;
* one deliberate deviation: replan after every block (``receding_horizon=1``)
  rather than LeWM's five. The reference history requires it, and this
  project measured the five-block cadence as the dominant control failure.

References on the same 64 fresh episodes: random actions; a PD controller
(100% success, median 33-56 steps on 256 starts); and oracle CEM, which plans
the SAME objective with the true simulator, so the gap between it and the learned
planner is attributable to the world model alone.
"""

import argparse
import json
from pathlib import Path

import torch
import yaml

from benchmarl.environments import VmasTask
from examples.world_model.cem import CEMConfig
from examples.world_model.closed_loop import goal_costs, reward_costs
from examples.world_model.model_input import ObservationBuilder, ReferenceHistory
from examples.world_model.mpc import MPCConfig, evaluate_policy, goal_oracle_costs
from examples.world_model.snapshot_restore import snapshot_state
from examples.world_model.train import load_model


def navigation_outcome(env):
    """(success, failure, distance): the scenario's own goal test; no failure mode."""
    scenario = env._env.scenario
    agent = env._env.world.agents[0]
    distance = torch.linalg.vector_norm(agent.state.pos - agent.goal.state.pos, dim=-1)
    reached = scenario.done()
    return reached, torch.zeros_like(reached), distance


def pd_policy(gain=4.0, damping=4.0):
    """Reactive reference: saturated PD toward the goal."""

    def act(env):
        agent = env._env.world.agents[0]
        force = gain * (agent.goal.state.pos - agent.state.pos) - damping * agent.state.vel
        return force.clamp(-1, 1).reshape(env.batch_size[0], 1, -1)

    return act


def goal_observation(env):
    """The agent at its goal and at rest, in observation space: (B,1,6)."""
    goal = env._env.world.agents[0].goal.state.pos
    zeros = torch.zeros_like(goal)
    return torch.cat([goal, zeros, zeros], dim=-1).unsqueeze(1)


def summarize(rows):
    success = torch.tensor([r["success"] for r in rows], dtype=torch.float)
    length = torch.tensor([r["length"] for r in rows], dtype=torch.float)
    final = torch.tensor([r["final_goal_distance"] for r in rows])
    best = torch.tensor([r["best_goal_distance"] for r in rows])
    return {
        "episodes": len(rows),
        "success_rate": float(success.mean()),
        "successes": int(success.sum()),
        "median_steps_to_success": float(length[success.bool()].median())
        if success.any()
        else None,
        "mean_final_distance": float(final.mean()),
        "mean_best_distance": float(best.mean()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("runs", type=Path, nargs="*", help="checkpoint run directories")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=64)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=9100)
    parser.add_argument("--objectives", default="goal,reward")
    parser.add_argument("--skip-oracle", action="store_true")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    task = VmasTask.NAVIGATION.get_from_yaml()
    task.config.update(n_agents=1, collisions=False, max_steps=args.max_steps)
    env = task.get_env_fun(args.episodes, True, args.seed, args.device)()
    env.set_seed(args.seed)
    env.reset()
    initial = snapshot_state(env)
    goal = goal_observation(env)
    cem = CEMConfig()  # LeWM: horizon 5, 300 samples, 30 elites, 30 iters, std 1
    mpc = MPCConfig(receding_horizon=1, action_block=5, warm_start=True)
    results, timings = {}, {}

    def run(label, policy, plan_costs=None, observe=None, scratch=None):
        generator = torch.Generator(device=args.device).manual_seed(args.seed)
        kwargs = {"observe": observe} if observe is not None else {}
        rows, timing, _ = evaluate_policy(
            env,
            initial,
            policy=policy,
            generator=generator,
            cem_config=cem,
            mpc_config=mpc,
            scratch_env=scratch,
            outcome_fn=navigation_outcome,
            plan_costs=plan_costs,
            goal_observation=goal,
            **kwargs,
        )
        # A reactive policy is a callable; rows must carry a name, not the function.
        rows = [{**row, "policy": label} for row in rows]
        results[label] = {"summary": summarize(rows), "rows": rows}
        timings[label] = timing["seconds"]
        s = results[label]["summary"]
        print(
            f"{label:40s} success {s['successes']:3d}/{s['episodes']} "
            f"({s['success_rate']:.3f})  median steps {s['median_steps_to_success']}  "
            f"final dist {s['mean_final_distance']:.3f}  best {s['mean_best_distance']:.3f}  "
            f"[{timing['seconds']:.0f}s]",
            flush=True,
        )

    try:
        run("random", "random")
        run("pd_heuristic", pd_policy())
        if not args.skip_oracle:
            scratch = task.get_env_fun(
                args.episodes * cem.num_samples, True, args.seed, args.device
            )()
            scratch.reset()
            run("oracle_cem_goal", "mpc", goal_oracle_costs(scratch, goal), scratch=scratch)
            scratch.close()
        for run_dir in args.runs:
            config = yaml.safe_load((run_dir / "resolved_config.yaml").read_text())
            model = load_model(run_dir / "model.pt", args.device)
            model.eval()
            if model.profile != "lewm_reference":
                raise ValueError("Single-agent control expects lewm_reference checkpoints")
            observe = ReferenceHistory(
                ObservationBuilder("observation", 3, int(model.obs_mean.shape[-1])),
                history_size=config["model"].get("history_size", 3),
                action_block=mpc.action_block,
            )
            name = run_dir.name
            if "goal" in args.objectives:
                run(f"lewm_goal|{name}", "mpc",
                    goal_costs(model, goal, mpc.action_block, args.device),
                    observe=observe)
            if "reward" in args.objectives:
                run(f"lewm_reward|{name}", "mpc",
                    reward_costs(model, mpc.action_block, args.device),
                    observe=observe)
            del model
            torch.cuda.empty_cache()
    finally:
        env.close()

    out = {
        "question": "Can the learned single-agent world model plan navigation to its goal?",
        "episodes": args.episodes,
        "max_steps": args.max_steps,
        "seed": args.seed,
        "cem": vars(cem),
        "receding_horizon": mpc.receding_horizon,
        "action_block": mpc.action_block,
        "timings_seconds": timings,
        "results": results,
    }
    (args.output / "control.json").write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {args.output / 'control.json'}", flush=True)


if __name__ == "__main__":
    main()
