#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#
"""M5's remaining third: closed-loop MPC driven by a learned world model.

Everything measured so far is open-loop. C7, plan ranking and the goal-cost
prototype all score plans the planner never executes, so the project has no
evidence about control -- the last link in the chain the proposal names,
``counterfactual accuracy -> plan ranking -> closed-loop control``.

This runs the same CEM-MPC loop the oracle already validated, on the same states,
with the same horizon and search budget, and changes exactly one thing: the
dynamics the planner rolls. Four policies share those states:

    random    the floor
    oracle    the true simulator as dynamics -- the ceiling, and the oracle gap
    reward    learned latents + the learned reward/termination readout,
              scoring J = -sum_t sum_i r_i,t, the objective the plan committed to
    goal      learned latents + LeWM's terminal latent-goal distance

The two learned costs are reported separately on purpose. The experiment plan
fixes task reward as the planning objective and defers latent-goal scoring, so
``goal`` is a labelled alternative, not a substitute. It is here because job 1203
showed the readout is where the chain breaks on Buzz Wire -- anti-correlated at
rho ~ -0.25 even given the simulator's own latents, because the reward depends on
the ball and no agent observes the ball -- which predicts that ``reward`` fails
there and ``goal``, needing no readout, does not. That is a real prediction this
run can falsify.

Goal semantics follow LeWM rather than the task: the goal is an *achieved*
observation, reached by rolling a random plan from the same state, so it is
reachable by construction. ``goal`` therefore measures goal-reaching, not task
success, and its success column is distance-thresholded rather than the
scenario's ``done()``. Only ``reward`` and ``oracle`` are comparable on task
success.
"""

from pathlib import Path

import numpy as np
import torch

from examples.world_model.cem import CEMConfig
from examples.world_model.goal_planning import goal_plan_costs, terminal_observations
from examples.world_model.metrics import mean_interval, success_interval
from examples.world_model.mpc import (
    agent_observations,
    buzz_wire_outcome,
    evaluate_policy,
    MPCConfig,
    transport_outcome,
)
from examples.world_model.plan_ranking import model_costs, select_anchor_states
from examples.world_model.readout_diagnostic import simulate


def reward_costs(model, action_block, device):
    """Plan cost from the learned readout: J = -sum_t sum_i r_i,t.

    The oracle's objective exactly, with the rewards coming from the readout
    applied to rolled-out latents instead of from the simulator, so the only
    thing that differs between the two planners is the dynamics.
    """

    def costs(snapshot, observation, candidates):
        # Both helpers return on CPU, because their own callers compare on CPU.
        # CEM picks elites with topk and gathers from candidates living on the
        # planner's device, so the cost has to come back to that device.
        return model_costs(model, observation, candidates, action_block, device).to(
            candidates.device
        )

    return costs


def goal_costs(model, goal_observation, action_block, device, history_size=3):
    """Plan cost from LeWM's terminal latent-goal distance.

    Needs neither a reward head nor a termination head, which is the whole
    reason it is worth measuring separately here.
    """

    def costs(snapshot, observation, candidates):
        return goal_plan_costs(
            model,
            observation,
            goal_observation,
            candidates,
            action_block,
            device,
            history_size,
        ).to(candidates.device)

    return costs


@torch.no_grad()
def achieved_goals(scratch_env, snapshot, candidates, action_block):
    """Terminal observation of the first plan per evaluation state: (B,N,O).

    LeWM draws its goal from a trajectory rather than from the task definition,
    so the goal is reachable and the planning question is well posed. Rolling the
    simulator here is a property of the evaluation *protocol*, not of the planner:
    the goal is fixed before planning starts and the learned planner never touches
    the simulator.
    """
    _, block_valid, observation = simulate(
        scratch_env, snapshot, candidates, action_block
    )
    # `simulate` returns on CPU because its own caller compares on CPU. The goal
    # is then differenced against live observations, so hand it back on the
    # environment's device rather than leaving that to every caller.
    goals = terminal_observations(observation, block_valid)[:, 0]
    return goals.to(scratch_env.device)


def persist(args, manifest, states, chosen, cem_config, rows, timings):
    """Write everything scored so far; return the summary.

    Separated from ``main`` so the evaluation loop can call it after each policy
    rather than only once at the end.
    """
    import json

    summary = summarize(rows, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "task": manifest["task_name"],
                "states": states,
                "anchor_ids": chosen.tolist(),
                "cem": vars(cem_config),
                "summary": summary,
                "timing": timings,
                "rows": rows,
            },
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )
    return summary


def summarize(rows, seed=0):
    """Per-policy episode summary, reusing the oracle pilot's interval helpers.

    ``goal_observation_distance`` is distance to the LeWM target; the task's own
    ``final_goal_distance`` is distance to the scenario goal. They are different
    objectives and both are reported for every policy.
    """
    rng = np.random.default_rng(seed)
    summary = {}
    for policy in dict.fromkeys(row["policy"] for row in rows):
        episodes = [row for row in rows if row["policy"] == policy]
        summary[policy] = {
            "episodes": len(episodes),
            "return": mean_interval([r["return"] for r in episodes], rng),
            "team_return": mean_interval([r["team_return"] for r in episodes], rng),
            "success": success_interval([r["success"] for r in episodes]),
            "collision_rate": float(np.mean([r["collision"] for r in episodes])),
            "timeout_rate": float(np.mean([r["timeout"] for r in episodes])),
            "final_goal_distance": mean_interval(
                [r["final_goal_distance"] for r in episodes], rng
            ),
            "goal_observation_distance": mean_interval(
                [r["goal_observation_distance"] for r in episodes], rng
            ),
            "episode_length": mean_interval([r["length"] for r in episodes], rng),
        }
    return summary


def main():
    """Closed-loop MPC on shared states: random, oracle, and each learned model."""
    import argparse
    import json

    import yaml

    from benchmarl.environments import VmasTask
    from examples.world_model.train import load_model

    parser = argparse.ArgumentParser()
    parser.add_argument("runs", type=Path, help="sweep directory of trained models")
    parser.add_argument("--data", type=Path, required=True, help="bank with manifest")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--states", type=int, default=20)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=7100)
    parser.add_argument("--num-samples", type=int, default=300)
    parser.add_argument("--num-iters", type=int, default=30)
    parser.add_argument(
        "--num-elites",
        type=int,
        default=None,
        help="default: LeWM's 10%% of num_samples (30 of 300)",
    )
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--goal-offset", type=int, default=5, help="blocks ahead")
    parser.add_argument("--skip-oracle", action="store_true")
    parser.add_argument(
        "--seeds",
        default=None,
        help="comma-separated training seeds to score; default all. Closed-loop "
        "costs minutes per checkpoint, so the full grid rarely fits.",
    )
    parser.add_argument(
        "--max-runs",
        type=int,
        default=None,
        help="score only the first N checkpoints, for a pre-sweep smoke",
    )
    args = parser.parse_args()

    manifest = json.loads((args.data / "manifest.json").read_text())
    block = manifest["action_block"]
    joint_dim = torch.as_tensor(manifest["action_low"]).numel()
    task = VmasTask[manifest["task_name"].split("/")[-1].upper()].get_from_yaml()
    cem_config = CEMConfig(
        horizon=args.horizon,
        num_samples=args.num_samples,
        num_iters=args.num_iters,
        # LeWM keeps 30 of 300; hold the ratio so a smaller budget stays a
        # smaller version of the same search rather than a greedier one.
        num_elites=args.num_elites or max(1, round(0.1 * args.num_samples)),
    )
    mpc_config = MPCConfig(receding_horizon=args.horizon, action_block=block)

    # Evaluation states come from the bank's test split, so no learned model has
    # trained on the states it is asked to control.
    anchors = torch.load(
        args.data / "anchors.pt", map_location="cpu", weights_only=True
    )
    test = (anchors["split"] == 2).nonzero(as_tuple=True)[0]
    generator = torch.Generator().manual_seed(args.seed)
    chosen = test[torch.randperm(test.numel(), generator=generator)[: args.states]]
    states = chosen.numel()
    initial_state = select_anchor_states(anchors, chosen, args.device)

    env = task.get_env_fun(states, True, 0, args.device)()
    scratch = task.get_env_fun(states * args.num_samples, True, 0, args.device)()
    env.reset()
    scratch.reset()

    try:
        # One random plan per state supplies the LeWM goal, drawn before any
        # planning so every policy is scored against the same fixed targets.
        goal_plans = (
            torch.rand(
                states,
                args.num_samples,
                args.goal_offset * block,
                joint_dim,
                generator=generator,
            )
            * 2
            - 1
        ).to(args.device)
        goal_observation = achieved_goals(scratch, initial_state, goal_plans, block)

        outcome_fn = (
            transport_outcome
            if manifest["task_name"] == "vmas/transport"
            else buzz_wire_outcome
        )
        rows, timings = [], {}

        def run(label, policy, plan_costs=None):
            planner = torch.Generator(device=args.device).manual_seed(args.seed)
            episodes, timing = evaluate_policy(
                env,
                initial_state,
                policy=policy,
                generator=planner,
                cem_config=cem_config,
                mpc_config=mpc_config,
                scratch_env=scratch,
                outcome_fn=outcome_fn,
                plan_costs=plan_costs,
            )
            # The task's own final_goal_distance says nothing about the LeWM
            # objective, which targets an achieved observation rather than the
            # scenario goal. Measure that reached distance for every policy so
            # the goal planner is scored on what it actually optimises, and the
            # others give it a floor and a ceiling.
            reached = agent_observations(env)
            goal_distance = (reached - goal_observation).flatten(1).norm(dim=-1)
            for row, distance in zip(episodes, goal_distance.tolist()):
                row["policy"] = label
                row["goal_observation_distance"] = distance
            rows.extend(episodes)
            decisions = timing["decisions"]
            timings[label] = {
                "seconds": timing["seconds"],
                "decisions": len(decisions),
                "seconds_per_decision": (
                    sum(d["seconds"] for d in decisions) / len(decisions)
                    if decisions
                    else 0.0
                ),
            }
            print(f"  {label}: {timings[label]['seconds']:.1f}s", flush=True)
            # Persist after every policy. These runs take hours, and writing
            # only at the end means a wall-clock kill destroys all of it --
            # which is how job 1212 would have ended.
            persist(args, manifest, states, chosen, cem_config, rows, timings)

        print(f"task {manifest['task_name']}, {states} test states", flush=True)
        run("random", "random")
        if not args.skip_oracle:
            run("oracle", "mpc")

        wanted = (
            {int(s) for s in args.seeds.split(",")} if args.seeds else None
        )
        checkpoints = [
            directory
            for directory in sorted(args.runs.glob("[0-9]*"))
            if (directory / "model.pt").exists()
            and (
                wanted is None
                or yaml.safe_load((directory / "resolved_config.yaml").read_text())[
                    "seed"
                ]
                in wanted
            )
        ]
        print(f"scoring {len(checkpoints)} checkpoints", flush=True)
        for directory in checkpoints[: args.max_runs]:
            config = yaml.safe_load((directory / "resolved_config.yaml").read_text())
            kind, regime, seed = (
                config["model"]["kind"],
                config["data"]["regime"],
                config["seed"],
            )
            model = load_model(directory / "model.pt", args.device)
            tag = f"{kind}|{regime}|{seed}"
            run(f"reward|{tag}", "mpc", reward_costs(model, block, args.device))
            run(
                f"goal|{tag}",
                "mpc",
                goal_costs(model, goal_observation, block, args.device),
            )
    finally:
        env.close()
        scratch.close()

    summary = persist(args, manifest, states, chosen, cem_config, rows, timings)

    print(f"\n{'policy':34s}{'success':>12s}{'return':>10s}{'coll':>7s}{'timeout':>9s}")
    print("-" * 72)
    for policy, values in summary.items():
        success = values["success"]
        print(
            f"{policy:34s}{success['count']:>4d}/{values['episodes']:<3d}"
            f"{success['rate']:>5.0%}{values['return']['mean']:>10.3f}"
            f"{values['collision_rate']:>7.2f}{values['timeout_rate']:>9.2f}"
        )


if __name__ == "__main__":
    main()
