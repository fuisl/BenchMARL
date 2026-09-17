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
dynamics the planner rolls. Five policies share those states:

    random       the floor: uniform noise resampled every primitive step
    heuristic    VMAS's own hand-written policy, where the scenario ships one --
                 the baseline a learned model has to be worth more than. Only
                 Transport and Wheel have one; on Buzz Wire and Dropout the
                 column is absent rather than faked.
    oracle       the true simulator as dynamics, scoring task reward -- the
                 ceiling for ``reward``, and the oracle gap
    goal_oracle  the true simulator scoring goal distance -- the ceiling for
                 ``goal``. A separate policy because job 1218 measured the
                 reward oracle ending *further* from the generated goal than
                 random did (0.839 against 0.256 on Buzz Wire): maximizing task
                 reward moves away from a goal drawn from an arbitrary
                 trajectory, so one oracle cannot bound both objectives.
    reward       learned latents + the learned reward/termination readout,
                 scoring J = -sum_t sum_i r_i,t, the objective the plan committed to
    goal         learned latents + LeWM's terminal latent-goal distance

The two learned costs are reported separately on purpose. The experiment plan
fixes task reward as the planning objective and defers latent-goal scoring, so
``goal`` is a labelled alternative, not a substitute. It is here because job 1203
showed the readout is where the chain breaks on Buzz Wire -- anti-correlated at
rho ~ -0.25 even given the simulator's own latents, because the reward depends on
the ball and no agent observes the ball -- which predicts that ``reward`` fails
there and ``goal``, needing no readout, does not. That is a real prediction this
run can falsify.

Goal semantics follow LeWM rather than the task: the goal is an *achieved*
observation, reached by rolling a plan from the same state, so it is reachable by
construction. ``goal`` therefore measures goal-reaching, not task success, and
its success column is distance-thresholded rather than the scenario's ``done()``.
Only ``reward``, ``heuristic`` and ``oracle`` are comparable on task success;
``goal`` belongs against ``goal_oracle`` and ``random``.

Which plan supplies that goal decides whether any of it means anything. Taking
candidate 0 -- an arbitrary random plan, LeWM's own convention -- produces a goal
the goal oracle reaches in 20/20 episodes on both tasks while ending no closer to
the task than random behaviour does (jobs 1218/1221/1226; 0.8993 against 0.8988
on Transport, 1.0937 against 1.0852 on Buzz Wire). So ``--goal-source task``
selects the candidate endpoint that gets furthest on the task's own distance
instead, which leaves reachability untouched and makes the goal worth reaching.
``--goal-source arbitrary`` restores the old behaviour for reproduction only.
Every table must therefore carry the task distance beside the goal distance:
without it, a planner that perfectly solves a meaningless goal reads as a
success.
"""

from pathlib import Path

import numpy as np
import torch

from examples.world_model.cem import CEMConfig
from examples.world_model.collect import SPLITS
from examples.world_model.goal_planning import goal_plan_costs
from examples.world_model.metrics import mean_interval, success_interval
from examples.world_model.model_input import ObservationBuilder
from examples.world_model.mpc import (
    action_bounds,
    evaluate_policy,
    goal_dimension_weight,
    goal_oracle_costs,
    MPCConfig,
    scenario_heuristic,
    task_outcome,
)
from examples.world_model.plan_ranking import model_costs, select_anchor_states
from examples.world_model.snapshot_restore import agent_observations, broadcast_state
from examples.world_model.train import MARL_EVAL_FILE, MODEL_NAME
from examples.world_model.readout_diagnostic import simulate
from tensordict import TensorDict


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
    _, _complete, _block_valid, _observation, endpoint, _entities = simulate(
        scratch_env, snapshot, candidates, action_block
    )
    # `simulate` returns on CPU because its own caller compares on CPU. The goal
    # is then differenced against live observations, so hand it back on the
    # environment's device rather than leaving that to every caller.
    return endpoint[:, 0].to(scratch_env.device)


@torch.no_grad()
def reference_goals(
    env, initial_state, policy, *, generator, cem_config, mpc_config, scratch_env,
    outcome_fn,
):
    """Goal observation per evaluation state: (B,N,O) -- where a competent
    controller ends up, having started from the same state with the same budget.

    A goal must be *reachable*, or the planning question is ill posed, and it
    must carry *task content*, or reaching it means nothing. LeWM's convention --
    the endpoint of an arbitrary random plan, `achieved_goals` -- satisfies only
    the first, and jobs 1218/1221/1226 measured what that costs: the goal oracle
    reaches such a goal in 20/20 episodes on both tasks and still ends at exactly
    the task distance random behaviour reaches (0.8993 against 0.8988 on
    Transport, 1.0937 against 1.0852 on Buzz Wire). Every "fraction of the oracle
    gap closed" measured against it is a fraction of nothing.

    Choosing the best of the random bank by task distance does not repair it:
    over 300 candidates and 25 steps the best endpoint improves task distance by
    0.0012 on Transport and 0.045 on Buzz Wire, because random action sequences
    make almost no task progress at all. The goal has to come from something that
    can actually do the task.

    So it comes from the strongest controller the task has: the scenario's own
    hand-written policy where one exists, and otherwise the reward oracle, which
    on Buzz Wire moves the ball from 1.095 to 0.557 and succeeds 5/20.

    The goal is the observation at each episode's *best* task frame rather than
    its terminal one. That reward oracle collides in 35% of episodes, so its
    terminal frames are the states it crashed in; a planner aimed at one would be
    aimed at a crash. The best frame is reachable for exactly the same reason the
    terminal frame is -- the policy was there, from this state, within this
    budget -- and it is the most the trajectory ever achieved.
    """
    episodes, _timing, terminal = evaluate_policy(
        env,
        initial_state,
        policy=policy,
        generator=generator,
        cem_config=cem_config,
        mpc_config=mpc_config,
        scratch_env=scratch_env,
        outcome_fn=outcome_fn,
    )
    best = np.mean([row["best_goal_distance"] for row in episodes])
    final = np.mean([row["final_goal_distance"] for row in episodes])
    reached = sum(row["success"] for row in episodes)
    collided = sum(row["collision"] for row in episodes)
    print(
        f"  goals from {'the scenario heuristic' if callable(policy) else 'the reward oracle'}: "
        f"task distance {best:.4f} at the best frame, {final:.4f} at the last; "
        f"{reached}/{len(episodes)} native successes, {collided} collided",
        flush=True,
    )
    return terminal


def zero_policy(env):
    """Do nothing. The baseline the audit's contract asks for and the project
    never ran: a goal that a motionless agent already satisfies is not a goal."""
    n_agents = len(env._env.world.agents)
    action_dim = env.full_action_spec_unbatched["agents", "action"].shape[-1]
    return torch.zeros(
        env.batch_size[0], 1, n_agents * action_dim, device=env.device
    )


@torch.no_grad()
def solved_states(task, anchors, pool, args, manifest, cem_config, mpc_config, outcome_fn):
    """Anchor indices the reference controller actually solves, and the goal it
    reached on each: (M,), (M,N,O), with M <= args.states.

    Job 1229 ruled out the alternative. Taking the goal from wherever the
    reference controller *ended* gives, on Buzz Wire, seven goals out of twenty
    that are the states it crashed in -- the oracle approaches monotonically
    until it collides, so its best frame is its crash frame in 20/20 episodes.
    Aiming a planner at those is aiming it at a crash, and the goal oracle duly
    collided in 90% of episodes and finished worse on the task than random.

    A goal taken from an episode the controller *solved* cannot have that
    problem: the ball is at its target and nothing has collided. The cost is that
    only some states yield one -- the Buzz Wire reward oracle succeeds 5/20 -- so
    a larger pool is run and the states it fails are dropped. Dropping them
    changes what the evaluation set is, which is why it is reported: these are
    the states a competent controller can solve, not a random sample of states.
    """
    pool_env = task.get_env_fun(pool.numel(), True, 0, args.device)()
    pool_env.reset()
    scratch = task.get_env_fun(
        pool.numel() * args.num_samples, True, 0, args.device
    )()
    scratch.reset()
    heuristic = scenario_heuristic(manifest["task_name"], action_bounds(pool_env)[1])
    try:
        episodes, _timing, best = evaluate_policy(
            pool_env,
            select_anchor_states(anchors, pool, args.device),
            policy=heuristic or "mpc",
            generator=torch.Generator(device=args.device).manual_seed(args.seed),
            cem_config=cem_config,
            mpc_config=mpc_config,
            scratch_env=scratch,
            outcome_fn=outcome_fn,
        )
    finally:
        pool_env.close()
        scratch.close()

    solved = torch.tensor(
        [row["success"] for row in episodes], device=best.device
    ).nonzero(as_tuple=True)[0][: args.states]
    distance = np.mean([episodes[i]["best_goal_distance"] for i in solved.tolist()])
    print(
        f"  reference controller solved {len(solved)} of {pool.numel()} pooled "
        f"states; goals at task distance {distance:.4f}",
        flush=True,
    )
    if solved.numel() == 0:
        raise ValueError(
            f"The reference controller solved none of {pool.numel()} states, so "
            "there is no task-relevant goal to plan toward on this task"
        )
    return pool[solved.cpu()], best[solved]


def log_policy(args, manifest, label, episodes, timing):
    """One wandb run and one marl-eval file per evaluated policy.

    BenchMARL's own convention, the same one `train.py` follows: the run name is
    {algorithm}_{task}_{model}, the wandb group is the task, and the id is the
    experiment name. A closed-loop policy is an algorithm in that sense -- it is
    what is being compared on a shared task at a matched planning budget -- so
    `relational_correlated_goal` and `oracle` sit in the same slot MAPPO and
    IPPO occupy there.

    Control metrics are genuinely per-episode, which is the shape JsonWriter
    expects, so these files carry real distributions rather than single values.
    """
    if not args.wandb:
        return
    from benchmarl.experiment.logger import JsonWriter
    from torchrl.record.loggers import get_logger
    from torchrl.record.loggers.utils import generate_exp_name

    environment, task_name = manifest["task_name"].split("/")
    algorithm = label.replace("|", "_")
    experiment_name = generate_exp_name(f"{algorithm}_{task_name}_{MODEL_NAME}", "")
    metrics = {
        "return": torch.tensor([row["return"] for row in episodes]),
        "success": torch.tensor([float(row["success"]) for row in episodes]),
        "collision": torch.tensor([float(row["collision"]) for row in episodes]),
        "episode_length": torch.tensor([float(row["length"]) for row in episodes]),
    }
    if "goal_reached" in episodes[0]:
        metrics["goal_reached"] = torch.tensor(
            [float(row["goal_reached"]) for row in episodes]
        )
        metrics["neg_goal_distance"] = torch.tensor(
            [-row["goal_observation_distance"] for row in episodes]
        )

    folder = args.output.parent / experiment_name
    folder.mkdir(parents=True, exist_ok=True)
    JsonWriter(
        folder=str(folder),
        name=MARL_EVAL_FILE,
        algorithm_name=algorithm,
        task_name=task_name,
        environment_name=environment,
        seed=args.seed,
    ).write(total_frames=len(episodes), metrics=metrics, evaluation_step=0)

    logger = get_logger(
        logger_type="wandb",
        logger_name=str(folder),
        experiment_name=experiment_name,
        wandb_kwargs={
            "group": task_name,
            "id": experiment_name,
            "project": args.project,
            "entity": args.entity,
            "config": {
                "algorithm": algorithm,
                "task": task_name,
                "environment": environment,
                "model": MODEL_NAME,
                "policy": label,
                "seed": args.seed,
                "states": args.states,
                "num_samples": args.num_samples,
                "num_iters": args.num_iters,
                "horizon": args.horizon,
                "execute_blocks": args.execute_blocks,
            },
        },
    )
    for name, values in metrics.items():
        logger.log_scalar(name, float(values.mean()), step=0)
    logger.log_scalar("seconds_per_decision", timing["seconds_per_decision"], step=0)
    if hasattr(logger, "experiment"):
        logger.experiment.finish()


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
            **(
                {
                    "goal_reached": success_interval(
                        [r["goal_reached"] for r in episodes]
                    ),
                    "goal_observation_distance": mean_interval(
                        [r["goal_observation_distance"] for r in episodes], rng
                    ),
                }
                if "goal_reached" in episodes[0]
                else {}
            ),
            "collision_rate": float(np.mean([r["collision"] for r in episodes])),
            "timeout_rate": float(np.mean([r["timeout"] for r in episodes])),
            "final_goal_distance": mean_interval(
                [r["final_goal_distance"] for r in episodes], rng
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
    parser.add_argument("--horizon", type=int, default=5, help="blocks planned")
    parser.add_argument(
        "--execute-blocks",
        type=int,
        default=1,
        help="blocks executed per decision, of --horizon planned. Jobs up to "
        "1233 tied this to --horizon, so at H=5 and block 5 all 25 primitive "
        "actions ran before the next observation and a 100-step episode held "
        "at most four decisions. One block gives feedback every five steps, "
        "which is what a contact task needs; it also multiplies planning calls "
        "per episode, so report episode compute alongside per-decision budget.",
    )
    parser.add_argument("--goal-offset", type=int, default=5, help="blocks ahead")
    parser.add_argument(
        "--goal-source",
        choices=("reference", "success", "arbitrary"),
        default="reference",
        help="where the goal comes from. `reference`: where the strongest "
        "controller the task has actually ends up. `arbitrary`: the endpoint of "
        "a random plan, LeWM's own convention and what jobs 1218-1226 used -- "
        "reachable but task-orthogonal, kept only to reproduce them.",
    )
    parser.add_argument(
        "--goal-threshold",
        type=float,
        default=0.05,
        help="observation-space L2 within which a goal counts as reached. A "
        "declared parameter, not a calibrated one: the graded distance is the "
        "primary number and this only names a cut through it.",
    )
    parser.add_argument(
        "--goal-metric",
        choices=("full", "position"),
        default="full",
        help="which observation components the goal distance scores. `full`: "
        "every component, including velocity -- what jobs 1218-1229 used. "
        "`position`: velocity dropped. Job 1229 measured why that matters: "
        "matching a goal's velocity means arriving at speed, and on Buzz Wire "
        "that means driving through the wire, which the objective cannot see.",
    )
    parser.add_argument(
        "--state-pool",
        type=int,
        default=None,
        help="with --goal-source success: how many candidate states to run the "
        "reference controller over before keeping the ones it solved. The Buzz "
        "Wire reward oracle succeeds 5/20, so ~4x the wanted states.",
    )
    parser.add_argument(
        "--roots-split",
        choices=SPLITS,
        default=None,
        help="evaluate from episode STARTS in `initial_states.pt` of this "
        "split, instead of the mid-episode anchors of the test split. Anchors "
        "are not independent: Balance's 197 test anchors come from 16 root "
        "episodes sampled at steps 0/20/40/60/80, so an interval over anchors "
        "understates uncertainty, and an anchor at step 80 has 20 steps of "
        "episode left rather than 100. Roots give one independent full-length "
        "episode each. Use `train` for development and keep `test` frozen for "
        "the final comparison -- no learned model is involved in a "
        "true-simulator reference, so a train root leaks nothing.",
    )
    parser.add_argument(
        "--objectives",
        choices=("both", "reward", "goal"),
        default="both",
        help="which cost functions are scored, for the true-simulator "
        "references AND for every checkpoint. `reward` optimises the native "
        "task reward; `goal` optimises observation-space distance to a goal. "
        "Each costs a full CEM search at every decision, so scoring both "
        "doubles a sweep. Job 1237 measured the goal objective to be worse "
        "than doing nothing on Balance even with true dynamics, so a run "
        "asking about control should usually say `reward`.",
    )
    parser.add_argument(
        "--skip-oracle",
        action="store_true",
        help="skip the true-simulator planners but keep random/zero/heuristic. "
        "Orthogonal to --objectives: this is about cost, that is about which "
        "question is being asked.",
    )
    parser.add_argument("--project", default="counterfactual-wm")
    parser.add_argument("--entity", default="cair-traffic")
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="log each policy through BenchMARL's logger conventions",
    )
    parser.add_argument(
        "--skip-references",
        action="store_true",
        help="score only checkpoints; references come from --reference-cache",
    )
    parser.add_argument(
        "--reference-cache",
        type=Path,
        default=None,
        help="random/oracle episodes and goals, computed once and reused. The "
        "oracle rolls states x num_samples simulator environments per CEM "
        "iteration and is the one policy that must not share the slice.",
    )
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
    outcome_fn = task_outcome(manifest["task_name"])
    cem_config = CEMConfig(
        horizon=args.horizon,
        num_samples=args.num_samples,
        num_iters=args.num_iters,
        # LeWM keeps 30 of 300; hold the ratio so a smaller budget stays a
        # smaller version of the same search rather than a greedier one.
        num_elites=args.num_elites or max(1, round(0.1 * args.num_samples)),
    )
    mpc_config = MPCConfig(
        receding_horizon=args.execute_blocks, action_block=block
    )
    mpc_config.validate(cem_config.horizon)

    # Evaluation states come from the bank's test split, so no learned model has
    # trained on the states it is asked to control.
    if args.roots_split is None:
        anchors = torch.load(
            args.data / "anchors.pt", map_location="cpu", weights_only=True
        )
        wanted_split = SPLITS.index("test")
    else:
        anchors = torch.load(
            args.data / "initial_states.pt", map_location="cpu", weights_only=True
        )
        wanted_split = SPLITS.index(args.roots_split)
    available = (anchors["split"] == wanted_split).nonzero(as_tuple=True)[0]
    generator = torch.Generator().manual_seed(args.seed)
    shuffled = available[torch.randperm(available.numel(), generator=generator)]
    chosen = shuffled[: args.states]
    solved_goals = None
    if args.goal_source == "success":
        # The evaluation set becomes the states a competent controller can
        # solve, so it is chosen by running one rather than by sampling.
        chosen, solved_goals = solved_states(
            task,
            anchors,
            shuffled[: args.state_pool or 4 * args.states],
            args,
            manifest,
            cem_config,
            mpc_config,
            outcome_fn,
        )
    states = chosen.numel()
    initial_state = select_anchor_states(anchors, chosen, args.device)

    # Everything the references depend on. A cache built under any other
    # identity is a different experiment and must not be reused silently.
    reference_identity = {
        "task_name": manifest["task_name"],
        "anchors_sha256": manifest["anchors_sha256"],
        "states": args.states,
        "seed": args.seed,
        "num_samples": args.num_samples,
        "num_iters": args.num_iters,
        "horizon": args.horizon,
        "execute_blocks": args.execute_blocks,
        "goal_offset": args.goal_offset,
        "goal_source": args.goal_source,
        "goal_metric": args.goal_metric,
        "state_pool": args.state_pool,
        "roots_split": args.roots_split,
        "objectives": args.objectives,
        "skip_oracle": args.skip_oracle,
        "references": "random,zero,heuristic"
        + (
            ""
            if args.skip_oracle
            else {
                "both": ",oracle,goal_oracle",
                "reward": ",oracle",
                "goal": ",goal_oracle",
            }[args.objectives]
        ),
    }

    env = task.get_env_fun(states, True, 0, args.device)()
    env.reset()
    # Resolved here rather than beside the reference runs because the goal
    # source needs it: where a scenario ships a hand-written policy, that is the
    # strongest controller the task has.
    heuristic = scenario_heuristic(manifest["task_name"], action_bounds(env)[1])
    goal_weight = (
        None
        if args.goal_metric == "full"
        else goal_dimension_weight(
            manifest["task_name"],
            env.observation_spec["agents", "observation"].shape[-1],
            args.device,
        )
    )

    # The scratch simulator exists only for the oracle and for generating goals:
    # states x num_samples environments, which is the bulk of a process's
    # memory. A shard that takes both from a reference cache never touches it,
    # so it is not built -- which is what lets many shards share one slice.
    cached = args.reference_cache is not None and args.reference_cache.exists()
    needs_scratch = not (cached and args.skip_references)
    scratch = None
    if needs_scratch:
        scratch = task.get_env_fun(
            states * args.num_samples, True, 0, args.device
        )()
        scratch.reset()

    try:
        # One random plan per state supplies the LeWM goal, drawn before any
        # planning so every policy is scored against the same fixed targets.
        # Drawn from the shared generator before any shard-dependent branching,
        # so concurrent shards target identical goals whether cached or not.
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

        # The random and oracle references are per-state, not per-checkpoint,
        # and the oracle is by far the most expensive policy here: it rolls
        # states x num_samples simulator environments through every CEM
        # iteration. Job 1216 measured it at 4827 s while sharing a slice
        # against 299 s alone, so it is computed once, alone, and reused.
        cache = args.reference_cache
        if cache is not None and cache.exists():
            stored = torch.load(cache, map_location="cpu", weights_only=False)
            if stored["identity"] != reference_identity:
                raise ValueError(
                    f"Reference cache {cache} was built for a different "
                    f"experiment.\n  cached: {stored['identity']}"
                    f"\n  wanted: {reference_identity}"
                )
            goal_observation = stored["goal_observation"].to(args.device)
            reference_rows = stored["rows"]
            print(f"loaded reference cache for {states} states", flush=True)
        else:
            if solved_goals is not None:
                goal_observation = solved_goals
            elif args.goal_source == "arbitrary":
                goal_observation = achieved_goals(
                    scratch, initial_state, goal_plans, block
                )
            else:
                # The heuristic where the scenario ships one, the reward oracle
                # otherwise. On Transport the heuristic is the only policy that
                # has ever scored a native success; on Buzz Wire the reward
                # oracle is, at 5/20.
                goal_observation = reference_goals(
                    env,
                    initial_state,
                    heuristic or "mpc",
                    generator=torch.Generator(device=args.device).manual_seed(
                        args.seed
                    ),
                    cem_config=cem_config,
                    mpc_config=mpc_config,
                    scratch_env=scratch,
                    outcome_fn=outcome_fn,
                )
            reference_rows = None

        rows, timings = [], {}

        def run(label, policy, plan_costs=None, observe=None):
            planner = torch.Generator(device=args.device).manual_seed(args.seed)
            if observe is not None:
                # Stateful for `history`; a stale buffer would carry the previous
                # checkpoint's frames into this episode set.
                observe.reset()
            episodes, timing, terminal = evaluate_policy(
                env,
                initial_state,
                policy=policy,
                generator=planner,
                cem_config=cem_config,
                mpc_config=mpc_config,
                scratch_env=scratch,
                outcome_fn=outcome_fn,
                plan_costs=plan_costs,
                observe=agent_observations if observe is None else observe,
                goal_observation=goal_observation,
                goal_threshold=args.goal_threshold,
                goal_weight=goal_weight,
            )
            # Goal distance and goal reaching are latched inside EpisodeStats
            # at each episode's own terminal frame. Reading them here, after the
            # loop, measured whichever state a finished slot had drifted to
            # while its neighbours kept running -- on Buzz Wire, where most
            # episodes collide early, that was almost every episode.
            for row in episodes:
                row["policy"] = label
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
            log_policy(args, manifest, label, episodes, timings[label])
            # Persist after every policy. These runs take hours, and writing
            # only at the end means a wall-clock kill destroys all of it --
            # which is how job 1212 would have ended.
            persist(args, manifest, states, chosen, cem_config, rows, timings)

        print(f"task {manifest['task_name']}, {states} test states", flush=True)
        if reference_rows is not None:
            rows.extend(reference_rows)
            print(f"  reused {len(reference_rows)} cached reference episodes")
        elif not args.skip_references:
            run("random", "random")
            # A goal a motionless agent already satisfies is not a goal. The
            # audit's contract asks for this baseline and the project has never
            # run it.
            run("zero", zero_policy)
            if heuristic is not None:
                run("heuristic", heuristic)
            else:
                print(
                    f"  {manifest['task_name']} ships no heuristic; "
                    "random is the only non-planning reference",
                    flush=True,
                )
            if not args.skip_oracle and args.objectives in ("both", "reward"):
                run("oracle", "mpc")
            if not args.skip_oracle and args.objectives in ("both", "goal"):
                run(
                    "goal_oracle",
                    "mpc",
                    goal_oracle_costs(scratch, goal_observation, goal_weight),
                )
            if cache is not None:
                cache.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "identity": reference_identity,
                        "goal_observation": goal_observation.cpu(),
                        "rows": list(rows),
                    },
                    cache,
                )
                print(f"  wrote reference cache {cache}", flush=True)

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
        checkpoints = checkpoints[: args.max_runs]
        print(f"scoring {len(checkpoints)} checkpoints", flush=True)
        for directory in checkpoints:
            config = yaml.safe_load((directory / "resolved_config.yaml").read_text())
            kind, regime, seed = (
                config["model"]["kind"],
                config["data"]["regime"],
                config["seed"],
            )
            model = load_model(directory / "model.pt", args.device)
            # The checkpoint's own recorded data config decides what it is fed.
            # Job 1223 trained three input conditions on this bank, and a model
            # whose encoder was fitted on 24 dimensions cannot be planned with
            # from the agents' 6. `state_input` defaults to `observation`, so
            # every bank collected before Stage 2 keeps its existing behaviour.
            state_input = config["data"].get("state_input", "observation")
            observe = ObservationBuilder(
                state_input,
                config["data"].get("history_frames", 3),
                int(model.obs_mean.shape[-1]),
                action_block_stride=mpc_config.receding_horizon,
            )
            # The input condition is part of the model's identity, so it belongs
            # in the label: without it three conditions collapse into one row.
            tag = f"{kind}|{regime}|{seed}"
            if state_input != "observation":
                tag = f"{state_input}|{tag}"
            if args.objectives in ("both", "reward"):
                run(
                    f"reward|{tag}",
                    "mpc",
                    reward_costs(model, block, args.device),
                    observe=observe,
                )
            if args.objectives in ("both", "goal"):
                if state_input != "observation":
                    # The goal is an achieved *agent* observation. Encoding it
                    # with a model fitted on another input would need the entity
                    # frame at that same step, which is not captured. Fail here
                    # rather than silently pad or truncate the goal.
                    raise ValueError(
                        "The goal objective needs goals in the model's own input "
                        f"format; state_input={state_input} has none recorded. "
                        "Use --objectives reward."
                    )
                run(
                    f"goal|{tag}",
                    "mpc",
                    goal_costs(model, goal_observation, block, args.device),
                )
    finally:
        env.close()
        if scratch is not None:
            scratch.close()

    summary = persist(args, manifest, states, chosen, cem_config, rows, timings)

    # `task_dist` is the scenario's own distance and is the only column
    # comparable across every row: a goal planner ignores task reward and the
    # heuristic ignores the goal, but both move the task or they do not. Reading
    # the goal column alone is what let a task-orthogonal goal look like control
    # for three jobs (1218/1221/1226).
    print(
        f"\n{'policy':34s}{'success':>12s}{'return':>10s}{'task_dist':>11s}"
        f"{'goal_dist':>11s}{'coll':>7s}{'timeout':>9s}"
    )
    print("-" * 94)
    for policy, values in summary.items():
        success = values["success"]
        goal = values.get("goal_observation_distance")
        goal_column = f"{goal['mean']:.4f}" if goal else "--"
        print(
            f"{policy:34s}{success['count']:>4d}/{values['episodes']:<3d}"
            f"{success['rate']:>5.0%}{values['return']['mean']:>10.3f}"
            f"{values['final_goal_distance']['mean']:>11.4f}{goal_column:>11s}"
            f"{values['collision_rate']:>7.2f}{values['timeout_rate']:>9.2f}"
        )


if __name__ == "__main__":
    main()
