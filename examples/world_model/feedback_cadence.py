# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the license found in the LICENSE file in the root directory.
"""Experiment 25: does replanning cadence compensate for the frozen structured
surrogate's dynamics error, independent of the mechanisms Gate 5 (job 1335)
already localized?

Job 1331 already closes the control loop at receding_horizon (K) = 1: it
re-observes the true state and replans every action block, all 25 primitive
steps of a 5-block imagined horizon. Its own `control_summary`/`control_rows`
for the three full-coverage seeds are the K=1 condition here, reused verbatim
rather than recomputed, so this file only adds K in {2, 5} -- coarser
feedback, down to K=5 (plan once, execute the whole horizon open-loop before
observing again).

Model, test roots, CEM budget, objective and control seed are frozen exactly
as job 1331/1335 used them. Nothing is retrained and nothing about the
dynamics changes: only how much of a plan gets executed before the planner is
allowed to see the true state again.

CEM's first decision does not depend on K -- the same model, state, control
seed and CEM budget produce the same first population regardless of how many
of its blocks later get executed -- so "first-action quality" (does a bad
plan already exist before any stale-plan execution can explain it) is not
recomputed here. Gate 5's iteration-30, horizon-5, teacher-forced row for each
seed is exactly that first decision's regret/rank/percentile, and is cited
from job 1335 rather than duplicated.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from benchmarl.environments import VmasTask
from examples.world_model.cem import CEMConfig
from examples.world_model.decision_information import take_snapshot
from examples.world_model.mpc import MPCConfig, evaluate_policy, task_outcome
from examples.world_model.planner_tail_failure import load_surrogate
from examples.world_model.structured_surrogate import controller_cost, live_structured_state


K_VALUES = (1, 2, 5)
SEEDS = (9100, 9101, 9102)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def summarize_episodes(episodes):
    success = sum(row["success"] for row in episodes)
    collision = sum(row["collision"] for row in episodes)
    timeout = sum(row["timeout"] for row in episodes)
    n = len(episodes)
    return {
        "episodes": n,
        "success_rate": success / n,
        "collision_rate": collision / n,
        "timeout_rate": timeout / n,
        "return_mean": sum(row["return"] for row in episodes) / n,
        "final_goal_distance_mean": sum(row["final_goal_distance"] for row in episodes) / n,
        "episode_length_mean": sum(row["length"] for row in episodes) / n,
    }


def k1_row_from_job_1331(prior, seed):
    """Reuse job 1331's own closed-loop K=1 run rather than recomputing it."""
    label = f"structured_full_{seed}"
    episodes = [row for row in prior["control_rows"] if row["policy"] == label]
    timing = prior["timing"][label]
    decisions = timing["decisions"]
    return {
        "seed": seed,
        "K": 1,
        "source": "job_1331_control_rows",
        "summary": summarize_episodes(episodes),
        "timing": {
            "seconds": timing["seconds"],
            "decisions": len(decisions),
            "seconds_per_decision": (
                sum(d["seconds"] for d in decisions) / len(decisions) if decisions else 0.0
            ),
        },
    }


def first_action_quality_from_gate5(gate5_result, seed):
    """Cite, not recompute: Gate 5's iteration-30/horizon-5 teacher-forced row
    is CEM's first decision from this same frozen state, model and control
    seed -- identical for every K here, since nothing about replanning cadence
    can act before the first decision is made."""
    row = next(
        row
        for row in gate5_result["rows"]
        if row["seed"] == seed and row["iteration"] == 30 and row["horizon"] == 5
    )["teacher_forced"]
    return {
        "seed": seed,
        "selected_regret": row["selected_regret"],
        "oracle_percentile": row["oracle_percentile"],
        "spearman": row["spearman"],
        "source": "job_1335_iteration30_horizon5_teacher_forced",
    }


def render(rows, output):
    by_k = {}
    for row in rows:
        by_k.setdefault(row["K"], []).append(row)
    ks = sorted(by_k)

    def series(key):
        return [sum(r["summary"][key] for r in by_k[k]) / len(by_k[k]) for k in ks]

    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    axes[0].plot(ks, series("collision_rate"), marker="o", label="collision rate")
    axes[0].plot(ks, series("success_rate"), marker="o", label="success rate")
    axes[0].set_title("Outcome rate vs replanning cadence K")
    axes[0].legend(fontsize=8)

    axes[1].plot(ks, series("final_goal_distance_mean"), marker="o", color="tab:red")
    axes[1].set_title("Mean final ball-goal distance")

    axes[2].plot(ks, series("return_mean"), marker="o", color="tab:green")
    axes[2].set_title("Mean per-agent return")

    for axis in axes:
        axis.set_xlabel("K (blocks executed before replanning)")
        axis.set_xticks(ks)
        axis.grid(alpha=0.25)
    figure.suptitle("Experiment 25: feedback cadence, frozen structured surrogate")
    figure.savefig(output / "feedback_cadence.png", dpi=180)
    figure.savefig(output / "feedback_cadence.pdf")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source", type=Path, default=Path("outputs/structured_surrogate_1331")
    )
    parser.add_argument(
        "--gate5", type=Path, default=Path("outputs/planner_tail_1335")
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    prior = json.loads((args.source / "result/result.json").read_text())
    gate5_result = json.loads((args.gate5 / "result/result.json").read_text())
    stored = torch.load(
        args.source / "coverage/oracle_plans.pt", map_location="cpu", weights_only=False
    )

    config = prior["config"]
    frozen = {
        "horizon": int(config["horizon"]),
        "num_samples": int(config["num_samples"]),
        "num_elites": int(config["num_elites"]),
        "num_iters": int(config["num_iters"]),
        "control_seed": int(config["control_seed"]),
        "action_block": 5,
        "model_seeds": SEEDS,
        "objective": prior["selected_objective"]["full"],
    }
    if frozen["objective"] != "probability":
        raise ValueError("Experiment 25 freezes job 1331's full/probability objective")

    test_ids = (stored["split"] == 2).nonzero(as_tuple=True)[0]
    if test_ids.tolist() != prior["test_root_ids"]:
        raise ValueError("Frozen test roots differ from job 1331's own record")
    if test_ids.numel() != 16:
        raise ValueError(f"Expected 16 frozen test roots, found {test_ids.numel()}")

    task = VmasTask.BUZZ_WIRE.get_from_yaml()
    outcome_fn = task_outcome("vmas/buzz_wire")
    cem_config = CEMConfig(
        horizon=frozen["horizon"],
        num_samples=frozen["num_samples"],
        num_elites=frozen["num_elites"],
        num_iters=frozen["num_iters"],
    )
    initial_state = take_snapshot(stored["snapshot"], test_ids, args.device)

    env = task.get_env_fun(test_ids.numel(), True, 0, args.device)()
    env.reset()

    all_rows = []
    first_action = {}
    try:
        for seed in SEEDS:
            all_rows.append(k1_row_from_job_1331(prior, seed))
            first_action[str(seed)] = first_action_quality_from_gate5(
                gate5_result, seed
            )

            model, _checkpoint = load_surrogate(
                args.source / f"result/model_full_{seed}.pt", args.device
            )
            plan_costs = controller_cost(model, frozen["action_block"], frozen["objective"])
            for k in K_VALUES[1:]:
                mpc_config = MPCConfig(receding_horizon=k, action_block=frozen["action_block"])
                mpc_config.validate(cem_config.horizon)
                episodes, timing, _terminal = evaluate_policy(
                    env,
                    initial_state,
                    policy="mpc",
                    generator=torch.Generator(device=args.device).manual_seed(
                        frozen["control_seed"]
                    ),
                    cem_config=cem_config,
                    mpc_config=mpc_config,
                    outcome_fn=outcome_fn,
                    plan_costs=plan_costs,
                    observe=live_structured_state,
                )
                decisions = timing["decisions"]
                row = {
                    "seed": seed,
                    "K": k,
                    "source": "computed",
                    "summary": summarize_episodes(episodes),
                    "timing": {
                        "seconds": timing["seconds"],
                        "decisions": len(decisions),
                        "seconds_per_decision": (
                            sum(d["seconds"] for d in decisions) / len(decisions)
                            if decisions
                            else 0.0
                        ),
                    },
                }
                all_rows.append(row)
                print(f"seed {seed} K={k} complete: {row['summary']}", flush=True)
    finally:
        env.close()

    result = {
        "question": (
            "Does more frequent replanning (smaller K) recover control that "
            "one-shot open-loop planning (K=horizon) cannot, on the same "
            "frozen structured surrogate Gate 5 diagnosed?"
        ),
        "source_job": 1331,
        "gate5_job": 1335,
        "frozen_protocol": frozen,
        "k_values": list(K_VALUES),
        "first_action_quality": first_action,
        "first_action_quality_note": (
            "CEM's first decision is identical across K given the same model, "
            "state, control seed and budget, so it is cited from Gate 5's "
            "iteration-30/horizon-5 teacher-forced row rather than "
            "recomputed here."
        ),
        "rows": all_rows,
    }
    write_json(args.output / "result.json", result)
    render(all_rows, args.output)
    print(f"wrote Experiment 25 results to {args.output}", flush=True)


if __name__ == "__main__":
    main()
