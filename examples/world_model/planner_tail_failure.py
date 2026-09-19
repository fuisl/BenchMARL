# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the license found in the LICENSE file in the root directory.
"""Gate 5: localize planner-induced tail failure without retraining.

The diagnostic freezes the three full-coverage structured surrogates from job
1331 and records their CEM populations at iterations 1/5/10/20/30.  Simulator
truth is evaluated only *after* CEM has finished.  Each candidate then has:

* a simulator trajectory (future truth);
* a teacher-forced surrogate trajectory (true state re-supplied each block);
* a fully recursive surrogate trajectory.

Job 1331's 14-D structured state is privileged.  This file does not relabel it
as an observation: the output contract marks the run diagnostic-only and the
planner as ineligible for deployment.  Future simulator truth is never passed
to candidate selection.
"""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from tensordict import TensorDict

from benchmarl.environments import VmasTask
from examples.world_model.cem import CEMConfig, cem_plan
from examples.world_model.decision_information import take_snapshot
from examples.world_model.mpc import action_bounds, task_outcome, unpack_actions
from examples.world_model.plan_ranking import spearman
from examples.world_model.snapshot_restore import broadcast_state
from examples.world_model.structured_surrogate import (
    STATE_DIM,
    StructuredSurrogate,
    bank_state,
    live_structured_state,
    surrogate_cost,
    wire_clearance,
)


STAGES = (1, 5, 10, 20, 30)
TOPKS = (1, 5, 10, 30)
BUFFER_NAMES = (
    "state_mean",
    "state_std",
    "action_mean",
    "action_std",
    "delta_mean",
    "delta_std",
    "clearance_mean",
    "clearance_std",
    "penalty_mean",
    "penalty_std",
)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def information_contract():
    """Declare information roles without pretending Gate 5 is deployable."""
    variables = {
        "agent_pose_velocity": "P",
        "ball_pose_velocity": "P",
        "goal": "O",
        "joint_candidate_actions": "O",
        "current_clearance_from_privileged_geometry": "P",
        "future_state": "T",
        "future_clearance": "T",
        "future_collision": "T",
        "simulator_reward": "T",
        "simulator_rollout": "T",
    }
    candidate_selection = {
        "initial_structured_state": "P",
        "joint_candidate_actions": "O",
        "surrogate_predictions_from_privileged_state": "P",
    }
    if "T" in candidate_selection.values():
        raise ValueError("Future simulator truth may not enter candidate selection")
    return {
        "roles": {
            "O": "deployable observation or task specification",
            "P": "privileged training/evaluation information",
            "T": "future simulator truth",
        },
        "variables": variables,
        "deployment_rule": "A deployed planner may consume only O.",
        "candidate_selection_inputs": candidate_selection,
        "future_truth_used_by_candidate_selection": False,
        "deployment_contract_satisfied": all(
            role == "O" for role in candidate_selection.values()
        ),
        "diagnostic_only": True,
        "reason": (
            "The frozen job-1331 model requires exact 14-D structured state. "
            "Gate 5 diagnoses that legacy privileged-state controller; it is "
            "not evidence for a deployable observation model."
        ),
    }


def load_surrogate(path, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint["state_dict"]
    hidden = state["body.0.weight"].shape[0]
    model = StructuredSurrogate(
        *(state[name] for name in BUFFER_NAMES), hidden=hidden
    ).to(device)
    model.load_state_dict(state)
    model.eval()
    return model, checkpoint


def validate_frozen_protocol(prior, stored, seeds):
    config = prior["config"]
    expected = {
        "horizon": 5,
        "num_samples": 300,
        "num_elites": 30,
        "num_iters": 30,
        "control_seed": 8700,
    }
    for key, value in expected.items():
        if int(config[key]) != value:
            raise ValueError(f"Job-1331 {key}={config[key]}, expected frozen {value}")
    if prior["selected_objective"].get("full") != "probability":
        raise ValueError("Gate 5 freezes job 1331's full/probability objective")
    fitted = sorted(
        int(row["seed"]) for row in prior["fits"] if row["mix"] == "full"
    )
    if fitted != sorted(seeds):
        raise ValueError(f"Expected full model seeds {seeds}, found {fitted}")
    test_ids = (stored["split"] == 2).nonzero(as_tuple=True)[0]
    if test_ids.tolist() != prior["test_root_ids"]:
        raise ValueError("Frozen Gate-5 roots differ from job 1331's test roots")
    if test_ids.numel() != 16:
        raise ValueError(f"Expected 16 frozen test roots, found {test_ids.numel()}")
    return test_ids, expected


def body_positions(state):
    return torch.stack(
        [state[..., 0:2], state[..., 4:6], state[..., 8:10]], dim=-2
    )


@torch.no_grad()
def simulator_trajectory(task, snapshot, candidates, action_block, device):
    """Evaluate candidates in VMAS after planning, retaining structured truth."""
    roots, count, horizon, plan_dim = candidates.shape
    primitive = unpack_actions(candidates.to(device), action_block)
    agents = 2
    action_dim = primitive.shape[-1] // agents
    primitive = primitive.reshape(
        roots * count, horizon * action_block, agents, action_dim
    )
    env = task.get_env_fun(roots * count, True, 0, device)()
    env.reset()
    source = torch.arange(roots, device=device).repeat_interleave(count)
    broadcast_state(env, snapshot, source_indices=source)
    outcome = task_outcome("vmas/buzz_wire")
    td = TensorDict({}, batch_size=[roots * count], device=device)
    live = ~env._env.done()
    frozen_state = live_structured_state(env).clone()
    states = [frozen_state.cpu()]
    rewards, collisions, clearances = [], [], []
    live_starts, dynamics_valid = [], []
    try:
        for block in range(horizon):
            started = live.clone()
            through = live.clone()
            reward_sum = torch.zeros(roots * count, device=device)
            collided = torch.zeros_like(live)
            minimum = torch.full_like(reward_sum, float("inf"))
            for offset in range(action_block):
                before = live.clone()
                step = block * action_block + offset
                td.set(("agents", "action"), primitive[:, step])
                td = env.step(td)["next"]
                reward = td["agents", "reward"].sum(dim=1).squeeze(-1)
                reward_sum += reward.masked_fill(~before, 0)
                _success, crash, _distance = outcome(env)
                collided |= before & crash
                current = live_structured_state(env)
                clearance = wire_clearance(body_positions(current)).min(dim=-1).values
                minimum = torch.minimum(
                    minimum, clearance.masked_fill(~before, float("inf"))
                )
                frozen_state = torch.where(before[:, None], current, frozen_state)
                live = before & ~td["done"].squeeze(-1)
                through &= live
            states.append(frozen_state.cpu())
            rewards.append(reward_sum.cpu())
            collisions.append(collided.cpu())
            clearances.append(
                minimum.masked_fill(~started, 0).cpu()
            )
            live_starts.append(started.cpu())
            dynamics_valid.append(through.cpu())
    finally:
        env.close()

    def shaped(value, frames=False):
        stacked = torch.stack(value, dim=1)
        length = horizon + 1 if frames else horizon
        return stacked.reshape(roots, count, length, *stacked.shape[2:])

    block_reward = shaped(rewards)
    return {
        "state": shaped(states, frames=True),
        "block_reward": block_reward,
        "cost": -block_reward.cumsum(dim=-1),
        "collision": shaped(collisions),
        "minimum_clearance": shaped(clearances),
        "live_start": shaped(live_starts),
        "dynamics_valid": shaped(dynamics_valid),
    }


@torch.no_grad()
def surrogate_trajectories(model, truth, candidates, action_block, device):
    roots, count, horizon, plan_dim = candidates.shape
    agents = 2
    primitive = plan_dim // (agents * action_block)
    actions = candidates.reshape(
        roots, count, horizon, action_block, agents, primitive
    )
    actions = actions.permute(0, 1, 2, 4, 3, 5).reshape(
        roots * count, horizon, agents, action_block * primitive
    ).to(device)
    true_state = truth["state"].reshape(
        roots * count, horizon + 1, STATE_DIM
    ).to(device)
    recursive_state = true_state[:, 0]
    recursive_states = [recursive_state.cpu()]
    result = {
        "teacher_forced": defaultdict(list),
        "recursive": defaultdict(list),
    }
    survival = {
        "teacher_forced": torch.ones(roots * count, device=device),
        "recursive": torch.ones(roots * count, device=device),
    }
    total = {
        "teacher_forced": torch.zeros(roots * count, device=device),
        "recursive": torch.zeros(roots * count, device=device),
    }
    for step in range(horizon):
        inputs = {
            "teacher_forced": true_state[:, step],
            "recursive": recursive_state,
        }
        for name, state in inputs.items():
            next_state, clearance, logits, penalty = model(state, actions[:, step])
            probability = torch.sigmoid(logits)
            progress = (
                torch.linalg.vector_norm(state[:, 8:10] - state[:, 12:14], dim=-1)
                - torch.linalg.vector_norm(
                    next_state[:, 8:10] - next_state[:, 12:14], dim=-1
                )
            )
            reward = 2 * progress - 20 * probability
            total[name] += survival[name] * reward
            survival[name] *= 1 - probability
            result[name]["next_state"].append(next_state.cpu())
            result[name]["minimum_clearance"].append(clearance.cpu())
            result[name]["collision_probability"].append(probability.cpu())
            result[name]["collision_penalty"].append(penalty.cpu())
            result[name]["cost"].append((-total[name]).cpu())
            if name == "recursive":
                recursive_state = next_state
                recursive_states.append(next_state.cpu())

    output = {}
    for name, values in result.items():
        output[name] = {
            key: torch.stack(items, dim=1).reshape(
                roots, count, horizon, *items[0].shape[1:]
            )
            for key, items in values.items()
        }
    output["recursive"]["state"] = torch.stack(
        recursive_states, dim=1
    ).reshape(roots, count, horizon + 1, STATE_DIM)
    return output


def vector_rmse(error, mask, coordinate=False):
    selected = error[mask]
    if not selected.numel():
        return None
    value = selected.square().mean() if coordinate else selected.square().sum(-1).mean()
    return float(value.sqrt())


def scalar_rmse(error, mask):
    selected = error[mask]
    return None if not selected.numel() else float(selected.square().mean().sqrt())


def calibration(probability, collision, bins=10):
    probability = probability.flatten().float()
    collision = collision.flatten().float()
    brier = float((probability - collision).square().mean())
    ece = 0.0
    for index in range(bins):
        low, high = index / bins, (index + 1) / bins
        mask = (probability >= low) & (
            probability <= high if index == bins - 1 else probability < high
        )
        if mask.any():
            ece += float(mask.float().mean()) * abs(
                float(probability[mask].mean()) - float(collision[mask].mean())
            )
    positive = collision.bool()
    false_negative = (
        None
        if not positive.any()
        else float((probability[positive] < 0.5).float().mean())
    )
    return {
        "collision_brier": brier,
        "collision_ece_10bin": ece,
        "collision_false_negative_rate_at_0.5": false_negative,
        "true_collision_rate": float(collision.mean()),
        "predicted_collision_probability": float(probability.mean()),
    }


def rank_and_tail_metrics(predicted, truth, collision, predicted_collision, oracle):
    roots, count = predicted.shape
    per_root = []
    for root in range(roots):
        pred, real = predicted[root], truth[root]
        selected = int(pred.argmin())
        rho = spearman(pred, real)
        item = {
            "spearman": rho if math.isfinite(rho) else None,
            "selected_regret": float(real[selected] - real.min()),
            "selected_true_cost": float(real[selected]),
            "selected_true_collision": bool(collision[root, selected]),
            "oracle_percentile": float((pred < oracle[root]).float().mean()),
        }
        for k in TOPKS:
            width = min(k, count)
            predicted_top = torch.topk(pred, width, largest=False).indices
            true_top = torch.topk(real, width, largest=False).indices
            overlap = len(set(predicted_top.tolist()) & set(true_top.tolist()))
            union = len(set(predicted_top.tolist()) | set(true_top.tolist()))
            item[f"top{k}_recall"] = overlap / width
            item[f"top{k}_elite_jaccard"] = overlap / union
            item[f"true_collision_given_predicted_top{k}"] = float(
                collision[root, predicted_top].float().mean()
            )
            item[f"predicted_collision_given_predicted_top{k}"] = float(
                predicted_collision[root, predicted_top].mean()
            )
        per_root.append(item)
    keys = per_root[0].keys()
    aggregate = {}
    for key in keys:
        values = [row[key] for row in per_root if row[key] is not None]
        aggregate[key] = (
            sum(float(value) for value in values) / len(values) if values else None
        )
    aggregate["roots"] = roots
    aggregate["candidates"] = count
    aggregate["per_root"] = per_root
    return aggregate


def path_metrics(path, comparison_state, truth, horizon, population_count):
    index = horizon - 1
    state = path["next_state"][:, :population_count, index]
    target = comparison_state[:, :population_count, index + 1]
    valid = truth["dynamics_valid"][:, :population_count, index]
    outcome = truth["live_start"][:, :population_count, index]
    clearance_error = (
        path["minimum_clearance"][:, :population_count, index]
        - truth["minimum_clearance"][:, :population_count, index]
    )
    return {
        "agent_state_coordinate_rmse": vector_rmse(
            state[..., :8] - target[..., :8], valid, coordinate=True
        ),
        "ball_position_rmse": vector_rmse(
            state[..., 8:10] - target[..., 8:10], valid
        ),
        "ball_velocity_rmse": vector_rmse(
            state[..., 10:12] - target[..., 10:12], valid
        ),
        "clearance_rmse": scalar_rmse(clearance_error, outcome),
    }


def analyze_stage(seed, iteration, truth, paths, population_count):
    rows = []
    for horizon in range(1, truth["block_reward"].shape[-1] + 1):
        true_cost = truth["cost"][:, :population_count, horizon - 1]
        true_collision = truth["collision"][
            :, :population_count, :horizon
        ].any(dim=-1)
        row = {"seed": seed, "iteration": iteration, "horizon": horizon}
        for name in ("teacher_forced", "recursive"):
            path = paths[name]
            predicted = path["cost"][:, :population_count, horizon - 1]
            probability = 1 - torch.prod(
                1 - path["collision_probability"][:, :population_count, :horizon],
                dim=-1,
            )
            oracle_cost = path["cost"][:, population_count, horizon - 1]
            metrics = path_metrics(path, truth["state"], truth, horizon, population_count)
            metrics["cost_rmse"] = float((predicted - true_cost).square().mean().sqrt())
            metrics.update(calibration(probability, true_collision))
            metrics.update(
                rank_and_tail_metrics(
                    predicted, true_cost, true_collision, probability, oracle_cost
                )
            )
            row[name] = metrics

        # The gap between recursive and teacher-forced predictions is the
        # accumulation term.  It is not a second comparison to simulator truth.
        recursive = paths["recursive"]
        teacher = paths["teacher_forced"]
        index = horizon - 1
        valid = truth["dynamics_valid"][:, :population_count, index]
        outcome = truth["live_start"][:, :population_count, index]
        row["recursive_vs_teacher_forced"] = {
            "agent_state_coordinate_rmse": vector_rmse(
                recursive["next_state"][:, :population_count, index, :8]
                - teacher["next_state"][:, :population_count, index, :8],
                valid,
                coordinate=True,
            ),
            "ball_position_rmse": vector_rmse(
                recursive["next_state"][:, :population_count, index, 8:10]
                - teacher["next_state"][:, :population_count, index, 8:10],
                valid,
            ),
            "ball_velocity_rmse": vector_rmse(
                recursive["next_state"][:, :population_count, index, 10:12]
                - teacher["next_state"][:, :population_count, index, 10:12],
                valid,
            ),
            "clearance_rmse": scalar_rmse(
                recursive["minimum_clearance"][:, :population_count, index]
                - teacher["minimum_clearance"][:, :population_count, index],
                outcome,
            ),
            "cost_rmse": float(
                (
                    recursive["cost"][:, :population_count, index]
                    - teacher["cost"][:, :population_count, index]
                ).square().mean().sqrt()
            ),
        }
        rows.append(row)
    return rows


def summarize(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["iteration"], row["horizon"]].append(row)
    output = []
    keys = (
        "ball_position_rmse",
        "ball_velocity_rmse",
        "agent_state_coordinate_rmse",
        "clearance_rmse",
        "cost_rmse",
        "collision_brier",
        "collision_ece_10bin",
        "collision_false_negative_rate_at_0.5",
        "spearman",
        "top10_recall",
        "top30_recall",
        "top10_elite_jaccard",
        "top30_elite_jaccard",
        "oracle_percentile",
        "selected_regret",
        "true_collision_given_predicted_top1",
        "true_collision_given_predicted_top5",
        "true_collision_given_predicted_top10",
        "true_collision_given_predicted_top30",
        "predicted_collision_given_predicted_top1",
        "predicted_collision_given_predicted_top5",
        "predicted_collision_given_predicted_top10",
        "predicted_collision_given_predicted_top30",
    )
    gap_keys = (
        "ball_position_rmse",
        "ball_velocity_rmse",
        "agent_state_coordinate_rmse",
        "clearance_rmse",
        "cost_rmse",
    )
    for (iteration, horizon), items in sorted(grouped.items()):
        row = {"iteration": iteration, "horizon": horizon}
        for path in ("teacher_forced", "recursive"):
            row[path] = {}
            for key in keys:
                values = [item[path].get(key) for item in items]
                values = [value for value in values if value is not None]
                row[path][key] = sum(values) / len(values) if values else None
        row["recursive_vs_teacher_forced"] = {
            key: sum(item["recursive_vs_teacher_forced"][key] for item in items)
            / len(items)
            for key in gap_keys
        }
        output.append(row)
    return output


def registered_branch_result(rows):
    indexed = {
        (row["seed"], row["iteration"], row["horizon"]): row for row in rows
    }
    seeds = sorted({row["seed"] for row in rows})
    ood_seeds = []
    recursion_seeds = []
    false_safe_seeds = []
    ood_ratios = {}
    for seed in seeds:
        qualifying = 0
        ood_ratios[str(seed)] = {}
        for horizon in range(1, 6):
            early = indexed[seed, 1, horizon]["teacher_forced"]
            late = indexed[seed, 30, horizon]["teacher_forced"]
            ratios = {}
            for key in ("ball_position_rmse", "clearance_rmse", "cost_rmse"):
                denominator = early[key]
                ratios[key] = (
                    None
                    if denominator is None or denominator <= 0
                    else late[key] / denominator
                )
            ood_ratios[str(seed)][str(horizon)] = ratios
            finite = [value for value in ratios.values() if value is not None]
            qualifying += int(bool(finite) and max(finite) >= 1.5)
        if qualifying >= 3:
            ood_seeds.append(seed)

        final = indexed[seed, 30, 5]
        teacher = final["teacher_forced"]["ball_position_rmse"]
        recursive = final["recursive"]["ball_position_rmse"]
        gap = final["recursive_vs_teacher_forced"]["ball_position_rmse"]
        if (
            teacher is not None
            and recursive is not None
            and gap is not None
            and recursive >= 2 * teacher
            and gap >= 0.01
        ):
            recursion_seeds.append(seed)

        true_tail = final["recursive"]["true_collision_given_predicted_top10"]
        predicted_tail = final["recursive"][
            "predicted_collision_given_predicted_top10"
        ]
        if true_tail >= 0.25 and true_tail - predicted_tail >= 0.15:
            false_safe_seeds.append(seed)

    fired = {
        "G6a_planner_aware_aggregation": len(ood_seeds) >= 2,
        "G6b_action_prefix_dynamics": len(recursion_seeds) >= 2,
        "G6c_conservative_collision_scoring": len(false_safe_seeds) >= 2,
    }
    order = list(fired)
    primary = next((name for name in order if fired[name]), "audit_before_training")
    return {
        "primary_next_intervention": primary,
        "all_rules": fired,
        "passing_seeds": {
            "G6a_planner_aware_aggregation": ood_seeds,
            "G6b_action_prefix_dynamics": recursion_seeds,
            "G6c_conservative_collision_scoring": false_safe_seeds,
        },
        "ood_late_over_early_ratios": ood_ratios,
        "priority_if_multiple_fire": order,
    }


def render(summary, output):
    def series(path, key, horizon):
        rows = [row for row in summary if row["horizon"] == horizon]
        return [row["iteration"] for row in rows], [row[path][key] for row in rows]

    colours = plt.cm.viridis(torch.linspace(0.05, 0.95, 5).numpy())
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    for horizon, colour in zip(range(1, 6), colours):
        for axis, path, title in (
            (axes[0], "teacher_forced", "Simulator vs teacher-forced"),
            (axes[1], "recursive", "Simulator vs recursive"),
            (axes[2], "recursive_vs_teacher_forced", "Recursive accumulation"),
        ):
            x, y = series(path, "ball_position_rmse", horizon)
            axis.plot(x, y, marker="o", color=colour, label=f"h={horizon}")
            axis.set_title(title)
            axis.set_xlabel("CEM iteration")
            axis.set_ylabel("ball-position RMSE (m)")
            axis.grid(alpha=0.25)
    axes[0].legend(ncol=2, fontsize=8)
    figure.suptitle("Gate 5: physical error across the planner-induced tail")
    figure.savefig(output / "tail_physics.png", dpi=180)
    figure.savefig(output / "tail_physics.pdf")
    plt.close(figure)

    rows = [row for row in summary if row["horizon"] == 5]
    x = [row["iteration"] for row in rows]
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    axes[0].plot(x, [row["recursive"]["spearman"] for row in rows], marker="o")
    axes[0].set_title("Recursive Plan--Real Spearman, h=5")
    axes[1].plot(
        x, [row["recursive"]["oracle_percentile"] for row in rows], marker="o"
    )
    axes[1].set_title("Oracle percentile (lower is better), h=5")
    for k in TOPKS:
        axes[2].plot(
            x,
            [row["recursive"][f"true_collision_given_predicted_top{k}"] for row in rows],
            marker="o",
            label=f"top {k}",
        )
    axes[2].set_title("P(true collision | predicted top-k), h=5")
    axes[2].legend(fontsize=8)
    for axis in axes:
        axis.set_xlabel("CEM iteration")
        axis.grid(alpha=0.25)
    figure.suptitle("Gate 5: decision quality in successive CEM populations")
    figure.savefig(output / "tail_decisions.png", dpi=180)
    figure.savefig(output / "tail_decisions.pdf")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source", type=Path, default=Path("outputs/structured_surrogate_1331")
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    prior = json.loads((args.source / "result/result.json").read_text())
    stored = torch.load(
        args.source / "coverage/oracle_plans.pt", map_location="cpu", weights_only=False
    )
    seeds = (9100, 9101, 9102)
    test_ids, frozen = validate_frozen_protocol(prior, stored, seeds)
    contract = information_contract()
    block = 5
    task = VmasTask.BUZZ_WIRE.get_from_yaml()
    snapshot = take_snapshot(stored["snapshot"], test_ids, args.device)
    initial_state = bank_state(task, snapshot, args.device)
    oracle = stored["plans"][test_ids]

    env = task.get_env_fun(test_ids.numel(), True, 0, args.device)()
    env.reset()
    low, high = action_bounds(env)
    env.close()
    cem_config = CEMConfig(
        horizon=frozen["horizon"],
        num_samples=frozen["num_samples"],
        num_elites=frozen["num_elites"],
        num_iters=frozen["num_iters"],
    )

    all_rows = []
    for seed in seeds:
        model, _checkpoint = load_surrogate(
            args.source / f"result/model_full_{seed}.pt", args.device
        )

        def cost_fn(candidates):
            # Only the frozen initial P-state, actions and surrogate enter this
            # function.  No simulator rollout or future T-value is in scope.
            primitive = unpack_actions(candidates, block)
            return surrogate_cost(
                model, initial_state, primitive, block, objective="probability"
            )

        result = cem_plan(
            cost_fn,
            action_dim=20,
            action_low=low,
            action_high=high,
            config=cem_config,
            batch_size=test_ids.numel(),
            device=args.device,
            generator=torch.Generator(device=args.device).manual_seed(
                frozen["control_seed"]
            ),
            record_candidates=True,
        )
        population = result.candidate_history[
            torch.tensor([stage - 1 for stage in STAGES])
        ]
        torch.save(
            {
                "seed": seed,
                "iterations": STAGES,
                "candidates": population,
                "test_root_ids": test_ids,
                "planner_used_future_truth": False,
            },
            args.output / f"populations_{seed}.pt",
        )

        for stage_index, iteration in enumerate(STAGES):
            candidates = population[stage_index]
            # Oracle is appended only after planning.  It cannot affect the CEM
            # proposal update or elite selection and exists solely to measure
            # its percentile in the frozen candidate population.
            evaluated = torch.cat([candidates, oracle[:, None]], dim=1)
            truth = simulator_trajectory(
                task, snapshot, evaluated, block, args.device
            )
            paths = surrogate_trajectories(
                model, truth, evaluated, block, args.device
            )
            population_count = candidates.shape[1]
            recomputed = paths["recursive"]["cost"][
                :, :population_count, -1
            ]
            expected = cost_fn(candidates.to(args.device)).cpu()
            difference = float((recomputed - expected).abs().max())
            if difference > 2e-5:
                raise ValueError(
                    f"Recorded recursive cost differs from planner by {difference}"
                )
            all_rows.extend(
                analyze_stage(seed, iteration, truth, paths, population_count)
            )
            torch.save(
                {
                    "seed": seed,
                    "iteration": iteration,
                    "candidates": evaluated,
                    "candidate_roles": {
                        "0:300": "CEM population",
                        "300": "post-planning oracle diagnostic",
                    },
                    "simulator": truth,
                    "teacher_forced": paths["teacher_forced"],
                    "recursive": paths["recursive"],
                    "planner_cost_max_abs_check": difference,
                },
                args.output / f"trajectories_seed{seed}_iter{iteration}.pt",
            )
            print(f"seed {seed} iteration {iteration} complete", flush=True)

    summary = summarize(all_rows)
    branch = registered_branch_result(all_rows)
    result = {
        "question": (
            "Does the structured planner fail from one-step OOD error, recursive "
            "accumulation, or false-safe collision tails?"
        ),
        "source_job": 1331,
        "information_contract": contract,
        "frozen_protocol": {
            **frozen,
            "action_block": block,
            "model_seeds": seeds,
            "test_root_ids": test_ids.tolist(),
            "cem_iterations_analyzed": STAGES,
            "topk": TOPKS,
            "objective": "probability",
            "retraining": False,
        },
        "rows": all_rows,
        "summary_across_model_seeds": summary,
        "registered_branch_result": branch,
    }
    write_json(args.output / "result.json", result)
    render(summary, args.output)
    write_json(args.output / "information_contract.json", contract)
    print(f"wrote Gate 5 results to {args.output}", flush=True)


if __name__ == "__main__":
    main()
