# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the license found in the LICENSE file in the root directory.
"""Planner-induced coverage and a 14-D privileged Buzz Wire diagnostic.

This is Baseline B after Gate 0d.  It deliberately removes both learned-latent
coordinates and scalar reward prediction:

* collection queries the frozen seed-4100 planner at CEM iterations
  1/5/10/20/30, then executes its preferred candidates in VMAS;
* competent true-dynamics plans and local joint-action perturbations are added;
* a small surrogate predicts the selected 14-D state, within-block clearance,
  and collision probability;
* CEM scores rollouts with Buzz Wire's known progress and collision cost.

Commands are separate so the launcher enforces collection before training:

``collect`` writes a split-safe coverage bank. ``run`` fits behavior-only and
full-mixture controls, evaluates held-out plan ranking, then runs receding-
horizon MPC on the frozen test roots.
"""

import argparse
import json
import math
import subprocess
from collections import defaultdict
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
import yaml
from tensordict import TensorDict
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from benchmarl.environments import VmasTask
from examples.world_model.cem import CEMConfig, cem_plan
from examples.world_model.closed_loop import summarize
from examples.world_model.collect import SPLITS, branch_rollouts
from examples.world_model.cost_landscape import true_scores
from examples.world_model.decision_information import (
    blocked_actions,
    checkpoint_rows,
    rank_metrics,
    reward_cost,
    rollout_truth,
    take_snapshot,
    tree_map,
)
from examples.world_model.model_input import compose_frames, entity_frame
from examples.world_model.mpc import (
    MPCConfig,
    action_bounds,
    evaluate_policy,
    task_outcome,
    unpack_actions,
)
from examples.world_model.snapshot_restore import (
    agent_observations,
    physical_state,
    restore_state,
)
from examples.world_model.train import load_model


FAMILY_NAMES = ("behavior", "competent_local", "cem_hard")
HARD_STAGES = (1, 5, 10, 20, 30)
# This legacy diagnostic omits the two movable linkage bodies. An audit
# counterexample now proves it is not Markov-sufficient; retain its dimensions
# only so jobs 1331--1337 remain reproducible while a full-state successor is
# implemented separately.
STATE_DIM = 14  # two agents pos/vel, ball pos/vel, goal position
DYNAMIC_DIM = 12


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def wire_clearance(position):
    """Signed sphere clearance to Buzz Wire's rectangular boundary."""
    horizontal = 0.125 - position[..., 0].abs() - 0.03
    vertical = 1.0 - position[..., 1].abs() - 0.03
    return torch.minimum(horizontal, vertical)


def structured_state(agent_state, package_state, observation=None, goal=None):
    """Build [...,14] task state from recorded or live physical tensors."""
    agents = agent_state[..., :4].flatten(-2)
    ball = package_state[..., 0, :4]
    if goal is None:
        if observation is None:
            raise ValueError("observation or goal is required")
        goal = (agent_state[..., :2] - observation[..., 4:6]).mean(dim=-2)
    return torch.cat([agents, ball, goal], dim=-1)


def live_structured_state(env):
    agents = physical_state(env._env.world.agents)
    ball = physical_state([env._env.scenario.ball])
    goal = env._env.scenario.goal.state.pos
    return structured_state(agents, ball, goal=goal)


def blockify(data, split, family, source, root_id, stage, action_block):
    """Convert primitive VMAS trajectories to task-labelled block transitions."""
    count, steps, agents, action_dim = data["action"].shape
    if steps % action_block:
        raise ValueError("Primitive trajectory length must divide into blocks")
    blocks = steps // action_block
    start = torch.arange(0, steps, action_block)
    goal = (
        data["agent_state"][:, start, :, :2]
        - data["observation"][:, start, :, 4:6]
    ).mean(dim=2)
    state = structured_state(
        data["agent_state"][:, start],
        data["package_state"][:, start],
        goal=goal,
    )

    valid = data["valid"].reshape(count, blocks, action_block)
    outcome_valid = valid.any(dim=2)
    dynamics_valid = valid.all(dim=2)
    length = valid.sum(dim=2).clamp_min(1)
    endpoint_index = (length - 1).view(count, blocks, 1, 1, 1)

    next_agent_steps = data["next_agent_state"].reshape(
        count, blocks, action_block, agents, 6
    )
    next_package_steps = data["next_package_state"].reshape(
        count, blocks, action_block, data["package_state"].shape[2], 6
    )
    end_agent = next_agent_steps.gather(
        2, endpoint_index.expand(count, blocks, 1, agents, 6)
    ).squeeze(2)
    end_package = next_package_steps.gather(
        2,
        endpoint_index.expand(
            count, blocks, 1, next_package_steps.shape[3], 6
        ),
    ).squeeze(2)
    next_state = structured_state(end_agent, end_package, goal=goal)

    primitive_action = data["action"].reshape(
        count, blocks, action_block, agents, action_dim
    )
    action = primitive_action.permute(0, 1, 3, 2, 4).reshape(
        count, blocks, agents, action_block * action_dim
    )
    reward = data["reward"].reshape(
        count, blocks, action_block, agents, 1
    )
    team_reward = reward.sum(dim=(2, 3, 4))
    collision = (reward < -1.0).any(dim=(2, 3, 4))

    positions = torch.cat(
        [next_agent_steps[..., :2], next_package_steps[..., :1, :2]], dim=3
    )
    primitive_clearance = wire_clearance(positions).min(dim=3).values
    primitive_clearance = primitive_clearance.masked_fill(~valid, float("inf"))
    minimum_clearance = primitive_clearance.min(dim=2).values

    start_distance = torch.linalg.vector_norm(state[..., 8:10] - goal, dim=-1)
    end_distance = torch.linalg.vector_norm(next_state[..., 8:10] - goal, dim=-1)
    progress = start_distance - end_distance
    collision_cost = (2 * progress - team_reward).clamp_min(0)

    def trajectory_field(value, dtype=torch.long):
        value = torch.as_tensor(value, dtype=dtype)
        if value.ndim == 0:
            value = value.expand(count)
        if value.shape != (count,):
            raise ValueError(f"Metadata must have {count} trajectory rows")
        return value[:, None].expand(count, blocks)

    flat_mask = outcome_valid.flatten()
    result = {
        "state": state.reshape(-1, STATE_DIM)[flat_mask],
        "action": action.reshape(-1, agents, action_block * action_dim)[flat_mask],
        "next_state": next_state.reshape(-1, STATE_DIM)[flat_mask],
        "dynamics_valid": dynamics_valid.flatten()[flat_mask],
        "minimum_clearance": minimum_clearance.flatten()[flat_mask],
        "collision": collision.flatten()[flat_mask],
        "collision_cost": collision_cost.flatten()[flat_mask],
        "team_reward": team_reward.flatten()[flat_mask],
        "progress": progress.flatten()[flat_mask],
        "split": trajectory_field(split).flatten()[flat_mask],
        "family": trajectory_field(family).flatten()[flat_mask],
        "source": trajectory_field(source).flatten()[flat_mask],
        "root_id": trajectory_field(root_id).flatten()[flat_mask],
        "stage": trajectory_field(stage).flatten()[flat_mask],
    }
    if not all(torch.isfinite(value).all() for key, value in result.items() if value.is_floating_point()):
        raise ValueError("Non-finite structured transition")
    return result


def cat_records(records):
    keys = records[0].keys()
    return {key: torch.cat([record[key] for record in records]) for key in keys}


def repeat_snapshot(snapshot, repeats):
    return tree_map(snapshot, lambda value: value.repeat_interleave(repeats, dim=0))


@torch.no_grad()
def replay_blocked(task, snapshot, plans, action_block, batch_size, device):
    roots, count, horizon, plan_dim = plans.shape
    expanded = repeat_snapshot(snapshot, count)
    primitive = unpack_actions(plans.to(device), action_block)
    agents = 2
    action_dim = primitive.shape[-1] // agents
    actions = primitive.reshape(
        roots * count, horizon * action_block, agents, action_dim
    )
    return branch_rollouts(
        task, {"snapshot": expanded}, actions, batch_size, device
    )


@torch.no_grad()
def learned_costs(model, observation, candidates, action_block, device):
    batch, count = candidates.shape[:2]
    agents, obs_dim = observation.shape[1:]
    actions = blocked_actions(candidates, agents, action_block).to(device)
    start = observation[:, None].expand(batch, count, agents, obs_dim)
    start = start.reshape(batch * count, 1, agents, obs_dim).to(device)
    initial = model.encode(start)
    sequence = torch.cat([initial, model.rollout(initial, actions)], dim=1)
    return reward_cost(model, sequence).reshape(batch, count)


def root_model_inputs(task, snapshot, device):
    roots = snapshot["steps"].shape[0]
    env = task.get_env_fun(roots, True, 0, device)()
    env.reset()
    restore_state(env, snapshot)
    observation = agent_observations(env).cpu()
    entities = entity_frame(env).cpu()
    low, high = action_bounds(env)
    agents = len(env._env.world.agents)
    primitive = env.full_action_spec_unbatched["agents", "action"].shape[-1]
    env.close()
    model_input = compose_frames(
        "physical", observation.unsqueeze(1), entities.unsqueeze(1), 3
    )[:, 0]
    return model_input, agents, primitive, low, high


def trajectory_summary(data, stage):
    valid = data["valid"]
    total = data["reward"].sum(dim=(1, 2, 3))
    collided = (data["reward"] < -1.0).any(dim=(1, 2, 3))
    length = valid.sum(dim=1).clamp_min(1)
    row = torch.arange(valid.shape[0])
    end = length - 1
    ball = data["next_package_state"][row, end, 0, :2]
    goal = (
        data["agent_state"][:, 0, :, :2]
        - data["observation"][:, 0, :, 4:6]
    ).mean(dim=1)
    distance = torch.linalg.vector_norm(ball - goal, dim=-1)
    result = {}
    for value in torch.as_tensor(stage).unique(sorted=True):
        mask = torch.as_tensor(stage) == value
        result[str(int(value))] = {
            "trajectories": int(mask.sum()),
            "return": float(total[mask].mean()),
            "collision_rate": float(collided[mask].float().mean()),
            "final_distance": float(distance[mask].mean()),
            "mean_length": float(length[mask].float().mean()),
        }
    return result


def choose_anchor_ids(anchors, limits, seed):
    selected = []
    generator = torch.Generator().manual_seed(seed)
    for split, limit in enumerate(limits):
        available = (anchors["split"] == split).nonzero(as_tuple=True)[0]
        order = torch.randperm(available.numel(), generator=generator)
        selected.append(available[order[: min(limit, available.numel())]])
    return torch.cat(selected)


def collect_coverage(args):
    started = perf_counter()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "coverage.pt").exists():
        raise FileExistsError("Refusing to overwrite coverage.pt")
    manifest = json.loads((args.data / "manifest.json").read_text())
    if manifest["task_name"] != "vmas/buzz_wire":
        raise ValueError("Structured surrogate is scoped to Buzz Wire")
    block = manifest["action_block"]
    task = VmasTask.BUZZ_WIRE.get_from_yaml()
    records, collection_summary = [], {}

    anchors = torch.load(args.data / "anchors.pt", map_location="cpu", weights_only=False)
    for source, regime in enumerate(("independent", "correlated")):
        samples = torch.load(
            args.data / f"samples_{regime}.pt", map_location="cpu", weights_only=False
        )
        split = anchors["split"][samples["anchor_id"]]
        records.append(
            blockify(
                samples,
                split,
                family=0,
                source=source,
                root_id=samples["anchor_id"],
                stage=-1,
                action_block=block,
            )
        )

    rows = checkpoint_rows(args.runs, "physical", "correlated", args.model_seed)
    run, config = next((row for row in rows if row[1]["model"]["kind"] == args.kind), (None, None))
    if run is None:
        raise ValueError(f"No physical/correlated/{args.kind} checkpoint")
    model = load_model(run / "model.pt", args.device)
    hard_ids = choose_anchor_ids(
        anchors, (args.hard_train, args.hard_validation, args.hard_test), args.seed
    )
    hard_summaries = defaultdict(list)
    for start in range(0, hard_ids.numel(), args.root_batch):
        ids = hard_ids[start : start + args.root_batch]
        snapshot = take_snapshot(anchors["snapshot"], ids, args.device)
        observation, agents, primitive, low, high = root_model_inputs(
            task, snapshot, args.device
        )
        plan_dim = agents * primitive * block

        def cost_fn(candidates):
            return learned_costs(model, observation, candidates, block, args.device)

        cem = cem_plan(
            cost_fn,
            action_dim=plan_dim,
            action_low=low,
            action_high=high,
            config=CEMConfig(
                horizon=args.horizon,
                num_samples=args.num_samples,
                num_elites=args.num_elites,
                num_iters=max(HARD_STAGES),
            ),
            batch_size=ids.numel(),
            device=args.device,
            generator=torch.Generator(device=args.device).manual_seed(args.seed + start),
            record_candidates=True,
        )
        selected, stage_values = [], []
        for stage in HARD_STAGES:
            candidates = cem.candidate_history[stage - 1]
            costs = cost_fn(candidates.to(args.device)).cpu()
            elite = costs.topk(args.hard_plans, largest=False).indices
            chosen = candidates.gather(
                1,
                elite[..., None, None].expand(
                    ids.numel(), args.hard_plans, args.horizon, plan_dim
                ),
            )
            selected.append(chosen)
            stage_values.extend([stage] * args.hard_plans)
        plans = torch.cat(selected, dim=1)
        data = replay_blocked(
            task, snapshot, plans, block, args.replay_batch, args.device
        )
        plan_count = plans.shape[1]
        split = anchors["split"][ids].repeat_interleave(plan_count)
        root_id = ids.repeat_interleave(plan_count)
        stage = torch.tensor(stage_values).repeat(ids.numel())
        records.append(
            blockify(
                data,
                split,
                family=2,
                source=4,
                root_id=root_id,
                stage=stage,
                action_block=block,
            )
        )
        summary = trajectory_summary(data, stage)
        for key, value in summary.items():
            hard_summaries[key].append(value)
        print(f"hard roots {start + ids.numel()}/{hard_ids.numel()}", flush=True)
        del cem, data
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    del model

    initial = torch.load(
        args.data / "initial_states.pt", map_location="cpu", weights_only=False
    )
    available_roots = initial["split"].numel()
    root_count = min(available_roots, args.oracle_roots or available_roots)
    oracle_plans = torch.empty(root_count, args.horizon, 20)
    competent_summaries = defaultdict(list)
    for start in range(0, root_count, args.oracle_root_batch):
        ids = torch.arange(start, min(start + args.oracle_root_batch, root_count))
        snapshot = take_snapshot(initial["snapshot"], ids, args.device)
        _observation, agents, primitive, low, high = root_model_inputs(
            task, snapshot, args.device
        )
        plan_dim = agents * primitive * block
        scratch = task.get_env_fun(ids.numel() * args.num_samples, True, 0, args.device)()
        scratch.reset()
        try:
            def oracle_cost(candidates):
                total, _distance, _collision = true_scores(
                    scratch,
                    snapshot,
                    unpack_actions(candidates, block),
                    block,
                    task_outcome(manifest["task_name"]),
                )
                return (-total).to(args.device)

            oracle_result = cem_plan(
                oracle_cost,
                action_dim=plan_dim,
                action_low=low,
                action_high=high,
                config=CEMConfig(
                    horizon=args.horizon,
                    num_samples=args.num_samples,
                    num_elites=args.num_elites,
                    num_iters=args.oracle_iters,
                ),
                batch_size=ids.numel(),
                device=args.device,
                generator=torch.Generator(device=args.device).manual_seed(
                    args.oracle_seed + start
                ),
            )
            # The CEM mean is an unscored interpolation of the final elites and
            # can cross the wire even when every elite is safe.  Collection
            # needs demonstrably competent data, so retain the best candidate
            # whose simulator cost was actually evaluated.
            best = oracle_result.costs.argmin(dim=1)
            oracle = oracle_result.candidates[
                torch.arange(ids.numel(), device=args.device), best
            ].cpu()
        finally:
            scratch.close()
        oracle_plans[ids] = oracle
        generator = torch.Generator().manual_seed(args.oracle_seed + 1000 + start)
        local = []
        for index in range(args.local_plans):
            sigma = args.local_sigma_small if index < args.local_plans // 2 else args.local_sigma_large
            noise = torch.randn(oracle.shape, generator=generator) * sigma
            local.append((oracle + noise).clamp(low, high))
        plans = torch.cat([oracle[:, None], torch.stack(local, dim=1)], dim=1)
        data = replay_blocked(
            task, snapshot, plans, block, args.replay_batch, args.device
        )
        count = plans.shape[1]
        split = initial["split"][ids].repeat_interleave(count)
        root_id = (1_000_000 + ids).repeat_interleave(count)
        variant = torch.tensor([0] + [1] * args.local_plans).repeat(ids.numel())
        records.append(
            blockify(
                data,
                split,
                family=1,
                source=2 + variant,
                root_id=root_id,
                stage=-1,
                action_block=block,
            )
        )
        summary = trajectory_summary(data, variant)
        for key, value in summary.items():
            competent_summaries[key].append(value)
        print(f"oracle roots {ids[-1].item() + 1}/{root_count}", flush=True)

    coverage = cat_records(records)
    if coverage["state"].shape[1] != STATE_DIM:
        raise ValueError("Unexpected structured state width")
    root_splits = {}
    for root, split in zip(coverage["root_id"].tolist(), coverage["split"].tolist()):
        if root in root_splits and root_splits[root] != split:
            raise ValueError("Root leaked across splits")
        root_splits[root] = split
    torch.save(coverage, output / "coverage.pt")
    torch.save(
        {
            "plans": oracle_plans,
            "split": initial["split"][:root_count],
            "snapshot": tree_map(
                initial["snapshot"], lambda value: value[:root_count]
            ),
        },
        output / "oracle_plans.pt",
    )

    def pooled(parts):
        result = {}
        for key in parts[0]:
            weight = sum(item["trajectories"] for item in parts)
            result[key] = (
                weight
                if key == "trajectories"
                else sum(item[key] * item["trajectories"] for item in parts) / weight
            )
        return result

    summary = {
        "seconds": perf_counter() - started,
        "checkpoint": str(run),
        "transitions": int(coverage["state"].shape[0]),
        "by_split_family": {
            SPLITS[split]: {
                FAMILY_NAMES[family]: int(
                    ((coverage["split"] == split) & (coverage["family"] == family)).sum()
                )
                for family in range(len(FAMILY_NAMES))
            }
            for split in range(len(SPLITS))
        },
        "collision_rate_by_family": {
            FAMILY_NAMES[family]: float(
                coverage["collision"][coverage["family"] == family].float().mean()
            )
            for family in range(len(FAMILY_NAMES))
        },
        "hard_by_cem_stage": {
            key: pooled(value) for key, value in sorted(hard_summaries.items(), key=lambda x: int(x[0]))
        },
        "competent": {
            "oracle": pooled(competent_summaries["0"]),
            "local": pooled(competent_summaries["1"]),
        },
        "root_split_leakage": False,
    }
    write_json(output / "collection_summary.json", summary)
    write_json(
        output / "manifest.json",
        {
            "task_name": manifest["task_name"],
            "action_block": block,
            "state_fields": "agent0 pos/vel, agent1 pos/vel, ball pos/vel, goal pos",
            "families": FAMILY_NAMES,
            "hard_stages": HARD_STAGES,
            "source_data": str(args.data),
            "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "config": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        },
    )
    print(json.dumps(summary, indent=2), flush=True)


class TransitionDataset(Dataset):
    def __init__(self, tensors, indices):
        self.tensors = tensors
        self.indices = indices

    def __len__(self):
        return self.indices.numel()

    def __getitem__(self, index):
        row = int(self.indices[index])
        return {key: value[row] for key, value in self.tensors.items()}


class StructuredSurrogate(nn.Module):
    def __init__(
        self,
        state_mean,
        state_std,
        action_mean,
        action_std,
        delta_mean,
        delta_std,
        clearance_mean,
        clearance_std,
        penalty_mean,
        penalty_std,
        hidden=256,
    ):
        super().__init__()
        width = state_mean.numel() + action_mean.numel()
        self.body = nn.Sequential(
            nn.Linear(width, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
        )
        self.delta = nn.Linear(hidden, DYNAMIC_DIM)
        self.clearance = nn.Linear(hidden, 1)
        self.collision = nn.Linear(hidden, 1)
        self.penalty = nn.Linear(hidden, 1)
        for name, value in (
            ("state_mean", state_mean), ("state_std", state_std),
            ("action_mean", action_mean), ("action_std", action_std),
            ("delta_mean", delta_mean), ("delta_std", delta_std),
            ("clearance_mean", clearance_mean.reshape(1)),
            ("clearance_std", clearance_std.reshape(1)),
            ("penalty_mean", penalty_mean.reshape(1)),
            ("penalty_std", penalty_std.reshape(1)),
        ):
            self.register_buffer(name, value.clone().float())

    def forward(self, state, action):
        action = action.flatten(-2)
        features = torch.cat(
            [(state - self.state_mean) / self.state_std, (action - self.action_mean) / self.action_std],
            dim=-1,
        )
        hidden = self.body(features)
        delta = self.delta(hidden) * self.delta_std + self.delta_mean
        clearance = self.clearance(hidden).squeeze(-1) * self.clearance_std + self.clearance_mean
        collision = self.collision(hidden).squeeze(-1)
        penalty = self.penalty(hidden).squeeze(-1) * self.penalty_std + self.penalty_mean
        next_state = torch.cat([state[..., :DYNAMIC_DIM] + delta, state[..., DYNAMIC_DIM:]], dim=-1)
        return next_state, clearance, collision, penalty


def statistics(data, mask):
    valid = mask & data["dynamics_valid"]
    state = data["state"][mask]
    action = data["action"][mask].flatten(1)
    delta = data["next_state"][valid, :DYNAMIC_DIM] - data["state"][valid, :DYNAMIC_DIM]

    def stats(value):
        return value.mean(0), value.std(0, unbiased=False).clamp_min(1e-5)

    state_mean, state_std = stats(state)
    action_mean, action_std = stats(action)
    delta_mean, delta_std = stats(delta)
    clearance_mean, clearance_std = stats(data["minimum_clearance"][mask])
    penalty_mean, penalty_std = stats(data["collision_cost"][mask])
    return (
        state_mean,
        state_std,
        action_mean,
        action_std,
        delta_mean,
        delta_std,
        clearance_mean,
        clearance_std,
        penalty_mean,
        penalty_std,
    )


def batch_loss(model, batch, pos_weight):
    state, action = batch["state"], batch["action"]
    predicted, clearance, logits, penalty = model(state, action)
    valid = batch["dynamics_valid"]
    dynamics = (((predicted[..., :DYNAMIC_DIM] - batch["next_state"][..., :DYNAMIC_DIM]) / model.delta_std).square()[valid]).mean()
    clearance_loss = (((clearance - batch["minimum_clearance"]) / model.clearance_std).square()).mean()
    collision = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, batch["collision"].float(), pos_weight=pos_weight
    )
    penalty_loss = (
        (penalty - batch["collision_cost"]) / model.penalty_std
    ).square().mean()
    return (
        dynamics + clearance_loss + collision + penalty_loss,
        dynamics,
        clearance_loss,
        collision,
        penalty_loss,
    )


def binary_auc(score, target):
    positive, negative = score[target], score[~target]
    if not positive.numel() or not negative.numel():
        return None
    return float(((positive[:, None] > negative).float() + 0.5 * (positive[:, None] == negative).float()).mean())


@torch.no_grad()
def evaluate_transitions(model, data, indices, device):
    batch = {key: value[indices].to(device) for key, value in data.items()}
    predicted, clearance, logits, penalty = model(batch["state"], batch["action"])
    valid = batch["dynamics_valid"]
    error = predicted[valid, :DYNAMIC_DIM] - batch["next_state"][valid, :DYNAMIC_DIM]
    collision = batch["collision"].bool()
    probability = torch.sigmoid(logits)
    result = {
        "dynamic_coordinate_rmse": float(error.square().mean().sqrt()),
        "ball_position_rmse": float(error[:, 8:10].square().sum(-1).mean().sqrt()),
        "clearance_rmse": float((clearance - batch["minimum_clearance"]).square().mean().sqrt()),
        "collision_brier": float((probability - collision.float()).square().mean()),
        "collision_auc": binary_auc(probability, collision),
        "collision_penalty_rmse": float(
            (penalty - batch["collision_cost"]).square().mean().sqrt()
        ),
        "collision_rate": float(collision.float().mean()),
        "transitions": int(indices.numel()),
    }
    return result


def training_sampling_weights(
    data,
    indices,
    scheme="family_balanced",
    augmentation_family=None,
):
    """Return per-transition weights and the registered expected group mass.

    Historical structured-surrogate runs balance the three collection families.
    G6a instead needs a causal base-vs-augmentation comparison: both targeted
    and generic additions receive exactly half of the expected optimizer draws,
    independent of their transition count or semantic source.
    """
    family = data["family"][indices]
    if scheme == "family_balanced":
        counts = torch.bincount(family).clamp_min(1)
        weights = (1.0 / counts[family]).double()
        present = torch.unique(family, sorted=True)
        mass = {
            f"family_{int(value)}": 1.0 / present.numel()
            for value in present
        }
        return weights, mass
    if scheme != "base_augmentation":
        raise ValueError(f"Unknown training sampling scheme: {scheme}")
    if augmentation_family is None:
        raise ValueError("base_augmentation sampling needs augmentation_family")

    augmentation = family == augmentation_family
    augmentation_count = int(augmentation.sum())
    base_count = int((~augmentation).sum())
    if not base_count or not augmentation_count:
        raise ValueError(
            "base_augmentation sampling requires both base and augmentation data"
        )
    weights = torch.empty(indices.numel(), dtype=torch.double)
    weights[~augmentation] = 0.5 / base_count
    weights[augmentation] = 0.5 / augmentation_count
    return weights, {"base": 0.5, "augmentation": 0.5}


def train_surrogate(
    data,
    mix,
    seed,
    args,
    *,
    sampling_scheme="family_balanced",
    augmentation_family=None,
    samples_per_epoch=None,
    fixed_epochs=False,
):
    eligible = data["family"] == 0 if mix == "behavior" else torch.ones_like(data["family"], dtype=torch.bool)
    train_mask = eligible & (data["split"] == 0)
    validation_mask = eligible & (data["split"] == 1)
    stats = statistics(data, train_mask)
    torch.manual_seed(seed)
    model = StructuredSurrogate(*stats, hidden=args.hidden).to(args.device)
    positives = data["collision"][train_mask].sum()
    negatives = train_mask.sum() - positives
    pos_weight = (negatives / positives.clamp_min(1)).to(args.device)
    indices = train_mask.nonzero(as_tuple=True)[0]
    weights, expected_mass = training_sampling_weights(
        data,
        indices,
        scheme=sampling_scheme,
        augmentation_family=augmentation_family,
    )
    if samples_per_epoch is None:
        samples_per_epoch = len(indices)
    if samples_per_epoch < 1:
        raise ValueError("samples_per_epoch must be positive")
    sampler = WeightedRandomSampler(
        weights,
        samples_per_epoch,
        replacement=True,
        generator=torch.Generator().manual_seed(seed),
    )
    loader = DataLoader(TransitionDataset(data, indices), batch_size=args.batch_size, sampler=sampler)
    validation_indices = validation_mask.nonzero(as_tuple=True)[0]
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    best, best_epoch, best_state, stale, history = math.inf, -1, None, 0, []
    optimizer_steps = 0
    for epoch in range(args.epochs):
        model.train()
        running, batches = 0.0, 0
        for batch in loader:
            batch = {key: value.to(args.device) for key, value in batch.items()}
            loss = batch_loss(model, batch, pos_weight)[0]
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer_steps += 1
            running += float(loss)
            batches += 1
        model.eval()
        with torch.no_grad():
            validation = {key: value[validation_indices].to(args.device) for key, value in data.items()}
            score = float(batch_loss(model, validation, pos_weight)[0])
        history.append({"epoch": epoch, "train_loss": running / batches, "validation_loss": score})
        if score < best - 1e-5:
            best, best_epoch = score, epoch
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if not fixed_epochs and stale >= args.patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model, {
        "best_epoch": best_epoch,
        "validation_loss": best,
        "epochs": len(history),
        "history": history,
        "sampling_scheme": sampling_scheme,
        "expected_sampling_mass": expected_mass,
        "samples_per_epoch": samples_per_epoch,
        "optimizer_steps": optimizer_steps,
        "fixed_epochs": fixed_epochs,
    }


@torch.no_grad()
def surrogate_cost(model, initial_state, candidates, action_block, objective="probability"):
    batch, count, primitive_steps, joint = candidates.shape
    agents = 2
    primitive = joint // agents
    horizon = primitive_steps // action_block
    action = candidates.reshape(batch, count, horizon, action_block, agents, primitive)
    action = action.permute(0, 1, 2, 4, 3, 5).reshape(batch * count, horizon, agents, action_block * primitive)
    state = initial_state[:, None].expand(batch, count, STATE_DIM).reshape(batch * count, STATE_DIM)
    survival = torch.ones(batch * count, device=state.device)
    total = torch.zeros_like(survival)
    for step in range(horizon):
        next_state, clearance, logits, penalty = model(state, action[:, step])
        progress = torch.linalg.vector_norm(state[:, 8:10] - state[:, 12:14], dim=-1) - torch.linalg.vector_norm(next_state[:, 8:10] - next_state[:, 12:14], dim=-1)
        if objective == "probability":
            collision = torch.sigmoid(logits)
            collision_penalty = 20 * collision
        elif objective == "penalty":
            collision = torch.sigmoid(logits)
            collision_penalty = penalty.clamp_min(0)
        elif objective == "clearance":
            collision = (clearance <= 0).float()
            collision_penalty = 20 * collision
        else:
            raise ValueError(f"Unknown structured objective {objective}")
        total += survival * (2 * progress - collision_penalty)
        survival = survival * (1 - collision)
        state = next_state
    return -total.reshape(batch, count)


def build_bank(task, snapshot, oracle, action_block, random_plans, seed, device):
    roots, horizon, plan_dim = oracle.shape
    generator = torch.Generator().manual_seed(seed)
    random = torch.rand(roots, random_plans, horizon, plan_dim, generator=generator) * 2 - 1
    candidates = torch.cat([oracle[:, None], torch.zeros(roots, 1, horizon, plan_dim), random], dim=1)
    truth = rollout_truth(task, tree_map(snapshot, lambda x: x.to(device)), candidates, action_block, device)
    return {"candidates": candidates, "truth": truth}


def bank_state(task, snapshot, device):
    env = task.get_env_fun(snapshot["steps"].shape[0], True, 0, device)()
    env.reset()
    restore_state(env, tree_map(snapshot, lambda x: x.to(device)))
    state = live_structured_state(env)
    env.close()
    return state


def zero_policy(env):
    agents = len(env._env.world.agents)
    primitive = env.full_action_spec_unbatched["agents", "action"].shape[-1]
    return torch.zeros(env.batch_size[0], 1, agents * primitive, device=env.device)


def controller_cost(model, action_block, objective):
    def costs(_snapshot, state, candidates):
        return surrogate_cost(model, state, candidates, action_block, objective)
    return costs


def render_results(result, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels, success, returns, distance, collision = [], [], [], [], []
    for name, summary in result["control_summary"].items():
        labels.append(name)
        success.append(summary["success"]["rate"])
        returns.append(summary["team_return"]["mean"])
        distance.append(summary["final_goal_distance"]["mean"])
        collision.append(summary["collision_rate"])
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for axis, values, title in zip(axes.flat, (success, returns, distance, collision), ("Success", "Team return", "Final distance", "Collision rate")):
        axis.bar(range(len(labels)), values, color="#4477AA")
        axis.set_xticks(range(len(labels)), labels, rotation=25, ha="right")
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle("Structured physical surrogate — held-out Buzz Wire control", fontsize=15)
    figure.savefig(output / "structured_control.png", dpi=180)
    figure.savefig(output / "structured_control.pdf")
    plt.close(figure)

    rows = result["ranking"]
    labels = [f"{row['mix']} s{row['seed']}" for row in rows]
    rho = [row["test"]["spearman"] for row in rows]
    oracle = [row["test"]["oracle_percentile"] for row in rows]
    figure, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    axes[0].bar(range(len(rows)), rho, color="#228833")
    axes[0].set_title("Held-out Plan–Real Spearman")
    axes[1].bar(range(len(rows)), oracle, color="#CC6677")
    axes[1].set_title("Held-out oracle percentile (lower is better)")
    for axis in axes:
        axis.set_xticks(range(len(rows)), labels, rotation=25, ha="right")
        axis.grid(axis="y", alpha=0.25)
    figure.savefig(output / "structured_ranking.png", dpi=180)
    figure.savefig(output / "structured_ranking.pdf")
    plt.close(figure)


def run_surrogate(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    data = torch.load(args.coverage / "coverage.pt", map_location="cpu", weights_only=False)
    stored = torch.load(args.coverage / "oracle_plans.pt", map_location="cpu", weights_only=False)
    manifest = json.loads((args.data / "manifest.json").read_text())
    task = VmasTask.BUZZ_WIRE.get_from_yaml()
    block = manifest["action_block"]
    models, records = {}, []
    seeds = [int(value) for value in args.seeds.split(",")]
    for mix in ("behavior", "full"):
        for seed in seeds:
            print(f"train {mix} seed {seed}", flush=True)
            model, fit = train_surrogate(data, mix, seed, args)
            test_mask = (data["split"] == 2) & ((data["family"] == 0) if mix == "behavior" else torch.ones_like(data["family"], dtype=torch.bool))
            test = evaluate_transitions(model, data, test_mask.nonzero(as_tuple=True)[0], args.device)
            checkpoint = output / f"model_{mix}_{seed}.pt"
            torch.save({"state_dict": model.state_dict(), "mix": mix, "seed": seed, "fit": fit, "test": test}, checkpoint)
            models[mix, seed] = model
            records.append({"mix": mix, "seed": seed, "fit": fit, "test_transitions": test, "checkpoint": str(checkpoint)})

    banks = {}
    for name, split in (("validation", 1), ("test", 2)):
        ids = (stored["split"] == split).nonzero(as_tuple=True)[0]
        snapshot = take_snapshot(stored["snapshot"], ids, "cpu")
        bank = build_bank(task, snapshot, stored["plans"][ids], block, args.random_plans, args.bank_seed + split, args.device)
        bank["state"] = bank_state(task, snapshot, args.device).cpu()
        bank["snapshot"] = snapshot
        banks[name] = bank
        torch.save(bank, output / f"{name}_bank.pt")

    objective_scores = defaultdict(list)
    ranking = []
    for (mix, seed), model in models.items():
        row = {"mix": mix, "seed": seed, "objectives": {}}
        for objective in ("probability", "penalty", "clearance"):
            metrics = {}
            for name, bank in banks.items():
                cost = surrogate_cost(
                    model,
                    bank["state"].to(args.device),
                    unpack_actions(bank["candidates"].to(args.device), block),
                    block,
                    objective,
                ).cpu()
                metrics[name] = rank_metrics(cost, bank["truth"], args.topk, named_candidates=True)
            row["objectives"][objective] = metrics
            objective_scores[mix, objective].append(metrics["validation"]["spearman"])
        ranking.append(row)

    selected = {}
    for mix in ("behavior", "full"):
        selected[mix] = max(
            ("probability", "penalty", "clearance"),
            key=lambda objective: sum(objective_scores[mix, objective]) / len(objective_scores[mix, objective]),
        )
    compact_ranking = []
    for row in ranking:
        objective = selected[row["mix"]]
        compact_ranking.append(
            {"mix": row["mix"], "seed": row["seed"], "objective": objective, **row["objectives"][objective]}
        )

    test_ids = (stored["split"] == 2).nonzero(as_tuple=True)[0]
    initial_state = take_snapshot(stored["snapshot"], test_ids, args.device)
    env = task.get_env_fun(test_ids.numel(), True, 0, args.device)()
    env.reset()
    cem_config = CEMConfig(
        horizon=args.horizon,
        num_samples=args.num_samples,
        num_elites=args.num_elites,
        num_iters=args.num_iters,
    )
    mpc_config = MPCConfig(receding_horizon=1, action_block=block)
    rows, timings = [], {}
    try:
        reference_seed = args.control_seed
        for policy, callable_policy in (("random", "random"), ("zero", zero_policy)):
            episode_rows, timing, _ = evaluate_policy(
                env,
                initial_state,
                policy=callable_policy,
                generator=torch.Generator(device=args.device).manual_seed(reference_seed),
                cem_config=cem_config,
                mpc_config=mpc_config,
                outcome_fn=task_outcome(manifest["task_name"]),
            )
            for row in episode_rows:
                row["policy"] = policy
            rows.extend(episode_rows)
            timings[policy] = timing
        for (mix, seed), model in models.items():
            policy = f"structured_{mix}_{seed}"
            episode_rows, timing, _ = evaluate_policy(
                env,
                initial_state,
                policy="mpc",
                generator=torch.Generator(device=args.device).manual_seed(args.control_seed),
                cem_config=cem_config,
                mpc_config=mpc_config,
                outcome_fn=task_outcome(manifest["task_name"]),
                plan_costs=controller_cost(model, block, selected[mix]),
                observe=live_structured_state,
            )
            for row in episode_rows:
                row["policy"] = policy
            rows.extend(episode_rows)
            timings[policy] = timing
            print(f"controlled {policy}", flush=True)
    finally:
        env.close()
    control_summary = summarize(rows, seed=args.control_seed)
    result = {
        "question": (
            "Can a cheap 14-D privileged structured-state diagnostic control "
            "Buzz Wire?"
        ),
        "selected_objective": selected,
        "fits": records,
        "ranking_all_objectives": ranking,
        "ranking": compact_ranking,
        "control_summary": control_summary,
        "control_rows": rows,
        "timing": timings,
        "test_root_ids": test_ids.tolist(),
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    }
    write_json(output / "result.json", result)
    render_results(result, output)
    print(json.dumps({"selected_objective": selected, "control_summary": control_summary}, indent=2), flush=True)


def build_parser():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    collect = sub.add_parser("collect")
    collect.add_argument("--data", type=Path, required=True)
    collect.add_argument("--runs", type=Path, required=True)
    collect.add_argument("--output", type=Path, required=True)
    collect.add_argument("--device", default="cuda")
    collect.add_argument("--model-seed", type=int, default=4100)
    collect.add_argument("--kind", default="joint")
    collect.add_argument("--seed", type=int, default=8400)
    collect.add_argument("--horizon", type=int, default=5)
    collect.add_argument("--num-samples", type=int, default=300)
    collect.add_argument("--num-elites", type=int, default=30)
    collect.add_argument("--root-batch", type=int, default=8)
    collect.add_argument("--replay-batch", type=int, default=256)
    collect.add_argument("--hard-train", type=int, default=256)
    collect.add_argument("--hard-validation", type=int, default=64)
    collect.add_argument("--hard-test", type=int, default=64)
    collect.add_argument("--hard-plans", type=int, default=4)
    collect.add_argument("--oracle-root-batch", type=int, default=8)
    collect.add_argument(
        "--oracle-roots",
        type=int,
        default=None,
        help="default: every initial root; a small value is for smoke tests",
    )
    collect.add_argument("--oracle-iters", type=int, default=30)
    collect.add_argument("--oracle-seed", type=int, default=8500)
    collect.add_argument("--local-plans", type=int, default=8)
    collect.add_argument("--local-sigma-small", type=float, default=0.05)
    collect.add_argument("--local-sigma-large", type=float, default=0.15)

    run = sub.add_parser("run")
    run.add_argument("--data", type=Path, required=True)
    run.add_argument("--coverage", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--device", default="cuda")
    run.add_argument("--seeds", default="9100,9101,9102")
    run.add_argument("--hidden", type=int, default=256)
    run.add_argument("--batch-size", type=int, default=512)
    run.add_argument("--epochs", type=int, default=200)
    run.add_argument("--patience", type=int, default=25)
    run.add_argument("--learning-rate", type=float, default=3e-4)
    run.add_argument("--weight-decay", type=float, default=1e-4)
    run.add_argument("--random-plans", type=int, default=300)
    run.add_argument("--bank-seed", type=int, default=8600)
    run.add_argument("--topk", type=int, default=30)
    run.add_argument("--horizon", type=int, default=5)
    run.add_argument("--num-samples", type=int, default=300)
    run.add_argument("--num-elites", type=int, default=30)
    run.add_argument("--num-iters", type=int, default=30)
    run.add_argument("--control-seed", type=int, default=8700)
    return parser


def main():
    args = build_parser().parse_args()
    if args.command == "collect":
        collect_coverage(args)
    else:
        run_surrogate(args)


if __name__ == "__main__":
    main()
