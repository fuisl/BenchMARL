#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#
"""Is the learned cost landscape wrong, or is the search failing to climb it?

Job 1273 left three explanations standing for `physical` models that stop
colliding and make no progress:

  (a) **safe but static** -- the model believes inaction is optimal, and the
      planner is faithfully executing that belief;
  (b) **cannot predict progress** -- the learned cost does not order plans by
      how much task progress they make, so there is no signal to follow;
  (c) **search failure** -- the signal is there and CEM does not reach it.

They are separable by scoring ONE candidate bank three ways. The bank is built
so that it spans the range the planner cannot: 300 random plans (what CEM starts
from), the all-zero plan, and the plan the TRUE-dynamics CEM chose from the same
state with the same budget -- which is known to make progress, because that
planner solves 25/32.

Each candidate is scored by the simulator (true return, true final task
distance) and by the learned cost each model's planner actually optimises. Then:

* if the **zero plan** ranks at or near the learned cost's minimum, (a) holds:
  the model prefers doing nothing and the planner is not malfunctioning;
* if the **oracle plan** ranks poorly under the learned cost while the
  simulator scores it best, (b) holds: the landscape does not reward progress;
* if the oracle plan ranks near the top under the learned cost and CEM still did
  not find it, (c) holds -- and (c) would be surprising, because the same CEM
  budget found that plan when the cost came from the simulator.

Run:
    python -m examples.world_model.cost_landscape \\
        outputs/state_input_1223/runs --data outputs/buzz_wire_1196/data \\
        --device cuda --roots 12
"""

import argparse
from collections import defaultdict
from pathlib import Path

import torch
import yaml

from benchmarl.environments import VmasTask
from examples.world_model.cem import CEMConfig, cem_plan
from examples.world_model.model_input import compose_frames
from examples.world_model.mpc import action_bounds, task_outcome, unpack_actions
from examples.world_model.plan_ranking import model_costs, spearman
from examples.world_model.model_input import entity_frame
from examples.world_model.snapshot_restore import (
    agent_observations,
    broadcast_state,
    restore_state,
)
from examples.world_model.train import load_model
from tensordict import TensorDict

KINDS = ("independent", "joint", "relational")
INPUTS = ("observation", "history", "physical")


@torch.no_grad()
def true_scores(env, snapshot, candidates, action_block, outcome_fn):
    """Simulator truth per candidate: native return, final task distance, collided.

    `candidates` is (B, K, H*action_block, joint). Every candidate starts from
    its own root, so slot b*K+k restores root b.
    """
    batch, n_candidates, steps, joint = candidates.shape
    count = batch * n_candidates
    agents = len(env._env.world.agents)
    primitive = joint // agents
    indices = torch.arange(batch, device=candidates.device).repeat_interleave(
        n_candidates
    )
    broadcast_state(env, snapshot, source_indices=indices)
    live = ~env._env.done()
    td = TensorDict({}, batch_size=[count], device=candidates.device)
    actions = candidates.reshape(count, steps, agents, primitive)

    total = torch.zeros(count, device=candidates.device)
    collided = torch.zeros(count, dtype=torch.bool, device=candidates.device)
    _reached, _coll, distance = outcome_fn(env)
    final = distance.clone()
    for step in range(steps):
        td.set(("agents", "action"), actions[:, step])
        td = env.step(td)["next"]
        reward = td["agents", "reward"].sum(dim=1).squeeze(-1)
        total += reward.masked_fill(~live, 0)
        _reached, crash, distance = outcome_fn(env)
        # Freeze the task distance at the frame the episode ended on, exactly as
        # EpisodeStats does; a finished slot keeps being stepped beside its
        # neighbours and must not contribute the state it drifted to.
        final = torch.where(live, distance, final)
        collided |= live & crash
        live = live & ~td["done"].squeeze(-1)
    shape = (batch, n_candidates)
    return total.view(shape).cpu(), final.view(shape).cpu(), collided.view(shape).cpu()


def percentile_rank(values, index):
    """Fraction of candidates scoring strictly better (lower) than `index`."""
    return float((values < values[index]).float().mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", type=Path)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--roots", type=int, default=12)
    ap.add_argument("--random-plans", type=int, default=300)
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--num-samples", type=int, default=300)
    ap.add_argument("--num-iters", type=int, default=30)
    ap.add_argument("--seed", type=int, default=7100)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    manifest = yaml.safe_load((args.data / "manifest.json").read_text())
    task = VmasTask[manifest["task_name"].split("/")[-1].upper()].get_from_yaml()
    block = manifest["action_block"]
    outcome_fn = task_outcome(manifest["task_name"])
    initial = torch.load(args.data / "initial_states.pt", map_location="cpu",
                         weights_only=False)
    # Snapshots nest dicts per entity, so slice and move recursively; a
    # half-moved snapshot fails inside the physics step rather than here.
    def take(value):
        if isinstance(value, dict):
            return {k: take(v) for k, v in value.items()}
        return value[: args.roots].to(args.device) if torch.is_tensor(value) else value

    snapshot = take(initial["snapshot"])
    roots = args.roots

    # The root observations, read from a restored simulator rather than from
    # `initial_states.pt`, which stores snapshots only.
    probe = task.get_env_fun(roots, True, 0, args.device)()
    probe.reset()
    restore_state(probe, snapshot)
    agents = len(probe._env.world.agents)
    primitive = probe.full_action_spec_unbatched["agents", "action"].shape[-1]
    low, high = action_bounds(probe)
    root_observation = agent_observations(probe).cpu()
    root_entities = entity_frame(probe).cpu()
    probe.close()
    joint_dim = agents * primitive
    plan_dim = joint_dim * block

    # 1. The plan the TRUE-dynamics planner chooses, at the same budget.
    scratch = task.get_env_fun(roots * args.num_samples, True, 0, args.device)()
    scratch.reset()
    try:
        def oracle_cost(candidates):
            steps = unpack_actions(candidates, block)
            total, _final, _coll = true_scores(
                scratch, snapshot, steps, block, outcome_fn
            )
            # `true_scores` returns on CPU because the tables below compare
            # there; CEM gathers elites on the planner's device.
            return (-total).to(candidates.device)

        result = cem_plan(
            oracle_cost,
            action_dim=plan_dim,
            action_low=low,
            action_high=high,
            config=CEMConfig(
                horizon=args.horizon,
                num_samples=args.num_samples,
                num_iters=args.num_iters,
            ),
            batch_size=roots,
            device=args.device,
            generator=torch.Generator(device=args.device).manual_seed(args.seed),
        )
        oracle_plan = result.plan  # (roots, H, plan_dim)
    finally:
        scratch.close()

    # 2. The bank: oracle plan, the zero plan, then random plans.
    generator = torch.Generator().manual_seed(args.seed + 1)
    random_plans = (
        torch.rand(roots, args.random_plans, args.horizon, plan_dim,
                   generator=generator) * (high - low) + low
    ).to(args.device)
    bank = torch.cat(
        [
            oracle_plan.unsqueeze(1),
            torch.zeros(roots, 1, args.horizon, plan_dim, device=args.device),
            random_plans,
        ],
        dim=1,
    )
    ORACLE, ZERO = 0, 1
    n_candidates = bank.shape[1]

    env = task.get_env_fun(roots * n_candidates, True, 0, args.device)()
    env.reset()
    try:
        returns, distances, collided = true_scores(
            env, snapshot, unpack_actions(bank, block), block, outcome_fn
        )
    finally:
        env.close()

    print(f"{manifest['task_name']}: {roots} roots x {n_candidates} candidates "
          f"(1 oracle + 1 zero + {args.random_plans} random), H={args.horizon} "
          f"blocks = {args.horizon * block} steps")
    print(f"  simulator: oracle return {returns[:, ORACLE].mean():+.3f} "
          f"dist {distances[:, ORACLE].mean():.4f} coll {collided[:, ORACLE].float().mean():.2f}"
          f" | zero {returns[:, ZERO].mean():+.3f} dist {distances[:, ZERO].mean():.4f}"
          f" | random {returns[:, 2:].mean():+.3f} dist {distances[:, 2:].mean():.4f}")

    # 3. The learned cost each planner actually optimises.
    rows = defaultdict(list)
    for path in sorted(args.runs.rglob("resolved_config.yaml")):
        run = path.parent
        if not (run / "model.pt").exists():
            continue
        cfg = yaml.safe_load(path.read_text())
        state_input = cfg["data"].get("state_input", "observation")
        model = load_model(run / "model.pt", args.device)
        # Observations at the roots, in this checkpoint's own input format.
        observation = compose_frames(
            state_input,
            root_observation.unsqueeze(1),
            root_entities.unsqueeze(1),
            cfg["data"].get("history_frames", 3),
        )[:, 0]
        cost = model_costs(
            model, observation, unpack_actions(bank, block).cpu(), block, args.device
        )
        best = cost.argmin(dim=1)
        rows[(state_input, cfg["model"]["kind"], cfg["data"]["regime"])].append(
            {
                "zero_rank": sum(
                    percentile_rank(cost[b], ZERO) for b in range(roots)
                ) / roots,
                "oracle_rank": sum(
                    percentile_rank(cost[b], ORACLE) for b in range(roots)
                ) / roots,
                "rho_return": sum(
                    spearman(-cost[b], returns[b]) for b in range(roots)
                ) / roots,
                "rho_distance": sum(
                    spearman(cost[b], distances[b]) for b in range(roots)
                ) / roots,
                "chosen_return": float(returns[torch.arange(roots), best].mean()),
                "chosen_distance": float(distances[torch.arange(roots), best].mean()),
                "spread": float((cost.std(dim=1) / cost.abs().mean(dim=1)).mean()),
            }
        )

    header = (
        f"{'input':12s}{'kind':12s}{'regime':12s}{'zero%ile':>10s}{'oracle%ile':>12s}"
        f"{'rho(ret)':>10s}{'rho(dist)':>11s}{'chosen ret':>12s}{'chosen dist':>13s}"
        f"{'spread':>9s}"
    )
    print("\n" + header)
    print("-" * len(header))
    for key in sorted(rows):
        values = rows[key]
        mean = {k: sum(v[k] for v in values) / len(values) for k in values[0]}
        print(
            f"{key[0]:12s}{key[1]:12s}{key[2]:12s}{mean['zero_rank']:10.3f}"
            f"{mean['oracle_rank']:12.3f}{mean['rho_return']:10.4f}"
            f"{mean['rho_distance']:11.4f}{mean['chosen_return']:12.3f}"
            f"{mean['chosen_distance']:13.4f}{mean['spread']:9.3f}"
        )
    print(
        "\nzero%ile / oracle%ile: fraction of the bank the learned cost prefers to"
        "\nthat plan. 0.00 = the learned cost's own favourite; 1.00 = its worst."
        "\nrho(ret) is against true return, rho(dist) against final task distance"
        "\n(both: higher = the learned cost orders plans the way the simulator does)."
    )


if __name__ == "__main__":
    main()
