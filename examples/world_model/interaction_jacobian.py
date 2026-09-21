#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#
"""T-A1: the TRUE interaction Jacobian, measured on the simulator alone.

Task A asks whether a learned latent transition preserves counterfactual
joint-action effects. That question is only answerable where the effect exists
and is larger than what we can resolve. This module measures the effect itself.
No learned model is loaded here, deliberately: a model comparison run on cells
whose true effect is at the noise floor is the single failure this project has
repeated most often (see `14_gate2_gate4.md` on Balance, where the true response
sat 9-70x below probe error and produced a model ordering that was pure noise).

From one restored held-out state we hold every action coordinate at a sampled
reference joint action A, then move ONE coordinate -- agent `i`'s axis `c` --
to each end of its own action range. The difference in the simulator's next
physical state is a finite-difference column of

                 [ dY_A/da_A   dY_A/da_B ]
    J         =  [                       ]
     interaction [ dY_B/da_A   dY_B/da_B ]

plus a shared row for the ball, which is nobody's private state. The diagonal
blocks are self-dynamics. The off-diagonal blocks are the entire multi-agent
content of the problem: an independent predictor sets them to exactly zero by
construction, so wherever they are nonzero it is structurally misspecified, and
wherever they are zero it is sufficient.

The step is the full action range for every cell rather than a reflection of the
sampled value, so all four blocks share one finite-difference step and are
directly comparable. The bank's historical `low + high - a` reflection on agent
1's x is reported alongside as a consistency check with the existing
counterfactual bank, not as the primary measurement.

Y is the common physical target `physical_response.py` already uses -- each
body's (pos_x, pos_y, vel_x, vel_y) -- scaled by its per-dimension standard
deviation on TRAINING rows, so the number is unitless and every later model
comparison can quote the same scale.

Run:
    python -m examples.world_model.interaction_jacobian \\
        --data outputs/buzz_wire_1196/data --device cuda --out outputs/ta1
"""

import argparse
import json
from pathlib import Path

import torch

from benchmarl.environments import VmasTask
from examples.world_model.collect import branch_rollouts, sample_actions
from examples.world_model.physical_response import MOTION
from examples.world_model.plan_ranking import select_anchor_states

SPLITS = ("train", "validation", "test")
# Bit-exact replay is a contract the bank already verified; re-checking it here
# means a silent simulator/version change cannot be read as an interaction.
REPLAY_TOLERANCE = 0.0


def load_bank(data_root):
    anchors = torch.load(
        data_root / "anchors.pt", map_location="cpu", weights_only=True
    )
    manifest = json.loads((data_root / "manifest.json").read_text())
    return anchors, manifest


def resolve_task(manifest):
    """Build the task and require it to agree with the bank that produced Y.

    `get_from_yaml` reads the current checked-in task config. If that config has
    drifted from the one the bank was collected under, every number below would
    describe a different environment than the one the models were trained on.
    """
    task = VmasTask[manifest["task_name"].split("/")[-1].upper()].get_from_yaml()
    recorded = manifest.get("task", {})
    config = dict(task.config)
    mismatched = {
        key: {"manifest": value, "resolved": config.get(key)}
        for key, value in recorded.items()
        if key in config and config[key] != value
    }
    if mismatched:
        raise ValueError(f"Task config disagrees with the bank manifest: {mismatched}")
    return task, recorded


def action_bounds(task, device):
    env = task.get_env_fun(1, True, 0, device)()
    try:
        spec = env.full_action_spec_unbatched["agents", "action"]
        return spec.low.cpu().clone(), spec.high.cpu().clone()
    finally:
        env.close()


def constant_plan(action, steps):
    """(B,N,d) held for `steps` primitive steps -> (B,steps,N,d).

    The Jacobian is evaluated at one action point, so the intervened coordinate
    must stay intervened for the whole horizon. A block that reverted to the
    reference after five steps would measure a transient, not a column of J.
    """
    return action.unsqueeze(1).expand(-1, steps, -1, -1).contiguous()


def witness_response(reference_data, branch_data, scale_agent, scale_shared):
    """Scaled |dY| per anchor, per horizon step, split by responding body.

    Returns agent (B,T,N) and shared (B,T,M) magnitudes, plus the joint validity
    mask. Following `collect.effect_summary`, a step counts only while BOTH
    branches are still live: after either terminates the stored rows are zeroed
    and their difference would be an artifact of termination, not of physics.
    """
    valid = reference_data["valid"] & branch_data["valid"]
    agent_delta = (
        branch_data["next_agent_state"][..., MOTION]
        - reference_data["next_agent_state"][..., MOTION]
    ) / scale_agent
    shared_delta = (
        branch_data["next_package_state"][..., MOTION]
        - reference_data["next_package_state"][..., MOTION]
    ) / scale_shared
    return agent_delta.norm(dim=-1), shared_delta.norm(dim=-1), valid


def cumulative_valid(valid):
    """Mask a step only if every step up to it was live in both branches."""
    return valid.cumprod(dim=1).bool()


def bootstrap_by_episode(values, episode_ids, samples=2000, seed=7301):
    """Mean with a 95% interval resampling ROOT EPISODES, not anchors.

    Anchors branch from shared source episodes, so treating them as independent
    samples overstates precision -- audit section 5.3.
    """
    values = values.double()
    unique = torch.unique(episode_ids)
    groups = [values[episode_ids == episode] for episode in unique.tolist()]
    groups = [group for group in groups if group.numel() > 0]
    if not groups:
        # A deep-horizon cell can have every anchor terminated. Report it as
        # empty rather than crashing the caller that formats it.
        return {
            "mean": float("nan"),
            "low": float("nan"),
            "high": float("nan"),
            "episodes": 0,
            "anchors": 0,
        }
    generator = torch.Generator().manual_seed(seed)
    means = []
    for _ in range(samples):
        picks = torch.randint(len(groups), (len(groups),), generator=generator)
        means.append(torch.cat([groups[p] for p in picks.tolist()]).mean())
    means = torch.stack(means)
    return {
        "mean": float(values.mean()),
        "low": float(means.quantile(0.025)),
        "high": float(means.quantile(0.975)),
        "episodes": int(len(groups)),
        "anchors": int(values.numel()),
    }


def training_scale(data_root, anchors, regimes):
    """Per-dimension std of Y on TRAINING anchors, shared by every later model.

    Read from the same stored samples `physical_response.py` scales with, so a
    ratio reported there and a magnitude reported here are on one scale.
    """
    train_mask = anchors["split"] == SPLITS.index("train")
    agent_rows, shared_rows = [], []
    for regime in regimes:
        path = data_root / f"samples_{regime}.pt"
        if not path.exists():
            continue
        samples = torch.load(path, map_location="cpu", weights_only=True)
        rows = train_mask[: samples["next_agent_state"].shape[0]]
        valid = samples["valid"][rows]
        agent = samples["next_agent_state"][rows][..., MOTION]
        shared = samples["next_package_state"][rows][..., MOTION]
        agent_rows.append(agent[valid].reshape(-1, agent.shape[-1]))
        shared_rows.append(shared[valid].reshape(-1, shared.shape[-1]))
    if not agent_rows:
        raise FileNotFoundError("No samples_<regime>.pt found to fit the Y scale")
    agent_scale = torch.cat(agent_rows).std(dim=0).clamp_min(1e-6)
    shared_scale = torch.cat(shared_rows).std(dim=0).clamp_min(1e-6)
    return agent_scale, shared_scale


def replay_check(task, sub_anchors, plan, batch_size, device):
    """Two identical rollouts from one snapshot must agree bit for bit."""
    first = branch_rollouts(task, sub_anchors, plan, batch_size, device)
    second = branch_rollouts(task, sub_anchors, plan, batch_size, device)
    gaps = {
        key: float((first[key].double() - second[key].double()).abs().max())
        for key in ("next_agent_state", "next_package_state", "reward")
    }
    if max(gaps.values()) > REPLAY_TOLERANCE:
        raise ValueError(f"Simulator replay is not deterministic: {gaps}")
    return gaps


def run(args):
    data_root = Path(args.data)
    anchors, manifest = load_bank(data_root)
    task, task_config = resolve_task(manifest)
    block = args.block or manifest.get("action_block", 5)
    steps = args.horizon * block

    split_index = SPLITS.index(args.split)
    rows = (anchors["split"] == split_index).nonzero().squeeze(-1)
    if args.max_anchors:
        rows = rows[: args.max_anchors]
    episode_ids = anchors["episode_id"][rows]
    sub = {"snapshot": select_anchor_states(anchors, rows, "cpu")}
    count = len(rows)

    low, high = action_bounds(task, args.device)
    agents, action_dim = low.shape
    agent_scale, shared_scale = training_scale(
        data_root, anchors, ("correlated", "independent")
    )

    print(
        f"T-A1: {count} {args.split} anchors from {len(torch.unique(episode_ids))} "
        f"root episodes, {agents} agents x {action_dim} axes, "
        f"horizon {args.horizon} blocks of {block} steps",
        flush=True,
    )

    results = {
        "data_root": str(data_root),
        "task_name": manifest["task_name"],
        "task_config": task_config,
        "split": args.split,
        "anchors": int(count),
        "root_episodes": int(len(torch.unique(episode_ids))),
        "action_block": int(block),
        "horizon_blocks": int(args.horizon),
        "primitive_steps": int(steps),
        "references": int(args.references),
        "reference_seed": int(args.seed),
        "action_low": low.tolist(),
        "action_high": high.tolist(),
        "y_columns": "pos_x, pos_y, vel_x, vel_y",
        "agent_scale": agent_scale.tolist(),
        "shared_scale": shared_scale.tolist(),
        "cells": {},
        "reflection_check": {},
    }

    # Preflight on the first reference action, before anything is interpreted.
    warm = sample_actions(
        (count, 1, agents, action_dim), low, high, "independent", args.seed
    )[:, 0]
    results["replay_determinism_max_abs_difference"] = replay_check(
        task, sub, constant_plan(warm, block), args.batch_size, args.device
    )
    print(f"  replay determinism: {results['replay_determinism_max_abs_difference']}", flush=True)

    per_cell = {}
    reflection = {}
    for reference in range(args.references):
        seed = args.seed + reference
        base = sample_actions(
            (count, 1, agents, action_dim), low, high, "independent", seed
        )[:, 0]
        base_plan = constant_plan(base, steps)
        base_data = branch_rollouts(task, sub, base_plan, args.batch_size, args.device)

        for intervened in range(agents):
            for axis in range(action_dim):
                endpoints = {}
                for name, bound in (("low", low), ("high", high)):
                    moved = base.clone()
                    moved[:, intervened, axis] = bound[intervened, axis]
                    endpoints[name] = branch_rollouts(
                        task, sub, constant_plan(moved, steps), args.batch_size, args.device
                    )
                agent_mag, shared_mag, valid = witness_response(
                    endpoints["low"], endpoints["high"], agent_scale, shared_scale
                )
                live = cumulative_valid(valid)
                for horizon in range(1, args.horizon + 1):
                    step = horizon * block - 1
                    mask = live[:, step]
                    for responder in range(agents):
                        key = (intervened, axis, f"agent_{responder}", horizon)
                        per_cell.setdefault(key, []).append(
                            (agent_mag[:, step, responder][mask], episode_ids[mask])
                        )
                    for body in range(shared_mag.shape[-1]):
                        key = (intervened, axis, f"shared_{body}", horizon)
                        per_cell.setdefault(key, []).append(
                            (shared_mag[:, step, body][mask], episode_ids[mask])
                        )

        # Historical consistency: the bank's own intervention is a reflection of
        # agent 1's sampled x about the action midpoint, not an endpoint pair.
        mirrored = base.clone()
        mirrored[:, 1, 0] = low[1, 0] + high[1, 0] - mirrored[:, 1, 0]
        mirror_data = branch_rollouts(
            task, sub, constant_plan(mirrored, steps), args.batch_size, args.device
        )
        agent_mag, shared_mag, valid = witness_response(
            base_data, mirror_data, agent_scale, shared_scale
        )
        live = cumulative_valid(valid)
        step = block - 1
        mask = live[:, step]
        reflection.setdefault("agent_0_at_h1", []).append(
            (agent_mag[:, step, 0][mask], episode_ids[mask])
        )
        print(f"  reference {reference} complete", flush=True)

    def summarize(entries):
        values = torch.cat([value for value, _ in entries])
        ids = torch.cat([episode for _, episode in entries])
        summary = bootstrap_by_episode(values, ids, seed=args.seed + 900)
        summary["active_fraction_above_1e-6"] = float((values > 1e-6).double().mean())
        summary["median"] = float(values.median()) if values.numel() else float("nan")
        summary["p90"] = (
            float(values.double().quantile(0.9)) if values.numel() else float("nan")
        )
        summary["max"] = float(values.max()) if values.numel() else float("nan")
        return summary

    for (intervened, axis, responder, horizon), entries in per_cell.items():
        name = f"a{intervened}_axis{axis}__{responder}__h{horizon}"
        summary = summarize(entries)
        summary.update(
            {
                "intervened_agent": intervened,
                "intervened_axis": axis,
                "responder": responder,
                "horizon_blocks": horizon,
                "block": "self"
                if responder == f"agent_{intervened}"
                else ("shared" if responder.startswith("shared") else "cross"),
            }
        )
        results["cells"][name] = summary

    for name, entries in reflection.items():
        results["reflection_check"][name] = summarize(entries)

    # Headline: how much of the response an independent model cannot represent.
    for horizon in range(1, args.horizon + 1):
        self_terms, cross_terms = [], []
        for cell in results["cells"].values():
            if cell["horizon_blocks"] != horizon:
                continue
            if cell["block"] == "self":
                self_terms.append(cell["mean"])
            elif cell["block"] == "cross":
                cross_terms.append(cell["mean"])
        total = sum(self_terms) + sum(cross_terms)
        results.setdefault("cross_share", {})[f"h{horizon}"] = {
            "mean_self_response": sum(self_terms) / max(len(self_terms), 1),
            "mean_cross_response": sum(cross_terms) / max(len(cross_terms), 1),
            "cross_over_self": (
                (sum(cross_terms) / max(len(cross_terms), 1))
                / max(sum(self_terms) / max(len(self_terms), 1), 1e-12)
            ),
            "cross_share_of_total": sum(cross_terms) / max(total, 1e-12),
        }

    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    (output / "interaction_jacobian.json").write_text(json.dumps(results, indent=2))
    print(f"\nWrote {output / 'interaction_jacobian.json'}")
    print_report(results)
    return results


def print_report(results):
    print("\nTrue interaction Jacobian, scaled |dY| per unit action-range step")
    print("(an independent predictor sets every `cross` cell to exactly 0)\n")
    horizons = sorted({c["horizon_blocks"] for c in results["cells"].values()})
    for horizon in horizons:
        print(f"horizon {horizon} block(s):")
        rows = [
            (name, cell)
            for name, cell in results["cells"].items()
            if cell["horizon_blocks"] == horizon
        ]
        for name, cell in sorted(rows, key=lambda r: (r[1]["block"], r[0])):
            print(
                f"  {cell['block']:<6} {name:<34} "
                f"mean {cell['mean']:.5f} [{cell['low']:.5f}, {cell['high']:.5f}]  "
                f"active {cell['active_fraction_above_1e-6']:.3f}  "
                f"n={cell['anchors']} over {cell['episodes']} episodes"
            )
        share = results["cross_share"][f"h{horizon}"]
        print(
            f"  -> cross/self = {share['cross_over_self']:.4f}, "
            f"cross share of total = {share['cross_share_of_total']:.4f}\n"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--split", default="test", choices=SPLITS)
    parser.add_argument("--horizon", type=int, default=5, help="blocks")
    parser.add_argument("--block", type=int, default=None, help="primitive steps/block")
    parser.add_argument("--references", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7301)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-anchors", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
