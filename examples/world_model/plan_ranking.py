#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#
"""M5 link 1: do better predictions rank candidate plans better?

Same planner, same objective, different dynamics. For each evaluation state we
sample one shared set of candidate joint-action plans, score them with the
simulator (exact cost) and with each learned model (latent rollout plus the
reward readout), then ask how well the model's ordering matches the truth.

Two properties of Transport shape the protocol and are reported rather than
worked around:

* **Most states are unrankable.** From a reset state no 25-step plan earns any
  reward, so every candidate ties at exactly zero and Spearman is undefined --
  the M2 candidate banks have one unique cost across all 300 candidates at all
  20 states. Only states whose true costs actually vary are scored, and the
  fraction kept is reported alongside the result.
* **Candidates are drawn uniformly**, which for a model trained on the
  correlated regime is deliberately outside its action coverage. That is the
  planner's real query, not a nuisance.

Run:
    python -m examples.world_model.plan_ranking \\
        outputs/interaction_control_1194/transport --data outputs/transport_data_1190
"""

import argparse
import json
from pathlib import Path

import torch
import yaml

from benchmarl.environments import VmasTask
from examples.world_model.compare_baselines import bootstrap_interval, mean
from examples.world_model.oracle_dynamics import oracle_plan_costs
from examples.world_model.train import load_model

BASELINES = ("independent", "joint", "relational")


def select_anchor_states(anchors, indices, device="cpu"):
    """Slice a batched snapshot down to the chosen anchor rows.

    Banks are loaded on CPU but the scratch environment may live on CUDA, and
    `broadcast_state` writes these tensors straight into the world, so the
    snapshot has to be moved with the slice rather than left behind.
    """

    def take(value):
        return value[indices].clone().to(device)

    snapshot = anchors["snapshot"]
    return {
        "entities": {
            name: {
                group: {k: take(v) for k, v in fields.items()}
                for group, fields in entity.items()
            }
            for name, entity in snapshot["entities"].items()
        },
        "scenario": {k: take(v) for k, v in snapshot["scenario"].items()},
        "steps": take(snapshot["steps"]),
    }


def true_costs(data_root: Path, indices, candidates, device):
    """Exact simulator cost per candidate: (B, K).

    The task comes from the bank's own manifest rather than being hard-coded, so
    the same protocol runs on whichever task produced the data.
    """
    anchors = torch.load(
        data_root / "anchors.pt", map_location="cpu", weights_only=True
    )
    manifest = json.loads((data_root / "manifest.json").read_text())
    snapshot = select_anchor_states(anchors, indices, device)
    batch, n_candidates = candidates.shape[:2]
    task = VmasTask[manifest["task_name"].split("/")[-1].upper()].get_from_yaml()
    scratch = task.get_env_fun(batch * n_candidates, True, 0, device)()
    scratch.reset()
    try:
        return oracle_plan_costs(scratch, snapshot, candidates.to(device)).cpu()
    finally:
        scratch.close()


@torch.no_grad()
def model_costs(model, observation, candidates, action_block, device):
    """Predicted cost per candidate under the learned dynamics: (B, K).

    Mirrors the oracle's objective exactly -- J = -sum_t sum_i r_i,t -- but the
    rewards come from the readout applied to rolled-out latents rather than from
    the simulator, so the only thing that differs between the two is the model.
    """
    batch, n_candidates, steps, joint_dim = candidates.shape
    agents, obs_dim = observation.shape[1:]
    action_dim = joint_dim // agents
    blocks = steps // action_block

    # (B,K,T,N*d_a) primitive -> (B*K,L,N,block*d_a), primitive time before coordinate.
    plans = candidates.view(
        batch, n_candidates, blocks, action_block, agents, action_dim
    )
    plans = plans.permute(0, 1, 2, 4, 3, 5).reshape(
        batch * n_candidates, blocks, agents, action_block * action_dim
    )
    plans = plans.to(device)

    start = observation.unsqueeze(1).expand(batch, n_candidates, agents, obs_dim)
    start = start.reshape(batch * n_candidates, 1, agents, obs_dim).to(device)

    latent = model.encode(start)
    rolled = model.rollout(latent, plans)
    sequence = torch.cat([latent, rolled], dim=1)
    reward, _ = model.readout(sequence[:, :-1], sequence[:, 1:])
    cost = -reward.sum(dim=(1, 2, 3))
    return cost.view(batch, n_candidates).cpu()


def spearman(a, b):
    """Rank correlation of two 1-D tensors, with average ranks for ties."""

    def ranks(x):
        order = x.argsort()
        result = torch.empty_like(x)
        result[order] = torch.arange(x.numel(), dtype=x.dtype)
        # Average tied ranks so ties do not manufacture an ordering.
        unique, inverse = x.unique(return_inverse=True)
        for index in range(unique.numel()):
            mask = inverse == index
            if mask.sum() > 1:
                result[mask] = result[mask].mean()
        return result

    ra, rb = ranks(a), ranks(b)
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denominator = ra.norm() * rb.norm()
    if denominator == 0:
        return float("nan")
    return float((ra * rb).sum() / denominator)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", type=Path)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--candidates", type=int, default=64)
    parser.add_argument("--states", type=int, default=128)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=5100)
    parser.add_argument("--cache", type=Path, default=Path("plan_ranking_truth.pt"))
    args = parser.parse_args()

    anchors = torch.load(
        args.data / "anchors.pt", map_location="cpu", weights_only=True
    )
    test = (anchors["split"] == 2).nonzero(as_tuple=True)[0]
    generator = torch.Generator().manual_seed(args.seed)
    chosen = test[torch.randperm(test.numel(), generator=generator)[: args.states]]

    if args.cache.exists():
        cached = torch.load(args.cache, map_location="cpu", weights_only=True)
        chosen, candidates, truth = (
            cached["indices"],
            cached["candidates"],
            cached["costs"],
        )
        print(f"loaded cached truth for {chosen.numel()} states")
    else:
        manifest = json.loads((args.data / "manifest.json").read_text())
        steps = manifest["sequence_steps"]
        joint_dim = torch.as_tensor(manifest["action_low"]).numel()
        candidates = (
            torch.rand(
                chosen.numel(), args.candidates, steps, joint_dim, generator=generator
            )
            * 2
            - 1
        )
        truth = true_costs(args.data, chosen, candidates, args.device)
        torch.save(
            {"indices": chosen, "candidates": candidates, "costs": truth}, args.cache
        )
        print(f"computed simulator truth for {chosen.numel()} states")

    rankable = truth.std(dim=1) > 1e-9
    print(
        f"rankable states (true costs actually vary): "
        f"{int(rankable.sum())}/{rankable.numel()} "
        f"({100 * float(rankable.float().mean()):.0f}%)"
    )
    if not bool(rankable.any()):
        raise ValueError("No state has varying true costs; ranking is undefined")

    results = {}
    samples = None
    for directory in sorted(args.runs.glob("[0-9]*")):
        checkpoint = directory / "model.pt"
        if not checkpoint.exists():
            continue
        config = yaml.safe_load((directory / "resolved_config.yaml").read_text())
        regime, kind, seed = (
            config["data"]["regime"],
            config["model"]["kind"],
            config["seed"],
        )
        if samples is None or samples[0] != regime:
            loaded = torch.load(
                args.data / f"samples_{regime}.pt",
                map_location="cpu",
                weights_only=True,
            )
            samples = (regime, loaded["observation"])
        observation = samples[1][chosen, 0]
        model = load_model(checkpoint, args.device)
        predicted = model_costs(
            model, observation, candidates, config["data"]["action_block"], args.device
        )

        correlations, regrets = [], []
        for state in rankable.nonzero(as_tuple=True)[0]:
            rho = spearman(predicted[state], truth[state])
            if rho == rho:  # skip NaN from a constant prediction
                correlations.append(rho)
            picked = int(predicted[state].argmin())
            regrets.append(float(truth[state][picked] - truth[state].min()))
        results[(regime, kind, seed)] = {
            "spearman": mean(correlations) if correlations else float("nan"),
            "regret": mean(regrets),
            "scored_states": len(correlations),
        }

    regimes = sorted({r for r, _, _ in results})
    seeds = sorted({s for _, _, s in results})
    print(f"scored {len(results)} checkpoints over {len(seeds)} seeds\n")

    header = f"{'regime':12s} {'kind':12s} {'spearman':>10s} {'regret':>10s}"
    print(header)
    print("-" * len(header))
    for regime in regimes:
        for kind in BASELINES:
            values = [v for (r, k, _), v in results.items() if (r, k) == (regime, kind)]
            if values:
                print(
                    f"{regime:12s} {kind:12s} "
                    f"{mean([v['spearman'] for v in values]):10.4f} "
                    f"{mean([v['regret'] for v in values]):10.4f}"
                )

    print("\npaired vs independent (positive spearman change = better ranking):")
    header = f"{'regime':12s} {'kind':12s} {'metric':10s} {'mean':>10s} {'95% CI':>24s} {'seeds better':>13s}"
    print(header)
    print("-" * len(header))
    for regime in regimes:
        for kind in ("joint", "relational"):
            for metric, better_is_lower in (("spearman", False), ("regret", True)):
                pairs = []
                for seed in seeds:
                    base = results.get((regime, "independent", seed))
                    treatment = results.get((regime, kind, seed))
                    if base and treatment:
                        pairs.append(treatment[metric] - base[metric])
                if not pairs:
                    continue
                low, high = bootstrap_interval(pairs, mean)
                # Higher Spearman is better; lower regret is better.
                better = sum(1 for v in pairs if (v < 0) == better_is_lower)
                print(
                    f"{regime:12s} {kind:12s} {metric:10s} {mean(pairs):+10.4f} "
                    f"{f'[{low:+.4f}, {high:+.4f}]':>24s} {f'{better}/{len(pairs)}':>13s}"
                )

    print(
        json.dumps({"rankable_states": int(rankable.sum()), "total": rankable.numel()})
    )


if __name__ == "__main__":
    main()
