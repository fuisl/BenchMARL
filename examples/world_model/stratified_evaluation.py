#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#
"""Does the relational advantage live where the interaction lives?

The Dropout control compares two *tasks*, so a null there has several possible
causes besides the absence of cross-agent dynamics. This is the tighter,
within-task version: the M3 bank labels which test anchors actually exhibit a
physical cross-agent effect under a one-agent action intervention, so the same
models on the same data can be scored separately on interaction-active and
interaction-inactive states.

- advantage concentrated on interaction-active anchors -> interaction modelling
- advantage uniform across both strata -> a conditioning/regularisation effect

Absolute errors live in each model's own latent space, so only the paired
relational-vs-independent change *within* a stratum is comparable, and the
question is whether that change differs between strata.

Run:
    python -m examples.world_model.stratified_evaluation \\
        outputs/interaction_control_1194/transport --data outputs/transport_data_1190
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
import yaml

from examples.world_model.compare_baselines import bootstrap_interval, mean, median
from examples.world_model.dataset import OfflineSequences
from examples.world_model.train import load_model
from torch.utils.data import DataLoader

BASELINES = ("independent", "joint", "relational")


def effect_labels(data_root: Path, horizon: int):
    """Anchor ids whose intervention moved another agent or the package.

    Mirrors the M3 audit: compare the logged correlated snippet against the
    counterfactual snippet in which only agent 1's x action was flipped, and
    look for a change in absolute position/velocity above 1e-6, masked once
    either branch ends. Relative-observation changes do not count; this is a
    physical effect, not an observational one.
    """
    load = lambda name: torch.load(  # noqa: E731
        data_root / name, map_location="cpu", weights_only=True
    )
    counterfactual = load("counterfactual_test.pt")
    reference = load("samples_correlated.pt")
    ids = counterfactual["anchor_id"]
    valid = reference["valid"][ids] & counterfactual["valid"]

    # The collector intervenes on agent 1; every other agent is a witness whose
    # own action is unchanged. Derived from the data so 2-agent tasks work too.
    n_agents = reference["next_agent_state"].shape[2]
    other_agents = [i for i in range(n_agents) if i != 1]
    labels = torch.zeros(ids.numel(), dtype=torch.bool)
    for key, entities in (
        ("next_agent_state", other_agents),
        ("next_package_state", slice(None)),
    ):
        delta = (
            (
                reference[key][ids][:, :, entities, :4]
                - counterfactual[key][:, :, entities, :4]
            )
            .norm(dim=-1)
            .max(-1)
            .values
        )
        labels |= ((delta[:, :horizon] > 1e-6) & valid[:, :horizon]).any(-1)
    return ids, labels


@torch.no_grad()
def per_anchor_rollout_error(model, dataset, device):
    """Rollout error for each snippet, kept per anchor rather than pooled."""
    loader = DataLoader(dataset, batch_size=64, collate_fn=torch.stack)
    errors, anchors = [], []
    for batch in loader:
        batch = batch.to(device)
        latent = model.encode(batch["observation"])
        rolled = model.rollout(latent[:, :1], batch["action"])
        target = latent[:, 1:]
        valid = batch["valid"].unsqueeze(-1).unsqueeze(-1).float()
        squared = (rolled - target).square() * valid
        counts = valid.expand_as(squared).sum(dim=(1, 2, 3))
        errors.append(squared.sum(dim=(1, 2, 3)) / counts.clamp_min(1))
        anchors.append(batch["anchor_id"])
    return torch.cat(anchors).cpu(), torch.cat(errors).cpu()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", type=Path, help="sweep directory of trained runs")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    ids, labels = effect_labels(args.data, args.horizon)
    label_of = {int(i): bool(v) for i, v in zip(ids, labels)}
    print(
        f"interaction-active test anchors at {args.horizon} primitive steps: "
        f"{int(labels.sum())}/{labels.numel()}"
    )

    # (regime, kind, seed) -> {stratum: mean error}
    scores = {}
    datasets = {}
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
        if regime not in datasets:
            datasets[regime] = OfflineSequences(
                config["data"]["root"],
                regime,
                "test",
                action_block=config["data"]["action_block"],
            )
        model = load_model(checkpoint, args.device)
        anchors, errors = per_anchor_rollout_error(model, datasets[regime], args.device)
        active = torch.tensor([label_of[int(a)] for a in anchors])
        scores[(regime, kind, seed)] = {
            "active": float(errors[active].mean()),
            "inactive": float(errors[~active].mean()),
            "n_active": int(active.sum()),
            "n_inactive": int((~active).sum()),
        }

    regimes = sorted({regime for regime, _, _ in scores})
    seeds = sorted({seed for _, _, seed in scores})
    print(f"loaded {len(scores)} checkpoints over {len(seeds)} seeds\n")

    print("mean rollout error by stratum (each model's own latent space):")
    header = f"{'regime':12s} {'kind':12s} {'active':>10s} {'inactive':>10s} {'act/inact':>10s}"
    print(header)
    print("-" * len(header))
    for regime in regimes:
        for kind in BASELINES:
            values = [v for (r, k, _), v in scores.items() if (r, k) == (regime, kind)]
            act = mean([v["active"] for v in values])
            ina = mean([v["inactive"] for v in values])
            print(f"{regime:12s} {kind:12s} {act:10.4f} {ina:10.4f} {act/ina:9.2f}x")

    print(
        "\nTHE TEST -- paired relational vs independent, % change within each stratum."
        "\nConcentration on the active stratum is the interaction signature;"
        "\nequal change in both strata is a conditioning effect.\n"
    )
    header = (
        f"{'regime':12s} {'stratum':10s} {'mean':>9s} {'median':>9s} "
        f"{'95% CI':>22s} {'seeds better':>13s}"
    )
    print(header)
    print("-" * len(header))
    summary = defaultdict(dict)
    for regime in regimes:
        for stratum in ("active", "inactive"):
            pairs = []
            for seed in seeds:
                base = scores.get((regime, "independent", seed))
                treatment = scores.get((regime, "relational", seed))
                if base and treatment:
                    pairs.append(
                        (treatment[stratum] - base[stratum]) / base[stratum] * 100
                    )
            if not pairs:
                continue
            low, high = bootstrap_interval(pairs, mean)
            summary[regime][stratum] = (mean(pairs), low, high)
            better = sum(1 for value in pairs if value < 0)
            print(
                f"{regime:12s} {stratum:10s} {mean(pairs):+8.1f}% {median(pairs):+8.1f}% "
                f"{f'[{low:+.1f}, {high:+.1f}]':>22s} {f'{better}/{len(pairs)}':>13s}"
            )

    # Relative and absolute can disagree: interaction-active states carry ~4x the
    # error, so a percentage can be compressed there even when the model removes
    # more error outright. Both are reported rather than choosing the flattering one.
    print("\npaired ABSOLUTE error reduction, relational vs independent:")
    header = f"{'regime':12s} {'stratum':10s} {'mean':>12s} {'95% CI':>26s}"
    print(header)
    print("-" * len(header))
    for regime in regimes:
        for stratum in ("active", "inactive"):
            pairs = []
            for seed in seeds:
                base = scores.get((regime, "independent", seed))
                treatment = scores.get((regime, "relational", seed))
                if base and treatment:
                    pairs.append(treatment[stratum] - base[stratum])
            if pairs:
                low, high = bootstrap_interval(pairs, mean)
                print(
                    f"{regime:12s} {stratum:10s} {mean(pairs):+12.5f} "
                    f"{f'[{low:+.5f}, {high:+.5f}]':>26s}"
                )
    for regime in regimes:
        pairs = []
        for seed in seeds:
            base = scores.get((regime, "independent", seed))
            treatment = scores.get((regime, "relational", seed))
            if base and treatment:
                pairs.append(
                    (treatment["active"] - base["active"])
                    - (treatment["inactive"] - base["inactive"])
                )
        if pairs:
            low, high = bootstrap_interval(pairs, mean)
            verdict = (
                "MORE absolute error removed on interaction-active states"
                if high < 0
                else "LESS absolute error removed on active states"
                if low > 0
                else "no detectable difference"
            )
            print(
                f"  {regime:12s} active-minus-inactive {mean(pairs):+.5f} "
                f"[{low:+.5f}, {high:+.5f}] -> {verdict}"
            )

    Path("stratified_scores.json").write_text(
        json.dumps({f"{r}|{k}|{s}": v for (r, k, s), v in scores.items()}, indent=2)
    )

    print("\ndifference between strata (active minus inactive, per seed), RELATIVE:")
    for regime in regimes:
        pairs = []
        for seed in seeds:
            base = scores.get((regime, "independent", seed))
            treatment = scores.get((regime, "relational", seed))
            if not (base and treatment):
                continue
            active = (treatment["active"] - base["active"]) / base["active"] * 100
            inactive = (
                (treatment["inactive"] - base["inactive"]) / base["inactive"] * 100
            )
            pairs.append(active - inactive)
        if pairs:
            low, high = bootstrap_interval(pairs, mean)
            verdict = (
                "concentrated on interaction-active states"
                if high < 0
                else "no detectable concentration"
            )
            print(
                f"  {regime:12s} {mean(pairs):+7.1f} points "
                f"[{low:+.1f}, {high:+.1f}] -> {verdict}"
            )

    print(
        json.dumps({"anchors": {"active": int(labels.sum()), "total": labels.numel()}})
    )


if __name__ == "__main__":
    main()
