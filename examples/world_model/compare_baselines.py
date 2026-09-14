#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#
"""Summarise M4 baseline sweeps and M1 Row 4's weak-interaction control.

Compares baselines *within* a task only. Absolute latent errors are not
comparable across tasks or across baselines -- each model learns its own latent
space, and Dropout observes 7 features where Transport observes 11 -- so the
reported quantity is the within-task relational-vs-independent ratio, and the
control is whether that ratio survives on a task with no cross-agent dynamics.

Seeds are shared across baselines (the seed fixes both initialisation and batch
order), so differences are paired by seed rather than compared as independent
group means.

Run:
    python -m examples.world_model.compare_baselines outputs/interaction_control_1194
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import yaml

BASELINES = ("independent", "joint", "relational")


def load_runs(root: Path):
    """(task, regime, kind, seed) -> metrics, for every completed run beneath root."""
    runs = {}
    for metrics_path in sorted(root.glob("*/*/metrics.json")):
        directory = metrics_path.parent
        config = yaml.safe_load((directory / "resolved_config.yaml").read_text())
        task = directory.parent.name
        key = (task, config["data"]["regime"], config["model"]["kind"], config["seed"])
        if key in runs:
            raise ValueError(f"Duplicate run for {key}")
        runs[key] = json.loads(metrics_path.read_text())
    if not runs:
        raise ValueError(f"No completed runs found under {root}")
    return runs


def bootstrap_interval(values, statistic, resamples=10000, seed=0):
    """Percentile bootstrap over seeds. Reported because eight seeds is few
    enough that a standard deviation alone invites over-reading."""
    rng = random.Random(seed)
    draws = []
    for _ in range(resamples):
        sample = [values[rng.randrange(len(values))] for _ in values]
        draws.append(statistic(sample))
    draws.sort()
    low = draws[int(0.025 * len(draws))]
    high = draws[min(int(0.975 * len(draws)), len(draws) - 1)]
    return low, high


def paired_ratio(runs, task, regime, metric, treatment, control):
    """Per-seed percentage change of `treatment` against `control`."""
    seeds = sorted(
        seed
        for (t, r, kind, seed) in runs
        if t == task and r == regime and kind == control
    )
    pairs = []
    for seed in seeds:
        a = runs.get((task, regime, control, seed))
        b = runs.get((task, regime, treatment, seed))
        if a is None or b is None:
            continue
        pairs.append((b[metric] - a[metric]) / a[metric] * 100)
    return pairs


def mean(values):
    return sum(values) / len(values)


def median(values):
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--metric", default="rollout_error")
    args = parser.parse_args()

    runs = load_runs(args.root)
    tasks = sorted({task for task, _, _, _ in runs})
    regimes = sorted({regime for _, regime, _, _ in runs})

    print(f"metric: {args.metric}\n")
    print("per-baseline values (mean over seeds):")
    header = f"{'task':12s} {'regime':12s} " + " ".join(f"{b:>22s}" for b in BASELINES)
    print(header)
    print("-" * len(header))
    for task in tasks:
        for regime in regimes:
            cells = []
            for kind in BASELINES:
                values = [
                    m[args.metric]
                    for (t, r, k, _), m in runs.items()
                    if (t, r, k) == (task, regime, kind)
                ]
                cells.append(
                    f"{mean(values):.4f} (n={len(values)})" if values else "missing"
                )
            print(f"{task:12s} {regime:12s} " + " ".join(f"{c:>22s}" for c in cells))

    for treatment in ("relational", "joint"):
        print(
            f"\npaired {treatment} vs independent, "
            "% change per seed (negative = better):"
        )
        header = (
            f"{'task':12s} {'regime':12s} {'mean':>9s} {'median':>9s} "
            f"{'95% CI (bootstrap)':>24s} {'seeds better':>13s}"
        )
        print(header)
        print("-" * len(header))
        for task in tasks:
            for regime in regimes:
                pairs = paired_ratio(
                    runs, task, regime, args.metric, treatment, "independent"
                )
                if not pairs:
                    continue
                low, high = bootstrap_interval(pairs, mean)
                better = sum(1 for value in pairs if value < 0)
                print(
                    f"{task:12s} {regime:12s} {mean(pairs):+8.1f}% "
                    f"{median(pairs):+8.1f}% "
                    f"{f'[{low:+.1f}, {high:+.1f}]':>24s} "
                    f"{f'{better}/{len(pairs)}':>13s}"
                )

    print("\nper-seed dispersion (a wide spread means the estimate is unstable):")
    header = f"{'task':12s} {'regime':12s} {'kind':12s} {'min':>9s} {'median':>9s} {'max':>9s}"
    print(header)
    print("-" * len(header))
    for task in tasks:
        for regime in regimes:
            for kind in BASELINES:
                values = [
                    m[args.metric]
                    for (t, r, k, _), m in runs.items()
                    if (t, r, k) == (task, regime, kind)
                ]
                if values:
                    print(
                        f"{task:12s} {regime:12s} {kind:12s} "
                        f"{min(values):9.4f} {median(values):9.4f} {max(values):9.4f}"
                    )

    health = defaultdict(list)
    for (task, _, kind, _), m in runs.items():
        health[(task, kind)].append(m)
    print("\nlatent health and readout (mean over seeds and regimes):")
    header = (
        f"{'task':12s} {'kind':12s} {'rank':>7s} {'lat.var':>8s} "
        f"{'reward_rel':>11s} {'term_rate':>10s} {'reload_max':>11s}"
    )
    print(header)
    print("-" * len(header))
    for (task, kind), values in sorted(health.items()):

        def pick(key, values=values):
            present = [v[key] for v in values if key in v]
            return mean(present) if present else float("nan")

        print(
            f"{task:12s} {kind:12s} {pick('effective_rank'):7.1f} "
            f"{pick('latent_variance'):8.3f} {pick('reward_relative_error'):11.3f} "
            f"{pick('terminated_rate'):10.4f} "
            f"{max(v['checkpoint_reload_max_difference'] for v in values):11.1f}"
        )


if __name__ == "__main__":
    main()
