#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#
"""Aggregate a sweep with BenchMARL's own reporting.

`train.py` writes a marl-eval file per run through BenchMARL's ``JsonWriter``,
so aggregation is `benchmarl.eval_results`' job and this module only drives it:
merge the files, build the rliable matrices, print the scores.

The one thing it cannot delegate is the interval. ``Plotting``'s figures call
``rliable.library.get_interval_estimates``, which builds an ``arch``
``StratifiedBootstrap``; arch 7.2 fails to import against the installed pandas
3.0, and arch 8.0 renamed the ``random_state`` argument rliable still passes, so
neither version works. Pinning pandas back to marl-eval's 1.4.4 would mean
downgrading the environment the experiments run in. Point estimates still come
from ``rliable.metrics``; only the interval is computed here.
"""

import json
from pathlib import Path

import numpy as np
from rliable import metrics as rliable_metrics
from scipy.stats import trim_mean

from benchmarl.eval_results import load_and_merge_json_dicts, Plotting
from examples.world_model.train import MARL_EVAL_FILE

ENVIRONMENT = "vmas"


def merge_sweeps(sweeps):
    """Merge the marl-eval files under several multirun folders.

    ``get_raw_dict_from_multirun_folder`` would be the natural call, but it
    walks for every json under the folder and our run directories also hold
    metrics, parameters and provenance. The files are named, so name them.
    """
    files = [
        str(path)
        for sweep in sweeps
        for path in sorted(Path(sweep).rglob(MARL_EVAL_FILE))
    ]
    if not files:
        raise ValueError(f"No {MARL_EVAL_FILE} under {', '.join(map(str, sweeps))}")
    return load_and_merge_json_dicts(files)


def bootstrap_iqm(values, resamples=20000, seed=0):
    """IQM with a percentile interval.

    IQM is ``trim_mean`` at 25%, so the vectorised bootstrap computes exactly
    that statistic rather than an approximation of it. Resampling through
    rliable's per-call estimator would be millions of Python calls.
    """
    values = np.asarray(values, dtype=float)
    estimate = float(rliable_metrics.aggregate_iqm(values[:, None]))
    generator = np.random.default_rng(seed)
    draws = values[generator.integers(0, len(values), size=(resamples, len(values)))]
    low, high = np.quantile(
        trim_mean(draws, proportiontocut=0.25, axis=1), [0.025, 0.975]
    )
    return estimate, float(low), float(high)


def aggregate(matrix):
    """IQM per algorithm over the pooled (seeds x tasks) scores."""
    return {
        algorithm: bootstrap_iqm(np.asarray(scores).reshape(-1))
        for algorithm, scores in sorted(matrix.items())
    }


def format_table(title, rows):
    header = f"{'algorithm':28s}{'IQM':>10s}{'95% CI':>24s}"
    lines = [title, header, "-" * len(header)]
    for algorithm, (estimate, low, high) in rows.items():
        lines.append(
            f"{algorithm:28s}{estimate:>10.4f}{f'[{low:.4f}, {high:.4f}]':>24s}"
        )
    return "\n".join(lines)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("sweeps", type=Path, nargs="+", help="multirun folders")
    parser.add_argument("--output", type=Path, default=Path("outputs/report"))
    parser.add_argument("--metric", default="neg_rollout_error")
    args = parser.parse_args()

    raw = merge_sweeps(args.sweeps)
    tasks = raw.get(ENVIRONMENT, {})
    runs = sum(len(seeds) for task in tasks.values() for seeds in task.values())
    print(f"tasks: {', '.join(sorted(tasks))}  ({runs} runs)")

    normalize = [args.metric]
    processed = Plotting.process_data(raw, metrics_to_normalize=normalize)
    environment_matrix, _ = Plotting.create_matrices(
        processed, env_name=ENVIRONMENT, metrics_to_normalize=normalize
    )

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "results.json").write_text(json.dumps(raw, indent=2) + "\n")

    rows = aggregate(environment_matrix[f"mean_norm_{args.metric}"])
    report = format_table(f"{args.metric}, normalised, pooled over tasks", rows)
    (args.output / "aggregate_scores.txt").write_text(report + "\n")
    print("\n" + report)
    print(f"\nwrote {args.output}/results.json and aggregate_scores.txt")


if __name__ == "__main__":
    main()
