# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the license in the LICENSE file in the root directory.
"""Episode-level uncertainty for the paired oracle/random pilot."""

import math

import numpy as np


def mean_interval(values, rng, resamples=10000):
    values = np.asarray(values, dtype=float)
    means = rng.choice(values, size=(resamples, len(values)), replace=True).mean(axis=1)
    return {
        "mean": float(values.mean()),
        "ci95": np.quantile(means, [0.025, 0.975]).tolist(),
    }


def success_interval(successes):
    n = len(successes)
    p = sum(successes) / n
    z = 1.959963984540054
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    radius = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return {
        "count": int(sum(successes)),
        "rate": p,
        "ci95": [max(0, center - radius), min(1, center + radius)],
    }


def summarize(rows, seed=0):
    rng = np.random.default_rng(seed)
    summary = {}
    returns = {}
    for policy in ("mpc", "random"):
        episodes = [row for row in rows if row["policy"] == policy]
        returns[policy] = [row["return"] for row in episodes]
        summary[policy] = {
            "episodes": len(episodes),
            "return": mean_interval(returns[policy], rng),
            "team_return": mean_interval([r["team_return"] for r in episodes], rng),
            "success": success_interval([r["success"] for r in episodes]),
            "collision_rate": float(np.mean([r["collision"] for r in episodes])),
            "timeout_rate": float(np.mean([r["timeout"] for r in episodes])),
        }
    summary["paired_return_difference"] = mean_interval(
        np.asarray(returns["mpc"]) - np.asarray(returns["random"]), rng
    )
    summary["m2_return_gate"] = (
        summary["mpc"]["return"]["ci95"][0] > summary["random"]["return"]["ci95"][1]
    )
    summary["objective_status"] = (
        "unvalidated: no task successes"
        if summary["mpc"]["success"]["count"] == 0
        else "pilot goals observed; numerical success gate and held-out confirmation pending"
    )
    return summary
