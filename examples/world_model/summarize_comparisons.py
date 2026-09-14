# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the license in the LICENSE file in the root directory.
"""Summarize paired-state runs without treating repeated seeds as new episodes."""

import argparse
import csv
import json
from pathlib import Path

from omegaconf import OmegaConf


def summarize_comparisons(root: Path):
    rows = []
    random_reference = None
    for path in sorted(root.glob("*/summary.json"), key=lambda p: int(p.parent.name)):
        summary = json.loads(path.read_text())
        config = OmegaConf.load(path.parent / "resolved_config.yaml")
        timing = json.loads((path.parent / "timing.json").read_text())
        with (path.parent / "episodes.csv").open() as file:
            episodes = list(csv.DictReader(file))
        random_rows = [r for r in episodes if r["policy"] == "random"]
        if random_reference is None:
            random_reference = random_rows
        if random_rows != random_reference:
            raise ValueError("Random reference episodes differ across comparisons")
        rows.append(
            {
                "run": path.parent.name,
                "comparison": config.comparison_name,
                "seed": config.seed,
                "horizon": config.cem.horizon,
                "receding_horizon": config.mpc.receding_horizon,
                "action_block": config.mpc.action_block,
                "samples": config.cem.num_samples,
                "iterations": config.cem.num_iters,
                "return": summary["mpc"]["return"]["mean"],
                "return_ci_low": summary["mpc"]["return"]["ci95"][0],
                "return_ci_high": summary["mpc"]["return"]["ci95"][1],
                "success_rate": summary["mpc"]["success"]["rate"],
                "collision_rate": summary["mpc"]["collision_rate"],
                "timeout_rate": summary["mpc"]["timeout_rate"],
                "seconds": timing["mpc"]["seconds"],
                "m2_return_gate": summary["m2_return_gate"],
                "state_bank_sha256": summary["state_bank_sha256"],
            }
        )
    if not rows:
        raise ValueError("No completed comparison runs found")
    if len({row["state_bank_sha256"] for row in rows}) != 1:
        raise ValueError("Comparison runs used different state banks")
    with (root / "comparison.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(
        f"Summarized {len(rows)} runs; identical state bank and random episodes verified."
    )
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    summarize_comparisons(args.root)
