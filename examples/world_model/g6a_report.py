# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the license found in the LICENSE file in the root directory.
"""Experiment 24/G6a comparison report.

Reruns Gate 5 (`planner_tail_failure.py`) unchanged on each aggregation round
produce a `result.json` in exactly Gate 5's own schema, so the pre-registered
`registered_branch_result` rule is reapplied verbatim rather than re-derived:
this reports whether G6a's own registered OOD signature -- iteration-30
teacher-forced ball-position/clearance/cost RMSE at least 1.5x iteration-1,
at 3+ of 5 horizons, for 2+ seeds -- still fires after planner-aware
aggregation, and beside it, whether the matched generic-data control fires
too. If both fire equally, aggregation is not the mechanism; if only the
generic arm still fires, planner-aware coverage is what closed the gap.
"""

import argparse
import json
from pathlib import Path

from examples.world_model.planner_tail_failure import registered_branch_result


SEEDS = (9100, 9101, 9102)


def endpoint(result, seed):
    rows = {
        (row["iteration"], row["horizon"]): row["teacher_forced"]
        for row in result["rows"]
        if row["seed"] == seed
    }
    early, late = rows[(1, 5)], rows[(30, 5)]
    ratio = (
        late["ball_position_rmse"] / early["ball_position_rmse"]
        if early["ball_position_rmse"]
        else None
    )
    recursive_row = next(
        row for row in result["rows"]
        if row["seed"] == seed and row["iteration"] == 30 and row["horizon"] == 5
    )
    return {
        "seed": seed,
        "tf_ball_rmse_iter1_h5": early["ball_position_rmse"],
        "tf_ball_rmse_iter30_h5": late["ball_position_rmse"],
        "tf_ratio_30_over_1_h5": ratio,
        "recursive_vs_tf_gap_iter30_h5": recursive_row["recursive_vs_teacher_forced"][
            "ball_position_rmse"
        ],
        "false_safe_top10_true": recursive_row["recursive"][
            "true_collision_given_predicted_top10"
        ],
        "false_safe_top10_predicted": recursive_row["recursive"][
            "predicted_collision_given_predicted_top10"
        ],
        "selected_regret_tf_iter30": late["selected_regret"],
    }


def summarize_source(path):
    """`path` is the directory holding `result.json` directly -- job 1335's
    own `result/` subdirectory, or a G6a round's `gate5_result/` output."""
    result = json.loads((path / "result.json").read_text())
    branch = registered_branch_result(result["rows"])
    return {
        "endpoints": [endpoint(result, seed) for seed in SEEDS],
        "registered_branch_result": branch,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--baseline", type=Path, required=True,
        help="directory holding result.json directly, e.g. job 1335's own result/",
    )
    parser.add_argument("--targeted-round1", type=Path, required=True)
    parser.add_argument("--targeted-round2", type=Path, required=True)
    parser.add_argument("--generic-round1", type=Path, required=True)
    parser.add_argument("--generic-round2", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    sources = {
        "M0_baseline": args.baseline,
        "G6a_targeted_round1": args.targeted_round1,
        "G6a_targeted_round2": args.targeted_round2,
        "generic_round1": args.generic_round1,
        "generic_round2": args.generic_round2,
    }
    report = {label: summarize_source(path) for label, path in sources.items()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    header = (
        f"{'label':22s}{'seed':>6s}{'TF@1,h5':>9s}{'TF@30,h5':>10s}{'ratio':>7s}"
        f"{'rec-tf gap':>12s}{'fs top10 T':>12s}{'fs top10 P':>12s}"
    )
    print(header)
    print("-" * len(header))
    for label, entry in report.items():
        for row in entry["endpoints"]:
            print(
                f"{label:22s}{row['seed']:>6d}{row['tf_ball_rmse_iter1_h5']:>9.4f}"
                f"{row['tf_ball_rmse_iter30_h5']:>10.4f}"
                f"{(row['tf_ratio_30_over_1_h5'] or 0):>7.2f}"
                f"{row['recursive_vs_tf_gap_iter30_h5']:>12.4f}"
                f"{row['false_safe_top10_true']:>12.3f}"
                f"{row['false_safe_top10_predicted']:>12.3f}"
            )
    print()
    print(f"{'label':22s}{'G6a fires':>11s}{'G6b fires':>11s}{'G6c fires':>11s}")
    for label, entry in report.items():
        fired = entry["registered_branch_result"]["all_rules"]
        print(
            f"{label:22s}"
            f"{str(fired['G6a_planner_aware_aggregation']):>11s}"
            f"{str(fired['G6b_action_prefix_dynamics']):>11s}"
            f"{str(fired['G6c_conservative_collision_scoring']):>11s}"
        )


if __name__ == "__main__":
    main()
