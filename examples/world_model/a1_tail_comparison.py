"""A1.2 paired comparison: does the optimizer tail survive the Markov repair?

Each arm of A1.2 is measured by ``planner_tail_failure --protocol a1_paired``
on banks that differ only in the state representation.  This script reads those
per-arm results and answers the one registered question:

    does teacher-forced error still grow from CEM iteration 1 to 30 once the
    linkage alias that A1.1 demonstrated in ``legacy14`` is removed?

It selects nothing and trains nothing.  The pre-registered branch that a run
lands in is decided here from the recorded numbers alone.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

STAGES = (1, 5, 10, 20, 30)
HORIZON = 5
# The audit's decision rule, registered before the run produced any numbers.
# A ratio at or above this is "the tail still grows"; at or below its inverse
# side, the blow-up is considered removed rather than merely reduced.
GROWTH_THRESHOLD = 1.5
REPAIR_FRACTION = 0.5


def load_arm(path):
    result = json.loads((path / "result.json").read_text())
    rows = {
        (row["iteration"], row["horizon"]): row
        for row in result["summary_across_model_seeds"]
    }
    return result, rows


def curve(rows, path_name, key):
    return [rows[(stage, HORIZON)][path_name].get(key) for stage in STAGES]


def ratio(values):
    first, last = values[0], values[-1]
    if first is None or last is None or first <= 0:
        return None
    return last / first


def arm_summary(result, rows):
    teacher = curve(rows, "teacher_forced", "ball_position_rmse")
    recursive = curve(rows, "recursive", "ball_position_rmse")
    gap = [
        rows[(stage, HORIZON)]["recursive_vs_teacher_forced"]["ball_position_rmse"]
        for stage in STAGES
    ]
    final = rows[(30, HORIZON)]
    return {
        "state_profile": result["state_profile"],
        "model_parameters": result.get("model_parameters"),
        "teacher_forced_ball_rmse": dict(zip(map(str, STAGES), teacher)),
        "recursive_ball_rmse": dict(zip(map(str, STAGES), recursive)),
        "teacher_forced_growth_1_to_30": ratio(teacher),
        "recursive_growth_1_to_30": ratio(recursive),
        "recursive_minus_teacher_gap_at_30": gap[-1],
        "link_body_rmse_at_30": final["teacher_forced"].get(
            "link_body_coordinate_rmse"
        ),
        "full_dynamic_rmse_at_30": final["teacher_forced"].get(
            "full_dynamic_coordinate_rmse"
        ),
        "clearance_rmse_at_30": final["teacher_forced"].get("clearance_rmse"),
        "spearman_at_30": final["recursive"].get("spearman"),
        "false_safe_top10_at_30": final["recursive"].get(
            "true_collision_given_predicted_top10"
        ),
        "predicted_safe_top10_at_30": final["recursive"].get(
            "predicted_collision_given_predicted_top10"
        ),
        "collision_brier_at_30": final["recursive"].get("collision_brier"),
        "oracle_percentile_at_30": final["recursive"].get("oracle_percentile"),
        "selected_regret_at_30": final["recursive"].get("selected_regret"),
        "registered_branch_result": result["registered_branch_result"],
    }


def decide(full, legacy):
    """Return the pre-registered branch this run lands in."""
    full_growth = full["teacher_forced_growth_1_to_30"]
    legacy_growth = legacy["teacher_forced_growth_1_to_30"]
    if full_growth is None or legacy_growth is None:
        return {
            "branch": "undetermined",
            "reason": "teacher-forced growth was not measurable in some arm",
        }
    tail_survives = full_growth >= GROWTH_THRESHOLD
    repaired = (
        legacy_growth >= GROWTH_THRESHOLD
        and full_growth <= REPAIR_FRACTION * legacy_growth
    )
    recursive_explodes = (
        full["recursive_growth_1_to_30"] is not None
        and full["recursive_growth_1_to_30"] >= GROWTH_THRESHOLD
    )
    false_safe = (
        full["false_safe_top10_at_30"] is not None
        and full["predicted_safe_top10_at_30"] is not None
        and full["false_safe_top10_at_30"] >= 0.25
        and full["false_safe_top10_at_30"] - full["predicted_safe_top10_at_30"]
        >= 0.15
    )
    if tail_survives:
        branch, action = "C", "corrected G6a on full32"
    elif repaired and not recursive_explodes and not false_safe:
        branch, action = "A", "the alias was upstream of the Gate-5 tail; do not run G6a"
    elif recursive_explodes:
        branch, action = "B", "multi-step compounding remains; G6b"
    elif false_safe:
        branch, action = "D", "event calibration remains; G6c"
    else:
        branch, action = "A", "no registered failure signature fired"
    return {
        "branch": branch,
        "next_experiment": action,
        "full32_teacher_forced_growth": full_growth,
        "legacy14_teacher_forced_growth": legacy_growth,
        "growth_threshold": GROWTH_THRESHOLD,
        "tail_survives_in_full32": tail_survives,
        "tail_substantially_repaired": repaired,
        "recursive_still_explodes": recursive_explodes,
        "false_safe_tail_remains": false_safe,
    }


def render(arms, output):
    metrics = (
        ("teacher_forced", "ball_position_rmse", "Teacher-forced ball RMSE"),
        ("recursive", "ball_position_rmse", "Recursive ball RMSE"),
        ("recursive", "spearman", "Plan-ranking Spearman"),
        (
            "recursive",
            "true_collision_given_predicted_top10",
            "True collision in predicted best 10%",
        ),
    )
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for axis, (path_name, key, title) in zip(axes.flat, metrics):
        for name, (_result, rows) in arms.items():
            values = curve(rows, path_name, key)
            if all(value is None for value in values):
                continue
            axis.plot(STAGES, values, marker="o", label=name)
        axis.set_title(title)
        axis.set_xlabel("CEM iteration")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    figure.suptitle(
        "A1.2: does the optimizer tail survive the full32 Markov repair?",
        fontsize=15,
    )
    figure.savefig(output / "a1_tail_comparison.png", dpi=180)
    figure.savefig(output / "a1_tail_comparison.pdf")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--arms",
        nargs="+",
        default=["full32_h256", "legacy14_h256", "legacy14_h264"],
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    arms = {name: load_arm(args.run / f"tail_{name}") for name in args.arms}
    summaries = {
        name: arm_summary(result, rows) for name, (result, rows) in arms.items()
    }
    roots = {
        name: tuple(result["frozen_protocol"]["test_root_ids"])
        for name, (result, _rows) in arms.items()
    }
    if len(set(roots.values())) != 1:
        raise ValueError(f"Arms were measured on different roots: {roots}")

    # The capacity-matched legacy arm is the primary control; the historical
    # width is retained so the job-1331 lineage stays visible.
    comparison = {
        "question": (
            "Does late-CEM teacher-forced error still grow after removing the "
            "state alias A1.1 demonstrated in legacy14?"
        ),
        "cem_iterations": STAGES,
        "horizon": HORIZON,
        "test_root_ids": list(next(iter(roots.values()))),
        "arms": summaries,
        "registered_decision": decide(
            summaries["full32_h256"], summaries["legacy14_h264"]
        ),
    }
    if "legacy14_h256" in summaries:
        comparison["decision_against_historical_width"] = decide(
            summaries["full32_h256"], summaries["legacy14_h256"]
        )
    (args.output / "comparison.json").write_text(
        json.dumps(comparison, indent=2, allow_nan=False) + "\n"
    )
    render(arms, args.output)
    print(json.dumps(comparison["registered_decision"], indent=2), flush=True)


if __name__ == "__main__":
    main()
