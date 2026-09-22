#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#
"""The cheap A-vs-S admission gate, and the cell it should be run on.

Job 1540 separated two things this project had been treating as one. Balance at
`h=3` has a real simulator interaction -- job 1525 measured `J_cross` rising
0.049 -> 1.110 over `h = 1..5` -- and the audit still could not measure
anything, because under the sampled-reference design the privileged physical
state did not beat the action-only blind head. So:

    a true interaction EXISTS          (J_cross, activity: descriptive)
    the instrument can RESOLVE it      (E_A - E_S > 0: the admission criterion)

are different claims, and only the second admits a task to the expensive
representation ladder. Activity alone is demoted to a descriptor; the old 50%
activity threshold is no longer an admission rule on its own.

Two entry points:

    --pick-cell   read a T-A1 JSON, print the strongest CROSS cell as
                  AGENT:AXIS. The gate is run where the effect is largest: a
                  reference that cannot beat blind THERE will not beat it
                  anywhere, so this is the most favourable test available.

    --summarize   read the T-A1 and localization JSONs for every (task,
                  horizon) cell and print one admission table.
"""

import argparse
import json
import math
from pathlib import Path

# A gate whose gap is positive but within bootstrap noise is not a pass. The
# interval on the blind floor and on the reference must not overlap.
ADMIT = "ADMIT"
REJECT_NO_TRUE_EFFECT = "no true effect to resolve"
REJECT_NO_EFFECT = "no resolvable effect"
REJECT_NOT_SEPARATED = "reference does not beat blind"
REJECT_NO_SUPPORT = "too little surviving support"

# Minimum support for the separation test to mean anything. Buzz Wire at h=3
# leaves ONE anchor in ONE episode -- everything else terminates by step 14 --
# and a bootstrap over a single episode returns that anchor's value every time.
# The interval collapses to a point, so `E_S high < E_A low` is satisfied
# trivially and the cell was admitted on a sample of one.
#
# These thresholds were chosen AFTER seeing that failure, not before it. Their
# justification is structural rather than tuned: the banks hold 16 root
# episodes, so requiring half of them keeps the episode-clustered resample from
# being dominated by any single episode, and 50 anchors is the floor below
# which the head is fitting a handful of points. Every cell in job 1541 either
# clears both comfortably (311-655 anchors, 16 episodes) or fails them badly
# (1 anchor, 1 episode), so no result in that run is sensitive to where exactly
# between those the line sits.
MIN_EPISODES = 8
MIN_ANCHORS = 50


def _best_cross(jacobian_path):
    """The strongest cross cell AT THE RUN'S OWN HORIZON.

    A T-A1 run at horizon h writes cells for every horizon 1..h, because the
    loop reports each block boundary along the way. Ranking all of them
    together picks whichever horizon happens to carry the largest response,
    which is not the horizon the gate then runs at -- on transport h=3 that
    returned an h=2 cell, and the descriptor columns described a different
    measurement from the one being gated.
    """
    payload = json.loads(Path(jacobian_path).read_text())
    horizon = payload["horizon_blocks"]
    cross = [
        v for v in payload["cells"].values()
        if v.get("block") == "cross" and v.get("horizon_blocks") == horizon
    ]
    if not cross:
        return None
    return max(cross, key=lambda v: v["mean"])


def strongest_cross_cell(jacobian_path):
    """The (agent, axis) whose CROSS response is largest, as `AGENT:AXIS`."""
    best = _best_cross(jacobian_path)
    if best is None:
        raise ValueError(f"{jacobian_path} has no cross cells at its own horizon")
    return f"{best['intervened_agent']}:{best['intervened_axis']}"


def cross_summary(jacobian_path):
    """Largest cross cell's magnitude and activity, for the descriptor columns."""
    best = _best_cross(jacobian_path)
    if best is None:
        return {"j_cross": 0.0, "active": 0.0, "cell": "-"}
    return {
        "j_cross": best["mean"],
        "active": best["active_fraction_above_1e-6"],
        "cell": f"{best['intervened_agent']}:{best['intervened_axis']}",
    }


def gate(localization_path):
    """E_A, E_S, the gap, and whether the intervals separate."""
    shared = json.loads(Path(localization_path).read_text())["shared"]
    blind = shared["actions_only"]["test_cross"]
    reference = shared["physical"]["test_cross"]
    gap = blind["mean"] - reference["mean"]
    # Bootstrapped over root episodes, so this is the clustered interval, not
    # an anchor-level one that would overstate separation.
    separated = reference["high"] < blind["low"]
    episodes = min(blind["episodes"], reference["episodes"])
    anchors = min(blind["anchors"], reference["anchors"])
    return {
        "e_a": blind["mean"],
        "e_a_low": blind["low"],
        "e_s": reference["mean"],
        "e_s_high": reference["high"],
        "gap": gap,
        "separated": bool(separated),
        "episodes": int(episodes),
        "anchors": int(anchors),
        "supported": bool(episodes >= MIN_EPISODES and anchors >= MIN_ANCHORS),
    }


def verdict(cross, gated):
    # Dropout's cross response is identically zero, so the pooled ratio is 0/0.
    # That is the negative control behaving correctly, and it must not be
    # reported as "the reference failed to beat blind" -- there is nothing for
    # either head to predict. `nan` fails every comparison, so without this
    # branch it would fall through to the separation test and be mislabelled.
    if math.isnan(gated["gap"]) or cross["j_cross"] <= 0.0:
        return REJECT_NO_TRUE_EFFECT
    # Checked BEFORE separation: a degenerate interval separates trivially, so
    # support has to gate the test rather than be reported next to it.
    if not gated.get("supported", True):
        return REJECT_NO_SUPPORT
    if gated["gap"] <= 0:
        return REJECT_NO_EFFECT
    if not gated["separated"]:
        return REJECT_NOT_SEPARATED
    return ADMIT


def summarize(root):
    root = Path(root)
    rows = []
    for cell_dir in sorted(root.glob("*/h*")):
        task, horizon = cell_dir.parent.name, int(cell_dir.name[1:])
        jacobian = cell_dir / "ta1" / "interaction_jacobian.json"
        localization = cell_dir / "gate" / "counterfactual_localization.json"
        if not jacobian.exists():
            rows.append({"task": task, "h": horizon, "verdict": "T-A1 missing"})
            continue
        cross = cross_summary(jacobian)
        if not localization.exists():
            rows.append({"task": task, "h": horizon, **cross,
                         "verdict": "gate missing"})
            continue
        gated = gate(localization)
        rows.append({"task": task, "h": horizon, **cross, **gated,
                     "verdict": verdict(cross, gated)})
    return rows


def print_table(rows):
    print("\n=== A-vs-S admission gate ===")
    print("J_cross and activity DESCRIBE the effect; the gap ADMITS the task.")
    print("`sep` = the reference's 95% upper bound sits below the blind's lower")
    print("bound, both resampled over root episodes.\n")
    header = (f"  {'task':<11}{'h':>2} {'cell':>5} {'J_cross':>9} {'active':>7} "
              f"{'anch':>5} {'E_A':>7} {'E_S':>7} {'E_A-E_S':>9} {'sep':>4}  verdict")
    print(header)
    print("  " + "-" * (len(header) - 2))
    for r in rows:
        if "e_a" not in r:
            print(f"  {r['task']:<11}{r['h']:>2} {r.get('cell', '-'):>5} "
                  f"{r.get('j_cross', float('nan')):>9.3f} "
                  f"{r.get('active', float('nan')):>7.3f} "
                  f"{'':>5}{'':>7}{'':>7}{'':>9}{'':>4}  {r['verdict']}")
            continue
        print(f"  {r['task']:<11}{r['h']:>2} {r['cell']:>5} {r['j_cross']:>9.3f} "
              f"{r['active']:>7.3f} {r['anchors']:>5} {r['e_a']:>7.4f} "
              f"{r['e_s']:>7.4f} {r['gap']:>+9.4f} "
              f"{'yes' if r['separated'] else 'no':>4}  {r['verdict']}")
    admitted = [r for r in rows if r.get("verdict") == ADMIT]
    print(f"\n  ADMITTED: {len(admitted)} of {len(rows)} cells")
    for r in admitted:
        print(f"    {r['task']} h={r['h']} cell {r['cell']}  "
              f"gap {r['gap']:+.4f}  J_cross {r['j_cross']:.3f}  "
              f"({r['anchors']} anchors / {r['episodes']} episodes)")
    if not admitted:
        print("    none -- no task/horizon supports the representation ladder.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pick-cell", type=Path, default=None)
    parser.add_argument("--summarize", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    if args.pick_cell:
        print(strongest_cross_cell(args.pick_cell))
        return
    if args.summarize:
        rows = summarize(args.summarize)
        print_table(rows)
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(rows, indent=2))
            print(f"\nWrote {args.out}")
        return
    parser.error("pass --pick-cell or --summarize")


if __name__ == "__main__":
    main()
