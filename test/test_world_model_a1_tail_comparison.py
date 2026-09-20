"""The A1.2 branch rule is registered before the run produces numbers.

These tests pin that rule so a later edit cannot quietly move the threshold
that decides which of G6a/G6b/G6c is justified.
"""

import pytest

from examples.world_model.a1_tail_comparison import GROWTH_THRESHOLD, decide


def arm(
    teacher=(0.01, 0.01, 0.01, 0.01, 0.01),
    recursive=(0.02, 0.02, 0.02, 0.02, 0.02),
    false_safe=0.05,
    predicted_safe=0.05,
):
    stages = ("1", "5", "10", "20", "30")
    return {
        "teacher_forced_ball_rmse": dict(zip(stages, teacher)),
        "recursive_ball_rmse": dict(zip(stages, recursive)),
        "teacher_forced_growth_1_to_30": teacher[-1] / teacher[0],
        "recursive_growth_1_to_30": recursive[-1] / recursive[0],
        "false_safe_top10_at_30": false_safe,
        "predicted_safe_top10_at_30": predicted_safe,
    }


def test_branch_c_when_teacher_forced_tail_survives_in_full32():
    """The old G6a diagnosis survives the stronger test."""
    decision = decide(
        arm(teacher=(0.01, 0.02, 0.03, 0.05, 0.08)),
        arm(teacher=(0.01, 0.02, 0.03, 0.05, 0.09)),
    )
    assert decision["branch"] == "C"
    assert decision["tail_survives_in_full32"] is True
    assert "G6a" in decision["next_experiment"]


def test_branch_a_when_full32_removes_the_blow_up():
    """legacy14 blows up, full32 does not, and nothing else fires."""
    decision = decide(
        arm(teacher=(0.01, 0.01, 0.01, 0.01, 0.011)),
        arm(teacher=(0.01, 0.02, 0.04, 0.06, 0.08)),
    )
    assert decision["branch"] == "A"
    assert decision["tail_substantially_repaired"] is True
    assert "do not run G6a" in decision["next_experiment"]


def test_branch_b_when_only_recursive_error_explodes():
    decision = decide(
        arm(
            teacher=(0.01, 0.01, 0.01, 0.01, 0.011),
            recursive=(0.02, 0.05, 0.10, 0.20, 0.40),
        ),
        arm(teacher=(0.01, 0.02, 0.04, 0.06, 0.08)),
    )
    assert decision["branch"] == "B"
    assert decision["recursive_still_explodes"] is True
    assert "G6b" in decision["next_experiment"]


def test_branch_d_when_only_the_false_safe_tail_remains():
    decision = decide(
        arm(
            teacher=(0.01, 0.01, 0.01, 0.01, 0.011),
            false_safe=0.40,
            predicted_safe=0.05,
        ),
        arm(teacher=(0.01, 0.02, 0.04, 0.06, 0.08)),
    )
    assert decision["branch"] == "D"
    assert decision["false_safe_tail_remains"] is True
    assert "G6c" in decision["next_experiment"]


def test_growth_at_the_threshold_counts_as_surviving():
    """The rule is >= the threshold, so the boundary is not silently excluded."""
    at_threshold = arm(teacher=(0.01, 0.01, 0.01, 0.01, 0.01 * GROWTH_THRESHOLD))
    decision = decide(at_threshold, arm())
    assert decision["full32_teacher_forced_growth"] == pytest.approx(
        GROWTH_THRESHOLD
    )
    assert decision["branch"] == "C"


def test_unmeasurable_growth_is_undetermined_rather_than_a_branch():
    """A degenerate arm must not be reported as evidence for any branch."""
    broken = arm()
    broken["teacher_forced_growth_1_to_30"] = None
    decision = decide(broken, arm())
    assert decision["branch"] == "undetermined"
    assert "next_experiment" not in decision
