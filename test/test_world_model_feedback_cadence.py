from examples.world_model.feedback_cadence import (
    first_action_quality_from_gate5,
    k1_row_from_job_1331,
    summarize_episodes,
)


def test_summarize_episodes_reduces_to_rates_and_means():
    episodes = [
        {
            "success": True, "collision": False, "timeout": False,
            "return": 1.0, "final_goal_distance": 0.1, "length": 20,
        },
        {
            "success": False, "collision": True, "timeout": False,
            "return": -3.0, "final_goal_distance": 0.5, "length": 40,
        },
    ]
    summary = summarize_episodes(episodes)
    assert summary["episodes"] == 2
    assert summary["success_rate"] == 0.5
    assert summary["collision_rate"] == 0.5
    assert summary["timeout_rate"] == 0.0
    assert summary["return_mean"] == -1.0
    assert summary["final_goal_distance_mean"] == 0.3
    assert summary["episode_length_mean"] == 30


def test_k1_row_reuses_job_1331_control_rows_without_recomputation():
    prior = {
        "control_rows": [
            {
                "policy": "structured_full_9100", "success": True, "collision": False,
                "timeout": False, "return": 2.0, "final_goal_distance": 0.05, "length": 10,
            },
            {
                "policy": "structured_full_9101", "success": False, "collision": True,
                "timeout": False, "return": -5.0, "final_goal_distance": 0.9, "length": 60,
            },
        ],
        "timing": {
            "structured_full_9100": {
                "seconds": 12.0,
                "decisions": [{"seconds": 0.6}, {"seconds": 0.6}],
            }
        },
    }
    row = k1_row_from_job_1331(prior, 9100)
    assert row["K"] == 1
    assert row["source"] == "job_1331_control_rows"
    assert row["summary"]["episodes"] == 1
    assert row["summary"]["success_rate"] == 1.0
    assert row["timing"]["decisions"] == 2
    assert row["timing"]["seconds_per_decision"] == 0.6


def test_first_action_quality_cites_gate5_iteration30_horizon5_teacher_forced():
    gate5_result = {
        "rows": [
            {
                "seed": 9100, "iteration": 1, "horizon": 5,
                "teacher_forced": {"selected_regret": 99.0, "oracle_percentile": 0.9, "spearman": 0.1},
            },
            {
                "seed": 9100, "iteration": 30, "horizon": 5,
                "teacher_forced": {"selected_regret": 3.5, "oracle_percentile": 0.8, "spearman": -0.2},
            },
            {
                "seed": 9100, "iteration": 30, "horizon": 4,
                "teacher_forced": {"selected_regret": 1.0, "oracle_percentile": 0.1, "spearman": 0.4},
            },
        ]
    }
    quality = first_action_quality_from_gate5(gate5_result, 9100)
    assert quality["selected_regret"] == 3.5
    assert quality["oracle_percentile"] == 0.8
    assert quality["spearman"] == -0.2
    assert quality["source"] == "job_1335_iteration30_horizon5_teacher_forced"
