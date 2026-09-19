from examples.world_model.g6a_report import endpoint


def _row(seed, iteration, horizon, ball_rmse, regret=0.0):
    return {
        "seed": seed, "iteration": iteration, "horizon": horizon,
        "teacher_forced": {
            "ball_position_rmse": ball_rmse, "selected_regret": regret,
        },
        "recursive": {
            "true_collision_given_predicted_top10": 0.3,
            "predicted_collision_given_predicted_top10": 0.1,
        },
        "recursive_vs_teacher_forced": {"ball_position_rmse": 0.02},
    }


def test_endpoint_reads_iteration_1_and_30_at_horizon_5():
    rows = [
        _row(9100, 1, 5, ball_rmse=0.01),
        _row(9100, 30, 5, ball_rmse=0.03, regret=4.0),
        _row(9100, 30, 4, ball_rmse=0.5),  # a different horizon must be ignored
    ]
    result = {"rows": rows}
    metrics = endpoint(result, 9100)
    assert metrics["tf_ball_rmse_iter1_h5"] == 0.01
    assert metrics["tf_ball_rmse_iter30_h5"] == 0.03
    assert metrics["tf_ratio_30_over_1_h5"] == 3.0
    assert metrics["selected_regret_tf_iter30"] == 4.0
    assert metrics["recursive_vs_tf_gap_iter30_h5"] == 0.02
    assert metrics["false_safe_top10_true"] == 0.3
    assert metrics["false_safe_top10_predicted"] == 0.1
