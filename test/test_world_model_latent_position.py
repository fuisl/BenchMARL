# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the license in the LICENSE file in the root directory.
"""Physical-position measurement for latent world-model rollouts."""

import torch

from examples.world_model.latent_position import curve_summary, position_summary


def test_position_summary_separates_error_from_persistence():
    truth = torch.tensor([[[[0.03, 0.04], [0.00, 0.02]]]])
    baseline = torch.zeros_like(truth)
    valid = torch.ones(1, 1, dtype=torch.bool)
    perfect = position_summary(truth, truth, baseline, valid)
    stationary = position_summary(baseline, truth, baseline, valid)
    assert perfect["coordinate_rmse"] == 0.0
    assert perfect["relative_mse_to_persistence"] == 0.0
    assert stationary["relative_mse_to_persistence"] == 1.0


def test_position_summary_masks_invalid_blocks():
    truth = torch.tensor([[[[1.0, 0.0]], [[100.0, 0.0]]]])
    prediction = torch.tensor([[[[0.0, 0.0]], [[-100.0, 0.0]]]])
    baseline = torch.zeros_like(truth)
    valid = torch.tensor([[True, False]])
    result = position_summary(prediction, truth, baseline, valid)
    assert result["positions"] == 1
    assert result["relative_mse_to_persistence"] == 1.0


def test_curve_summary_uses_horizon_specific_masks():
    truth = torch.tensor([[[[1.0, 0.0]], [[2.0, 0.0]]]])
    prediction = truth.clone()
    baseline = torch.zeros_like(truth)
    valid = torch.ones(1, 2, dtype=torch.bool)
    curve = curve_summary(prediction, truth, baseline, valid)
    assert len(curve) == 2
    assert all(point["coordinate_rmse"] == 0.0 for point in curve)
