# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the license in the LICENSE file in the root directory.
"""Wheel replay, reward semantics and rotation-aware intervention labels."""

import json
import sys

import torch

from benchmarl.environments import VmasTask
from examples.world_model.collect import rollout_actions
from examples.world_model.counterfactual_evaluation import main as counterfactual_main
from examples.world_model.snapshot_restore import snapshot_state
from examples.world_model.stratified_evaluation import effect_labels
from examples.world_model.stratified_evaluation import main as stratified_main


def test_rotation_labels_respect_anchor_ids_intervention_and_boundary(tmp_path):
    reference = {
        "valid": torch.ones(4, 3, dtype=torch.bool),
        "next_agent_state": torch.zeros(4, 3, 4, 6),
        "next_package_state": torch.zeros(4, 3, 1, 6),
    }
    ids = torch.tensor([3, 1, 2])
    counterfactual = {key: value[ids].clone() for key, value in reference.items()}
    counterfactual["anchor_id"] = ids
    # A pure angular-velocity response is a physical effect on Wheel.
    counterfactual["next_package_state"][0, 1, 0, 5] = 0.1
    # The intervened agent's own motion is not a cross-agent effect.
    counterfactual["next_agent_state"][1, :, 1, 0] = 0.2
    # Rotation after either branch has stopped must not count.
    reference["valid"][2, 1:] = False
    counterfactual["next_package_state"][2, 1:, 0, 4] = 0.3
    torch.save(reference, tmp_path / "samples_correlated.pt")
    torch.save(counterfactual, tmp_path / "counterfactual_test.pt")
    returned, labels = effect_labels(tmp_path, 3, full_state=True)
    assert torch.equal(returned, ids)
    assert labels.tolist() == [True, False, False]
    assert not effect_labels(tmp_path, 1, full_state=True)[1].any()
    assert not effect_labels(tmp_path, 3)[1].any()


def test_wheel_rotation_replays_and_reward_matches_observed_speed_error():
    task = VmasTask.WHEEL.get_from_yaml()
    task.config["max_steps"] = 6
    env = task.get_env_fun(2, True, 37, "cpu")()
    try:
        env.reset()
        env._env.scenario.line.state.ang_vel[:] = torch.tensor([[0.1], [-0.1]])
        initial = snapshot_state(env)
        actions = torch.zeros(2, 8, 4, 2)
        data, _ = rollout_actions(env, initial, actions)
        replay, _ = rollout_actions(env, initial, actions)
        for key in data:
            assert torch.equal(data[key], replay[key]), key
        valid = data["valid"]
        assert valid.sum(1).tolist() == [6, 6]
        assert not data["terminated"].any()
        assert torch.equal(data["done"], data["truncated"])
        assert data["next_observation"].shape[-1] == 13
        torch.testing.assert_close(
            data["reward"][valid], -data["next_observation"][valid][..., -1:]
        )
        assert not data["next_package_state"][valid][..., :4].count_nonzero()
        assert data["next_package_state"][valid][..., 4:].count_nonzero()
    finally:
        env.close()


def test_empty_interaction_stratum_reports_unmeasurable(tmp_path, monkeypatch, capsys):
    reference = {
        "valid": torch.ones(1, 5, dtype=torch.bool),
        "action": torch.ones(1, 5, 4, 2),
        "observation": torch.zeros(1, 5, 4, 13),
        "next_observation": torch.zeros(1, 5, 4, 13),
        "next_agent_state": torch.zeros(1, 5, 4, 6),
        "next_package_state": torch.zeros(1, 5, 1, 6),
    }
    counterfactual = {key: value.clone() for key, value in reference.items()}
    counterfactual["anchor_id"] = torch.tensor([0])
    counterfactual["action"][:, :, 1, 0] *= -1
    torch.save(reference, tmp_path / "samples_correlated.pt")
    torch.save(counterfactual, tmp_path / "counterfactual_test.pt")
    args = ["evaluate", str(tmp_path / "runs"), "--data", str(tmp_path), "--full-state"]
    monkeypatch.setattr(sys, "argv", args)
    counterfactual_main()
    assert '"status": "no_active_anchors"' in capsys.readouterr().out
    output = tmp_path / "scores.json"
    monkeypatch.setattr(sys, "argv", args + ["--output", str(output)])
    stratified_main()
    assert "unmeasurable" in capsys.readouterr().out
    scores = json.loads(output.read_text())
    assert scores["status"] == "empty_stratum"
    assert scores["active"] == 0
    assert scores["total"] == 1
