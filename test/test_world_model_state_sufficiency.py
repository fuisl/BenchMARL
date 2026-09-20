# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the license found in the LICENSE file in the root directory.

from examples.world_model.state_sufficiency import evaluate


def test_paired_snapshot_audit_separates_physics_cache_and_timeout():
    result = evaluate(roots=4, warmup_steps=0, seed=99, device="cpu")
    verdict = result["verdict"]
    omitted = result["reachable_omitted_state"]
    assert verdict["arbitrary_snapshot_physics_sufficient_for_tested_omissions"] is False
    assert verdict["reachable_physics_sufficient_for_tested_omissions"] is True
    assert verdict["off_manifold_nonzero_torque_changes_physics"] is True
    assert verdict["pos_shaping_is_derived_but_reward_sensitive_if_corrupted"] is True
    assert verdict["episode_clock_required_for_timeout"] is True
    assert verdict["represented_link_control_changes_full32"] is True
    assert omitted["agent_0_torque_max_abs"] == 0
    assert omitted["agent_1_torque_max_abs"] == 0
    assert omitted["pos_shaping_cache_max_residual"] == 0
