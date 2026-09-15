# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the license in the LICENSE file in the root directory.
"""Controlled action coverage and real Transport branch replay regressions."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from benchmarl.environments import VmasTask
from examples.world_model.collect import (
    branch_rollouts,
    effect_summary,
    episode_split,
    leakage_checks,
    map_tensors,
    rollout_actions,
    sample_actions,
    tracked_entities,
)
from examples.world_model.dataset import OfflineSequences
from examples.world_model.evaluate import state_digest
from examples.world_model.snapshot_restore import restore_state, snapshot_state
from tensordict import TensorDict
from vmas.scenarios.transport import HeuristicPolicy


@pytest.fixture
def transport_env():
    envs = []

    def make(count=3, max_steps=6):
        task = VmasTask.TRANSPORT.get_from_yaml()
        task.config["max_steps"] = max_steps
        env = task.get_env_fun(count, True, 37, "cpu")()
        env.reset()
        envs.append(env)
        return task, env

    yield make
    for env in envs:
        env.close()


def test_correlated_actions_keep_marginals_and_exclude_joint_region():
    low = torch.tensor([[-2.0, 1.0]]).expand(4, -1)
    high = torch.tensor([[3.0, 5.0]]).expand(4, -1)
    rng = torch.random.get_rng_state()
    independent = sample_actions((20000, 4, 2), low, high, "independent", 81)
    correlated = sample_actions((20000, 4, 2), low, high, "correlated", 81)
    assert torch.equal(rng, torch.random.get_rng_state())
    normalized = [
        2 * (actions - low) / (high - low) - 1 for actions in (independent, correlated)
    ]
    torch.testing.assert_close(
        normalized[0].abs(), normalized[1].abs(), atol=1e-6, rtol=0
    )
    for actions, unit in zip((independent, correlated), normalized):
        assert ((actions >= low) & (actions <= high)).all()
        # Every individual coordinate retains both signs and interior support.
        assert unit.mean(0).abs().max() < 0.025
        for agent in range(4):
            for coordinate in range(2):
                assert (
                    torch.histc(
                        unit[:, agent, coordinate], bins=10, min=-1, max=1
                    ).min()
                    > 1700
                )
    opposed = normalized[0][:, 0, 0] * normalized[0][:, 1, 0] < 0
    assert 0.48 < opposed.float().mean() < 0.52
    assert (normalized[1] * normalized[1][:, :1] >= 0).all()
    assert torch.equal(
        correlated, sample_actions((20000, 4, 2), low, high, "correlated", 81)
    )


def test_related_anchors_inherit_episode_split_and_duplicates_cannot_leak(
    transport_env,
):
    _, env = transport_env(count=8)
    split = episode_split(8, 17)
    assert torch.equal(split, episode_split(8, 17))
    assert torch.bincount(split).tolist() == [6, 1, 1]
    episode_ids = torch.arange(8).repeat_interleave(2)
    anchors = {
        "episode_id": episode_ids,
        "split": split[episode_ids],
        "snapshot": map_tensors(snapshot_state(env), lambda x: x[episode_ids]),
    }
    initial = {"split": split, "snapshot": snapshot_state(env)}
    checks = leakage_checks(initial, anchors)
    assert checks["unique_anchor_states"] == 8
    with pytest.raises(ValueError, match="excluded evaluation bank"):
        leakage_checks(initial, anchors, excluded=initial["snapshot"])
    anchors["split"][0] = (anchors["split"][0] + 1) % 3
    with pytest.raises(ValueError, match="source episode split"):
        leakage_checks(initial, anchors)
    anchors["split"] = split[episode_ids]
    destination = (anchors["split"] != anchors["split"][0]).nonzero()[0, 0]
    anchors["snapshot"] = map_tensors(
        anchors["snapshot"],
        lambda x: x.clone().index_copy(0, destination.reshape(1), x[:1]),
    )
    with pytest.raises(ValueError, match="leaked across splits"):
        leakage_checks(initial, anchors)


def test_saved_transitions_replay_through_timeout_and_keep_terminal_reward(
    transport_env,
):
    _, env = transport_env(max_steps=4)
    package = env._env.scenario.packages[0]
    toward_goal = package.goal.state.pos - package.state.pos
    package.state.vel = 0.2 * toward_goal / toward_goal.norm(dim=-1, keepdim=True)
    env._env.steps[:] = torch.tensor([0.0, 2.0, 3.0])
    initial = snapshot_state(env)
    actions = sample_actions(
        (3, 5, 4, 2), torch.tensor(-1.0), torch.tensor(1.0), "independent", 4
    )
    data, anchors = rollout_actions(env, initial, actions, anchor_stride=1)
    replay, _ = rollout_actions(env, initial, actions)
    for key in data:
        assert torch.equal(data[key], replay[key]), key
    assert data["valid"].sum(1).tolist() == [4, 2, 1]
    assert not data["terminated"].any()
    assert torch.equal(data["done"], data["truncated"])
    assert sum(len(anchor["episode_id"]) for anchor in anchors) == 7
    assert [anchor["source_step"].tolist() for anchor in anchors] == [
        [0, 2, 3],
        [1, 3],
        [2],
        [3],
    ]

    # Check saved records against actual execution, independently of collector replay.
    restore_state(env, initial)
    for step in range(4):
        live = data["valid"][:, step]
        action = actions[:, step].clone()
        action[~live] = 0
        nxt = env.step(TensorDict({("agents", "action"): action}, [3]))["next"]
        assert torch.equal(data["reward"][live, step], nxt["agents", "reward"][live])
        assert torch.equal(
            data["next_observation"][live, step], nxt["agents", "observation"][live]
        )
    for episode, terminal_step in enumerate([3, 1, 0]):
        assert data["reward"][episode, terminal_step].abs().sum() > 0
        assert data["done"][episode, terminal_step]
    for key, value in data.items():
        assert not value[~data["valid"]].count_nonzero(), key


def test_real_goal_at_time_limit_is_termination_not_truncation(transport_env):
    _, env = transport_env(count=1, max_steps=1)
    scenario = env._env.scenario
    package = scenario.packages[0]
    package.state.pos[:] = torch.tensor([0.0, 0.0])
    package.state.vel[:] = torch.tensor([1.0, 0.0])
    package.goal.state.pos[:] = torch.tensor([0.3, 0.0])
    for agent, position in zip(
        env._env.world.agents, [[-0.8, -0.8], [-0.8, 0.8], [0.8, -0.8], [0.8, 0.8]]
    ):
        agent.state.pos[:] = torch.tensor(position)
    package.on_goal = env._env.world.is_overlapping(package, package.goal)
    package.global_shaping = (package.state.pos - package.goal.state.pos).norm(
        dim=-1
    ) * scenario.shaping_factor
    assert not scenario.done().any()
    data, _ = rollout_actions(env, snapshot_state(env), torch.zeros(1, 3, 4, 2))
    assert data["valid"].tolist() == [[True, False, False]]
    assert data["terminated"].tolist() == [[True, False, False]]
    assert not data["truncated"].any()
    assert data["done"].tolist() == [[True, False, False]]


def test_chunked_branches_preserve_source_action_order_and_live_environment(
    transport_env,
):
    task, live = transport_env(count=5)
    live._env.steps[:] = torch.arange(5)
    initial = snapshot_state(live)
    identity = state_digest(initial)
    anchors = {"snapshot": initial}
    spec = live.full_action_spec_unbatched["agents", "action"]
    independent = sample_actions((5, 4, 4, 2), spec.low, spec.high, "independent", 123)
    correlated = sample_actions((5, 4, 4, 2), spec.low, spec.high, "correlated", 123)
    batched = branch_rollouts(task, anchors, independent, batch_size=2, device="cpu")
    serial = branch_rollouts(task, anchors, independent, batch_size=1, device="cpu")
    matched = branch_rollouts(task, anchors, correlated, batch_size=2, device="cpu")
    assert state_digest(snapshot_state(live)) == identity
    assert batched["valid"].sum(1).tolist() == [4, 4, 4, 3, 2]
    for key in batched:
        torch.testing.assert_close(batched[key], serial[key], atol=1e-5, rtol=1e-5)
    assert torch.equal(
        batched["action"][batched["valid"]], independent[batched["valid"]]
    )
    assert torch.equal(batched["observation"][:, 0], matched["observation"][:, 0])
    observations = torch.stack(
        [live._env.scenario.observation(agent) for agent in live._env.world.agents], 1
    )
    assert torch.equal(batched["observation"][:, 0], observations)


def test_one_agent_intervention_measures_physical_effect_not_relative_observation(
    transport_env,
):
    task, env = transport_env(count=2, max_steps=30)
    scenario = env._env.scenario
    package = scenario.packages[0]
    package.state.pos.zero_()
    package.goal.state.pos[:] = torch.tensor([0.8, 0.0])
    for agent, position in zip(
        env._env.world.agents, [[0.108, 0.0], [-0.108, 0.0], [-0.7, 0.7], [0.7, 0.7]]
    ):
        agent.state.pos[:] = torch.tensor(position)
    # Only slot 0 lets the package transmit force to agent 0. Slot 1 is a control.
    env._env.world.agents[0].state.pos[1] = torch.tensor([0.6, 0.6])
    package.on_goal = env._env.world.is_overlapping(package, package.goal)
    package.global_shaping = (package.state.pos - package.goal.state.pos).norm(
        dim=-1
    ) * scenario.shaping_factor
    anchors = {"snapshot": snapshot_state(env)}
    original = torch.zeros(2, 20, 4, 2)
    original[:, :, :2, 0] = -1
    intervention = original.clone()
    intervention[:, :, 1, 0] = 1
    reference = branch_rollouts(task, anchors, original, 2, "cpu")
    counterfactual = branch_rollouts(task, anchors, intervention, 2, "cpu")
    assert torch.equal(
        reference["action"][:, :, [0, 2, 3]], counterfactual["action"][:, :, [0, 2, 3]]
    )
    assert torch.equal(
        reference["observation"][:, 0], counterfactual["observation"][:, 0]
    )
    effects = effect_summary(reference, counterfactual)
    assert effects["package"]["anchors_with_effect"] == 2
    assert effects["other_agent"]["anchors_with_effect"] == 1
    assert (
        reference["next_agent_state"][0, :, 0]
        - counterfactual["next_agent_state"][0, :, 0]
    ).abs().max() > 1e-5
    assert torch.equal(
        reference["next_agent_state"][1, :, 0],
        counterfactual["next_agent_state"][1, :, 0],
    )
    assert not torch.equal(
        reference["next_observation"][1, :, 0],
        counterfactual["next_observation"][1, :, 0],
    )


def test_shipped_policy_uses_current_observations_and_replays(transport_env):
    _, env = transport_env(count=3, max_steps=6)
    heuristic = HeuristicPolicy(continuous_action=True)

    def policy(observation):
        batch, agents, features = observation.shape
        return heuristic.compute_action(
            observation.reshape(batch * agents, features), 1.0
        ).reshape(batch, agents, 2)

    initial = snapshot_state(env)
    data, _ = rollout_actions(env, initial, torch.zeros(3, 8, 4, 2), policy=policy)
    assert not torch.equal(data["action"][:, 0], data["action"][:, 1])
    for step in range(6):
        assert torch.equal(
            data["action"][:, step], policy(data["observation"][:, step])
        )
    replay, _ = rollout_actions(env, initial, data["action"])
    for key in data:
        assert torch.equal(data[key], replay[key]), key


def test_complete_hydra_collection_reloads_for_m4(tmp_path):
    subprocess.run(
        [
            sys.executable,
            "-m",
            "examples.world_model.collect",
            "--config-name",
            "sweep/offline_data_smoke",
            "dataset.include_heuristic=true",
            f"hydra.run.dir={tmp_path}",
        ],
        cwd=Path(__file__).resolve().parents[1],
        env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
        check=True,
        capture_output=True,
        text=True,
    )
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    checks = json.loads((tmp_path / "leakage_checks.json").read_text())
    assert manifest["source_regimes"] == ["independent", "correlated", "heuristic"]
    assert (
        checks["replay_bit_exact"]
        and checks["counterfactual_reference_replay_bit_exact"]
    )
    for regime in ("independent", "correlated"):
        assert len(OfflineSequences(tmp_path, regime, "train", action_block=5)) == 54


def test_tracked_entities_keeps_a_body_that_only_rotates():
    """Wheel's line is pinned at the origin, so a movable-only filter drops it.

    It is the one body the task is about; selecting the static `center` instead
    would record constant columns and read as "no interaction" for any diagnostic
    built on them.
    """
    task = VmasTask.WHEEL.get_from_yaml()
    env = task.get_env_fun(2, True, 0, "cpu")()
    try:
        env.reset()
        assert [entity.name for entity in tracked_entities(env)] == ["line"]
    finally:
        env.close()


def test_tracked_entities_is_unchanged_where_a_movable_body_exists():
    """Admitting rotatable bodies must not re-select what the existing banks used."""
    for task_enum, expected in (
        (VmasTask.TRANSPORT, ["package 0"]),
        (VmasTask.BUZZ_WIRE, ["ball", "joint agent_0 ball", "joint agent_1 ball"]),
    ):
        env = task_enum.get_from_yaml().get_env_fun(2, True, 0, "cpu")()
        try:
            env.reset()
            assert [entity.name for entity in tracked_entities(env)] == expected
        finally:
            env.close()


def test_effect_summary_sees_a_purely_rotational_effect():
    """Position and velocity are identically zero for a pinned rotating body.

    The historical keys must stay zero there -- they are measuring what they say
    -- while the full-state keys must report the rotation the intervention caused.
    """
    shape = (3, 4, 1, 6)
    reference = {
        "valid": torch.ones(3, 4, dtype=torch.bool),
        "action": torch.zeros(3, 4, 2, 2),
        "next_agent_state": torch.zeros(shape),
        "next_package_state": torch.zeros(shape),
    }
    counterfactual = {key: value.clone() for key, value in reference.items()}
    # Rotation and angular velocity only; position and velocity stay identical.
    counterfactual["next_package_state"][..., 4] = 0.5

    effects = effect_summary(reference, counterfactual)
    assert effects["package"]["anchors_with_effect"] == 0
    assert effects["package"]["mean_position_velocity_l2"] == 0.0
    assert effects["package_full_state"]["anchors_with_effect"] == 3
    assert effects["package_full_state"]["mean_full_state_l2"] > 0.0
