# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the license found in the LICENSE file in the root directory.
"""Paired-snapshot sufficiency audit for the Buzz Wire ``full32`` state.

Each intervention duplicates reachable simulator snapshots, changes one field
that ``full32`` does not encode in the second copy, applies identical actions,
and compares the next physical state, reward, collision, and termination. The
link-velocity intervention is a represented-state positive control. The timeout
intervention tests the episode clock separately from physical dynamics.
"""

import argparse
import hashlib
import json
import subprocess
from importlib.metadata import version
from pathlib import Path

import torch
from tensordict import TensorDict

from benchmarl.environments import VmasTask
from examples.world_model.snapshot_restore import restore_state, snapshot_state
from examples.world_model.structured_surrogate import live_structured_state


def tree_map(value, function):
    if isinstance(value, dict):
        return {key: tree_map(item, function) for key, item in value.items()}
    return function(value)


def paired_snapshot(snapshot):
    return tree_map(snapshot, lambda value: value.repeat_interleave(2, dim=0))


def pair_metrics(value, tolerance=1e-7):
    first, second = value[0::2], value[1::2]
    difference = (first - second).abs()
    flat = difference.reshape(difference.shape[0], -1)
    maximum = flat.max(dim=1).values
    return {
        "mean_absolute_difference": float(difference.float().mean()),
        "maximum_absolute_difference": float(difference.max()),
        "different_pair_rate": float((maximum > tolerance).float().mean()),
    }


def bool_pair_metrics(value):
    first, second = value[0::2], value[1::2]
    different = first != second
    return {
        "different_pair_rate": float(different.float().mean()),
        "different_pairs": int(different.sum()),
    }


def apply_intervention(snapshot, name, max_steps):
    odd = slice(1, None, 2)
    if name.startswith("agent_"):
        _prefix, agent, field = name.split("_")
        key = "_force" if field == "force" else "_torque"
        snapshot["entities"][f"agent_{agent}"]["state"][key][odd] += 0.75
    elif name == "pos_rew_cache":
        snapshot["scenario"]["pos_rew"][odd] += 5.0
    elif name == "collision_rew_cache":
        snapshot["scenario"]["collision_rew"][odd] -= 5.0
    elif name == "collided_cache":
        snapshot["scenario"]["collided"][odd] = True
    elif name == "pos_shaping_cache":
        snapshot["scenario"]["pos_shaping"][odd] += 0.5
    elif name == "step_counter_local":
        snapshot["steps"][odd] += 1
    elif name == "step_counter_timeout":
        snapshot["steps"][odd] = max_steps - 1
    elif name == "link_velocity_positive_control":
        link = snapshot["entities"]["joint agent_0 ball"]["state"]["_vel"]
        link[odd, 0] += 0.5
    else:
        raise ValueError(f"Unknown intervention: {name}")


@torch.no_grad()
def evaluate_intervention(task, source, actions, name, device, max_steps):
    roots = actions.shape[0]
    snapshot = paired_snapshot(source)
    apply_intervention(snapshot, name, max_steps)
    env = task.get_env_fun(roots * 2, True, 0, device)()
    env.reset()
    try:
        restore_state(env, tree_map(snapshot, lambda value: value.to(device)))
        before = live_structured_state(env, "full32")
        td = TensorDict({}, batch_size=[roots * 2], device=device)
        td.set(("agents", "action"), actions.repeat_interleave(2, dim=0))
        td = env.step(td)["next"]
        after = live_structured_state(env, "full32")
        reward = td["agents", "reward"].sum(dim=(1, 2))
        done = td["done"].squeeze(-1)
        collided = env._env.scenario.collided
    finally:
        env.close()
    return {
        "before_full32": pair_metrics(before.cpu()),
        "next_full32": pair_metrics(after.cpu()),
        "team_reward": pair_metrics(reward.cpu()),
        "collision": bool_pair_metrics(collided.cpu()),
        "done": bool_pair_metrics(done.cpu()),
    }


@torch.no_grad()
def evaluate(roots=64, warmup_steps=3, seed=9700, device="cpu"):
    task = VmasTask.BUZZ_WIRE.get_from_yaml()
    task.config["max_steps"] = 100
    env = task.get_env_fun(roots, True, seed, device)()
    env.set_seed(seed)
    td = env.reset()
    generator = torch.Generator(device=device).manual_seed(seed + 1)
    try:
        for _ in range(warmup_steps):
            action = torch.rand(
                roots, 2, 2, device=device, generator=generator
            ) * 2 - 1
            td.set(("agents", "action"), action)
            td = env.step(td)["next"]
        alive = ~td["done"].squeeze(-1)
        indices = alive.nonzero(as_tuple=True)[0]
        if not indices.numel():
            raise RuntimeError("No live roots remain after warmup")
        source = tree_map(
            snapshot_state(env), lambda value: value.index_select(0, indices)
        )
        full = live_structured_state(env, "full32").index_select(0, indices)
        distance = torch.linalg.vector_norm(
            full[:, 12:14] - full[:, -2:], dim=-1
        )
        shaping_residual = (
            source["scenario"]["pos_shaping"].cpu() - distance.cpu()
        ).abs()
        reachable_omitted = {
            "agent_0_force_max_abs": float(
                source["entities"]["agent_0"]["state"]["_force"].abs().max()
            ),
            "agent_1_force_max_abs": float(
                source["entities"]["agent_1"]["state"]["_force"].abs().max()
            ),
            "agent_0_torque_max_abs": float(
                source["entities"]["agent_0"]["state"]["_torque"].abs().max()
            ),
            "agent_1_torque_max_abs": float(
                source["entities"]["agent_1"]["state"]["_torque"].abs().max()
            ),
            "pos_shaping_cache_max_residual": float(shaping_residual.max()),
        }
    finally:
        env.close()

    surviving_roots = indices.numel()
    actions = torch.rand(
        surviving_roots, 2, 2, device=device, generator=generator
    ) * 2 - 1
    interventions = (
        "agent_0_force",
        "agent_0_torque",
        "agent_1_force",
        "agent_1_torque",
        "pos_rew_cache",
        "collision_rew_cache",
        "collided_cache",
        "pos_shaping_cache",
        "step_counter_local",
        "step_counter_timeout",
        "link_velocity_positive_control",
    )
    results = {
        name: evaluate_intervention(
            task, source, actions, name, device, task.config["max_steps"]
        )
        for name in interventions
    }
    physical_omissions = (
        "agent_0_force",
        "agent_0_torque",
        "agent_1_force",
        "agent_1_torque",
        "pos_rew_cache",
        "collision_rew_cache",
        "collided_cache",
        "step_counter_local",
    )
    arbitrary_snapshot_physics_pass = all(
        results[name]["next_full32"]["different_pair_rate"] == 0
        for name in physical_omissions
    )
    reachable_physics_pass = (
        results["agent_0_force"]["next_full32"]["different_pair_rate"] == 0
        and results["agent_1_force"]["next_full32"]["different_pair_rate"] == 0
        and reachable_omitted["agent_0_torque_max_abs"] == 0
        and reachable_omitted["agent_1_torque_max_abs"] == 0
        and results["pos_rew_cache"]["next_full32"]["different_pair_rate"] == 0
        and results["collision_rew_cache"]["next_full32"][
            "different_pair_rate"
        ]
        == 0
        and results["collided_cache"]["next_full32"]["different_pair_rate"]
        == 0
        and results["step_counter_local"]["next_full32"][
            "different_pair_rate"
        ]
        == 0
    )
    source_path = Path(__file__)
    return {
        "question": (
            "Is full32 sufficient for one-step physical and modeled-outcome "
            "prediction under reachable Buzz Wire states?"
        ),
        "config": {
            "requested_roots": roots,
            "surviving_roots": surviving_roots,
            "warmup_steps": warmup_steps,
            "seed": seed,
            "device": device,
            "max_steps": task.config["max_steps"],
            "continuous_action_range": [-1.0, 1.0],
            "force_intervention": 0.75,
            "torque_intervention": 0.75,
            "link_velocity_intervention": 0.5,
            "pos_shaping_intervention": 0.5,
        },
        "provenance": {
            "commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            "git_status": subprocess.check_output(
                ["git", "status", "--short"], text=True
            ).splitlines(),
            "versions": {
                "torch": version("torch"),
                "vmas": version("vmas"),
            },
            "source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        },
        "reachable_omitted_state": reachable_omitted,
        "interventions": results,
        "verdict": {
            "arbitrary_snapshot_physics_sufficient_for_tested_omissions": (
                arbitrary_snapshot_physics_pass
            ),
            "reachable_physics_sufficient_for_tested_omissions": (
                reachable_physics_pass
            ),
            "off_manifold_nonzero_torque_changes_physics": (
                results["agent_0_torque"]["next_full32"][
                    "different_pair_rate"
                ]
                > 0
            ),
            "pos_shaping_is_derived_but_reward_sensitive_if_corrupted": (
                results["pos_shaping_cache"]["team_reward"][
                    "different_pair_rate"
                ]
                > 0
            ),
            "episode_clock_required_for_timeout": (
                results["step_counter_timeout"]["done"]["different_pair_rate"]
                > 0
            ),
            "represented_link_control_changes_full32": (
                results["link_velocity_positive_control"]["before_full32"][
                    "different_pair_rate"
                ]
                > 0
            ),
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--roots", type=int, default=64)
    parser.add_argument("--warmup-steps", type=int, default=3)
    parser.add_argument("--seed", type=int, default=9700)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    result = evaluate(args.roots, args.warmup_steps, args.seed, args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    printed = result["verdict"] if args.quiet else result
    print(json.dumps(printed, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
