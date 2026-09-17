#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#
"""M2 (docs/paper/experiment_plan.md): validate that a VMAS scenario's full
state can be snapshotted and restored, and that replaying the same actions
from a restored state reproduces the original trajectory exactly.

This is the primitive M2's oracle CEM-MPC and M3's counterfactual dataset
branches both need: roll a live environment forward, snapshot a state,
try a hypothetical action sequence, then return to that exact state to try
another -- without disturbing whatever "live" rollout the environment is
otherwise used for.

Run directly: `python examples/world_model/snapshot_restore.py`
"""

import torch

from benchmarl.environments import VmasTask


def _batch_tensor_attrs(obj, batch_dim: int):
    """Every tensor attribute on `obj` whose leading dim is the batch size."""
    return {
        name: value
        for name, value in vars(obj).items()
        if isinstance(value, torch.Tensor)
        and value.dim() > 0
        and value.shape[0] == batch_dim
    }


def snapshot_state(env):
    """Clone the full scenario state: core physics (pos/vel/rot/ang_vel) for
    every entity, plus each entity's and the scenario's own bookkeeping
    tensors (e.g. potential-based shaping, collision flags).

    Attributes are discovered generically (any batch-shaped tensor found on
    the entity, its state, or the scenario object) rather than hand-listed
    per task, so this works unchanged for any of the six scenarios in
    docs/paper/vmas_task_survey.md whose extra state is plain tensor
    attributes. A scenario driven through a stateful controller or an
    action-delay queue (e.g. give_way's `VelocityController`/`input_queue`)
    would need that handled separately -- not needed for Buzz Wire.
    """
    world = env._env.world
    scenario = env._env.scenario
    batch_dim = world.batch_dim

    entities = {}
    for entity in world.entities:
        entities[entity.name] = {
            "state": {
                k: v.clone()
                for k, v in _batch_tensor_attrs(entity.state, batch_dim).items()
            },
            "entity": {
                k: v.clone() for k, v in _batch_tensor_attrs(entity, batch_dim).items()
            },
        }
    scenario_attrs = {
        k: v.clone() for k, v in _batch_tensor_attrs(scenario, batch_dim).items()
    }
    return {
        "entities": entities,
        "scenario": scenario_attrs,
        "steps": env._env.steps.clone(),
    }


def restore_state(env, snapshot) -> None:
    """Write a snapshot taken by `snapshot_state` back into `env`, in place."""
    world = env._env.world
    if snapshot["steps"].shape != env._env.steps.shape:
        raise ValueError("Snapshot batch must match the environment batch")
    scenario = env._env.scenario
    entities_by_name = {entity.name: entity for entity in world.entities}

    for name, entity_snapshot in snapshot["entities"].items():
        entity = entities_by_name[name]
        for attr, value in entity_snapshot["state"].items():
            setattr(entity.state, attr, value.clone())
        for attr, value in entity_snapshot["entity"].items():
            setattr(entity, attr, value.clone())

    for attr, value in snapshot["scenario"].items():
        setattr(scenario, attr, value.clone())
    env._env.steps = snapshot["steps"].clone()


def broadcast_state(env, snapshot, env_index: int = 0, *, source_indices=None) -> None:
    """Copy source slots into scratch slots, including their episode clocks.

    ``source_indices`` has one source index per destination slot. For B states
    and K candidates use ``arange(B).repeat_interleave(K)``. Without indices,
    retain the single-source branching convention.
    """
    world = env._env.world
    scenario = env._env.scenario
    k = world.batch_dim
    entities_by_name = {entity.name: entity for entity in world.entities}
    if source_indices is None:
        source_indices = torch.full(
            (k,), env_index, dtype=torch.long, device=snapshot["steps"].device
        )
    if source_indices.shape != (k,):
        raise ValueError("One source index is required per scratch environment")

    def spread(value):
        return value.index_select(0, source_indices)

    for name, entity_snapshot in snapshot["entities"].items():
        entity = entities_by_name[name]
        for attr, value in entity_snapshot["state"].items():
            setattr(entity.state, attr, spread(value))
        for attr, value in entity_snapshot["entity"].items():
            setattr(entity, attr, spread(value))

    for attr, value in snapshot["scenario"].items():
        setattr(scenario, attr, spread(value))
    env._env.steps = spread(snapshot["steps"])


def tracked_entities(env):
    """Landmarks whose physics we record as an interaction diagnostic.

    Transport exposes ``scenario.packages`` (the shared object agents push);
    tasks without one fall back to the world's landmarks, so the stored schema
    is identical across tasks. These states are diagnostics only and are never
    model inputs, so the stored key keeps its ``package_state`` name rather than
    breaking the schema that the existing Transport bank was written with.
    """
    scenario = env._env.scenario
    packages = getattr(scenario, "packages", None)
    if packages:
        return packages
    # Prefer the bodies that can actually move -- Buzz Wire's ball is the
    # reward-relevant one, while its walls and floors are static and would only
    # pad the diagnostic with constant columns. Rotatable counts as moving:
    # Wheel's line is pinned at the origin and *only* rotates, so a movable-only
    # filter would drop the single body the task is about and silently record a
    # constant. World order is preserved, so tasks that already have a movable
    # body select exactly what they selected before.
    dynamic = [
        e
        for e in env._env.world.landmarks
        if getattr(e, "movable", False) or getattr(e, "rotatable", False)
    ]
    return dynamic or env._env.world.landmarks


def physical_state(entities):
    return torch.stack(
        [
            torch.cat([e.state.pos, e.state.vel, e.state.rot, e.state.ang_vel], -1)
            for e in entities
        ],
        dim=1,
    )


def agent_observations(env):
    """(B, N, obs_dim) -- what the agents themselves see at the current state.

    Lives here rather than in ``mpc`` because ``oracle_dynamics`` needs it too,
    and ``mpc`` imports ``oracle_dynamics``.
    """
    return torch.stack(
        [env._env.scenario.observation(agent) for agent in env._env.world.agents], dim=1
    )


def _rollout(env, td, action_sequence):
    """Step `env` through a fixed action sequence, recording the trajectory."""
    observations, rewards, dones = [], [], []
    for action in action_sequence:
        td.set(("agents", "action"), action.clone())
        td = env.step(td)
        observations.append(td["next", "agents", "observation"].clone())
        rewards.append(td["next", "agents", "reward"].clone())
        dones.append(td["next", "done"].clone())
        td = td["next"]
    return observations, rewards, dones


def _assert_same_trajectory(traj_a, traj_b, label: str) -> None:
    obs_a, rew_a, done_a = traj_a
    obs_b, rew_b, done_b = traj_b
    for h, (oa, ob, ra, rb, da, db) in enumerate(
        zip(obs_a, obs_b, rew_a, rew_b, done_a, done_b)
    ):
        assert torch.equal(oa, ob), f"{label}: observation mismatch at step {h}"
        assert torch.equal(ra, rb), f"{label}: reward mismatch at step {h}"
        assert torch.equal(da, db), f"{label}: done mismatch at step {h}"


def _trajectories_differ(traj_a, traj_b) -> bool:
    obs_a, _, _ = traj_a
    obs_b, _, _ = traj_b
    return any(not torch.equal(oa, ob) for oa, ob in zip(obs_a, obs_b))


if __name__ == "__main__":
    torch.manual_seed(0)
    num_envs = 4
    warmup_steps = 10
    horizon = 8

    task = VmasTask.BUZZ_WIRE.get_from_yaml()
    env = task.get_env_fun(
        num_envs=num_envs, continuous_actions=True, seed=0, device="cpu"
    )()

    td = env.reset()
    for _ in range(warmup_steps):
        td = env.rand_action(td)
        td = env.step(td)
        td = td["next"]

    snapshot = snapshot_state(env)

    action_sequence_a = [
        env.rand_action(td.clone())["agents", "action"] for _ in range(horizon)
    ]
    trajectory_a = _rollout(env, td.clone(), action_sequence_a)

    restore_state(env, snapshot)
    trajectory_b = _rollout(env, td.clone(), action_sequence_a)
    _assert_same_trajectory(trajectory_a, trajectory_b, "same actions after restore")
    print(
        f"PASS: identical {horizon}-step trajectory (obs/reward/done) "
        f"replaying the same {num_envs}-env action sequence after restore."
    )

    restore_state(env, snapshot)
    action_sequence_c = [
        env.rand_action(td.clone())["agents", "action"] for _ in range(horizon)
    ]
    trajectory_c = _rollout(env, td.clone(), action_sequence_c)
    assert _trajectories_differ(
        trajectory_a, trajectory_c
    ), "sanity check failed: different action sequences produced identical observations"
    print("PASS: a different action sequence from the same restored state diverges.")
