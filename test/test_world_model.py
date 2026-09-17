# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the license in the LICENSE file in the root directory.
"""Oracle/MPC regression checks: branching, episode boundaries and action order."""

from types import SimpleNamespace

import pytest
import torch

from benchmarl.environments import VmasTask
from examples.world_model import mpc
from examples.world_model.cem import cem_plan, CEMConfig
from examples.world_model.evaluate import state_digest
from examples.world_model.metrics import success_interval, summarize
from examples.world_model.oracle_dynamics import (
    NegativeTaskReward,
    oracle_plan_costs,
    oracle_rollout,
)
from examples.world_model.snapshot_restore import (
    broadcast_state,
    restore_state,
    snapshot_state,
)
from tensordict import TensorDict


@pytest.fixture
def make_env():
    envs = []

    def make(count=3, max_steps=100):
        task = VmasTask.BUZZ_WIRE.get_from_yaml()
        task.config["max_steps"] = max_steps
        env = task.get_env_fun(count, True, 0, "cpu")()
        env.reset()
        envs.append(env)
        return env

    yield make
    for env in envs:
        env.close()


def assert_snapshot_equal(a, b):
    assert a.keys() == b.keys()
    for key in a:
        if isinstance(a[key], dict):
            assert_snapshot_equal(a[key], b[key])
        else:
            assert torch.equal(a[key], b[key]), key


def test_snapshot_restores_timeout(make_env):
    env = make_env(max_steps=2)
    td = TensorDict({("agents", "action"): torch.zeros(3, 2, 2)}, [3])
    snapshot = snapshot_state(env)
    first = env.step(td)["next"]
    restore_state(env, snapshot)
    replay = env.step(td)["next"]
    assert not replay["done"].any()
    assert torch.equal(first["done"], replay["done"])
    assert torch.equal(first["agents", "reward"], replay["agents", "reward"])
    assert env.step(td)["next", "done"].all()


def test_batched_oracle_matches_serial_and_preserves_live(make_env):
    batch_size, samples, horizon = 3, 4, 4
    env = make_env(batch_size)
    env._env.steps[:] = torch.tensor([1, 98, 99])
    snapshot = snapshot_state(env)
    candidates = (
        torch.rand(
            batch_size, samples, horizon, 4, generator=torch.Generator().manual_seed(8)
        )
        * 2
        - 1
    )
    scratch = make_env(batch_size * samples)
    rng_before = torch.random.get_rng_state()
    rollout = oracle_rollout(scratch, snapshot, candidates)
    costs = NegativeTaskReward()(rollout)
    assert torch.equal(rng_before, torch.random.get_rng_state())
    assert_snapshot_equal(snapshot, snapshot_state(env))
    assert rollout["live"][:, 0].sum(-1).tolist() == [4, 2, 1]
    assert costs.std() > 0
    assert torch.equal(costs, oracle_plan_costs(scratch, snapshot, candidates))
    single = make_env(1)
    serial_scratch = make_env(samples)
    for index in range(batch_size):
        broadcast_state(single, snapshot, env_index=index)
        serial = oracle_plan_costs(
            serial_scratch, snapshot_state(single), candidates[index : index + 1]
        )
        # Different VMAS batch widths can change float32 reduction rounding.
        torch.testing.assert_close(costs[index], serial[0], atol=1e-5, rtol=1e-5)


def test_terminal_reward_and_block_execution_match_oracle(make_env):
    env = make_env(2, max_steps=3)
    env._env.steps[1] = 1
    snapshot = snapshot_state(env)
    blocked = torch.rand(2, 1, 2, 8, generator=torch.Generator().manual_seed(5)) * 0.1
    actions = mpc.unpack_actions(blocked, 2)
    rollout = oracle_rollout(make_env(2, max_steps=3), snapshot, actions)
    assert rollout["live"][:, 0].sum(-1).tolist() == [3, 2]
    expected = NegativeTaskReward()(rollout)[:, 0]
    stats = mpc.EpisodeStats(env, mpc.buzz_wire_outcome)
    td = TensorDict({}, [2])
    for action in actions[:, 0].unbind(1):
        td.set(("agents", "action"), action.reshape(2, 2, 2))
        td = env.step(td)["next"]
        stats.update(env, td)
    torch.testing.assert_close(-stats.team_return, expected)
    assert stats.length.tolist() == [3, 2]
    assert stats.timeout.all()
    # The terminal reward is included, and post-terminal NaNs cannot leak in.
    reward = torch.tensor([[[1.0, 2.0, float("nan")]]])
    live = torch.tensor([[[True, True, False]]])
    assert NegativeTaskReward()({"reward": reward, "live": live}).item() == -3


def test_first_outcome_latches_with_collision_precedence(make_env):
    env = make_env(3)
    stats = mpc.EpisodeStats(env, mpc.buzz_wire_outcome)
    scenario = env._env.scenario
    scenario.ball.state.pos[:2] = scenario.goal.state.pos[:2]
    scenario.collided[1] = True
    env._env.steps[2] = 100
    td = TensorDict(
        {
            ("agents", "reward"): torch.ones(3, 2, 1),
            "done": torch.ones(3, 1, dtype=torch.bool),
        },
        [3],
    )
    stats.update(env, td)
    assert stats.success.tolist() == [True, False, False]
    assert stats.collision.tolist() == [False, True, False]
    assert stats.timeout.tolist() == [False, False, True]
    scenario.collided[:] = False
    scenario.ball.state.pos[:] = scenario.goal.state.pos
    stats.update(env, td)
    assert stats.success.tolist() == [True, False, False]
    assert stats.team_return.tolist() == [2, 2, 2]
    assert stats.length.tolist() == [1, 1, 1]


def test_loop_order_warm_start_and_mixed_endings(make_env, monkeypatch, tmp_path):
    env = make_env(2, max_steps=5)
    env._env.steps[1] = 3
    initial = snapshot_state(env)
    plan = (
        torch.arange(16, dtype=torch.float32).reshape(1, 2, 8).expand(2, -1, -1) / 100
    )
    starts = []

    def planner(cost_fn, **kwargs):
        starts.append(kwargs["init_mean"])
        return SimpleNamespace(
            plan=plan,
            candidates=plan[:, None],
            costs=torch.zeros(2, 1),
            elite_idx=torch.zeros(2, 1, dtype=torch.long),
            elite_cost_history=torch.zeros(1, 2),
            best_cost_history=torch.zeros(1, 2),
        )

    monkeypatch.setattr(mpc, "cem_plan", planner)
    rows, timing, _terminal = mpc.evaluate_policy(
        env,
        initial,
        policy="mpc",
        generator=torch.Generator().manual_seed(0),
        cem_config=CEMConfig(horizon=2, num_samples=1, num_elites=1, num_iters=1),
        mpc_config=mpc.MPCConfig(receding_horizon=1, action_block=2),
        scratch_env=make_env(2, max_steps=5),
        diagnostics_path=tmp_path,
        outcome_fn=mpc.buzz_wire_outcome,
    )
    assert len(starts) == 3 and starts[0] is None
    torch.testing.assert_close(starts[1][:, 0], plan[:, 1])
    assert starts[1][:, 1].count_nonzero() == 0
    trajectory = torch.load(tmp_path / "mpc_trajectory.pt", weights_only=True)
    actual = torch.stack([step["action"][0].flatten() for step in trajectory])
    primitive = torch.arange(8, dtype=torch.float32).reshape(2, 4) / 100
    torch.testing.assert_close(actual, primitive.repeat(3, 1)[:5])
    assert [row["length"] for row in rows] == [5, 2]
    assert all(step["action"][1].count_nonzero() == 0 for step in trajectory[2:])
    assert torch.equal(mpc.shift_plan(plan, 2), torch.zeros_like(plan))


def test_cem_batch_reproducibility_and_validation():
    config = CEMConfig(horizon=2, num_samples=100, num_elites=10, num_iters=30)
    target = torch.tensor([-0.5, 0.0, 0.5]).reshape(3, 1, 1).expand(3, 2, 3)

    def run():
        return cem_plan(
            lambda actions: (actions - target[:, None]).square().sum((-2, -1)),
            action_dim=3,
            action_low=-1,
            action_high=1,
            config=config,
            batch_size=3,
            generator=torch.Generator().manual_seed(4),
        )

    result = run()
    torch.testing.assert_close(result.plan, target, atol=1e-3, rtol=0)
    assert torch.equal(result.plan, run().plan)
    assert (result.best_cost_history.diff(dim=0) <= 0).all()
    traced = cem_plan(
        lambda actions: (actions - target[:, None]).square().sum((-2, -1)),
        action_dim=3,
        action_low=-1,
        action_high=1,
        config=CEMConfig(horizon=2, num_samples=20, num_elites=5, num_iters=4),
        batch_size=3,
        generator=torch.Generator().manual_seed(4),
        record_candidates=True,
    )
    assert traced.candidate_history.shape == (4, 3, 20, 2, 3)
    with pytest.raises(ValueError, match="non-finite"):
        cem_plan(
            lambda c: torch.full(c.shape[:2], float("nan")),
            action_dim=3,
            action_low=-1,
            action_high=1,
            config=config,
        )
    with pytest.raises(ValueError):
        CEMConfig(num_iters=0)
    with pytest.raises(ValueError):
        CEMConfig(num_elites=301)
    with pytest.raises(ValueError):
        mpc.MPCConfig(receding_horizon=5).validate(1)


def test_metrics_pairing_and_zero_success():
    rows = [
        {
            "policy": policy,
            "episode": i,
            "return": ret + i,
            "team_return": 2 * (ret + i),
            "success": False,
            "collision": False,
            "timeout": True,
        }
        for policy, ret in (("mpc", 10), ("random", 0))
        for i in range(20)
    ]
    result = summarize(rows)
    assert result["paired_return_difference"]["ci95"] == [10, 10]
    assert result["m2_return_gate"]
    assert result["objective_status"].startswith("unvalidated")
    interval = success_interval([False] * 20)
    assert interval["ci95"][0] == 0
    assert 0 < interval["ci95"][1] < 1
    assert result == summarize(rows)


def test_state_bank_identity_and_reload(make_env, tmp_path):
    env = make_env(3)
    bank = {
        "snapshot": snapshot_state(env),
        "state_seed": 0,
        "task": {"max_steps": 100},
    }
    identity = state_digest(bank)
    torch.save(bank, tmp_path / "states.pt")
    loaded = torch.load(tmp_path / "states.pt", weights_only=True)
    assert state_digest(loaded) == identity
    loaded["snapshot"]["steps"][0] += 1
    assert state_digest(loaded) != identity


@pytest.mark.parametrize("cause", ["goal", "collision"])
def test_real_termination_inside_block_matches_oracle(make_env, cause):
    env = make_env(2)
    scenario = env._env.scenario
    for body in env._env.world.agents + [scenario.ball]:
        body.state.vel[:, 0] = 1.0
    td = TensorDict({("agents", "action"): torch.ones(2, 2, 2)}, [2])
    if cause == "goal":
        # Place a non-colliding goal at the next true ball position. The source
        # remains nonterminal; the first executed action reaches the goal.
        before = snapshot_state(env)
        env.step(td)
        next_position = scenario.ball.state.pos.clone()
        restore_state(env, before)
        scenario.goal.state.pos = next_position
        scenario.pos_shaping = (
            torch.linalg.vector_norm(scenario.ball.state.pos - next_position, dim=-1)
            * scenario.pos_shaping_factor
        )
    initial = snapshot_state(env)
    actions = mpc.unpack_actions(torch.ones(2, 1, 1, 20), action_block=5)
    rollout = oracle_rollout(make_env(2), initial, actions)
    expected_length = 1 if cause == "goal" else 2
    assert rollout["live"][:, 0].sum(-1).tolist() == [expected_length] * 2
    stats = mpc.EpisodeStats(env, mpc.buzz_wire_outcome)
    for action in actions[:, 0].unbind(1):
        td.set(("agents", "action"), action.reshape(2, 2, 2))
        td = env.step(td)["next"]
        stats.update(env, td)
    torch.testing.assert_close(-stats.team_return, NegativeTaskReward()(rollout)[:, 0])
    assert stats.length.tolist() == [expected_length] * 2
    assert (stats.success if cause == "goal" else stats.collision).all()


@pytest.mark.parametrize("n_packages", [1, 2])
def test_transport_oracle_replay_and_outcomes(n_packages):
    task = VmasTask.TRANSPORT.get_from_yaml()
    task.config.update(max_steps=6, n_packages=n_packages)
    env = task.get_env_fun(3, True, 0, "cpu")()
    scratch = task.get_env_fun(12, True, 0, "cpu")()
    serial = task.get_env_fun(4, True, 0, "cpu")()
    single = task.get_env_fun(1, True, 0, "cpu")()
    try:
        for item in (env, scratch, serial, single):
            item.reset()
        # Moving packages exercise potential-based rewards even without contact.
        for package in env._env.scenario.packages:
            package.state.vel[:, 0] = 0.1
        td = TensorDict({("agents", "action"): torch.zeros(3, 4, 2)}, [3])
        env.step(td)
        initial = snapshot_state(env)
        candidates = (
            torch.rand(3, 4, 4, 8, generator=torch.Generator().manual_seed(7)) * 2 - 1
        )
        costs = oracle_plan_costs(scratch, initial, candidates)
        assert costs.abs().max() > 0
        assert torch.equal(costs, oracle_plan_costs(scratch, initial, candidates))
        assert_snapshot_equal(initial, snapshot_state(env))
        for index in range(3):
            broadcast_state(single, initial, env_index=index)
            score = oracle_plan_costs(
                serial, snapshot_state(single), candidates[index : index + 1]
            )
            torch.testing.assert_close(score[0], costs[index], atol=1e-4, rtol=1e-5)
        stats = mpc.EpisodeStats(env, mpc.transport_outcome)
        for action in candidates[:, 0].unbind(1):
            td.set(("agents", "action"), action.reshape(3, 4, 2))
            td = env.step(td)["next"]
            stats.update(env, td)
        torch.testing.assert_close(
            -stats.team_return, costs[:, 0], atol=1e-4, rtol=1e-5
        )
        # Goal signals are per-package: one package alone is insufficient.
        for package in env._env.scenario.packages:
            package.on_goal[:] = False
        env._env.scenario.packages[0].on_goal[0] = True
        goal, collision, _ = mpc.transport_outcome(env)
        assert goal[0].item() == (n_packages == 1)
        assert not collision.any()
        for package in env._env.scenario.packages:
            package.on_goal[0] = True
        # Goal completion on the time-limit step wins over timeout.
        td.set(("agents", "reward"), torch.zeros(3, 4, 1))
        td.set("done", torch.ones(3, 1, dtype=torch.bool))
        env._env.steps[:] = 6
        stats.update(env, td)
        assert stats.success.tolist() == [True, False, False]
        assert stats.timeout.tolist() == [False, True, True]
        assert not stats.collision.any()
    finally:
        for item in (env, scratch, serial, single):
            item.close()
