# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the license in the LICENSE file in the root directory.
"""M3: controlled Transport data, using exact shared simulator anchors.

Run: python -m examples.world_model.collect [Hydra overrides].
All stored time indices count primitive actions. Simulator state is retained for
replay and physical-effect diagnostics, never silently added to model inputs.
"""

import hashlib
import json
import platform
import subprocess
import sys
import time
from importlib.metadata import version
from pathlib import Path

import hydra
import torch

from benchmarl.hydra_config import load_task_config_from_hydra
from examples.world_model.evaluate import state_digest, write_json
from examples.world_model.snapshot_restore import restore_state, snapshot_state
from hydra.core.hydra_config import HydraConfig
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from tensordict import TensorDict
from torchrl.record.loggers import get_logger
from torchrl.record.loggers.wandb import WandbLogger
from vmas.scenarios import transport as transport_scenario


REGIMES = ("independent", "correlated")
SPLITS = ("train", "validation", "test")


def map_tensors(tree, fn):
    return {
        key: map_tensors(value, fn) if isinstance(value, dict) else fn(value)
        for key, value in tree.items()
    }


def cat_trees(trees):
    return {
        key: cat_trees([tree[key] for tree in trees])
        if isinstance(trees[0][key], dict)
        else torch.cat([tree[key] for tree in trees])
        for key in trees[0]
    }


def episode_split(count, seed):
    """Assign root episodes before producing any related trajectory or branch."""
    if count < 8:
        raise ValueError("At least eight root episodes are needed for three splits")
    order = torch.randperm(count, generator=torch.Generator().manual_seed(seed))
    heldout = max(1, count // 8)
    split = torch.zeros(count, dtype=torch.long)
    split[order[-2 * heldout : -heldout]] = 1
    split[order[-heldout:]] = 2
    return split


def sample_actions(shape, low, high, regime, seed):
    """Uniform marginals; correlated actions share signs across all agents.

    The same seed gives exactly matched absolute normalized actions across
    regimes. Only signs change. The excluded region is a0_x * a1_x < 0 in
    normalized coordinates, an event of probability 1/2 under independent data.
    """
    if regime not in REGIMES:
        raise ValueError(f"Unknown action regime: {regime}")
    unit = torch.rand(shape, generator=torch.Generator().manual_seed(seed)) * 2 - 1
    if regime == "correlated":
        sign = torch.where(unit[..., :1, :] >= 0, 1.0, -1.0)
        unit = unit.abs() * sign
    return low.cpu() + (unit + 1) * 0.5 * (high - low).cpu()


def physical_state(entities):
    return torch.stack(
        [
            torch.cat([e.state.pos, e.state.vel, e.state.rot, e.state.ang_vel], -1)
            for e in entities
        ],
        dim=1,
    )


@torch.no_grad()
def rollout_actions(env, snapshot, actions, anchor_stride=None, policy=None):
    """Return (B,T,...) transitions; first terminal reward is valid, later data zero.

    Unlike TorchRL's VMAS `terminated` field, task termination here excludes
    time limits. No auto-reset is used. Snapshot clocks govern remaining budget.
    """
    restore_state(env, snapshot)
    count, length = actions.shape[:2]
    if count != env.batch_size[0] or length < 1:
        raise ValueError("Actions must match the environment batch and have time steps")
    live = ~env._env.done()
    records, anchors = [], []
    td = TensorDict({}, batch_size=[count], device=env.device)
    for step in range(length):
        if anchor_stride is not None and step % anchor_stride == 0 and live.any():
            indices = live.nonzero().squeeze(-1)
            anchors.append(
                {
                    "snapshot": map_tensors(
                        snapshot_state(env), lambda x, indices=indices: x[indices].cpu()
                    ),
                    "episode_id": indices.cpu(),
                    "source_step": env._env.steps[indices].long().cpu(),
                }
            )
        observation = torch.stack(
            [env._env.scenario.observation(agent) for agent in env._env.world.agents], 1
        )
        agent_state = physical_state(env._env.world.agents)
        package_state = physical_state(env._env.scenario.packages)
        action = actions[:, step].to(env.device).clone()
        if policy is not None and live.any():
            action[live] = policy(observation[live])
        action[~live] = 0
        if not torch.isfinite(action).all():
            raise ValueError("Collection policy produced non-finite actions")
        td.set(("agents", "action"), action)
        nxt = env.step(td)["next"]
        done = nxt["done"].squeeze(-1)
        terminated = env._env.scenario.done()
        record = {
            "observation": observation,
            "action": action,
            "next_observation": nxt["agents", "observation"],
            "reward": nxt["agents", "reward"],
            "done": done,
            "terminated": terminated,
            "truncated": done & ~terminated,
            "valid": live,
            "agent_state": agent_state,
            "next_agent_state": physical_state(env._env.world.agents),
            "package_state": package_state,
            "next_package_state": physical_state(env._env.scenario.packages),
        }
        records.append(
            {
                key: value.masked_fill(
                    ~live.reshape(count, *([1] * (value.ndim - 1))), 0
                ).cpu()
                for key, value in record.items()
            }
        )
        live = live & ~done
    return {
        key: torch.stack([r[key] for r in records], 1) for key in records[0]
    }, anchors


def branch_rollouts(task, anchors, actions, batch_size, device):
    """Bound scratch memory while retaining anchor/candidate ordering."""
    count = actions.shape[0]
    # Keep the same simulator width for full banks and smaller test subsets.
    # Padding avoids batch-width-dependent float32 differences in effect probes.
    width = batch_size
    env = task.get_env_fun(width, True, 0, device)()
    batches = []
    try:
        env.reset()
        for start in range(0, count, width):
            size = min(width, count - start)
            indices = torch.arange(start, start + width) % count
            snapshot = map_tensors(
                anchors["snapshot"], lambda x, indices=indices: x[indices].to(device)
            )
            data, _ = rollout_actions(env, snapshot, actions[indices])
            batches.append({key: value[:size] for key, value in data.items()})
    finally:
        env.close()
    return cat_trees(batches)


def action_coverage(data, low, high):
    action = data["action"][data["valid"]]
    unit = 2 * (action - low) / (high - low) - 1
    bins = [
        torch.histc(unit[:, i, j], bins=10, min=-1, max=1).long().tolist()
        for i in range(unit.shape[1])
        for j in range(unit.shape[2])
    ]
    return {
        "transitions": len(action),
        "opposed_x_fraction": ((unit[:, 0, 0] * unit[:, 1, 0]) < 0)
        .float()
        .mean()
        .item(),
        "marginal_mean": unit.mean(0).tolist(),
        "marginal_std": unit.std(0, unbiased=False).tolist(),
        "normalized_histograms_10_bins_agent_then_coordinate": bins,
        "nonzero_reward_fraction": (
            data["reward"][data["valid"]].abs().sum((-1, -2)) > 0
        )
        .float()
        .mean()
        .item(),
        "moving_package_fraction": (
            data["next_package_state"][data["valid"]][..., 2:4]
            .norm(dim=-1)
            .max(-1)
            .values
            > 1e-6
        )
        .float()
        .mean()
        .item(),
        "task_terminations": int(data["terminated"].sum()),
        "timeouts": int(data["truncated"].sum()),
        "mean_snippet_return": data["reward"].sum((1, 2, 3)).mean().item()
        / action.shape[1],
        "mean_snippet_length": data["valid"].sum(-1).float().mean().item(),
    }


def leakage_checks(initial, anchors, excluded=None):
    """Shared roots and duplicate physical anchors must never cross splits."""
    root_split = initial["split"]
    if not torch.equal(anchors["split"], root_split[anchors["episode_id"]]):
        raise ValueError("Anchor split differs from its source episode split")
    seen = {}
    for row, split in enumerate(anchors["split"].tolist()):
        identity = state_digest(
            map_tensors(anchors["snapshot"], lambda x, row=row: x[row : row + 1])
        )
        if identity in seen and seen[identity] != split:
            raise ValueError("Identical simulator anchor leaked across splits")
        seen[identity] = split
    excluded_hashes = set()
    if excluded is not None:
        for row in range(len(excluded["steps"])):
            excluded_hashes.add(
                state_digest(map_tensors(excluded, lambda x, row=row: x[row : row + 1]))
            )
    for row in range(len(root_split)):
        identity = state_digest(
            map_tensors(initial["snapshot"], lambda x, row=row: x[row : row + 1])
        )
        if identity in excluded_hashes:
            raise ValueError("Dataset initial state overlaps excluded evaluation bank")
    return {
        "episode_splits_disjoint": True,
        "anchor_splits_inherit_source_episode": True,
        "duplicate_anchor_cross_split_count": 0,
        "unique_anchor_states": len(seen),
        "excluded_evaluation_states_checked": len(excluded_hashes),
        "excluded_evaluation_state_overlap": 0,
        "root_episodes_by_split": {
            name: int((root_split == i).sum()) for i, name in enumerate(SPLITS)
        },
    }


def effect_summary(reference, counterfactual):
    """Change only agent 1's x action; observe other agents' absolute physics.

    Relative observation changes alone are not evidence of physical interaction.
    Comparisons stop once either branch has terminated.
    """
    valid = reference["valid"] & counterfactual["valid"]
    other = [i for i in range(reference["action"].shape[2]) if i != 1]
    result = {}
    for name, key, indices in (
        ("other_agent", "next_agent_state", other),
        ("package", "next_package_state", slice(None)),
    ):
        delta = (
            (counterfactual[key][:, :, indices, :4] - reference[key][:, :, indices, :4])
            .norm(dim=-1)
            .max(-1)
            .values
        )
        result[name] = {
            "mean_position_velocity_l2": delta[valid].mean().item(),
            "max_position_velocity_l2": delta[valid].max().item(),
            "fraction_above_1e-6": (delta[valid] > 1e-6).float().mean().item(),
            "anchors_with_effect": int(((delta > 1e-6) & valid).any(-1).sum()),
            "anchors": len(delta),
            "first_step_anchors_with_effect": int(
                ((delta[:, 0] > 1e-6) & valid[:, 0]).sum()
            ),
        }
    return result


def run_collection(cfg, output, task_name):
    if task_name != "vmas/transport":
        raise ValueError("M3 collection currently validates Transport only")
    settings = cfg.dataset
    if (
        min(
            settings.episodes,
            settings.anchor_stride,
            settings.sequence_steps,
            settings.branch_batch_size,
            cfg.task.max_steps,
        )
        < 1
    ):
        raise ValueError("Data budgets must be positive")
    if settings.action_block < 1 or settings.sequence_steps % settings.action_block:
        raise ValueError("Sequence steps must be divisible by a positive action block")
    if cfg.task.n_agents < 2:
        raise ValueError("The joint-action intervention requires at least two agents")
    if cfg.experiment.render:
        raise ValueError("Offline data collection does not render")
    if settings.include_heuristic and (
        cfg.task.n_packages != 1
        or cfg.task.package_width != 0.15
        or cfg.task.package_length != 0.15
    ):
        raise ValueError("VMAS Transport heuristic assumes one default-sized package")
    if (output / "manifest.json").exists():
        raise FileExistsError("Refusing to overwrite an existing fixed dataset")
    started = time.perf_counter()
    task = load_task_config_from_hydra(cfg.task, task_name)
    device = cfg.experiment.sampling_device
    if torch.device(device).type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    output.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, output / "resolved_config.yaml", resolve=True)
    loggers = [
        get_logger(
            logger_type=name,
            logger_name=str(output),
            experiment_name="offline_data",
            wandb_kwargs={
                "project": cfg.experiment.project_name,
                "group": "m3-transport-data",
                **OmegaConf.to_container(
                    cfg.experiment.wandb_extra_kwargs, resolve=True
                ),
            },
        )
        for name in cfg.experiment.loggers
    ]
    for logger in loggers:
        logger.log_hparams(OmegaConf.to_container(cfg, resolve=True))
    env = task.get_env_fun(settings.episodes, True, settings.state_seed, device)()
    try:
        env.set_seed(settings.state_seed)
        env.reset()
        initial_snapshot = snapshot_state(env)
        spec = env.full_action_spec_unbatched["agents", "action"]
        low, high = spec.low.cpu(), spec.high.cpu()
        initial = {
            "snapshot": map_tensors(initial_snapshot, lambda x: x.cpu()),
            "episode_id": torch.arange(settings.episodes),
            "split": episode_split(settings.episodes, settings.split_seed),
        }
        torch.save(initial, output / "initial_states.pt")
        source_regimes = list(REGIMES)
        if settings.include_heuristic:
            source_regimes.append("heuristic")
        anchor_parts, report = [], {"trajectories": {}, "branches": {}}
        for regime_id, regime in enumerate(source_regimes):
            policy = None
            if regime == "heuristic":
                # Shipped VMAS baseline; its competence is measured, not assumed.
                heuristic = transport_scenario.HeuristicPolicy(continuous_action=True)
                if not torch.equal(low, -torch.ones_like(low)) or not torch.equal(
                    high, torch.ones_like(high)
                ):
                    raise ValueError(
                        "Transport heuristic requires native [-1,1] actions"
                    )

                def policy(observation, heuristic=heuristic):
                    count, agents, features = observation.shape
                    return heuristic.compute_action(
                        observation.reshape(count * agents, features), 1.0
                    ).reshape(count, agents, -1)

                actions = torch.zeros(settings.episodes, cfg.task.max_steps, *low.shape)
            else:
                actions = sample_actions(
                    (settings.episodes, cfg.task.max_steps, *low.shape),
                    low,
                    high,
                    regime,
                    settings.action_seed,
                )
            data, parts = rollout_actions(
                env, initial_snapshot, actions, settings.anchor_stride, policy=policy
            )
            for part in parts:
                part["source_regime"] = torch.full_like(part["episode_id"], regime_id)
                part["split"] = initial["split"][part["episode_id"]]
            anchor_parts.extend(parts)
            torch.save(
                {
                    **data,
                    "episode_id": initial["episode_id"],
                    "split": initial["split"],
                },
                output / f"trajectories_{regime}.pt",
            )
            report["trajectories"][regime] = action_coverage(data, low, high)
        anchors = cat_trees(anchor_parts)
        excluded = None
        if settings.excluded_states_file is not None:
            excluded = torch.load(
                to_absolute_path(settings.excluded_states_file),
                map_location="cpu",
                weights_only=True,
            )["snapshot"]
        checks = leakage_checks(initial, anchors, excluded)
        torch.save(anchors, output / "anchors.pt")
        print(
            f"Collected {len(anchors['episode_id'])} anchors from {settings.episodes} root episodes",
            flush=True,
        )
        count = len(anchors["episode_id"])
        samples = {}
        for regime in REGIMES:
            actions = sample_actions(
                (count, settings.sequence_steps, *low.shape),
                low,
                high,
                regime,
                settings.branch_seed,
            )
            data = branch_rollouts(
                task, anchors, actions, settings.branch_batch_size, device
            )
            data["anchor_id"] = torch.arange(count)
            torch.save(data, output / f"samples_{regime}.pt")
            samples[regime] = data
            report["branches"][regime] = {
                name: action_coverage(
                    {
                        key: value[anchors["split"] == i]
                        for key, value in data.items()
                        if key != "anchor_id"
                    },
                    low,
                    high,
                )
                for i, name in enumerate(SPLITS)
            }
            print(f"Completed {regime} branches", flush=True)
        if not torch.equal(
            samples["independent"]["observation"][:, 0],
            samples["correlated"]["observation"][:, 0],
        ):
            raise ValueError(
                "Matched branches did not start from identical observations"
            )
        if not all(data["valid"][:, 0].all() for data in samples.values()):
            raise ValueError(
                "Every matched anchor must supply a valid first transition"
            )
        checks["matched_branch_initial_observations"] = True
        checks["matched_branch_anchors"] = True
        checks["matched_first_step_counts"] = True
        # Strict out-of-support probes exist only for test root episodes.
        test_ids = (anchors["split"] == 2).nonzero().squeeze(-1)
        test_anchors = map_tensors(anchors, lambda x: x[test_ids])
        reference = {
            key: value[test_ids] for key, value in samples["correlated"].items()
        }
        replay_reference = branch_rollouts(
            task, test_anchors, reference["action"], settings.branch_batch_size, device
        )
        for key, value in replay_reference.items():
            if not torch.equal(value, reference[key]):
                raise ValueError(f"Counterfactual reference replay mismatch: {key}")
        intervention = sample_actions(
            (count, settings.sequence_steps, *low.shape),
            low,
            high,
            "correlated",
            settings.branch_seed,
        )[test_ids]
        intervention[:, :, 1, 0] = low[1, 0] + high[1, 0] - intervention[:, :, 1, 0]
        cf = branch_rollouts(
            task, test_anchors, intervention, settings.branch_batch_size, device
        )
        cf["anchor_id"] = test_ids
        torch.save(cf, output / "counterfactual_test.pt")
        report["counterfactual"] = action_coverage(cf, low, high)
        report["intervention_effects"] = effect_summary(reference, cf)
        report["intervention_effects_by_source"] = {
            name: effect_summary(
                {
                    key: value[test_anchors["source_regime"] == i]
                    for key, value in reference.items()
                },
                {
                    key: value[test_anchors["source_regime"] == i]
                    for key, value in cf.items()
                },
            )
            for i, name in enumerate(source_regimes)
        }
        checks["counterfactual_reference_replay_bit_exact"] = True
        # Replay a whole scratch-width prefix twice, including boundary masks.
        replay_count = min(count, settings.branch_batch_size)
        replay_anchors = map_tensors(anchors, lambda x: x[:replay_count])
        replay = branch_rollouts(
            task,
            replay_anchors,
            samples["independent"]["action"][:replay_count],
            settings.branch_batch_size,
            device,
        )
        for key, value in replay.items():
            if not torch.equal(value, samples["independent"][key][:replay_count]):
                raise ValueError(f"Saved-data simulator replay mismatch: {key}")
        checks["replayed_anchors"] = replay_count
        checks["replay_bit_exact"] = True
        seconds = time.perf_counter() - started
        write_json(output / "coverage.json", report)
        write_json(output / "leakage_checks.json", checks)
        timing = {"seconds": seconds}
        if torch.device(device).type == "cuda":
            timing["peak_allocated_mib"] = (
                torch.cuda.max_memory_allocated(device) / 2**20
            )
            timing["peak_reserved_mib"] = (
                torch.cuda.max_memory_reserved(device) / 2**20
            )
        write_json(output / "timing.json", timing)
        source_dir = output / "source"
        source_dir.mkdir(exist_ok=True)
        for path in Path(__file__).parent.glob("*.py"):
            (source_dir / path.name).write_bytes(path.read_bytes())
        provenance = {
            "commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            "git_status": subprocess.check_output(
                ["git", "status", "--short"], text=True
            ),
            "versions": {
                name: version(name)
                for name in ("torch", "torchrl", "tensordict", "vmas", "benchmarl")
            },
            "python": platform.python_version(),
            "torch_cuda": torch.version.cuda,
            "vmas_transport_source_sha256": hashlib.sha256(
                Path(transport_scenario.__file__).read_bytes()
            ).hexdigest(),
            "device": str(device),
            "device_name": torch.cuda.get_device_name(device)
            if torch.device(device).type == "cuda"
            else "cpu",
            "source_sha256": {
                p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in source_dir.glob("*.py")
            },
        }
        write_json(output / "provenance.json", provenance)
        manifest = {
            "schema_version": 1,
            "task_name": task_name,
            "task": task.config,
            "seeds": {
                key: settings[key]
                for key in ("state_seed", "split_seed", "action_seed", "branch_seed")
            },
            "action_low": low.tolist(),
            "action_high": high.tolist(),
            "sequence_steps": settings.sequence_steps,
            "action_block": settings.action_block,
            "regimes": list(REGIMES),
            "source_regimes": source_regimes,
            "splits": list(SPLITS),
            "anchors": count,
            "initial_states_sha256": state_digest(initial),
            "anchors_sha256": state_digest(anchors),
            "excluded_states_sha256": state_digest(excluded)
            if excluded is not None
            else None,
            "control": "Identical source snapshots and first-step budgets; later branch states depend on actions.",
            "heldout_region": "normalized a0_x*a1_x < 0; absent from correlated actions, supported by independent actions",
            "files": {
                p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(output.iterdir())
                if p.suffix in (".pt", ".json", ".yaml")
            },
        }
        write_json(output / "manifest.json", manifest)
        for logger in loggers:
            logger.log_scalar("data/anchors", count, step=0)
            logger.log_scalar("data/collection_complete", 1, step=0)
            logger.log_scalar("data/seconds", seconds, step=0)
            for source, values in report["trajectories"].items():
                for key in (
                    "transitions",
                    "moving_package_fraction",
                    "task_terminations",
                    "mean_snippet_return",
                ):
                    logger.log_scalar(
                        f"data/source/{source}/{key}", values[key], step=0
                    )
            for regime in REGIMES:
                for split in SPLITS:
                    for key in (
                        "transitions",
                        "opposed_x_fraction",
                        "moving_package_fraction",
                    ):
                        logger.log_scalar(
                            f"data/{regime}/{split}/{key}",
                            report["branches"][regime][split][key],
                            step=0,
                        )
            for key, values in report["intervention_effects"].items():
                logger.log_scalar(
                    f"data/intervention/{key}_anchors_with_effect",
                    values["anchors_with_effect"],
                    step=0,
                )
        print(
            json.dumps(
                {
                    "output": str(output),
                    "seconds": seconds,
                    "checks": checks,
                    "effects": report["intervention_effects"],
                },
                indent=2,
            ),
            flush=True,
        )
        return manifest
    finally:
        env.close()
        for logger in loggers:
            if isinstance(logger, WandbLogger):
                logger.experiment.finish(exit_code=int(sys.exc_info()[0] is not None))


@hydra.main(
    version_base=None, config_path="../../benchmarl/conf", config_name="offline_data"
)
def main(cfg: DictConfig):
    run_collection(
        cfg,
        Path(HydraConfig.get().runtime.output_dir),
        HydraConfig.get().runtime.choices.task,
    )


if __name__ == "__main__":
    main()
