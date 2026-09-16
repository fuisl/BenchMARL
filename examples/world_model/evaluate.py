# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the license in the LICENSE file in the root directory.
"""Hydra entry point: python -m examples.world_model.evaluate [overrides]."""

import csv
import hashlib
import json
import subprocess
from importlib.metadata import version
from pathlib import Path

import hydra
import torch

from benchmarl.hydra_config import load_task_config_from_hydra
from examples.world_model.cem import CEMConfig
from examples.world_model.metrics import summarize
from examples.world_model.mpc import (
    evaluate_policy,
    MPCConfig,
    task_outcome,
)
from examples.world_model.snapshot_restore import snapshot_state
from hydra.core.hydra_config import HydraConfig
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from torchrl.record.loggers import get_logger
from torchrl.record.loggers.wandb import WandbLogger


def state_digest(bank):
    """Content identity independent of torch.save's filename/device metadata."""
    digest = hashlib.sha256()

    def update(value):
        if isinstance(value, dict):
            for key in sorted(value):
                digest.update(key.encode())
                update(value[key])
        elif isinstance(value, torch.Tensor):
            digest.update(str((value.dtype, tuple(value.shape))).encode())
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
        else:
            digest.update(json.dumps(value, sort_keys=True).encode())

    update(bank)
    return digest.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def run_evaluation(cfg, output: Path):
    """Use task/logging defaults without constructing a training Experiment."""
    task_name = HydraConfig.get().runtime.choices.task
    # Raises for a task with no recorded success/failure contract rather than
    # scoring it with another task's.
    outcome_fn = task_outcome(task_name)
    cem = CEMConfig(**OmegaConf.to_container(cfg.cem, resolve=True))
    mpc = MPCConfig(**OmegaConf.to_container(cfg.mpc, resolve=True))
    mpc.validate(cem.horizon)
    count = cfg.experiment.evaluation_episodes
    if count < 1 or cfg.task.max_steps < 1:
        raise ValueError("Episode count and task max_steps must be positive")
    if cfg.experiment.render:
        raise ValueError("The oracle evaluator does not render")
    task = load_task_config_from_hydra(cfg.task, task_name)
    device = cfg.experiment.sampling_device
    output.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, output / "resolved_config.yaml", resolve=True)
    versions = {
        name: version(name) for name in ("torch", "torchrl", "vmas", "benchmarl")
    }
    provenance = {
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "git_status": subprocess.check_output(["git", "status", "--short"], text=True),
        "versions": versions,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device)
        if torch.device(device).type == "cuda"
        else "cpu",
        "planner_seed": cfg.seed,
        "task": task_name,
        "random_seed": cfg.evaluation.random_seed,
        "statistics_seed": cfg.evaluation.statistics_seed,
    }
    # Retain the actual research sources for dirty-checkout experiments.
    sources = Path(__file__).parent
    provenance["source_sha256"] = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(sources.glob("*.py"))
    }
    source_dir = output / "source"
    source_dir.mkdir(exist_ok=True)
    for path in sources.glob("*.py"):
        (source_dir / path.name).write_bytes(path.read_bytes())
    write_json(output / "provenance.json", provenance)

    loggers = [
        get_logger(
            logger_type=name,
            logger_name=str(output),
            experiment_name="oracle_mpc",
            wandb_kwargs={
                "project": cfg.experiment.project_name,
                "group": f"m2-{task_name.split('/')[-1]}-oracle",
                **OmegaConf.to_container(
                    cfg.experiment.wandb_extra_kwargs, resolve=True
                ),
            },
        )
        for name in cfg.experiment.loggers
    ]
    for logger in loggers:
        logger.log_hparams(OmegaConf.to_container(cfg, resolve=True))

    env = task.get_env_fun(count, True, cfg.evaluation.state_seed, device)()
    scratch = task.get_env_fun(
        count * cem.num_samples, True, cfg.evaluation.state_seed, device
    )()
    try:
        env.set_seed(cfg.evaluation.state_seed)
        env.reset()
        scratch.reset()
        if cfg.evaluation.states_file is None:
            bank = {
                "task": task.config,
                "state_seed": cfg.evaluation.state_seed,
                "snapshot": snapshot_state(env),
            }
        else:
            bank = torch.load(
                to_absolute_path(cfg.evaluation.states_file),
                map_location=device,
                weights_only=True,
            )
            if bank["task"] != task.config:
                raise ValueError("State bank task settings differ from this evaluation")
        if bank["snapshot"]["steps"].shape != (count,):
            raise ValueError(
                "State bank episode count differs from evaluation_episodes"
            )
        bank_id = state_digest(bank)
        torch.save(bank, output / "initial_states.pt")
        write_json(
            output / "state_bank.json",
            {"sha256": bank_id, "state_seed": bank["state_seed"]},
        )
        rows, timings = [], {}
        # Full-budget MPC runs first; random restores the identical saved states.
        for policy, seed in (("mpc", cfg.seed), ("random", cfg.evaluation.random_seed)):
            episodes, timing, _terminal = evaluate_policy(
                env,
                bank["snapshot"],
                policy=policy,
                generator=torch.Generator(device=device).manual_seed(seed),
                cem_config=cem,
                mpc_config=mpc,
                scratch_env=scratch,
                diagnostics_path=output,
                outcome_fn=outcome_fn,
            )
            rows.extend(episodes)
            timings[policy] = timing
            with (output / "episodes.csv").open("w", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            write_json(output / "timing.json", timings)
            print(
                f"{policy}: {timing['seconds']:.2f}s, {sum(r['success'] for r in episodes)}/{count} successes",
                flush=True,
            )

        summary = summarize(rows, seed=cfg.evaluation.statistics_seed)
        summary["state_bank_sha256"] = bank_id
        write_json(output / "summary.json", summary)
        for logger in loggers:
            for policy in ("mpc", "random"):
                for metric, value in {
                    "return_mean": summary[policy]["return"]["mean"],
                    "return_ci_low": summary[policy]["return"]["ci95"][0],
                    "return_ci_high": summary[policy]["return"]["ci95"][1],
                    "success_rate": summary[policy]["success"]["rate"],
                    "collision_rate": summary[policy]["collision_rate"],
                    "timeout_rate": summary[policy]["timeout_rate"],
                    "seconds": timings[policy]["seconds"],
                }.items():
                    logger.log_scalar(f"eval/{policy}/{metric}", value, step=0)
            logger.log_scalar(
                "eval/m2_return_gate", float(summary["m2_return_gate"]), step=0
            )
        print(json.dumps(summary, indent=2), flush=True)
        return summary
    finally:
        env.close()
        scratch.close()
        for logger in loggers:
            if isinstance(logger, WandbLogger):
                logger.experiment.finish()


@hydra.main(
    version_base=None, config_path="../../benchmarl/conf", config_name="oracle_mpc"
)
def main(cfg: DictConfig):
    run_evaluation(cfg, Path(HydraConfig.get().runtime.output_dir))


if __name__ == "__main__":
    main()
