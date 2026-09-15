#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#
"""M4: train one latent world-model baseline on a fixed M3 dataset.

Two stages, deliberately separated:

1. **Dynamics** -- LeWM's objective exactly: masked next-latent MSE against an
   undetached target, plus ``sigreg_weight * SIGReg(latents)``. This is the only
   stage the three baselines are compared on, so nothing but the prediction and
   anti-collapse terms shapes the representation.
2. **Readout** -- the reward/termination head for ``J = -sum_t sum_i r_i,t``,
   fitted on *frozen* dynamics. Planning consumes predicted latents, so the head
   is fitted on predicted latents; the teacher-forced error is reported beside it
   as a diagnostic.

Run: ``python -m examples.world_model.train [overrides]``
"""

import csv
import hashlib
import json
import subprocess
from importlib.metadata import version
from pathlib import Path

import hydra
import torch

from benchmarl.experiment.logger import JsonWriter
from examples.world_model.dataset import OfflineSequences
from examples.world_model.models import (
    conditioning_gate_scale,
    effective_rank,
    latent_variance,
    masked_mean,
    masked_sum_count,
    MultiAgentWorldModel,
    parameter_counts,
    SIGReg,
)
from omegaconf import DictConfig, OmegaConf
from tensordict import TensorDict
from torch.utils.data import DataLoader
from hydra.utils import to_absolute_path
from torchrl.record.loggers import get_logger
from torchrl.record.loggers.utils import generate_exp_name


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def loaders(cfg):
    splits = {}
    for split in ("train", "validation"):
        data = OfflineSequences(
            cfg.data.root, cfg.data.regime, split, action_block=cfg.data.action_block
        )
        splits[split] = DataLoader(
            data,
            batch_size=cfg.train.batch_size,
            shuffle=split == "train",
            drop_last=False,
            collate_fn=torch.stack,
            generator=torch.Generator().manual_seed(cfg.seed),
        )
    return splits


def observation_statistics(loader):
    """Normalisation fitted on valid TRAINING frames only (M3 handoff contract)."""
    total, square, count = 0.0, 0.0, 0
    for batch in loader:
        frames = batch["observation"][batch["observation_valid"]]
        flat = frames.reshape(-1, frames.size(-1)).double()
        total = total + flat.sum(dim=0)
        square = square + flat.square().sum(dim=0)
        count += flat.size(0)
    mean = total / count
    variance = (square / count - mean.square()).clamp_min(1e-8)
    return mean.float(), variance.sqrt().float()


def block_reward(batch):
    """Sum primitive rewards inside each action block: (B,L,block,N,1) -> (B,L,N,1).

    The planner's objective sums every primitive reward, so the block-level
    target must be the sum, not the mean. Padded primitive steps are already
    zero-filled by the collector but are masked again by ``valid`` downstream.
    """
    rewards = batch["primitive_reward"]
    valid = batch["primitive_valid"].unsqueeze(-1).unsqueeze(-1)
    return (rewards * valid).sum(dim=2)


def dynamics_losses(model, sigreg, batch, cfg):
    """LeWM's two-term objective, masked to valid blocks/frames."""
    latent = model.encode(batch["observation"])
    predicted = model.predict(latent[:, :-1], batch["action"])
    target = latent[:, 1:]  # undetached, as in the reference
    prediction = masked_mean((predicted - target).square(), batch["valid"])
    frames = latent[batch["observation_valid"]]
    population = frames.reshape(1, -1, frames.size(-1))
    anti_collapse = sigreg(population)
    return {
        "prediction": prediction,
        "sigreg": anti_collapse,
        "loss": prediction + cfg.train.sigreg_weight * anti_collapse,
    }


def readout_losses(model, batch):
    """Reward/termination fit on frozen, predicted latents."""
    with torch.no_grad():
        latent = model.encode(batch["observation"])
        predicted = model.predict(latent[:, :-1], batch["action"])
    reward, terminated = model.readout(latent[:, :-1], predicted)
    valid = batch["valid"]
    reward_loss = masked_mean((reward - block_reward(batch)).square(), valid)
    target = batch["terminated"].float()
    termination = torch.nn.functional.binary_cross_entropy_with_logits(
        terminated, target, reduction="none"
    )
    return {
        "reward": reward_loss,
        "termination": masked_mean(termination, valid),
        "loss": reward_loss + masked_mean(termination, valid),
    }


@torch.no_grad()
def evaluate(model, sigreg, loader, cfg, device):
    """Validation metrics: prediction, latent health, rollout error, readout.

    Every masked metric is pooled by valid-entry count across the whole split,
    so a batch whose snippets all terminate early contributes proportionally
    rather than being averaged in as an equal-weight batch mean.
    """
    model.eval()
    sums, counts, latents = {}, {}, []
    sigreg_total, sigreg_batches = 0.0, 0
    for batch in loader:
        batch = batch.to(device)
        latent = model.encode(batch["observation"])
        predicted = model.predict(latent[:, :-1], batch["action"])
        # Multi-step: roll from frame 0 on actions alone, no teacher forcing.
        rolled = model.rollout(latent[:, :1], batch["action"])
        reward, _ = model.readout(latent[:, :-1], predicted)
        valid = batch["valid"]
        target = latent[:, 1:]
        pairs = {
            "one_step_error": ((predicted - target).square(), valid),
            "rollout_error": ((rolled - target).square(), valid),
            "final_step_error": (
                (rolled[:, -1:] - target[:, -1:]).square(),
                valid[:, -1:],
            ),
            "reward_error": ((reward - block_reward(batch)).square(), valid),
        }
        # Reward targets, to normalise the readout error by the variance a
        # constant predictor would already achieve.
        target_reward = block_reward(batch)
        for key, (values, mask) in list(pairs.items()) + [
            ("reward_target_sum", (target_reward, valid)),
            ("reward_target_square", (target_reward.square(), valid)),
        ]:
            total, count = masked_sum_count(values, mask)
            sums[key] = sums.get(key, 0.0) + float(total)
            counts[key] = counts.get(key, 0.0) + float(count)
        terminated_total, terminated_count = masked_sum_count(
            batch["terminated"].float(), valid
        )
        sums["terminated_rate"] = sums.get("terminated_rate", 0.0) + float(
            terminated_total
        )
        counts["terminated_rate"] = counts.get("terminated_rate", 0.0) + float(
            terminated_count
        )
        frames = latent[batch["observation_valid"]]
        sigreg_total += float(sigreg(frames.reshape(1, -1, frames.size(-1))))
        sigreg_batches += 1
        latents.append(frames.reshape(-1, model.dim))

    model.train()
    metrics = {key: sums[key] / counts[key] for key in sums if counts[key] > 0}

    # A constant predictor scores exactly the target variance, so anything at or
    # above 1.0 here means the readout has learned nothing usable for planning.
    reward_variance = (
        metrics.pop("reward_target_square") - metrics.pop("reward_target_sum") ** 2
    )
    metrics["reward_target_variance"] = reward_variance
    metrics["reward_relative_error"] = (
        metrics["reward_error"] / reward_variance
        if reward_variance > 0
        else float("nan")
    )
    metrics["sigreg"] = sigreg_total / sigreg_batches
    population = torch.cat(latents)
    mask = torch.ones(1, population.size(0), dtype=torch.bool, device=population.device)
    metrics["latent_variance"] = float(latent_variance(population.unsqueeze(0), mask))
    metrics["effective_rank"] = effective_rank(population.unsqueeze(0), mask)
    metrics["evaluated_elements"] = counts.get("one_step_error", 0.0)
    return metrics


def run_stage(model, parameters, loader, step_fn, epochs, cfg, device, logger, stage):
    optimizer = torch.optim.AdamW(
        parameters, lr=cfg.train.learning_rate, weight_decay=cfg.train.weight_decay
    )
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(epochs, 1)
    )
    history = []
    for epoch in range(epochs):
        totals, weight = {}, 0.0
        for batch in loader:
            batch = batch.to(device)
            losses = step_fn(batch)
            optimizer.zero_grad(set_to_none=True)
            losses["loss"].backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                parameters, cfg.train.gradient_clip
            )
            if not torch.isfinite(grad_norm):
                raise ValueError(f"{stage}: non-finite gradient at epoch {epoch}")
            optimizer.step()
            size = batch.batch_size[0]
            for key, value in losses.items():
                totals[key] = totals.get(key, 0.0) + float(value) * size
            totals["grad_norm"] = totals.get("grad_norm", 0.0) + float(grad_norm) * size
            weight += size
        schedule.step()
        record = {key: value / weight for key, value in totals.items()}
        record["epoch"] = epoch
        history.append(record)
        for key, value in record.items():
            if key != "epoch":
                logger.log_scalar(f"{stage}/{key}", value, step=epoch)
    return history


MODEL_NAME = "lewm"
MARL_EVAL_FILE = "marl_eval.json"

# Metrics written to the marl-eval file, and whether smaller is better. The
# reporting stack assumes larger is better, so those are negated on the way out
# and renamed, because a field called `rollout_error` holding a negative number
# would mislead anyone reading the file without this table beside it.
REPORTED_METRICS = {
    "one_step_error": True,
    "rollout_error": True,
    "final_step_error": True,
    "latent_variance": False,
    "effective_rank": False,
}


def run_training(cfg, output: Path):
    torch.manual_seed(cfg.seed)
    device = cfg.device
    output.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, output / "resolved_config.yaml", resolve=True)

    sources = Path(__file__).parent
    write_json(
        output / "provenance.json",
        {
            "commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            "git_status": subprocess.check_output(
                ["git", "status", "--short"], text=True
            ),
            "versions": {
                name: version(name) for name in ("torch", "tensordict", "benchmarl")
            },
            "device": str(device),
            "lewm_reference": "8edfeb336732b5f3ce7b8b210d0ba370a09e2cac",
            "source_sha256": {
                path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                for path in sorted(sources.glob("*.py"))
            },
        },
    )

    splits = loaders(cfg)
    train_loader, validation_loader = splits["train"], splits["validation"]
    mean, std = observation_statistics(train_loader)

    sample = next(iter(train_loader))
    frames, agents, obs_dim = sample["observation"].shape[1:]
    action_dim = sample["action"].shape[-1]

    model = MultiAgentWorldModel(
        cfg.model.kind,
        obs_dim,
        action_dim,
        agents,
        dim=cfg.model.dim,
        hidden_dim=cfg.model.hidden_dim,
        conditioner_budget=cfg.model.conditioner_budget,
        frames=frames,
        depth=cfg.model.depth,
        heads=cfg.model.heads,
        dim_head=cfg.model.dim_head,
        mlp_dim=cfg.model.mlp_dim,
        dropout=cfg.model.dropout,
        obs_mean=mean,
        obs_std=std,
    ).to(device)
    sigreg = SIGReg(cfg.train.sigreg_knots, cfg.train.sigreg_projections).to(device)

    # BenchMARL's own convention, so these runs group and aggregate like every
    # other run in the project: the name carries algorithm, task and model, the
    # wandb group is the task, and the id is the unique experiment name. Our
    # three predictors are the algorithm -- they are what is compared on a
    # shared task at matched capacity, exactly as MAPPO and IPPO are there.
    task_name = json.loads(
        (Path(to_absolute_path(cfg.data.root)) / "manifest.json").read_text()
    )["task_name"]
    environment_name, task_name = task_name.split("/")
    algorithm_name = f"{cfg.model.kind}_{cfg.data.regime}"
    experiment_name = generate_exp_name(
        f"{algorithm_name}_{task_name}_{MODEL_NAME}", ""
    )
    loggers = [
        get_logger(
            logger_type=name,
            logger_name=str(output),
            experiment_name=experiment_name,
            wandb_kwargs={
                "group": task_name,
                "id": experiment_name,
                "project": cfg.wandb.project,
                "entity": cfg.wandb.entity,
                # Config fields, so the UI can group or filter on any axis.
                "config": {
                    "algorithm": algorithm_name,
                    "kind": cfg.model.kind,
                    "regime": cfg.data.regime,
                    "task": task_name,
                    "environment": environment_name,
                    "seed": cfg.seed,
                    "model": MODEL_NAME,
                },
            },
        )
        for name in cfg.loggers
    ]

    # BenchMARL's marl-eval reporting file, written per run exactly as
    # Experiment does, so benchmarl.eval_results reads these runs natively.
    # Fixed name rather than the experiment name: BenchMARL's own run folders
    # hold exactly one json, so its loader walks for any of them, while ours
    # also hold metrics/parameters/provenance. A known name keeps the reporting
    # file identifiable without renaming our diagnostics.
    json_writer = JsonWriter(
        folder=str(output),
        name=MARL_EVAL_FILE,
        algorithm_name=algorithm_name,
        task_name=task_name,
        environment_name=environment_name,
        seed=cfg.seed,
    )

    class Fan:
        def log_scalar(self, key, value, step=None):
            for logger in loggers:
                logger.log_scalar(key, value, step=step)

    fan = Fan()
    counts = parameter_counts(model)
    write_json(output / "parameters.json", counts)

    dynamics_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if not name.startswith("readout.")
    ]
    dynamics_history = run_stage(
        model,
        dynamics_parameters,
        train_loader,
        lambda batch: dynamics_losses(model, sigreg, batch, cfg),
        cfg.train.dynamics_epochs,
        cfg,
        device,
        fan,
        "dynamics",
    )

    for parameter in dynamics_parameters:
        parameter.requires_grad_(False)
    readout_history = run_stage(
        model,
        list(model.readout.parameters()),
        train_loader,
        lambda batch: readout_losses(model, batch),
        cfg.train.readout_epochs,
        cfg,
        device,
        fan,
        "readout",
    )

    metrics = evaluate(model, sigreg, validation_loader, cfg, device)
    metrics["latent_dimensions"] = model.dim
    metrics["conditioning_gate_scale"] = conditioning_gate_scale(model)
    # Transport branch snippets contain no task terminations, so the termination
    # head has no positive examples and cannot be validated on this bank. Record
    # that rather than letting a trivially "accurate" always-false head look fit
    # for the `J = -sum r through first termination` objective.
    metrics["termination_head_validated"] = bool(metrics["terminated_rate"] > 0)
    metrics["parameters"] = counts["total"]
    metrics["conditioner_parameters"] = counts["conditioner"]
    metrics["conditioner_hidden"] = model.conditioner_hidden
    metrics["dynamics_parameters"] = counts["dynamics_total"]

    checkpoint = output / "model.pt"
    torch.save(
        {
            "kind": cfg.model.kind,
            "state_dict": model.state_dict(),
            "config": OmegaConf.to_container(cfg, resolve=True),
            "observation_mean": mean,
            "observation_std": std,
            "shapes": {
                "agents": agents,
                "obs_dim": obs_dim,
                "action_dim": action_dim,
                "frames": frames,
            },
        },
        checkpoint,
    )
    metrics["checkpoint_reload_max_difference"] = verify_reload(
        checkpoint, sample, device
    )

    for key, value in metrics.items():
        fan.log_scalar(f"validation/{key}", value, step=cfg.train.dynamics_epochs)
    write_json(output / "metrics.json", metrics)

    # The marl-eval file BenchMARL's tooling reads. Errors are negated into
    # scores because everything downstream -- rliable's aggregates, marl-eval's
    # normalisation -- assumes larger is better. One value per metric: these are
    # per-run figures, not per-episode ones.
    json_writer.write(
        total_frames=len(train_loader.dataset) * cfg.train.dynamics_epochs,
        metrics={
            (f"neg_{name}" if lower_is_better else name): torch.tensor(
                [-value if lower_is_better else value]
            )
            for name, lower_is_better in REPORTED_METRICS.items()
            if (value := metrics.get(name)) is not None
        },
        evaluation_step=0,
    )
    with (output / "history.csv").open("w", newline="") as stream:
        rows = [{"stage": "dynamics", **row} for row in dynamics_history]
        rows += [{"stage": "readout", **row} for row in readout_history]
        writer = csv.DictWriter(
            stream, fieldnames=sorted({key for row in rows for key in row})
        )
        writer.writeheader()
        writer.writerows(rows)
    for logger in loggers:
        if hasattr(logger, "experiment") and hasattr(logger.experiment, "finish"):
            logger.experiment.finish()
    return metrics


def load_model(checkpoint_path, device="cpu"):
    """Rebuild a trained model from a checkpoint, for M5 and the reload check."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = OmegaConf.create(checkpoint["config"])
    shapes = checkpoint["shapes"]
    model = MultiAgentWorldModel(
        checkpoint["kind"],
        shapes["obs_dim"],
        shapes["action_dim"],
        shapes["agents"],
        dim=cfg.model.dim,
        hidden_dim=cfg.model.hidden_dim,
        conditioner_budget=cfg.model.conditioner_budget,
        frames=shapes["frames"],
        depth=cfg.model.depth,
        heads=cfg.model.heads,
        dim_head=cfg.model.dim_head,
        mlp_dim=cfg.model.mlp_dim,
        dropout=cfg.model.dropout,
        obs_mean=checkpoint["observation_mean"],
        obs_std=checkpoint["observation_std"],
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


def verify_reload(checkpoint_path, sample: TensorDict, device):
    """A reloaded checkpoint must reproduce predictions exactly, not approximately."""
    original = load_model(checkpoint_path, device)
    reloaded = load_model(checkpoint_path, device)
    batch = sample.to(device)
    with torch.no_grad():
        first = original.predict(
            original.encode(batch["observation"])[:, :-1], batch["action"]
        )
        second = reloaded.predict(
            reloaded.encode(batch["observation"])[:, :-1], batch["action"]
        )
    difference = float((first - second).abs().max())
    if difference != 0.0:
        raise ValueError(f"Checkpoint reload changed predictions by {difference}")
    return difference


@hydra.main(
    version_base=None, config_path="../../benchmarl/conf", config_name="world_model"
)
def main(cfg: DictConfig):
    from hydra.core.hydra_config import HydraConfig

    output = Path(HydraConfig.get().runtime.output_dir)
    metrics = run_training(cfg, output)
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
