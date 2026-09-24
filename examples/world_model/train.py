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
import math
import subprocess
from contextlib import nullcontext
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
    ReferenceMultiAgentWorldModel,
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
            cfg.data.root,
            cfg.data.regime,
            split,
            action_block=cfg.data.action_block,
            state_input=cfg.data.state_input,
            history_frames=cfg.data.history_frames,
            train_fraction=cfg.data.get("train_fraction", 1.0) if split == "train" else 1.0,
        )
        splits[split] = DataLoader(
            data,
            batch_size=cfg.train.batch_size,
            shuffle=split == "train",
            # Pinned LeWM drops the incomplete training batch.  Preserve the
            # historical behavior for every legacy checkpoint and experiment.
            drop_last=(
                split == "train"
                and cfg.model.get("profile", "legacy_compact") == "lewm_reference"
            ),
            collate_fn=torch.stack,
            generator=torch.Generator().manual_seed(cfg.seed),
        )
    return splits


def reference_training_view(model, batch, random_window=False):
    """Return LeWM's registered 3-context/1-prediction training window.

    Pinned LeWM loads four frames, feeds the first three embeddings/actions to
    the predictor, and uses frames 1..3 as shifted targets.  Existing offline
    snippets are longer, so step 1 takes their first exact reference window;
    it does not reinterpret the legacy concatenated-history input condition.

    ``random_window`` draws each row's window start uniformly instead, the way
    LeWM's loader samples windows from whole trajectories. The first-window
    default uses only four frames of every snippet and discards the rest.
    """
    if getattr(model, "profile", "legacy_compact") != "lewm_reference":
        return batch
    required_frames = model.history_size + 1
    if batch["observation"].shape[1] < required_frames:
        raise ValueError(
            f"Reference training needs {required_frames} observations, got "
            f"{batch['observation'].shape[1]}"
        )
    frame_keys = {
        "observation",
        "agent_state",
        "package_state",
        "observation_valid",
    }
    block_keys = {
        "action",
        "primitive_action",
        "primitive_reward",
        "primitive_valid",
        "valid",
        "outcome_valid",
        "done",
        "terminated",
        "truncated",
    }
    frames = batch["observation"].shape[1]
    rows = batch["observation"].shape[0]
    device = batch["observation"].device
    if random_window and frames > required_frames:
        start = torch.randint(0, frames - required_frames + 1, (rows,), device=device)
    else:
        start = torch.zeros(rows, dtype=torch.long, device=device)
    row = torch.arange(rows, device=device).unsqueeze(1)
    frame_index = start.unsqueeze(1) + torch.arange(required_frames, device=device)
    block_index = frame_index[:, : model.history_size]
    view = {}
    for key, value in batch.items():
        if key in frame_keys:
            value = value[row, frame_index]
        elif key in block_keys:
            value = value[row, block_index]
        view[key] = value
    return view


def observation_statistics(loader, max_frames=None):
    """Normalisation fitted on valid TRAINING frames only (M3 handoff contract)."""
    total, square, count = 0.0, 0.0, 0
    for batch in loader:
        observation = batch["observation"]
        valid = batch["observation_valid"]
        if max_frames is not None:
            observation = observation[:, :max_frames]
            valid = valid[:, :max_frames]
        frames = observation[valid]
        flat = frames.reshape(-1, frames.size(-1)).double()
        total = total + flat.sum(dim=0)
        square = square + flat.square().sum(dim=0)
        count += flat.size(0)
    mean = total / count
    variance = (square / count - mean.square()).clamp_min(1e-8)
    return mean.float(), variance.sqrt().float()


def action_statistics(loader, max_blocks=None, action_block=1):
    """Per-coordinate sample statistics from valid TRAINING actions only.

    Pinned LeWM fits ``get_column_normalizer`` on the raw ``action`` column and
    only afterwards concatenates ``frameskip`` steps for its action encoder, so
    one statistic per primitive coordinate is shared by every block position.
    We reproduce that by reducing over block positions and tiling the result
    across the blocked width, which also keeps the transform independent of an
    action's offset inside its block.

    ``torch.std``'s default corrected (sample) denominator and the reference
    drop of non-finite rows are retained.  We fit after the split rather than
    on the full dataset as the reference script does, to keep validation
    information out of this experiment.  Agent rows are samples of one shared
    action interface.
    """
    total = square = None
    count = 0
    for batch in loader:
        action = batch["action"]
        valid = batch["valid"]
        if max_blocks is not None:
            action = action[:, :max_blocks]
            valid = valid[:, :max_blocks]
        width = action.size(-1)
        if width % action_block:
            raise ValueError(
                f"Blocked action width {width} is not divisible by "
                f"action_block={action_block}"
            )
        # (L,N,block*A) is block-major, so this recovers the primitive rows
        # that the reference normalizer would have seen.
        flat = action[valid].reshape(-1, width // action_block).double()
        flat = flat[torch.isfinite(flat).all(dim=1)]
        if not flat.numel():
            continue
        batch_total = flat.sum(dim=0)
        batch_square = flat.square().sum(dim=0)
        total = batch_total if total is None else total + batch_total
        square = batch_square if square is None else square + batch_square
        count += flat.size(0)
    if count < 2:
        raise ValueError(
            "Reference action normalization needs at least two finite valid "
            f"training primitive actions, got {count}"
        )
    mean = total / count
    variance = ((square - total.square() / count) / (count - 1)).clamp_min(1e-8)
    return mean.float().repeat(action_block), variance.sqrt().float().repeat(
        action_block
    )


def block_reward(batch):
    """Sum primitive rewards inside each action block: (B,L,block,N,1) -> (B,L,N,1).

    The planner's objective sums every primitive reward, so the block-level
    target must be the sum, not the mean. Padded primitive steps are already
    zero-filled by the collector but are masked again by ``valid`` downstream.
    """
    rewards = batch["primitive_reward"]
    valid = batch["primitive_valid"].unsqueeze(-1).unsqueeze(-1)
    return (rewards * valid).sum(dim=2)


def sigreg_loss(model, sigreg, latent, observation_valid, cfg):
    """Apply the registered SIGReg population semantics for one model profile."""
    profile = getattr(model, "profile", "legacy_compact")
    mode = cfg.train.get(
        "sigreg_population",
        "legacy_flattened" if profile == "legacy_compact" else "joint",
    )
    if profile == "legacy_compact":
        if mode != "legacy_flattened":
            raise ValueError("Legacy checkpoints require legacy_flattened SIGReg")
        frames = latent[observation_valid]
        return sigreg(frames.reshape(1, -1, frames.size(-1)))
    if mode not in ("joint", "per_agent"):
        raise ValueError(f"Unknown reference SIGReg population: {mode}")

    # Preserve the temporal axis and one rectangular population, as pinned
    # LeWM does.  Padded/partial snippets are excluded from SIGReg only; their
    # valid transitions remain available to the masked prediction objective.
    complete = observation_valid.all(dim=1)
    encoded = latent[complete]
    if encoded.shape[0] == 0:
        raise ValueError("Reference SIGReg needs at least one complete sequence")
    if mode == "joint":
        population = encoded.permute(1, 0, 2, 3).reshape(
            encoded.shape[1], encoded.shape[0] * encoded.shape[2], encoded.shape[3]
        )
        return sigreg(population)
    return torch.stack(
        [
            sigreg(encoded[:, :, agent].transpose(0, 1))
            for agent in range(encoded.shape[2])
        ]
    ).mean()


def dynamics_losses(model, sigreg, batch, cfg):
    """LeWM's two-term objective, masked to valid blocks/frames."""
    batch = reference_training_view(
        model, batch, cfg.train.get("reference_window", "first") == "random"
    )
    latent = model.encode(batch["observation"])
    predicted = model.predict(latent[:, :-1], batch["action"])
    target = latent[:, 1:]  # undetached, as in the reference
    prediction = masked_mean((predicted - target).square(), batch["valid"])
    anti_collapse = sigreg_loss(
        model, sigreg, latent, batch["observation_valid"], cfg
    )
    return {
        "prediction": prediction,
        "sigreg": anti_collapse,
        "loss": prediction + cfg.train.sigreg_weight * anti_collapse,
    }


def readout_losses(model, batch):
    """Reward/termination fit on frozen, predicted latents."""
    batch = reference_training_view(model, batch)
    with torch.no_grad():
        latent = model.encode(batch["observation"])
        predicted = model.predict(latent[:, :-1], batch["action"])
    reward, terminated = model.readout(latent[:, :-1], predicted)
    # Outcome validity, not dynamics validity: the readout reads the block-start
    # latent and the model's own predicted next latent, never the unobserved
    # end-of-block frame, so a partial terminal block is a legitimate target.
    # Masking it out dropped most terminations and every collision penalty.
    valid = batch["outcome_valid"]
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
        batch = reference_training_view(model, batch)
        latent = model.encode(batch["observation"])
        predicted = model.predict(latent[:, :-1], batch["action"])
        if getattr(model, "profile", "legacy_compact") == "lewm_reference":
            rolled = model.rollout_from_context(
                latent[:, : model.history_size],
                batch["action"][:, : model.history_size - 1],
                batch["action"][:, model.history_size - 1 : model.history_size],
            )
            rollout_target = latent[:, model.history_size : model.history_size + 1]
            rollout_valid = batch["valid"][
                :, model.history_size - 1 : model.history_size
            ]
        else:
            # Historical metric: roll from frame 0 on actions alone.
            rolled = model.rollout(latent[:, :1], batch["action"])
            rollout_target = latent[:, 1:]
            rollout_valid = batch["valid"]
        reward, _ = model.readout(latent[:, :-1], predicted)
        valid = batch["valid"]
        # Reward and termination are scored on outcome validity, matching how
        # they are trained; prediction stays on dynamics validity, which is the
        # only mask for which an end-of-block latent target exists.
        outcome = batch["outcome_valid"]
        target = latent[:, 1:]
        pairs = {
            "one_step_error": ((predicted - target).square(), valid),
            "rollout_error": ((rolled - rollout_target).square(), rollout_valid),
            "final_step_error": (
                (rolled[:, -1:] - rollout_target[:, -1:]).square(),
                rollout_valid[:, -1:],
            ),
            "reward_error": ((reward - block_reward(batch)).square(), outcome),
        }
        # Reward targets, to normalise the readout error by the variance a
        # constant predictor would already achieve.
        target_reward = block_reward(batch)
        for key, (values, mask) in list(pairs.items()) + [
            ("reward_target_sum", (target_reward, outcome)),
            ("reward_target_square", (target_reward.square(), outcome)),
        ]:
            total, count = masked_sum_count(values, mask)
            sums[key] = sums.get(key, 0.0) + float(total)
            counts[key] = counts.get(key, 0.0) + float(count)
        terminated_total, terminated_count = masked_sum_count(
            batch["terminated"].float(), outcome
        )
        sums["terminated_rate"] = sums.get("terminated_rate", 0.0) + float(
            terminated_total
        )
        counts["terminated_rate"] = counts.get("terminated_rate", 0.0) + float(
            terminated_count
        )
        frames = latent[batch["observation_valid"]]
        sigreg_total += float(
            sigreg_loss(model, sigreg, latent, batch["observation_valid"], cfg)
        )
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


def reference_lr_factor(step, total_steps, warmup_steps):
    """One-percent linear warmup followed by cosine decay to zero."""
    if total_steps < 1 or warmup_steps < 1 or warmup_steps >= total_steps:
        raise ValueError("Invalid reference scheduler extent")
    if step < warmup_steps:
        return step / warmup_steps
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))


def run_stage(model, parameters, loader, step_fn, epochs, cfg, device, logger, stage):
    optimizer = torch.optim.AdamW(
        parameters, lr=cfg.train.learning_rate, weight_decay=cfg.train.weight_decay
    )
    reference = cfg.model.get("profile", "legacy_compact") == "lewm_reference"
    if reference:
        total_steps = max(1, epochs * len(loader))
        warmup_steps = max(1, int(0.01 * total_steps))
        if total_steps == 1:
            schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _step: 1.0)
        else:
            schedule = torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                lambda step: reference_lr_factor(step, total_steps, warmup_steps),
            )
    else:
        schedule = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(epochs, 1)
        )
    history = []
    device_type = torch.device(device).type
    use_bf16 = reference and device_type == "cuda"
    for epoch in range(epochs):
        totals, weight = {}, 0.0
        for batch in loader:
            batch = batch.to(device)
            precision = (
                torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                if use_bf16
                else nullcontext()
            )
            with precision:
                losses = step_fn(batch)
            optimizer.zero_grad(set_to_none=True)
            losses["loss"].backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                parameters, cfg.train.gradient_clip
            )
            if not torch.isfinite(grad_norm):
                raise ValueError(f"{stage}: non-finite gradient at epoch {epoch}")
            optimizer.step()
            if reference:
                schedule.step()
            size = batch.batch_size[0]
            for key, value in losses.items():
                totals[key] = totals.get(key, 0.0) + float(value) * size
            totals["grad_norm"] = totals.get("grad_norm", 0.0) + float(grad_norm) * size
            weight += size
        if not reference:
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


def world_model_from_config(
    cfg,
    shapes,
    observation_mean,
    observation_std,
    device,
    action_mean=None,
    action_std=None,
):
    """Construct the recorded model profile without guessing from its weights."""
    profile = cfg.model.get("profile", "legacy_compact")
    common = {
        "kind": cfg.model.kind,
        "obs_dim": shapes["obs_dim"],
        "action_dim": shapes["action_dim"],
        "agents": shapes["agents"],
        "dim": cfg.model.dim,
        "hidden_dim": cfg.model.hidden_dim,
        "conditioner_budget": cfg.model.conditioner_budget,
        "depth": cfg.model.depth,
        "heads": cfg.model.heads,
        "dim_head": cfg.model.dim_head,
        "mlp_dim": cfg.model.mlp_dim,
        "dropout": cfg.model.dropout,
        "obs_mean": observation_mean,
        "obs_std": observation_std,
        # Absent from older checkpoint configs, which therefore rebuild unchanged.
        "encoder_hidden_dim": cfg.model.get("encoder_hidden_dim"),
        "encoder_depth": cfg.model.get("encoder_depth", 1),
    }
    if profile == "legacy_compact":
        model = MultiAgentWorldModel(frames=shapes["frames"], **common)
    elif profile == "lewm_reference":
        model = ReferenceMultiAgentWorldModel(
            history_size=cfg.model.history_size,
            projector_hidden_dim=cfg.model.projector_hidden_dim,
            action_mean=action_mean,
            action_std=action_std,
            **common,
        )
    else:
        raise ValueError(f"Unknown world-model profile: {profile}")
    return model.to(device)


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
    window = cfg.train.get("reference_window", "first")
    if window not in ("first", "random"):
        raise ValueError(f"train.reference_window must be first or random, not {window}")
    # A random window trains on every frame, so it must be normalized on every
    # frame; the first-window default keeps its historical statistics.
    reference_frames = (
        cfg.model.history_size + 1
        if cfg.model.get("profile", "legacy_compact") == "lewm_reference"
        and window == "first"
        else None
    )
    mean, std = observation_statistics(train_loader, reference_frames)
    reference = cfg.model.get("profile", "legacy_compact") == "lewm_reference"
    if reference:
        action_mean, action_std = action_statistics(
            train_loader,
            max_blocks=cfg.model.history_size if window == "first" else None,
            action_block=cfg.data.action_block,
        )
    else:
        action_mean = action_std = None

    sample = next(iter(train_loader))
    frames, agents, obs_dim = sample["observation"].shape[1:]
    action_dim = sample["action"].shape[-1]

    shapes = {
        "agents": agents,
        "obs_dim": obs_dim,
        "action_dim": action_dim,
        "frames": frames,
    }
    model = world_model_from_config(
        cfg,
        shapes,
        mean,
        std,
        device,
        action_mean=action_mean,
        action_std=action_std,
    )
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
    if model.profile != "legacy_compact":
        algorithm_name = f"{model.profile}_{algorithm_name}"
    # The Stage 2 input condition is a third axis with no slot in the schema, so
    # it joins the algorithm name exactly as the data regime does. Only when it
    # is not the default, so the 192 already-logged baseline runs keep their
    # identity and stay poolable with new ones.
    if cfg.data.state_input != "observation":
        algorithm_name = f"{algorithm_name}_{cfg.data.state_input}"
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
                    "profile": model.profile,
                    "regime": cfg.data.regime,
                    "state_input": cfg.data.state_input,
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
    checkpoint_payload = {
        "kind": cfg.model.kind,
        "profile": model.profile,
        "state_dict": model.state_dict(),
        "config": OmegaConf.to_container(cfg, resolve=True),
        "observation_mean": mean,
        "observation_std": std,
        "shapes": shapes,
    }
    if reference:
        checkpoint_payload.update(
            action_mean=action_mean,
            action_std=action_std,
            action_normalization="train_valid_sample_std",
        )
    torch.save(checkpoint_payload, checkpoint)
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
    recorded_profile = checkpoint.get("profile", "legacy_compact")
    configured_profile = cfg.model.get("profile", "legacy_compact")
    if configured_profile != recorded_profile:
        raise ValueError(
            "Checkpoint model profile disagrees with its recorded config: "
            f"{recorded_profile} != {configured_profile}"
        )
    action_mean = checkpoint.get("action_mean")
    action_std = checkpoint.get("action_std")
    if recorded_profile == "lewm_reference":
        if action_mean is None or action_std is None:
            raise ValueError(
                "lewm_reference checkpoint is missing fitted action statistics"
            )
        state = checkpoint["state_dict"]
        if (
            "action_mean" not in state
            or "action_std" not in state
            or not torch.equal(state["action_mean"], action_mean)
            or not torch.equal(state["action_std"], action_std)
        ):
            raise ValueError(
                "lewm_reference checkpoint action statistics disagree with state_dict"
            )
    model = world_model_from_config(
        cfg,
        shapes,
        checkpoint["observation_mean"],
        checkpoint["observation_std"],
        device,
        action_mean=action_mean,
        action_std=action_std,
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


def verify_reload(checkpoint_path, sample: TensorDict, device):
    """A reloaded checkpoint must reproduce predictions exactly, not approximately."""
    original = load_model(checkpoint_path, device)
    reloaded = load_model(checkpoint_path, device)
    batch = sample.to(device)
    batch = reference_training_view(original, batch)
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
