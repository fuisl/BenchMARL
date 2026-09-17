# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the license in the LICENSE file in the root directory.
"""Validate one-block physical position prediction before latent planning.

Latent rollout MSE cannot answer whether an imagined agent is in the right
place: each checkpoint learns its own coordinates, and the visualization adds a
separately fitted latent-to-position probe. This diagnostic trains a deliberately
small world model directly against simulator ``(x, y)`` displacement.

One transition function is shared by every agent. The independent, joint and
relational variants differ only in which other-agent observations/actions can
reach agent ``i``'s prediction. Evaluation is on episode-disjoint test roots and
reports each agent separately, together with a persistence baseline that
predicts no movement.

One model step is one dataset action block (five primitive steps in the current
banks), not one primitive simulator step.

Example:
    python -m examples.world_model.agent_position \
        --data outputs/buzz_wire_1196/data \
        --output outputs/agent_position_pilot \
        --regimes correlated independent \
        --state-inputs observation \
        --device cpu
"""

import argparse
import json
import math
import subprocess
from pathlib import Path

import torch

from examples.world_model.dataset import OfflineSequences, STATE_INPUTS
from examples.world_model.models import SharedAgentPositionModel, masked_mean
from torch.utils.data import DataLoader


KINDS = ("independent", "joint", "relational")


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def loader(dataset, batch_size, shuffle, seed):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        collate_fn=torch.stack,
        generator=torch.Generator().manual_seed(seed),
    )


def transitions(batch):
    """Inputs and physical displacement at consecutive block boundaries."""
    observation = batch["observation"][:, :-1]
    action = batch["action"]
    state = batch["agent_state"]
    delta = state[:, 1:, :, :2] - state[:, :-1, :, :2]
    return observation, action, delta, batch["valid"]


def feature_statistics(data, batch_size):
    """Fit normalizers only on valid training transitions."""
    observations, actions, deltas = [], [], []
    for batch in loader(data, batch_size, False, 0):
        observation, action, delta, valid = transitions(batch)
        observations.append(observation[valid].reshape(-1, observation.size(-1)))
        actions.append(action[valid].reshape(-1, action.size(-1)))
        deltas.append(delta[valid].reshape(-1, 2))

    def summarize(parts):
        values = torch.cat(parts).double()
        mean = values.mean(dim=0)
        std = values.std(dim=0, unbiased=False).clamp_min(1e-6)
        return mean.float(), std.float()

    obs_mean, obs_std = summarize(observations)
    action_mean, action_std = summarize(actions)
    delta_mean, delta_std = summarize(deltas)
    return {
        "obs_mean": obs_mean,
        "obs_std": obs_std,
        "action_mean": action_mean,
        "action_std": action_std,
        "delta_mean": delta_mean,
        "delta_std": delta_std,
    }


@torch.no_grad()
def validation_loss(model, data_loader, device):
    model.eval()
    total, count = 0.0, 0
    for batch in data_loader:
        batch = batch.to(device)
        observation, action, target, valid = transitions(batch)
        predicted = model(observation, action)
        error = ((predicted - target) / model.delta_std).square()
        expanded = valid.unsqueeze(-1).unsqueeze(-1).expand_as(error)
        total += float(error[expanded].sum())
        count += int(expanded.sum())
    if count == 0:
        raise ValueError("Validation split has no valid transitions")
    return total / count


def train_model(model, train_data, validation_data, args):
    """Train with validation-selected early stopping and restore the best epoch."""
    train_loader = loader(train_data, args.batch_size, True, args.seed)
    validation_loader = loader(
        validation_data, args.batch_size, False, args.seed
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    best_loss, best_epoch, best_state = math.inf, -1, None
    stale, history = 0, []

    for epoch in range(args.epochs):
        model.train()
        train_total, examples = 0.0, 0
        for batch in train_loader:
            batch = batch.to(args.device)
            observation, action, target, valid = transitions(batch)
            predicted = model(observation, action)
            error = ((predicted - target) / model.delta_std).square()
            loss = masked_mean(error, valid)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), args.gradient_clip
            )
            if not torch.isfinite(grad_norm):
                raise ValueError(f"Non-finite gradient at epoch {epoch}")
            optimizer.step()
            train_total += float(loss) * batch.batch_size[0]
            examples += batch.batch_size[0]

        score = validation_loss(model, validation_loader, args.device)
        history.append(
            {
                "epoch": epoch,
                "train_standardized_mse": train_total / examples,
                "validation_standardized_mse": score,
            }
        )
        if score < best_loss - args.min_delta:
            best_loss = score
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                break

    if best_state is None:
        raise ValueError("Training produced no finite validation checkpoint")
    model.load_state_dict(best_state)
    return {
        "best_epoch": best_epoch,
        "best_validation_standardized_mse": best_loss,
        "epochs_ran": len(history),
        "history": history,
    }


def error_summary(error, target):
    """Position error and two simple displacement baselines."""
    squared = error.square()
    persistence = target.square()
    error_sum = float(squared.sum())
    persistence_sum = float(persistence.sum())
    samples = error.shape[0]
    return {
        "transitions": samples,
        "coordinate_rmse": float(squared.mean().sqrt()),
        "euclidean_rmse": float(squared.sum(dim=-1).mean().sqrt()),
        "coordinate_mae": float(error.abs().mean()),
        "axis_rmse": squared.mean(dim=0).sqrt().tolist(),
        "persistence_coordinate_rmse": float(persistence.mean().sqrt()),
        "relative_mse_to_persistence": (
            error_sum / persistence_sum if persistence_sum > 0 else None
        ),
        "improvement_over_persistence_percent": (
            100.0 * (1.0 - error_sum / persistence_sum)
            if persistence_sum > 0
            else None
        ),
        "within_1cm": float((error.norm(dim=-1) <= 0.01).float().mean()),
        "within_5cm": float((error.norm(dim=-1) <= 0.05).float().mean()),
    }


@torch.no_grad()
def evaluate(model, data, args):
    model.eval()
    predictions, targets, episodes = [], [], []
    for batch in loader(data, args.batch_size, False, args.seed):
        batch = batch.to(args.device)
        observation, action, target, valid = transitions(batch)
        predicted = model(observation, action)
        predictions.append(predicted[valid].cpu())
        targets.append(target[valid].cpu())
        episode = batch["episode_id"].unsqueeze(1).expand_as(valid)
        episodes.append(episode[valid].cpu())

    predicted = torch.cat(predictions)
    target = torch.cat(targets)
    episode = torch.cat(episodes)
    error = predicted - target
    agents = target.shape[1]
    result = error_summary(error.reshape(-1, 2), target.reshape(-1, 2))
    result["root_episodes"] = int(episode.unique().numel())
    result["per_agent"] = {
        str(agent): error_summary(error[:, agent], target[:, agent])
        for agent in range(agents)
    }
    result["per_root"] = {}
    for root in episode.unique(sorted=True):
        selected = episode == root
        root_error = error[selected].reshape(-1, 2)
        root_target = target[selected].reshape(-1, 2)
        result["per_root"][str(int(root))] = error_summary(
            root_error, root_target
        )

    mean_error = target - model.delta_mean.cpu()
    mean_squared = float(mean_error.square().sum())
    persistence_squared = float(target.square().sum())
    result["train_mean_delta_relative_mse"] = (
        mean_squared / persistence_squared
        if persistence_squared > 0
        else None
    )
    return result


def model_checkpoint(model, args, regime, state_input, stats, metrics):
    return {
        "kind": model.kind,
        "state_dict": model.state_dict(),
        "config": vars(args),
        "regime": regime,
        "state_input": state_input,
        "statistics": stats,
        "test_metrics": metrics,
        "shapes": {
            "agents": model.agents,
            "obs_dim": model.obs_mean.numel(),
            "action_dim": model.action_mean.numel(),
        },
    }


def serializable(value):
    if torch.is_tensor(value):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [serializable(item) for item in value]
    return value


def run(args):
    manifest = json.loads((args.data / "manifest.json").read_text())
    action_block = manifest["action_block"]
    args.output.mkdir(parents=True, exist_ok=True)
    summary_path = args.output / "summary.json"
    if summary_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite completed result: {summary_path}"
        )

    result = {
        "question": (
            "Can one transition function shared across agents predict each "
            "agent's next physical position?"
        ),
        "target": "agent (x, y) displacement over one action block",
        "task_name": manifest["task_name"],
        "action_block": action_block,
        "config": serializable(vars(args)),
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "git_status": subprocess.check_output(
            ["git", "status", "--short"], text=True
        ),
        "runs": [],
    }

    for state_input in args.state_inputs:
        for regime in args.regimes:
            datasets = {
                split: OfflineSequences(
                    args.data,
                    regime,
                    split,
                    action_block=action_block,
                    state_input=state_input,
                    history_frames=args.history_frames,
                )
                for split in ("train", "validation", "test")
            }
            stats = feature_statistics(datasets["train"], args.batch_size)
            sample = datasets["train"][0]
            agents = sample["observation"].shape[-2]
            obs_dim = sample["observation"].shape[-1]
            action_dim = sample["action"].shape[-1]

            for kind in args.kinds:
                torch.manual_seed(args.seed)
                model = SharedAgentPositionModel(
                    kind,
                    obs_dim,
                    action_dim,
                    agents,
                    hidden_dim=args.hidden_dim,
                    conditioner_budget=args.conditioner_budget,
                    **stats,
                ).to(args.device)
                training = train_model(
                    model, datasets["train"], datasets["validation"], args
                )
                metrics = evaluate(model, datasets["test"], args)
                directory = args.output / state_input / regime / kind
                directory.mkdir(parents=True, exist_ok=True)
                checkpoint = directory / "model.pt"
                torch.save(
                    model_checkpoint(
                        model, args, regime, state_input, stats, metrics
                    ),
                    checkpoint,
                )
                write_json(directory / "history.json", training["history"])
                record = {
                    "state_input": state_input,
                    "regime": regime,
                    "kind": kind,
                    "seed": args.seed,
                    "train_examples": len(datasets["train"]),
                    "validation_examples": len(datasets["validation"]),
                    "test_examples": len(datasets["test"]),
                    "parameters": sum(
                        parameter.numel() for parameter in model.parameters()
                    ),
                    "conditioner_parameters": sum(
                        parameter.numel()
                        for parameter in model.conditioner.parameters()
                    ),
                    "conditioner_hidden": model.conditioner_hidden,
                    **{
                        key: value
                        for key, value in training.items()
                        if key != "history"
                    },
                    "test": metrics,
                    "checkpoint": str(checkpoint),
                }
                result["runs"].append(record)
                print(
                    f"{state_input:11s} {regime:11s} {kind:12s} "
                    f"RMSE={metrics['coordinate_rmse']:.5f} "
                    f"relative={metrics['relative_mse_to_persistence']:.3f} "
                    f"roots={metrics['root_episodes']}",
                    flush=True,
                )

    write_json(summary_path, serializable(result))
    print(f"wrote {summary_path}")
    return result


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--regimes",
        nargs="+",
        choices=("correlated", "independent"),
        default=("correlated", "independent"),
    )
    parser.add_argument(
        "--state-inputs",
        nargs="+",
        choices=STATE_INPUTS,
        default=("observation",),
    )
    parser.add_argument("--kinds", nargs="+", choices=KINDS, default=KINDS)
    parser.add_argument("--history-frames", type=int, default=3)
    parser.add_argument("--seed", type=int, default=4100)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--min-delta", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--conditioner-budget", type=int, default=100000)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
