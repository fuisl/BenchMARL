# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the license in the LICENSE file in the root directory.
"""Decode latent world-model predictions into agent positions.

This is the physical counterpart to ``horizon_rollout``.  It separates three
errors on the episode-disjoint test split:

* probe: simulator frame -> encoder -> position probe;
* teacher: true current latent + action -> predicted next latent -> probe;
* rollout: initial latent + actions -> recursive predicted latents -> probe.

The probe is shared across agent identities but local to one agent's latent. It
is fitted on train roots, selected on validation roots, and never sees test
roots.  ``evaluate`` handles one input/regime group so six processes can share
one MIG allocation; ``merge`` produces the aggregate tables and figures.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from examples.world_model.dataset import OfflineSequences
from examples.world_model.horizon_rollout import windowed_rollout
from examples.world_model.physical_response import (
    MLP_DECAYS,
    RIDGE_STRENGTHS,
    apply_probe,
    fit_mlp,
    ridge,
)
from examples.world_model.train import load_model


KINDS = ("independent", "joint", "relational")
COLOURS = {
    "independent": "#777777",
    "joint": "#2d6fb3",
    "relational": "#c8452e",
}
AGENT_COLOURS = ("#1b7f4b", "#7b3fa0", "#1f77b4", "#8c564b")


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def stack_dataset(dataset):
    """One deterministic full-split batch."""
    return next(
        iter(
            DataLoader(
                dataset,
                batch_size=len(dataset),
                shuffle=False,
                collate_fn=torch.stack,
            )
        )
    )


@torch.no_grad()
def probe_examples(model, batch, device, source="true"):
    """Return valid per-agent latent/position pairs on CPU in float64.

    ``true`` probes the encoder representation. ``predicted`` matches the
    planning readout contract: teacher-forced predicted latents are paired with
    their next physical positions.
    """
    latent = model.encode(batch["observation"].to(device))
    if source == "true":
        positions = batch["agent_state"][..., :2].double()
        valid = batch["observation_valid"]
    elif source == "predicted":
        latent = model.predict(latent[:, :-1], batch["action"].to(device))
        positions = batch["agent_state"][:, 1:, :, :2].double()
        valid = batch["valid"]
    else:
        raise ValueError("source must be true or predicted")
    latent = latent.cpu().double()
    latent = latent[valid]
    positions = positions[valid]
    return latent.reshape(-1, latent.shape[-1]), positions.reshape(-1, 2)


def fit_position_probe(model, train, validation, device, kind="mlp", source="true"):
    """Fit on train roots and select one regularizer on validation roots."""
    train_x, train_y = probe_examples(model, train, device, source)
    val_x, val_y = probe_examples(model, validation, device, source)
    strengths = MLP_DECAYS if kind == "mlp" else RIDGE_STRENGTHS
    candidates = []
    for strength in strengths:
        fitted = (
            fit_mlp(train_x, train_y, strength, device)
            if kind == "mlp"
            else ridge(train_x, train_y, strength)
        )
        prediction = apply_probe(fitted, val_x)
        candidates.append((float((prediction - val_y).square().mean()), strength, fitted))
    error, strength, fitted = min(candidates, key=lambda row: row[0])
    return fitted, {"strength": strength, "validation_coordinate_mse": error}


def decode_agents(probe, latent):
    """Decode ``(..., agents, dim)`` to ``(..., agents, 2)`` on CPU."""
    shape = latent.shape[:-1]
    decoded = apply_probe(probe, latent.reshape(-1, latent.shape[-1]).double())
    return decoded.reshape(*shape, 2)


def position_summary(predicted, truth, baseline, valid):
    """Physical position metrics with an explicitly supplied persistence state."""
    selected = valid.unsqueeze(-1).expand(*valid.shape, truth.shape[-2])
    error = (predicted - truth)[selected]
    motion = (truth - baseline)[selected]
    if error.numel() == 0:
        raise ValueError("No valid physical targets")
    squared = error.square()
    denominator = float(motion.square().sum())
    numerator = float(squared.sum())
    return {
        "positions": int(error.shape[0]),
        "coordinate_rmse": float(squared.mean().sqrt()),
        "euclidean_rmse": float(squared.sum(dim=-1).mean().sqrt()),
        "coordinate_mae": float(error.abs().mean()),
        "relative_mse_to_persistence": numerator / denominator if denominator else None,
        "within_1cm": float((error.norm(dim=-1) <= 0.01).float().mean()),
        "within_5cm": float((error.norm(dim=-1) <= 0.05).float().mean()),
    }


def curve_summary(predicted, truth, baseline, valid):
    return [
        position_summary(
            predicted[:, step : step + 1],
            truth[:, step : step + 1],
            baseline[:, step : step + 1],
            valid[:, step : step + 1],
        )
        for step in range(truth.shape[1])
    ]


@torch.no_grad()
def evaluate_model(model, probe, predicted_probe, test, device):
    observation = test["observation"].to(device)
    action = test["action"].to(device)
    latent = model.encode(observation)
    truth_latent = latent[:, 1:]
    teacher_latent = model.predict(latent[:, :-1], action)
    rollout_latent = windowed_rollout(model, latent[:, :1], action)

    truth = test["agent_state"][:, 1:, :, :2].double()
    current = test["agent_state"][:, :-1, :, :2].double()
    start = test["agent_state"][:, :1, :, :2].double().expand_as(truth)
    valid = test["valid"]
    decoded = {
        "probe": decode_agents(probe, truth_latent.cpu()),
        "teacher": decode_agents(probe, teacher_latent.cpu()),
        "rollout": decode_agents(probe, rollout_latent.cpu()),
        "adapted_teacher": decode_agents(predicted_probe, teacher_latent.cpu()),
        "adapted_rollout": decode_agents(predicted_probe, rollout_latent.cpu()),
    }
    result = {
        "probe_overall": position_summary(decoded["probe"], truth, current, valid),
        "teacher_overall": position_summary(decoded["teacher"], truth, current, valid),
        "rollout_overall": position_summary(decoded["rollout"], truth, start, valid),
        "adapted_teacher_overall": position_summary(
            decoded["adapted_teacher"], truth, current, valid
        ),
        "adapted_rollout_overall": position_summary(
            decoded["adapted_rollout"], truth, start, valid
        ),
        "probe_curve": curve_summary(decoded["probe"], truth, current, valid),
        "teacher_curve": curve_summary(decoded["teacher"], truth, current, valid),
        "rollout_curve": curve_summary(decoded["rollout"], truth, start, valid),
        "adapted_teacher_curve": curve_summary(
            decoded["adapted_teacher"], truth, current, valid
        ),
        "adapted_rollout_curve": curve_summary(
            decoded["adapted_rollout"], truth, start, valid
        ),
        # This removes simulator/probe reconstruction from the comparison. It
        # is not physical error by itself, but localises error after decoding.
        "decoded_transition_overall": position_summary(
            decoded["teacher"], decoded["probe"], current, valid
        ),
        "decoded_rollout_curve": curve_summary(
            decoded["rollout"], decoded["probe"], start, valid
        ),
    }
    return result, decoded


def checkpoint_rows(runs, state_input, regime):
    rows = []
    for config_path in sorted(Path(runs).glob("[0-9]*/resolved_config.yaml")):
        run = config_path.parent
        if not (run / "model.pt").exists():
            continue
        config = yaml.safe_load(config_path.read_text())
        if (
            config["data"].get("state_input", "observation") == state_input
            and config["data"]["regime"] == regime
        ):
            rows.append((run, config))
    return rows


def direct_results(root):
    values = {}
    if root is None:
        return values
    for path in Path(root).rglob("summary.json"):
        summary = json.loads(path.read_text())
        for run in summary.get("runs", []):
            key = (run["state_input"], run["regime"], run["kind"], run["seed"])
            values[key] = run["test"]["coordinate_rmse"]
    return values


def render_filmstrip(batch, decoded, state_input, regime, seed, output):
    """Render the most-moving complete test snippet for three pipelines."""
    valid = batch["valid"].all(dim=1)
    position = batch["agent_state"][..., :2].double()
    travel = (position[:, 1:] - position[:, :-1]).norm(dim=-1).sum(dim=(1, 2))
    travel[~valid] = -1
    episode = int(travel.argmax())
    truth = position[episode].numpy()
    rows = [("simulator", truth, None, None)]
    for kind in KINDS:
        item = decoded[kind]
        probe = torch.cat([position[episode, :1], item["probe"][episode]], dim=0).numpy()
        rollout = torch.cat(
            [position[episode, :1], item["adapted_rollout"][episode]], dim=0
        ).numpy()
        rows.append((kind, rollout, truth, probe))

    all_xy = np.concatenate([row[1].reshape(-1, 2) for row in rows])
    pad = 0.15 * max(np.ptp(all_xy[:, 0]), np.ptp(all_xy[:, 1]), 1e-3)
    xlim = (all_xy[:, 0].min() - pad, all_xy[:, 0].max() + pad)
    ylim = (all_xy[:, 1].min() - pad, all_xy[:, 1].max() + pad)
    columns = truth.shape[0]
    figure, axes = plt.subplots(
        len(rows), columns, figsize=(1.45 * columns + 1.7, 1.55 * len(rows) + 1.0),
        squeeze=False,
    )
    for row_index, (label, frames, ghost, probe) in enumerate(rows):
        for step in range(columns):
            axis = axes[row_index, step]
            if ghost is not None:
                axis.scatter(
                    ghost[step, :, 0], ghost[step, :, 1], s=52,
                    facecolor="none", edgecolor="0.65", linewidth=1.2,
                    label="simulator" if step == 0 else None,
                )
                axis.scatter(
                    probe[step, :, 0], probe[step, :, 1], s=28, marker="x",
                    c="0.35", linewidth=1.0,
                    label="probe on truth" if step == 0 else None,
                )
            for agent in range(frames.shape[1]):
                axis.scatter(
                    frames[step, agent, 0], frames[step, agent, 1], s=48,
                    color=AGENT_COLOURS[agent], edgecolor="white", linewidth=0.7,
                    label=f"agent {agent}" if step == 0 else None,
                )
            if ghost is not None and step:
                error = np.linalg.norm(frames[step] - ghost[step], axis=-1).mean()
                axis.set_xlabel(f"{error:.3f} m", fontsize=7, color="0.35")
            axis.set_xlim(xlim)
            axis.set_ylim(ylim)
            axis.set_aspect("equal", adjustable="box")
            axis.set_xticks([])
            axis.set_yticks([])
            for spine in axis.spines.values():
                spine.set_color("0.75")
            if row_index == 0:
                axis.set_title("given" if step == 0 else f"h={step}", fontsize=8)
            if step == 0:
                axis.set_ylabel(
                    label, fontsize=8.5, fontweight="bold",
                    color="black" if label == "simulator" else COLOURS[label],
                )
    handles, labels = axes[-1, 0].get_legend_handles_labels()
    figure.legend(handles, labels, ncol=len(labels), loc="lower center", frameon=False)
    figure.suptitle(
        f"Buzz Wire latent imagination: {state_input}, {regime}, seed {seed}\n"
        "coloured = predicted-latent-head rollout; grey circle = simulator; "
        "x = true-latent probe; "
        "cell labels = mean agent-position error",
        fontsize=10,
    )
    figure.tight_layout(rect=(0, 0.07, 1, 0.90))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=200)
    figure.savefig(output.with_suffix(".pdf"))
    plt.close(figure)


def evaluate_group(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    datasets = {
        split: OfflineSequences(
            args.data,
            args.regime,
            split,
            action_block=args.action_block,
            state_input=args.state_input,
            history_frames=args.history_frames,
        )
        for split in ("train", "validation", "test")
    }
    batches = {name: stack_dataset(data) for name, data in datasets.items()}
    rows = checkpoint_rows(args.runs, args.state_input, args.regime)
    if args.max_runs is not None:
        rows = rows[: args.max_runs]
    if not rows:
        raise ValueError(f"No checkpoints for {args.state_input}/{args.regime}")
    direct = direct_results(args.direct_results)
    records, render = [], {}
    for index, (run, config) in enumerate(rows, 1):
        kind, seed = config["model"]["kind"], config["seed"]
        print(
            f"[{index:02d}/{len(rows)}] {args.state_input} {args.regime} "
            f"{kind} seed {seed}",
            flush=True,
        )
        model = load_model(run / "model.pt", args.device)
        probe, probe_selection = fit_position_probe(
            model, batches["train"], batches["validation"], args.device,
            args.probe, "true"
        )
        predicted_probe, predicted_probe_selection = fit_position_probe(
            model, batches["train"], batches["validation"], args.device,
            args.probe, "predicted"
        )
        metrics, decoded = evaluate_model(
            model, probe, predicted_probe, batches["test"], args.device
        )
        key = (args.state_input, args.regime, kind, seed)
        records.append(
            {
                "run": str(run),
                "state_input": args.state_input,
                "regime": args.regime,
                "kind": kind,
                "seed": seed,
                "probe": args.probe,
                "probe_selection": probe_selection,
                "predicted_probe_selection": predicted_probe_selection,
                "direct_coordinate_rmse": direct.get(key),
                **metrics,
            }
        )
        if seed == args.render_seed:
            render[kind] = decoded
        del model, probe, predicted_probe
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if set(render) == set(KINDS):
        render_filmstrip(
            batches["test"], render, args.state_input, args.regime,
            args.render_seed, output / "filmstrip.png",
        )
    summary = {
        "question": "Can the latent world model recover agent positions one-step and recursively?",
        "state_input": args.state_input,
        "regime": args.regime,
        "probe": args.probe,
        "train_examples": len(datasets["train"]),
        "validation_examples": len(datasets["validation"]),
        "test_examples": len(datasets["test"]),
        "test_roots": int(batches["test"]["episode_id"].unique().numel()),
        "records": records,
    }
    write_json(output / "summary.json", summary)
    print(f"wrote {output / 'summary.json'}", flush=True)


def mean_curve(rows, key):
    horizons = len(rows[0][key])
    return [
        float(np.mean([row[key][step]["coordinate_rmse"] for row in rows]))
        for step in range(horizons)
    ]


def merge_results(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for path in args.summaries:
        records.extend(json.loads(Path(path).read_text())["records"])
    expected = 3 * 2 * 3 * 8
    if len(records) != expected:
        raise ValueError(f"Expected {expected} records, found {len(records)}")
    groups = defaultdict(list)
    for row in records:
        groups[(row["state_input"], row["regime"], row["kind"])].append(row)

    aggregate = []
    print(
        f"{'input':12s} {'regime':12s} {'kind':12s} {'probe':>9s} "
        f"{'aligned':>9s} {'adapted':>9s} {'direct':>9s} {'adapt-h5':>10s} "
        f"{'adapt/direct':>13s}"
    )
    for state_input in ("observation", "history", "physical"):
        for regime in ("correlated", "independent"):
            for kind in KINDS:
                rows = groups[(state_input, regime, kind)]
                probe = float(np.mean([r["probe_overall"]["coordinate_rmse"] for r in rows]))
                teacher = float(np.mean([r["teacher_overall"]["coordinate_rmse"] for r in rows]))
                adapted = float(np.mean([
                    r["adapted_teacher_overall"]["coordinate_rmse"] for r in rows
                ]))
                direct_values = [r["direct_coordinate_rmse"] for r in rows]
                direct = float(np.mean(direct_values)) if all(v is not None for v in direct_values) else None
                rollout = mean_curve(rows, "rollout_curve")
                adapted_rollout = mean_curve(rows, "adapted_rollout_curve")
                probe_curve = mean_curve(rows, "probe_curve")
                teacher_curve = mean_curve(rows, "teacher_curve")
                row = {
                    "state_input": state_input,
                    "regime": regime,
                    "kind": kind,
                    "seeds": len(rows),
                    "probe_coordinate_rmse": probe,
                    "teacher_coordinate_rmse": teacher,
                    "adapted_teacher_coordinate_rmse": adapted,
                    "direct_coordinate_rmse": direct,
                    "latent_to_direct_rmse_ratio": teacher / direct if direct else None,
                    "adapted_to_direct_rmse_ratio": adapted / direct if direct else None,
                    "rollout_curve": rollout,
                    "adapted_rollout_curve": adapted_rollout,
                    "probe_curve": probe_curve,
                    "teacher_curve": teacher_curve,
                }
                aggregate.append(row)
                direct_text = "--" if direct is None else f"{direct:.5f}"
                ratio_text = "--" if direct is None else f"{adapted / direct:.2f}x"
                print(
                    f"{state_input:12s} {regime:12s} {kind:12s} {probe:9.5f} "
                    f"{teacher:9.5f} {adapted:9.5f} {direct_text:>9s} "
                    f"{adapted_rollout[-1]:10.5f} "
                    f"{ratio_text:>14s}"
                )

    write_json(output / "aggregate.json", {"records": records, "aggregate": aggregate})
    render_aggregate(aggregate, output)
    print(f"wrote aggregate data and figures under {output}")


def render_aggregate(aggregate, output):
    index = {(r["state_input"], r["regime"], r["kind"]): r for r in aggregate}
    figure, axes = plt.subplots(3, 2, figsize=(10.5, 11), sharex=True)
    for row, state_input in enumerate(("observation", "history", "physical")):
        for column, regime in enumerate(("correlated", "independent")):
            axis = axes[row, column]
            for kind in KINDS:
                item = index[(state_input, regime, kind)]
                horizon = np.arange(1, len(item["rollout_curve"]) + 1)
                axis.plot(
                    horizon, item["adapted_rollout_curve"], marker="o",
                    color=COLOURS[kind],
                    linewidth=1.8, label=kind,
                )
                axis.plot(
                    horizon, item["rollout_curve"], linestyle="--",
                    color=COLOURS[kind], linewidth=1.0, alpha=0.7,
                )
                axis.plot(
                    horizon, item["probe_curve"], linestyle=":", color=COLOURS[kind],
                    linewidth=1.2, alpha=0.8,
                )
            axis.set_title(f"{state_input} / {regime}")
            axis.set_ylabel("coordinate RMSE (m)")
            axis.grid(alpha=0.25)
            axis.set_xticks(horizon)
    for axis in axes[-1]:
        axis.set_xlabel("recursive rollout horizon (5 simulator steps per block)")
    architecture = [
        plt.Line2D([], [], color=COLOURS[k], marker="o", label=k) for k in KINDS
    ]
    styles = [
        plt.Line2D([], [], color="black", linestyle="-", label="predicted-latent head"),
        plt.Line2D([], [], color="black", linestyle="--", label="true-latent head"),
        plt.Line2D([], [], color="black", linestyle=":", label="probe on true latent"),
    ]
    figure.legend(architecture + styles, [h.get_label() for h in architecture + styles],
                  ncol=6, loc="lower center", frameon=False)
    figure.suptitle("Latent world-model position error: test roots, 8 training seeds")
    figure.tight_layout(rect=(0, 0.05, 1, 0.97))
    figure.savefig(output / "rollout_curves.png", dpi=200)
    figure.savefig(output / "rollout_curves.pdf")
    plt.close(figure)

    figure, axes = plt.subplots(3, 2, figsize=(11, 10.5), sharey=False)
    width = 0.20
    for row, state_input in enumerate(("observation", "history", "physical")):
        for column, regime in enumerate(("correlated", "independent")):
            axis = axes[row, column]
            items = [index[(state_input, regime, kind)] for kind in KINDS]
            x = np.arange(len(KINDS))
            axis.bar(x - 1.5 * width, [r["probe_coordinate_rmse"] for r in items], width,
                     color="#d9d9d9", label="probe floor")
            axis.bar(x - 0.5 * width, [r["teacher_coordinate_rmse"] for r in items], width,
                     color="#5b8db8", label="true-latent head")
            axis.bar(x + 0.5 * width,
                     [r["adapted_teacher_coordinate_rmse"] for r in items], width,
                     color="#5aae72", label="predicted-latent head")
            if all(r["direct_coordinate_rmse"] is not None for r in items):
                axis.bar(x + 1.5 * width,
                         [r["direct_coordinate_rmse"] for r in items], width,
                         color="#d97945", label="direct dx,dy")
            axis.set_xticks(x, KINDS, rotation=12)
            axis.set_title(f"{state_input} / {regime}")
            axis.set_ylabel("pooled one-block coordinate RMSE (m)")
            axis.grid(axis="y", alpha=0.25)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, ncol=4, loc="lower center", frameon=False)
    figure.suptitle("One-block agent position: latent pipeline versus direct diagnostic")
    figure.tight_layout(rect=(0, 0.05, 1, 0.97))
    figure.savefig(output / "one_step_comparison.png", dpi=200)
    figure.savefig(output / "one_step_comparison.pdf")
    plt.close(figure)


def parser():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="command", required=True)
    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("runs", type=Path)
    evaluate.add_argument("--data", type=Path, required=True)
    evaluate.add_argument("--direct-results", type=Path)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--state-input", choices=("observation", "history", "physical"), required=True)
    evaluate.add_argument("--regime", choices=("correlated", "independent"), required=True)
    evaluate.add_argument("--probe", choices=("linear", "mlp"), default="mlp")
    evaluate.add_argument("--device", default="cpu")
    evaluate.add_argument("--action-block", type=int, default=5)
    evaluate.add_argument("--history-frames", type=int, default=3)
    evaluate.add_argument("--render-seed", type=int, default=4100)
    evaluate.add_argument("--max-runs", type=int)
    merge = sub.add_parser("merge")
    merge.add_argument("summaries", nargs="+", type=Path)
    merge.add_argument("--output", type=Path, required=True)
    return ap


def main():
    args = parser().parse_args()
    if args.command == "evaluate":
        evaluate_group(args)
    else:
        merge_results(args)


if __name__ == "__main__":
    main()
