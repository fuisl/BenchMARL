#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#
"""Render the latent rollout error curves written by `horizon_rollout --output`.

Why not BenchMARL's own plotting. `benchmarl.eval_results.Plotting` wraps
marl-eval, and every figure it draws calls `rliable.library.get_interval_estimates`,
which fails in this environment -- arch 8.0 renamed the `random_state` argument
rliable still passes, and arch 7.2 will not import against pandas 3.0. That is
already recorded in `report.py`, and re-checked here before writing this module.
Its figures are also the wrong shape: marl-eval plots normalised return against
*training step*, per algorithm and task, while this is prediction error against
*rollout horizon*. What is reused is the repository's own interval convention,
`metrics.mean_interval`, so the bands here are the same bootstrap every table in
the paper reports.

Two scales matter, and both are honest only if stated:

* **Normalised error.** Each model learns its own latent geometry, so a raw MSE
  is not comparable across predictors. Every curve is divided by *that
  checkpoint's own* latent variance -- the error a constant mean-predictor makes
  in that same space -- so 1.0 means "no better than predicting the mean" and the
  three tasks can share an axis.
* **Paired differences.** Even normalised, the size of a gap between two rows is
  not a physical quantity. The second panel therefore shows the per-seed paired
  difference against the single-agent model, which is a within-seed, within-task
  comparison and is the only ranking the data supports.

Run:
    python -m examples.world_model.plot_horizon outputs/horizon_curves \\
        --output outputs/horizon_curves/rollout_error.png
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless server
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from examples.world_model.metrics import mean_interval

KINDS = ("independent", "joint", "relational")
REGIMES = ("correlated", "independent")
COLOUR = {"independent": "#9a9a9a", "joint": "#2d6fb3", "relational": "#c8452e"}
STYLE = {"correlated": "-", "independent": "--"}
LABEL = {
    "independent": "independent (no cross-agent path)",
    "joint": "joint (fixed order)",
    "relational": "relational (permutation-equivariant)",
}


def load(directory):
    """Every `<name>.json` in `directory`, keyed by name."""
    sets = {}
    for path in sorted(Path(directory).glob("*.json")):
        sets[path.stem] = json.loads(path.read_text())
    if not sets:
        raise ValueError(f"No horizon JSON in {directory}; run horizon_rollout --output")
    return sets


def normalised(record):
    """One checkpoint's curve as a fraction of its own latent variance.

    Dividing by the model's *own* variance is what makes two separately learned
    latent spaces comparable at all: the quantity becomes "how much of the
    variance this model fails to explain", which is dimensionless and bounded
    below by zero, with 1.0 the constant-predictor baseline.
    """
    return np.asarray(record["curve"], dtype=float) / record["latent_variance"]


def curves(data, regime, kind):
    """(seeds, blocks) normalised curves for one cell, ordered by seed."""
    rows = [r for r in data["runs"] if r["regime"] == regime and r["kind"] == kind]
    rows.sort(key=lambda r: r["seed"])
    return np.stack([normalised(r) for r in rows]) if rows else None


def band(values, rng):
    """Bootstrap mean and 95% interval per column, via the project's own helper."""
    mean, low, high = [], [], []
    for column in values.T:
        finite = column[np.isfinite(column)]
        if finite.size == 0:
            mean.append(np.nan), low.append(np.nan), high.append(np.nan)
            continue
        interval = mean_interval(finite, rng)
        mean.append(interval["mean"])
        low.append(interval["ci95"][0])
        high.append(interval["ci95"][1])
    return np.array(mean), np.array(low), np.array(high)


def draw_task(axis, data, rng):
    blocks = np.arange(1, data["blocks"] + 1)
    for regime in REGIMES:
        for kind in KINDS:
            values = curves(data, regime, kind)
            if values is None:
                continue
            mean, low, high = band(values, rng)
            keep = np.isfinite(mean)
            axis.plot(
                blocks[keep], mean[keep],
                color=COLOUR[kind], linestyle=STYLE[regime], linewidth=1.7,
                marker="o" if regime == "correlated" else "^", markersize=3.2,
                label=None,
            )
            axis.fill_between(
                blocks[keep], low[keep], high[keep],
                color=COLOUR[kind], alpha=0.12, linewidth=0,
            )
    axis.axhline(1.0, color="black", linewidth=0.9, linestyle=":", zorder=1)
    trained = data["trained_frames"]
    axis.axvspan(trained, data["blocks"] + 1, color="black", alpha=0.045, zorder=0)
    axis.axvline(trained, color="black", linewidth=0.8, alpha=0.5)
    axis.set_xlim(1, max(b for b, c in zip(blocks, data["episodes_live"]) if c > 0))
    # Log scale because the tasks differ by an order of magnitude: Transport
    # sits at 2-5% of latent variance where Buzz Wire sits at 20-50%. On a
    # linear axis one of them is a flat line against the floor, which hides
    # exactly the result -- that rollout error stays low and grows slowly.
    axis.set_yscale("log")
    axis.set_ylim(0.01, 1.6)
    axis.set_xlabel(f"rollout horizon (blocks of {data['action_block']} steps)")
    axis.grid(alpha=0.25, linewidth=0.5)


def paired(data, kind, rng):
    """Per-seed difference against the single-agent model, at each block."""
    base = curves(data, "correlated", "independent")
    other = curves(data, "correlated", kind)
    if base is None or other is None:
        return None
    return band(other - base, rng)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("directory", type=Path)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--sets", default="transport,buzz_wire,balance")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    sets = load(args.directory)
    names = [n for n in args.sets.split(",") if n in sets]
    if not names:
        raise ValueError(f"None of {args.sets} found in {sorted(sets)}")
    rng = np.random.default_rng(args.seed)

    figure, axes = plt.subplots(
        2, len(names), figsize=(4.3 * len(names), 7.9), squeeze=False,
        gridspec_kw={"height_ratios": [2.1, 1.0]},
    )
    for column, name in enumerate(names):
        data = sets[name]
        top = axes[0][column]
        draw_task(top, data, rng)
        live = data["episodes_live"]
        top.set_title(
            f"{data['task_name'].split('/')[-1].replace('_', ' ').title()}\n"
            f"{live[0]} episodes at h=1, {max(live[4], 0)} at h=5",
            fontsize=10,
        )
        if column == 0:
            top.set_ylabel("latent rollout error / own latent variance")

        bottom = axes[1][column]
        blocks = np.arange(1, data["blocks"] + 1)
        for kind in ("joint", "relational"):
            result = paired(data, kind, rng)
            if result is None:
                continue
            mean, low, high = result
            keep = np.isfinite(mean)
            bottom.plot(blocks[keep], mean[keep], color=COLOUR[kind], linewidth=1.7,
                        marker="o", markersize=3.0,
                        label=LABEL[kind].split(" (")[0] if column == 0 else None)
            bottom.fill_between(blocks[keep], low[keep], high[keep],
                                color=COLOUR[kind], alpha=0.15, linewidth=0)
        bottom.axhline(0.0, color="black", linewidth=0.9)
        bottom.axvline(data["trained_frames"], color="black", linewidth=0.8, alpha=0.5)
        bottom.axvspan(data["trained_frames"], data["blocks"] + 1,
                       color="black", alpha=0.045, zorder=0)
        bottom.set_xlim(top.get_xlim())
        bottom.set_xlabel("rollout horizon (blocks)")
        bottom.grid(alpha=0.25, linewidth=0.5)
        if column == 0:
            bottom.set_ylabel("paired Δ vs independent\n(negative = better)")
            bottom.legend(fontsize=8, frameon=False, loc="best")

    # Two independent visual channels, so two legends rather than one with six
    # combined labels: colour is the predictor, dash pattern is the training
    # data regime. A combined legend needs 3x2 long entries and clips.
    predictors = [
        Line2D([], [], color=COLOUR[k], linewidth=2.0, marker="o", markersize=4,
               label=LABEL[k])
        for k in KINDS
    ]
    regimes = [
        Line2D([], [], color="black", linewidth=1.6, linestyle=STYLE[r],
               marker="o" if r == "correlated" else "^", markersize=4,
               label=f"{r} actions")
        for r in REGIMES
    ]
    figure.legend(handles=predictors, fontsize=8.4, frameon=False, ncol=3,
                  loc="lower center", bbox_to_anchor=(0.5, 0.052), title="predictor",
                  title_fontsize=8.4)
    figure.legend(handles=regimes, fontsize=8.4, frameon=False, ncol=2,
                  loc="lower center", bbox_to_anchor=(0.5, -0.006),
                  title="training data", title_fontsize=8.4)
    figure.suptitle(
        "Latent rollout error against horizon, per predictor and data regime\n"
        "bands: 95% bootstrap over 8 seeds · dotted line: predicting the latent "
        "mean · shaded: beyond the trained context",
        fontsize=10.5,
    )
    figure.tight_layout(rect=(0, 0.145, 1, 0.94))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=200)
    figure.savefig(args.output.with_suffix(".pdf"))
    print(f"wrote {args.output} and {args.output.with_suffix('.pdf')}")

    # The numbers behind the figure, so the plot is never the only record.
    for name in names:
        data = sets[name]
        print(f"\n{data['task_name']} -- normalised by each model's own latent variance")
        print(f"{'regime':12s}{'kind':13s}" + "".join(f"{f'h={h}':>9s}" for h in (1, 3, 5, 10, 15)))
        for regime in REGIMES:
            for kind in KINDS:
                values = curves(data, regime, kind)
                if values is None:
                    continue
                mean = np.nanmean(values, axis=0)
                cells = "".join(
                    f"{mean[h - 1]:9.4f}" if h <= len(mean) and np.isfinite(mean[h - 1])
                    else "       --" for h in (1, 3, 5, 10, 15)
                )
                print(f"{regime:12s}{kind:13s}{cells}")


if __name__ == "__main__":
    main()
