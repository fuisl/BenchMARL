#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#
"""Imagined rollouts beside the simulator that produced them.

The horizon tables say how wrong a rollout is. They cannot say *how* it is
wrong, and a scalar averaged over 128 episodes and 8 seeds hides whether a model
drifts, lags, or invents an interaction that never happens. This draws the
trajectories themselves, which is the qualitative figure the LeWM paper carries
and the one this project has never had.

Our predictor rolls in latent space and there is no decoder, so "imagined
position" needs one. `physical_response` already fits exactly that: a probe from
latent to the simulator's own (pos, vel), with an agent head that reads only its
own agent's latent and a global head for the shared object. It is reused here
unchanged, fitted on train-split anchors no evaluation touches.

That makes every panel a three-way comparison, and all three lines are needed
for the figure to be honest:

  * **simulator** -- ground truth, what actually happened.
  * **probe on TRUE latents** -- the simulator's own states, encoded and decoded
    back. The gap to `simulator` is the PROBE's error and has nothing to do with
    the dynamics. Without this line, a readout that cannot represent position
    reads as a world model that cannot predict it.
  * **probe on IMAGINED latents** -- encode the first frame, roll the model
    forward on the same actions, decode each block. The gap to the line above is
    the model's own rollout error, which is the quantity of interest.

Actions come from `collect.sample_actions` in the regime the checkpoint was
trained on, so the model is asked about the distribution it was fitted on, and
the same action sequence drives the simulator and the imagination.

Run:
    python -m examples.world_model.plot_imagination \\
        outputs/buzz_wire_1196/baselines --data outputs/buzz_wire_1196/data \\
        --device cuda --output outputs/horizon_curves/imagination.png
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless server
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from benchmarl.environments import VmasTask
from examples.world_model.collect import sample_actions
from examples.world_model.horizon_rollout import to_device, windowed_rollout
from examples.world_model.mpc import action_bounds
from examples.world_model.physical_response import apply_probe, fit_probes, MOTION
from examples.world_model.snapshot_restore import (
    agent_observations,
    physical_state,
    restore_state,
    tracked_entities,
)
from examples.world_model.train import load_model
from tensordict import TensorDict

KINDS = ("independent", "joint", "relational")
COLOUR = {"independent": "#9a9a9a", "joint": "#2d6fb3", "relational": "#c8452e"}
OBJECT_COLOUR = "#d98200"


def agent_colours(n):
    """One colour per agent; tasks range from 2 agents (Buzz Wire) to 4."""
    palette = ("#1b7f4b", "#7b3fa0", "#1f77b4", "#8c564b", "#e377c2", "#17becf")
    if n > len(palette):
        raise ValueError(f"No colour for {n} agents; extend the palette")
    return palette[:n]


@torch.no_grad()
def true_trace(task, snapshot, blocks, action_block, regime, seed, device):
    """Observations, actions and PHYSICAL state at every block boundary.

    Deliberately separate from `horizon_rollout.simulator_truth`, which returns
    observations only and whose three-value contract four measurement paths
    unpack positionally. Rendering needs the simulator's own positions as well,
    and widening a function the horizon tables depend on, to serve a figure, is
    the wrong trade. The action sampling and stepping are identical.
    """
    env = task.get_env_fun(snapshot["steps"].numel(), True, 0, device)()
    env.reset()
    restore_state(env, to_device(snapshot, device))
    agents = len(env._env.world.agents)
    action_dim = env.full_action_spec_unbatched["agents", "action"].shape[-1]
    low, high = action_bounds(env)
    batch = env.batch_size[0]

    actions = sample_actions(
        (batch, blocks * action_block, agents, action_dim),
        torch.as_tensor(low, dtype=torch.float32),
        torch.as_tensor(high, dtype=torch.float32),
        regime,
        seed,
    ).to(device)

    td = TensorDict({}, batch_size=[batch], device=device)
    live = ~env._env.done()
    frames = [agent_observations(env).cpu()]
    agent_state = [physical_state(env._env.world.agents).cpu()]
    object_state = [physical_state(tracked_entities(env)).cpu()]
    valid = []
    for block in range(blocks):
        through = live.clone()
        for step in range(action_block):
            td.set(("agents", "action"), actions[:, block * action_block + step])
            td = env.step(td)["next"]
            live = live & ~td["done"].squeeze(-1)
            through = through & live
        frames.append(agent_observations(env).cpu())
        agent_state.append(physical_state(env._env.world.agents).cpu())
        object_state.append(physical_state(tracked_entities(env)).cpu())
        valid.append(through.cpu())
    env.close()
    blocked = actions.reshape(batch, blocks, action_block, agents, action_dim)
    blocked = blocked.permute(0, 1, 3, 2, 4).reshape(batch, blocks, agents, -1).cpu()
    return {
        "frames": torch.stack(frames, dim=1),
        "actions": blocked,
        "agent_state": torch.stack(agent_state, dim=1),
        "object_state": torch.stack(object_state, dim=1),
        "valid": torch.stack(valid, dim=1),
    }


def decode(probes, latent):
    """(B,N,dim) latents -> agent (B,N,4) and object (B,P,4) in raw units."""
    anchors, agents, dim = latent.shape
    per_agent = apply_probe(probes["agent"][0], latent.reshape(-1, dim))
    per_agent = per_agent.reshape(anchors, agents, -1)
    shared = apply_probe(probes["object"][0], latent.reshape(anchors, agents * dim))
    return per_agent, shared.reshape(anchors, -1, per_agent.shape[-1])


@torch.no_grad()
def imagine(model, trace, device):
    """Encode frame 0, roll on the recorded actions, decode nothing yet.

    Returns TRUE and IMAGINED latents at every block boundary after the first:
    (B, blocks, N, dim) each. `windowed_rollout` is the same sliding window the
    horizon tables use, so the imagination shown is the one that was measured --
    it never indexes the positional slot that received no prediction gradient.
    """
    frames = trace["frames"].to(device)
    true_latent = model.encode(frames)[:, 1:].cpu().double()
    start = model.encode(frames[:, :1])
    rolled = windowed_rollout(model, start, trace["actions"].to(device))
    return true_latent, rolled.cpu().double()


def choose_checkpoints(runs, regime, seed):
    """One checkpoint per predictor, all from the same regime and seed."""
    chosen = {}
    for path in sorted(Path(runs).rglob("resolved_config.yaml")):
        run = path.parent
        if not (run / "model.pt").exists():
            continue
        cfg = yaml.safe_load(path.read_text())
        if cfg["data"].get("state_input", "observation") != "observation":
            continue
        if cfg["data"]["regime"] != regime or cfg["seed"] != seed:
            continue
        chosen[cfg["model"]["kind"]] = run
    missing = [k for k in KINDS if k not in chosen]
    if missing:
        raise ValueError(f"No {missing} checkpoint at regime={regime} seed={seed}")
    return chosen


def draw(axis, true_xy, probe_xy, imagined_xy, colour, label):
    """One body: simulator, probe-on-truth, probe-on-imagination."""
    axis.plot(true_xy[:, 0], true_xy[:, 1], "-", color=colour, linewidth=2.0,
              marker="o", markersize=3.0, label=label, zorder=3)
    axis.plot(probe_xy[:, 0], probe_xy[:, 1], ":", color=colour, linewidth=1.2,
              alpha=0.75, zorder=2)
    axis.plot(imagined_xy[:, 0], imagined_xy[:, 1], "--", color=colour,
              linewidth=1.6, marker="^", markersize=3.0, alpha=0.95, zorder=4)
    axis.scatter(true_xy[:1, 0], true_xy[:1, 1], s=46, facecolor="white",
                 edgecolor=colour, linewidth=1.4, zorder=5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", type=Path)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--blocks", type=int, default=8)
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument(
        "--select",
        choices=("movers", "first"),
        default="movers",
        help="which surviving episodes to draw. `movers` takes those with the "
        "largest true displacement, because an episode whose bodies barely "
        "move renders the probe's own error rather than the model's: on Buzz "
        "Wire the ball can travel 0.06m over 8 blocks while the probe "
        "reconstructs position to only 0.1m. `first` restores index order.",
    )
    ap.add_argument("--regime", default="correlated")
    ap.add_argument("--seed", type=int, default=4100)
    ap.add_argument("--action-seed", type=int, default=9100)
    ap.add_argument("--probe", choices=("linear", "mlp"), default="mlp")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    manifest = yaml.safe_load((args.data / "manifest.json").read_text())
    task = VmasTask[manifest["task_name"].split("/")[-1].upper()].get_from_yaml()
    block = manifest["action_block"]
    initial = torch.load(args.data / "initial_states.pt", map_location="cpu",
                         weights_only=False)
    samples = torch.load(args.data / f"samples_{args.regime}.pt", map_location="cpu",
                         weights_only=True)
    anchors = torch.load(args.data / "anchors.pt", map_location="cpu",
                         weights_only=True)
    train_rows = (anchors["split"] == 0).nonzero(as_tuple=True)[0]
    train_rows = train_rows[samples["valid"][train_rows, 0]]

    trace = true_trace(task, initial["snapshot"], args.blocks, block,
                       args.regime, args.action_seed, args.device)
    survivors = trace["valid"].all(dim=1).nonzero(as_tuple=True)[0]
    if survivors.numel() == 0:
        raise ValueError(f"No episode survives {args.blocks} blocks; lower --blocks")
    if args.select == "movers":
        # Total path length of every body, so the panels show episodes where
        # there is motion to predict.
        moved = torch.cat([trace["agent_state"], trace["object_state"]], dim=2)
        travel = (
            moved[:, 1:, :, :2] - moved[:, :-1, :, :2]
        ).norm(dim=-1).sum(dim=(1, 2))[survivors]
        live = survivors[travel.argsort(descending=True)[: args.episodes]]
    else:
        live = survivors[: args.episodes]
    print(f"{manifest['task_name']}: {survivors.numel()} of "
          f"{trace['valid'].shape[0]} episodes live through {args.blocks} blocks "
          f"({args.blocks * block} primitive steps); drawing {live.tolist()}")

    checkpoints = choose_checkpoints(args.runs, args.regime, args.seed)
    figure, axes = plt.subplots(
        len(KINDS), live.numel(),
        figsize=(3.5 * live.numel(), 3.5 * len(KINDS)), squeeze=False,
    )
    for row, kind in enumerate(KINDS):
        model = load_model(checkpoints[kind] / "model.pt", args.device)
        probes = fit_probes(model, samples, train_rows, args.device, kind=args.probe)
        true_latent, imagined_latent = imagine(model, trace, args.device)
        drift = []
        for column, episode in enumerate(live.tolist()):
            axis = axes[row][column]
            true_agent = trace["agent_state"][episode][:, :, MOTION]
            colours = agent_colours(true_agent.shape[1])
            true_object = trace["object_state"][episode][:, :, MOTION]
            probe_agent, probe_object = decode(probes, true_latent[episode])
            imagined_agent, imagined_object = decode(probes, imagined_latent[episode])
            # The probe reads a block boundary, so its first point is block 1;
            # prepend the simulator's own frame 0 so all three lines start together.
            def with_start(series, start):
                return torch.cat([start.unsqueeze(0).double(), series], dim=0).numpy()

            for agent in range(true_agent.shape[1]):
                draw(
                    axis,
                    true_agent[:, agent].numpy(),
                    with_start(probe_agent[:, agent], true_agent[0, agent]),
                    with_start(imagined_agent[:, agent], true_agent[0, agent]),
                    colours[agent],
                    f"agent {agent}" if (row == 0 and column == 0) else None,
                )
            draw(
                axis,
                true_object[:, 0].numpy(),
                with_start(probe_object[:, 0], true_object[0, 0]),
                with_start(imagined_object[:, 0], true_object[0, 0]),
                OBJECT_COLOUR,
                "object" if (row == 0 and column == 0) else None,
            )
            # Both error scales, on every body, so a panel can never be read as
            # model error when it is readout error. `probe` is the floor: the
            # simulator's own states encoded and decoded back.
            true_all = torch.cat(
                [true_agent[1:, :, :2], true_object[1:, :1, :2]], dim=1
            ).double()
            probe_all = torch.cat(
                [probe_agent[:, :, :2], probe_object[:, :1, :2]], dim=1
            )
            imagined_all = torch.cat(
                [imagined_agent[:, :, :2], imagined_object[:, :1, :2]], dim=1
            )
            probe_rmse = float((probe_all - true_all).norm(dim=-1).mean())
            imagined_rmse = float((imagined_all - true_all).norm(dim=-1).mean())
            # Error is only interpretable against how far the bodies actually
            # moved: the same 0.1m is a total failure on a body that travelled
            # 0.06m and a good prediction on one that travelled 1m.
            travel = float(
                (true_all[1:] - true_all[:-1]).norm(dim=-1).sum(dim=0).mean()
            )
            drift.append((probe_rmse, imagined_rmse, travel))
            axis.set_aspect("equal", adjustable="datalim")
            axis.grid(alpha=0.22, linewidth=0.5)
            axis.tick_params(labelsize=7)
            axis.set_title(
                (f"episode {episode}\n" if row == 0 else "")
                + f"probe {probe_rmse:.3f} m · imagined {imagined_rmse:.3f} m"
                + f" · travelled {travel:.2f} m",
                fontsize=8.2,
            )
            if column == 0:
                axis.set_ylabel(f"{kind}\ny (m)", fontsize=9,
                                color=COLOUR[kind], fontweight="bold")
            if row == len(KINDS) - 1:
                axis.set_xlabel("x (m)", fontsize=9)
        print(
            f"  {kind:12s} "
            + "  ".join(
                f"[probe {p:.3f} imagined {i:.3f} travelled {t:.2f} "
                f"-> {i / t:.0%} of travel]"
                for p, i, t in drift
            )
        )

    handles, labels = axes[0][0].get_legend_handles_labels()
    style = [
        plt.Line2D([], [], color="black", linestyle="-", linewidth=2.0,
                   marker="o", markersize=3, label="simulator (truth)"),
        plt.Line2D([], [], color="black", linestyle=":", linewidth=1.2,
                   label="probe on true latents (readout error)"),
        plt.Line2D([], [], color="black", linestyle="--", linewidth=1.6,
                   marker="^", markersize=3, label="probe on imagined rollout"),
    ]
    figure.legend(handles=handles, fontsize=8.5, frameon=False, ncol=3,
                  loc="lower center", bbox_to_anchor=(0.5, 0.045), title="body",
                  title_fontsize=8.5)
    figure.legend(handles=style, fontsize=8.5, frameon=False, ncol=3,
                  loc="lower center", bbox_to_anchor=(0.5, -0.004))
    figure.suptitle(
        f"{manifest['task_name']} -- imagined rollout against the simulator, "
        f"{args.blocks} blocks ({args.blocks * block} steps)\n"
        f"{args.regime} actions, seed {args.seed}, {args.probe} probe fitted on "
        f"{train_rows.numel()} train anchors · open circle = start",
        fontsize=10.5,
    )
    figure.tight_layout(rect=(0, 0.11, 1, 0.93))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=200)
    figure.savefig(args.output.with_suffix(".pdf"))
    print(f"wrote {args.output} and {args.output.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
