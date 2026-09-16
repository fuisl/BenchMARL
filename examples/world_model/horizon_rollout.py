# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the license in the LICENSE file in the root directory.
"""Latent rollout error per horizon block, against simulator truth.

Run:
    python -m examples.world_model.horizon_rollout outputs/buzz_wire_1196/baselines \
        --data outputs/buzz_wire_1196/data --blocks 20 --device cuda

The offline banks hold six-frame snippets, so every rollout number in the project
stops at five blocks -- 25 primitive steps. That is the depth the planner uses,
but it is far too short to see where a learned model actually stops being usable.
This rolls the true simulator from episode starts for the full 100 steps and
scores the model's open-loop latent prediction at each block boundary against the
encoded truth.

Ground truth is generated here rather than loaded because no bank contains it.
Actions come from the same regime sampler the training data used, so the model is
asked about the action distribution it was fitted on.
"""

import argparse
import statistics as st
from collections import defaultdict
from pathlib import Path

import torch
import yaml

from benchmarl.environments import VmasTask
from examples.world_model.collect import sample_actions
from examples.world_model.mpc import action_bounds
from examples.world_model.snapshot_restore import agent_observations, restore_state
from examples.world_model.train import load_model
from tensordict import TensorDict


def to_device(value, device):
    """The banks store snapshots on CPU; restoring one into a CUDA env mixes
    devices inside the physics step."""
    if isinstance(value, dict):
        return {k: to_device(v, device) for k, v in value.items()}
    return value.to(device) if torch.is_tensor(value) else value


@torch.no_grad()
def simulator_truth(task, snapshot, blocks, action_block, regime, seed, device):
    """Observations at every block boundary and per-block validity.

    Returns frames (B, blocks+1, N, O), blocked actions (B, blocks, N, L*d_a)
    and valid (B, blocks): a block is valid when the episode was still running
    for all of it, which is the same rule the offline banks apply.
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
    frames, valid = [agent_observations(env)], []
    for block in range(blocks):
        alive_through = live.clone()
        for step in range(action_block):
            td.set(("agents", "action"), actions[:, block * action_block + step])
            td = env.step(td)["next"]
            live = live & ~td["done"].squeeze(-1)
            alive_through = alive_through & live
        frames.append(agent_observations(env))
        valid.append(alive_through)
    env.close()
    blocked = actions.reshape(batch, blocks, action_block, agents, action_dim)
    blocked = blocked.permute(0, 1, 3, 2, 4).reshape(batch, blocks, agents, -1)
    return torch.stack(frames, dim=1), blocked, torch.stack(valid, dim=1)


def trained_frames(model):
    """How many context positions ever received a prediction gradient.

    `dynamics_losses` predicts from `latent[:, :-1]` over a snippet of
    `pos_embedding.shape[1]` frames, so the last allocated position is never a
    predictor input and its embedding is only ever decayed. A gradient probe on
    a trained checkpoint gives nonzero norms for positions 0-4 and exactly 0.0
    for position 5 (outputs/review_20260916/evidence.json).

    Rolling through that position is what produced the h>=6 cliff first reported
    in docs/paper/experiments/09_horizon_rollout.md: at a 6-position window
    Transport's error jumps 0.0565 -> 0.2731 at h=6, while the same checkpoint
    on the same actions stays at 0.0595 with the trained 5. The cliff was the
    untrained slot, not the horizon.
    """
    return model.predictor.pos_embedding.shape[1] - 1


@torch.no_grad()
def windowed_rollout(model, latent, actions):
    """Autoregressive rollout past the trained context length.

    `MultiAgentWorldModel.rollout` grows its history without bound and indexes
    `pos_embedding`, so with 6 allocated positions it raises at block 7 -- but
    block 6 already reads the untrained position, so only 5 blocks are ever
    supported by training.

    Past that the only honest thing a model like this can do is slide its
    context, which is what any deployment past the training length would do. The
    window is re-indexed from position 0 each step, so blocks past
    `trained_frames` are extrapolation and are marked as such where reported.
    """
    frames = trained_frames(model)
    history, window, outputs = latent, actions[:, :0], []
    for step in range(actions.size(1)):
        # Both windows are trimmed to the same length before the call: `predict`
        # infers its frame count from the latents and reshapes the actions to
        # match, so a one-off difference is a silent shape error, not a warning.
        window = torch.cat([window, actions[:, step : step + 1]], dim=1)[:, -frames:]
        predicted = model.predict(history, window)[:, -1:]
        outputs.append(predicted)
        history = torch.cat([history, predicted], dim=1)[:, -frames:]
    return torch.cat(outputs, dim=1)


@torch.no_grad()
def errors(run, frames, actions, valid, device):
    """Masked mean squared latent error per block: (blocks,)."""
    model = load_model(run / "model.pt", device)
    model.eval()
    truth = model.encode(frames)[:, 1:]
    rolled = windowed_rollout(model, model.encode(frames[:, :1]), actions)
    squared = (rolled - truth).square().mean(dim=(-1, -2))  # (B, blocks)
    mask = valid.float()
    counts = mask.sum(dim=0)
    # A block no episode survives to has no error, not an error of zero.
    return torch.where(
        counts > 0,
        (squared * mask).sum(dim=0) / counts.clamp_min(1),
        torch.full_like(counts, float("nan")),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sweep", type=Path)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--blocks", type=int, default=20)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=9100)
    args = ap.parse_args()

    manifest = yaml.safe_load((args.data / "manifest.json").read_text())
    task = VmasTask[manifest["task_name"].split("/")[-1].upper()].get_from_yaml()
    initial = torch.load(args.data / "initial_states.pt", map_location="cpu",
                         weights_only=False)
    block = manifest["action_block"]

    truth = {}
    for regime in ("correlated", "independent"):
        truth[regime] = simulator_truth(
            task, initial["snapshot"], args.blocks, block, regime, args.seed,
            args.device,
        )
        print(f"{regime}: {int(truth[regime][2][:, -1].sum())} of "
              f"{truth[regime][2].shape[0]} episodes still live at block "
              f"{args.blocks}", flush=True)

    # Every checkpoint in a sweep shares an architecture, so one read states the
    # context the whole table was rolled at.
    trained = trained_frames(
        load_model(next(args.sweep.rglob("model.pt")), args.device)
    )

    grouped = defaultdict(list)
    for config_path in sorted(args.sweep.rglob("resolved_config.yaml")):
        run = config_path.parent
        if not (run / "model.pt").exists():
            continue
        cfg = yaml.safe_load(config_path.read_text())
        if cfg["data"].get("state_input", "observation") != "observation":
            continue
        regime = cfg["data"]["regime"]
        grouped[(regime, cfg["model"]["kind"])].append(
            errors(run, *truth[regime], args.device).cpu().tolist()
        )

    shown = [h for h in (1, 2, 3, 5, 8, 10, 15, 20) if h <= args.blocks]
    live = truth["correlated"][2].sum(dim=0).tolist()
    print("\nepisodes still live: " + "  ".join(
        f"h={h}:{int(live[h - 1])}" for h in shown))
    header = "".join(f"{f'h={h}':>9s}" for h in shown)
    print(f"\n{manifest['task_name']} -- latent rollout error, {args.blocks} blocks "
          f"({args.blocks * block} primitive steps); context slides over the "
          f"trained positions only, so h>{trained} is extrapolation")
    print(f"{'regime':12s}{'predictor':13s}{header}{'h20/h1':>9s}{'seeds':>7s}")
    print("-" * (25 + 9 * len(shown) + 16))
    for regime in ("correlated", "independent"):
        for kind in ("independent", "joint", "relational"):
            runs = grouped.get((regime, kind))
            if not runs:
                continue
            mean = [
                st.mean(r[h] for r in runs) if runs[0][h] == runs[0][h] else float("nan")
                for h in range(args.blocks)
            ]
            cells = "".join(
                ("       --" if mean[h - 1] != mean[h - 1] else f"{mean[h - 1]:9.4f}")
                for h in shown
            )
            usable = [v for v in mean if v == v]
            print(f"{regime:12s}{kind:13s}{cells}{usable[-1] / usable[0]:9.2f}{len(runs):7d}")


if __name__ == "__main__":
    main()
