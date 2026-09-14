#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#
"""Why does plan ranking fail -- the dynamics or the readout?

Plan ranking gives Spearman ~0.05 for every baseline even on Buzz Wire, where
the cost landscape is fully non-degenerate and the models demonstrably capture
26% of the cross-agent effect. A model plan cost passes through two learned
stages, so the failure could sit in either:

    predicted latents  ->  reward readout  ->  plan cost

This separates them with one extra scoring path. The simulator is rolled out for
the same candidates and its **true** observations at each block boundary are
encoded, then the *learned readout* is applied to those true latents:

    J_true      simulator states,  simulator rewards   (ground truth)
    J_readout   simulator states,  learned readout     (readout error only)
    J_model     predicted latents, learned readout     (both errors)

If J_readout ranks well, the readout is adequate and the rollout is what breaks;
if J_readout ranks no better than J_model, the readout cannot order plans and
improving the dynamics would not help.

Run:
    python -m examples.world_model.readout_diagnostic \\
        outputs/buzz_wire_1196/baselines --data outputs/buzz_wire_1196/data
"""

import argparse
import json
from pathlib import Path

import torch
import yaml

from benchmarl.environments import VmasTask
from examples.world_model.compare_baselines import mean
from examples.world_model.plan_ranking import (
    model_costs,
    select_anchor_states,
    spearman,
)
from examples.world_model.snapshot_restore import broadcast_state
from examples.world_model.train import load_model
from tensordict import TensorDict

BASELINES = ("independent", "joint", "relational")


@torch.no_grad()
def simulate(scratch_env, snapshot, candidates, action_block):
    """Roll the simulator and keep the observations, not only the rewards.

    Returns true per-candidate cost (B,K), block validity (B,K,L) and the
    observations at every block boundary (B,K,L+1,N,O).
    """
    batch, n_candidates, horizon, joint_dim = candidates.shape
    count = batch * n_candidates
    agents = len(scratch_env._env.world.agents)
    primitive = joint_dim // agents
    blocks = horizon // action_block

    indices = torch.arange(batch, device=candidates.device).repeat_interleave(
        n_candidates
    )
    broadcast_state(scratch_env, snapshot, source_indices=indices)
    live = ~scratch_env._env.done()
    actions = candidates.reshape(count, horizon, agents, primitive)
    td = TensorDict({}, batch_size=[count], device=candidates.device)

    frames = [
        torch.stack(
            [
                scratch_env._env.scenario.observation(agent)
                for agent in scratch_env._env.world.agents
            ],
            dim=1,
        ).clone()
    ]
    rewards, lives = [], []
    for step in range(horizon):
        td.set(("agents", "action"), actions[:, step])
        td = scratch_env.step(td)["next"]
        rewards.append(td["agents", "reward"].sum(dim=1).squeeze(-1))
        lives.append(live)
        live = live & ~td["done"].squeeze(-1)
        if (step + 1) % action_block == 0:
            frames.append(td["agents", "observation"].clone())

    reward = torch.stack(rewards, dim=-1)
    alive = torch.stack(lives, dim=-1)
    cost = -reward.masked_fill(~alive, 0).sum(dim=-1).reshape(batch, n_candidates)
    block_valid = (
        alive.reshape(count, blocks, action_block)
        .all(dim=-1)
        .reshape(batch, n_candidates, blocks)
    )
    observation = torch.stack(frames, dim=1).reshape(
        batch, n_candidates, blocks + 1, agents, -1
    )
    return cost.cpu(), block_valid.cpu(), observation.cpu()


@torch.no_grad()
def readout_costs(model, observation, block_valid, device):
    """Learned readout applied to the simulator's own latents: (B, K)."""
    batch, n_candidates, frames, agents, obs_dim = observation.shape
    flat = observation.reshape(batch * n_candidates, frames, agents, obs_dim).to(device)
    latent = model.encode(flat)
    reward, _ = model.readout(latent[:, :-1], latent[:, 1:])
    mask = block_valid.reshape(batch * n_candidates, frames - 1, 1, 1).to(device)
    cost = -(reward * mask).sum(dim=(1, 2, 3))
    return cost.view(batch, n_candidates).cpu()


def rank_against(truth, predicted, rankable):
    scores = []
    for state in rankable.nonzero(as_tuple=True)[0]:
        value = spearman(predicted[state], truth[state])
        if value == value:
            scores.append(value)
    return mean(scores) if scores else float("nan")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", type=Path)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--states", type=int, default=96)
    parser.add_argument("--candidates", type=int, default=48)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=5100)
    args = parser.parse_args()

    manifest = json.loads((args.data / "manifest.json").read_text())
    block = manifest["action_block"]
    steps = manifest["sequence_steps"]
    joint_dim = torch.as_tensor(manifest["action_low"]).numel()

    anchors = torch.load(
        args.data / "anchors.pt", map_location="cpu", weights_only=True
    )
    test = (anchors["split"] == 2).nonzero(as_tuple=True)[0]
    generator = torch.Generator().manual_seed(args.seed)
    chosen = test[torch.randperm(test.numel(), generator=generator)[: args.states]]
    candidates = (
        torch.rand(
            chosen.numel(), args.candidates, steps, joint_dim, generator=generator
        )
        * 2
        - 1
    )

    task = VmasTask[manifest["task_name"].split("/")[-1].upper()].get_from_yaml()
    scratch = task.get_env_fun(chosen.numel() * args.candidates, True, 0, args.device)()
    scratch.reset()
    try:
        truth, block_valid, observation = simulate(
            scratch,
            select_anchor_states(anchors, chosen, args.device),
            candidates.to(args.device),
            block,
        )
    finally:
        scratch.close()

    rankable = truth.std(dim=1) > 1e-9
    print(
        f"states {chosen.numel()}, candidates {args.candidates}, "
        f"rankable {int(rankable.sum())}/{rankable.numel()}"
    )

    rows = {}
    samples = None
    for directory in sorted(args.runs.glob("[0-9]*")):
        checkpoint = directory / "model.pt"
        if not checkpoint.exists():
            continue
        config = yaml.safe_load((directory / "resolved_config.yaml").read_text())
        regime, kind = config["data"]["regime"], config["model"]["kind"]
        if samples is None or samples[0] != regime:
            loaded = torch.load(
                args.data / f"samples_{regime}.pt",
                map_location="cpu",
                weights_only=True,
            )
            samples = (regime, loaded["observation"])
        model = load_model(checkpoint, args.device)
        predicted = model_costs(
            model, samples[1][chosen, 0], candidates, block, args.device
        )
        on_true = readout_costs(model, observation, block_valid, args.device)
        rows.setdefault((regime, kind), []).append(
            (
                rank_against(truth, predicted, rankable),
                rank_against(truth, on_true, rankable),
            )
        )

    print("\nSpearman against the simulator's own plan costs:")
    header = (
        f"{'regime':12s} {'kind':12s} {'J_model':>10s} {'J_readout':>11s} "
        f"{'recovered':>11s}"
    )
    print(header)
    print("-" * len(header))
    for (regime, kind), values in sorted(rows.items()):
        if kind not in BASELINES:
            continue
        model_rho = mean([v[0] for v in values])
        readout_rho = mean([v[1] for v in values])
        print(
            f"{regime:12s} {kind:12s} {model_rho:10.4f} {readout_rho:11.4f} "
            f"{readout_rho - model_rho:+11.4f}"
        )

    print(
        "\nJ_model uses predicted latents (dynamics + readout error);"
        "\nJ_readout uses the simulator's own latents (readout error only)."
        "\nA large positive 'recovered' means the rollout is what breaks ranking;"
        "\nnear zero means the readout cannot order plans either."
    )


if __name__ == "__main__":
    main()
