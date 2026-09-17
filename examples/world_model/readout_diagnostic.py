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
from examples.world_model.model_input import compose_frames, entity_frame
from examples.world_model.snapshot_restore import broadcast_state
from examples.world_model.train import load_model
from tensordict import TensorDict

BASELINES = ("independent", "joint", "relational")


@torch.no_grad()
def simulate(scratch_env, snapshot, candidates, action_block):
    """Roll the simulator and keep the observations, not only the rewards.

    Returns true per-candidate cost (B,K), that cost restricted to complete
    blocks (B,K) for comparison with block-boundary scorers, block validity
    (B,K,L), the
    observations at every block boundary (B,K,L+1,N,O), the endpoint each
    candidate actually reached (B,K,N,O) -- its first terminal frame, or its
    final frame when it never terminates -- and the tracked entity states at the
    same block boundaries (B,K,L+1,E*6).

    The entity frames are recorded because a model trained on `physical` inputs
    cannot be handed the agents' observations alone: its encoder was fitted on
    the world state appended. Tasks whose models use the default `observation`
    condition simply ignore them.
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
    # The endpoint each candidate actually reached: the frame at its first
    # termination, or the final frame if it never terminated. Block boundaries
    # cannot express this -- an episode ending inside a block has no boundary
    # frame at its endpoint, and the nearest ones are either before the
    # termination or padded after it.
    endpoint = frames[0].clone()
    ended = torch.zeros(count, dtype=torch.bool, device=candidates.device)

    entity_frames = [entity_frame(scratch_env).clone()]
    rewards, lives = [], []
    for step in range(horizon):
        td.set(("agents", "action"), actions[:, step])
        td = scratch_env.step(td)["next"]
        rewards.append(td["agents", "reward"].sum(dim=1).squeeze(-1))
        lives.append(live)
        observation = td["agents", "observation"]
        # Advance while running; freeze on the terminal frame itself.
        endpoint = torch.where((~ended).reshape(-1, 1, 1), observation, endpoint)
        done = td["done"].squeeze(-1)
        ended = ended | (live & done)
        live = live & ~done
        if (step + 1) % action_block == 0:
            frames.append(observation.clone())
            entity_frames.append(entity_frame(scratch_env).clone())

    reward = torch.stack(rewards, dim=-1)
    alive = torch.stack(lives, dim=-1)
    cost = -reward.masked_fill(~alive, 0).sum(dim=-1).reshape(batch, n_candidates)
    # The readout comparator scores block boundaries, and a partial terminal
    # block has no observed end-of-block frame to score -- admitting it would
    # feed the readout a padded observation. So the comparison is made over
    # complete blocks, and the truth it is compared against must cover exactly
    # those blocks too. Comparing a truth that includes the terminal partial
    # block against a readout that drops it measures the mismatch, not the
    # readout.
    complete = (
        alive.reshape(count, blocks, action_block)
        .all(dim=-1)
        .repeat_interleave(action_block, dim=1)
    )
    cost_complete = (
        -reward.masked_fill(~(alive & complete), 0)
        .sum(dim=-1)
        .reshape(batch, n_candidates)
    )
    block_valid = (
        alive.reshape(count, blocks, action_block)
        .all(dim=-1)
        .reshape(batch, n_candidates, blocks)
    )
    observation = torch.stack(frames, dim=1).reshape(
        batch, n_candidates, blocks + 1, agents, -1
    )
    entities = torch.stack(entity_frames, dim=1).reshape(
        batch, n_candidates, blocks + 1, -1
    )
    return (
        cost.cpu(),
        cost_complete.cpu(),
        block_valid.cpu(),
        observation.cpu(),
        endpoint.reshape(batch, n_candidates, *endpoint.shape[-2:]).cpu(),
        entities.cpu(),
    )


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
        truth, truth_complete, block_valid, observation, _endpoint, entities = simulate(
            scratch,
            select_anchor_states(anchors, chosen, args.device),
            candidates.to(args.device),
            block,
        )
    finally:
        scratch.close()

    rankable = truth.std(dim=1) > 1e-9
    rankable_complete = truth_complete.std(dim=1) > 1e-9
    print(
        f"states {chosen.numel()}, candidates {args.candidates}, "
        f"rankable {int(rankable.sum())}/{rankable.numel()}, "
        f"complete-block rankable {int(rankable_complete.sum())}/{rankable.numel()}"
    )

    # Block boundaries, reshaped so `compose_frames` sees one (B,T,...) record
    # per candidate. The composition is per input condition and identical for
    # every checkpoint sharing one, so it is built once and reused.
    states, n_candidates, boundaries = observation.shape[:3]
    flat_observation = observation.reshape(
        states * n_candidates, boundaries, *observation.shape[3:]
    )
    flat_entities = entities.reshape(states * n_candidates, boundaries, -1)
    composed = {}

    def boundary_frames(state_input, history_frames):
        key = (state_input, history_frames)
        if key not in composed:
            composed[key] = compose_frames(
                state_input, flat_observation, flat_entities, history_frames
            ).reshape(states, n_candidates, boundaries, observation.shape[3], -1)
        return composed[key]

    rows = {}
    samples = None
    for directory in sorted(args.runs.glob("[0-9]*")):
        checkpoint = directory / "model.pt"
        if not checkpoint.exists():
            continue
        config = yaml.safe_load((directory / "resolved_config.yaml").read_text())
        regime, kind = config["data"]["regime"], config["model"]["kind"]
        state_input = config["data"].get("state_input", "observation")
        history_frames = config["data"].get("history_frames", 3)
        if samples is None or samples[0] != regime:
            loaded = torch.load(
                args.data / f"samples_{regime}.pt",
                map_location="cpu",
                weights_only=True,
            )
            samples = (regime, loaded["observation"], loaded["package_state"])
        model = load_model(checkpoint, args.device)
        # The anchor frame in this checkpoint's own input format. Taking the
        # stored 6-dimension observation for a model fitted on 24 would feed the
        # encoder a vector it never saw, which is the defect this whole module
        # exists to measure -- so it must not be reintroduced here.
        start = compose_frames(
            state_input,
            samples[1][chosen, :1],
            samples[2][chosen, :1].flatten(2),
            history_frames,
        )[:, 0]
        predicted = model_costs(model, start, candidates, block, args.device)
        on_true = readout_costs(
            model, boundary_frames(state_input, history_frames), block_valid, args.device
        )
        rows.setdefault((state_input, regime, kind), []).append(
            (
                rank_against(truth, predicted, rankable),
                # Scored against the complete-block truth, which is the set of
                # blocks a boundary scorer can see at all.
                rank_against(truth_complete, on_true, rankable_complete),
            )
        )

    print("\nSpearman against the simulator's own plan costs:")
    header = (
        f"{'input':12s} {'regime':12s} {'kind':12s} {'J_model':>10s} "
        f"{'J_readout':>11s} {'recovered':>11s} {'seeds':>6s}"
    )
    print(header)
    print("-" * len(header))
    for (state_input, regime, kind), values in sorted(rows.items()):
        if kind not in BASELINES:
            continue
        model_rho = mean([v[0] for v in values])
        readout_rho = mean([v[1] for v in values])
        print(
            f"{state_input:12s} {regime:12s} {kind:12s} {model_rho:10.4f} "
            f"{readout_rho:11.4f} {readout_rho - model_rho:+11.4f} {len(values):6d}"
        )

    print(
        "\nJ_model uses predicted latents (dynamics + readout error);"
        "\nJ_readout uses the simulator's own latents (readout error only)."
        "\nA large positive 'recovered' means the rollout is what breaks ranking;"
        "\nnear zero means the readout cannot order plans either."
    )


if __name__ == "__main__":
    main()
