#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#
"""Prototype: LeWM's goal-conditioned plan cost, as an alternative to task reward.

Kept separate from the task-reward path on purpose. Nothing here changes
`plan_ranking`, `models` or `train`; this imports them and overrides only the two
pieces LeWM does differently, so the two objectives can be compared without
disturbing results already recorded.

**Why.** Our `J = -sum_t sum_i r_i,t` needs a reward head and a termination head,
and both are where the chain measurably breaks:

* Buzz Wire's readout is *anti-correlated* (rho ~ -0.25) even given the
  simulator's own latents, because its reward depends on the ball and no agent
  observes the ball;
* the termination head has never seen a positive example on Transport or
  Dropout, and plan costs discard it;
* Transport's reward telescopes to package displacement, which is identically
  zero for 77% of snippets, leaving nothing to rank.

LeWM's terminal latent-goal distance needs neither head, and varies continuously
across candidates, so it removes all three problems at once. The goal itself *is*
observable on these tasks -- Buzz Wire observes `pos - goal` directly.

**Fidelity to the reference** (revision 8edfeb33, `jepa.py`):

* `terminal_goal_cost` mirrors `JEPA.criterion`: terminal frame only, goal
  detached, squared error **summed** over feature dimensions rather than averaged.
* `lewm_rollout` mirrors `JEPA.rollout`: the predictor context is truncated to
  the last `history_size` frames at every step. `MultiAgentWorldModel.rollout`
  instead lets the context grow, so it is overridden rather than reused.

Deliberate multi-agent extension: the reference is single-agent, so the terminal
distance is summed over agents as well as latent dimensions.
"""

import torch

from examples.world_model.plan_ranking import spearman
from examples.world_model.readout_diagnostic import simulate

HISTORY_SIZE = 3  # LeWM config/train/lewm.yaml


@torch.no_grad()
def lewm_rollout(model, latent, actions, history_size=HISTORY_SIZE):
    """Autoregressive rollout with LeWM's context truncation.

    latent: (B,1,N,D) encoded start. actions: (B,H,N,block*d_a).
    Returns predicted latents (B,H,N,D) for steps 1..H.

    The reference keeps only the last `history_size` frames of context at each
    step and takes the final position of the prediction. Our own
    `MultiAgentWorldModel.rollout` feeds the whole growing history instead, which
    is a deviation worth isolating rather than silently inheriting.
    """
    history = latent
    predictions = []
    for step in range(actions.size(1)):
        context = history[:, -history_size:]
        window = actions[:, : step + 1][:, -history_size:]
        predicted = model.predict(context, window)[:, -1:]
        predictions.append(predicted)
        history = torch.cat([history, predicted], dim=1)
    return torch.cat(predictions, dim=1)


def terminal_goal_cost(predicted, goal):
    """LeWM ``JEPA.criterion`` for multi-agent latents.

    predicted: (B,K,H,N,D) rolled latents. goal: (B,1,1,N,D) encoded goal.
    Returns (B,K).

    Terminal frame only, goal detached, and the squared error is **summed** over
    agents and latent dimensions -- the reference uses `reduction="none"` then
    `.sum()` over every non-batch dimension, not a mean.
    """
    terminal = predicted[:, :, -1:]
    target = goal.expand_as(predicted)[:, :, -1:].detach()
    error = torch.nn.functional.mse_loss(terminal, target, reduction="none")
    return error.sum(dim=tuple(range(2, predicted.ndim)))


@torch.no_grad()
def goal_plan_costs(
    model,
    observation,
    goal_observation,
    candidates,
    action_block,
    device,
    history_size=HISTORY_SIZE,
):
    """Predicted terminal latent distance to the goal, per candidate: (B,K)."""
    batch, n_candidates, steps, joint_dim = candidates.shape
    agents, obs_dim = observation.shape[1:]
    action_dim = joint_dim // agents
    blocks = steps // action_block

    plans = candidates.view(
        batch, n_candidates, blocks, action_block, agents, action_dim
    )
    plans = (
        plans.permute(0, 1, 2, 4, 3, 5)
        .reshape(batch * n_candidates, blocks, agents, action_block * action_dim)
        .to(device)
    )

    start = (
        observation.unsqueeze(1)
        .expand(batch, n_candidates, agents, obs_dim)
        .reshape(batch * n_candidates, 1, agents, obs_dim)
        .to(device)
    )
    rolled = lewm_rollout(model, model.encode(start), plans, history_size)
    rolled = rolled.view(batch, n_candidates, blocks, agents, -1)

    goal = model.encode(goal_observation.unsqueeze(1).to(device))
    goal = goal.view(batch, 1, 1, agents, -1)
    return terminal_goal_cost(rolled, goal).cpu()


def terminal_observations(observation, block_valid):
    """Last observation each candidate actually reached: (B,K,N,O).

    Episodes end at different steps, so the frame after the last valid block is
    the reachable endpoint; reading the final frame unconditionally would compare
    against states past termination.
    """
    last = block_valid.sum(dim=-1).clamp_min(1)
    index = last.view(*last.shape, 1, 1).expand(*last.shape, *observation.shape[-2:])
    return observation.gather(2, index.unsqueeze(2)).squeeze(2)


def true_goal_costs(observation, block_valid, goal_observation):
    """Ground-truth analogue: terminal distance in observation space, (B,K).

    The simulator has no latent space, so the objective it can be held to is the
    same terminal distance measured on observations. Summed, to match the
    latent cost's reduction.
    """
    reached = terminal_observations(observation, block_valid)
    return (reached - goal_observation.unsqueeze(1)).square().sum(dim=(-1, -2))


def rank_scores(truth, predicted, rankable):
    scores = []
    for state in rankable.nonzero(as_tuple=True)[0]:
        value = spearman(predicted[state], truth[state])
        if value == value:
            scores.append(value)
    return sum(scores) / len(scores) if scores else float("nan")


__all__ = [
    "lewm_rollout",
    "terminal_goal_cost",
    "goal_plan_costs",
    "true_goal_costs",
    "terminal_observations",
    "rank_scores",
    "simulate",
]


def main():
    """Compare goal-cost ranking against the simulator's own terminal distance.

    The goal is an *achieved* observation: for each state one candidate's
    endpoint is held out and used as the target, so the goal is reachable by
    construction and the ranking question is well posed. That makes this a
    diagnostic of planning capability, NOT a measure of task success -- the
    scenario's own `done()` remains the only success criterion.
    """
    import argparse
    import json
    from pathlib import Path

    import yaml

    from benchmarl.environments import VmasTask
    from examples.world_model.plan_ranking import select_anchor_states
    from examples.world_model.train import load_model

    parser = argparse.ArgumentParser()
    parser.add_argument("runs", type=Path)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--states", type=int, default=96)
    parser.add_argument("--candidates", type=int, default=48)
    parser.add_argument("--history", type=int, default=HISTORY_SIZE)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=5100)
    args = parser.parse_args()

    manifest = json.loads((args.data / "manifest.json").read_text())
    block, steps = manifest["action_block"], manifest["sequence_steps"]
    joint_dim = torch.as_tensor(manifest["action_low"]).numel()

    anchors = torch.load(
        args.data / "anchors.pt", map_location="cpu", weights_only=True
    )
    test = (anchors["split"] == 2).nonzero(as_tuple=True)[0]
    generator = torch.Generator().manual_seed(args.seed)
    chosen = test[torch.randperm(test.numel(), generator=generator)[: args.states]]
    # One extra candidate supplies the goal and is then excluded from ranking.
    candidates = (
        torch.rand(
            chosen.numel(), args.candidates + 1, steps, joint_dim, generator=generator
        )
        * 2
        - 1
    )

    task = VmasTask[manifest["task_name"].split("/")[-1].upper()].get_from_yaml()
    scratch = task.get_env_fun(
        chosen.numel() * (args.candidates + 1), True, 0, args.device
    )()
    scratch.reset()
    try:
        _, block_valid, observation = simulate(
            scratch,
            select_anchor_states(anchors, chosen, args.device),
            candidates.to(args.device),
            block,
        )
    finally:
        scratch.close()

    goal = terminal_observations(observation, block_valid)[:, 0]
    scored = candidates[:, 1:]
    truth = true_goal_costs(observation[:, 1:], block_valid[:, 1:], goal)
    rankable = truth.std(dim=1) > 1e-9
    print(
        f"states {chosen.numel()}, candidates {args.candidates}, "
        f"rankable {int(rankable.sum())}/{rankable.numel()}, history {args.history}"
    )

    rows, samples = {}, None
    for directory in sorted(args.runs.glob("[0-9]*")):
        if not (directory / "model.pt").exists():
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
        model = load_model(directory / "model.pt", args.device)
        predicted = goal_plan_costs(
            model, samples[1][chosen, 0], goal, scored, block, args.device, args.history
        )
        rows.setdefault((regime, kind), []).append(
            rank_scores(truth, predicted, rankable)
        )

    print("\nSpearman of goal-cost ranking against the simulator's terminal distance:")
    header = f"{'regime':12s} {'kind':12s} {'spearman':>10s}"
    print(header)
    print("-" * len(header))
    for (regime, kind), values in sorted(rows.items()):
        print(f"{regime:12s} {kind:12s} {sum(values)/len(values):10.4f}")


if __name__ == "__main__":
    main()
