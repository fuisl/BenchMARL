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
    """Deprecated: block-boundary approximation of the endpoint.

    Kept only so existing callers fail loudly rather than silently returning the
    wrong frame. It counts complete blocks and clamps to at least one, so a
    candidate terminating inside its first block returns a frame from *after*
    termination, and one terminating inside a later block returns the boundary
    *before* it. Use the ``endpoint`` that ``simulate`` now returns.
    """
    raise NotImplementedError(
        "terminal_observations mis-locates endpoints; use simulate()'s endpoint"
    )


def true_goal_costs(endpoint, goal_observation):
    """Ground-truth analogue: terminal distance in observation space, (B,K).

    The simulator has no latent space, so the objective it can be held to is the
    same terminal distance measured on observations, at the endpoint each
    candidate actually reached. Summed, to match the latent cost's reduction.
    """
    return (endpoint - goal_observation.unsqueeze(1)).square().sum(dim=(-1, -2))


@torch.no_grad()
def encoded_goal_costs(model, endpoint, goal_observation, device):
    """B: latent distance between the TRUE endpoints and the goal, (B,K).

    The simulator supplies the endpoints, so nothing here depends on the
    model's rollout. It isolates whether the learned representation orders
    physically meaningful goals at all -- the question A-B answers, before
    rollout error enters in C.
    """
    batch, candidates, agents, obs_dim = endpoint.shape
    flat = endpoint.reshape(batch * candidates, 1, agents, obs_dim).to(device)
    latent = model.encode(flat).reshape(batch, candidates, 1, agents, -1)
    goal = model.encode(goal_observation.unsqueeze(1).to(device))
    goal = goal.reshape(batch, 1, 1, agents, -1)
    return terminal_goal_cost(latent, goal).cpu()


def selected_regret(reference, scores, rankable):
    """Physical cost of each scorer's chosen plan, above the best available.

    Rank correlation says whether the ordering is broadly right; regret says
    what the ordering costs when a single plan is actually executed, which is
    what a planner does.
    """
    values = []
    for state in rankable.nonzero(as_tuple=True)[0]:
        chosen = int(scores[state].argmin())
        values.append(float(reference[state, chosen] - reference[state].min()))
    return sum(values) / len(values) if values else float("nan")


def top_agreement(reference, scores, rankable):
    """Fraction of states where the scorer picks the physically best plan."""
    hits = [
        int(scores[state].argmin()) == int(reference[state].argmin())
        for state in rankable.nonzero(as_tuple=True)[0]
    ]
    return sum(hits) / len(hits) if hits else float("nan")


def rank_scores(truth, predicted, rankable):
    scores = []
    for state in rankable.nonzero(as_tuple=True)[0]:
        value = spearman(predicted[state], truth[state])
        if value == value:
            scores.append(value)
    return sum(scores) / len(scores) if scores else float("nan")


__all__ = [
    "encoded_goal_costs",
    "selected_regret",
    "top_agreement",
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
        _, _complete, _block_valid, _observation, endpoint = simulate(
            scratch,
            select_anchor_states(anchors, chosen, args.device),
            candidates.to(args.device),
            block,
        )
    finally:
        scratch.close()

    # Candidate 0 supplies the goal and is excluded from ranking. Both the goal
    # and the scored endpoints are the frames actually reached.
    goal = endpoint[:, 0]
    scored = candidates[:, 1:]
    truth = true_goal_costs(endpoint[:, 1:], goal)
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
        # B: true endpoints, learned representation. C: predicted endpoints.
        encoded = encoded_goal_costs(model, endpoint[:, 1:], goal, args.device)
        predicted = goal_plan_costs(
            model, samples[1][chosen, 0], goal, scored, block, args.device, args.history
        )
        rows.setdefault((regime, kind), []).append(
            {
                "B_vs_A": rank_scores(truth, encoded, rankable),
                "C_vs_A": rank_scores(truth, predicted, rankable),
                "C_vs_B": rank_scores(encoded, predicted, rankable),
                "B_regret": selected_regret(truth, encoded, rankable),
                "C_regret": selected_regret(truth, predicted, rankable),
                "B_top": top_agreement(truth, encoded, rankable),
                "C_top": top_agreement(truth, predicted, rankable),
            }
        )

    # A is the simulator's own physical goal distance, the reference ordering.
    # B adds the learned representation; C adds the learned rollout on top.
    # A-B is representation geometry, B-C is what prediction costs.
    random_regret = selected_regret(
        truth, torch.rand(truth.shape, generator=generator), rankable
    )
    print(
        f"\nA = simulator physical goal distance (reference). "
        f"Random-choice regret {random_regret:.4f}."
    )
    header = (
        f"{'regime':12s} {'kind':12s} {'B vs A':>8s} {'C vs A':>8s} {'C vs B':>8s} "
        f"{'B regret':>9s} {'C regret':>9s} {'B top':>7s} {'C top':>7s}"
    )
    print(header)
    print("-" * len(header))
    for (regime, kind), values in sorted(rows.items()):
        avg = {k: sum(v[k] for v in values) / len(values) for k in values[0]}
        print(
            f"{regime:12s} {kind:12s} {avg['B_vs_A']:8.4f} {avg['C_vs_A']:8.4f} "
            f"{avg['C_vs_B']:8.4f} {avg['B_regret']:9.4f} {avg['C_regret']:9.4f} "
            f"{avg['B_top']:7.2f} {avg['C_top']:7.2f}"
        )


if __name__ == "__main__":
    main()
