# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the license in the LICENSE file in the root directory.
"""Gate 2: compare models' intervention response in COMMON physical coordinates.

Every cross-model number this project has reported lives in each model's own
learned latent space. SIGReg does not make two encoders keep the same variables,
so "joint's error is lower than relational's" has never been a statement about
physics. Review 2026-09-16 section 6 asks for one comparison in coordinates that
do not depend on which model produced them.

Y is the physical quantity the intervention can move: the non-intervened agents'
position and velocity, and the shared object's. From one restored state we
compare a reference joint action A against A', which differs only in agent 1's
x component, and score

    response ratio = sum ||(predicted dY - true dY) / scale||^2
                     / sum ||true dY / scale||^2

`scale` is the per-dimension standard deviation of Y on TRAINING data, fitted
once and shared by every model, so the ratio is unitless and comparable. A model
predicting no response at all scores exactly 1.0.

Two probes, both ridge regressions in closed form -- no optimiser, so nothing
here adds a tuning surface:

  * agent head: one agent's latent -> that agent's (pos, vel). Deliberately NOT
    a global readout. A non-intervened agent's predicted state must depend only
    on its own latent, or the probe could turn a change in the intervened
    agent's latent into a fake response.
  * object head: all agents' latents -> the shared object's (pos, vel). The
    object is nobody's private state, so this one has to be global.

Probes are fitted on train-split anchors, the ridge strength is chosen on the
validation split, and everything is reported on the test split. Probe error on
TRUE encodings is reported separately from prediction error, so a model whose
latent simply does not carry position is visible as such rather than being
scored as a dynamics failure.

Run:
    python -m examples.world_model.physical_response \\
        outputs/balance_repair_1236/baselines \\
        --data outputs/balance_repair_1236/data --device cuda
"""

import argparse
import json
from pathlib import Path

import torch
import yaml

from examples.world_model.compare_baselines import bootstrap_interval, mean
from examples.world_model.counterfactual_evaluation import blocked, INTERVENED
from examples.world_model.stratified_evaluation import effect_labels
from examples.world_model.train import load_model

BASELINES = ("independent", "joint", "relational")
# [pos_x, pos_y, vel_x, vel_y] of the 6-column stored physical state; rotation
# and angular velocity are excluded because not every task's bodies rotate.
MOTION = slice(0, 4)
RIDGE_STRENGTHS = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)
# Fewer, because each one is a real training run rather than a solve.
MLP_DECAYS = (0.0, 1e-4, 1e-2)


def ridge(features, targets, strength):
    """Closed-form ridge with a bias column: (F,) -> (T,). Returns (F+1, T)."""
    ones = torch.ones(features.shape[0], 1, dtype=features.dtype)
    design = torch.cat([features, ones], dim=1)
    gram = design.T @ design
    penalty = strength * torch.eye(gram.shape[0], dtype=gram.dtype)
    penalty[-1, -1] = 0.0  # never penalise the bias
    return torch.linalg.solve(gram + penalty, design.T @ targets)


def apply_probe(readout_weights, features):
    """Apply either probe family; an MLP is a Module, a ridge fit is a matrix."""
    if isinstance(readout_weights, torch.nn.Module):
        device = next(readout_weights.parameters()).device
        with torch.no_grad():
            return readout_weights(features.to(device)).cpu()
    ones = torch.ones(features.shape[0], 1, dtype=features.dtype)
    return torch.cat([features, ones], dim=1) @ readout_weights


@torch.no_grad()
def probe_features(model, observation, device):
    """Encoded frames as probe inputs: per-agent (M,dim) and global (M,N*dim)."""
    latent = model.encode(observation.unsqueeze(1).to(device))[:, 0].cpu().double()
    anchors, agents, dim = latent.shape
    return latent.reshape(anchors * agents, dim), latent.reshape(anchors, agents * dim)


def fit_mlp(features, targets, strength, device, seed=0):
    """One hidden layer, fixed budget, no schedule -- a capacity check, not a model.

    A linear probe failing could mean the latent does not carry physical state,
    or only that the readout is too weak. This separates them. Architecture and
    optimiser are fixed constants so `strength` stays the single selected knob.
    """
    torch.manual_seed(seed)
    net = torch.nn.Sequential(
        torch.nn.Linear(features.shape[1], 256),
        torch.nn.GELU(),
        torch.nn.Linear(256, targets.shape[1]),
    ).to(device).double()
    x, y = features.to(device), targets.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=strength)
    for _ in range(400):
        optimizer.zero_grad()
        loss = (net(x) - y).square().mean()
        loss.backward()
        optimizer.step()
    if not torch.isfinite(loss):
        raise ValueError("Probe training diverged")
    return net


def fit_probes(model, samples, rows, device, kind="linear", shaped=None):
    """Fit both heads on `rows`; choose the regularisation on a held-out quarter.

    Returns (agent_weights, object_weights, validation errors), all in the raw
    physical units the simulator stores.
    """
    observation = (
        samples["observation"][rows, 0]
        if shaped is None
        else shaped(samples, rows, 0, False)
    )
    agent_target = samples["agent_state"][rows, 0][..., MOTION].double()
    object_target = samples["package_state"][rows, 0][..., MOTION].double()
    per_agent, global_latent = probe_features(model, observation, device)
    agent_target = agent_target.reshape(-1, agent_target.shape[-1])
    object_target = object_target.reshape(object_target.shape[0], -1)

    # The split is by anchor, and anchors from one root episode are correlated,
    # so this is a selection split rather than a clean generalisation estimate.
    # It only picks a scalar, and the test anchors below are a different split.
    cut = int(0.75 * global_latent.shape[0])
    agent_cut = cut * (per_agent.shape[0] // global_latent.shape[0])
    chosen = {}
    strengths = RIDGE_STRENGTHS if kind == "linear" else MLP_DECAYS
    fit = (
        (lambda f, y, s: ridge(f, y, s))
        if kind == "linear"
        else (lambda f, y, s: fit_mlp(f, y, s, device))
    )
    for name, features, target, split in (
        ("agent", per_agent, agent_target, agent_cut),
        ("object", global_latent, object_target, cut),
    ):
        scored = []
        for strength in strengths:
            fitted = fit(features[:split], target[:split], strength)
            error = (apply_probe(fitted, features[split:]) - target[split:]).square()
            scored.append((float(error.mean()), strength))
        held_out, best = min(scored)
        # Refit on everything at the selected strength; the split above only
        # ever chose that scalar.
        chosen[name] = (fit(features, target, best), best, held_out)
    return chosen


@torch.no_grad()
def encoded(model, observation, device):
    """Encoded frame, CPU double, matching `predicted_latent`'s layout."""
    return model.encode(observation.unsqueeze(1).to(device))[:, 0].cpu().double()


@torch.no_grad()
def predicted_latent(model, observation, action, device):
    """Rollout to the end of `action`: (B, N, dim), on CPU.

    action is (B, H, N, block*d_a). At H=1 this is one teacher-forced step; past
    that it is the model's own autoregressive rollout, which is what a planner
    would use and the only way to reach a horizon where the cross-agent response
    is large enough to measure.
    """
    start = model.encode(observation.unsqueeze(1).to(device))
    return model.rollout(start, action.to(device))[:, -1].cpu().double()


def readout(probes, latent, witnesses):
    """Latent (B,N,dim) -> physical Y (B, K*4 + P*4) for the witnesses + object."""
    anchors, agents, dim = latent.shape
    per_agent = apply_probe(probes["agent"][0], latent.reshape(-1, dim))
    per_agent = per_agent.reshape(anchors, agents, -1)[:, witnesses]
    shared = apply_probe(probes["object"][0], latent.reshape(anchors, agents * dim))
    return torch.cat([per_agent.reshape(anchors, -1), shared], dim=1)


def true_physical(source, index, step, witnesses):
    """The simulator's own Y at `step`, same column order as `readout`."""
    agents = source["next_agent_state"][index][:, step - 1][:, witnesses, MOTION]
    shared = source["next_package_state"][index][:, step - 1][..., MOTION]
    return torch.cat(
        [agents.reshape(agents.shape[0], -1), shared.reshape(shared.shape[0], -1)],
        dim=1,
    ).double()


def state_input_frames(source, index, step, state_input, history_frames, following):
    """The observation a Stage 2 model expects, at `step`, for rows `index`.

    Mirrors `OfflineSequences._apply_state_input` exactly. Applied to BOTH
    counterfactual branches from each branch's own stored frames, so the
    intervention is the only thing that differs between them.

    `following` selects `next_observation` over `observation`, which is what the
    dynamics target and the probe floor read.
    """
    key = "next_observation" if following else "observation"
    base = source[key][index][:, step]  # (B, N, D)
    if state_input == "observation":
        return base
    if state_input == "physical":
        # One shared world state, given identically to every agent.
        entity_key = "next_package_state" if following else "package_state"
        entities = source[entity_key][index][:, step]
        flat = entities.reshape(entities.shape[0], 1, -1)
        return torch.cat([base, flat.expand(-1, base.shape[1], -1)], dim=-1)
    # history: frame t carries [t, t-1, ..., t-k+1], clamped at the snippet
    # start because an anchor's earlier frames are not stored.
    offsets = torch.arange(history_frames)
    past = source["observation"][index][:, (step - offsets).clamp_min(0)]
    stacked = past.permute(0, 2, 1, 3).reshape(base.shape[0], base.shape[1], -1)
    if not following:
        return stacked
    return torch.cat([base, stacked[..., : -base.shape[-1]]], dim=-1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", type=Path)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--block", type=int, default=5)
    parser.add_argument(
        "--horizon-blocks",
        type=int,
        default=1,
        help="how many action blocks to roll before reading the response. The "
        "intervention flips agent 1's x action for the whole 25-step snippet, "
        "but one block reads it after only 5 primitive steps, where the "
        "physical effect on the other agents measured 0.00346 raw units -- 70x "
        "below a linear probe's own reconstruction error. Sweep this to find "
        "the horizon at which the response is resolvable at all.",
    )
    parser.add_argument(
        "--probe",
        choices=("linear", "mlp"),
        default="linear",
        help="readout family. A linear probe failing cannot distinguish `the "
        "latent does not carry physical state` from `the readout is too weak`; "
        "run both and compare the reported probe resolution.",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--full-state", action="store_true")
    parser.add_argument(
        "--min-response",
        type=float,
        default=1e-8,
        help="anchors whose true scaled response is below this are dropped as "
        "uninformative rather than allowed to manufacture a stable ratio",
    )
    args = parser.parse_args()

    load = lambda name: torch.load(  # noqa: E731
        args.data / name, map_location="cpu", weights_only=True
    )
    samples = load("samples_correlated.pt")
    counterfactual = load("counterfactual_test.pt")
    anchors = load("anchors.pt")
    ids = counterfactual["anchor_id"]
    block = args.block

    observation = samples["observation"][ids, 0]
    n_agents = observation.shape[1]
    witnesses = [i for i in range(n_agents) if i != INTERVENED]
    steps = block * args.horizon_blocks
    logged = blocked(samples["action"][ids, :steps], block)
    intervened = blocked(counterfactual["action"][:, :steps], block)

    live = samples["valid"][ids, :steps].all(dim=1) & counterfactual["valid"][
        :, :steps
    ].all(dim=1)
    _, active = effect_labels(args.data, steps, full_state=args.full_state)
    active = active & live

    # Probes are fitted on anchors no evaluation touches.
    train_rows = (anchors["split"] == 0).nonzero(as_tuple=True)[0]
    train_rows = train_rows[samples["valid"][train_rows, 0]]
    print(
        f"probe training anchors: {train_rows.numel()}   "
        f"test anchors: {ids.numel()}   interaction-active: {int(active.sum())}"
    )
    if not active.any():
        print(json.dumps({"active": 0, "status": "no_active_anchors"}))
        return

    # One scale for every model, from training physical state, so the ratio
    # below is the same quantity no matter which model produced the prediction.
    scale = true_physical(samples, train_rows, steps, witnesses).std(dim=0)
    scale = scale.clamp_min(1e-6)

    truth_id = true_physical(samples, ids, steps, witnesses)
    every = torch.arange(ids.numel())
    truth_cf = true_physical(counterfactual, every, steps, witnesses)
    true_delta = (truth_cf - truth_id) / scale
    informative = active & (true_delta.square().sum(dim=1) > args.min_response)
    print(
        f"informative active anchors (true scaled response above "
        f"{args.min_response:g}): {int(informative.sum())}"
    )

    results = {}
    for directory in sorted(args.runs.glob("[0-9]*")):
        checkpoint = directory / "model.pt"
        if not checkpoint.exists():
            continue
        config = yaml.safe_load((directory / "resolved_config.yaml").read_text())
        # Stage 2's `history` and `physical` conditions rewrite the observation
        # before training, so each checkpoint is fed the frames its own encoder
        # was fitted on. Y, the scale and the anchors are untouched by this, so
        # all three input widths are still scored against the SAME physical
        # target -- which is the whole point of the comparison.
        state_input = config["data"].get("state_input", "observation")
        frames = config["data"].get("history_frames", 3)
        shaped = lambda src, idx, stp, follow: state_input_frames(  # noqa: E731
            src, idx, stp, state_input, frames, follow
        )
        start_frames = shaped(samples, ids, 0, False)
        logged_frames = shaped(samples, ids, steps - 1, True)
        reference_frames = shaped(counterfactual, every, steps - 1, True)
        # state_input must be part of the key: Stage 2 trains the same
        # regime/kind/seed under three input conditions, and collapsing them
        # silently keeps only whichever directory sorted last.
        key = (
            config["data"]["regime"],
            config["model"]["kind"],
            config["seed"],
            state_input,
        )
        model = load_model(checkpoint, args.device)
        model.eval()
        probes = fit_probes(
            model, samples, train_rows, args.device, args.probe, shaped
        )

        # The probe's own ceiling, measured on the SAME quantity as the models:
        # encode both true branches and read the response straight off them. A
        # perfect dynamics model could do no better than this, and comparing the
        # probe's absolute state error against a response magnitude instead
        # would be two different scales.
        floor_delta = (
            readout(probes, encoded(model, reference_frames, args.device), witnesses)
            - readout(probes, encoded(model, logged_frames, args.device), witnesses)
        ) / scale
        probe_floor = (floor_delta - true_delta).square().sum(dim=1)

        predicted = {
            "id": predicted_latent(model, start_frames, logged, args.device),
            "cf": predicted_latent(model, start_frames, intervened, args.device),
        }
        y_id = readout(probes, predicted["id"], witnesses)
        y_cf = readout(probes, predicted["cf"], witnesses)
        predicted_delta = (y_cf - y_id) / scale

        residual = (predicted_delta - true_delta).square().sum(dim=1)
        # Direction, which a magnitude ratio cannot see: a prediction pointing
        # the wrong way can still have the right size.
        cosine = torch.nn.functional.cosine_similarity(
            predicted_delta, true_delta, dim=1, eps=1e-12
        )
        results[key] = {
            "reconstruction": (
                readout(probes, encoded(model, logged_frames, args.device), witnesses)
                - truth_id
            ).norm(dim=1),
            "residual": residual,
            "denominator": true_delta.square().sum(dim=1),
            "absolute": (y_cf - y_id - (truth_cf - truth_id)).norm(dim=1),
            "cosine": cosine,
            "probe_floor": probe_floor,
            "probe_strengths": {n: probes[n][1] for n in probes},
        }

    regimes = sorted({r for r, _, _, _ in results})
    seeds = sorted({s for _, _, s, _ in results})
    conditions = sorted({i for _, _, _, i in results})
    print(f"scored {len(results)} checkpoints over {len(seeds)} seeds")
    print(f"input conditions: {', '.join(conditions)}\n")

    mask = informative
    # Whether the instrument can resolve the thing at all. The response is a
    # small difference between two nearby states; if a probe's absolute
    # reconstruction error is the same size as the response, no comparison
    # built on that probe means anything, however many seeds it averages.
    print("=== CAN THE PROBE RESOLVE THE RESPONSE? ===")
    for condition in conditions:
        rec = mean(
            [
                float(v["reconstruction"][mask].mean())
                for (_, _, _, i), v in results.items()
                if i == condition
            ]
        )
        size = float((true_delta[mask] * scale).norm(dim=1).mean())
        verdict = "  -- the probe cannot resolve it" if rec > size else ""
        print(
            f"  {condition:12s} |true response| {size:.5f}   "
            f"|probe error| {rec:.5f}   {rec / size:.2f}x{verdict}"
        )
    reconstruction = mean(
        [float(v["reconstruction"][mask].mean()) for v in results.values()]
    )
    response_size = float((true_delta[mask] * scale).norm(dim=1).mean())
    print(f"mean |true response|            {response_size:.5f} (raw units)")
    print(f"mean |probe absolute error|     {reconstruction:.5f} (raw units)")
    print(
        f"error / response                {reconstruction / response_size:.2f}x"
        + ("  -- the probe cannot resolve it" if reconstruction > response_size else "")
    )

    for condition in conditions:
        print(f"\n########## input condition: {condition} ##########")
        print("\n=== PHYSICAL RESPONSE RATIO, common coordinates ===")
        print("sum||predicted dY - true dY||^2 / sum||true dY||^2, both scaled by the")
        print("same training std. 1.0 = no better than predicting no response.")
        print("`probe floor` is the same ratio computed from the TRUE encodings of")
        print("both branches: the readout's own error, which bounds what follows.\n")
        header = (
            f"{'regime':12s} {'kind':12s} {'ratio':>9s} {'probe floor':>13s} "
            f"{'|abs err|':>11s} {'cos(dY)':>9s}"
        )
        print(header)
        print("-" * len(header))
        for regime in regimes:
            for kind in BASELINES:
                rows = [v for (r, k, _, i), v in results.items()
                        if (r, k) == (regime, kind) and i == condition]
                if not rows:
                    continue
                ratio = mean(
                    [
                        float(v["residual"][mask].sum() / v["denominator"][mask].sum())
                        for v in rows
                    ]
                )
                floor = mean(
                    [
                        float(v["probe_floor"][mask].sum() / v["denominator"][mask].sum())
                        for v in rows
                    ]
                )
                absolute = mean([float(v["absolute"][mask].mean()) for v in rows])
                direction = mean([float(v["cosine"][mask].mean()) for v in rows])
                print(
                    f"{regime:12s} {kind:12s} {ratio:9.4f} {floor:13.4f} "
                    f"{absolute:11.5f} {direction:9.4f}"
                )

        print("\npaired ratio vs independent (negative = captured more of the response):")
        paired = f"{'regime':12s} {'kind':12s} {'mean':>10s} {'95% CI':>24s} {'seeds':>8s}"
        print(paired)
        print("-" * len(paired))
        for regime in regimes:
            for kind in ("joint", "relational"):
                pairs = []
                for seed in seeds:
                    base = results.get((regime, "independent", seed, condition))
                    other = results.get((regime, kind, seed, condition))
                    if base and other:
                        pairs.append(
                            float(
                                other["residual"][mask].sum()
                                / other["denominator"][mask].sum()
                            )
                            - float(
                                base["residual"][mask].sum()
                                / base["denominator"][mask].sum()
                            )
                        )
                if not pairs:
                    continue
                low, high = bootstrap_interval(pairs, mean)
                print(
                    f"{regime:12s} {kind:12s} {mean(pairs):+10.4f} "
                    f"{f'[{low:+.4f}, {high:+.4f}]':>24s} "
                    f"{f'{sum(1 for v in pairs if v < 0)}/{len(pairs)}':>8s}"
                )

    print(
        "\n"
        + json.dumps(
            {
                "test_anchors": int(ids.numel()),
                "active": int(active.sum()),
                "informative": int(mask.sum()),
                "probe_training_anchors": int(train_rows.numel()),
                "output_dimensions": int(true_delta.shape[1]),
            }
        )
    )


if __name__ == "__main__":
    main()
