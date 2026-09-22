#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#
"""T-A2 stage 2: effect-normalized counterfactual fidelity across H0/H1/H2.

T-A1 measured the TRUE interaction Jacobian on the simulator. This scores every
trained model against exactly those interventions:

    E_CF = || dY_predicted - dY_true || / ( || dY_true || + eps )

in the common scaled physical coordinates `physical_response.py` established. A
model that predicts no response at all scores exactly 1, so `independent` is the
floor on every cross cell by construction rather than a competitor. `E_CF < 1`
means the prediction error is smaller than the effect being resolved.

The interventions are NOT read from a file. They are regenerated from the same
seed through the same helpers `interaction_jacobian` used, and the regenerated
true effects are then required to reproduce that run's recorded per-cell means.
A drifted seed, horizon or action convention fails loudly here instead of
silently scoring models against a different counterfactual than the one whose
floor was registered.

Three numbers accompany every cell, because `E_CF < 1` alone does not rescue a
measurement whose instrument is coarser than its effect -- that is exactly how
the Balance response claim (K8) had to be retired:

  1. the true effect size, in shared scaled units;
  2. the probe floor -- the same response read off TRUE encodings, which is what
     a perfect dynamics model would score;
  3. their ratio, against the registered 3x resolution rule.

Run:
    python -m examples.world_model.counterfactual_fidelity \\
        outputs/ta2_reference_baselines_1498/runs \\
        --data outputs/buzz_wire_1196/data \\
        --jacobian outputs/ta1_interaction_jacobian_1497/result/interaction_jacobian.json \\
        --out outputs/ta2_fidelity --device cuda
"""

import argparse
import json
from pathlib import Path

import torch
import yaml

from examples.world_model.counterfactual_evaluation import blocked
from examples.world_model.interaction_jacobian import (
    SPLITS,
    action_bounds,
    bootstrap_by_episode,
    constant_plan,
    cumulative_valid,
    load_bank,
    resolve_task,
    training_scale,
)
from examples.world_model.collect import branch_rollouts, sample_actions
from examples.world_model.physical_response import (
    MOTION,
    encoded,
    fit_probes,
    predicted_latent,
    readout,
    state_input_frames,
)
from examples.world_model.plan_ranking import select_anchor_states
from examples.world_model.train import load_model

# The registered T-A1 rule: a cell is usable only if its true effect clears the
# probe floor by this factor. Set by the experiment 14/16 precedent, not tuned.
RESOLUTION_RATIO = 3.0
# Regenerated interventions must reproduce T-A1's recorded means this closely.
JACOBIAN_AGREEMENT = 1e-4


def is_usable(relative_probe_floor):
    """The registered T-A1 rule, in the units the evaluator actually reports.

    `E_CF` and the probe floor are both divided by ||dY_true||, so the reported
    floor IS the inverse resolution: "true effect >= 3x probe floor" is exactly
    "relative floor <= 1/3". A floor at or above 1 means a model predicting no
    response scores better than the instrument's own ceiling, and no ordering
    built on that cell means anything.
    """
    return relative_probe_floor <= 1.0 / RESOLUTION_RATIO


def physical_target(source, index, step, agents):
    """Simulator Y at `step`: every agent's motion, then every shared body's.

    Unlike `physical_response.true_physical` this keeps ALL agents rather than
    only the non-intervened ones, because T-A2 scores the diagonal (self) blocks
    of the Jacobian as well as the off-diagonal (cross) ones.
    """
    per_agent = source["next_agent_state"][index][:, step - 1][:, agents, MOTION]
    shared = source["next_package_state"][index][:, step - 1][..., MOTION]
    return torch.cat(
        [per_agent.reshape(per_agent.shape[0], -1), shared.reshape(shared.shape[0], -1)],
        dim=1,
    ).double()


def collect_branches(task, sub, count, low, high, args, steps):
    """Regenerate T-A1's interventions: reference and both endpoints per cell.

    Deterministic in `args.seed`, so this reproduces the branches T-A1 scored
    without needing them archived. Returns {(reference, cell): {low, high}} plus
    the reference rollouts, all on CPU.
    """
    agents, action_dim = low.shape
    branches, references = {}, {}
    for reference in range(args.references):
        base = sample_actions(
            (count, 1, agents, action_dim), low, high, "independent", args.seed + reference
        )[:, 0]
        references[reference] = branch_rollouts(
            task, sub, constant_plan(base, steps), args.batch_size, args.device
        )
        for intervened in range(agents):
            for axis in range(action_dim):
                endpoints = {}
                for name, bound in (("low", low), ("high", high)):
                    moved = base.clone()
                    moved[:, intervened, axis] = bound[intervened, axis]
                    endpoints[name] = branch_rollouts(
                        task, sub, constant_plan(moved, steps), args.batch_size, args.device
                    )
                branches[(reference, intervened, axis)] = endpoints
        print(f"  regenerated reference {reference}", flush=True)
    return branches, references


def scaled_truth(endpoints, agents, step, scale):
    """True scaled dY and the joint live mask at `step`."""
    low_y = physical_target(endpoints["low"], slice(None), step + 1, agents)
    high_y = physical_target(endpoints["high"], slice(None), step + 1, agents)
    live = cumulative_valid(endpoints["low"]["valid"] & endpoints["high"]["valid"])
    return (high_y - low_y) / scale, live[:, step]


def verify_against_jacobian(branches, agents, block, scale_agent, recorded, horizon):
    """Require the regenerated branches to reproduce T-A1's recorded cell means.

    T-A1 reports each responding body separately, so this checks the per-body
    magnitudes rather than the concatenated vector the models are scored on.
    """
    step = horizon * block - 1
    observed = {}
    for (reference, intervened, axis), endpoints in branches.items():
        delta = (
            endpoints["high"]["next_agent_state"][..., MOTION]
            - endpoints["low"]["next_agent_state"][..., MOTION]
        ) / scale_agent
        live = cumulative_valid(endpoints["low"]["valid"] & endpoints["high"]["valid"])
        mask = live[:, step]
        for responder in range(agents):
            key = f"a{intervened}_axis{axis}__agent_{responder}__h{horizon}"
            observed.setdefault(key, []).append(delta[:, step, responder].norm(dim=-1)[mask])

    disagreements = {}
    for key, values in observed.items():
        if key not in recorded:
            continue
        got = float(torch.cat(values).mean())
        want = recorded[key]["mean"]
        if abs(got - want) > JACOBIAN_AGREEMENT * max(abs(want), 1.0):
            disagreements[key] = {"regenerated": got, "t_a1": want}
    if disagreements:
        raise ValueError(
            "Regenerated interventions do not reproduce T-A1. The seed, horizon "
            f"or action convention has drifted: {disagreements}"
        )
    return len(observed)


def resolve_regime_files(data_root, anchors, rows, observed):
    """Map each `source_regime` index to its trajectory, from the MANIFEST.

    The producer records the resolved list. `collect.py` builds
    `source_regimes = list(REGIMES)` and appends `"heuristic"` when that source
    is enabled, then numbers them by position, and writes the result to the
    manifest as `source_regimes`. So the mapping is stated by the bank itself
    and does not have to be inferred.

    Two earlier attempts were wrong. A hardcoded two-entry tuple, validated on
    Buzz Wire, raised `KeyError: 2` on Transport and Balance, which carry a
    third heuristic source (job 1521). Inferring the mapping by matching frames
    then proved ambiguous, because the regimes branch from a shared anchor bank
    and their observations coincide at many anchors.

    The manifest is authoritative; the frame check is retained only to catch a
    manifest that disagrees with the files beside it.
    """
    manifest = json.loads((data_root / "manifest.json").read_text())
    names = manifest.get("source_regimes")
    if not names:
        raise ValueError(
            f"{data_root}/manifest.json records no `source_regimes`; the index "
            "order cannot be recovered and must not be guessed"
        )
    present = sorted(set(anchors["source_regime"][rows].tolist()))
    missing = [i for i in present if i >= len(names)]
    if missing:
        raise ValueError(
            f"source_regime {missing} exceeds the manifest's {names}"
        )

    mapping, episode, step = {}, anchors["episode_id"][rows], anchors["source_step"][rows]
    regime = anchors["source_regime"][rows]
    for index in present:
        path = data_root / f"trajectories_{names[index]}.pt"
        if not path.exists():
            raise FileNotFoundError(
                f"manifest names source_regime {index} as '{names[index]}' but "
                f"{path.name} is absent"
            )
        source = torch.load(path, map_location="cpu", weights_only=True)
        here = (regime == index).nonzero().squeeze(-1)
        probe = here[: min(32, len(here))]
        looked_up = torch.stack(
            [source["observation"][int(e), int(s)]
             for e, s in zip(episode[probe], step[probe])]
        )
        if not torch.equal(looked_up, observed[probe]):
            raise ValueError(
                f"{path.name} disagrees with the anchors it is supposed to "
                f"describe for source_regime {index}: max difference "
                f"{float((looked_up - observed[probe]).abs().max()):.3e}"
            )
        mapping[index] = source
    return mapping


def reference_context(anchors, rows, data_root, history_size, block, observed):
    """The model's OWN context layout at a restored anchor: block-strided.

    Audit F13: `rolled_latent` used to hand `lewm_reference` the episode-start
    convention -- current frame repeated, zero past actions -- at anchors that
    are 73% mid-episode. Training always fed real windows, so the model was
    evaluated out of distribution.

    The stride matters and is easy to get wrong. `dataset.py` builds the
    reference sequence as ``[observation[0]] + next_observation[block-1::block]``
    with actions blocked to ``(L, N, block*A)``, so a frame is a BLOCK BOUNDARY.
    A 3-frame context spans ``2*block`` primitive steps, not 2. Reusing the
    stride-1 window that the G0 diagnostic uses would reproduce F13 in a new
    place.

    Returns observations ``(B, history_size, N, O)`` and blocked past actions
    ``(B, history_size-1, N, block*A)``, both ending at the anchor.
    """
    episode = anchors["episode_id"][rows]
    step = anchors["source_step"][rows]
    regime = anchors["source_regime"][rows]
    trajectories = resolve_regime_files(data_root, anchors, rows, observed)

    frames, actions = [], []
    for e, s, r in zip(episode.tolist(), step.tolist(), regime.tolist()):
        source = trajectories[r]
        frame_steps = [max(0, s - offset * block) for offset in reversed(range(history_size))]
        frames.append(source["observation"][e, frame_steps])
        # Each past position carries the `block` primitive actions that produced
        # the frame after it, in the same time-then-coordinate order `blocked`
        # uses, so training and evaluation agree on the layout.
        #
        # A position whose SOURCE frame was clamped describes an artificial
        # padding transition (o_0 -> o_0), and the registered convention in
        # `model_input.PlanningContext` is explicit that "observation is
        # repeated and historical actions are zero". Carrying a real action
        # across a padded transition would tell the predictor that motion
        # occurred between two identical frames.
        window = []
        for offset in reversed(range(1, history_size)):
            start = s - offset * block
            if start < 0:
                window.append(torch.zeros(block, *source["action"].shape[2:]))
                continue
            span = [min(start + i, source["action"].shape[1] - 1) for i in range(block)]
            window.append(source["action"][e, span])  # (block, N, A)
        stacked = torch.stack(window)  # (history_size-1, block, N, A)
        agents, action_dim = stacked.shape[2], stacked.shape[3]
        actions.append(
            stacked.permute(0, 2, 1, 3).reshape(history_size - 1, agents, block * action_dim)
        )
    frames = torch.stack(frames)
    actions = torch.stack(actions)

    if not torch.equal(frames[:, -1], observed):
        raise ValueError(
            "Reference context is misaligned with the anchor it describes: the "
            f"frame at source_step differs by {float((frames[:, -1] - observed).abs().max()):.3e}"
        )
    return frames, actions


@torch.no_grad()
def rolled_latent(model, observation, plan, device, context=None, source_step=None):
    """Roll `plan` from a restored anchor, honouring the model's own profile.

    `lewm_reference` refuses a one-frame rollout by design (A0 step 2), because
    silently reconstructing history from the current frame is the defect that
    guard exists to prevent.

    A restored mid-episode anchor **does** have a past: it lives in the source
    trajectory, and the branch object simply does not carry it. That anchor must
    therefore be supplied its authentic source-trajectory context. The synthetic
    repeated-frame / null-action context is valid **only** at a genuine episode
    boundary. Asserting otherwise is what audit F13 records.
    """
    if getattr(model, "profile", "legacy_compact") != "lewm_reference":
        return predicted_latent(model, observation, plan, device)

    frames = model.history_size
    if context is None:
        # Registered contract (audit F13): only a genuine episode start may use
        # the synthetic repeated-frame context. A mid-episode anchor has a real
        # past and must be given it.
        if source_step is not None and int((source_step > 0).sum()):
            raise ValueError(
                f"{int((source_step > 0).sum())} anchors have source_step > 0 but "
                "no real context was supplied; synthetic-start context is "
                "forbidden off an episode boundary (audit F13)"
            )
        history = observation.unsqueeze(1).expand(-1, frames, -1, -1).to(device)
        past_actions = torch.zeros(
            plan.shape[0], frames - 1, plan.shape[2], plan.shape[3], device=device
        )
    else:
        history, past_actions = context
        history = history.to(device)
        past_actions = past_actions.to(device)
    latent_history = model.encode(history)
    rolled = model.rollout_from_context(latent_history, past_actions, plan.to(device))
    return rolled[:, -1].cpu().double()


def column_groups(agents, shared_bodies, intervened):
    """Which Y columns belong to the self, cross and shared blocks.

    Y is [agent_0 motion, ..., agent_{N-1} motion, shared_0, ..., shared_{M-1}],
    four columns each. E_CF must be computed per block: an off-diagonal claim
    pooled with the diagonal would be dominated by self-dynamics, which every
    arm including `independent` can represent.
    """
    width = len(range(*MOTION.indices(6)))
    groups = {"self": [], "cross": [], "shared": []}
    for agent in range(agents):
        columns = list(range(agent * width, (agent + 1) * width))
        groups["self" if agent == intervened else "cross"].extend(columns)
    for body in range(shared_bodies):
        base = agents * width + body * width
        groups["shared"].extend(range(base, base + width))
    return {name: torch.tensor(cols) for name, cols in groups.items() if cols}


def score_checkpoint(directory, args, samples, train_rows, branches, agents,
                     shared_bodies, block, scale, horizon, context=None,
                     source_step=None):
    """One model's E_CF, probe floor and direction on every Jacobian block."""
    config = yaml.safe_load((directory / "resolved_config.yaml").read_text())
    state_input = config["data"].get("state_input", "observation")
    frames = config["data"].get("history_frames", 3)
    shaped = lambda src, idx, stp, follow: state_input_frames(  # noqa: E731
        src, idx, stp, state_input, frames, follow
    )
    model = load_model(directory / "model.pt", args.device)
    model.eval()
    probes = fit_probes(model, samples, train_rows, args.device, args.probe, shaped)

    step = horizon * block - 1
    every = slice(None)
    all_agents = list(range(agents))
    cells = {}
    for (reference, intervened, axis), endpoints in branches.items():
        true_delta, mask = scaled_truth(endpoints, all_agents, step, scale)
        predicted, floor = {}, {}
        for name in ("low", "high"):
            source = endpoints[name]
            start = shaped(source, every, 0, False)
            plan = blocked(source["action"][:, : step + 1], block)
            latent = rolled_latent(
                model, start, plan, args.device, context, source_step
            )
            predicted[name] = readout(probes, latent, all_agents)
            # The probe's own ceiling on the SAME quantity: read the response
            # straight off TRUE encodings. A perfect dynamics model scores this.
            floor[name] = readout(
                probes,
                encoded(model, shaped(source, every, step, True), args.device),
                all_agents,
            )
        predicted_delta = (predicted["high"] - predicted["low"]) / scale
        floor_delta = (floor["high"] - floor["low"]) / scale

        for name, columns in column_groups(agents, shared_bodies, intervened).items():
            true_block = true_delta[:, columns]
            denominator = true_block.norm(dim=1) + 1e-12
            cells[(reference, intervened, axis, name)] = {
                # ||dY_predicted|| / ||dY_true||. With E_CF and cosine this
                # separates a response of the wrong SIZE from one pointing the
                # wrong WAY -- two failures a single ratio cannot tell apart.
                "magnitude_ratio": (
                    predicted_delta[:, columns].norm(dim=1) / denominator
                )[mask],
                "e_cf": (
                    (predicted_delta[:, columns] - true_block).norm(dim=1) / denominator
                )[mask],
                "probe_floor": (
                    (floor_delta[:, columns] - true_block).norm(dim=1) / denominator
                )[mask],
                "true_size": true_block.norm(dim=1)[mask],
                "cosine": torch.nn.functional.cosine_similarity(
                    predicted_delta[:, columns], true_block, dim=1, eps=1e-12
                )[mask],
                "mask": mask,
            }
    return config, cells, {name: probes[name][1] for name in probes}


def summarize(scored, episode_ids, args):
    """Pool by block type, apply the registered resolution rule, compare arms."""
    kinds = sorted({k for _, k, _ in scored})
    regimes = sorted({r for r, _, _ in scored})
    seeds = sorted({s for _, _, s in scored})
    results = {
        "checkpoints": len(scored),
        "kinds": kinds,
        "regimes": regimes,
        "seeds": seeds,
        "horizon_blocks": args.horizon,
        "probe": args.probe,
        "resolution_ratio_rule": RESOLUTION_RATIO,
        "per_arm": {},
        "paired": {},
    }

    def pooled(run_results, block_type):
        keys = ("e_cf", "probe_floor", "true_size", "cosine", "magnitude_ratio")
        parts = {key: [] for key in keys}
        parts["ids"] = []
        for run_result in run_results:
            for (_, _, _, name), cell in run_result["cells"].items():
                if name != block_type:
                    continue
                for key in keys:
                    parts[key].append(cell[key])
                parts["ids"].append(episode_ids[cell["mask"]])
        if not parts["e_cf"]:
            return None
        return {key: torch.cat(value) for key, value in parts.items()}

    for regime in regimes:
        for kind in kinds:
            runs = [scored[(regime, kind, s)] for s in seeds if (regime, kind, s) in scored]
            if not runs:
                continue
            for block_type in ("self", "cross", "shared"):
                pool = pooled(runs, block_type)
                if pool is None:
                    continue
                summary = bootstrap_by_episode(pool["e_cf"], pool["ids"], seed=args.seed + 500)
                # Both E_CF and the floor are already divided by ||dY_true||, so
                # the floor IS the inverse resolution: the registered "effect
                # >= 3x probe floor" rule is exactly floor <= 1/3.
                floor_mean = float(pool["probe_floor"].mean())
                summary.update(
                    {
                        "probe_floor_mean": floor_mean,
                        "true_effect_size_mean": float(pool["true_size"].mean()),
                        "resolution_ratio": 1.0 / max(floor_mean, 1e-12),
                        "cosine_mean": float(pool["cosine"].mean()),
                        "magnitude_ratio_mean": float(pool["magnitude_ratio"].mean()),
                        # E_CF if the gain were corrected SEPARATELY AT EVERY
                        # ANCHOR: min over r of ||r*u - v||/||v|| is
                        # sqrt(1 - cos^2), attained at a per-anchor r. This is
                        # an ORACLE bound -- it needs the true response to pick
                        # each r, so it is not deployable and is not a claim
                        # that one global scalar would do as well. It isolates
                        # how much of the gap is direction and how much is gain.
                        "oracle_per_anchor_gain_e_cf": float(
                            (1.0 - pool["cosine"].clamp(-1, 1) ** 2).clamp_min(0).sqrt().mean()
                        ),
                        "fraction_below_one": float((pool["e_cf"] < 1.0).double().mean()),
                        "usable_by_registered_rule": is_usable(floor_mean),
                    }
                )
                results["per_arm"][f"{regime}__{kind}__{block_type}"] = summary

    # Paired seed differences on the CROSS block: the registered comparison.
    # `independent` is structurally zero there, so this is where the hypotheses
    # actually differ.
    def cross_mean(run_result):
        values = [
            cell["e_cf"]
            for (_, _, _, name), cell in run_result["cells"].items()
            if name == "cross"
        ]
        return float(torch.cat(values).mean()) if values else float("nan")

    for regime in regimes:
        for better, worse in (("joint", "independent"), ("relational", "joint")):
            diffs, wins = [], 0
            for seed in seeds:
                a, b = (regime, better, seed), (regime, worse, seed)
                if a not in scored or b not in scored:
                    continue
                delta = cross_mean(scored[a]) - cross_mean(scored[b])
                diffs.append(delta)
                wins += delta < 0
            if diffs:
                results["paired"][f"{regime}__{better}_vs_{worse}"] = {
                    "seeds": len(diffs),
                    f"wins_for_{better}": wins,
                    "mean_delta_e_cf": sum(diffs) / len(diffs),
                    "per_seed_delta": diffs,
                }
    return results


def run(args):
    data_root = Path(args.data)
    anchors, manifest = load_bank(data_root)
    task, _ = resolve_task(manifest)
    block = args.block or manifest.get("action_block", 5)
    steps = args.horizon * block

    rows = (anchors["split"] == SPLITS.index(args.split)).nonzero().squeeze(-1)
    if args.max_anchors:
        rows = rows[: args.max_anchors]
    episode_ids = anchors["episode_id"][rows]
    sub = {"snapshot": select_anchor_states(anchors, rows, "cpu")}

    low, high = action_bounds(task, args.device)
    agents = low.shape[0]
    agent_scale, shared_scale = training_scale(
        data_root, anchors, ("correlated", "independent")
    )
    samples = torch.load(
        data_root / "samples_correlated.pt", map_location="cpu", weights_only=True
    )
    shared_bodies = samples["next_package_state"].shape[2]
    # Y is [every agent's motion, every shared body's motion]; one scale vector,
    # fitted on training rows and shared by every arm.
    scale = torch.cat(
        [agent_scale.repeat(agents), shared_scale.repeat(shared_bodies)]
    ).double()

    print(
        f"T-A2 stage 2: {len(rows)} {args.split} anchors, horizon {args.horizon} "
        f"block(s), {agents} agents, {shared_bodies} shared bodies",
        flush=True,
    )
    branches, _ = collect_branches(task, sub, len(rows), low, high, args, steps)

    recorded = json.loads(Path(args.jacobian).read_text())["cells"]
    checked = verify_against_jacobian(
        branches, agents, block, agent_scale, recorded, args.horizon
    )
    print(f"  regenerated interventions reproduce {checked} T-A1 cells", flush=True)

    # Audit F13: give the reference predictor the context it was trained on.
    source_step = anchors["source_step"][rows]
    context = None
    results_context = {
        "context_mode": "synthetic_episode_start",
        "context_history_frames": int(args.history_frames),
        "context_block_stride": None,
        "mid_episode_anchors": int((source_step > 0).sum()),
        "episode_start_anchors": int((source_step == 0).sum()),
        "padded_transitions_use_zero_actions": True,
        "superseded_by": "audit F13 -- pre-repair behaviour, retained only to regenerate old numbers",
    }
    if not args.synthetic_start_context:
        any_branch = next(iter(branches.values()))["low"]
        context = reference_context(
            anchors, rows, data_root, args.history_frames, block,
            any_branch["observation"][:, 0],
        )
        mid = int((source_step > 0).sum())
        # Artifact self-description (audit): six months from now the JSON alone
        # must say whether a result is pre- or post-F13.
        results_context = {
            "context_mode": "authentic_block_strided",
            "context_history_frames": int(args.history_frames),
            "context_block_stride": int(block),
            "mid_episode_anchors": mid,
            "episode_start_anchors": int(len(rows) - mid),
            "padded_transitions_use_zero_actions": True,
        }
        print(
            f"  real reference context: frames {tuple(context[0].shape)}, "
            f"past actions {tuple(context[1].shape)}, stride {block}, "
            f"{mid}/{len(rows)} anchors mid-episode, alignment verified",
            flush=True,
        )

    # Probes are fitted on anchors no evaluation touches.
    train_rows = (anchors["split"] == 0).nonzero(as_tuple=True)[0]
    train_rows = train_rows[samples["valid"][train_rows, 0]]

    scored = {}
    for directory in sorted(Path(args.runs).iterdir()):
        if not (directory / "model.pt").exists():
            continue
        config, cells, strengths = score_checkpoint(
            directory, args, samples, train_rows, branches,
            agents, shared_bodies, block, scale, args.horizon,
            context, source_step,
        )
        key = (config["data"]["regime"], config["model"]["kind"], config["seed"])
        scored[key] = {"cells": cells, "probe_strengths": strengths}
        print(f"  scored {directory.name}", flush=True)
    if not scored:
        raise FileNotFoundError(f"No checkpoints with model.pt under {args.runs}")

    results = summarize(scored, episode_ids, args)
    results["context"] = results_context
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    (output / "counterfactual_fidelity.json").write_text(json.dumps(results, indent=2))
    print(f"\nWrote {output / 'counterfactual_fidelity.json'}")
    print_report(results)
    return results


def print_report(results):
    print("\nE_CF by arm (lower is better; 1.0 = predicts no response at all)\n")
    for name, arm in sorted(results["per_arm"].items()):
        print(
            f"  {name:<40} E_CF {arm['mean']:.4f} [{arm['low']:.4f}, {arm['high']:.4f}]  "
            f"floor {arm['probe_floor_mean']:.4f}  "
            f"cos {arm['cosine_mean']:+.3f}  mag {arm['magnitude_ratio_mean']:.3f}  "
            f"oracle-gain {arm['oracle_per_anchor_gain_e_cf']:.3f}  "
            f"<1 in {arm['fraction_below_one']:.3f}  n={arm['anchors']}"
        )
    print("\nPaired seed comparisons (negative delta favours the first arm)\n")
    for name, pair in sorted(results["paired"].items()):
        key = [k for k in pair if k.startswith("wins_for_")][0]
        print(
            f"  {name:<40} delta {pair['mean_delta_e_cf']:+.4f}  "
            f"{pair[key]}/{pair['seeds']} seeds"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path, help="T-A2 stage 1 run directory")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--jacobian", type=Path, required=True, help="T-A1 result JSON")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--split", default="test", choices=SPLITS)
    parser.add_argument("--horizon", type=int, default=1, help="blocks")
    parser.add_argument("--block", type=int, default=None)
    parser.add_argument("--references", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7301)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-anchors", type=int, default=None)
    parser.add_argument("--probe", choices=("linear", "mlp"), default="linear")
    parser.add_argument("--history-frames", type=int, default=3)
    parser.add_argument(
        "--synthetic-start-context",
        action="store_true",
        help="reproduce the pre-F13 behaviour: repeated current frame and null "
        "past actions at every anchor. Retained only to regenerate the "
        "superseded numbers; it is wrong off an episode boundary.",
    )
    parser.add_argument("--device", default="cpu")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
