#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#
"""Cross-scenario admission benchmark: which tasks deserve H0/H1/H2 training?

Buzz Wire taught us that information sufficiency must be checked *before*
architecture: its cross-agent effect is large and real, and no model could learn
it because the observation omits the mediating body. Spending 48 checkpoints per
task to rediscover that is the mistake this module exists to prevent.

One protocol, run identically on every scenario, producing one comparable
ADMISSION REPORT. No world model is trained.

**Stage 0 - interaction structure.** From each restored anchor, hold every
action coordinate at a sampled reference and sweep BOTH agents over
``{-1, 0, +1}`` on one axis, giving a 3x3 joint-action surface. From that single
grid:

    J_own    = Y(+,0) - Y(-,0)        first-order own effect
    J_cross  = Y(0,+) - Y(0,-)        cross-agent effect, the multi-agent content
    C        = Y(+,+) - Y(+,-) - Y(-,+) + Y(-,-)
                                      MIXED interaction: the second difference,
                                      which is zero for any additively separable
                                      response no matter how large J_cross is

State dependence is the spread of the per-anchor effect across anchors. A large
but *constant* effect is uninteresting: the actions-only baseline learns it.

**Stage 1 - information sufficiency.** Fit one diagnostic family to predict the
true cross effect from increasingly informative inputs,

    A -> O -> H_dense -> H_model -> S

and report ``R_info(X) = (E_A - E_X) / (E_A - E_S)``. ``S`` is privileged state
used ONLY as a diagnostic reference, never as a model input. ``H_model`` samples
history at the world model's own block stride, which is a different information
representation from dense history -- dense windows mostly expose instantaneous
velocity, block-strided windows expose slower coupled motion.

**Y is task-specific by construction.** Every body's full
``(pos, vel, rot, ang_vel)`` is used, scaled by its training standard deviation,
and columns that are constant on training data are dropped. Wheel's line is
pinned and only rotates, so a position-only target would read exactly zero there
however much it turns; Buzz Wire's ball moves translationally. Forcing one
task's physical target onto every scenario is how Wheel was previously recorded
as having no interaction.

Run:
    python -m examples.world_model.admission_benchmark \\
        --scenario buzz_wire=outputs/buzz_wire_1196/data \\
        --scenario transport=outputs/transport_data_1190 \\
        --out outputs/admission --device cuda
"""

import argparse
import json
from pathlib import Path

import torch

from examples.world_model.collect import branch_rollouts, sample_actions
from examples.world_model.counterfactual_fidelity import reference_context
from examples.world_model.counterfactual_localization import (
    history_window,
    score,
    select_and_fit,
)
from examples.world_model.interaction_jacobian import (
    SPLITS,
    action_bounds,
    bootstrap_by_episode,
    constant_plan,
    cumulative_valid,
    load_bank,
    resolve_task,
)
from examples.world_model.plan_ranking import select_anchor_states

# Registered admission thresholds, fixed before any scenario is measured.
RESOLUTION_RATIO = 3.0        # effect must clear the probe floor by this factor
ACTIVE_FRACTION = 0.5         # ...on at least this fraction of anchors
INFO_RECOVERY_GATE = 0.5      # a legitimate input must recover this much of the gap
# The privileged reference must beat the actions-only baseline by this fraction
# for a recovery ratio to mean anything at all.
REFERENCE_SPAN_FLOOR = 0.10
LEVELS = (-1.0, 0.0, 1.0)


def task_scale(data_root, anchors, regimes):
    """Per-column std of the FULL body state on training anchors, plus a mask.

    All six columns, not just position/velocity: Wheel's line is pinned and only
    rotates, so a motion-only target is identically zero there. Columns that are
    constant on training data carry no information and are dropped rather than
    contributing divide-by-epsilon noise.
    """
    train = anchors["split"] == SPLITS.index("train")
    agent_rows, shared_rows = [], []
    for regime in regimes:
        path = data_root / f"samples_{regime}.pt"
        if not path.exists():
            continue
        samples = torch.load(path, map_location="cpu", weights_only=True)
        rows = train[: samples["next_agent_state"].shape[0]]
        valid = samples["valid"][rows]
        agent = samples["next_agent_state"][rows]
        shared = samples["next_package_state"][rows]
        agent_rows.append(agent[valid].reshape(-1, agent.shape[-1]))
        shared_rows.append(shared[valid].reshape(-1, shared.shape[-1]))
    if not agent_rows:
        raise FileNotFoundError(f"No samples_<regime>.pt under {data_root}")
    out = {}
    for name, rows in (("agent", agent_rows), ("shared", shared_rows)):
        std = torch.cat(rows).std(dim=0)
        out[name] = (std.clamp_min(1e-8), std > 1e-6)
    return out


def body_target(source, step, scale):
    """Scaled body state at `step`: (B, N*k + M*k) over informative columns."""
    (agent_std, agent_keep), (shared_std, shared_keep) = scale["agent"], scale["shared"]
    agent = source["next_agent_state"][:, step - 1] / agent_std
    shared = source["next_package_state"][:, step - 1] / shared_std
    agent = agent[..., agent_keep]
    shared = shared[..., shared_keep]
    return torch.cat(
        [agent.reshape(agent.shape[0], -1), shared.reshape(shared.shape[0], -1)], dim=1
    ).double()


def grid_rollouts(task, sub, count, low, high, axis, args, steps):
    """The 3x3 joint-action surface on `axis`, plus the live mask.

    Both agents move on the SAME axis so the mixed second difference is defined.
    Every other coordinate stays at the sampled reference, so this is a surface
    through one action point rather than a sweep of everything at once.
    """
    agents, action_dim = low.shape
    mid = (low + high) / 2
    half = (high - low) / 2
    surface, live = {}, None
    for reference in range(args.references):
        base = sample_actions(
            (count, 1, agents, action_dim), low, high, "independent",
            args.seed + reference,
        )[:, 0]
        for a_level in LEVELS:
            for b_level in LEVELS:
                action = base.clone()
                action[:, 0, axis] = mid[0, axis] + a_level * half[0, axis]
                if agents > 1:
                    action[:, 1, axis] = mid[1, axis] + b_level * half[1, axis]
                data = branch_rollouts(
                    task, sub, constant_plan(action, steps), args.batch_size, args.device
                )
                surface[(reference, a_level, b_level)] = data
                mask = cumulative_valid(data["valid"])
                live = mask if live is None else (live & mask)
    return surface, live


def body_columns(scale, agents, shared_bodies):
    """Which Y columns belong to each agent and to the shared bodies.

    `J_ij(s) = d_{a_j} Y_i` is defined on ONE body's state. Measuring it on the
    whole concatenated Y would mix the intervened agent's own large response
    into the cross term and make every task look strongly coupled.
    """
    width = int(scale["agent"][1].sum())
    shared_width = int(scale["shared"][1].sum())
    groups = {}
    for agent in range(agents):
        groups[f"agent_{agent}"] = torch.arange(agent * width, (agent + 1) * width)
    base = agents * width
    groups["shared"] = torch.arange(base, base + shared_bodies * shared_width)
    return groups


def interaction_quantities(surface, live, step, scale, references, agents,
                           shared_bodies, episode_ids):
    """J_own, J_cross and the mixed second difference C, decomposed by responder.

    Agent 0 is the reference "self": J_own is its response to its OWN action,
    J_cross is its response to agent 1's action, and C is the mixed second
    difference on the same body. `shared` is reported separately because the
    jointly controlled object belongs to neither agent.
    """
    def Y(reference, a, b):
        return body_target(surface[(reference, a, b)], step + 1, scale)

    mask = live[:, step]
    groups = body_columns(scale, agents, shared_bodies)
    out = {}
    for responder, columns in groups.items():
        parts = {"J_own": [], "J_cross": [], "C": [], "ids": []}
        for reference in range(references):
            take = lambda a, b: Y(reference, a, b)[:, columns][mask]  # noqa: E731
            parts["J_own"].append(take(1.0, 0.0) - take(-1.0, 0.0))
            parts["J_cross"].append(take(0.0, 1.0) - take(0.0, -1.0))
            parts["C"].append(
                take(1.0, 1.0) - take(1.0, -1.0) - take(-1.0, 1.0) + take(-1.0, -1.0)
            )
            parts["ids"].append(episode_ids[mask])
        out[responder] = {k: torch.cat(v) for k, v in parts.items()}
    return out


def describe(effect, ids, seed):
    """Magnitude, uncertainty, activity and STATE DEPENDENCE of an effect."""
    magnitude = effect.norm(dim=1)
    summary = bootstrap_by_episode(magnitude, ids, seed=seed)
    summary["active_fraction"] = float((magnitude > 1e-6).double().mean())
    summary["median"] = float(magnitude.median()) if magnitude.numel() else float("nan")
    # A large but CONSTANT effect is uninteresting: an actions-only head learns
    # it. Dispersion relative to the mean is what makes the effect a function of
    # state rather than of the action alone.
    summary["state_dependence"] = (
        float(magnitude.std() / magnitude.mean().clamp_min(1e-12))
        if magnitude.numel() > 1 else float("nan")
    )
    return summary


def ladder_inputs(data_root, anchors, anchor_rows, branch, mask, args, block):
    """The five inputs, built once and reused by every target."""
    actions = torch.cat(
        [
            branch["low"]["action"][:, 0].reshape(branch["low"]["action"].shape[0], -1),
            branch["high"]["action"][:, 0].reshape(branch["high"]["action"].shape[0], -1),
        ],
        dim=1,
    ).double()
    reference_frame = branch["low"]["observation"][:, 0]
    observation = reference_frame.reshape(reference_frame.shape[0], -1).double()
    privileged = torch.cat(
        [
            branch["low"]["agent_state"][:, 0].reshape(actions.shape[0], -1),
            branch["low"]["package_state"][:, 0].reshape(actions.shape[0], -1),
        ],
        dim=1,
    ).double()
    dense = history_window(
        anchors, anchor_rows, data_root, args.dense_frames, reference_frame
    )
    frames, past = reference_context(
        anchors, anchor_rows, data_root, args.model_frames, block, reference_frame
    )
    model_history = torch.cat(
        [frames.reshape(frames.shape[0], -1), past.reshape(past.shape[0], -1)], dim=1
    ).double()
    return {
        "A": actions[mask],
        "O": torch.cat([observation, actions], 1)[mask],
        "H_dense": torch.cat([dense, actions], 1)[mask],
        "H_model": torch.cat([model_history, actions], 1)[mask],
        "S": torch.cat([privileged, actions], 1)[mask],
    }


def information_ladder(rows, targets, ids, label, args):
    """A -> O -> H_dense -> H_model -> S for ONE target.

    Run per target, never pooled. Pooling `d_{a_1}[Y_0, Y_1, Y_shared]` lets a
    large, easily predicted self or shared response carry the recovery fraction
    while the cross term the taxonomy turns on stays hidden -- on Buzz Wire the
    shared response is 7.43 against a 2.24 cross term, so the pooled ladder
    reported `R_O = 0.380` for a quantity whose cross-specific value is 0.016.
    """
    span_columns = torch.arange(targets["test"].shape[1]).unsqueeze(0)
    ladder = {}
    for condition in ("A", "O", "H_dense", "H_model", "S"):
        net, decay, stats = select_and_fit(
            rows["train"][condition], targets["train"], ids["train"], args.device, args.seed
        )
        ladder[condition] = score(
            net, stats,
            {
                split: (
                    rows[split][condition], targets[split],
                    {"cross": span_columns.expand(targets[split].shape[0], -1),
                     "self": span_columns.expand(targets[split].shape[0], -1)},
                    ids[split],
                )
                for split in ("train", "test")
            },
            args.device, args.seed, decay,
        )
    base = ladder["A"]["test_cross"]["mean"]
    ceiling = ladder["S"]["test_cross"]["mean"]
    span = base - ceiling
    # A recovery fraction divides by (base - ceiling). When the privileged
    # reference barely beats the actions-only baseline, that span is near zero
    # and the ratio explodes: the axis-1 mixed target produced R = -13 and -20
    # from a span of 0.026. Such a target is simply unpredictable from any
    # input, and the honest report is NaN, not a large number.
    informative = span > REFERENCE_SPAN_FLOOR * max(base, 1e-9)
    for entry in ladder.values():
        entry["R_info"] = (
            (base - entry["test_cross"]["mean"]) / span if informative else float("nan")
        )
        entry["reference_informative"] = bool(informative)
    print(f"    [{label}] " + "  ".join(
        f"{c}={ladder[c]['test_cross']['mean']:.3f}" for c in ("A", "O", "H_dense", "H_model", "S")
    ) + f"   R_O={ladder['O']['R_info']:+.3f} R_Hd={ladder['H_dense']['R_info']:+.3f} "
        f"R_Hm={ladder['H_model']['R_info']:+.3f}", flush=True)
    return ladder


def diagnostic_overfit(ladder):
    """True when the ladder memorised instead of generalising.

    The first smoke of this module, on 40 anchors, produced train errors of
    0.003-0.15 against test errors of 1e4-1e6 with healthy cosine: predictions
    pointing the right way at wildly wrong scale. That is memorisation, and any
    classification built on it is meaningless. The guard refuses to classify
    rather than reporting the artifact.
    """
    offenders = {}
    for condition, entry in ladder.items():
        train = entry["train_cross"]["mean"]
        test = entry["test_cross"]["mean"]
        if test > 1.0 and train < 0.5 * test:
            offenders[condition] = {"train": train, "test": test}
    return offenders


def classify(cross, ladders):
    """The registered taxonomy. Fixed before any scenario was measured.

    Every gate is read off the CROSS target's own ladder, and the mixed term off
    the MIXED target's own ladder. Borrowing one quantity's instrument
    resolution to judge another is what the first draft did.
    """
    # `score` divides by ||true||, so an S-condition error IS a relative error
    # and its inverse is the resolution of that specific target.
    cross_reference = ladders["cross"]["S"]["test_cross"]["mean"]
    mixed_reference = ladders["mixed"]["S"]["test_cross"]["mean"]

    resolvable = (
        cross_reference <= 1.0 / RESOLUTION_RATIO
        and cross["active_fraction"] >= ACTIVE_FRACTION
    )
    cross_informative = ladders["cross"]["S"].get("reference_informative", True)
    recoveries = [ladders["cross"][c]["R_info"] for c in ("O", "H_dense", "H_model")]
    recoveries = [r for r in recoveries if r == r]  # drop NaN
    best_legit = max(recoveries) if recoveries else float("nan")
    # The mixed second difference now carries its OWN measurement floor rather
    # than borrowing the first-order cross-effect instrument.
    mixed_real = mixed_reference <= 1.0 / RESOLUTION_RATIO

    overfit = {}
    for name, ladder in ladders.items():
        found = diagnostic_overfit(ladder)
        if found:
            overfit[name] = found

    if overfit:
        verdict, reason = "undetermined_diagnostic_overfit", (
            f"the diagnostic memorised on {sorted(overfit)}; classification "
            "refused. Increase anchors or references and rerun"
        )
    elif not resolvable:
        verdict, reason = "weak_interaction_control", (
            "the privileged reference cannot resolve the cross-agent effect, or "
            "it is active on too few anchors"
        )
    elif not cross_informative:
        verdict, reason = "undetermined_reference_uninformative", (
            "the privileged reference barely beats the actions-only baseline on "
            "the cross target, so no recovery fraction is defined"
        )
    elif not (best_legit >= INFO_RECOVERY_GATE):
        verdict, reason = "partially_observable", (
            "cross-agent effect is real but no legitimate input recovers it "
            "(the Buzz Wire class)"
        )
    elif not mixed_real:
        verdict, reason = "observable_additive", (
            "cross-agent effect is observable but the mixed second difference is "
            "below its own measurement floor, so the response is additively "
            "separable as far as this instrument can tell"
        )
    else:
        verdict, reason = "ADMIT", (
            "observable cross-agent effect with a resolvable mixed interaction "
            "-- H0/H1/H2 have something to distinguish"
        )
    return {
        "verdict": verdict,
        "reason": reason,
        "cross_resolvable": bool(resolvable),
        "best_legitimate_cross_recovery": float(best_legit),
        "mixed_above_own_floor": bool(mixed_real),
        "cross_reference_relative_error": float(cross_reference),
        "mixed_reference_relative_error": float(mixed_reference),
        # Descriptive only. No state-dependence threshold is gated on, because
        # choosing a coefficient-of-variation cut after seeing the first
        # scenario would be arbitrary. `E_O < E_A` inside R_info is the
        # meaningful empirical test of whether state information matters.
        "shared_recovery_descriptive": float(
            max(ladders["shared"][c]["R_info"] for c in ("O", "H_dense", "H_model"))
        ),
        "self_recovery_descriptive": float(
            max(ladders["self"][c]["R_info"] for c in ("O", "H_dense", "H_model"))
        ),
    }


def audit_scenario(name, data_root, args):
    """One scenario, one comparable admission artifact."""
    data_root = Path(data_root)
    anchors, manifest = load_bank(data_root)
    task, _ = resolve_task(manifest)
    block = manifest.get("action_block", 5)
    steps = args.horizon * block
    step = steps - 1

    splits, episode_ids, subs = {}, {}, {}
    for split in ("train", "test"):
        rows = (anchors["split"] == SPLITS.index(split)).nonzero().squeeze(-1)
        if args.max_anchors:
            rows = rows[: args.max_anchors]
        splits[split] = rows
        episode_ids[split] = anchors["episode_id"][rows]
        subs[split] = {"snapshot": select_anchor_states(anchors, rows, "cpu")}

    low, high = action_bounds(task, args.device)
    agents, action_dim = low.shape
    regimes = [r for r in ("correlated", "independent")
               if (data_root / f"samples_{r}.pt").exists()]
    scale = task_scale(data_root, anchors, regimes)
    informative = int(scale["agent"][1].sum()) * agents + int(scale["shared"][1].sum()) * 1

    print(f"\n=== {name} ({manifest['task_name']}) ===", flush=True)
    print(f"  {len(splits['train'])} train / {len(splits['test'])} test anchors, "
          f"{agents} agents x {action_dim} axes, block {block}, "
          f"informative Y columns per body: agent "
          f"{int(scale['agent'][1].sum())}/6, shared {int(scale['shared'][1].sum())}/6",
          flush=True)

    report = {
        "scenario": name,
        "task_name": manifest["task_name"],
        "data_root": str(data_root),
        "agents": agents,
        "action_dim": action_dim,
        "action_block": block,
        "horizon_blocks": args.horizon,
        "references": args.references,
        "train_anchors": int(len(splits["train"])),
        "test_anchors": int(len(splits["test"])),
        "informative_y_columns": {
            "agent": int(scale["agent"][1].sum()),
            "shared": int(scale["shared"][1].sum()),
        },
        "axes": {},
    }

    for axis in range(action_dim):
        surface, live = {}, {}
        for split in ("train", "test"):
            surface[split], live[split] = grid_rollouts(
                task, subs[split], len(splits[split]), low, high, axis, args, steps
            )
        shared_bodies = surface["test"][(0, 0.0, 0.0)]["next_package_state"].shape[2]
        quantities = interaction_quantities(
            surface["test"], live["test"], step, scale, args.references,
            agents, shared_bodies, episode_ids["test"],
        )
        # Agent 0 is the reference responder: J_cross is its response to agent
        # 1's action, which is the multi-agent content the taxonomy turns on.
        responder = quantities["agent_0"]
        own = describe(responder["J_own"], responder["ids"], args.seed + 1)
        cross = describe(responder["J_cross"], responder["ids"], args.seed + 2)
        mixed = describe(responder["C"], responder["ids"], args.seed + 3)
        shared = describe(
            quantities["shared"]["J_cross"], quantities["shared"]["ids"], args.seed + 4
        )
        print(f"  axis {axis}: J_own={own['mean']:.4f}  J_cross={cross['mean']:.4f} "
              f"(active {cross['active_fraction']:.2f}, state-dep "
              f"{cross['state_dependence']:.2f})  C={mixed['mean']:.4f}  "
              f"shared={shared['mean']:.4f}", flush=True)

        # Four targets, each with its own ladder and its own measurement
        # floor. `cross` decides admission; `self` and `shared` are descriptive
        # and may reveal, e.g., that Transport's observation identifies the
        # package's response far better than the partner agent's.
        groups = body_columns(scale, agents, shared_bodies)
        mask = live["test"][:, step]
        targets_by_name = {
            "cross": ("agent_0", "J_cross"),
            "self": ("agent_1", "J_cross"),
            "shared": ("shared", "J_cross"),
            "mixed": ("agent_0", "C"),
        }
        ladders = {}
        for label, (responder, quantity) in targets_by_name.items():
            rows, targets, ids = {}, {}, {}
            for split in ("train", "test"):
                split_mask = live[split][:, step]
                built, target, episode = [], [], []
                for reference in range(args.references):
                    branch = {
                        "low": surface[split][(reference, 0.0, -1.0)],
                        "high": surface[split][(reference, 0.0, 1.0)],
                    }
                    if quantity == "C":
                        branch = {
                            "low": surface[split][(reference, -1.0, -1.0)],
                            "high": surface[split][(reference, 1.0, 1.0)],
                        }
                    if int(split_mask.sum()) == 0:
                        continue
                    built.append(
                        ladder_inputs(
                            data_root, anchors, splits[split], branch, split_mask,
                            args, block,
                        )
                    )
                    columns = groups[responder]
                    if quantity == "C":
                        value = (
                            body_target(surface[split][(reference, 1.0, 1.0)], step + 1, scale)
                            - body_target(surface[split][(reference, 1.0, -1.0)], step + 1, scale)
                            - body_target(surface[split][(reference, -1.0, 1.0)], step + 1, scale)
                            + body_target(surface[split][(reference, -1.0, -1.0)], step + 1, scale)
                        )
                    else:
                        value = (
                            body_target(surface[split][(reference, 0.0, 1.0)], step + 1, scale)
                            - body_target(surface[split][(reference, 0.0, -1.0)], step + 1, scale)
                        )
                    target.append(value[:, columns][split_mask])
                    episode.append(episode_ids[split][split_mask])
                rows[split] = {
                    key: torch.cat([b[key] for b in built]) for key in built[0]
                }
                targets[split] = torch.cat(target)
                ids[split] = torch.cat(episode)
            ladders[label] = information_ladder(rows, targets, ids, label, args)

        report["axes"][str(axis)] = {
            "J_own": own, "J_cross": cross, "C_mixed": mixed,
            "shared_response": shared,
            "information_ladders": {
                label: {
                    c: {"E": v["test_cross"]["mean"],
                        "cosine": v["test_cross"]["cosine_mean"],
                        "R_info": v["R_info"],
                        "train_E": v["train_cross"]["mean"]}
                    for c, v in ladder.items()
                }
                for label, ladder in ladders.items()
            },
            "classification": classify(cross, ladders),
        }

    # The scenario's verdict is its best axis, chosen by cross-effect size.
    best = max(report["axes"], key=lambda a: report["axes"][a]["J_cross"]["mean"])
    report["selected_axis"] = best
    report["verdict"] = report["axes"][best]["classification"]
    print(f"  -> axis {best}: {report['verdict']['verdict']} "
          f"({report['verdict']['reason']})", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenario", action="append", required=True, metavar="NAME=PATH",
        help="repeatable, e.g. --scenario wheel=outputs/wheel_1205/data",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--horizon", type=int, default=1, help="blocks")
    parser.add_argument("--references", type=int, default=2)
    parser.add_argument("--dense-frames", type=int, default=3)
    parser.add_argument("--model-frames", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7301)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-anchors", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    reports = {}
    for entry in args.scenario:
        name, _, path = entry.partition("=")
        reports[name] = audit_scenario(name, path, args)
        # Written after each scenario so a later failure cannot lose earlier work.
        (output / "admission_report.json").write_text(json.dumps(reports, indent=2))

    print("\n=== ADMISSION REPORT ===")
    print("R_* are recovery fractions on the CROSS-AGENT target only; "
          "shared/self are descriptive")
    print(f"{'scenario':<12} {'J_cross':>8} {'C_mix':>7} {'st-dep':>7} "
          f"{'R_O':>6} {'R_Hd':>6} {'R_Hm':>6} {'R_shr':>6}  verdict")
    for name, report in reports.items():
        axis = report["axes"][report["selected_axis"]]
        cross = axis["information_ladders"]["cross"]
        verdict = report["verdict"]
        print(
            f"{name:<12} {axis['J_cross']['mean']:>8.3f} {axis['C_mixed']['mean']:>7.3f} "
            f"{axis['J_cross']['state_dependence']:>7.2f} "
            f"{cross['O']['R_info']:>6.3f} {cross['H_dense']['R_info']:>6.3f} "
            f"{cross['H_model']['R_info']:>6.3f} "
            f"{verdict['shared_recovery_descriptive']:>6.3f}  {verdict['verdict']}"
        )
    print(f"\nWrote {output / 'admission_report.json'}")


if __name__ == "__main__":
    main()
