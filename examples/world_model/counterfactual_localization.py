#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#
"""T-A2b: localize where the counterfactual effect is lost, E vs P vs interface.

T-A2 scored the composite ``E -> P -> probe`` and found it fails: every
conditioned arm sits above ``E_CF = 1`` on the cross block while carrying a
clearly positive cosine. That number cannot say which stage lost the effect, and
attributing it to the encoder would be exactly the unlocalized causal claim the
audit exists to prevent.

Three tests share one target, one scale and one set of interventions:

  **Test B -- true future latent.** Encode the TRUE next observation of each
  branch and read the response off with the fixed probe. Already computed by
  ``counterfactual_fidelity`` as ``probe_floor``: on Buzz Wire cross cells it is
  0.173-0.234, so the encoder *does* represent the realized counterfactual
  difference after it happens.

  **Test C -- predicted future latent.** The T-A2 measurement itself, 1.05-1.19.

  **Test A -- current-latent sufficiency.** THE DISCRIMINATOR, and the only one
  that is new. Fit a diagnostic head

      g(z_t, a^low, a^high) -> dY

  on training anchors and score it on held-out ones. This never rolls the
  predictor forward, so it asks one thing only: does the latent the predictor
  starts from carry enough state information to determine the counterfactual
  effect?

Test A is meaningless without its two controls, which are the whole point:

  ``actions_only``  g(a^low, a^high) -> dY, the STATE-BLIND floor. It can learn
                    the average response to an intervention but nothing
                    state-specific. If ``latent`` cannot beat it, ``z_t`` adds
                    no state information about the interaction.
  ``physical``      g(s_t, a^low, a^high) -> dY on the recorded simulator state,
                    the information CEILING. If it also fails to beat
                    ``actions_only``, the effect is simply not predictable from
                    the current state at this horizon and Test A cannot
                    discriminate anything -- report that and stop, rather than
                    reading a null as an encoder result.

Reading the result (the registered table is in experiment 30):

  latent ~ actions_only         -> z_t is not counterfactually sufficient
  latent ~ physical << blind    -> z_t is sufficient; the PREDICTOR loses it
  physical ~ actions_only       -> the diagnostic is uninformative; stop

Run:
    python -m examples.world_model.counterfactual_localization \
        outputs/ta2_reference_baselines_1498/runs \
        --data outputs/buzz_wire_1196/data \
        --jacobian outputs/ta1_interaction_jacobian_1497/result/interaction_jacobian.json \
        --out outputs/ta2b --device cuda
"""

import argparse
import json
from pathlib import Path

import torch
import yaml

from examples.world_model.counterfactual_evaluation import blocked
from examples.world_model.counterfactual_fidelity import (
    collect_branches,
    resolve_regime_files,
    column_groups,
    scaled_truth,
    verify_against_jacobian,
)
from examples.world_model.interaction_jacobian import (
    SPLITS,
    action_bounds,
    bootstrap_by_episode,
    pooled_ratio_by_episode,
    load_bank,
    resolve_task,
    training_scale,
)
from examples.world_model.physical_response import encoded, state_input_frames
from examples.world_model.plan_ranking import select_anchor_states
from examples.world_model.train import load_model, world_model_from_config
from omegaconf import OmegaConf


def untrained_like(checkpoint_path, device, seed):
    """The checkpoint's exact architecture and input normalization, random weights.

    The control for "the encoder loses two thirds of what the observation has"
    (job 1516). A deterministic 6 -> 192 map can only lose information by being
    non-injective, so part of the observation-vs-latent gap may be the diagnostic
    head's own difficulty with a 192-D input at ~700 anchors. If an UNTRAINED
    encoder scores like the trained one, the gap is the head; if it scores like
    the raw observation, JEPA training discarded the information.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    torch.manual_seed(seed)
    model = world_model_from_config(
        OmegaConf.create(checkpoint["config"]),
        checkpoint["shapes"],
        checkpoint["observation_mean"],
        checkpoint["observation_std"],
        device,
        action_mean=checkpoint.get("action_mean"),
        action_std=checkpoint.get("action_std"),
    )
    model.eval()
    return model


def random_projection(obs_dim, width, seed):
    """(obs_dim, width) Gaussian map: full column rank, so it loses nothing.

    Applied per agent, it gives the raw observation the latent's width without
    changing its information -- a pure input-dimensionality control.
    """
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(obs_dim, width, generator=generator, dtype=torch.float64)



def history_window(anchors, rows, data_root, frames, observed):
    """(B, frames*N*obs + (frames-1)*N*act): what the agents actually saw.

    G0's question is whether the mediating state -- Buzz Wire's ball, which no
    agent observes -- is recoverable from legitimate observations once MOTION is
    available. Two rigid joints constrain the ball to at most two solutions given
    the agents' positions, so a single frame cannot disambiguate it but a short
    history might.

    Indices clamp at 0, repeating the earliest available frame, which is the
    episode-start convention `model_input.PlanningContext` already registers.

    `observed` is each anchor's own step-0 observation from the branch rollout.
    The looked-up frame at `source_step` must equal it exactly, or the window is
    aligned to the wrong episode and every number built on it is meaningless.
    """
    episode = anchors["episode_id"][rows]
    step = anchors["source_step"][rows]
    regime = anchors["source_regime"][rows]
    # Discovered and verified per bank, never assumed: see
    # `counterfactual_fidelity.resolve_regime_files`.
    trajectories = resolve_regime_files(data_root, anchors, rows, observed)

    observations, actions = [], []
    for e, s, r in zip(episode.tolist(), step.tolist(), regime.tolist()):
        source = trajectories[r]
        frame_index = [max(0, s - offset) for offset in reversed(range(frames))]
        action_index = [max(0, s - offset) for offset in reversed(range(1, frames))]
        observations.append(source["observation"][e, frame_index])
        actions.append(source["action"][e, action_index])
    observations = torch.stack(observations)  # (B, frames, N, obs)
    actions = torch.stack(actions)  # (B, frames-1, N, act)

    aligned = observations[:, -1]
    if not torch.equal(aligned, observed):
        raise ValueError(
            "History lookup is misaligned with the anchor it describes: the "
            f"frame at source_step differs by {float((aligned - observed).abs().max()):.3e}"
        )
    batch = observations.shape[0]
    return torch.cat(
        [observations.reshape(batch, -1), actions.reshape(batch, -1)], dim=1
    ).double()


# One architecture for every input condition. The inputs differ in width because
# that is the question being asked, so the head itself must not also vary.
HIDDEN = 512
DECAYS = (0.0, 1e-4, 1e-3, 1e-2)
EPOCHS = 400
BATCH = 512


def standardize(train, *others):
    """z-score on TRAIN statistics only.

    Raw VMAS coordinates and native-unit actions have very different scales, and
    an unnormalized design matrix makes the head underfit at a fixed budget --
    which would show up as "the state does not predict the effect" when it
    really means "the optimizer did not converge". Fitting the statistics on
    train only keeps test information out.
    """
    mean = train.mean(dim=0, keepdim=True)
    std = train.std(dim=0, keepdim=True).clamp_min(1e-8)
    return [(x - mean) / std for x in (train, *others)]


def fit_head(features, targets, decay, device, seed=0):
    """One hidden layer, minibatch Adam, fixed budget. A capacity check.

    Minibatched rather than full-batch: 300 full-batch steps is a few hundred
    updates and underfits this design matrix badly. The fitted TRAIN error is
    reported beside the test error so underfitting stays visible instead of
    being read as a property of the input.
    """
    torch.manual_seed(seed)
    net = (
        torch.nn.Sequential(
            torch.nn.Linear(features.shape[1], HIDDEN),
            torch.nn.GELU(),
            torch.nn.Linear(HIDDEN, HIDDEN),
            torch.nn.GELU(),
            torch.nn.Linear(HIDDEN, targets.shape[1]),
        )
        .to(device)
        .double()
    )
    x, y = features.to(device), targets.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=decay)
    rows = x.shape[0]
    generator = torch.Generator(device="cpu").manual_seed(seed)
    for _ in range(EPOCHS):
        order = torch.randperm(rows, generator=generator).to(device)
        for start in range(0, rows, BATCH):
            batch = order[start : start + BATCH]
            optimizer.zero_grad()
            (net(x[batch]) - y[batch]).square().mean().backward()
            optimizer.step()
    return net


def select_and_fit(features, targets, episodes, device, seed=0):
    """Choose weight decay on held-out ROOT EPISODES, then refit on everything.

    Splitting by row would leak: anchors from one episode are correlated, so a
    row-wise split lets the head memorise an episode and score on its siblings.
    That is the defect audit section 5.3 records.

    Returns the head, the chosen decay, and the train statistics it was
    standardized with, so scoring applies the identical transform.
    """
    unique = torch.unique(episodes)
    keep = unique[: max(1, int(0.75 * len(unique)))]
    inner = torch.isin(episodes, keep)
    scored = []
    for decay in DECAYS:
        inner_x, held_x = standardize(features[inner], features[~inner])
        net = fit_head(inner_x, targets[inner], decay, device, seed)
        with torch.no_grad():
            held = (
                net(held_x.to(device)) - targets[~inner].to(device)
            ).square().mean()
        scored.append((float(held), decay))
    _, best = min(scored)
    mean = features.mean(dim=0, keepdim=True)
    std = features.std(dim=0, keepdim=True).clamp_min(1e-8)
    net = fit_head((features - mean) / std, targets, best, device, seed)
    return net, best, (mean, std)


@torch.no_grad()
def latent_features(model, branches, shaped, device):
    """z_t per cell, flattened over agents.

    At step 0 no action has been applied, so both branches of a cell share the
    anchor's observation and therefore one latent. Encoding that one frame is
    the whole of what the reference predictor starts from at an episode
    boundary, where its context is that frame repeated.
    """
    features = {}
    for key, endpoints in branches.items():
        frames = shaped(endpoints["low"], slice(None), 0, False)
        latent = encoded(model, frames, device)  # (B, N, dim)
        features[key] = latent.reshape(latent.shape[0], -1)
    return features


def build_rows(branches, block, step, agents, scale, episode_ids, latents, history=None,
               projection=None, untrained=None):
    """Stack every (anchor, reference, cell) into one design matrix per input."""
    rows = {
        "actions_only": [],
        "observation_raw": [],
        "observation_projected": [],
        "physical": [],
        "latent": [],
        "latent_plus_agentphys": [],
        "latent_plus_state": [],
        "latent_plus_observation": [],
        "latent_untrained": [],
        "history": [],
    }
    targets, episodes = [], []
    columns = {"cross": [], "self": []}
    for key, endpoints in sorted(branches.items()):
        _, intervened, _ = key
        true_delta, mask = scaled_truth(endpoints, list(range(agents)), step, scale)
        count = int(mask.sum())
        if count == 0:
            continue
        low_action = blocked(endpoints["low"]["action"][:, : step + 1], block)[:, 0]
        high_action = blocked(endpoints["high"]["action"][:, : step + 1], block)[:, 0]
        actions = torch.cat(
            [
                low_action.reshape(low_action.shape[0], -1),
                high_action.reshape(high_action.shape[0], -1),
            ],
            dim=1,
        ).double()
        physical = torch.cat(
            [
                endpoints["low"]["agent_state"][:, 0].reshape(actions.shape[0], -1),
                endpoints["low"]["package_state"][:, 0].reshape(actions.shape[0], -1),
            ],
            dim=1,
        ).double()
        observation = endpoints["low"]["observation"][:, 0].reshape(
            actions.shape[0], -1
        ).double()
        agent_physical = endpoints["low"]["agent_state"][:, 0].reshape(
            actions.shape[0], -1
        ).double()
        rows["actions_only"].append(actions[mask])
        # The RAW observation, unencoded. If it beats the latent, the encoder is
        # discarding cross-agent information it was given -- a JEPA result. If
        # they match, the encoder preserved what was there and the deficit is
        # the observation itself.
        rows["observation_raw"].append(
            torch.cat([observation, actions], dim=1)[mask]
        )
        rows["physical"].append(torch.cat([physical, actions], dim=1)[mask])
        if projection is not None:
            # (B, N, obs) @ (obs, width) -> (B, N*width): the observation at the
            # latent's width, information unchanged.
            per_agent = endpoints["low"]["observation"][:, 0].double() @ projection
            rows["observation_projected"].append(
                torch.cat([per_agent.reshape(actions.shape[0], -1), actions], dim=1)[mask]
            )
        if untrained is not None:
            rows["latent_untrained"].append(
                torch.cat([untrained[key], actions], dim=1)[mask]
            )
        if history is not None:
            rows["history"].append(torch.cat([history, actions], dim=1)[mask])
        if latents is not None:
            rows["latent"].append(torch.cat([latents[key], actions], dim=1)[mask])
            # T-A2b-2: the latent PLUS the mediating state the observation omits.
            # Buzz Wire's observation is [pos, vel, pos - goal] -- no ball, no
            # linkage, no partner -- while the ball is rigidly jointed to both
            # agents and mediates the cross-agent effect. If this reaches the
            # physical ceiling, the deficit IS that omitted state.
            shared_state = endpoints["low"]["package_state"][:, 0].reshape(
                actions.shape[0], -1
            ).double()
            rows["latent_plus_state"].append(
                torch.cat([latents[key], shared_state, actions], dim=1)[mask]
            )
            # Agent rotation and angular velocity, which the observation omits
            # but which are NOT the mediating body. Separates "the ball is
            # required" from "any omitted physical state would do".
            rows["latent_plus_agentphys"].append(
                torch.cat([latents[key], agent_physical, actions], dim=1)[mask]
            )
            # The encoder's own input handed back beside its output. If this
            # scores BELOW `observation_raw`, the extra latent dimensions cost
            # the head more than they add -- the head, not the encoder.
            rows["latent_plus_observation"].append(
                torch.cat([latents[key], observation, actions], dim=1)[mask]
            )
        targets.append(true_delta[mask])
        groups = column_groups(agents, 0, intervened)
        # NOT `block`: that name is the action-block width used above, and
        # rebinding it here silently passed a string into `blocked()`.
        for name in ("cross", "self"):
            columns[name].append(groups[name].unsqueeze(0).expand(count, -1))
        episodes.append(episode_ids[mask])
    stacked = {name: torch.cat(v) for name, v in rows.items() if v}
    return (
        stacked,
        torch.cat(targets),
        {block: torch.cat(v) for block, v in columns.items()},
        torch.cat(episodes),
    )


def block_e_cf(predicted, truth, columns):
    """Per-anchor residual norm, truth norm, ratio and cosine on one block.

    The residual and truth norms are returned UNDIVIDED so the caller can pool
    them before taking the ratio. Dividing per anchor is only sound where every
    anchor's effect clears the floor; see `pooled_ratio_by_episode`.
    """
    picked_pred = predicted.gather(1, columns)
    picked_true = truth.gather(1, columns)
    residual = (picked_pred - picked_true).norm(dim=1)
    magnitude = picked_true.norm(dim=1)
    return (
        residual,
        magnitude,
        residual / (magnitude + 1e-12),
        torch.nn.functional.cosine_similarity(
            picked_pred, picked_true, dim=1, eps=1e-12
        ),
    )


def restrict_to_block(target, columns, block):
    """Gather one block's columns into a fixed-width target.

    The cross columns depend on which agent was intervened, so they vary row by
    row and cannot be sliced with a fixed index. Gathering makes the head
    predict that block DIRECTLY instead of predicting the whole next-state
    response and having a subset read off it afterwards.

    That distinction is the whole point of this switch: on Buzz Wire the shared
    response is 7.43 against a 2.24 cross term, so a head minimising MSE over
    the full target spends its capacity elsewhere.
    """
    gathered = target.gather(1, columns[block])
    identity = torch.arange(gathered.shape[1]).unsqueeze(0).expand(gathered.shape[0], -1)
    return gathered, {"cross": identity, "self": identity}


def score(net, stats, rows, device, seed, decay):
    """E_CF on both blocks, on TRAIN and TEST.

    The self block is the head's own sanity control: T-A2 shows self-dynamics
    are well captured, so a head that cannot predict the self response either is
    underfit, and its cross number says nothing about the input. The train
    numbers make underfitting visible rather than letting it read as a property
    of the input.
    """
    mean, std = stats
    summary = {"weight_decay": decay}
    for split, (features, target, columns, episodes) in rows.items():
        with torch.no_grad():
            predicted = net(((features - mean) / std).to(device)).cpu()
        for block in ("cross", "self"):
            residual, magnitude, e_cf, cosine = block_e_cf(
                predicted, target, columns[block]
            )
            entry = pooled_ratio_by_episode(
                residual, magnitude, episodes, seed=seed + 11
            )
            # Both forms are reported side by side. They coincide where the
            # effect is bounded away from zero on every anchor (Buzz Wire) and
            # diverge where it is not (Balance at h=3), so a reader can see
            # which regime a number came from instead of trusting the label.
            entry["mean_of_ratios"] = bootstrap_by_episode(
                e_cf, episodes, seed=seed + 11
            )
            entry["ratio_median"] = float(e_cf.median())
            entry["effect_median"] = float(magnitude.median())
            entry["cosine_mean"] = float(cosine.mean())
            entry["fraction_below_one"] = float((e_cf < 1.0).double().mean())
            summary[f"{split}_{block}"] = entry
    return summary


def run(args):
    data_root = Path(args.data)
    anchors, manifest = load_bank(data_root)
    task, _ = resolve_task(manifest)
    block = args.block or manifest.get("action_block", 5)
    steps = args.horizon * block
    step = steps - 1

    splits, episode_ids, branches = {}, {}, {}
    for name in ("train", "test"):
        rows = (anchors["split"] == SPLITS.index(name)).nonzero().squeeze(-1)
        if args.max_anchors:
            rows = rows[: args.max_anchors]
        splits[name] = rows
        episode_ids[name] = anchors["episode_id"][rows]

    low, high = action_bounds(task, args.device)
    agents = low.shape[0]
    agent_scale, shared_scale = training_scale(
        data_root, anchors, ("correlated", "independent")
    )
    samples = torch.load(
        data_root / "samples_correlated.pt", map_location="cpu", weights_only=True
    )
    shared_bodies = samples["next_package_state"].shape[2]
    scale = torch.cat(
        [agent_scale.repeat(agents), shared_scale.repeat(shared_bodies)]
    ).double()

    for name, rows in splits.items():
        sub = {"snapshot": select_anchor_states(anchors, rows, "cpu")}
        print(f"regenerating {name} interventions ({len(rows)} anchors)", flush=True)
        branches[name], _ = collect_branches(
            task, sub, len(rows), low, high, args, steps
        )
        if args.cells:
            # One head serving four heterogeneous (agent, axis) intervention
            # cells is a different, harder problem than one head per cell.
            # Restricting the fit isolates that factor while holding the
            # intervention design, anchors, seeds, scale and head family fixed.
            keep = {tuple(int(v) for v in cell.split(":")) for cell in args.cells}
            branches[name] = {
                key: value for key, value in branches[name].items()
                if (key[1], key[2]) in keep
            }
            if not branches[name]:
                raise ValueError(f"No branches match --cells {args.cells}")
            print(f"  restricted to cells {sorted(keep)}: "
                  f"{len(branches[name])} branch sets", flush=True)

    recorded = json.loads(Path(args.jacobian).read_text())["cells"]
    checked = verify_against_jacobian(
        branches["test"], agents, block, agent_scale, recorded, args.horizon
    )
    print(f"test interventions reproduce {checked} T-A1 cells\n", flush=True)

    results = {
        "data_root": str(data_root),
        "horizon_blocks": args.horizon,
        "references": args.references,
        "hidden": HIDDEN,
        "epochs": EPOCHS,
        "train_anchors": int(len(splits["train"])),
        "test_anchors": int(len(splits["test"])),
        "shared": {},
        "per_run": {},
    }

    # The state-blind floor and the physical ceiling do not depend on any model,
    # so they are fitted once. Refitting them per checkpoint would invite
    # reading their seed noise as a model difference.
    shared, history = {}, {}
    projection = None
    if args.encoder_controls:
        obs_dim = next(iter(branches["train"].values()))["low"]["observation"].shape[-1]
        projection = random_projection(obs_dim, args.projection_width, args.seed)
    for name in ("train", "test"):
        if args.history_frames > 1:
            any_branch = next(iter(branches[name].values()))["low"]
            history[name] = history_window(
                anchors, splits[name], data_root, args.history_frames,
                any_branch["observation"][:, 0],
            )
            print(f"  {name} history window: {tuple(history[name].shape)} "
                  f"({args.history_frames} frames, alignment verified)", flush=True)
        shared[name] = build_rows(
            branches[name], block, step, agents, scale, episode_ids[name], None,
            history.get(name), projection,
        )
    conditions = ["actions_only", "observation_raw", "physical"]
    if args.history_frames > 1:
        conditions.insert(2, "history")
    if args.encoder_controls:
        conditions.insert(2, "observation_projected")
    if args.conditions:
        # The A-vs-S admission gate needs only the blind floor and the
        # privileged reference. Fitting `observation_raw` and `history` as well
        # is the expensive part of the expensive ladder, and it answers a
        # question that is meaningless until the gate says the effect is
        # resolvable at all.
        unknown = set(args.conditions) - set(conditions)
        if unknown:
            raise ValueError(f"unknown conditions {sorted(unknown)}; have {conditions}")
        conditions = [c for c in conditions if c in set(args.conditions)]
    def fitted_view(entry, condition):
        design, target, columns, episodes = entry
        if args.fit_target != "full":
            target, columns = restrict_to_block(target, columns, args.fit_target)
        return design[condition], target, columns, episodes

    for condition in conditions:
        train_design, train_target, _, train_episodes = fitted_view(
            shared["train"], condition
        )
        net, decay, stats = select_and_fit(
            train_design, train_target, train_episodes, args.device, args.seed
        )
        summary = score(
            net, stats,
            {split: fitted_view(shared[split], condition) for split in ("train", "test")},
            args.device, args.seed, decay,
        )
        results["shared"][condition] = summary
        cross, self_ = summary["test_cross"], summary["test_self"]
        print(
            f"{condition:<14} cross {cross['mean']:.4f} "
            f"[{cross['low']:.4f}, {cross['high']:.4f}]  "
            f"self {self_['mean']:.4f}  "
            f"(train cross {summary['train_cross']['mean']:.4f})",
            flush=True,
        )

    # The shared conditions are model-independent, so a history-length sweep
    # does not need the per-checkpoint loop that dominates runtime.
    for directory in sorted(Path(args.runs).iterdir()):
        if args.shared_only or not (directory / "model.pt").exists():
            continue
        config = yaml.safe_load((directory / "resolved_config.yaml").read_text())
        state_input = config["data"].get("state_input", "observation")
        frames = config["data"].get("history_frames", 3)
        shaped = lambda src, idx, stp, follow: state_input_frames(  # noqa: E731
            src, idx, stp, state_input, frames, follow
        )
        model = load_model(directory / "model.pt", args.device)
        model.eval()
        random_model = (
            untrained_like(directory / "model.pt", args.device, config["seed"])
            if args.encoder_controls else None
        )

        fitted = {}
        for name in ("train", "test"):
            latents = latent_features(model, branches[name], shaped, args.device)
            untrained = (
                latent_features(random_model, branches[name], shaped, args.device)
                if random_model is not None else None
            )
            fitted[name] = build_rows(
                branches[name], block, step, agents, scale, episode_ids[name], latents,
                history.get(name), None, untrained,
            )
        kind = config["model"]["kind"]
        if args.label_from_dir:
            kind = directory.name.rsplit("_seed", 1)[0]
        key = f"{config['data']['regime']}__{kind}__{config['seed']}"
        per_checkpoint = ["latent", "latent_plus_agentphys", "latent_plus_state"]
        if args.encoder_controls:
            per_checkpoint += ["latent_plus_observation", "latent_untrained"]
        for condition in per_checkpoint:
            train_design, train_target, _, train_episodes = fitted_view(
                fitted["train"], condition
            )
            net, decay, stats = select_and_fit(
                train_design, train_target, train_episodes, args.device, args.seed
            )
            summary = score(
                net, stats,
                {split: fitted_view(fitted[split], condition)
                 for split in ("train", "test")},
                args.device, args.seed, decay,
            )
            results["per_run"].setdefault(condition, {})[key] = summary
            print(f"  {condition:<18} {key:<38} cross {summary['test_cross']['mean']:.4f} "
                  f"cos {summary['test_cross']['cosine_mean']:+.3f}", flush=True)

    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    (output / "counterfactual_localization.json").write_text(json.dumps(results, indent=2))
    print(f"\nWrote {output / 'counterfactual_localization.json'}")
    print_report(results)
    return results


def print_report(results):
    print("\n=== Test A: is the CURRENT latent counterfactually sufficient? ===")
    print("cross = the question; self = the head's sanity control; train = underfit check")
    print("(E_CF as a POOLED RATIO OF SUMS; 1.0 = predicts no response at all.)")
    print("`mean-of-ratios` is the superseded per-anchor form, reported so the")
    print("two can be compared wherever the effect is not uniformly active.\n")

    def line(label, summary):
        cross, self_ = summary["test_cross"], summary["test_self"]
        ratios = cross.get("mean_of_ratios", {}).get("mean", float("nan"))
        print(
            f"  {label:<34} cross {cross['mean']:.4f} "
            f"[{cross['low']:.4f}, {cross['high']:.4f}]   "
            f"self {self_['mean']:.4f}   "
            f"train-cross {summary['train_cross']['mean']:.4f}   "
            f"mean-of-ratios {ratios:.4f}"
        )

    for name, summary in results["shared"].items():
        line(name, summary)

    by_arm = {}
    if not results["per_run"]:
        print("  (shared conditions only; no per-checkpoint arms requested)")
    for condition, runs in results["per_run"].items():
        for key, summary in runs.items():
            regime, kind, _ = key.split("__")
            by_arm.setdefault(f"{condition}[{regime}/{kind}]", []).append(summary)
    for name, runs in sorted(by_arm.items()):
        mean = {
            field: sum(r[field]["mean"] for r in runs) / len(runs)
            for field in ("test_cross", "test_self", "train_cross")
        }
        print(
            f"  {name:<40} cross {mean['test_cross']:.4f}   "
            f"self {mean['test_self']:.4f}   "
            f"train-cross {mean['train_cross']:.4f}   ({len(runs)} seeds)"
        )

    # The agreement check. R is recomputed end to end under each aggregation so
    # the comparison is between two complete measurements, not between two
    # numbers that happen to share a denominator.
    for form in ("pooled", "mean_of_ratios"):
        report_recovery(results, by_arm, form)


def _pick(entry, form):
    return entry["mean"] if form == "pooled" else entry["mean_of_ratios"]["mean"]


def report_recovery(results, by_arm, form):
    print(f"\n--- recovery ratios under `{form}` ---")
    blind_entry = results["shared"].get("actions_only", {}).get("test_cross")
    ceiling_entry = results["shared"].get("physical", {}).get("test_cross")
    if blind_entry is None or ceiling_entry is None:
        return
    if form == "mean_of_ratios" and "mean_of_ratios" not in blind_entry:
        return
    blind, ceiling = _pick(blind_entry, form), _pick(ceiling_entry, form)
    for name, entry in results["shared"].items():
        value = _pick(entry["test_cross"], form)
        print(f"  E[{name:<40}] = {value:.4f}")
    gap = blind - ceiling
    print(f"\n  blind-to-ceiling gap on cross: {gap:+.4f}")
    if gap <= 0:
        print("  -> the true physical state does not beat the state-blind head.")
        print("     Registered rule: the diagnostic is UNINFORMATIVE at this horizon.")
        return
    blind_cos = results["shared"]["actions_only"]["test_cross"]["cosine_mean"]
    ceiling_cos = results["shared"]["physical"]["test_cross"]["cosine_mean"]
    for name, runs in sorted(by_arm.items()):
        latent = sum(_pick(r["test_cross"], form) for r in runs) / len(runs)
        cos = sum(r["test_cross"]["cosine_mean"] for r in runs) / len(runs)
        # Both, because they can disagree: a head can recover the DIRECTION of
        # the response while still missing its scale, and E_CF alone hides that.
        print(
            f"  R[{name:<40}] = {(blind - latent) / gap:+.3f} on E_CF, "
            f"{(cos - blind_cos) / max(ceiling_cos - blind_cos, 1e-12):+.3f} on cosine"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--jacobian", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--block", type=int, default=None)
    # 4 matches T-A1. The test-split interventions must reproduce its recorded
    # cell means exactly, so this is a contract, not a tuning knob.
    parser.add_argument("--references", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7301)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-anchors", type=int, default=None)
    # 3 matches the reference profile's history_size. 1 disables the condition.
    parser.add_argument("--history-frames", type=int, default=3)
    parser.add_argument(
        "--cells", nargs="*", default=None, metavar="AGENT:AXIS",
        help="restrict the fit to these intervention cells, e.g. 1:0. Default "
        "pools every cell into one head, which is what G0/G0b did.",
    )
    parser.add_argument(
        "--fit-target", choices=("full", "cross", "self"), default="full",
        help="what the diagnostic head is fitted to predict. `full` is the "
        "historical behaviour -- fit the whole next-state response, score a "
        "subset. `cross` fits the cross block directly, which is the better-"
        "posed question when asking whether the cross effect is recoverable.",
    )
    parser.add_argument(
        "--conditions", nargs="*", default=None,
        help="restrict the shared conditions that are fitted, e.g. "
        "`--conditions actions_only physical` for the A-vs-S admission gate. "
        "Default fits all of them.",
    )
    parser.add_argument(
        "--shared-only",
        action="store_true",
        help="skip the per-checkpoint latent conditions. The state-blind floor, "
        "raw observation, history window and physical ceiling do not depend on "
        "any model, so a history-length sweep needs only these.",
    )
    parser.add_argument(
        "--encoder-controls",
        action="store_true",
        help="add the head-dimensionality controls for K36: the observation "
        "through a fixed random linear map to the latent width, the latent plus "
        "the raw observation, and an untrained encoder of the same architecture.",
    )
    parser.add_argument("--projection-width", type=int, default=192)
    parser.add_argument(
        "--label-from-dir", action="store_true",
        help="label each checkpoint by its run directory (minus `_seed<N>`) "
        "instead of model.kind, for sweeps whose variants share a kind",
    )
    parser.add_argument("--device", default="cpu")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
