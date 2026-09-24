# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the license in the LICENSE file in the root directory.
"""Single-agent gate: can the latent world model imagine one agent's own motion?

Every multi-agent claim assumes the model first gets an agent's OWN dynamics
right. This scores that on a one-agent VMAS bank (``vmas/navigation``,
``n_agents=1``), where there is no cross-agent term to hide behind:

* rollout   -- three real context frames, then recursive imagination over the
               remaining blocks, decoded to position by a probe fitted on
               TRAIN true latents, against simulator truth;
* teacher   -- one-step prediction from real context at every block;
* self-CF   -- the collector's counterfactual file flips the agent's own x
               action from ``intervention_start_step`` on, with identical
               context before it. The model rolls both action sequences from
               the SAME context; ``E_CF``, gain and cosine are pooled per
               horizon (the frozen convention of note 33).

References, all on the same anchors and horizons:

* persistence and constant velocity -- trivial physical baselines;
* probe floor -- the probe applied to the TRUE encoding of each future frame,
  the best any latent prediction can score through this readout;
* direct      -- an observation-space model f(o_t, a_t) -> o_{t+1} - o_t fitted
  on the same train roots and rolled out the same way. It needs no probe, so it
  is the "is this learnable from this data at all" reference.

Registered pass rule (fixed before any sweep result, 2026-09-23):
  G1  rollout position MSE below persistence at every horizon, every seed;
  G2  self-CF pooled ``E_CF`` < 0.5 and cosine > 0.8 at horizons 1-3 after the
      intervention, with the probe floor itself <= 1/3 there.
A cell failing G1 or G2 means the pipeline cannot model a single agent, and no
multi-agent result built on it is interpretable.
"""

import argparse
import json
from pathlib import Path

import torch
import yaml

from examples.world_model.dataset import OfflineSequences
from examples.world_model.latent_position import (
    decode_agents,
    fit_position_probe,
    stack_dataset,
)
from examples.world_model.physical_response import (
    MLP_DECAYS,
    RIDGE_STRENGTHS,
    apply_probe,
    fit_mlp,
    ridge,
)
from examples.world_model.train import load_model

CONTEXT = 3  # lewm_reference history_size


def block_snippets(samples, block):
    """(B,T,N,...) primitive snippets -> the dataset's block-boundary view."""
    count, steps, agents, action_dim = samples["action"].shape
    length = steps // block

    def frames(key, nxt):
        return torch.cat([samples[key][:, :1], samples[nxt][:, block - 1 :: block]], dim=1)

    primitive = samples["action"].reshape(count, length, block, agents, action_dim)
    return {
        "observation": frames("observation", "next_observation"),
        "position": frames("agent_state", "next_agent_state")[..., :2].double(),
        "action": primitive.permute(0, 1, 3, 2, 4).reshape(
            count, length, agents, block * action_dim
        ),
        "valid": samples["valid"].reshape(count, length, block).all(dim=2),
    }


def alive(valid):
    """(B,L) block validity -> (B,L) 'every block up to here was valid'."""
    return valid.long().cumprod(dim=1).bool()


@torch.no_grad()
def imagine(model, observation, action, device):
    """Encode, then roll out from the first CONTEXT real frames.

    Returns true latents (B,L+1,N,D) and imagined latents for frames
    CONTEXT..L, shape (B,L+1-CONTEXT,N,D).
    """
    latent = model.encode(observation.to(device))
    action = action.to(device)
    rolled = model.rollout_from_context(
        latent[:, :CONTEXT], action[:, : CONTEXT - 1], action[:, CONTEXT - 1 :]
    )
    return latent, rolled


@torch.no_grad()
def teacher(model, latent, action, device):
    """One-step prediction of frame t from real frames t-3..t-1, t >= CONTEXT."""
    action = action.to(device)
    out = [
        model.predict(latent[:, t - CONTEXT : t], action[:, t - CONTEXT : t])[:, -1:]
        for t in range(CONTEXT, latent.shape[1])
    ]
    return torch.cat(out, dim=1)


def per_horizon_mse(predicted, truth, mask):
    """Mean squared Euclidean position error per horizon: (H,)."""
    squared = (predicted - truth).square().sum(dim=-1).mean(dim=-1)  # (B,H)
    mask = mask.double()
    return (squared * mask).sum(0) / mask.sum(0).clamp_min(1)


def pooled_cf(predicted_delta, true_delta, mask):
    """Per-horizon pooled E_CF, gain and cosine of a counterfactual delta."""
    m = mask.double()
    flat_p = predicted_delta.flatten(2)
    flat_t = true_delta.flatten(2)
    residual = ((flat_p - flat_t).square().sum(-1) * m).sum(0)
    truth = (flat_t.square().sum(-1) * m).sum(0)
    size = (flat_p.square().sum(-1) * m).sum(0)
    cosine = torch.nn.functional.cosine_similarity(flat_p, flat_t, dim=-1, eps=1e-12)
    active = mask & (flat_t.norm(dim=-1) > 1e-6)
    cos = (cosine * active.double()).sum(0) / active.double().sum(0).clamp_min(1)
    return {
        "e_cf": (residual / truth.clamp_min(1e-12)).sqrt().tolist(),
        "gain": (size / truth.clamp_min(1e-12)).sqrt().tolist(),
        "cosine": cos.tolist(),
        "active_anchors": active.sum(0).tolist(),
    }


def fit_direct(train, validation):
    """Observation-space transition f(o, a) -> o' - o; ridge or MLP by validation."""

    def pairs(batch):
        o, a, v = batch["observation"], batch["action"], batch["valid"]
        x = torch.cat([o[:, :-1], a], dim=-1)[v]
        y = (o[:, 1:] - o[:, :-1])[v]
        return x.reshape(-1, x.shape[-1]).double(), y.reshape(-1, y.shape[-1]).double()

    tx, ty = pairs(train)
    vx, vy = pairs(validation)
    mean, std = tx.mean(0), tx.std(0).clamp_min(1e-8)
    tx, vx = (tx - mean) / std, (vx - mean) / std
    device = "cuda" if torch.cuda.is_available() else "cpu"
    best = None
    for family, strengths in (("ridge", RIDGE_STRENGTHS), ("mlp", MLP_DECAYS)):
        for strength in strengths:
            fitted = (
                ridge(tx, ty, strength)
                if family == "ridge"
                else fit_mlp(tx, ty, strength, device)
            )
            error = float((apply_probe(fitted, vx) - vy).square().mean())
            if best is None or error < best[0]:
                best = (error, family, strength, fitted)
    error, family, strength, fitted = best

    def step(observation, action):
        x = (torch.cat([observation.double(), action.double()], dim=-1) - mean) / std
        delta = apply_probe(fitted, x.reshape(-1, x.shape[-1])).reshape(observation.shape)
        return observation.double() + delta

    return step, {"family": family, "strength": strength, "validation_mse": error}


def direct_rollout(step, observation, action):
    """Recursive observation-space rollout from the last context frame."""
    current = observation[:, CONTEXT - 1].double()
    out = []
    for t in range(CONTEXT - 1, action.shape[1]):
        current = step(current, action[:, t])
        out.append(current)
    return torch.stack(out, dim=1)


def load_cf_pair(data_root, block):
    """Reference and self-intervention branches for the test anchors."""
    root = Path(data_root)
    cf = torch.load(root / "counterfactual_test.pt", map_location="cpu", weights_only=True)
    ref = torch.load(root / "samples_correlated.pt", map_location="cpu", weights_only=True)
    ids = cf["anchor_id"]
    ref = {k: v[ids] for k, v in ref.items() if k != "anchor_id"}
    cf = {k: v for k, v in cf.items() if k != "anchor_id"}
    manifest = json.loads((root / "manifest.json").read_text())
    start = manifest.get("intervention_start_step", 0)
    if start % block or start // block < CONTEXT:
        raise ValueError(
            f"intervention_start_step={start} must leave {CONTEXT} identical context blocks"
        )
    return block_snippets(ref, block), block_snippets(cf, block), start // block


def evaluate_run(run, batches, cf_pair, direct, args):
    config = yaml.safe_load((run / "resolved_config.yaml").read_text())
    model = load_model(run / "model.pt", args.device)
    model.eval()
    probe, selection = fit_position_probe(
        model, batches["train"], batches["validation"], args.device, args.probe, "true"
    )
    test = batches["test"]
    obs, act = test["observation"], test["action"]
    truth = test["agent_state"][..., :2].double()
    live = alive(test["valid"])
    ok = live[:, : CONTEXT - 1].all(dim=1)
    mask = live[:, CONTEXT - 1 :] & ok.unsqueeze(1)  # horizons 1..H

    latent, rolled = imagine(model, obs, act, args.device)
    one_step = teacher(model, latent, act, args.device)
    future = truth[:, CONTEXT:]
    last = truth[:, CONTEXT - 1 : CONTEXT]
    step = torch.arange(1, future.shape[1] + 1, dtype=torch.double).view(1, -1, 1, 1)
    velocity = last - truth[:, CONTEXT - 2 : CONTEXT - 1]
    # Navigation observations start with [pos_x, pos_y]: the direct model's
    # position needs no probe.
    references = {
        "rollout": decode_agents(probe, rolled.cpu()),
        "teacher_one_step": decode_agents(probe, one_step.cpu()),
        "probe_floor": decode_agents(probe, latent[:, CONTEXT:].cpu()),
        "persistence": last.expand_as(future),
        "constant_velocity": last + step * velocity,
        "direct_rollout": direct_rollout(direct, obs, act)[..., :2],
    }
    mse = {k: per_horizon_mse(v, future, mask).tolist() for k, v in references.items()}
    persistence = torch.tensor(mse["persistence"])
    relative = {k: (torch.tensor(v) / persistence).tolist() for k, v in mse.items()}

    # Latent-space rollout error on this model's own scale.
    target = latent[:, CONTEXT:]
    latent_error = (rolled - target).square().mean(dim=(-1, -2)).double().cpu()
    latent_scale = target[mask.to(target.device)].var(dim=0).mean().item()
    latent_rel = (
        (latent_error * mask.double()).sum(0) / mask.double().sum(0).clamp_min(1)
        / latent_scale
    ).tolist()

    ref, cf, first = cf_pair
    ref_latent, ref_rolled = imagine(model, ref["observation"], ref["action"], args.device)
    cf_latent, cf_rolled = imagine(model, cf["observation"], cf["action"], args.device)
    if not torch.allclose(ref_latent[:, :CONTEXT], cf_latent[:, :CONTEXT]):
        raise ValueError("Counterfactual branches do not share their context")
    cf_mask = alive(ref["valid"] & cf["valid"])[:, CONTEXT - 1 :]
    true_delta = (cf["position"] - ref["position"])[:, CONTEXT:]
    model_delta = decode_agents(probe, cf_rolled.cpu()) - decode_agents(probe, ref_rolled.cpu())
    floor_delta = decode_agents(probe, cf_latent[:, CONTEXT:].cpu()) - decode_agents(
        probe, ref_latent[:, CONTEXT:].cpu()
    )
    direct_delta = (
        direct_rollout(direct, cf["observation"], cf["action"])
        - direct_rollout(direct, ref["observation"], ref["action"])
    )[..., :2]
    counterfactual = {
        # Rollout horizon h scores frame CONTEXT-1+h; the first changed action
        # is block `first`, so frame first+1 = horizon first-CONTEXT+2.
        "first_affected_horizon": first - CONTEXT + 2,
        "model": pooled_cf(model_delta, true_delta, cf_mask),
        "probe_floor": pooled_cf(floor_delta, true_delta, cf_mask),
        "direct": pooled_cf(direct_delta, true_delta, cf_mask),
    }
    record = {
        "run": str(run),
        "cell": run.name.rsplit("_seed", 1)[0],
        "seed": config["seed"],
        "sigreg_weight": config["train"]["sigreg_weight"],
        "reference_window": config["train"].get("reference_window", "first"),
        "train_fraction": config["data"].get("train_fraction", 1.0),
        "dynamics_epochs": config["train"]["dynamics_epochs"],
        "probe": args.probe,
        "probe_selection": selection,
        "anchors_per_horizon": mask.sum(0).tolist(),
        "position_mse": mse,
        "relative_to_persistence": relative,
        "latent_rollout_error_over_variance": latent_rel,
        "self_counterfactual": counterfactual,
    }
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("runs", type=Path, help="directory of <cell>_seed<N> runs")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--probe", choices=("linear", "mlp"), default="mlp")
    parser.add_argument("--regime", default="independent")
    parser.add_argument("--action-block", type=int, default=5)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    batches = {
        split: stack_dataset(
            OfflineSequences(
                args.data, args.regime, split, action_block=args.action_block,
                state_input="observation",
            )
        )
        for split in ("train", "validation", "test")
    }
    agents = batches["test"]["observation"].shape[2]
    if agents != 1:
        raise ValueError(f"The single-agent gate needs a one-agent bank, got {agents}")
    direct, direct_selection = fit_direct(batches["train"], batches["validation"])
    cf_pair = load_cf_pair(args.data, args.action_block)
    runs = sorted(p.parent for p in args.runs.glob("*/model.pt"))
    if not runs:
        raise ValueError(f"No checkpoints under {args.runs}")
    records = []
    for index, run in enumerate(runs, 1):
        record = evaluate_run(run, batches, cf_pair, direct, args)
        rel = record["relative_to_persistence"]["rollout"]
        cf = record["self_counterfactual"]["model"]
        print(
            f"[{index:02d}/{len(runs)}] {run.name}: rollout/persistence "
            f"h1 {rel[0]:.3f} h{len(rel)} {rel[-1]:.3f} | self-CF E "
            + " ".join(f"{e:.2f}" for e in cf["e_cf"])
            + " | gain " + " ".join(f"{g:.2f}" for g in cf["gain"]),
            flush=True,
        )
        records.append(record)
    summary = {
        "question": "Can the latent world model imagine a single agent's own motion?",
        "data": str(args.data),
        "probe": args.probe,
        "context_frames": CONTEXT,
        "test_snippets": int(batches["test"]["observation"].shape[0]),
        "test_roots": int(batches["test"]["episode_id"].unique().numel()),
        "direct_selection": direct_selection,
        "records": records,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"wrote {args.output / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
