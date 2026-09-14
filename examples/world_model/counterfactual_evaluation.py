#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#
"""C7: in-distribution error against counterfactual error on the same states.

The paper's headline claim. From one restored state we compare the model's
prediction under the logged joint action against its prediction under a joint
action where only agent 1's x component is flipped, scoring both against the
simulator's true outcome.

The measurement is taken on the **non-intervened agents** (0, 2, 3). Their own
actions are identical in both branches, so their next state can only differ
through a cross-agent effect. That makes the gap

    G_CF = E_CF - E_ID

a direct read on joint-action dependence rather than a proxy. `independent` is
bit-exactly unable to react to agent 1 (asserted in the model tests), so it is
the floor; a relational model that captures the coupling should show a smaller
gap, and only where the simulator says an effect exists.

Every model is scored on the *same* evaluation pairs regardless of which regime
it trained on, so a correlated-trained model is being asked about a region its
data never covered while an independent-trained model has seen it. That contrast
is M1 Row 2.

Errors live in each model's own latent space, so the reported quantity is the
scale-free relative gap G_CF / E_ID, with absolutes shown alongside.

Run:
    python -m examples.world_model.counterfactual_evaluation \\
        outputs/interaction_control_1194/transport --data outputs/transport_data_1190
"""

import argparse
import json
from pathlib import Path

import torch
import yaml

from examples.world_model.compare_baselines import bootstrap_interval, mean
from examples.world_model.stratified_evaluation import effect_labels
from examples.world_model.train import load_model

BASELINES = ("independent", "joint", "relational")
INTERVENED = 1  # the agent whose x action the collector flips


def blocked(action, block):
    """(B,T,N,d_a) primitive -> (B,T/block,N,block*d_a), time before coordinate."""
    batch, steps, agents, dim = action.shape
    return (
        action.view(batch, steps // block, block, agents, dim)
        .permute(0, 1, 3, 2, 4)
        .reshape(batch, steps // block, agents, block * dim)
    )


@torch.no_grad()
def branch(model, observation, action, target, agents, device):
    """One-block prediction and its target, restricted to `agents`.

    Returns per-anchor squared error plus the raw prediction and truth, so the
    caller can also measure how the prediction *moves* under an intervention.
    """
    start = observation.unsqueeze(1).to(device)  # (B,1,N,O)
    predicted = model.predict(model.encode(start), action.to(device))[:, 0, agents]
    truth = model.encode(target.unsqueeze(1).to(device))[:, 0, agents]
    error = (predicted - truth).square().mean(dim=(1, 2)).cpu()
    return error, predicted.cpu(), truth.cpu()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", type=Path)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--block", type=int, default=5)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    load = lambda name: torch.load(  # noqa: E731
        args.data / name, map_location="cpu", weights_only=True
    )
    reference = load("samples_correlated.pt")
    counterfactual = load("counterfactual_test.pt")
    ids = counterfactual["anchor_id"]

    # One action block from the shared anchor state; the M3 audit shows the
    # simulator has no cross-agent effect at all after a single primitive step,
    # so a one-step probe would test a coupling that does not yet exist.
    block = args.block
    observation = reference["observation"][ids, 0]
    non_intervened = [i for i in range(observation.shape[1]) if i != INTERVENED]
    logged = blocked(reference["action"][ids, :block], block)
    intervened = blocked(counterfactual["action"][:, :block], block)
    target_id = reference["next_observation"][ids, block - 1]
    target_cf = counterfactual["next_observation"][:, block - 1]

    # The collector zeroes actions once an episode stops being live, and the two
    # branches terminate at different steps, so only anchors whose block is live
    # in BOTH branches carry a comparable intervention. On Transport nothing
    # terminates inside a snippet and this keeps everything; on Buzz Wire, which
    # ends on wall contact, it is what makes the comparison well posed.
    live = reference["valid"][ids, :block].all(dim=1) & counterfactual["valid"][
        :, :block
    ].all(dim=1)
    if not torch.equal(observation, counterfactual["observation"][:, 0]):
        raise ValueError("Reference and counterfactual start from different states")
    if not torch.equal(
        logged[live][:, :, non_intervened], intervened[live][:, :, non_intervened]
    ):
        raise ValueError("Non-intervened agents' actions must be identical while live")
    if torch.equal(logged[live][:, :, INTERVENED], intervened[live][:, :, INTERVENED]):
        raise ValueError("The intervention did not change agent 1's action")

    _, active = effect_labels(args.data, block)
    active = active & live
    moved = (target_cf[:, non_intervened] - target_id[:, non_intervened]).abs().amax(
        dim=(1, 2)
    ) > 0
    print(
        f"test anchors: {ids.numel()}   live through the block in both branches: "
        f"{int(live.sum())}   simulator says a cross-agent effect exists in "
        f"{int(active.sum())} (label) / {int((moved & live).sum())} (observation moved)"
    )

    results = {}
    for directory in sorted(args.runs.glob("[0-9]*")):
        checkpoint = directory / "model.pt"
        if not checkpoint.exists():
            continue
        config = yaml.safe_load((directory / "resolved_config.yaml").read_text())
        key = (config["data"]["regime"], config["model"]["kind"], config["seed"])
        model = load_model(checkpoint, args.device)
        e_id, pred_id, truth_id = branch(
            model, observation, logged, target_id, non_intervened, args.device
        )
        e_cf, pred_cf, truth_cf = branch(
            model, observation, intervened, target_cf, non_intervened, args.device
        )
        # Intervention response: does the model move its prediction the way the
        # simulator actually moved? This removes the target-difficulty confound
        # in E_CF - E_ID, which compares errors against two different targets.
        # `independent` predicts zero response by construction, so its score is
        # exactly the true effect size and forms the floor.
        response = (
            ((pred_cf - pred_id) - (truth_cf - truth_id)).square().mean(dim=(1, 2))
        )
        inertia = (truth_cf - truth_id).square().mean(dim=(1, 2))
        results[key] = {
            "id": e_id,
            "cf": e_cf,
            "response": response,
            "inertia": inertia,
        }

    regimes = sorted({r for r, _, _ in results})
    seeds = sorted({s for _, _, s in results})
    print(f"scored {len(results)} checkpoints over {len(seeds)} seeds")

    for stratum, mask in (
        ("interaction-active", active),
        ("inactive", (~active) & live),
    ):
        print(f"\n=== {stratum} anchors (n={int(mask.sum())}) ===")
        header = (
            f"{'regime':12s} {'kind':12s} {'E_ID':>10s} {'E_CF':>10s} "
            f"{'G_CF':>11s} {'G_CF/E_ID':>11s}"
        )
        print(header)
        print("-" * len(header))
        summary = {}
        for regime in regimes:
            for kind in BASELINES:
                rows = [
                    v for (r, k, _), v in results.items() if (r, k) == (regime, kind)
                ]
                if not rows:
                    continue
                e_id = mean([float(v["id"][mask].mean()) for v in rows])
                e_cf = mean([float(v["cf"][mask].mean()) for v in rows])
                summary[(regime, kind)] = (e_id, e_cf)
                print(
                    f"{regime:12s} {kind:12s} {e_id:10.5f} {e_cf:10.5f} "
                    f"{e_cf - e_id:+11.5f} {(e_cf - e_id) / e_id:+11.3f}"
                )

        print(
            f"\npaired relative gap vs independent ({stratum}); negative = smaller gap:"
        )
        header = f"{'regime':12s} {'kind':12s} {'mean':>10s} {'95% CI':>24s} {'seeds better':>13s}"
        print(header)
        print("-" * len(header))
        for regime in regimes:
            for kind in ("joint", "relational"):
                pairs = []
                for seed in seeds:
                    base = results.get((regime, "independent", seed))
                    other = results.get((regime, kind, seed))
                    if not (base and other):
                        continue

                    def gap(v, mask=mask):
                        a = float(v["id"][mask].mean())
                        return (float(v["cf"][mask].mean()) - a) / a

                    pairs.append(gap(other) - gap(base))
                if not pairs:
                    continue
                low, high = bootstrap_interval(pairs, mean)
                better = sum(1 for v in pairs if v < 0)
                print(
                    f"{regime:12s} {kind:12s} {mean(pairs):+10.4f} "
                    f"{f'[{low:+.4f}, {high:+.4f}]':>24s} {f'{better}/{len(pairs)}':>13s}"
                )

    print("\n=== INTERVENTION RESPONSE on interaction-active anchors ===")
    print("Error in predicting the CHANGE the intervention causes, relative to")
    print("predicting no change at all. 1.0 = no better than `independent`'s")
    print("structural zero response; below 1.0 = the model captured some of it.\n")
    header = f"{'regime':12s} {'kind':12s} {'response':>11s} {'vs no-response':>16s}"
    print(header)
    print("-" * len(header))
    for regime in regimes:
        floor = mean(
            [
                float(v["inertia"][active].mean())
                for (r, k, _), v in results.items()
                if (r, k) == (regime, "independent")
            ]
        )
        for kind in BASELINES:
            rows = [v for (r, k, _), v in results.items() if (r, k) == (regime, kind)]
            if not rows:
                continue
            value = mean([float(v["response"][active].mean()) for v in rows])
            print(f"{regime:12s} {kind:12s} {value:11.6f} {value / floor:15.3f}x")

    print("\npaired response vs independent (negative = captured more of the effect):")
    header = f"{'regime':12s} {'kind':12s} {'mean':>12s} {'95% CI':>28s} {'seeds better':>13s}"
    print(header)
    print("-" * len(header))
    for regime in regimes:
        for kind in ("joint", "relational"):
            pairs = []
            for seed in seeds:
                base = results.get((regime, "independent", seed))
                other = results.get((regime, kind, seed))
                if base and other:
                    pairs.append(
                        float(other["response"][active].mean())
                        - float(base["response"][active].mean())
                    )
            if not pairs:
                continue
            low, high = bootstrap_interval(pairs, mean)
            better = sum(1 for v in pairs if v < 0)
            print(
                f"{regime:12s} {kind:12s} {mean(pairs):+12.6f} "
                f"{f'[{low:+.6f}, {high:+.6f}]':>28s} {f'{better}/{len(pairs)}':>13s}"
            )

    print(
        json.dumps(
            {
                "anchors": int(ids.numel()),
                "active": int(active.sum()),
                "observation_moved": int(moved.sum()),
            }
        )
    )


if __name__ == "__main__":
    main()
