# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the license found in the LICENSE file in the root directory.
"""Experiment 24/G6a: does planner-aware data aggregation remove the
optimizer-induced OOD failure Gate 5 (job 1335) registered as the primary
intervention?

Gate 5 measured teacher-forced ball-position RMSE growing 3.8x from CEM
iteration 1 to 30 on the frozen job-1331 structured surrogate: CEM optimizes
into states the training distribution under-covers. The causal hypothesis
here is that adding the model's *own* CEM hard negatives to training removes
that gap; a matched-size generic augmentation is the control that separates
"planner-aware coverage helps" from "more data helps".

Two DAgger rounds, three seeds, each seed its own aggregation chain:

    D0 --train--> M0 --own CEM on collection roots--> D_CEM,0
    D1 = D0 u D_CEM,0 --train--> M1 --own CEM--> D_CEM,1
    D2 = D1 u D_CEM,1 --train--> M2

M0 and D0 are job 1331's own full-coverage model/dataset, reused rather than
retrained. Collection roots are the 96 split=0 (train) root episodes of the
same `initial_states.pt` bank Gate 5's frozen 16 test roots (split=2) come
from -- disjoint by construction, and checked explicitly below. Candidates
are drawn at CEM iterations 1/5/10/20/30 (`HARD_STAGES`), with the model's
own predicted cost selecting the hardest `--hard-plans` per stage per root,
then replayed in the true simulator for labels -- simulator truth never
enters candidate selection, only supervision, matching Gate 5's contract.

The generic control collects the same roots and the identical per-root plan
count from uniform random actions instead of any model's CEM, and is shared
across seeds each round since it does not depend on a model. Both additions
use one augmentation family. Training draws 50% base and 50% augmentation with
a fixed epoch/step budget, so source is the only optimizer-mass difference.

The underlying 14-D surrogate omits movable linkage bodies and is now known to
be non-Markov. This corrected runner remains available for provenance and code
review, but must not be submitted until a full-state successor replaces it.

Every round of every arm is written out as a Gate-5-compatible source
directory (`result/model_full_{seed}.pt` + `result/result.json` +
`coverage/oracle_plans.pt`, the last copied unchanged from job 1331) so
`planner_tail_failure.py --source <dir>` reruns Gate 5 on it unmodified.
"""

import argparse
import json
import shutil
from pathlib import Path

import torch

from benchmarl.environments import VmasTask
from examples.world_model.cem import CEMConfig, cem_plan
from examples.world_model.decision_information import take_snapshot
from examples.world_model.mpc import action_bounds, unpack_actions
from examples.world_model.planner_tail_failure import load_surrogate
from examples.world_model.snapshot_restore import restore_state
from examples.world_model.structured_surrogate import (
    HARD_STAGES,
    blockify,
    cat_records,
    live_structured_state,
    replay_blocked,
    surrogate_cost,
    train_surrogate,
)


SEEDS = (9100, 9101, 9102)
G6A_AUGMENTATION_FAMILY = 3
G6A_ROOT_OFFSET = 2_000_000
G6A_TARGETED_SOURCE = 9
G6A_GENERIC_SOURCE = 10


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def assert_disjoint_from_frozen_test_roots(collection_ids, frozen_test_ids):
    """Refuse to run if collection would ever touch Gate 5's frozen 16."""
    frozen_test_ids = torch.as_tensor(frozen_test_ids)
    if torch.isin(collection_ids, frozen_test_ids).any():
        raise ValueError(
            "G6a collection roots must be disjoint from Gate 5's frozen test roots"
        )


def namespaced_g6a_root_ids(ids):
    """Keep initial-state-bank roots distinct from historical anchor IDs."""
    ids = torch.as_tensor(ids, dtype=torch.long)
    if (ids < 0).any():
        raise ValueError("G6a source root IDs must be non-negative")
    if (ids >= G6A_ROOT_OFFSET).any():
        raise ValueError("G6a source root IDs exceed their namespace range")
    return ids + G6A_ROOT_OFFSET


def structured_state_and_bounds(task, snapshot, device):
    """The live structured state and action bounds at a batch of snapshots.

    Mirrors `structured_surrogate.root_model_inputs`, but returns the
    *structured* 14-D state the CEM cost function needs, not the "physical"
    observation frame that one uses.
    """
    roots = snapshot["steps"].shape[0]
    env = task.get_env_fun(roots, True, 0, device)()
    env.reset()
    restore_state(env, snapshot)
    state = live_structured_state(env)
    low, high = action_bounds(env)
    agents = len(env._env.world.agents)
    primitive = env.full_action_spec_unbatched["agents", "action"].shape[-1]
    env.close()
    return state, agents, primitive, low, high


@torch.no_grad()
def collect_own_cem_hard_negatives(
    task, model, initial, collection_ids, *, hard_plans, block, cem_config,
    root_batch, replay_batch, seed, device,
):
    """CEM hard negatives scored by the *current* model, on roots disjoint
    from Gate 5's frozen 16 by construction (`collection_ids` is restricted
    to split=0 roots before this is ever called). Simulator truth labels the
    replayed candidates; it never enters CEM's own candidate scoring."""
    records = []
    for start in range(0, collection_ids.numel(), root_batch):
        ids = collection_ids[start : start + root_batch]
        snapshot = take_snapshot(initial["snapshot"], ids, device)
        state, agents, primitive, low, high = structured_state_and_bounds(
            task, snapshot, device
        )
        plan_dim = agents * primitive * block

        def cost_fn(candidates):
            return surrogate_cost(
                model, state, unpack_actions(candidates, block), block,
                objective="probability",
            )

        cem = cem_plan(
            cost_fn,
            action_dim=plan_dim,
            action_low=low,
            action_high=high,
            config=CEMConfig(
                horizon=cem_config.horizon,
                num_samples=cem_config.num_samples,
                num_elites=cem_config.num_elites,
                num_iters=max(HARD_STAGES),
            ),
            batch_size=ids.numel(),
            device=device,
            generator=torch.Generator(device=device).manual_seed(seed + start),
            record_candidates=True,
        )
        selected, stage_values = [], []
        for stage in HARD_STAGES:
            candidates = cem.candidate_history[stage - 1]
            costs = cost_fn(candidates.to(device)).cpu()
            elite = costs.topk(hard_plans, largest=False).indices
            chosen = candidates.gather(
                1,
                elite[..., None, None].expand(
                    ids.numel(), hard_plans, cem_config.horizon, plan_dim
                ),
            )
            selected.append(chosen)
            stage_values.extend([stage] * hard_plans)
        plans = torch.cat(selected, dim=1)
        data = replay_blocked(task, snapshot, plans, block, replay_batch, device)
        plan_count = plans.shape[1]
        root_id = namespaced_g6a_root_ids(ids).repeat_interleave(plan_count)
        stage = torch.tensor(stage_values).repeat(ids.numel())
        records.append(
            blockify(
                data,
                split=0,
                family=G6A_AUGMENTATION_FAMILY,
                source=G6A_TARGETED_SOURCE,
                root_id=root_id,
                stage=stage,
                action_block=block,
            )
        )
        del cem, data
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return records


@torch.no_grad()
def collect_generic_matched(
    task, initial, collection_ids, *, plans_per_root, horizon, plan_dim, block,
    root_batch, replay_batch, low, high, seed, device,
):
    """The same roots and the same per-root plan count as one targeted-arm
    collection pass, drawn uniformly at random instead of from any model's
    CEM -- the matched-size control for "more data" against "planner-aware
    data". Does not depend on a model, so it is shared across seeds."""
    records = []
    generator = torch.Generator().manual_seed(seed)
    for start in range(0, collection_ids.numel(), root_batch):
        ids = collection_ids[start : start + root_batch]
        snapshot = take_snapshot(initial["snapshot"], ids, "cpu")
        plans = (
            torch.rand(ids.numel(), plans_per_root, horizon, plan_dim, generator=generator)
            * (high - low)
            + low
        )
        data = replay_blocked(task, snapshot, plans, block, replay_batch, device)
        root_id = namespaced_g6a_root_ids(ids).repeat_interleave(plans_per_root)
        records.append(
            blockify(
                data,
                split=0,
                family=G6A_AUGMENTATION_FAMILY,
                source=G6A_GENERIC_SOURCE,
                root_id=root_id,
                stage=-1,
                action_block=block,
            )
        )
    return records


def write_gate5_source(output_dir, seeds, models, frozen, coverage_source_dir):
    """Assemble a `planner_tail_failure.py --source` directory unchanged by
    Gate 5: the frozen protocol config plus this round's checkpoints, with
    job 1331's own `oracle_plans.pt` -- the frozen 16 test roots never
    change -- copied through verbatim."""
    result_dir = output_dir / "result"
    result_dir.mkdir(parents=True, exist_ok=True)
    for seed, (model, fit) in models.items():
        torch.save(
            {"state_dict": model.state_dict(), "mix": "full", "seed": seed, "fit": fit},
            result_dir / f"model_full_{seed}.pt",
        )
    result = {
        "selected_objective": {"full": "probability"},
        "fits": [{"mix": "full", "seed": seed} for seed in seeds],
        "test_root_ids": frozen["test_root_ids"],
        "config": {
            "horizon": frozen["horizon"],
            "num_samples": frozen["num_samples"],
            "num_elites": frozen["num_elites"],
            "num_iters": frozen["num_iters"],
            "control_seed": frozen["control_seed"],
        },
    }
    write_json(result_dir / "result.json", result)
    coverage_dir = output_dir / "coverage"
    coverage_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(
        coverage_source_dir / "oracle_plans.pt", coverage_dir / "oracle_plans.pt"
    )


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source", type=Path, default=Path("outputs/structured_surrogate_1331")
    )
    parser.add_argument("--data", type=Path, default=Path("outputs/buzz_wire_1196/data"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--root-batch", type=int, default=8)
    parser.add_argument("--replay-batch", type=int, default=256)
    parser.add_argument("--hard-plans", type=int, default=4)
    parser.add_argument("--targeted-seed", type=int, default=9400)
    parser.add_argument("--generic-seed", type=int, default=9500)
    # Training hyperparameters, matching job 1331's `structured_surrogate run`.
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    return parser


def main():
    args = build_parser().parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    prior = json.loads((args.source / "result/result.json").read_text())
    if prior["selected_objective"]["full"] != "probability":
        raise ValueError("G6a freezes job 1331's full/probability objective")
    base_data = torch.load(
        args.source / "coverage/coverage.pt", map_location="cpu", weights_only=False
    )
    manifest = json.loads((args.data / "manifest.json").read_text())
    initial = torch.load(
        args.data / "initial_states.pt", map_location="cpu", weights_only=False
    )
    task = VmasTask.BUZZ_WIRE.get_from_yaml()
    block = manifest["action_block"]

    frozen = {
        "horizon": int(prior["config"]["horizon"]),
        "num_samples": int(prior["config"]["num_samples"]),
        "num_elites": int(prior["config"]["num_elites"]),
        "num_iters": int(prior["config"]["num_iters"]),
        "control_seed": int(prior["config"]["control_seed"]),
        "test_root_ids": prior["test_root_ids"],
    }
    cem_config = CEMConfig(
        horizon=frozen["horizon"],
        num_samples=frozen["num_samples"],
        num_elites=frozen["num_elites"],
        num_iters=frozen["num_iters"],
    )

    collection_ids = (initial["split"] == 0).nonzero(as_tuple=True)[0]
    assert_disjoint_from_frozen_test_roots(collection_ids, frozen["test_root_ids"])

    probe = take_snapshot(initial["snapshot"], collection_ids[:1], args.device)
    _state, agents, primitive, low, high = structured_state_and_bounds(
        task, probe, args.device
    )
    plan_dim = agents * primitive * block
    plans_per_root = len(HARD_STAGES) * args.hard_plans
    base_train_transitions = int((base_data["split"] == 0).sum())
    samples_per_epoch = 2 * base_train_transitions
    matched_training = {
        "sampling_scheme": "base_augmentation",
        "augmentation_family": G6A_AUGMENTATION_FAMILY,
        "samples_per_epoch": samples_per_epoch,
        "fixed_epochs": True,
    }

    data_targeted = {seed: base_data for seed in SEEDS}
    models_targeted = {
        seed: load_surrogate(args.source / f"result/model_full_{seed}.pt", args.device)[0]
        for seed in SEEDS
    }
    data_generic = base_data

    rounds_summary = []
    for round_index in range(1, args.rounds + 1):
        round_summary = {"round": round_index}

        print(f"round {round_index}: collecting generic control", flush=True)
        generic_records = collect_generic_matched(
            task, initial, collection_ids,
            plans_per_root=plans_per_root, horizon=cem_config.horizon,
            plan_dim=plan_dim, block=block, root_batch=args.root_batch,
            replay_batch=args.replay_batch, low=low, high=high,
            seed=args.generic_seed + round_index, device=args.device,
        )
        data_generic = cat_records([data_generic] + generic_records)
        round_summary["generic_transitions"] = int(
            data_generic["state"].shape[0]
        )
        generic_models = {}
        for seed in SEEDS:
            model, fit = train_surrogate(
                data_generic, "full", seed, args, **matched_training
            )
            generic_models[seed] = (model, fit)
            round_summary[f"generic_validation_loss_{seed}"] = fit["validation_loss"]
            round_summary[f"generic_optimizer_steps_{seed}"] = fit[
                "optimizer_steps"
            ]
        write_gate5_source(
            args.output / f"g6a_generic_round{round_index}", SEEDS, generic_models,
            frozen, args.source / "coverage",
        )
        print(f"round {round_index}: generic control trained and written", flush=True)

        targeted_models = {}
        for seed in SEEDS:
            print(f"round {round_index}: seed {seed} own-CEM collection", flush=True)
            targeted_records = collect_own_cem_hard_negatives(
                task, models_targeted[seed], initial, collection_ids,
                hard_plans=args.hard_plans, block=block, cem_config=cem_config,
                root_batch=args.root_batch, replay_batch=args.replay_batch,
                seed=args.targeted_seed + round_index * 1000 + seed, device=args.device,
            )
            data_targeted[seed] = cat_records([data_targeted[seed]] + targeted_records)
            model, fit = train_surrogate(
                data_targeted[seed], "full", seed, args, **matched_training
            )
            models_targeted[seed] = model
            targeted_models[seed] = (model, fit)
            round_summary[f"targeted_validation_loss_{seed}"] = fit["validation_loss"]
            round_summary[f"targeted_optimizer_steps_{seed}"] = fit[
                "optimizer_steps"
            ]
            round_summary[f"targeted_transitions_{seed}"] = int(
                data_targeted[seed]["state"].shape[0]
            )
        write_gate5_source(
            args.output / f"g6a_targeted_round{round_index}", SEEDS,
            targeted_models,
            frozen, args.source / "coverage",
        )
        print(f"round {round_index}: targeted arm trained and written", flush=True)
        rounds_summary.append(round_summary)

    write_json(
        args.output / "aggregation_summary.json",
        {
            "question": (
                "Does planner-aware CEM-hard-negative aggregation remove the "
                "optimizer-induced OOD Gate 5 (job 1335) registered as G6a?"
            ),
            "source_job": 1331,
            "frozen_protocol": frozen,
            "collection_roots": collection_ids.tolist(),
            "hard_plans_per_stage": args.hard_plans,
            "plans_per_root": plans_per_root,
            "root_id_namespace": {
                "kind": "initial_states",
                "offset": G6A_ROOT_OFFSET,
            },
            "augmentation_family": G6A_AUGMENTATION_FAMILY,
            "augmentation_sources": {
                "targeted": G6A_TARGETED_SOURCE,
                "generic": G6A_GENERIC_SOURCE,
            },
            "training_contract": {
                "sampling_scheme": "base_augmentation",
                "expected_sampling_mass": {"base": 0.5, "augmentation": 0.5},
                "samples_per_epoch": samples_per_epoch,
                "epochs": args.epochs,
                "fixed_epochs": True,
            },
            "rounds": rounds_summary,
        },
    )
    print(f"wrote G6a aggregation summary to {args.output}", flush=True)


if __name__ == "__main__":
    main()
