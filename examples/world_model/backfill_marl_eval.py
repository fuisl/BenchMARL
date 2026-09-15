#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#
"""Write the marl-eval file for runs that finished before `train.py` wrote one.

Job 1201 produced 192 checkpoints -- 144 width/regularization settings and a 48
run replication on a second Buzz Wire bank -- and none of them carry a
`marl_eval.json`, so `report.py` cannot see them at all. The audit records them
as available for rescoring with no validated result attached.

Nothing is recomputed. Each run already holds the metrics `train.py` evaluated
and the configuration it resolved; this writes the same file from them, through
the same `JsonWriter`, so the runs read natively alongside every other one.

Two naming decisions, both to stop distinct experiments being pooled as if they
were extra seeds of one:

* A width/regularization setting is a different algorithm, not a different seed
  of the baseline, so its latent dimension and SIGReg weight join the algorithm
  name. Pooling dim 16 with dim 192 under `independent_correlated` would average
  models of 1.24M and 3.81M parameters into one row.
* A second bank is a different evaluation, not more seeds, so the replication
  gets its own task name. The audit is explicit that bank replication is
  reported separately.

Run:
    python -m examples.world_model.backfill_marl_eval outputs/bw_remaining_1201/sigreg
"""

import argparse
import json
from pathlib import Path

import torch
import yaml

from benchmarl.experiment.logger import JsonWriter
from examples.world_model.train import MARL_EVAL_FILE, MODEL_NAME, REPORTED_METRICS


def describe(config, directory):
    """Algorithm, task and environment names for one finished run."""
    environment, task = config["data"]["root"], None
    manifest = json.loads((Path(environment) / "manifest.json").read_text())
    environment, task = manifest["task_name"].split("/")
    algorithm = f"{config['model']['kind']}_{config['data']['regime']}"

    # The sweep directory names the setting; the resolved config carries the
    # values, so read them rather than parsing the path.
    if "sigreg" in directory.parts:
        algorithm = (
            f"{algorithm}_d{config['model']['dim']}"
            f"_w{config['train']['sigreg_weight']}"
        )
    if "holdout" in directory.parts:
        task = f"{task}_holdout"
    return algorithm, task, environment


def backfill(directory):
    """Write one run's marl-eval file; return the identity it was written under."""
    config = yaml.safe_load((directory / "resolved_config.yaml").read_text())
    metrics = json.loads((directory / "metrics.json").read_text())
    algorithm, task, environment = describe(config, directory)
    writer = JsonWriter(
        folder=str(directory),
        name=MARL_EVAL_FILE,
        algorithm_name=algorithm,
        task_name=task,
        environment_name=environment,
        seed=config["seed"],
    )
    # Identical negation and renaming to train.py: everything downstream assumes
    # larger is better, and a field called `rollout_error` holding a negative
    # number would mislead anyone reading the file without that table beside it.
    writer.write(
        total_frames=0,
        metrics={
            (f"neg_{name}" if lower_is_better else name): torch.tensor(
                [-value if lower_is_better else value]
            )
            for name, lower_is_better in REPORTED_METRICS.items()
            if (value := metrics.get(name)) is not None
        },
        evaluation_step=0,
    )
    return algorithm, task, config["seed"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("sweeps", type=Path, nargs="+")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="rewrite files that already exist, after a naming change",
    )
    args = parser.parse_args()

    written, skipped, identities = 0, 0, {}
    for sweep in args.sweeps:
        for config_path in sorted(sweep.rglob("resolved_config.yaml")):
            directory = config_path.parent
            if not (directory / "metrics.json").exists():
                continue
            if (directory / MARL_EVAL_FILE).exists() and not args.overwrite:
                skipped += 1
                continue
            algorithm, task, seed = backfill(directory)
            identities.setdefault((task, algorithm), []).append(seed)
            written += 1

    for (task, algorithm), seeds in sorted(identities.items()):
        print(f"  {task:20s} {algorithm:40s} {len(seeds)} seeds")
    print(f"\nwrote {written} {MARL_EVAL_FILE} files, skipped {skipped} existing")


if __name__ == "__main__":
    main()
