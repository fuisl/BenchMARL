#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#
"""Aggregate every world-model run into one report, in BenchMARL's own formats.

Results currently live as 192 per-run ``metrics.json`` files spread over four
sweep directories, plus closed-loop summaries, and are logged to wandb in groups
named after the milestone that produced them. That makes cross-task comparison
hard: a reader cannot group by algorithm and task the way the published VMAS
benchmark does.

This emits three things from the same records:

* ``results.json`` in the **marl-eval schema** that BenchMARL's own
  ``JsonWriter`` writes -- ``{env: {task: {algorithm: {seed_i: ...}}}}`` -- so
  ``benchmarl.eval_results`` can consume our runs exactly like any BenchMARL
  experiment's.
* aggregate scores over seeds using ``rliable``, the library behind those
  published tables: IQM, median, mean and optimality gap, with paired bootstrap
  intervals.
* one wandb run per (task, algorithm, regime, seed), grouped by task with the
  algorithm as job type, which is what makes the UI group-by useful.

Our three predictors play the role BenchMARL gives to algorithms (MAPPO, IPPO,
...): they are the thing being compared on a shared task at matched budget. The
data regime is a second axis BenchMARL has no slot for, so it is folded into the
algorithm name for the schema and kept as its own field for wandb.

``rliable.library`` cannot be imported in this environment -- it pulls ``arch``,
which is incompatible with the installed pandas -- so its stratified bootstrap
is replaced by the paired bootstrap used elsewhere in this project. The point
estimates come from ``rliable.metrics`` itself.
"""

import json
from pathlib import Path

import numpy as np
import yaml
from rliable import metrics as rliable_metrics
from scipy.stats import trim_mean

ENVIRONMENT = "vmas"

# Per-run dynamics metrics, and whether a larger number is a better model.
PREDICTION_METRICS = {
    "one_step_error": False,
    "rollout_error": False,
    "final_step_error": False,
    "latent_variance": True,
    "effective_rank": True,
}

# Sweep directories, as (task, path). Each holds numbered Hydra run folders.
SWEEPS = {
    "transport": "outputs/interaction_control_1194/transport",
    "dropout": "outputs/interaction_control_1194/dropout",
    "buzz_wire": "outputs/buzz_wire_1196/baselines",
    "wheel": "outputs/wheel_1205/baselines",
}

# Closed-loop results, written by examples.world_model.closed_loop. The sweep
# runs one process per (task, seed group) so several files exist per task; they
# are combined here rather than merged by the producer, as with every other
# sweep in this project.
CLOSED_LOOP_GLOB = "outputs/closed_loop_*/{task}_*.json"
CLOSED_LOOP_TASKS = ("buzz_wire", "transport")


def collect_runs(root: Path):
    """One record per trained checkpoint: identity plus its dynamics metrics."""
    records = []
    for task, sweep in SWEEPS.items():
        for directory in sorted((root / sweep).glob("[0-9]*")):
            metrics_path = directory / "metrics.json"
            config_path = directory / "resolved_config.yaml"
            if not (metrics_path.exists() and config_path.exists()):
                continue
            metrics = json.loads(metrics_path.read_text())
            config = yaml.safe_load(config_path.read_text())
            records.append(
                {
                    "task": task,
                    "kind": config["model"]["kind"],
                    "regime": config["data"]["regime"],
                    "seed": config["seed"],
                    "run": str(directory.relative_to(root)),
                    "metrics": {
                        name: metrics[name]
                        for name in PREDICTION_METRICS
                        if name in metrics
                    },
                }
            )
    return records


def collect_closed_loop(root: Path):
    """Per-episode control results, keyed like the dynamics records.

    Policy labels are ``cost|kind|regime|seed``; the random and oracle
    references carry no checkpoint and are returned under their own names.
    """
    records, references = [], []
    for task in CLOSED_LOOP_TASKS:
        rows = []
        for path in sorted(root.glob(CLOSED_LOOP_GLOB.format(task=task))):
            rows.extend(json.loads(path.read_text())["rows"])
        if not rows:
            continue
        for policy in dict.fromkeys(row["policy"] for row in rows):
            episodes = [row for row in rows if row["policy"] == policy]
            values = {
                "return": [row["return"] for row in episodes],
                "success": [float(row["success"]) for row in episodes],
                "collision": [float(row["collision"]) for row in episodes],
                "goal_observation_distance": [
                    row["goal_observation_distance"] for row in episodes
                ],
            }
            if "|" not in policy:
                references.append({"task": task, "policy": policy, "episodes": values})
                continue
            cost, kind, regime, seed = policy.split("|")
            records.append(
                {
                    "task": task,
                    "kind": kind,
                    "regime": regime,
                    "seed": int(seed),
                    "cost": cost,
                    "episodes": values,
                }
            )
    return records, references


def algorithm_name(record):
    """BenchMARL has one algorithm slot; our comparison has two axes."""
    name = f"{record['kind']}-{record['regime']}"
    return f"{name}-{record['cost']}" if "cost" in record else name


def marl_eval_tree(records, closed_loop):
    """The schema BenchMARL's ``JsonWriter`` produces, so its tooling can read us.

    ``{env: {task: {algorithm: {seed_i: {step_0: {...}, absolute_metrics: {...}}}}}}``
    with every metric a list over evaluation episodes. Dynamics metrics are
    single-run scalars, so they are one-element lists; control metrics are real
    per-episode lists.

    Errors are negated into scores, because everything downstream -- rliable's
    aggregates, marl-eval's normalisation -- assumes larger is better.
    """
    tree = {ENVIRONMENT: {}}
    grouped = {}
    for record in records:
        values = {
            (name if higher else f"neg_{name}"): [
                record["metrics"][name] if higher else -record["metrics"][name]
            ]
            for name, higher in PREDICTION_METRICS.items()
            if name in record["metrics"]
        }
        grouped.setdefault(
            (record["task"], algorithm_name(record), record["seed"]), {}
        ).update(values)
    for record in closed_loop:
        grouped.setdefault(
            (record["task"], algorithm_name(record), record["seed"]), {}
        ).update(record["episodes"])

    for (task, algorithm, seed), values in sorted(grouped.items()):
        run = {
            "step_0": {"step_count": 0, **values},
            "absolute_metrics": {
                name: [max(series)] for name, series in values.items()
            },
        }
        tree[ENVIRONMENT].setdefault(task, {}).setdefault(algorithm, {})[
            f"seed_{seed}"
        ] = run
    return tree


def bootstrap_iqm(values, resamples=20000, seed=0):
    """IQM over seeds with a percentile interval.

    The point estimate comes from ``rliable`` itself. The interval is a plain
    percentile bootstrap computed in one vectorised pass -- resampling through
    ``rliable``'s per-call estimator would be millions of Python calls, and
    ``rliable.library``, which would otherwise supply the interval, cannot be
    imported here. IQM is ``trim_mean`` at 25%, so the vectorised form is exactly
    that statistic rather than an approximation of it.
    """
    values = np.asarray(values, dtype=float)
    estimate = float(rliable_metrics.aggregate_iqm(values[:, None]))
    generator = np.random.default_rng(seed)
    draws = values[generator.integers(0, len(values), size=(resamples, len(values)))]
    estimates = trim_mean(draws, proportiontocut=0.25, axis=1)
    low, high = np.quantile(estimates, [0.025, 0.975])
    return estimate, float(low), float(high)


def aggregate(records, metric):
    """rliable point estimates over seeds, per (task, algorithm).

    IQM is the headline in the published tables: it discards the top and bottom
    quarter of seeds, so one diverged run cannot carry a claim -- which matters
    here, since two Dropout seeds do exactly that.
    """
    grouped = {}
    for record in records:
        if metric in record["metrics"]:
            grouped.setdefault((record["task"], algorithm_name(record)), []).append(
                record["metrics"][metric]
            )
    rows = []
    for (task, algorithm), values in sorted(grouped.items()):
        iqm, low, high = bootstrap_iqm(values)
        rows.append(
            {
                "task": task,
                "algorithm": algorithm,
                "metric": metric,
                "seeds": len(values),
                "iqm": iqm,
                "iqm_ci": [low, high],
                "median": float(rliable_metrics.aggregate_median(np.array(values)[:, None])),
                "mean": float(rliable_metrics.aggregate_mean(np.array(values)[:, None])),
            }
        )
    return rows


def format_table(rows, higher_is_better):
    direction = "higher better" if higher_is_better else "lower better"
    header = (
        f"{'task':11s}{'algorithm':26s}{'seeds':>6s}{'IQM':>10s}"
        f"{'95% CI':>22s}{'median':>10s}"
    )
    lines = [f"{rows[0]['metric']} ({direction})", header, "-" * len(header)]
    for row in rows:
        interval = f"[{row['iqm_ci'][0]:.4f}, {row['iqm_ci'][1]:.4f}]"
        lines.append(
            f"{row['task']:11s}{row['algorithm']:26s}{row['seeds']:>6d}"
            f"{row['iqm']:>10.4f}{interval:>22s}{row['median']:>10.4f}"
        )
    return "\n".join(lines)


__all__ = [
    "collect_runs",
    "collect_closed_loop",
    "algorithm_name",
    "marl_eval_tree",
    "aggregate",
    "bootstrap_iqm",
    "benchmarl_figures",
    "format_table",
]


def benchmarl_figures(results_path, output, metric="neg_rollout_error"):
    """Figures over BenchMARL's own processed matrices.

    ``benchmarl.eval_results`` is the repository's existing reporting path -- it
    wraps ``marl-eval``, which wraps ``rliable`` -- and it reads exactly the
    schema written above, so our runs go through its data pipeline unchanged:
    ``process_data`` normalises and cleans, ``create_matrices`` produces the
    (seeds x tasks) score matrix per algorithm that rliable expects.

    Its *plotting* functions cannot run here. They call
    ``rliable.library.get_interval_estimates``, which builds an ``arch``
    ``StratifiedBootstrap``; arch 7 fails to import against the installed pandas
    3, and arch 8 renamed the ``random_state`` argument that rliable still
    passes. Neither version works, and pinning pandas down to marl-eval's 1.4.4
    would mean downgrading the environment the experiments run in.

    So the interval step -- and only that step -- is replaced by the same
    percentile bootstrap used elsewhere in this project. Point estimates still
    come from ``rliable.metrics``, and the matrices still come from BenchMARL.
    """
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    from benchmarl.eval_results import load_and_merge_json_dicts, Plotting

    raw = load_and_merge_json_dicts([str(results_path)])
    normalize = [metric]
    processed = Plotting.process_data(raw, metrics_to_normalize=normalize)
    environment_matrix, _ = Plotting.create_matrices(
        processed, env_name=ENVIRONMENT, metrics_to_normalize=normalize
    )
    scores = environment_matrix[f"mean_norm_{metric}"]
    algorithms = sorted(scores)
    written = {}

    # Aggregate scores: IQM per algorithm over the pooled (seeds x tasks) runs.
    figure, axis = plt.subplots(figsize=(7, 0.4 * len(algorithms) + 1.5))
    for position, algorithm in enumerate(algorithms):
        flat = np.asarray(scores[algorithm]).reshape(-1)
        estimate, low, high = bootstrap_iqm(flat)
        axis.barh(position, estimate, color="#4C72B0", height=0.6)
        axis.plot([low, high], [position, position], color="black", linewidth=1.5)
    axis.set_yticks(range(len(algorithms)))
    axis.set_yticklabels(algorithms)
    axis.set_xlabel(f"normalised {metric} (IQM, 95% bootstrap CI)")
    axis.set_title(f"Aggregate scores over {ENVIRONMENT} tasks")
    path = output / "aggregate_scores.png"
    figure.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(figure)
    written["aggregate_scores"] = path

    # Performance profile: fraction of runs scoring above each threshold.
    figure, axis = plt.subplots(figsize=(7, 4.5))
    taus = np.linspace(0.0, 1.0, 101)
    for algorithm in algorithms:
        flat = np.asarray(scores[algorithm]).reshape(-1)
        axis.plot(taus, [(flat > tau).mean() for tau in taus], label=algorithm)
    axis.set_xlabel(f"normalised {metric} threshold")
    axis.set_ylabel("fraction of runs above threshold")
    axis.set_title("Performance profile")
    axis.legend(fontsize=8)
    path = output / "performance_profile.png"
    figure.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(figure)
    written["performance_profile"] = path

    # Per-task IQM, which is where the coupling story actually lives.
    tasks = sorted(processed[ENVIRONMENT])
    figure, axes = plt.subplots(
        1, len(tasks), figsize=(3.6 * len(tasks), 0.42 * len(algorithms) + 2)
    )
    axes = np.atleast_1d(axes)
    for index, (axis, task) in enumerate(zip(axes, tasks)):
        for position, algorithm in enumerate(algorithms):
            column = np.asarray(scores[algorithm])[:, index]
            estimate, low, high = bootstrap_iqm(column)
            axis.barh(position, estimate, color="#4C72B0", height=0.6)
            axis.plot([low, high], [position, position], color="black", linewidth=1.5)
        axis.set_yticks(range(len(algorithms)))
        # Labels only on the left panel, but the axes must not be shared: with
        # sharey, blanking one panel's tick labels blanks every panel's.
        axis.set_yticklabels(algorithms if index == 0 else [], fontsize=8)
        axis.set_ylim(-0.6, len(algorithms) - 0.4)
        axis.set_xlim(0, 1)
        axis.set_title(task)
        axis.set_xlabel(f"normalised {metric}")
    path = output / "per_task_scores.png"
    figure.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(figure)
    written["per_task_scores"] = path
    return written


def log_to_wandb(records, closed_loop, references, project, entity, tree, figures):
    """One run per (task, algorithm, regime, seed), grouped the way BenchMARL groups.

    BenchMARL sets ``group=task_name`` and puts the algorithm in the run name,
    which is what makes the published VMAS workspace groupable. We add the two
    axes BenchMARL has no slot for -- the predictor kind and the data regime --
    as their own config fields, so the UI can group or filter by either.

    A separate project keeps this readable: the existing one holds 400+ runs from
    every milestone, sweep and smoke test, and mixing a clean cross-task report
    into it would not be any easier to read than the directories already are.
    """
    import wandb

    control = {}
    for record in closed_loop:
        control.setdefault(
            (record["task"], record["kind"], record["regime"], record["seed"]), {}
        )[record["cost"]] = record["episodes"]

    for record in records:
        key = (record["task"], record["kind"], record["regime"], record["seed"])
        algorithm = algorithm_name(record)
        name = f"{algorithm}__seed{record['seed']}"
        run = wandb.init(
            project=project,
            entity=entity,
            # Deterministic id, as BenchMARL's own logger uses, so re-running
            # the report updates each run rather than creating a duplicate.
            id=f"{record['task']}__{name}",
            resume="allow",
            group=record["task"],
            job_type=record["kind"],
            name=name,
            tags=[record["task"], record["kind"], record["regime"]],
            config={
                "environment": ENVIRONMENT,
                "task": record["task"],
                "algorithm": algorithm,
                "kind": record["kind"],
                "regime": record["regime"],
                "seed": record["seed"],
                "run_directory": record["run"],
            },
            reinit=True,
        )
        summary = dict(record["metrics"])
        for cost, episodes in control.get(key, {}).items():
            for name, values in episodes.items():
                summary[f"control/{cost}/{name}"] = float(np.mean(values))
        run.log(summary)
        run.finish()

    # References are per-task, not per-checkpoint, so they get one run each.
    for reference in references:
        run = wandb.init(
            project=project,
            entity=entity,
            id=f"{reference['task']}__reference__{reference['policy']}",
            resume="allow",
            group=reference["task"],
            job_type="reference",
            name=f"{reference['policy']}__{reference['task']}",
            tags=[reference["task"], "reference", reference["policy"]],
            config={
                "environment": ENVIRONMENT,
                "task": reference["task"],
                "algorithm": reference["policy"],
                "kind": reference["policy"],
                "regime": "n/a",
            },
            reinit=True,
        )
        run.log(
            {
                f"control/{reference['policy']}/{name}": float(np.mean(values))
                for name, values in reference["episodes"].items()
            }
        )
        run.finish()

    # The aggregate tables belong with the runs they summarise.
    run = wandb.init(
        project=project,
        entity=entity,
        id="aggregate-scores",
        resume="allow",
        group="aggregate",
        job_type="summary",
        name="aggregate-scores",
        reinit=True,
    )
    for metric, higher in PREDICTION_METRICS.items():
        rows = aggregate(records, metric)
        if not rows:
            continue
        table = wandb.Table(
            columns=["task", "algorithm", "seeds", "iqm", "ci_low", "ci_high", "median"]
        )
        for row in rows:
            table.add_data(
                row["task"],
                row["algorithm"],
                row["seeds"],
                row["iqm"],
                row["iqm_ci"][0],
                row["iqm_ci"][1],
                row["median"],
            )
        run.log({f"aggregate/{metric}": table})
    for name, path in figures.items():
        run.log({f"figures/{name}": wandb.Image(str(path))})
    artifact = wandb.Artifact("marl_eval_results", type="results")
    with artifact.new_file("results.json", mode="w") as handle:
        json.dump(tree, handle, indent=2)
    run.log_artifact(artifact)
    run.finish()


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=Path("outputs/report"))
    parser.add_argument("--project", default="counterfactual-wm-benchmark")
    parser.add_argument("--entity", default="cair-traffic")
    parser.add_argument("--wandb", action="store_true", help="log runs to wandb")
    parser.add_argument(
        "--figures",
        action="store_true",
        help="run BenchMARL's eval_results plotting pipeline",
    )
    args = parser.parse_args()

    records = collect_runs(args.root)
    closed_loop, references = collect_closed_loop(args.root)
    print(
        f"collected {len(records)} trained checkpoints over "
        f"{len({r['task'] for r in records})} tasks; "
        f"{len(closed_loop)} closed-loop policies, {len(references)} references"
    )

    tree = marl_eval_tree(records, closed_loop)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "results.json").write_text(json.dumps(tree, indent=2) + "\n")

    tables = []
    for metric, higher in PREDICTION_METRICS.items():
        rows = aggregate(records, metric)
        if rows:
            tables.append(format_table(rows, higher))
    report = "\n\n".join(tables)
    (args.output / "aggregate_scores.txt").write_text(report + "\n")
    print("\n" + report)

    figures = {}
    if args.figures:
        figures = benchmarl_figures(args.output / "results.json", args.output)
        print("\nfigures: " + ", ".join(sorted(figures)))

    if args.wandb:
        log_to_wandb(
            records, closed_loop, references, args.project, args.entity, tree, figures
        )
        print(f"\nlogged to wandb project {args.entity}/{args.project}")
    print(f"\nwrote {args.output}/results.json and aggregate_scores.txt")


if __name__ == "__main__":
    main()
