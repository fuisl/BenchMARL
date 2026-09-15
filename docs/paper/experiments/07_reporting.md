# Aggregate reporting

2026-09-15. How every run in this project is turned into one comparable report,
and where the published-benchmark tooling does and does not apply.

## Problem

Results lived as 192 per-run `metrics.json` files under four sweep directories,
plus closed-loop summaries, and reached wandb in groups named after the milestone
that produced them (`m4-buzz-wire`, `m3-transport-data`, ...). With 400+ runs in
one project, nobody could group by algorithm and task the way the published VMAS
benchmark does, so cross-task comparison meant reading directories.

## What `examples/world_model/report.py` produces

| Output | Content |
|---|---|
| `results.json` | marl-eval schema, identical to BenchMARL's own `JsonWriter` |
| `aggregate_scores.txt` | IQM, median and mean over seeds, with bootstrap intervals |
| `aggregate_scores.png` | IQM per algorithm, pooled over tasks |
| `performance_profile.png` | fraction of runs above each normalised score |
| `per_task_scores.png` | IQM per algorithm within each task |
| wandb | one run per (task, algorithm, regime, seed), plus an aggregate run |

```bash
.venv/bin/python -m examples.world_model.report --output outputs/report \
    --figures --wandb
```

No GPU: every number already exists in the run directories, so this reads and
aggregates on CPU.

## Mapping our comparison onto BenchMARL's schema

BenchMARL's schema is `{environment: {task: {algorithm: {seed_i: ...}}}}`. Our
three predictors take the slot it gives algorithms -- independent, joint and
relational are what is compared on a shared task at matched capacity, exactly as
MAPPO and IPPO are there. The **data regime** is a second axis that schema has no
room for, so it is folded into the algorithm name (`relational-correlated`) and
also kept as its own wandb field, where it can be grouped or filtered
independently.

Errors are negated into scores (`neg_rollout_error`) because everything
downstream -- rliable's aggregates, marl-eval's normalisation -- assumes larger
is better. Metrics are lists per evaluation episode: single-element for a
per-run dynamics metric, genuinely per-episode for closed-loop control.

wandb organisation follows BenchMARL's own logger, which sets `group=task_name`
and a deterministic run id. Runs go to `counterfactual-wm-benchmark` rather than
the existing project, which holds every milestone, sweep and smoke test; ids are
deterministic, so re-running the report updates runs instead of duplicating them.

## What the published tooling can and cannot do here

`benchmarl.eval_results` wraps `marl-eval`, which wraps `rliable`. Its **data
pipeline runs on our results unchanged**: `process_data` normalises and cleans,
`create_matrices` yields the (8 seeds x 4 tasks) matrices rliable expects.

Its **plotting cannot run in this environment**. Those functions call
`rliable.library.get_interval_estimates`, which constructs an `arch`
`StratifiedBootstrap`:

* `arch` 7.2 fails to import against the installed pandas 3.0 --
  `deprecate_kwarg` changed signature;
* `arch` 8.0 imports, but rejects the `random_state` argument `rliable` still
  passes, having renamed it.

Neither version works, and `id-marl-eval` pins pandas 1.4.4, which would mean
downgrading the environment the experiments run in -- including a venv with jobs
executing in it. Upgrading `arch` to 8.0 was the one change made, because it is
what lets BenchMARL's data pipeline import at all; nothing else depends on it.

So only the interval step is substituted, by the percentile bootstrap used
elsewhere in this project. Point estimates remain `rliable.metrics`'.

## A result that depends on which statistic is used

Job 1194's Dropout control was reported from paired seed means: relational
*worse* than independent (-9% and -28%, better on 3/8 seeds). By IQM, the
statistic the published tables use, relational-correlated is **better**
(0.0772 against 0.0888). IQM trims the top and bottom quarter of seeds, and
Dropout has two runs (4101, 4106) that diverge badly for every predictor.

The intervals overlap heavily, so neither direction is established -- but the
Dropout control reads differently under the two statistics, and the claim it
supports should say which one it rests on.
