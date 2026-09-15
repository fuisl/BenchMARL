# Logging and aggregate reporting

2026-09-15. How world-model runs reach wandb and the aggregate tables, and why
that is almost entirely BenchMARL's code rather than ours.

## The mistake this replaces

`train.py` called `get_logger` directly with `experiment_name =
"m4_{kind}_{regime}"` and `group = "m4-transport-baselines"`. That name carries
no task and no seed, no wandb id was passed, and no marl-eval file was written.
In `counterfactual-wm` the result was 400 runs sharing **14 distinct names**,
with `m4_independent_correlated` appearing 59 times, grouped by milestone
string. The repository's own runs in that same project -- `transport`,
`give_way`, `balance` -- group correctly by task, because they go through
`Logger`.

Everything downstream followed from that. A 538-line `report.py` existed to
reconstruct, after the fact, what the logger would have recorded: a
reimplementation of `JsonWriter`'s schema, a reimplementation of `Logger`'s
wandb conventions, and a deterministic id scheme that was only needed because
runs had no ids.

## What happens now

`train.py` uses BenchMARL's convention directly:

| | |
|---|---|
| run name | `{algorithm}_{task}_{model}` via `generate_exp_name` |
| wandb group | `task_name` |
| wandb id | the experiment name |
| marl-eval file | `JsonWriter`, one per run |

The three predictors are the **algorithm**: they are what is compared on a
shared task at matched capacity, the slot BenchMARL gives MAPPO and IPPO. The
data regime has no slot in that schema, so it joins the algorithm name
(`relational_correlated`) and is also a config field for filtering. Task and
environment are read from the bank's own manifest, not passed in.

```bash
# Aggregate any set of sweeps
.venv/bin/python -m examples.world_model.report \
    outputs/interaction_control_1194 outputs/buzz_wire_1196/baselines \
    outputs/wheel_1205/baselines
```

`report.py` is 112 lines: merge with `load_and_merge_json_dicts`, build matrices
with `Plotting`, print IQM. No GPU -- the numbers already exist in the run
directories.

## Two deliberate deviations

**Negated metrics are renamed.** The reporting stack assumes larger is better,
so errors are negated -- but a field called `rollout_error` holding `-0.93`
would mislead anyone reading the file without this note beside it. They are
written as `neg_rollout_error`.

**The marl-eval file has a fixed name**, `marl_eval.json`.
`get_raw_dict_from_multirun_folder` walks for *every* json under a sweep, which
is correct for BenchMARL run folders because they hold exactly one; ours also
hold `metrics.json`, `parameters.json` and `provenance.json`, and merging those
raises `AttributeError`. `report.py` therefore names the files and passes them
to the explicit `load_and_merge_json_dicts`, which is the same BenchMARL API one
level down.

## What the published tooling cannot do here

`Plotting`'s figures call `rliable.library.get_interval_estimates`, which builds
an `arch` `StratifiedBootstrap`. `arch` 7.2 fails to import against the
installed pandas 3.0, and `arch` 8.0 renamed the `random_state` argument that
`rliable` still passes. Neither works, and `id-marl-eval` pins pandas 1.4.4,
which would mean downgrading the environment the experiments run in. `arch` was
upgraded to 8.0 because that is what lets BenchMARL's data pipeline import at
all; nothing else depends on it.

Only the interval is substituted, by a percentile bootstrap. Point estimates are
`rliable.metrics`'.

## Migration

The 192 runs from jobs 1194, 1196 and 1205 predate this and were re-logged from
their `metrics.json` and resolved configs: a `marl_eval.json` written into each
run directory, and a wandb run under the new convention. They carry deterministic
ids derived from the source job and run index rather than a uuid, so the
migration is idempotent. `counterfactual-wm` now holds 192 runs grouped 48 per
task, 32 per algorithm, beside the older `m4_*` runs describing the same
experiments.

## What was audited and left alone

`oracle_comparison/` was flagged as config sprawl and is not. Config groups are
BenchMARL's own pattern -- `algorithm/`, `experiment/`, `model/`, `task/` are all
groups -- and all nine presets are referenced by the sweeps that produced jobs
1182 and 1183, whose results are recorded. Collapsing them would cost
reproducibility and buy nothing.

The real sprawl was `wandb.group` in nine submission scripts, which existed only
to compensate for the group not being the task. Those are gone. What remains in
those scripts is mostly Hydra output plumbing (`hydra.run.dir`,
`hydra.sweep.dir`, `launcher.n_jobs`) that every submission needs.
