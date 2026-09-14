# M2 — Transport comparison batch

2026-09-14: user requested one GPU and a batch of comparisons, explicitly switching
from Buzz Wire to Transport. Buzz Wire runs are paused; its results are retained.
The user subsequently authorized this machine, a full A100, and 18 runs packed
onto that one GPU with wandb logging and OOM monitoring. This explicitly overrides
the usual H100/MIG routing for this batch. Slurm job **1181** was submitted but
stayed pending for the full A100 the whole window (held by another user's job 1180)
and was cancelled. The batch was resubmitted as **job 1182** on the 20 GB MIG slice
(`gpu:a100_3g.20gb:1`) with `TRANSPORT_WORKERS=4`, `--mem=32G`. Job 1182 **completed**
in 10m41s (06:33:03–06:43:44 UTC), all 18 comparisons produced, with no reported
OOM. The recorded 1331 MiB memory peak is invalid for MIG: the monitor sampled
device `0` with 40960 MiB capacity, rather than the allocated 20 GB slice.
Results and their status are below. A follow-up escalation
(job **1183**, same MIG allocation, longer horizon/budget/episode length) ran
afterward and is recorded in its own section below; the matrix, semantics, and
submission mechanics that follow this section are otherwise unchanged.

## Result (job 1182)

Every comparison timed out on every episode: **0 successes in 360 MPC episode
evaluations**, comprising six settings × three planner seeds × the same 20
development states. Each setting has 0/60 successes across its three runs; these
are repeated evaluations of 20 states, not 60 independent initial states.
`collision_rate = 0.0` throughout, as expected (Transport has no collision-ending
failure mode). Only `replan1` seed 0 clears the return-vs-random-CI comparison;
no setting clears it consistently across seeds. Random return is exactly zero
on all 20 states, with zero successes. Each run's zero-success Wilson interval
is [0%, 16.1%]; no interval is pooled across the repeated state bank.

| Comparison | mean return (3 seeds) | success | mean seconds (3 seeds) |
|---|---:|---:|---:|
| lewm | 0.259 | 0/60 | 88.7 |
| iterations10 | 0.179 | 0/60 | 31.9 |
| replan1 | 0.515 | 0/60 | 461.6 |
| horizon1 | 0.000 | 0/60 | 75.9 |
| samples100 | 0.132 | 0/60 | 76.0 |
| samples100_iterations10 | 0.123 | 0/60 | 29.4 |

Per-episode detail (e.g. `comparisons/6/episodes.csv`) shows final goal distances
of approximately 0.28–1.81. Many episodes have exactly zero primitive rewards and
the same final goal distance as random. These artifacts establish absent
reward-relevant progress; they do not record every package position or contact,
so they do not by themselves prove that no contact or movement occurred.
**The return gate passes for only one run; task-success objective validity remains
unvalidated.** M2's written return gate and its separate success-rate reporting
requirement must remain distinct. No numerical success gate has yet been chosen,
and the lack of goals must not be hidden by changing the success definition.
Task difficulty, finite-horizon search, objective limitations, and untested
pipeline defects remain possible explanations; existing replay tests narrow the
last possibility without resolving the others.

**Runtime check.** The ~15x spread in per-run wall time (29s to 465s) was checked
against a bug — every run's wandb-logged `_runtime` was pulled and compared to the
`seconds` field in `comparison.csv` and to the per-decision breakdown in each run's
`timing.json`; all three agree (wandb runtime consistently ~0–11% above the CSV
figure, the gap being init/logging overhead) and every episode ran the full 100
steps (`length = 100`). The grid's settings explain the main work difference:
`lewm`/`iterations10`/`samples100`/`samples100_iterations10` replan every 5 blocks
(4 replans across 100 steps), `replan1`/`horizon1` replan every block (20 replans);
`replan1` additionally simulates the full H=5-block horizon per replan while
`horizon1` simulates only H=1. The observed 461.6s vs. 75.9s vs. 88.7s is
consistent with that work difference, although concurrent-worker contention and
fixed overhead prevent interpreting wall time as an isolated scaling law. Every
recorded episode ran its full budget; the fast runs (~30s, `samples100_iterations10`,
`iterations10`) are simply the cheapest corner of the K/R budget grid by design.

## Coverage analysis and budget escalation (job 1183, 2026-09-14)

**Coverage evidence and working hypothesis.** Across all 18 job-1182 runs, only
states {0, 6, 12, 18} ever produced nonzero return. The stored primitive rewards
are exactly zero at every step for the other 16 states. The mean distance from
the package center to its nearest agent center is 0.613; the four responsive
states have distances 0.303, 0.277, 0.289, and 0.334. Distance alone does not
explain the result: state 5 starts at distance 0.291 but has zero reward in every
1182 run. These distances are not physical contact thresholds.

This supports investigating poor coverage of useful contact and approach
behaviors. It does not prove a dominant cause: trajectories do not store contact
history, and the saved candidate bank covers the initial solve, not every later
decision. CEM uses sampled cost comparisons, not gradients; flat candidate costs
can impede its search, but the claim that all candidates tie at every unproductive
state has not been established. M3 should measure actual cross-agent effects and
contact-related coverage instead of assuming diverse actions provide it.

This motivated a follow-up escalation, run as Slurm job **1183** (same MIG
allocation, `--time=03:00:00`, completed in 1h43m15s with exit code 0 and no
reported OOM) as two sweeps, both
reusing state-seed 0 so all 20 initial positions are identical to job 1182's:

- **`horizon_budget`** (100-step episode, reuses job 1182's exact state-bank
  file): `horizon10` (H=10 blocks, same K=300/R=30), `bigbudget` (K=600,
  R=45, H=5), `horizon10_bigbudget` (both combined). **All execute one block
  between replans**, so `replan1`, not `lewm`, is their controlled H=5 reference.
- **`longepisode`** (300-step episode, 3× Transport's configured default;
  each run regenerates its own state bank from state-seed 0, since a 300-step
  bank cannot equal a 100-step one under `evaluate.py`'s task-config check —
  verified bit-identical positions, different `state_bank_sha256` as expected):
  `replan1` and `horizon10` at the 300-step episode length.

| Sweep | Comparison | mean return (3 seeds) | return gate | success | mean seconds |
|---|---|---:|---:|---:|---:|
| horizon_budget (100 steps) | horizon10 | 0.722 | 2/3 | 0/60 | 1014 |
| horizon_budget (100 steps) | bigbudget | 0.464 | 0/3 | 0/60 | 700 |
| horizon_budget (100 steps) | horizon10_bigbudget | 0.698 | 2/3 | 0/60 | 1215 |
| longepisode (300 steps) | replan1 | 2.791 | 3/3 | 0/60 | 1555 |
| longepisode (300 steps) | horizon10 | 3.661 | 3/3 | 0/60 | 2339 |

At fixed execution cadence, `horizon10` raises mean return from `replan1`'s
0.515 to 0.722 (about 1.40×), with the return gate passing for 2/3 seeds instead
of 1/3. Comparing 0.722 to `lewm`'s 0.259 also changes execution cadence and
cannot isolate lookahead. `bigbudget` averages 0.464, below `replan1`'s 0.515;
`horizon10_bigbudget` averages 0.698, near `horizon10`'s 0.722. These are descriptive
development-bank results, not proof that horizon dominates search budget.
Final-elite-mean selection and closed-loop return have no monotonicity guarantee,
and no uncertainty estimate for the across-seed configuration difference is
reported here. The contact-coverage hypothesis remains to be tested directly.

At 300 steps, `replan1` and `horizon10` both clear the return gate on **all three
seeds**. Their means increase from 0.515 to 2.791 (5.42×) and from 0.722 to
3.661 (5.07×), respectively, compared with the same solver at 100 steps.
The longer episode also grants three times as many control steps and replans;
these are not equal-budget performance comparisons. The 300-step random
reference averages 0.0165, with episode-bootstrap CI [0, 0.0419] and no goals.
Episode-level detail shows progress on additional development states: `horizon10` at
300 steps shows nonzero return on 6–8/20 states per seed (planner-seed
dependent), a union of **10/20** states across its 3 seeds — {0, 1, 5, 6, 9, 11,
12, 16, 18, 19} — versus the fixed {0, 6, 12, 18} that was the *only* nonzero
set anywhere in job 1182.

**The escalation records 0 successes in 300 MPC episode evaluations:** nine
100-step runs (180 episodes) and six 300-step runs (120 episodes). Each of its
five settings has 0/60 successes on three repetitions of the same 20 initial
states. Together, jobs 1182 and 1183 contain 0/660 successful MPC episode
evaluations; they do not provide 660 independent evaluation states.

Approach geometry is another plausible explanation: state 7 starts at nearest
agent distance 0.428 yet has zero return across both jobs, and the task's shaping
reward rewards package-to-goal progress rather than preparatory agent positioning.
The installed VMAS heuristic explicitly approaches and pushes the package, which
suggests a useful coverage control. Neither this observation nor published PPO
results establishes the cause of CEM failure. A centralized critic/policy in PPO
and centralized action search are different algorithms; grouping them together
does not diagnose an architectural failure.

**Implication for M3/M4.** Transport provides a finite-budget oracle reference
with measurable progress and no demonstrated goal attainment. Keep its return,
goal-overlap success, and objective status separate; every saved summary correctly
marks the objective `unvalidated: no task successes`. Preserve the objective
`J = -sum_t sum_i r_i,t` and the actual success definition for learned-model
comparisons. M3 can proceed with controlled data and explicit interaction-coverage
diagnostics; a learned model cannot be assumed to infer missing interactions or
beat the true dynamics at matched search solely through better understanding.
Any improvement over this finite-budget CEM implementation needs attribution to
search, horizon, objective, or approximation effects, followed by simulator replay
and held-out confirmation. M2's development states must remain outside M3's
training and held-out datasets.

## Question and controlled comparisons

Can task-reward oracle MPC solve the proposal's Transport example, and how do
search budget, lookahead and execution cadence affect success and return?
Keep VMAS task defaults: four agents, one package of mass 50, and 100 primitive
steps per episode. Keep five primitive actions per planning block. All runs use
one saved 20-episode development state bank (state seed 0), random-policy seed 1,
and bootstrap seed 2. Repeat each configuration with planner seeds 0, 1, 2.
These repetitions do not make 60 independent initial evaluation states.

| Comparison | K | R | Elites | H (blocks) | Execute (blocks) |
|---|---:|---:|---:|---:|---:|
| lewm | 300 | 30 | 30 | 5 | 5 |
| iterations10 | 300 | 10 | 30 | 5 | 5 |
| replan1 | 300 | 30 | 30 | 5 | 1 |
| horizon1 | 300 | 30 | 30 | 1 | 1 |
| samples100 | 100 | 30 | 10 | 5 | 5 |
| samples100_iterations10 | 100 | 10 | 10 | 5 | 5 |

This gives 18 evaluations, each with a paired random reference. Compare `lewm`
with `replan1` for cadence and `replan1` with `horizon1` for lookahead. The other
four settings complete the K/R budget grid at a fixed 10% elite fraction.
No setting is selected automatically, and held-out confirmation follows the
results rather than reusing this development bank as held-out evidence.

## Transport semantics and validation

- Success follows the installed VMAS scenario: **all packages overlap their
  goal**, using `scenario.done()` and its per-package `on_goal` flags. A successful
  terminal transition at step 100 counts as success, otherwise the limit is a
  timeout. Physical contacts are not episode-ending failures: `collision_rate`
  in this shared report means terminal collision failure and is always zero for
  Transport, not a count of agent/package contacts.
- `final_goal_distance` is the maximum package-center-to-goal-center distance.
  Mean-agent return matches BenchMARL; summed team return is four times that
  value and is the quantity used by `J = -sum_t sum_i r_i,t`.
- Package `global_shaping`, `on_goal`, dynamics state and episode clocks are
  snapshotted. New tests exercise moving-package rewards, same-width exact replay,
  serial/batched scoring, live-execution agreement, all-package success semantics,
  and timeout classification with one and two packages.
- All 12 focused test cases pass, including retained Buzz Wire regressions. CPU
  end-to-end Transport smoke passes. GPU smoke and a full-width R=1 timing run
  are the first steps inside the requested allocation; failure stops the batch.
- The tiny reset-state smoke can return zero for every action because agents have
  not reached the package. That is not a success result. Actual evaluation retains
  task defaults and records zero/negative results without fallback or retuning.

## Submission and artifacts

The six Hydra presets live in `benchmarl/conf/oracle_comparison/`; the matrix is
`benchmarl/conf/sweep/transport_oracle.yaml`. Submit from the repository root:

```bash
sbatch scripts/slurm/transport_oracle.sbatch
```

The script originally requested one full local `gpu:a100:1` allocation (40 GB) with
18 concurrent workers; that allocation never became available (see job 1181 below),
so the script now requests the 20 GB MIG slice `gpu:a100_3g.20gb:1`, 8 CPUs, 32 GB
host RAM, a two-hour limit, and **4 concurrent workers** (`TRANSPORT_WORKERS`,
default 4), with OMP/MKL limited to one thread per worker. Each worker also batches
B*K simulator slots internally. `TRANSPORT_WORKERS` and the `--gres`/`--mem` lines
can be raised again for a future submission if a full A100 becomes available and
throughput measurements warrant it. On another cluster, override
partition/GRES/account/QoS according to that site.

The batch validates GPU execution, measures R=1 throughput, prepares the shared
GPU state bank, runs all 18 comparisons via Hydra's existing Joblib launcher,
and writes a consolidated `comparison.csv`. The summarizer verifies matching
state-bank identities and exact random-reference episode rows across runs.

Outputs are `outputs/transport_oracle_JOBID/`, including `batch.log`, `smoke/`,
`timing/`, `initial_states.pt`, and `comparisons/0..17/`. Each real comparison
retains resolved config, source copies/hashes, package versions, per-episode
CSV, return/success intervals, timing, candidate bank and executed trajectories.
Real runs log to CSV and wandb group `m2-transport-oracle`; preflight runs use CSV
only. GPU memory sampling was attempted every five seconds in `gpu_memory.csv`;
the historical 1182 and 1183 files contain numeric device `0` with 40960 MiB total
memory throughout, so neither establishes memory usage of the allocated MIG
slice. Resolve the actual allocated device UUID before using this monitor for a
future capacity decision. Worker errors, including CUDA OOMs, remain in
`batch.log` and Slurm output; both jobs completed with exit code 0 and no logged OOM.
Review individual seeds as well as the consolidated table; no pooled-seed
confidence interval is manufactured from duplicated initial states.

First submission: `sbatch --parsable scripts/slurm/transport_oracle.sbatch` →
**1181**, requesting the full A100 (18 workers). It stayed pending behind another
user's job 1180 from submission at 06:28:20 to cancellation at 06:31:59 UTC
on 2026-09-14 (`sacct`); its requested wall-time limit was two hours, not its
queue duration. `outputs/transport_oracle_1181/` holds the watcher and a
`monitor_status.json` reporting cancellation and 0/18 completed — no comparison data.

Resubmission: same command, now targeting the MIG slice → **1182**. Output root
`outputs/transport_oracle_1182/`; Slurm log `slurm-1182.out`. Job 1182
**completed** (`sacct`: COMPLETED, 06:33:03–06:43:44 UTC, elapsed 10m41s). Results
are in `outputs/transport_oracle_1182/comparisons/comparison.csv` and summarized
above.

A detached, read-only watcher ran from `outputs/transport_oracle_1182/watch_job.py`.
It polls Slurm and logs every 15 s, records transitions in `monitor.log`, and
updates `monitor_status.json` with completed-run count, peak sampled GPU memory,
and OOM detection — final state for 1182: `COMPLETED`, `oom_detected: false`,
`peak_sampled_gpu_memory_mib: 1331.0`, `completed_summaries: 18/18`. The peak field
inherits the invalid device selection described above and is retained only as
historical output. The watcher exits after Slurm reports a terminal job state;
it has already exited for 1182.

Audit for M3 (2026-09-14): counts, returns, gates, episode outcomes, and coverage
sets above were checked against all 33 real-run `summary.json`/`episodes.csv`
artifacts and both jobs' `comparison.csv` files. Primitive reward coverage was
checked against all 18 job-1182 `mpc_trajectory.pt` files. All entity and scenario
snapshot tensors in the 100-step bank and the first 300-step bank are bitwise
identical; the manifests differ because their task episode limits differ.
