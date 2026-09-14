# M2 — Transport comparison batch

2026-09-14: user requested one GPU and a batch of comparisons, explicitly switching
from Buzz Wire to Transport. Buzz Wire runs are paused; its results are retained.
The user subsequently authorized this machine, a full A100, and 18 runs packed
onto that one GPU with wandb logging and OOM monitoring. This explicitly overrides
the usual H100/MIG routing for this batch. Slurm job **1181** was submitted but
stayed pending for the full A100 the whole window (held by another user's job 1180)
and was cancelled. The batch was resubmitted as **job 1182** on the 20 GB MIG slice
(`gpu:a100_3g.20gb:1`) with `TRANSPORT_WORKERS=4`, `--mem=32G`. Job 1182 **completed**
in 10m41s (06:33:03–06:43:44 UTC), all 18 comparisons produced, no OOM, peak sampled
GPU memory 1331 MiB. Results and their status are below. A follow-up escalation
(job **1183**, same MIG allocation, longer horizon/budget/episode length) ran
afterward and is recorded in its own section below; the matrix, semantics, and
submission mechanics that follow this section are otherwise unchanged.

## Result (job 1182)

Every comparison timed out on every episode — **0/18 configurations reached any
collision-free package success** (`success_rate = 0.0`, `timeout_rate = 1.0` for all
60 seed-runs). `collision_rate = 0.0` throughout, as expected (Transport has no
collision-ending failure mode). Only `replan1` seed 0 clears the return-vs-random-CI
comparison; no configuration clears it consistently across seeds.

| Comparison | mean return (3 seeds) | success | mean seconds (3 seeds) |
|---|---:|---:|---:|
| lewm | 0.259 | 0/60 | 88.7 |
| iterations10 | 0.179 | 0/60 | 31.9 |
| replan1 | 0.515 | 0/60 | 461.6 |
| horizon1 | 0.000 | 0/60 | 75.9 |
| samples100 | 0.132 | 0/60 | 76.0 |
| samples100_iterations10 | 0.123 | 0/60 | 29.4 |

Per-episode detail (e.g. `comparisons/6/episodes.csv`) shows most `final_goal_distance`
values sitting at 0.3–1.8 at episode end (package starts far from goal and is pushed
partway, sometimes not at all — several episodes are byte-identical to the random
baseline, meaning agents never made contact with the package inside 100 steps).
**This fails M2's "Done when" gate for Transport** (return exceeds random with
non-overlapping CIs *and* a reportable collision-free success rate) even though it
passed for Buzz Wire. Whether this is genuine task difficulty at these budgets or a
pipeline/objective problem is not yet determined — the same "task hard vs. pipeline
wrong" ambiguity already flagged for Buzz Wire in
[01_protocol.md](01_protocol.md) applies here and should be resolved before treating
Transport oracle MPC as validated or before spending more compute on this grid.

**Runtime check.** The ~15x spread in per-run wall time (29s to 465s) was checked
against a bug — every run's wandb-logged `_runtime` was pulled and compared to the
`seconds` field in `comparison.csv` and to the per-decision breakdown in each run's
`timing.json`; all three agree (wandb runtime consistently ~0–11% above the CSV
figure, the gap being init/logging overhead) and every episode ran the full 100
steps (`length = 100`). The spread is fully explained by the grid's own settings:
`lewm`/`iterations10`/`samples100`/`samples100_iterations10` replan every 5 blocks
(4 replans across 100 steps), `replan1`/`horizon1` replan every block (20 replans);
`replan1` additionally simulates the full H=5-block horizon per replan while
`horizon1` simulates only H=1, so `replan1` costs ≈5× `horizon1` and ≈5× the lewm
family — matching the observed 461.6s vs. 75.9s vs. 88.7s almost exactly. No run
silently skipped work or returned early; the fast runs (~30s, `samples100_iterations10`,
`iterations10`) are simply the cheapest corner of the K/R budget grid by design.

## Coverage analysis and budget escalation (job 1183, 2026-09-14)

**Diagnosis before spending more compute.** Cross-referencing every job-1182
episode's return against the 20 development states' initial geometry (extracted
from `initial_states.pt`) shows package *contact* inside the CEM lookahead, not
search quality, is the dominant bottleneck: across every one of the 18 original
configurations, only states {0, 6, 12, 18} — the states where an agent starts
within ~0.3 units of the package — ever produced nonzero return; the other 16
states showed exactly zero package movement for the full 100 steps, every
config, every seed. Mean agent-to-package distance across the 20 states is 0.61,
roughly 2× that contact radius, so at horizon=5 (25-step lookahead) most states
never reach the package before the rollout ends and CEM has no cost gradient to
climb (all candidates score identically until contact happens).

This motivated a follow-up escalation, run as Slurm job **1183** (same MIG
allocation, `--time=03:00:00`, completed in 1h43m, no OOM) as two sweeps, both
reusing state-seed 0 so all 20 initial positions are identical to job 1182's:

- **`horizon_budget`** (100-step episode, reuses job 1182's exact state-bank
  file): `horizon10` (H=10 blocks, i.e. double the lookahead, same K=300/R=30),
  `bigbudget` (K=600, R=45, same H=5), `horizon10_bigbudget` (both combined).
- **`longepisode`** (300-step episode, 3×, matching VMAS's own Passage default;
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

**Horizon dominates raw budget, exactly as the contact diagnosis predicted:**
`horizon10` (2× lookahead) roughly triples `lewm`'s original return (0.259→0.722)
and clears the return-vs-random gate on 2/3 seeds; `bigbudget` (2× samples, 1.5×
iterations, same horizon) barely beats `lewm` and clears the gate on 0/3 —
confirming more search *within the same 25-step lookahead* has little to offer
when most states can't reach the package inside that lookahead regardless of how
well it's searched. Combining both (`horizon10_bigbudget`) tracks `horizon10`
alone, not a multiplicative gain — budget is not the limiting factor.

**Episode length matters even more.** At 300 steps, `replan1` and `horizon10`
both clear the return gate on **all 3 seeds**, with mean returns 5–14× job
1182's original numbers. Episode-level detail also shows the fix is partly
generalizing, not just deepening progress on the same 4 states: `horizon10` at
300 steps shows nonzero return on 6–8/20 states per seed (planner-seed
dependent), a union of **10/20** states across its 3 seeds — {0, 1, 5, 6, 9, 11,
12, 16, 18, 19} — versus the fixed {0, 6, 12, 18} that was the *only* nonzero
set anywhere in job 1182.

**Success is still 0/60 across every run in this escalation, at every budget and
episode length tested.** Even the best configuration (`horizon10` at 300 steps)
never gets one package fully onto its goal. Inspecting states that remain stuck
despite short agent-package distance (e.g. state 7, min distance 0.43, `return`
still 0 in *every* run across both jobs) points to a second-order cause beyond
raw distance: the shaping reward only rewards package-to-goal distance, not
agent positioning, so an agent that starts on the *wrong side* of the package
relative to the goal gets zero cost signal for the "circle around to the far
side, then push" maneuver VMAS's own shipped `HeuristicPolicy` for Transport
implements explicitly (a hermite-spline "dribble" controller) — flat-signal CEM
has no gradient to discover that maneuver by chance. This is also consistent
with the original VMAS paper (arXiv:2207.03530): only fully-decentralized IPPO
learns Transport, requiring ~24M environment interactions (400 iterations ×
60,000 interactions/iteration); centralized methods (CPPO, MAPPO) fail outright,
attributed explicitly to the task's need for extensive exploration under high
joint-state variance. Oracle CEM-MPC is architecturally centralized and
zero-shot per episode (no learning carried across episodes), so it sits in
exactly the category the source paper reports failing here — the difficulty is
better explained by task/architecture mismatch than by a pipeline defect.

**Implication for using this method.** The return-based signal (not binary
success) is now a working oracle-vs-random reference for Transport — `horizon10`
at either episode length clears the statistical gate reliably. Two paths forward,
not yet chosen: (a) accept return/goal-distance-reduction, not binary success, as
the operative Transport metric for M2/M5 (Buzz Wire keeps its binary gate; the
two tasks would use different success definitions, which needs to be stated
explicitly rather than silently), or (b) push horizon/episode length further
still (500–1000 steps, H=15–20) to chase literal success, understanding the
wrong-side-approach failure mode identified above is a search-blindness problem
that more budget alone has not fixed at any scale tested so far.

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
only. GPU memory and utilization are sampled every five seconds in
`gpu_memory.csv`; worker errors, including CUDA OOMs, remain in `batch.log` and
Slurm output. No runtime/OOM conclusion is possible until the allocation starts.
Review individual seeds as well as the consolidated table; no pooled-seed
confidence interval is manufactured from duplicated initial states.

First submission: `sbatch --parsable scripts/slurm/transport_oracle.sbatch` →
**1181**, requesting the full A100 (18 workers). It stayed pending the whole
two-hour window behind another user's job 1180 and was cancelled by the user at
2026-09-14T06:32 UTC; `outputs/transport_oracle_1181/` holds only the watcher and
an empty `monitor_status.json` (0/18 completed) — no comparison data.

Resubmission: same command, now targeting the MIG slice → **1182**. Output root
`outputs/transport_oracle_1182/`; Slurm log `slurm-1182.out`. Job 1182
**completed** (`sacct`: COMPLETED, 06:33:03–06:43:44 UTC, elapsed 10m41s). Results
are in `outputs/transport_oracle_1182/comparisons/comparison.csv` and summarized
above.

A detached, read-only watcher ran from `outputs/transport_oracle_1182/watch_job.py`.
It polls Slurm and logs every 15 s, records transitions in `monitor.log`, and
updates `monitor_status.json` with completed-run count, peak sampled GPU memory,
and OOM detection — final state for 1182: `COMPLETED`, `oom_detected: false`,
`peak_sampled_gpu_memory_mib: 1331.0`, `completed_summaries: 18/18`. The watcher
exits after Slurm reports a terminal job state; it has already exited for 1182.
