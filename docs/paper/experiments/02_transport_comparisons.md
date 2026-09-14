# M2 — Transport comparison batch

2026-09-14: user requested one GPU and a batch of comparisons, explicitly switching
from Buzz Wire to Transport. Buzz Wire runs are paused; its results are retained.
The Transport adapter, batch configuration, and submission script are prepared.
Submission location is awaiting the user's choice because only the local A100 is
accessible here and coding rule 13 normally routes multi-seed ablations to H100.

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

The script requests exactly one local `a100_3g.20gb` GPU allocation, eight CPUs,
24 GB RAM and a two-hour limit. Use these local settings only after the user's
explicit local-batch choice; on H100 override partition, GRES, account, QoS and
memory according to that cluster. The one worker runs configurations sequentially
within the allocation, with B*K parallel simulator slots within each CEM solve.

The batch validates GPU execution, measures R=1 throughput, prepares the shared
GPU state bank, runs all 18 comparisons via Hydra's existing Joblib launcher,
and writes a consolidated `comparison.csv`. The summarizer verifies matching
state-bank identities and exact random-reference episode rows across runs.

Outputs are `outputs/transport_oracle_JOBID/`, including `batch.log`, `smoke/`,
`timing/`, `initial_states.pt`, and `comparisons/0..17/`. Each real comparison
retains resolved config, source copies/hashes, package versions, per-episode
CSV, return/success intervals, timing, candidate bank and executed trajectories.
Real runs log to CSV and wandb group `m2-transport-oracle`; preflight runs use CSV
only. Review individual seeds as well as the consolidated table; no pooled-seed
confidence interval is manufactured from duplicated initial states.
