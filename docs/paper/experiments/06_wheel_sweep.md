# Wheel — matched baseline sweep and evaluation

2026-09-14. The user requested Wheel with concurrent runs on the 20 GB MIG,
matching the Transport and Buzz Wire experiments. This explicitly authorizes
the local MIG allocation for this sweep despite the general H100 offload policy.

## Question and fixed design

Does the relational model capture the cross-agent response transmitted through
Wheel's rotating line, and does that improve candidate-plan ranking? This is a
development comparison, not an untouched confirmation bank. No hyperparameter
search, new observation features, reward changes, or heuristic trajectories are
part of this run.

| Choice | Wheel protocol |
|---|---|
| Reference runs | Transport job 1194; Buzz Wire job 1196 |
| Baselines | independent, joint, relational |
| Action regimes | correlated and independent, on matched source anchors |
| Training seeds | 4100–4107, paired across all six model/regime settings |
| Real training runs | 3 × 2 × 8 = 48 |
| Dataset | 128 root episodes; state/split/action/branch seeds 3100/3101/3102/3103 |
| Split | root-episode train/validation/test = 96/16/16; anchors inherit split |
| Collection | stride 20, 25-step snippets, 5-step action blocks, branch batch 256 |
| Task | 4 agents, 100 steps, line length 1, mass 30, desired speed 0.05 |
| Capacity | conditioner budget 1,083,072; latent width 192; shared model defaults |
| Training | 100 dynamics + 50 frozen-dynamics readout epochs; batch size 128 |
| Optimizer | AdamW, lr 5e-5, weight decay 1e-3, gradient clip 1 |
| SIGReg | weight 0.09; 17 knots; 1,024 projections |
| C7 | same held-out anchor pairs, agent 1 x-action flipped, block horizon 5 |
| C8 | up to 128 test anchors × 64 shared candidate plans; seed 5100 |
| Reporting | paired seed differences and bootstrap intervals; rankable fraction |
| Resources | Slurm gpu partition, gpu:a100_3g.20gb:1, 12 CPUs, 48 GB RAM, 3 hours |
| Concurrency | five Joblib training workers share the allocated MIG |
| Logging | real collection/training: CSV + online wandb, counterfactual-wm/cair-traffic |

Eight training seeds quantify optimizer/initialization variation on one fixed
dataset. They do not establish uncertainty across independently collected banks.
The later Buzz Wire holdout and SIGReg search are separate experiments; they are
not added to this matched baseline comparison.

## Implementation and documentation audit

The installed `vmas==1.5.2` scenario is authoritative for this run. Its reward and
observation code also matches the [upstream Wheel source](https://raw.githubusercontent.com/proroklab/VectorizedMultiAgentSimulator/main/vmas/scenarios/wheel.py)
checked on 2026-09-14. The wrapper is a configuration dataclass; physics and reward
come from VMAS. BenchMARL deliberately supplies `line_length=1`, whereas the
standalone scenario defaults to 2; retain the repository task default.

- The line is pinned at the origin and can rotate. Its tracked physical state
  must include rotation and angular velocity. Existing uncommitted collector
  changes already select rotatable landmarks and add full-state coverage metrics;
  these were retained and tested. The old fallback selected both line and static
  center, so it did not lose the line entirely; the decisive problem was that
  position/velocity-only diagnostics cannot detect its rotation.
- The C7 and stratified evaluators still sliced physical state to four columns.
  Added explicit `--full-state` to both, used by the Wheel submission. Historical
  commands retain their original position/velocity labels. These are physical
  effects on another agent **or the shared object**, not necessarily movement of
  another agent's body. Purely observational changes do not define the label.
- Wheel has 13 observation features per agent. Its shared reward is
  `-abs(abs(line_ang_vel) - desired_velocity)`, equal to minus the last feature of
  the next observation. Either rotation direction can earn optimal reward; the
  earlier survey's signed-velocity formula was incorrect and is corrected.
- The scenario has no task-success termination. Episodes end at the configured
  time limit; a termination head with no positives cannot be called validated.
  Report return/speed tracking and plan ranking, not task-success percentage.
- Observing speed error makes instantaneous reward observable, but does not make
  a single agent's observation a full Markov state: other agents and the sign of
  angular velocity are absent. No latent-planning competence is assumed.
- The claim that mass alone proves one-agent solutions impossible is stronger
  than the implementation establishes. Coupling/contact coverage is measured in
  `coverage.json`; the task name or mass is not evidence of sufficient coverage.
- Snapshot/replay, episode masks and source splits are inherited unchanged.
  Collection asserts bit-exact reference replay and disjoint episode splits.
  Source hashes, resolved configs and dataset manifests record the dirty checkout.
- All evaluation commands explicitly select CUDA. Stratified JSON goes inside
  this job's output directory, preserving the pre-existing root-level artifact.

The [experiment plan](../experiment_plan.md) and [M1 protocol](01_protocol.md)
remain the framing; the signed-reward correction and this scheduled Wheel
comparison supersede their historical task-selection recommendations for this run.

## Validation and reproduction

CPU collection (`outputs/wheel_audit_cpu_20260914`): eight root episodes, 30 steps,
48 anchors; both action regimes reload and all replay/leakage checks pass.
This disposable bank is only a pipeline check, not a coverage result.

Three concurrent CUDA smoke runs (all baselines) completed on the disposable
bank with bit-exact checkpoint reloads. Dynamics parameter counts were
4,213,530 / 4,213,680 / 4,213,008 (independent/joint/relational), within 0.02%.
The same three counts appear in the smoke and in all 48 real runs. (An earlier
revision of this file quoted 3,816,216 / 3,816,366 / 3,815,694; those are
Transport's 11-observation shape, not Wheel's 13, and were a transcription
error. The matched-capacity claim is unaffected: the spread is 0.016%.)

Focused tests: 61 passed across `test_world_model_collection.py`,
`test_world_model_models.py`, `test_world_model_dataset.py`, `test_world_model.py`,
and `test_world_model_wheel.py`. New tests cover real rotating-line replay,
reward/timeout semantics, angular-only labels, source-id alignment, exclusion of
the intervened agent, masking after either branch ends, and explicit unmeasurable
reports for empty interaction strata.

Before the real training sweep, the job validates its actual MIG memory size,
collects and verifies the bank, and runs all six model/regime combinations for
two dynamics epochs plus one readout epoch with CSV only. Then it runs the 48
full-budget experiments and C7, C8 and stratified evaluation in that allocation.

```bash
sbatch scripts/slurm/wheel_pipeline.sbatch
squeue -u "$USER"
```

Artifacts: `outputs/wheel_<jobid>/data`, `baselines/<run>/model.pt`, per-run
`metrics.json`, `parameters.json`, `history.csv`, resolved configs and provenance;
top-level `counterfactual.txt`, `plan_ranking.txt`, `plan_ranking_truth.pt`,
`stratified.txt`, `stratified_scores.json`, `batch.log`, `gpu_memory.csv`, source
snapshot, submission script, sweep config and working-tree patch. Simulator truth
uses a fresh cache per job. Real training wandb group: `m4-wheel-baselines`.

## Submission and startup audit

Job **1205** (`wheel-pipeline`) started at **18:50:16 UTC**, 2026-09-14, on
`gpu-a240`. Slurm allocated `gpu:a100_3g.20gb:1`, 12 CPUs and 48 GB RAM. CUDA
reported **20,096 MiB** and the memory monitor resolved the allocated device to
`MIG-b41c5bd0-f3d2-564b-859a-e6f2e060b5ee`.

Artifacts: `outputs/wheel_1205/`; log: `slurm-1205.out`. Collection produced
**1,280 anchors** from 128 roots, with **96/16/16** train/validation/test roots.
Both replay checks passed, including 256 prefix anchors and counterfactual
reference replay; no duplicate state crossed splits. All six concurrent MIG
smoke runs completed with exact checkpoint reloads. Hydra launched the **48 real
runs at 18:51:09 UTC** with five workers; online wandb logging is active.

The startup coverage audit found **0/160 interaction-active test anchors at five
primitive steps**, even with full-state labels. Over 25 steps, four anchors show
a line-state response and one an other-agent response. Thus the matched C7
five-step measurement cannot establish interaction prediction on this bank.
The protocol and budgets remain fixed; this is a coverage limitation, not a
negative model result or a reason to tune against the test split.

After this became visible, the C7 and stratified evaluators were corrected to
report an explicit **unmeasurable** status for an empty active stratum (or either
empty stratum for the stratified comparison), avoiding NaN-based summaries.
This post-submission change affects evaluation reporting only; training and data
are unchanged. For job 1205, the exact updated evaluator files and diff are saved
separately under `evaluation_source/` and `evaluation_working_tree.patch`, so the
initial submission snapshot remains intact. The empty-stratum regression passes.

Training and C8 results are pending; successful startup is not sweep completion.
At **18:53:50 UTC**, five full-budget runs had completed with zero checkpoint
reload error and the next five had started. The job remained RUNNING. Observed
GPU memory reached 19,494 / 20,096 MiB during the first wave without an OOM;
keep the requested five-worker cap and retain the memory trace.

## Results

Job 1205 completed in 26 min 39 s. All 48 runs produced metrics, checkpoints
reloaded bit-exactly, and the M4 gate passed: latent variance 0.86, effective
rank 23.3-23.7 of 192, no collapse in any run.

### The bank behaves as the audit predicted

| diagnostic | value | reading |
|---|---|---|
| `nonzero_reward_fraction` | 1.000 | reward is dense; it is literally observation feature 13 |
| `moving_package_fraction` | 0.000 | the pinned line has no linear velocity, as expected |
| `changing_package_fraction` | 0.108 | the line does rotate, in ~11% of transitions |
| `package` effects | 0/160 | position/velocity labels cannot see rotation |
| `package_full_state` effects | 4/160 | rotation-aware labels can |
| `task_terminations` | 0 | Wheel has no `done()`; the termination head has no positives |

The two rotation fixes are therefore load-bearing: without them this bank would
have reported zero line motion and zero interaction, both false.

### C7 and the stratified comparison are unmeasurable

Interaction-active test anchors, rotation-aware labels, by horizon in primitive
steps: 0 at 1, **0 at 5** (the matched C7 horizon), 2 at 10, 3 at 15, 4 at 20,
5 at 25 -- out of 160. Both evaluators reported `no_active_anchors` /
`empty_stratum` rather than emitting NaN summaries. This is a coverage
limitation of the bank, not a negative model result.

For comparison at the same 25-step horizon: Transport 40/239 (17%), Buzz Wire
117/117 (100%), Dropout 0/160.

### Plan ranking is null

51/128 states rankable. Spearman is within +/-0.011 of zero for every baseline
and every regime, and every paired interval against `independent` contains zero.

### Relational wins anyway, and that falsifies C10

Paired by seed, relational against independent on rollout error:

| regime | independent | relational | reduction | 95% CI | seeds |
|---|---|---|---|---|---|
| correlated | 0.06878 | 0.05235 | **+21.7%** | [+12.0%, +31.7%] | 8/8 |
| independent | 0.07439 | 0.05523 | **+22.8%** | [+14.1%, +32.4%] | 8/8 |

Set against the other three tasks, the benefit does **not** track measured
coupling:

| task | coupling (25 steps, full-state) | relational rollout gain |
|---|---|---|
| Dropout | 0/160 (0%) | -9% / -28%, 3/8 seeds (noise) |
| **Wheel** | **5/160 (3%)** | **+22%, 8/8** |
| Transport | 40/239 (17%) | +12%, 8/8 |
| Buzz Wire | 117/117 (100%) | +23%, 8/8 |

Wheel has one sixth of Transport's coupling and twice its gain, and matches
Buzz Wire's gain with 3% of its coupling. C10 as stated -- that the relational
benefit tracks measured cross-agent coupling -- does not survive this.

### What was ruled out

1. **Observation redundancy.** A ridge probe predicting agent i's next
   observation from the other agents' current observations, fit on train anchors
   and scored on test, adds +0.0000 R^2 on all four tasks (own-observation R^2 is
   already 0.999). No support.
2. **Horizon mismatch.** Still 5/160 at 25 steps. Rejected.
3. **Shared-body prediction.** Stratifying test anchors by whether the line moved
   gives +11.7% (moved) against +25.1% (static). The relative difference is a
   denominator artifact -- absolute reductions are 0.0167 and 0.0158, effectively
   equal -- so this neither supports nor refutes the mechanism.
4. **Architecture alone.** Ruled out. Permuting which episode's partner latents
   are pooled, at evaluation time only, damages relational on every task and
   every regime, 8/8 seeds: Wheel +26.0%/+16.4%, Transport +26.4%/+23.8%,
   Buzz Wire +17.8%/+13.3%. On Wheel this returns relational (0.052 -> 0.065) to
   roughly the independent level (0.068). The models genuinely read their
   partners.

The resolution is that C7's label and the rollout gain measure different things.
C7 flips one action component and demands a physical response within five steps;
where contact is rare that fires almost never, yet partner *states* stay
predictive. C7 is sufficient evidence of coupling, not necessary.

The partner-shuffle control is not yet a committed evaluator. It is the
measurement that actually supports the relational claim, and unlike C7 it returns
a number on tasks where C7 is unmeasurable -- including Dropout, where it is
positive (+14.3%/+5.0%) even though relational still loses to independent
outright. No single axis yet orders all four tasks.
