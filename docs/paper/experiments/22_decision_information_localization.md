# Gate 0d: decision-information localization

Status: protocol frozen; overnight run pending. Date: 2026-09-17.

## Question

The model predicts some local interactions, physical observability reduces
collisions, and a true-dynamics CEM controller makes progress. Nevertheless,
the learned planner ranks a known-good oracle plan poorly. Gate 0d asks exactly
where the information required to rank plans disappears:

1. the encoded representation;
2. the learned transition rollout;
3. the scalar reward/termination readout; or
4. the candidate distribution CEM visits.

This is a measurement of the existing system, not another architecture sweep.
The encoder, transition, conditioner, latent dimension, SIGReg objective,
reward head, dataset, and planner budget are frozen.

## Frozen panel and shared bank

Use the 18 seed-4100 checkpoints from job 1223:

`3 inputs × 2 action regimes × 3 conditioner kinds`.

Every checkpoint receives the same 16 simulator roots and the exact same
five-block candidate bank from experiment 19:

`1 true-dynamics CEM oracle + 1 zero plan + 300 random plans`.

This panel retains the observability intervention and both action-coverage
conditions while avoiding an unjustified 144-checkpoint representation sweep.
Conditioner kind is reported as a nuisance/replication factor, not promoted as
the next design variable.

## Planning paths

Each candidate is evaluated through the following paths:

| Path | Trajectory source | Planner-facing score | Isolates |
|---|---|---|---|
| A | simulator | true accumulated return | metric sanity ceiling |
| B | true simulator frames encoded by the frozen encoder | existing frozen reward/termination readout | encoder + current scalar interface |
| C | autoregressive frozen latent rollout | existing frozen reward/termination readout | transition damage beyond B |
| D | autoregressive frozen latent rollout | newly fitted diagnostic progress/clearance/collision readout | information present after rollout but discarded by reward scalar |
| T→D | true simulator frames encoded by the frozen encoder | the same task-sufficient diagnostic family | encoder sufficiency independent of transition rollout |

The diagnostic readout predicts only quantities Buzz Wire needs:

- change in ball-to-goal distance;
- signed ball-to-wire clearance;
- ball position;
- collision probability and collision penalty.

It is a probe/planning-interface diagnostic. It does not update or replace the
world model. It is fitted separately on frozen true-latent transitions and
frozen predicted-latent transitions, selected on the validation split, and
reported once on the test split.

## Measurements

For every path and root report:

- Plan–Real Spearman rank correlation;
- top-30 recall;
- oracle-plan percentile (0 is best);
- selected-plan regret relative to the best true candidate;
- selected true return, final ball-goal distance, and collision rate.

Save raw fixed-bank scores as tensors so every aggregate can be audited.

Repeat the rank measurements on the 300 candidates sampled at every one of 30
iterations of the current reward-head CEM planner, for eight shared roots. This
stage diagnostic determines whether apparently acceptable global ranking fails
specifically in the increasingly narrow, model-selected region where control
depends on it.

## Decision rules

| Observation | Diagnosis | Licensed next experiment |
|---|---|---|
| B bad and T→D bad | true encoded paths lack usable task geometry | representation-shaping objective |
| B bad, T→D good | latent contains task information but scalar reward readout discards it | structured planning cost or preference/energy objective |
| B good, C bad | recursive transition destroys decision information | dynamics/rollout repair |
| fixed-bank ranking good, CEM-stage ranking degrades | model exploitation or local coverage gap | Gate 3 competent trajectories plus local joint-action perturbations |
| all offline rankings good, closed-loop control bad | execution/distribution-shift problem | closed-loop Gate 3, not another offline probe |

No architecture, reward-head-capacity, CEM-budget, or generic reconstruction
sweep is licensed before this decision table is resolved.

## Rendered outputs

The merged run produces:

- `fixed_bank_paths.{png,pdf}` — A/B/C/D/T→D rank and regret contrasts;
- `cem_stage_spearman.{png,pdf}` — rank fidelity over CEM iterations;
- `diagnostic_heads.{png,pdf}` — decoded progress, clearance, and collision quality;
- `aggregate.json` — all 18 records and their aggregate table.

Implementation: `examples/world_model/decision_information.py`.
Launcher: `scripts/slurm/decision_information_gate.sbatch`.

