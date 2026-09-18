# Gate 0d: decision-information localization

Status: **complete**. Job 1296, commit `8c8711a`, 33m15s, exit 0. Date:
2026-09-17. Artifacts: `outputs/decision_information_1296/`. Peak GPU memory:
16,476 MiB of 20,096 MiB. All six concurrent groups and all 18 checkpoints
completed; the submit-time worktree was clean.

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

## Result

**Decision information is lost at two interfaces, not one.** The current reward
head is poorly aligned even on the simulator's true encoded trajectories. With
physical input, however, the true latent trajectory supports a useful progress
ranking. Recursive rollout then weakens that ranking and specifically changes
the known-good oracle from a top-decile plan to a bottom-quartile plan. Finally,
reward-head CEM concentrates its population in progressively worse regions of
the real simulator.

This rules out “CEM did not search hard enough.” CEM is successfully optimizing
a learned landscape that becomes anti-aligned with reality where the search
concentrates.

### 1. Shared-bank ranking

All values below average the two data regimes and three conditioner kinds for
seed 4100. `ρ` is Plan–Real Spearman; oracle percentile is the fraction of the
302-plan bank scored better than the oracle, so lower is better. Random ordering
has expected top-30 recall `30/302 = 0.099` and oracle percentile near 0.5.

| input | path | ρ | top-30 recall | oracle percentile | selected distance |
|---|---|---:|---:|---:|---:|
| observation | B: true latent → reward | 0.118 | 0.116 | 0.565 | 0.902 |
| observation | C: rollout → reward | 0.020 | 0.132 | 0.219 | 0.885 |
| observation | T→progress | 0.214 | 0.197 | 0.439 | 0.893 |
| observation | rollout → progress | 0.128 | 0.145 | 0.729 | 0.898 |
| history | B: true latent → reward | 0.148 | 0.083 | 0.770 | 0.900 |
| history | C: rollout → reward | 0.085 | 0.101 | 0.690 | 0.898 |
| history | T→progress | 0.179 | 0.190 | 0.339 | 0.900 |
| history | rollout → progress | 0.165 | 0.146 | 0.657 | 0.897 |
| **physical** | **B: true latent → reward** | **0.097** | **0.040** | **0.827** | **0.902** |
| **physical** | **C: rollout → reward** | **0.019** | **0.072** | **0.537** | **0.904** |
| **physical** | **T→progress** | **0.478** | **0.341** | **0.098** | **0.762** |
| **physical** | **rollout → progress** | **0.309** | **0.241** | **0.738** | **0.886** |

The shared bank itself exactly reproduces experiment 19: oracle return `+0.384`
and final distance `0.7087`, zero return `0.000` and distance `0.9008`, random
mean return `−2.363` and distance `0.9013`.

Three contrasts localize the problem:

1. **The reward interface fails before rollout.** B has only `ρ=0.097–0.148`.
   Physical B places the oracle at the 82.7th percentile and its top-30 recall
   is 0.040, below random overlap. Using the true simulator survival mask makes
   B approximately uncorrelated (`ρ=−0.001, 0.002, 0.020`), so the result is not
   rescued by perfect termination information.
2. **Physical true latents do contain progress information.** T→progress reaches
   `ρ=0.478`, top-30 recall 0.341, and oracle percentile 0.098. This is stable
   across all six physical cells: T→progress `ρ=0.467–0.489`, whereas B is only
   `0.069–0.138`. Conditioner kind and data regime are not the dominant variable.
3. **Rollout damages the useful information and fails systematically on the
   competent plan.** Physical progress ranking falls from `ρ=0.478` to `0.309`,
   but the more decisive change is oracle percentile `0.098 → 0.738`. The model
   can weakly order ordinary random candidates while predicting the useful,
   distribution-shifted oracle trajectory incorrectly.

### 2. A good local probe is not a good planning metric

On held-out snippets, physical predicted latents produce the best ordinary
diagnostic scores:

| input | progress RMSE | wire-clearance RMSE | ball-x RMSE | ball-y RMSE | collision AUROC |
|---|---:|---:|---:|---:|---:|
| observation | 0.00589 | 0.01875 | 0.02775 | 0.08261 | 0.786 |
| history | 0.00598 | 0.01858 | 0.02835 | 0.09735 | 0.805 |
| **physical** | **0.00474** | **0.01094** | **0.00831** | **0.05355** | **0.938** |

Yet the physical structured collision/progress composite has only `ρ=0.141`,
oracle percentile 0.780, and selects distance 0.899: safe but effectively
stationary. The true-latent composite behaves similarly (`ρ=0.195`, oracle
percentile 0.777, no selected collisions, distance 0.899). The progress-only
true-latent score selects distance 0.762 but collides on 15.6% of selections.

This is the project's central dissociation in one controlled experiment:

> Low one-step task-variable error does not imply trajectory-level decision
> fidelity. Collision is readily decoded and dominates an additive scalar;
> progress is weaker, accumulates error, and the competent trajectory is not
> predicted reliably—consistent with the unresolved coverage gap.

### 3. CEM-stage diagnosis

The table follows the populations generated by the **existing reward-head CEM**.
The structured score is evaluated on those same populations but does not drive
their generation.

| input | C reward ρ, iter 1 → 30 | structured ρ, iter 1 → 30 | C-selected regret, iter 1 → 30 | best true return present, iter 1 → 30 |
|---|---:|---:|---:|---:|
| observation | `+0.003 → −0.023` | `+0.046 → −0.051` | `2.166 → 4.180` | `+0.076 → −6.664` |
| history | `+0.136 → −0.012` | `+0.121 → −0.002` | `0.085 → 3.761` | `+0.076 → −1.239` |
| physical | `+0.073 → −0.023` | `+0.171 → +0.226` | `0.497 → 2.512` | `+0.076 → −1.677` |

By iteration 30 the reward score is negatively correlated with simulator truth
for every input. More strongly, search has removed the initially positive-return
candidates: even the best candidate remaining in each final population has
negative true return. The final C-selected collision rates are 52%, 25%, and
21% for observation, history, and physical respectively.

Physical structured quantities retain moderate local ranking signal on the
visited population (`ρ≈0.23` at iteration 30), unlike observation and history.
This supports a structured interface, but the fixed-bank result shows that the
current additive composite and current rollout are not yet a controller.

## Diagnosis and next gate

The pre-registered decision table resolves to a compound branch:

- **B bad, T→progress good for physical:** replace the scalar reward interface;
- **T→progress good, rollout progress worse, oracle rank collapses:** repair
  transition coverage around competent coordinated behavior;
- **CEM-stage reward ranking tends to zero/negative while its population gets
  worse:** model exploitation is active; more CEM iterations are harmful;
- **observation/history remain weak:** retain physical observability for the
  next causal test rather than reopening architecture selection.

The next experiment should therefore train a trajectory-level ranking or
preference cost on separate competent trajectories plus local joint-action
perturbations, with a held-out-root fixed-bank gate. Progress and collision
should be handled explicitly—preferably progress maximization under a calibrated
collision-risk constraint—rather than summed through another unconstrained
scalar MSE head. Only after that score ranks the held-out oracle and remains
aligned through CEM iterations should it be run in closed loop.

Do **not** respond to this result with another conditioner sweep, a wider reward
head, a larger CEM budget, or a generic coordinate decoder. This experiment is
one seed and 16 shared roots, so it localizes mechanisms rather than estimating
population-level confidence; replication belongs after the next interface
passes the frozen gate.

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

The merged run produced:

- `fixed_bank_paths.{png,pdf}` — A/B/C/D/T→D rank and regret contrasts;
- `cem_stage_spearman.{png,pdf}` — rank fidelity over CEM iterations;
- `diagnostic_heads.{png,pdf}` — decoded progress, clearance, and collision quality;
- `aggregate.json` — all 18 records and their aggregate table.

Implementation: `examples/world_model/decision_information.py`.
Launcher: `scripts/slurm/decision_information_gate.sbatch`.
