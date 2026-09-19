# Gate 5: planner-induced tail failure

Status: implementation complete; submission pending. Date: 2026-09-18.

## Question

Why does optimization of the frozen full-coverage structured surrogate produce
increasingly bad real plans?

Gate 5 changes no model or planner component. It localizes the failure between:

1. one-step error on the distribution CEM induces;
2. recursive error accumulation;
3. collision probabilities that are accurate in aggregate but false-safe in
   the selected tail.

## Frozen source

The experiment loads job 1331 without retraining:

- full-coverage seeds 9100, 9101, and 9102;
- the same 16 frozen test roots;
- the same five-block horizon and five-primitive-step action block;
- 300 CEM candidates, 30 elites, and 30 iterations;
- expected collision probability objective;
- control seed 8700 and the same joint-action parameterization.

The entry point validates these values against job 1331's saved `result.json`
and refuses to run if they differ.

## Information contract

Every simulator-dependent quantity is explicitly labelled:

| variable | role |
|---|---|
| goal/task specification and candidate actions | **O**: deployable information |
| exact agent and ball pose/velocity used by job 1331 | **P**: privileged information |
| future state, clearance, collision, reward, and simulator rollout | **T**: future truth |

The deployment rule is: **a deployed planner may consume only O**. Job 1331's
14-D structured input is P, so Gate 5 is explicitly diagnostic-only and cannot
be cited as a deployable controller. CEM consumes the frozen initial P-state,
candidate actions, and surrogate predictions. It never receives future T. The
simulator runs only after CEM finishes and cannot affect candidate selection,
proposal updates, or elites. The saved artifact records that the strict
deployment contract is not satisfied.

## Population protocol

For each model seed, run one frozen CEM search from all 16 test roots and retain
the entire 300-plan population at iterations 1, 5, 10, 20, and 30. After search,
append the job-1331 true-dynamics oracle plan as a diagnostic candidate. The
oracle is not presented to CEM.

Every population candidate and the post-search oracle receive three paths:

- `simulator`: VMAS future truth;
- `teacher_forced`: one surrogate block from the true state at every boundary;
- `recursive`: the surrogate's own predicted state feeds its next block.

The recorded recursive cost must reproduce the planner's frozen cost to within
`2e-5` maximum absolute error or the job fails.

## Measurements

At every CEM iteration and rollout horizon 1--5, report:

- ball position and velocity error;
- agent-state coordinate error;
- minimum-clearance error;
- reconstructed-cost error;
- cumulative collision Brier score, ten-bin calibration error, and false
  negative rate at threshold 0.5;
- Plan--Real Spearman, top-10/top-30 recall, oracle percentile, selected regret,
  and elite-set Jaccard overlap;
- `P(true collision | predicted top-k)` for k = 1, 5, 10, and 30, beside the
  corresponding mean predicted collision probability.

Simulator-to-teacher-forced error measures one-step/OOD failure.
Teacher-forced-to-recursive error measures accumulation. All ranking and tail
metrics are computed separately for both paths. Roots, rather than the 4,800
candidate trajectories, are the sampling unit; the three model seeds are
reported separately and then averaged descriptively.

## Registered branch rule

The next model change is selected from the earliest mechanism whose rule fires:

1. **Planner-induced OOD -> structured DAgger (G6a).** Iteration-30
   teacher-forced ball-position or clearance/cost error is at least 1.5x its
   iteration-1 value at three or more horizons for at least two model seeds.
2. **Recursive instability -> action-prefix dynamics (G6b).** If rule 1 does
   not fire, recursive ball-position error at iteration 30, horizon 5 is at
   least 2x teacher-forced error, with a recursive-vs-teacher-forced gap of at
   least 0.01 m, for at least two seeds.
3. **False-safe tail -> conservative collision scoring (G6c).** If neither
   upstream rule fires, at iteration 30/horizon 5 the predicted top-10 has at
   least 25% true collision and underestimates that rate by at least 15
   percentage points for at least two seeds.
4. **None fires:** do not train a new model. Audit the task-cost reconstruction
   and the candidate comparison before authorizing another intervention.

If several rules fire, report all of them but intervene on the earliest causal
stage in the order above, changing only that stage.

Implementation: `examples/world_model/planner_tail_failure.py`.
Launcher: `scripts/slurm/planner_tail_gate.sbatch`.
