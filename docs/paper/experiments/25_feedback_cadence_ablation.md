# Experiment 25: feedback-cadence ablation

Status: implementation complete; submission pending. Date: 2026-09-18.

## Question

Can more frequent replanning compensate for the dynamics error Gate 5 (job
1335) localized in the frozen structured surrogate, or does the failure
already sit in the first decision regardless of how often the planner
re-observes the true state?

## Design

LeWM-style MPC separates the imagined horizon `H` from the executed prefix
`K`: plan `H` blocks, execute `K`, re-observe the true state, replan. Job
1331's own closed-loop evaluation already fixed `K = 1` -- replanning every
action block. This experiment holds everything else frozen and varies only
`K in {1, 2, 5}`, with `H = 5` fixed:

- `K = 1`: reused verbatim from job 1331's `control_rows`/`timing` for the
  three full-coverage seeds. Not recomputed.
- `K = 2`, `K = 5`: computed here with the same models, the same 16 frozen
  test roots, the same CEM budget (300 samples, 30 elites, 30 iterations),
  the same control seed (8700), and the same `probability` objective job
  1331 selected for the full mixture.
- `K = 5` executes the entire imagined horizon before observing again --
  the coarsest, most open-loop cadence the frozen horizon permits.

No model is retrained and no architecture or cost changes. `evaluate_policy`,
`controller_cost`, and `live_structured_state` are reused unchanged from
`structured_surrogate.py`; only `MPCConfig.receding_horizon` differs.

## First-action quality is not recomputed

CEM's first decision is a deterministic function of the model, the frozen
initial state, the control seed, and the CEM budget -- none of which K
touches. So the quality of the very first plan is identical across K by
construction, and recomputing it here would just reproduce Gate 5's own
one-shot search. Instead this experiment cites, per seed, job 1335's
iteration-30/horizon-5 teacher-forced row: `selected_regret`,
`oracle_percentile`, and Plan--Real `spearman`. If that regret is already
large, no feedback cadence can fix it -- the failure predates any execution
of a stale plan.

## Measurements

Per seed and K: success rate, collision rate, timeout rate, mean per-agent
return, mean final ball-goal distance, mean episode length, number of
replanning decisions, and mean wall-clock seconds per decision.

## Reading the result

- If collision/success/return improve monotonically as K shrinks (5 -> 2 ->
  1) and the improvement is large relative to Gate 5's registered failure
  magnitude, feedback substantially compensates for the dynamics error --
  useful context for interpreting G6a/G6b/G6c, but not a substitute for
  fixing the mechanism Gate 5 identified.
- If collision/success/return stay poor even at K = 1 -- which job 1331
  already ran -- feedback cannot rescue this model, consistent with a first
  decision that is already bad (see `first_action_quality`).
- Either way, this experiment does not change the Gate-5 registered branch
  decision (G6a: structured DAgger / planner-aware aggregation). It is
  supporting evidence, run cheaply because it needs no retraining.

Implementation: `examples/world_model/feedback_cadence.py`.
Launcher: `scripts/slurm/feedback_cadence_gate.sbatch`.
