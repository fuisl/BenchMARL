# Experiment 24/G6a: planner-aware data aggregation

Status: original result is provisional; sampler-matched implementation revised,
rerun not authorized. Updated: 2026-09-20.

## Audit correction before rerun

Job 1337 did not isolate targeted collection from generic collection. Targeted
rows entered collection family 2 while generic rows entered family 0, and the
family-balanced sampler consequently gave the two additions different expected
optimizer weight. Its arm difference remains descriptive but is not clean
causal evidence for planner-aware coverage.

The corrected implementation assigns both additions one common augmentation
family while retaining distinct `source` labels. Training samples 50% from the
unchanged base bank and 50% from cumulative augmentation in both arms. The
number of samples per epoch is fixed at twice the base training-transition
count, early stopping cannot shorten an arm, and both arms therefore receive
the same epoch and optimizer-step budget. These values are stored in each fit
record and in `aggregation_summary.json`. Initial-state roots are placed in a
separate integer namespace so they cannot collide with historical anchor IDs.

The audit also produced a direct counterexample to the 14-D state's claimed
Markov sufficiency: two snapshots with identical agent/ball/goal state and the
same zero action diverge after one simulator step when only an omitted movable
link body's velocity differs. Jobs 1331--1337 must therefore be described as a
**14-D privileged structured-state diagnostic**, not a task-sufficient or full
physical-state model. The sampler correction is ready for review, but no G6a
rerun should begin until the successor state includes the linkage dynamics.

## Question

Does adding the frozen structured surrogate's own CEM hard negatives to its
training set remove the optimizer-induced OOD failure Gate 5 (job 1335)
registered as the primary intervention -- and is that because the negatives
are planner-aware, or just because there is more data?

Job 1335 measured teacher-forced ball-position RMSE at horizon 5 growing from
0.009 m (CEM iteration 1) to 0.034 m (iteration 30), a 3.8x increase, for all
three full-coverage seeds -- rule G6a fired. Job 1336 (feedback-cadence
ablation) then showed replanning frequency cannot rescue this: success stays
near 0% even at K=1, so the failure predates any stale-plan execution and
sits in what CEM selects. G6a is the earliest-stage intervention the
registered branch rule names for that failure.

## Design

Two DAgger rounds, three seeds, each seed its own aggregation chain, starting
from job 1331's own model/dataset (not retrained from scratch):

```
D0 --train--> M0 --own CEM on collection roots--> D_CEM,0
D1 = D0 u D_CEM,0 --train--> M1 --own CEM--> D_CEM,1
D2 = D1 u D_CEM,1 --train--> M2
```

- `D0` = job 1331's `coverage.pt`; `M0` = job 1331's `model_full_{seed}.pt`.
- Collection roots: the 96 split=0 (train) root episodes of the same
  `initial_states.pt` bank that Gate 5's frozen 16 test roots (split=2) come
  from. Disjoint from those 16 by construction (a different split value) and
  checked explicitly (`assert_disjoint_from_frozen_test_roots`) before any
  collection runs.
- Candidates are drawn at CEM iterations 1/5/10/20/30 (matching Gate 5's own
  `HARD_STAGES`), scored by the *current* round's structured surrogate, with
  the `--hard-plans` hardest per stage per root kept. Simulator truth labels
  the kept candidates after selection; it never enters CEM's own candidate
  scoring, the same contract Gate 5 and job 1331 both keep.
- **Matched generic control.** Every round also collects the same roots and
  the identical per-root plan count (`len(HARD_STAGES) * hard_plans`) from
  uniform random actions instead of any model's CEM, then trains a second
  chain on `D0 + generic`. This arm does not depend on a model, so it is
  collected once per round and shared across the three training seeds.
  Comparing the two arms is what separates "planner-aware coverage helps"
  from "more simulator transitions help".

Frozen throughout: architecture, CEM budget (300 samples / 30 elites / 30
iterations / horizon 5), the `probability` objective, action parameterization,
and the training hyperparameters job 1331 used (hidden 256, batch 512, up to
200 epochs, patience 25, lr 3e-4). The 16 Gate-5 test roots participate in
nothing here -- not collection, not training, not model selection.

## Re-diagnosis, not a new diagnostic

After each round, Gate 5 (`planner_tail_failure.py`) reruns *unchanged* on
each resulting model, via a small compatibility shim
(`write_gate5_source`) that packages a round's checkpoints plus job 1331's
own `oracle_plans.pt` (copied verbatim -- the frozen roots never change)
into the directory layout Gate 5's `--source` already expects. No new
diagnostic logic is written; `registered_branch_result` is re-applied
verbatim to ask whether G6a's own registered signature still fires.

## Reading the result

Compare, per seed, targeted round 1/2 against the matched generic round 1/2
and the job-1335 baseline:

| | M0 (job 1335) | generic r1 | targeted r1 | generic r2 | targeted r2 |
|---|---|---|---|---|---|
| TF ball RMSE, iter 1 | 0.009 | | | | |
| TF ball RMSE, iter 30 | 0.034 | | | | |
| ratio (30/1) | 3.8x | | | | |
| recursive-TF gap, iter 30 | 0.121 | | | | |
| false-safe top-10 (true/pred) | | | | | |
| G6a/G6b/G6c fire? | all three | | | | |

- **Targeted collapses the ratio, generic does not (or collapses less):**
  planner-aware coverage is the mechanism. Proceed to rerun Gate 5's branch
  rule on the improved model; if recursion or false-safe tails remain, they
  now justify G6b/G6c on their own terms.
- **Both arms collapse similarly:** the fix was more data, not planner-aware
  targeting -- report this as a genuine, simpler finding rather than
  claiming G6a's specific mechanism.
- **Neither arm collapses despite two rounds:** the "planner-induced
  coverage gap is causally upstream" reading needs revision rather than a
  third DAgger round; audit the cost reconstruction and candidate comparison
  before authorizing another intervention, per Gate 5's own fallback rule.

Do not select between K values, aggregation rounds, or arms post hoc by
whichever number looks best on a single metric -- the comparison above is
read as a whole, against the pre-registered ratio/gap/tail thresholds Gate 5
already defined.

Implementation: `examples/world_model/planner_aware_aggregation.py`
(aggregation) and `examples/world_model/g6a_report.py` (comparison).
Launcher: `scripts/slurm/g6a_aggregation_gate.sbatch`.
