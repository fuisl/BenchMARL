# Gate 0e: planner coverage and structured physical surrogate

Status: **completed; preregistered failure**. Job 1331, commit `b3697fd`,
2026-09-18. Runtime 35:34, exit code 0. Artifacts:
`outputs/structured_surrogate_1331/`.

## Verdict

The full-coverage physical surrogate **does move the ball toward the goal**, so
planner-induced coverage has a real causal effect. It does not control Buzz
Wire safely: its three seeds collide in 69--88% of test episodes, all have
negative return, and only one of 48 model-seed episodes succeeds. This fails
both the strong and partial decision rules below.

The result is more specific than "physical models fail." One-step physical
prediction on competent and CEM-query data improves sharply, and held-out
Plan--Real Spearman rises from 0.162 to 0.286. The remaining failure is in the
competitive tail that CEM selects: the known-good plan still ranks near the
middle of the bank, the model-selected five-block plan has worse true return
than the behavior-only model's selection, and repeated replanning chooses
unsafe motion.

## Question

Can a cheap learned multi-agent surrogate control Buzz Wire when the system
removes learned latent coordinates and the scalar reward bottleneck?

This is Baseline B requested after Gate 0d. It is one fixed architecture, not a
conditioner or representation sweep.

## Stage 1 — collect the planner's query distribution

The collection is split by root episode before any model fitting and contains:

1. the existing independent and correlated behavior transitions;
2. true-dynamics CEM plans from every initial root, plus eight local joint-action
   perturbations around each competent plan;
3. the four candidates preferred by the frozen seed-4100 physical/joint reward
   planner at CEM iterations 1, 5, 10, 20, and 30.

The third source is replayed in VMAS and receives simulator truth. It measures
and trains on the distribution the learned planner actually exploits rather
than adding generic uniform actions. Hard-query roots are sampled separately
from the train, validation, and test episode splits; only train rows update the
model, validation selects checkpoints/objectives, and test is reported once.

Every primitive trajectory is converted to five-step blocks with:

- both agents' position and velocity;
- ball position and velocity;
- goal position;
- the joint blocked action;
- exact next physical state;
- minimum within-block wire clearance over agents and ball;
- collision event, collision penalty, progress, and team return.

Root identifiers are checked to occur in exactly one split.

## Stage 2 — Baseline B

The surrogate is a small joint MLP with one residual physical transition. Its
state is 14 numbers: `agent0 pos/vel + agent1 pos/vel + ball pos/vel + goal pos`.
It predicts:

- the next 12 dynamic state coordinates;
- minimum clearance during the action block;
- collision probability and the explicit collision penalty component.

It does **not** predict reward or a free scalar value. During rollout, Buzz
Wire's known task definition reconstructs the candidate score:

`team reward = 2 × decrease(ball-goal distance) − collision penalty`.

CEM uses five predicted blocks, 300 candidates, 30 elites, 30 iterations, and
executes one block before replanning—the same working cadence as the simulator
control ceiling.

## Controlled comparison

Train three seeds for two data conditions with an identical model and optimizer:

- `behavior`: the original behavior bank only;
- `full`: behavior + competent-local + CEM hard negatives.

This separates the physical representation intervention from the new coverage.
The collision realization is selected on validation roots between expected
collision probability, the explicit penalty prediction, and predicted
hard-clearance crossing. Test roots do not choose the objective.

Report:

- one-block ball/state/clearance error and collision AUROC by source;
- held-out Plan–Real Spearman, top-30 recall, oracle percentile, and regret;
- receding-horizon success, true return, final task distance, collision rate,
  and planning time on the 16 frozen test root episodes;
- random and do-nothing controls on exactly the same roots.

## Decision rule

- **Strong pass:** the full mixture has positive return, improves final distance
  over do-nothing, stays below random's collision rate, and produces native
  successes for at least two of three seeds.
- **Partial pass:** it makes replicated safe progress but does not finish.
- **Failure:** it remains stationary, collides like random, or still cannot rank
  the held-out competent plan.

If `full` improves over `behavior`, planner-induced coverage is causal evidence,
not merely a post-hoc explanation. If both structured models fail despite good
one-step physics, the remaining target is trajectory-level ranking/multi-step
dynamics rather than latent representation.

Implementation: `examples/world_model/structured_surrogate.py`.
Launcher: `scripts/slurm/structured_surrogate_gate.sbatch`.

## Results

### 1. Collection found the expected optimizer-induced shift

The split-safe bank contains 49,125 block transitions. No root identifier
crosses train, validation, or test.

| split | behavior | competent/local | CEM hard | total |
|---|---:|---:|---:|---:|
| train | 6,146 | 4,158 | 23,861 | 34,165 |
| validation | 939 | 690 | 5,811 | 7,440 |
| test | 1,043 | 680 | 5,797 | 7,520 |

The true-dynamics oracle trajectories are genuinely competent: mean team
return +0.390, zero collisions, and final distance 0.764 over 128 roots. The
action neighborhood is extremely brittle. The 1,024 small/large local
perturbations preserve a similar final distance (0.788) but fall to -8.253
return and 40.9% trajectory collision.

The frozen learned planner's simulator-replayed candidates improve through CEM
iteration 5 and then get worse as optimization continues:

| CEM iteration | true return | trajectory collision | final distance |
|---:|---:|---:|---:|
| 1 | -3.156 | 15.5% | 0.957 |
| 5 | -1.753 | 8.4% | 0.959 |
| 10 | -2.145 | 10.4% | 0.960 |
| 20 | -2.975 | 13.9% | 0.958 |
| 30 | -3.417 | 15.7% | 0.958 |

Each row has 1,536 executed trajectories. More CEM iterations do not produce
more progress; after iteration 5 they concentrate back into more collision and
lower true return. This independently reproduces Gate 0d's model-exploitation
mechanism on the collected query distribution.

### 2. Coverage repairs out-of-distribution one-block physics

The table reports the mean over the three saved seeds on frozen test
transitions. These per-source values were derived after the run from the six
saved checkpoints without fitting or model selection. AUROC is undefined for
the oracle rows because none of those trajectories collides.

| training mix | held-out source | dynamic RMSE | ball-position RMSE | clearance RMSE | collision AUROC |
|---|---|---:|---:|---:|---:|
| behavior | independent behavior | 0.0195 | 0.00724 | 0.00709 | 0.988 |
| full | independent behavior | 0.0197 | 0.00745 | 0.00492 | 0.990 |
| behavior | correlated behavior | 0.0176 | 0.00630 | 0.00656 | 0.980 |
| full | correlated behavior | 0.0176 | 0.00642 | 0.00460 | 0.994 |
| behavior | true-dynamics oracle | 0.0473 | 0.03631 | 0.01115 | -- |
| full | true-dynamics oracle | **0.0138** | **0.00747** | **0.00401** | -- |
| behavior | competent-local | 0.0460 | 0.03460 | 0.01074 | 0.870 |
| full | competent-local | **0.0148** | **0.00775** | **0.00410** | **0.962** |
| behavior | frozen-planner CEM hard | 0.0222 | 0.00859 | 0.00754 | 0.942 |
| full | frozen-planner CEM hard | **0.0205** | **0.00814** | **0.00481** | **0.983** |

The largest change is exactly where the behavior-only model extrapolates: full
coverage cuts oracle/local ball-position error by about 4.6--4.9x and raises
local collision AUROC from 0.870 to 0.962. This is positive evidence for the
coverage intervention, but one-block averages are not sufficient for planning.

### 3. Candidate ranking improves globally but not at the selected tail

Validation selected expected collision probability for both training mixes.
The following test-bank values average the three seeds; each bank has 16 unseen
roots and 302 plans per root (oracle, zero, and 300 random plans). Lower oracle
percentile is better.

| training mix | Spearman | top-30 recall | oracle percentile | selected true return | selected regret | selected collision |
|---|---:|---:|---:|---:|---:|---:|
| behavior only | 0.162 | 0.081 | 0.630 | +0.015 | 0.386 | 0.0% |
| full coverage | **0.286** | **0.190** | **0.592** | **-1.083** | **1.484** | **6.25%** |

Coverage raises correlation by 0.124 and more than doubles top-30 recall, but
the known-good oracle (true return +0.401) still ranks around the 59th
percentile. More importantly, the full model's argmin is worse than the
behavior model's despite the better global correlation. Average ranking quality
and the extreme tail optimized by CEM have separated.

### 4. Closed-loop control fails the registered gate

All policies use the same 16 frozen test roots. Model rows are separate training
seeds; the same roots are intentionally repeated to measure seed robustness.

| policy | team return | successes | collision rate | final distance |
|---|---:|---:|---:|---:|
| random | -16.234 | 0/16 | 81.25% | 0.960 |
| do nothing | 0.000 | 0/16 | 0.00% | 0.967 |
| behavior 9100 | -23.823 | 0/16 | 100.00% | 1.004 |
| behavior 9101 | -21.317 | 0/16 | 100.00% | 1.001 |
| behavior 9102 | -22.526 | 0/16 | 100.00% | 0.980 |
| full 9100 | -17.076 | 0/16 | 81.25% | 0.755 |
| full 9101 | -15.705 | 0/16 | 68.75% | **0.695** |
| full 9102 | -19.793 | **1/16** | 87.50% | 0.864 |

The behavior-only controller collides on every episode and makes no progress.
All full-coverage seeds improve mean distance over do-nothing; at the paired
episode level they finish closer on 16/16, 14/16, and 15/16 roots. Across seeds,
mean final distance is 0.771 rather than 0.967. That motion is unsafe: collision
rates are equal to, below, and above random respectively, returns remain
negative, and successes occur for only one seed (one episode total).

The registered strong pass required positive return, lower collision than
random, and successes from at least two seeds. None is met robustly. The partial
pass required *safe* replicated progress; the 69--88% collision rate rules it
out. The formal result is therefore **failure**, with a useful positive
mechanistic finding that coverage causes progress.

## Interpretation and boundary

This run removes learned latent coordinates and scalar reward prediction, so
neither can explain the remaining failure. It also shows that the one-step
physics problem is learnable on the added distribution. The evidence instead
points to three coupled planning issues:

1. **Tail safety is not captured by average fit or AUROC.** CEM only needs one
   falsely safe high-progress trajectory; those rare errors dominate the chosen
   action even when aggregate collision AUROC is about 0.98.
2. **Recursive ordering is still inadequate.** The full model improves average
   five-block ranking but cannot identify the known-good plan, and its selected
   tail is worse in simulator truth.
3. **The competent action basin is narrow.** Local perturbations around the
   oracle collide 40.9% of the time, so modest action/model error converts
   useful motion into a terminal penalty.

One protocol boundary matters. The CEM hard negatives came from the old frozen
latent/reward planner. They are planner-induced, but they are not on-policy
queries from the newly trained structured planner. The run therefore shows that
one-shot cross-planner coverage transfers enough to recover motion, not that an
iterated data-aggregation loop has failed.

The next controlled gate should collect the structured planner's own candidate
populations, measure collision calibration in its lowest-predicted-risk tail,
and separate recursive transition error from collision-tail error by horizon.
Only then should the model change to a direct multi-block/action-prefix target,
ranking loss, or conservative uncertainty penalty.

## Reproducibility

- In-job preflight: 8 tests passed.
- Peak GPU allocation used: 1,218 MiB on the 20 GB slice.
- Saved provenance: clean worktree, exact commit, source snapshot, launcher,
  logs, GPU trace, checkpoints, fixed validation/test banks, JSON results, and
  PNG/PDF figures.
- Primary machine-readable results:
  `outputs/structured_surrogate_1331/coverage/collection_summary.json` and
  `outputs/structured_surrogate_1331/result/result.json`.
