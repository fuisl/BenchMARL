# Gate 0e: planner coverage and structured physical surrogate

Status: job 1331 running on commit `b3697fd`. Date: 2026-09-18. Artifacts:
`outputs/structured_surrogate_1331/`.

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
