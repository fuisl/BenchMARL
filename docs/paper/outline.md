# Where this project stands, and how it finishes — AAMAS 2027

**Updated 2026-09-16.** Replaces the 2026-09-14 outline, which was written before
the [audit](audit_2026-09-15.md), before the Stage 0/1/2 repairs, and before jobs
1217–1226. Deadlines: **abstract 1 Oct 2026 (15 days), full paper 8 Oct 2026 (22
days).** 8 pages body, unlimited references, double-blind, ACM `sigconf`.

This document is written in plain language on purpose. Technical terms are
defined in [§9 Glossary](#9-glossary--what-the-words-mean). Every number in it
comes from a file in `outputs/`; the source job is named next to it.

---

## 1. The one-page version

**What we set out to show.** Give several robots a shared job. Let them learn a
compact mental model of how the world responds to what they all do, from logged
data only. Then, at decision time, search over joint action plans inside that
mental model and execute the best one. The bet was that a model which explicitly
represents *"what agent A does changes what happens to agent B"* would predict
better, rank plans better, and control better than models that don't.

**What we found.** The bet is right at the first link and does not survive to the
last one.

| Link in the chain | Result | Confidence |
|---|---|---|
| 1. Predicting the effect of action combinations never seen together | **Works.** The relational model captures 27–53% of the cross-agent effect the other models miss, 8/8 seeds, on a held-out second dataset | High |
| 2. Turning that into a better ranking of candidate plans | **Partly.** Only under a goal-distance objective, only on Buzz Wire, and small | Medium |
| 3. Turning a better ranking into better control | **Not measured.** The objective the closed loop scored is task-orthogonal — a controller that reaches its goal perfectly ends no closer to the task than random does. Zero native task successes for every learned controller | High |
| 4. The benefit appears exactly where the physical coupling is | **Works.** Four tasks, coupling measured directly, benefit tracks it | High |

**Why link 3 fails is now measured, and it is mostly not the world model.** Three
things stand between us and control, and only one of them is the model:

1. **The reward head is nearly useless on Buzz Wire.** It predicts team reward at
   0.955 relative error — barely better than always guessing the average. A
   planner cannot maximise a number it cannot predict. (Job 1217.)
2. **The planner is weak, and we can prove it.** On Transport, VMAS's own
   hand-written policy scores **3.179** return and gets the only native success we
   have ever seen, while our search planner *with the true simulator as its model*
   scores **1.278** and gets none. The "oracle" is not an upper bound. (Job 1222.)
3. **The loop is barely closed.** Each episode is 100 steps and the planner
   commits 25 steps at a time, so there are **4 decisions per episode** and the
   warm start is all zeros. This is close to open-loop planning. (Job 1218 timing.)

**What changed the picture most recently, and what it cost us.** We built a
second oracle. Until job 1221 there was only a reward-maximising oracle, which is
the wrong ruler for a goal-reaching controller. The new goal oracle reaches the
target in **20/20** episodes on both tasks, which briefly read as a ceiling: on it,
learned planners close about 30% (Buzz Wire) / 41% (Transport) of the distance
between random behaviour and the oracle.

That reading is withdrawn. Scoring the same episodes on the task's own distance
shows the goal oracle ends at **0.8993 on Transport where random ends at 0.8988**,
and at **1.0937 on Buzz Wire where random ends at 1.0852** — reaching the goal
perfectly leaves task performance at, or slightly below, doing nothing in
particular. The goal was the endpoint of a *uniform random* 25-step plan
(`closed_loop.py`, `achieved_goals`), and 25 steps of random action make almost no
task progress: over 300 candidates the best endpoint improves task distance by
0.0012 on Transport and 0.045 on Buzz Wire. So the fraction-of-gap-closed numbers
are fractions of nothing, and link 3 has not been measured rather than measured
and failed.

**What we can say instead** is sharper and is a protocol result: a goal-reaching
protocol in the LeWM lineage can be satisfied perfectly while task performance
stays at the random floor, and a matched task-metric column is what catches it.
Every control table in this project now carries that column.

**So the paper.** The original headline — *interaction modelling → better ranking
→ better control* — cannot be written honestly. The paper that **can** be written,
entirely from artifacts already on disk, is a diagnosis: interaction-aware
prediction demonstrably works and demonstrably does not reach control yet, and
here is exactly which link breaks, on which task, and why. See
[§5 The decision](#5-the-decision-which-paper-we-write).

---

## 2. The story, link by link

### 2.1 What the project is

There are two related problems and they are **not** the same problem. Keeping
them apart is the single most useful thing in this document.

| | **Reward maximising** | **Goal reaching** |
|---|---|---|
| The planner is told | "collect as much team reward as you can" | "end up in this specific situation" |
| The model must supply | a **reward head** and a **termination head** on top of the dynamics | nothing extra — just distance in its own latent space |
| Success means | the task's own `done()` fires | the final joint observation is within 0.05 of the target |
| Our reference ceiling | `oracle` (CEM with the true simulator, maximising reward) | `goal_oracle` (same, minimising goal distance) — **new, job 1221** |
| Where the target comes from | the task | the endpoint of one real trajectory from the same start, so it is reachable by construction |

These two pull in opposite directions, and we measured how hard. On Buzz Wire the
reward oracle ends **0.839** away from the generated goal, while *random actions*
end **0.256** away — chasing reward actively moves you away from the goal. In the
other direction the goal oracle earns **−3.999** return against the reward
oracle's **−2.962**. One controller cannot serve both objectives, which is why
both oracles now exist and why every table reports them separately.

The project originally committed to reward maximising
([experiment_plan.md](experiment_plan.md), "What to do about planning cost") and
treated goal reaching as a labelled alternative. **The evidence has reversed
that.** Goal reaching is where everything we can measure actually happens.

### 2.2 Link 1 — predicting unseen action combinations. This works.

The setup: freeze the simulator at a state, branch it twice — once with the
original joint action, once with one agent's action changed — and ask each model
to predict the *difference*. A model that ignores other agents predicts zero
difference by construction, so it is the floor.

Job 1224 re-derived this on every checkpoint with the repaired evaluator (the old
version divided one model's error by a *different* model's floor, which compared
magnitudes across two unrelated learned spaces — audit finding F6):

| Dataset | Action regime | independent | joint | relational |
|---|---|---:|---:|---:|
| held-out bank (1201) | correlated | 1.000× | 0.673× | **0.470×** |
| held-out bank (1201) | independent | 1.000× | 0.499× | **0.488×** |
| development bank (1196) | correlated | 1.000× | 0.813× | **0.734×** |
| development bank (1196) | independent | 1.000× | 0.889× | **0.741×** |

Lower is better; 1.000× means "no better than predicting no effect at all". The
relational model captures roughly **27% on the development bank and 53% on the
held-out bank** of the effect the independent model structurally cannot see, and
it beats the joint model, which has *identical information* and differs only in
how that information is wired. Every paired comparison is 8/8 seeds with
confidence intervals clear of zero.

A second, harder check: rank each model's *predicted* response size against the
simulator's *true* response size, in shared physical units. This is one common
target for all three models, so it does not depend on anyone's latent space.
Relational scores 0.479/0.470 on the held-out bank against joint's 0.327/0.427;
independent predicts exactly zero everywhere and is reported as undefined rather
than as a bad number.

**This is the project's strongest result and it survived the audit's hardest
correction.**

### 2.3 Link 4 — the benefit tracks coupling, not task names. This works.

Same code, same budgets, same capacity, same seeds, four tasks. First we measure
how much cross-agent coupling each task actually has, by intervention, then we
measure how much of it each model captures:

| Task | Anchors where one agent's action visibly moves another's world | What relational captures |
|---|---:|---|
| Buzz Wire (2 agents, rigid joint through a ball) | 113/117 and 110/115 | 27–53% |
| Transport (4 agents pushing a 50 kg package) | 31/239 | **nothing** (1.002×) |
| Wheel (4 agents, shared line) | 0/160 at this horizon | unmeasurable |
| Dropout (no dynamic coupling) | 0/160 | unmeasurable |

The benefit appears exactly where the coupling does. This is a clean
dose-response and it is the result that makes the negative Transport finding
*informative* rather than embarrassing: architecture cannot recover an
interaction the data never contains.

### 2.4 Link 2 — plan ranking. Split verdict.

Two cost families, two different answers, and the difference is itself a finding.

**Under the reward cost**, the model scores each block of the plan and sums.
Errors accumulate at every block:

| Task | With true latents (readout only) | With the model's own rollout | Reading |
|---|---:|---:|---|
| Buzz Wire | −0.01 to −0.06 | 0.015–0.061 | the reward head itself is uninformative; nothing to destroy |
| Transport | **0.705–0.726** | 0.076–0.159 | the reward head is *good*; the rollout destroys it |

Only **28 of 96** Transport states are rankable at all — most random plans
produce indistinguishable costs.

**Under the goal cost**, the model only reads the endpoint, so errors accumulate
once. We decompose it into three scores on the same candidate plans: **A** = true
simulator, true physical distance; **B** = learned representation over *true*
endpoints; **C** = learned representation over *predicted* endpoints.

| Task | B vs A (is the representation aligned?) | C vs B (what does prediction cost?) | selected-plan regret |
|---|---:|---:|---|
| Buzz Wire | 0.534 | 0.380 → **0.444** (relational best) | 0.0127 → 0.026–0.030, random 0.038 |
| Transport | 0.372 | 0.82 | 0.0565 → 0.062, random 0.100 |

Read across: **the bottleneck is task-specific.** Buzz Wire's representation
orders goals well and its rollout wrecks that ordering (top-plan agreement falls
from 41% to 12%). Transport's rollout preserves ordering nearly perfectly and the
representation is the limit. And crucially, **B vs A is identical to three
decimal places across all three architectures on both tasks** — how the latent
space is *shaped* has nothing to do with which predictor you bolt on.

### 2.5 Link 3 — control. Not achieved, and until now not measured.

Jobs 1218/1221/1222/1226, 20 held-out states, 4 training seeds, all policies
scored on the same states with the same planner seed. **"Task distance" is the
scenario's own distance** — ball-to-goal on Buzz Wire, package-to-landmark on
Transport — and it is the only column comparable across every row. "Goal
distance" is distance to the LeWM target, which each planner is free to ignore.

**Buzz Wire** (states start at task distance 1.0950)

| Policy | Return | Task distance | Goal distance | Goal reached | Native success |
|---|---:|---:|---:|---:|---:|
| random | −7.990 | 1.0852 | 0.2558 | 1/20 | 0/20 |
| do nothing | 0.000 | 1.0948 | — | — | 0/20 |
| reward oracle | −2.962 | **0.5567** | 0.8388 | 0/20 | **5/20** |
| goal oracle | −3.999 | 1.0937 | **0.0189** | **20/20** | 0/20 |
| learned, goal objective, relational/correlated | −6.378 | 1.0975 | 0.1838 | 7/80 | 0/80 |
| learned, goal objective, best of the rest | −6.6 to −7.9 | 1.0896–1.0983 | 0.199–0.229 | 2–4/80 | 0/80 |
| learned, reward objective, all | −8.4 to −9.6 | 1.0741–1.1042 | 0.32–0.36 | 0–1/80 | 0/80 |

**Transport** (states start at task distance 0.9006)

| Policy | Return | Task distance | Goal distance | Goal reached | Native success |
|---|---:|---:|---:|---:|---:|
| random | 0.175 | 0.8988 | 0.4533 | 0/20 | 0/20 |
| do nothing | 0.000 | 0.8994 | — | — | 0/20 |
| **VMAS hand-written heuristic** | **3.179** | **0.8687** | 1.6958 | 0/20 | **1/20** |
| reward oracle | 1.278 | 0.8878 | 0.4038 | 0/20 | 0/20 |
| goal oracle | 0.127 | 0.8993 | **0.0114** | **20/20** | 0/20 |
| learned, goal objective, all | 0.118–0.130 | 0.8993–0.8994 | 0.268–0.306 | 0/80 | 0/80 |
| learned, reward objective, all | 0.22–0.35 | — | 1.20–1.49 | 0/80 | 0/80 |

Four things to take from these tables, and the first one retracts a claim this
document made in its previous revision.

1. **The goal objective is task-orthogonal, so no control was measured on it.**
   The goal oracle reaches its target 20/20 on both tasks and ends at task
   distance 0.8993 (Transport, random 0.8988) and 1.0937 (Buzz Wire, random
   1.0852) — at or slightly below doing nothing. The goal was the endpoint of a
   uniform random 25-step plan, and random actions make almost no task progress:
   over 300 such plans the best endpoint improves task distance by 0.0012 on
   Transport and 0.045 on Buzz Wire. The earlier claim that learned planners
   "close 30%/41% of the random → goal-oracle gap" is withdrawn: the gap is
   real, but crossing it changes nothing about the task. Job 1227 re-measures
   every cell against a goal taken from the strongest controller each task has.
2. **The two tasks fail in opposite ways, and one fix will not cover both.**
   Buzz Wire is controllable from these anchors — the reward oracle moves the
   ball 1.0950 → 0.5567 and succeeds 5/20 — so it can carry a control claim
   either way. Transport is not: the best policy in existence, VMAS's own
   heuristic, closes 0.032 of the starting 0.9006 in a full 100-step episode,
   3.5% of the distance, and doing nothing loses 0.0012. Transport's 0/20 native
   successes are therefore **not** evidence about any world model.
3. **Reward-objective planners are worse than random on Buzz Wire.** Consistent
   with a reward head at 0.955 relative error: they are optimising noise.
4. **The hand-written heuristic beats the reward oracle on Transport.** Every
   "gap to oracle" number under the reward objective is a gap to a mediocre
   controller, not to the best achievable.

**What is left standing.** Nothing learned achieves native task success anywhere,
and that is still true and still belongs in the abstract. What changes is the
reason we can give: on Transport nothing does, and on Buzz Wire the objective the
learned planners were scored on did not ask them to.

### 2.6 Stage 2 — is it the model, or is the information just missing? (Job 1223, undocumented)

Buzz Wire agents see six numbers: own position, own velocity, own offset to the
goal. The task is defined on a **ball** jointed to both agents, whose movement
drives the reward, and **nobody observes the ball.** So every negative Buzz Wire
result was ambiguous. Job 1223 made the input an experimental axis: `observation`
(the 6 numbers), `history` (the last 3 real frames, 18 numbers), `physical` (the
recorded entity states appended, 24 numbers). 3 models × 2 regimes × 3 inputs × 8
seeds = 144 runs, all complete, none written up.

On reward relative error — which is a **shared physical target**, so it is
comparable across models and inputs:

| Input | independent | joint | relational |
|---|---:|---:|---:|
| `observation` (6) | 0.956 / 0.958 | 0.956 / 0.960 | 0.952 / 0.953 |
| `history` (18) | 0.968 / 0.973 | 0.968 / 0.975 | 0.967 / 0.973 |
| `physical` (24) | **0.862 / 0.890** | **0.857 / 0.882** | **0.860 / 0.889** |

(correlated / independent regimes.)

Three readings, and all three matter:

- **Handing over the hidden ball state helps; three frames of history does not.**
  The missing variable is not recoverable from what the agents see. This is the
  proper test the retracted partial-observability claim was standing in for.
- **Even with the ball handed over, reward prediction is 0.86 relative error** —
  still close to "always guess the average". Something beyond observability
  limits the reward head.
- **Architecture does not move this number; input does.** All three models sit
  within 0.005 of each other at every input level. Meanwhile in each model's own
  latent space, `joint` with `physical` input (0.189 rollout error) *beats*
  `relational` with `physical` input (0.205), reversing the ordering seen under
  partial observation. The honest hypothesis this suggests — **relational
  structure substitutes for missing state information, and stops paying once the
  information is present** — is attractive and currently **unconfirmed**, because
  that last comparison is exactly the cross-latent-space confound (F6) the audit
  told us not to trust. It needs the shared-target rescore in
  [§6 step 3](#6-next-steps-in-order).

---

### 2.7 Capacity and the regulariser gate the whole effect (Job 1224, undocumented)

Job 1224 also rescored the 144 width/regularisation checkpoints from job 1201 —
six settings, 24 models each, all on the **same** Buzz Wire development bank as
the headline numbers, so this is a controlled comparison. Nobody has ever
reported a result from these runs. It is the most actionable thing on disk.

Intervention response, correlated / independent regimes (lower is better,
1.000× = captured nothing):

| Latent dim | SIGReg weight λ | joint | relational |
|---:|---:|---:|---:|
| 16 | 0.009 | 1.003 / 1.004 | 0.988 / 0.995 |
| 16 | 0.09 | 1.005 / 1.007 | 0.995 / 0.994 |
| 48 | 0.009 | 1.003 / 1.003 | 0.933 / 0.978 |
| 48 | 0.09 | 1.016 / 1.005 | 0.965 / 0.983 |
| **192** | **0.009** | **0.762 / 0.913** | **0.612 / 0.719** |
| 192 | 0.09 | 0.766 / 0.870 | 0.731 / 0.739 |

And the shared physical probe — which anchors does each model know are coupled?
This is a common target, so it is comparable across everything in the table:

| Latent dim | λ | joint | relational |
|---:|---:|---:|---:|
| 16 | 0.009 | 0.112 / 0.086 | 0.141 / 0.179 |
| 48 | 0.009 | **0.002 / 0.072** | **0.264 / 0.287** |
| 192 | 0.009 | 0.253 / 0.302 | **0.397 / 0.383** |
| 192 | 0.09 | 0.213 / 0.239 | 0.293 / 0.352 |

Three findings, all free:

1. **Capacity gates everything.** Below dim 192, `joint` captures *nothing* —
   1.00× at every setting. The whole C7 result exists only at the largest width
   we ran. This needs saying in the paper before a reviewer finds it.
2. **Our default regulariser weight is costing us the headline.** Every run in
   the project uses λ = 0.09 ([`benchmarl/conf/world_model.yaml`](../../benchmarl/conf/world_model.yaml)).
   At dim 192, dropping to λ = 0.009 takes relational from 0.731× to **0.612×** —
   from 27% of the effect captured to 39% — while `joint` barely moves
   (0.766× → 0.762×). The regulariser is suppressing the relational model
   specifically. It also lowers latent rollout error for all three models
   (audit §2.2). We have been running the worse of the two settings all along.
3. **Relational structure buys parameters.** On the shared probe, relational at
   dim 48 (0.264) matches joint at dim 192 (0.253), and joint at dim 48 is
   *zero* (0.002). Same information, roughly a third of the width. This is a
   data/parameter-efficiency claim — the kind of positive architecture result the
   project has been looking for — and it needs only per-seed intervals to
   become a figure.

Caveat the audit already states: width changes total capacity (≈1.24 M
parameters at dim 16 against ≈3.81 M at dim 192), so this is not a clean
intrinsic-dimension experiment. Report it as capacity, not as dimension.

---

## 3. The planner — yes, it is plain CEM, not iCEM

This is worth stating precisely, because it changes what the negative control
results mean.

[`examples/world_model/cem.py`](../../examples/world_model/cem.py) is a port of
LeWorldModel's `CEMSolver`: sample a population from a diagonal Gaussian, keep
the 30 cheapest of 300, set the next mean and standard deviation to the elites'
mean and spread, repeat 30 times. Two deliberate deviations are recorded in the
file: we plan in the environment's own action units and clamp to bounds, and we
flatten the joint action so the optimiser stays agent-agnostic.

**This is textbook CEM.** iCEM (Pinneri et al., CoRL 2020) is the improved
variant, and we have **none** of what makes it improved:

| iCEM ingredient | What it does | Do we have it? |
|---|---|---|
| Coloured (correlated) noise | proposes smooth action sequences instead of per-step jitter — the single biggest win for physical control | **No** |
| Memory of elites between iterations | keeps good candidates instead of throwing them away each round | **No** |
| Decaying population size | spends the budget where it helps | **No** |
| Shift-initialised elites from the previous decision | reuses last decision's work | **No** |
| Keep the current mean as a candidate | | Yes |
| Clip to action bounds | | Yes |

On top of that, two settings in the loop hurt us independently of which CEM we
use:

- **`receding_horizon` equals the full planning horizon**, so the controller
  commits all 5 blocks = **25 primitive steps** before it looks at the world
  again. Episodes are 100 steps, so there are **4 decisions per episode**.
- Because the whole plan is consumed, the warm start (`shift_plan`) has nothing
  left to shift and hands the next decision **all zeros**.

The heuristic beating the reward oracle on Transport is the smoking gun: with
*perfect dynamics*, this planner still loses to a reactive hand-written policy.
Until that is fixed or measured, "the learned model is not good enough for
control" is not a supported conclusion — "our planner is not good enough for
control" fits the evidence at least as well. Fixing it is cheap (see
[§6](#6-next-steps-in-order)) and it is the highest-value experiment left.

**Also withdraw the speed claim.** Per-decision times exist but are unusable: the
reference policies ran alone on a GPU slice while the learned sweeps ran 4–8
processes packed onto one. Under that packing, learned planners measured 75–87 s
per decision on Transport against the oracle's 7.7 s. Any speed statement needs a
matched, isolated measurement.

---

## 4. Current status — what exists and what it supports

534 trained world models are on disk (96 in job 1194, 48 in 1196, 192 in 1201, 54
in 1205, 144 in 1223), plus complete closed-loop evaluations, two oracles, a
heuristic reference, and repaired evaluators. **The compute is largely done. The
gap is analysis and writing, not GPU time.**

| # | Claim | Status | Where the evidence is |
|---|---|---|---|
| C1 | Exact counterfactual replay works in VMAS | **Holds** | replay bit-exact, 6,000 candidates; `02_oracle_validation.md` |
| C2 | A simulator planner is a valid dynamics reference | **Holds for goal reaching (20/20 both tasks, job 1221). Fails for reward on Transport** — the heuristic beats it 3.179 vs 1.278 (job 1222) | `heuristic_ref_1222` |
| C3 | Planner settings are justified | **No.** Plain CEM, 4 decisions per episode, dead warm start, never compared against a stronger variant | §3 above |
| C4 | The chosen tasks contain the coupling the question needs | **Holds, and is now quantified per task** | job 1224 |
| C5 | Datasets isolate action coverage | **Holds** for the first block; later branch states differ by construction | `03_datasets.md` |
| C6 | Three models train reproducibly under matched budgets | **Holds** (parameter counts matched to 0.02%, bit-exact reload) | `04_model_baselines.md` |
| C7 | Models look alike on logged data but differ counterfactually | **Holds on Buzz Wire, 8/8 seeds, two independent banks. Null on Transport** | job 1224, `08_stage1_rescoring.md` |
| C8 | The counterfactual gap predicts plan-ranking quality | **Negative under reward cost. Weakly positive under goal cost on Buzz Wire only** | job 1217 |
| C9 | Ranking quality predicts closed-loop control | **Negative for native success. Positive but small for goal reaching** | jobs 1218/1221/1226 |
| C10 | The benefit tracks measured coupling, not task identity | **Holds** — four tasks, clean dose-response | job 1224 |
| C11 | Multi-step training matters more as interaction increases (RQ2) | **Not started** — recommend cutting | — |
| C12 | Results are not single-seed artefacts | **Partial.** 8 seeds on prediction, 4 on control; control uncertainty still bootstraps 20 anchors drawn from only 10 root episodes as if independent (audit F9) | — |
| C13 | Input information, not architecture, drives shared-target error | **Measured, undocumented, one confound left** | job 1223, §2.6 |
| C14 | Reward maximising and goal reaching conflict, and both need their own oracle | **Holds, measured on both tasks** | job 1221 |

### Documentation debt — the biggest risk right now

Jobs **1217 (partly), 1218, 1221, 1222, 1223, 1224, 1226** are finished science
that exists only as JSON and text in `outputs/`. Nothing in `docs/paper/` cites
them. If the paper is written from the notes, it will be written from a picture
that is two weeks out of date and contains claims the audit retracted. **Writing
these notes is step 1 and it needs no GPU.**

---

## 5. The decision: which paper we write

There is one real fork, and it decides how the next 22 days are spent.

### Option A — the diagnosis paper *(recommended)*

**Title direction:** *Where Multi-Agent Latent Planning Breaks: Interaction-Aware
Prediction Without Control.*

**Claim:** modelling how agents' actions affect each other measurably improves
prediction of action combinations that were never observed together, exactly in
proportion to how coupled the task physically is — and that improvement does not
reach control. We localise the break: on one task the learned rollout destroys a
plan ordering the representation got right; on another the representation is
misaligned from the start; and on both, the planner and the objective are
confounds large enough to account for the failure on their own.

**Contributions:**
1. A measurement protocol for counterfactual joint-action prediction that is valid
   across independently learned latent spaces — own-space floors plus a shared
   physical probe. The naïve version of this metric (which we published to
   ourselves and then retracted) is a mistake the field would plausibly repeat.
2. A dose-response result across four tasks with coupling measured, not assumed.
3. A failure decomposition (A/B/C) that separates representation geometry,
   rollout error and scoring, and shows the bottleneck is task-specific.
4. Two matched oracles, and the demonstration that reward maximising and goal
   reaching are different control problems needing different ceilings — plus a
   hand-written policy that beats the reward oracle, which is the evidence that
   the usual "gap to oracle" framing can quietly compare against a weak
   controller.

**Why it is the right call:** every one of those is already on disk. It is
honest, it is unusually well-instrumented for a paper of this kind, and negative
results with this much measurement behind them are publishable at AAMAS.
Limitation stated in the abstract, not buried: no learned controller achieves
native task success.

### Option B — fix the planner and chase a positive control result

Drop reward maximising entirely, fix the cadence, upgrade to iCEM, add the
`physical` input condition, and try to get learned goal planners to reach the
goal often enough to matter.

**Why it is risky:** it is new science with 22 days left, and if the models still
fail you have neither a positive result nor a written-up negative one.

**Recommendation: write Option A, and run Option B's two cheapest experiments as
controls inside it.** The cadence fix and the iCEM comparison are hours of
compute each and they convert §7.4's biggest weakness ("maybe your planner is
just bad") from a reviewer's objection into a measured column in a table. If they
happen to produce a positive control result, Option A absorbs it without changing
shape.

---

## 6. Next experiments, in order

### 6.1 Which of the audit's stages are actually still open

The [audit](audit_2026-09-15.md) §9 lists seven stages. Four are discharged;
what remains is smaller and more specific than the audit's table suggests.

| Audit item | Status today | Evidence |
|---|---|---|
| Stage 0 — repair measurement | **Done** | `c1e27eb`, Stage 0 tests pass |
| Stage 1 — salvage existing results | **Done** | jobs 1217, 1224 |
| Stage 2 — isolate information | **Run, not finished** — 144 models trained, never rescored on a shared target, never written up | job 1223, §2.6 |
| Stage 3 — identify useful coverage | **Not started** | — |
| Stage 4 — relational data efficiency | **Partly answered for free** — the capacity ladder in §2.7 is a data-efficiency result nobody has claimed. The *coverage* axis is still untouched | job 1224 |
| Stage 5 — confirm goal control | **Partly done.** Goals, endpoints, latching and a goal oracle all exist. Still missing: the **zero-action baseline**, a **new root bank**, and **separating horizon from executed blocks** | jobs 1218/1221/1222/1226 |
| Stage 6 — remaining rollout cost | **Gate is now met** (Stage 1 localised the Buzz Wire failure to the rollout) but not started | job 1217 |
| "Cheapest useful control check" — 1 block vs 5 at H=5 | **Not done** | §3 |
| §6.2 — mixed second-difference (paired) intervention | **Not implemented** — the evaluator only ever changes one agent | `counterfactual_evaluation.py:102` |
| §8 contract item 5 — zero-action / persistence baseline | **Not done** | — |
| F9 — cluster bootstrap over root episodes | **Not done** | — |

### 6.2 The experiments, ordered

**E0, running alongside all of these: write up jobs 1217–1226.** One note per
job in `experiments/`, no GPU, about a day. Seven finished jobs — including
every result in §2.5, §2.6 and §2.7 — exist only as JSON in `outputs/` and are
cited by nothing. This is the largest correctness risk in the project and it
blocks §7 of the paper. See [§4](#documentation-debt--the-biggest-risk-right-now).

| # | Experiment | Audit item | Cost | What it decides |
|---|---|---|---|---|
| **E1** | **Zero-action / persistence baseline** on both tasks, both goal banks. A policy that does nothing, scored on exactly the same goals | §8 contract item 5 | **~30 min** | **Run this before anything else.** The goal is the endpoint of a 25-step random plan from the same start. If random plans barely move the agents, a do-nothing policy may land at ~0.2 goal distance — and our learned planners sit at 0.18–0.27. This single number either confirms or **destroys** the project's only positive closed-loop result |
| **E2** | **Cadence control**: execute 1 block per decision at fixed H=5, references plus one seed of each model, both objectives, both tasks | §9 "cheapest useful control check" | ~4 h | Separates "the model is wrong" from "the controller looks at the world four times per episode". Also revives the warm start, which is currently all zeros |
| **E3** | **Paired (mixed second-difference) intervention.** Extend the evaluator from one changed agent to the audit's `f(a_i',a_j') − f(a_i',a_j) − f(a_i,a_j') + f(a_i,a_j)`, and re-measure coupling and C7 on every existing bank | §6.2 | ~1 day to implement, ~2 h to run | Transport has 4 agents pushing a 50 kg package. One agent alone may move nothing, while two changing together do. Our null result there was measured with **single-agent** interventions only, so it may be an artefact of the probe rather than of the task. Turns the Transport null into either a stronger null or a new positive — and either way it is the audit's own recommended measurement |
| **E4** | **Adopt λ = 0.009.** Report the existing six-setting ladder as the width/regularisation study, then re-run the headline Buzz Wire grid at λ = 0.009, dim 192 | §7.3 "reuse the six completed settings first… select on validation decision quality" | 0 (report) + ~6 h (re-run) | §2.7 shows the default λ = 0.09 costs relational 12 points of captured effect and raises rollout error for all three models. Every headline number in the project uses the worse setting |
| **E5** | **Finish Stage 2**: repaired C7 evaluator plus goal A/B/C over the 144 state-input checkpoints | §9 Stage 2 gate | ~3 h | The observation / history / physical comparison currently rests on per-model latent error, which is exactly the F6 confound. On a shared target it becomes the mechanism claim: *relational structure substitutes for missing state information* |
| **E6** | **Statistics repair**: cluster-bootstrap over the 10 root episodes, paired on identical anchors, training seeds as a separate axis | F9 | 0.5 day, no GPU | Every control interval in the project is currently too narrow |
| **E7** | **Isolated latency**: one policy at a time, no packing | §9 compute discipline | ~2 h | Either make the speed claim or drop it. References ran alone, learned sweeps ran 4–8 to a slice, so the current numbers compare nothing |
| **E8** | **iCEM**: coloured noise, elite memory, shift initialisation, same budget, references first | not in the audit — new, from §3 | ~1 day + ~2 h | The heuristic beating the reward oracle says the planner is a live confound. Either it is exonerated or every oracle-gap number gets an asterisk |
| **E9** | **New evaluation root bank** for control: more than 10 root episodes | §9 Stage 5 | ~4 h | The audit's Stage 5 gate explicitly asks for new roots. 20 states from 10 roots cannot carry a control claim |
| **E10** | *(cut first)* **Stage 3 coverage ladder** on Buzz Wire: fixed model, fixed input, three levels of joint-action support | §9 Stage 3 | ~2 days | The audit's proposed contribution, genuinely untested. Only start if E1–E6 are done by 23 Sep |
| — | **Cut** | | | Stage 6 / RQ2 (multi-step training), new architectures, new tasks, foundation features |

### 6.3 Why this order

E1 is first because it is the cheapest experiment in the project and it is the
only one that can *invalidate* a headline result. There is no point tuning a
planner whose apparent success might be reproduced by standing still.

E2 and E3 are next because each one can move a negative result to a positive one
without training anything. E2 attacks the controller; E3 attacks the probe. Both
are pure measurement on checkpoints that already exist.

E4 is the only item that asks for retraining, and it is justified by a controlled
comparison we have already paid for rather than by a hunch.

E5–E7 are cleanup that any reviewer will demand. E8–E10 are genuinely optional
given 22 days.

**Writing runs in parallel.** §3 Related Work, §4 Problem Formulation, §5 Method,
§6 Setup, §8 Limitations and figures F4/T3 are draftable today and depend on none
of the above — about 3,200 words, half the paper. Under Option A, §7 becomes
draftable as soon as the notes for jobs 1217–1226 exist. Target: full draft by
30 Sep, abstract 1 Oct, polish to 8 Oct.

---

## 7. Revised paper structure

8 pages ≈ 6,500 words of body text.

| Section | Words | Content | Blocked on |
|---|---:|---|---|
| Abstract | 200 | Write last. Four moves: multi-agent planning needs answers to action combinations never observed; standard validation error does not certify that; we measure it properly across four tasks with coupling quantified; the ability is real and does not reach control, and we say where it stops | step 1 |
| 1. Introduction | 800 | Open on the dissociation, not on "MARL has attracted attention". Latent planning works in single-agent control; multi-agent world models mostly amortise into a policy rather than search; the one thing that changes with many agents is that actions affect other agents | — |
| 2. Contributions | 150 (inside §1) | The four in [§5](#option-a--the-diagnosis-paper-recommended) | — |
| 3. Related work | 500 | Three paragraphs, each positioned against the counterfactual question, not summarised. Multi-agent world models (MAMBA, MBVD, CoDreamer, MARIE, DIMA); planning in learned models (MuZero → MAZero; PlaNet/Dreamer/TD-MPC2); world-model evaluation. Explicitly disclaim introducing world models or planning to MARL | — |
| 4. Problem formulation | 650 | Dec-POMDP, per-agent latents, the three predictor classes as hypothesis classes **over the same input**, and a tight definition of the counterfactual query. Say explicitly why joint-concatenated is a fair control and not a strawman | — |
| 5. Method | 900 | Encoder and latent; the pairwise interaction term; the two-term loss; **the planner, described honestly as plain CEM with its cadence stated**; both objectives and both oracles | — |
| 6. Experimental setup | 850 | Four tasks with **measured** coupling (T3); snapshot/restore and replay validation; datasets and coverage; metrics, including why raw error across separately learned latent spaces is not comparable and what we use instead | — |
| 7. Results | 1,700 | 7.1 logged parity · 7.2 counterfactual gap (**F1, the core figure**) · 7.3 coupling dose-response (**F2**) · 7.4 plan ranking, A/B/C decomposition (**F3**) · 7.5 closed-loop control against both oracles and the heuristic (**T1**) · 7.6 what the input ablation says (**T2**) | steps 1–4 |
| 8. Limitations | 500 | No native task success anywhere. Plain CEM, 4 decisions per episode. The reward oracle is beaten by a hand-written policy. 10 root episodes behind 20 evaluation states. Architecture cannot create information the data lacks. Vector observations only. Centralised planning only | — |
| 9. Conclusion | 250 | Restate the dissociation and the diagnosis, not the architecture | — |

### Figures and tables

| ID | Type | Content | Status |
|---|---|---|---|
| F1 | Bar + CI | Counterfactual response ratio, three models × two banks × two regimes | **Data exists** (job 1224) |
| F2 | Scatter | Coupling measured per task (x) against benefit captured (y), four tasks | **Data exists** (job 1224) |
| F3 | Grouped bar | Goal A/B/C decomposition per task — where the ordering is lost | **Data exists** (job 1217) |
| F4 | Schematic | The three predictor classes side by side | **Draftable now** |
| T1 | Table | Closed-loop control: both tasks, both objectives, random / heuristic / reward oracle / goal oracle / three models | **Data exists** (1218/1221/1222/1226) |
| T2 | Table | Input ablation on a shared physical target | **Blocked on step 3** |
| T3 | Table | The four tasks, their interaction mechanism, the intervention, and the measured coupling rate | **Draftable now** |

### Citations to resolve before drafting

- **LeWM** — the two-term objective, the CEM defaults and the decoder-free framing
  all come from it, but our notes cite a GitHub repo, not a paper. Find the
  citable artefact or make §5 stand on its own.
- **SIGReg / LeJEPA** — same issue; the regulariser appears in the loss without a
  resolved citation.
- **iCEM** — now needs citing whether or not we implement it, because §5 must say
  which CEM we ran.
- The `<!-- cite: turnNsearchM -->` markers in [litreview.md](litreview.md) are
  search-session artefacts. They must not reach the `.bib`.

---

## 8. What we are explicitly not claiming

Written down so it does not have to be re-litigated:

- Not that relational world models are new. C-SWM and CoDreamer exist.
- Not that we introduced world models or planning to MARL.
- Not that the relational model helps control. It does not, yet.
- Not that partial observability explains Buzz Wire's failure. That claim was
  retracted in `8d45946` — the evidence for it (a negative readout correlation)
  was a training bug, and the repaired readout is uninformative rather than
  misleading. Job 1223 is the proper test and it says the hidden state helps but
  does not rescue.
- Not that the learned model is faster than the simulator. Unmeasured under
  matched conditions.
- Not that our planner is a strong controller. It is beaten by a hand-written
  policy on Transport.

---

## 9. Glossary — what the words mean

Terms used in this project and in the papers around it, in plain language.

**World model.** A learned stand-in for the environment. You give it the current
situation and a proposed action, and it predicts what happens next, without
touching the real simulator.

**Latent / latent space.** The model's private compressed description of a
situation — a list of numbers it invented during training. Two models trained
separately invent *different* descriptions, which is why comparing their raw
prediction errors is meaningless and why we need shared physical targets.

**Encoder.** The part that turns what an agent observes into that compressed
description.

**Readout / reward head / termination head.** Small extra networks bolted on top
that convert the compressed description back into something meaningful — the
team's reward, or whether the episode ended. Needed for reward maximising, not
needed for goal reaching.

**Decoder-free.** The model never reconstructs the original observation. It only
has to predict the next compressed description. Cheaper, and the reason LeWM is
attractive.

**Rollout.** Feeding the model its own prediction back in, repeatedly, to look
several steps ahead. Errors compound, which is why rollout error ≫ one-step error.

**MPC (model predictive control).** At every decision point, imagine many possible
action sequences, score them with your model, execute the best one (or its first
part), then throw the rest away and re-plan. "Receding horizon" is this
re-planning.

**Cadence.** How many steps you actually execute before re-planning. We execute
25 of a 25-step plan, which is why our loop is barely closed.

**CEM (cross-entropy method).** The search inside MPC. Sample many random action
sequences, keep the best few ("elites"), re-sample around them, repeat. Simple and
gradient-free.

**iCEM.** The improved CEM: smooth correlated noise instead of per-step jitter,
remembering elites between rounds, shrinking the population, and reusing the
previous decision's plan. **We do not use it.** See [§3](#3-the-planner--yes-it-is-plain-cem-not-icem).

**Elites.** The lowest-cost candidates in a CEM round; the next round samples
around them.

**Horizon (H).** How far ahead the planner looks, here 5 blocks of 5 primitive
steps = 25 steps.

**Action block.** A group of primitive steps held constant to shorten the search.
Ours is 5.

**Counterfactual.** A "what if" the data never showed you. Here specifically: what
if agent A did this while agent B did that, when in the logs they always moved
together?

**Intervention.** How you create a counterfactual experimentally: freeze the
simulator, change exactly one agent's action, and compare the two futures.

**Anchor.** One frozen simulator state we branch from. All our comparisons are on
identical anchors so nothing differs except the action.

**Coupling.** How much one agent's action physically changes what another agent
experiences. We measure it rather than assume it from the task name.

**Independent / joint / relational.** The three predictors. *Independent* sees
only its own agent. *Joint* sees everyone's information concatenated. *Relational*
sees everyone's information through explicit pairwise terms. Joint and relational
have **identical information**, so any difference is the wiring, not the data —
which is what makes joint the fair control rather than a strawman.

**Regime (correlated / independent).** How the training data was collected.
*Correlated* means the agents' actions shared signs — realistic, but it hides
which action caused what. *Independent* means they were drawn separately — more
informative, less realistic.

**Coverage.** Whether the data actually contains the combinations the planner will
later ask about. Each agent's actions can look perfectly varied on their own while
the *combinations* are badly covered — that is the whole problem.

**Spearman correlation.** Agreement between two rankings, from −1 to 1. We use it
for "does the model rank plans the way the simulator does".

**Regret (selected-plan).** How much worse the plan the model chose is than the
best plan available, measured by the true simulator. Lower is better.

**Oracle.** A planner given the true simulator instead of a learned model. A
ceiling for the *dynamics*, not a globally optimal controller — and on Transport,
demonstrably not even a good one.

**Dec-POMDP.** The formal name for the setting: several agents, a shared reward,
each seeing only part of the world.

**SIGReg.** A regulariser that stops the latent space collapsing to a constant, by
pushing its distribution toward a target shape along random projections.

**Effective rank.** How many directions of the latent space are actually used. Low
means the model is wasting its capacity; we track it to detect collapse.

**Bootstrap / cluster bootstrap.** Resampling your data to get error bars. The
*cluster* version resamples whole groups (here: root episodes) because the
individual rows inside a group are not independent — which is the correction audit
finding F9 asks for.

**Ablation.** Removing or changing one component to see whether it mattered.

**Held-out / test split.** Data the model never trained on. We split by whole
episodes before branching, so no evaluation state shares a trajectory with
training.
