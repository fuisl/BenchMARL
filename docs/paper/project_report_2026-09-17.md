# Multi-agent world models for joint planning — where we are

**Consolidated through 2026-09-18 · branch `feat/le-wm` · target AAMAS 2027
(abstract 1 Oct, paper 8 Oct)**

A short version of the work so far, for people who have not been following the
job numbers. Every number here comes from a file in `outputs/`; the job that
produced it is named beside it.

---

## 1. The bet

Give several robots a shared job. Let them learn a compact latent model of how
the world responds to what they *all* do, from logged data only. At decision
time, search over **joint** action plans inside that model and execute the best
one.

The hypothesis: a model that explicitly represents *"what agent A does changes
what happens to agent B"* will **predict** better, **rank plans** better, and
**control** better than models that do not.

That is a chain of three links. We can now say what happens to each.

---

## 2. What we built

**Stack.** BenchMARL + VMAS (vectorised multi-agent physics), TorchRL /
TensorDict, Hydra multirun, Slurm on one A100 node.

**Pipeline**, five stages, each a separate entry point under
`examples/world_model/`:

```
collect.py    logged trajectories + physical state + anchor snapshots
   ↓
train.py      stage 1 dynamics (JEPA objective)  →  stage 2 readout (frozen)
   ↓
evaluate      plan_ranking · physical_response · horizon_rollout
   ↓
closed_loop.py   CEM-MPC in the learned model, against a true-simulator oracle
```

**Model** (`models.py`), following LeWM:

| part | what it is |
|---|---|
| encoder | MLP, observation → 192-d latent per agent |
| action embedder | primitive actions blocked in groups of 5 |
| **conditioner** | **the only structural difference between baselines** |
| predictor | causal autoregressive transformer, depth 4, AdaLN-zero |
| SIGReg | sketched isotropic-Gaussian anti-collapse term |
| readout | reward + termination heads |

Two training stages are deliberately separated: the three baselines are compared
**only** on the dynamics stage, so nothing but prediction and anti-collapse
shapes the representation being compared. The readout is fitted afterwards on
frozen dynamics.

`action_block = 5` — one model step is five primitive simulator steps, and
episodes are 100 steps.

---

## 3. The three baselines

The whole experiment turns on one module. All three receive agent *i*'s own
action and latent; they differ in what else they get.

| baseline | conditioning vector | cross-agent path |
|---|---|---|
| `independent` | `W[a_i, z_i]` | **none, structurally** |
| `joint` | `W[a_i, z_i, (z_j, a_j) in fixed order]` | full, fixed order |
| `relational` | `W[a_i, z_i, Σ_{j≠i} φ(z_i, z_j, a_i, a_j)]` | full, permutation-equivariant |

`joint` and `relational` receive **exactly the same information** and differ only
in inductive bias, which is what makes `joint` a control rather than a strawman.

**Capacity is matched, not shared.** The three conditioners consume different
input widths, so a common hidden width would give `joint` 3.7× the parameters of
`independent`. `Conditioner.hidden_for_budget` solves for the width that hits a
fixed parameter budget instead, so no difference can be dismissed as "one model
is bigger".

**Two data regimes.** `correlated` (agents' actions copy agent 0's sign per axis)
and `independent`. Generalising to action combinations never seen together is
the thing being tested, so the training distribution is a controlled variable.

---

## 4. How we test: gates

The gate structure is not ours — it comes from an external review
([`review_2026-09-16.md`](review_2026-09-16.md) §Gates 0–5), adopted wholesale
after it reproduced three defects we had missed. Gates run **in order**, and a
gate that fails stops the ones after it. That discipline is why we learned
control fails *before* spending a large allocation on more seeds of it.

| # | The question the gate asks | What we ran | Verdict |
|---|---|---|---|
| **0** | Repair and audit before another sweep — is the data what we think it is, and does each measurement read the thing it names? | 1235, 1236, contract tests | **passed** |
| **1** | Is the task controllable **with true dynamics at all**? Compare do-nothing, random, the scenario heuristic and simulator-MPC on the same roots and budget; hold H=5 and compare executing five blocks against one. | 1237 Balance, 1259 Buzz Wire | **passed** — and exposed the cadence bug |
| **2** | With encoder geometry and reward-head quality removed, can learned joint dynamics support that validated controller? Score all three variants in **one common coordinate system**. | 1239, 1264 | **passed on Buzz Wire**; showed Balance is unmeasurable |
| **3** | Is the **data coverage** adequate before adding model complexity? Competent-local actions and planner-induced hard negatives on split-safe roots. | 1331 | **coverage helps physics, ranking, and progress; it does not make control safe** |
| **4** | Does a learned model actually control, using exactly the admitted task, objective, inputs, horizon, cadence and search budget? Three-seed pilot first; expand only a *functioning* comparison. | 1261–1263 | **failed** |
| **5** | Choose the next model change **from the remaining failure** — and change one thing, not data + loss + architecture + planner at once. | 1273, 1276, 1291, 1294--1296, 1331 | **localized through a failed structured-surrogate control gate** |

Three things worth knowing about this:

* **Gate 1 is the cheapest and it paid for itself twice.** Asking "can the *true
  simulator* do this task at this cadence?" before comparing learned models is
  what revealed that every control result for three months had been measured
  through a planner that re-observed only four times per episode.
* **Gate 2 is the one we would recommend to other teams.** Before comparing
  models on a quantity, check the quantity is above your own measurement floor.
  We added an explicit probe-resolution check, and it retired a whole task's
  worth of previously reported numbers.
* **Gate 4's own rule stopped us expanding.** It says to grow to eight seeds only
  once the comparison *functions*. It did not, so the expansion was not
  scheduled — 18 cells at 0/32 is not a variance problem.

The later Gate 5 sequence first localized the latent/reward/rollout failures,
then job 1331 changed both representation and coverage as an explicitly named
existence test rather than an architecture comparison. Its behavior-only/full
pair isolates the coverage effect inside that new baseline.

**Deviation we are carrying.** Gate 4 asks for *fresh* root episodes at final
evaluation. Earlier learned-control tables use 32 train-split roots; job 1331
does use the 16 frozen test roots, but a final headline controller would still
need a newly generated untouched evaluation bank.

## 5. Main findings

### 5.1 Prediction works, and it replicates

Intervention response, scored in **common physical coordinates** (a probe maps
each model's latent to real positions and velocities, so models are compared in
one space rather than in their own private latents). Lower is better; 1.0 = no
better than predicting no response.

| bank | winner | paired vs `independent` | seeds |
|---|---|---|---|
| Buzz Wire primary (job 1239) | `relational` | −0.0841 | **8/8** |
| Buzz Wire holdout (job 1264) | `relational` | −0.1162 | **8/8** |

Two independently collected datasets, 8 seeds each, same winner, same magnitude.
`relational` also leads on *direction* (cos 0.37–0.40 against `independent`'s
0.20–0.25).

### 5.2 The benefit scales with observability

Same task, same capacity, three input conditions trained identically (job 1223,
rescored in job 1264 against a common physical target):

| what the encoder sees | effect of conditioning | seeds |
|---|---|---|
| full physical state (24-d) | **−0.34** | 8/8, all four cells |
| agent observations (6-d) | −0.085 | 8/8 |
| stacked history (18-d) | *unmeasurable* — probe error is 1.37–1.66× the effect | — |

This is the controlled within-task experiment the direction needed. More
observable state → larger benefit from modelling interaction.

### 5.3 Control fails, completely

Gate 4 on Buzz Wire (jobs 1261–1263): the same checkpoints that win 5.1,
planning against a true-simulator oracle on identical roots and budget.

| policy | success | return | task distance | collisions |
|---|---:|---:|---:|---:|
| do-nothing | 0/32 | 0.00 | 0.9528 | 0.00 |
| random | 0/32 | −6.25 | 0.9526 | 0.62 |
| **oracle (true simulator)** | **25/32 (78%)** | **+0.83** | **0.1259** | **0.00** |
| best learned cell | 0/32 | −8.86 | 0.954 | 0.84 |
| worst learned cell | 0/32 | −10.63 | 0.958 | 0.94 |

**18 of 18 cells worse than doing nothing. 17 of 18 worse than random.** No
architecture separates from any other — every paired interval spans zero.

### 5.4 Why — and it is not the world model's dynamics

The learned policies **collide more often than random actions do** and die by
step 20–38 of 100. That is not an inert planner; it is a planner being steered
into the wire.

The cause is in the observation design. Buzz Wire's agents observe
`[pos, vel, pos − goal]` and nothing else. **The ball jointed between them —
whose contact with the wire *is* the failure the reward punishes — appears in no
agent's observation.** Job 1203 measured the consequence directly: the reward
head ranks plans at Spearman ≈ **−0.25** when handed the *simulator's own*
latents, so the ordering is wrong before any rollout error enters. CEM then
faithfully maximises a predicted return that never falls when the plan crashes.

A world model cannot avoid a constraint it cannot represent. Image- and
video-based world models never meet this problem, because the obstacle is in the
frame — predicting the frame means predicting the obstacle. Hand-picking a vector
observation picked the constraint *out*, and no amount of self-supervision can
put back information the input does not carry.

---

## 6. Problems we hit, and what they cost

Most of this project's elapsed time went into discovering that numbers were
**unmeasured** rather than wrong. The recurring lesson is that in this setting,
measurement validity dominates modelling.

**The planning cadence.** Every closed-loop result for three months executed the
*whole* 25-step plan before re-observing — four decisions per 100-step episode,
which is nearly open-loop. Fixing it to one block per decision:

| task | before | after |
|---|---|---|
| Balance (job 1237) | +22.4 return, 47% failures | +54.3 return, **0%** failures |
| Buzz Wire (job 1259) | 47% success, 19% collisions | **78%** success, **0%** collisions |

Every Buzz Wire control result the project had ever produced was measured through
the broken cadence, so none of them had tested a working controller.

**A whole task turned out to be unmeasurable.** Balance's cross-agent effect sits
**9–70× below** any probe's resolution, and an MLP probe does not help — so it is
not readout weakness. Its probe floor even exceeds the models' own scores, which
is incoherent and is what a pure-noise measurement looks like. Every Balance
counterfactual number we had reported is withdrawn.

**The probe family silently decides the winner.** With a *linear* readout,
`relational` goes from best (0.85) to worst (1.01) — because a linear probe
decodes `relational`'s latent 35–80% worse, and that decodability gap gets
charged to the model as response error. An MLP decodes all three to within 13% of
each other. **Any cross-model claim in a shared space must report the per-model
reconstruction error beside it**, or the probe's own bias picks the winner.

**Withdrawn claims.** A reported "18.7× error cliff" at horizon 5→8 was 1.16×
once the rollout stopped passing through a positional slot that never received a
prediction gradient (job 1235). Balance's "competent" reference trajectories had
been generated by *Transport's* hand-written policy. Both are now caught by
contract tests in `test/test_world_model_contracts.py`.

---

## 7. The observability repair — and what it bought

§5.4 said the failure was information, not dynamics. That was testable without
training anything, because job 1223 had already trained 48 checkpoints whose
encoder sees the ball (24-d physical input) at matched capacity on the same
seeds. They had never been *planned* with — `closed_loop` sized its observation
from the environment spec, so a 24-d encoder could not be fed — so we built
`model_input.py`, which constructs all three input conditions from a live
simulator and from recorded trajectories, with contract tests asserting both
match the dataset's own transform value-for-value.

**Job 1273** then scored all 18 checkpoints of one seed against one set of
references, making `observation` an in-job control rather than a separate run.

| input | return | task distance | **collisions** | timeouts |
|---|---:|---:|---:|---:|
| `observation` (6-d) — the condition that failed | −9.91 | 0.9615 | **0.94** | 0.06 |
| `history` (18-d) | −5.20 | 0.9409 | **0.49** | 0.51 |
| `physical` (24-d) — the ball supplied | −1.81 | 0.9920 | **0.17** | 0.83 |
| random | −6.25 | 0.9526 | 0.62 | 0.38 |
| do-nothing | 0.00 | 0.9528 | 0.00 | 1.00 |
| **oracle** | **+0.83** | **0.1259** | 0.00 | 0.22 |

**The mechanism is confirmed.** Collisions fall five-fold, ordered exactly by how
much true state the encoder receives. Job 1276 finds plan-ranking quality moving
the same way — Spearman 0.03–0.06 for `observation` against **0.146–0.212** for
`physical`. The failure was information, and it is repairable without retraining.

**It is still not control.** `physical` times out in 83% of episodes and ends
*further* from the goal than doing nothing (0.9920 against 0.9528). **Zero of
eighteen cells beat the do-nothing baseline**, and every cell is 0/32. The entire
return gain is the removal of collision penalties, not task progress: the planner
went from actively harmful to inert.

**Observability, not architecture, is what moves this pipeline.** Across both
experiments the input condition produces 3–5× effects where the predictor
produces none that order consistently — within `physical`, `independent` ranks
best at plan ranking and `relational` worst, reversing the physical-response
result on the same bank.

### One correction to §5.4

§5.4 cited job 1203's readout Spearman of ≈ −0.25 as evidence. That number is not
valid as stated: `train.py:120-125` fits the readout on **predicted** latents by
design, so the simulator's true latents are off-distribution for it, and the
diagnostic measures that mismatch rather than readout quality. The diagnosis
survives — §7's two experiments confirm it directly — but that particular
supporting number needs re-measuring with a readout fitted on true latents.

### A bottom-up dynamics check

Job 1291 asks a narrower question than the latent rollout plots: can one
transition function, shared across agent identities, directly predict each
agent's next physical position over the five-step action block? Across 144 fits
(8 seeds × 3 inputs × 2 action regimes × 3 conditioners), every model beats
copying the current position. Relative coordinate MSE ranges from 0.187 for the
weakest observation-only independent cell to 0.019 for the best
physical-input correlated cells, where 1.0 is the no-motion baseline.

Joint and relational conditioning each beat the independent model on 8/8 seeds
in all twelve paired comparisons. Supplying physical state improves every
architecture/regime/seed and places 96–99% of conditioned predictions within
1 cm in the correlated bank. History is only a small and non-uniform repair.

This proves that one-block agent motion is learnable and that shared context
matters. It does **not** prove that the planning model's latent transition is
correct: job 1291 trains a separate `(dx, dy)` target and never feeds that head
to CEM.

Jobs 1294–1295 now close that comparison. A position head fitted on true
latents exposes a large immediate coordinate-system mismatch. Fitting a second
head directly on predicted latents—as the planning readout is fitted—reduces
one-block RMSE to about 0.031–0.034 m, but every one of 144 checkpoints is still
worse than persistence and 3.84–13.13× worse than its matched direct model.
Recursive h=5 error reaches about 0.049–0.055 m with an MLP head. The failure is
already present at h=1; it is not mainly long-horizon drift, and architecture
choice changes it far less than the latent-vs-direct gap. Full numbers and
renders are in [experiments 20](experiments/20_agent_position_validation.md)
and [21](experiments/21_latent_position_validation.md).

### Decision localization and the structured-surrogate test

Job 1296 asks the decision-level version of that question. Useful progress is
present in the physical model's true latents (Plan--Real Spearman 0.478 and the
known-good plan near the top decile), but the scalar reward interface is already
poor on true latents. Recursive rollout then degrades the progress ordering and
rejects the same known-good plan. Across CEM iterations, optimization removes
the remaining useful true candidates from its own population. This separates
reward-interface failure, rollout loss, and optimizer-induced model
exploitation; see [experiment 22](experiments/22_decision_information_localization.md).

Job 1331 then performs the requested existence test without learned latents or
a scalar reward head. A 14-state physical surrogate predicts next state,
clearance, collision probability, and collision penalty; the known Buzz Wire
cost is reconstructed during planning. The full training mixture adds
true-dynamics competent plans, local perturbations, and simulator-labelled
candidates queried from CEM iterations 1/5/10/20/30.

Coverage works on its intended intermediate targets. It cuts ball-position
error on held-out oracle/local actions from about 0.035--0.036 m to about
0.0075--0.0078 m, raises Plan--Real Spearman from 0.162 to 0.286, and all three
seeds move the ball closer than do-nothing. It still does not produce safe
control: the three full-mixture seeds collide in 69--88% of episodes, return is
negative, and only 1/48 episodes succeeds. The known-good plan remains around
the 59th predicted percentile, while the full model's selected test-bank plan
has worse true return despite better global correlation.

This is a registered failure of Baseline B, not a null result. It establishes
that coverage is causal for physical generalization and progress while showing
that average one-step accuracy, collision AUROC near 0.98, and improved global
ranking do not secure the extreme low-cost tail CEM uses. The competent action
basin is also narrow: local perturbations around the oracle collide 40.9% of
the time. Full results and the next controlled diagnostic are in
[experiment 23](experiments/23_planner_coverage_structured_surrogate.md).

---

## 8. Where this leaves the paper

**Solid.** Cross-agent response is measurable in coordinates that do not depend
on which model produced them; relational conditioning captures more of it than a
matched single-agent model, on two independent datasets, 8 seeds each; and the
benefit scales with state observability in a controlled within-task intervention.

**Falsified.** That better cross-agent prediction yields better joint planning.
Tested directly on the one task where both ends are measurable, re-tested after
repairing observability, and finally tested with a task-sufficient physical
surrogate and planner-induced coverage. The last intervention recovers motion
but not safe control.

So the paper is **a measurement paper**, and §7 makes it a stronger one than it
was yesterday. Cross-agent response is measurable; conditioning captures it; the
benefit scales with observability; **and none of it transfers to control** — now
demonstrated twice, with the intermediate failure diagnosed, repaired, and shown
to move the mechanism by 5× without moving the outcome at all. That last clause
is the contribution: it is a much harder result to dismiss than a single negative.

What it is *not* is a variance problem. Eighteen cells at 0/32, twice over, will
not be rescued by more seeds, so the 8-seed expansion has not been scheduled.

### The open measurement, now closed

The planner was never the limit — the same CEM with true dynamics solves 25/32 —
and the remaining suspect was the reward head. Job 1282 refits it on true latent
pairs, which is the comparison the diagnostic always needed. Two results:

* **A long-standing number is retired.** Job 1203's readout Spearman of ≈ −0.25,
  cited throughout this project as evidence the head cannot order plans, was the
  distribution mismatch. A fairly fitted head scores **+0.028 … +0.050**.
* **The readout is not the bottleneck.** The refitted head predicts reward
  **twice as accurately** on held-out data and ranks plans *worse* — near zero
  for `observation` and `history`, and negative for `physical`. Gate 4 was
  already using the best-ranking configuration available.

Reward-prediction accuracy and plan-ranking ability move in opposite directions
here. That is a third dissociation, beside response-vs-control and
architecture-vs-observability, and it closes the last cheap line of attack.

---

## Pointers

| | |
|---|---|
| Per-job notes | [`experiments/README.md`](experiments/README.md) maps every job to its artifacts |
| Current status | [`status_2026-09-16.md`](status_2026-09-16.md) |
| The failure in full | [`experiments/16_gate4_buzz_wire.md`](experiments/16_gate4_buzz_wire.md) |
| Decision-level localization | [`experiments/22_decision_information_localization.md`](experiments/22_decision_information_localization.md) |
| Structured-surrogate coverage gate | [`experiments/23_planner_coverage_structured_surrogate.md`](experiments/23_planner_coverage_structured_surrogate.md) |
| Measurability | [`experiments/14_gate2_gate4.md`](experiments/14_gate2_gate4.md) |
| Conventions | [`coding_rules.md`](coding_rules.md) |
| Figures | `outputs/horizon_curves/` — rollout-error curves and imagined-rollout filmstrips; regenerate with `scripts/horizon_curves.sh` and `scripts/imagination_figures.sh` (`outputs/` is gitignored, so only the code is versioned) |

**Known debt:** nine jobs still have no written note, including several
closed-loop control results. The outline calls this the project's largest
correctness risk.
