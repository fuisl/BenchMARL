# Multi-agent world models for joint planning — where we are

**2026-09-17 · branch `feat/le-wm` · target AAMAS 2027 (abstract 1 Oct, paper 8 Oct)**

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
| **3** | Is the **data coverage** adequate before adding model complexity? Competent-local actions vs controlled single/pair deviations vs broad independent actions, on matched anchors and budgets. | **not run** | open |
| **4** | Does a learned model actually control, using exactly the admitted task, objective, inputs, horizon, cadence and search budget? Three-seed pilot first; expand only a *functioning* comparison. | 1261–1263 | **failed** |
| **5** | Choose the next model change **from the remaining failure** — and change one thing, not data + loss + architecture + planner at once. | 1273, 1276 (running) | in progress |

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

Our current work is a **Gate 5** action and obeys its constraint: the input
condition changes, while data, loss, architecture and planner all stay fixed.

**Deviations we are carrying.** Gate 3 has not been run, so data coverage remains
an untested alternative explanation for the Gate 4 failure. And Gate 4 asks for
*fresh* root episodes at final evaluation; we use 32 train-split roots and hold
the 16 test roots frozen, so the final table still owes a run on unseen roots.

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

## 7. What is running today

The diagnosis in §5.4 is testable without training anything, because job 1223
already trained 48 checkpoints whose encoder sees the ball (24-d physical input),
at matched capacity, on the same seeds.

They had never been planned with — `closed_loop` sized its observation from the
environment spec, so a 24-d encoder could not be fed. Today we built
`model_input.py`, which constructs all three input conditions from a live
simulator and from recorded trajectories, with contract tests asserting both
constructions match the dataset's own transform value-for-value.

**Job 1273** re-runs Gate 4 with all three input conditions in a single job
against one set of references, so `observation` is an in-job control that should
reproduce the failure and `physical` tests the fix. **Job 1276** re-measures the
readout's plan-ranking correlation across all 192 checkpoints, which says whether
seeing the ball repairs the −0.25 anti-correlation at the root of §5.4.

Early and *not* a result: at a toy search budget the physical-input model finished
every episode with **zero collisions**, against 0.84–0.94 for the models that
cannot see the ball.

---

## 8. Where this leaves the paper

**Solid today.** Cross-agent response is measurable in coordinates that do not
depend on which model produced them; relational conditioning captures more of it
than a matched single-agent model, on two independent datasets, 8 seeds each; and
the benefit scales with state observability in a controlled within-task
intervention.

**Falsified today.** That better cross-agent prediction yields better joint
planning. Tested directly, on the one task where both ends are measurable, and it
failed — 0/32 everywhere, worse than random.

Two honest endings, and job 1273 chooses between them:

1. **The chain is repairable.** If the constraint enters the input and control
   follows, the paper is the original story plus a diagnosed and fixed failure —
   a stronger result than if it had simply worked.
2. **It is a measurement paper.** Cross-agent response is measurable, conditioning
   captures it, it scales with observability, **and it does not transfer to
   control** — demonstrated rather than asserted, with the mechanism identified.

What it is *not* is a variance problem. 18 cells at 0/32 with returns below random
will not be rescued by more seeds, so the 8-seed expansion has not been scheduled.

---

## Pointers

| | |
|---|---|
| Per-job notes | [`experiments/README.md`](experiments/README.md) maps every job to its artifacts |
| Current status | [`status_2026-09-16.md`](status_2026-09-16.md) |
| The failure in full | [`experiments/16_gate4_buzz_wire.md`](experiments/16_gate4_buzz_wire.md) |
| Measurability | [`experiments/14_gate2_gate4.md`](experiments/14_gate2_gate4.md) |
| Conventions | [`coding_rules.md`](coding_rules.md) |

**Known debt:** nine jobs still have no written note, including several
closed-loop control results. The outline calls this the project's largest
correctness risk.
