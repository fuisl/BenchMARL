# Cross-scenario admission benchmark

**Status: registered 2026-09-21, before any scenario has been measured.**
Thresholds and the taxonomy are fixed here ahead of results.

Direction: [`../direction_joint_representation_2026-09-21.md`](../direction_joint_representation_2026-09-21.md).
Prior results this replaces as the main line:
[`30_task_a_counterfactual_fidelity.md`](30_task_a_counterfactual_fidelity.md),
[`32_joint_representation.md`](32_joint_representation.md).

## Why

Buzz Wire's cross-agent effect is large, real, active on 100% of anchors — and
unlearnable, because the observation omits the mediating body (K28, K30). We
spent 48 checkpoints, a horizon sweep and a localization study establishing
that. **The purpose of this benchmark is to never spend that again without
checking first.**

Buzz Wire is therefore retained **scientifically** as the partial-observability
reference and retired **experimentally** as the primary task. Its value in the
paper is the contrast: a framework that explains *why* a world model cannot
learn a counterfactual effect, rather than reporting which network won.

## FROZEN MEASUREMENT CONVENTION (2026-09-21)

Four conventions were each chosen implicitly, and two of them moved the headline
by more than 10x (jobs 1514, 1515, 1516). They are now fixed. **Do not reopen
any of them unless a positive or negative control actually falsifies it.**

| Choice | Frozen value | Why |
|---|---|---|
| **Fitting target** | the **cross block**, fitted directly | Fitting the whole `ΔY` and scoring a subset spends head capacity on the shared response, which is 3.3x the cross term. Job 1514: 0.016 → 0.293 |
| **Cells per fit** | **one** `(agent, axis)` cell per head | Pooling four heterogeneous cells charges the input for a representational burden that is not about information. Job 1515: 0.293 → 0.605-0.638 |
| **Ladder intervention** | **sampled reference** (G0 design) | A planner queries joint actions in which both agents act. The midpoint grid leaves the partner passive |
| **`Y`** | every body's full `(pos, vel, rot, ang_vel)`, per-column training-scaled, constant columns dropped | Wheel's line is pinned and only rotates; a motion-only target reads exactly zero there |
| **Grid** | 3x3 midpoint surface, for `J_own`/`J_cross`/`C` **only** | A second difference needs a regular surface; it is not the information question |

The information question is *is the effect recoverable from this input* — an
upper bound on availability. Per-cell, cross-fitted answers that. The pooled
variant answers a different and also legitimate question (can one function serve
the whole intervention space), and is closer to what a world model faces; it is
simply not the admission gate.

### Known limitation, accepted for this screening pass

`grid_rollouts` and `sampled_branches` intersect the live mask across **all**
branches, so an anchor is dropped if *any* unrelated intervention terminated it.
That can bias the surviving sample. Contrast-specific masks are the correct fix
and are **owed before any final statistical claim**; for a screening pass whose
purpose is ranking tasks, the bias is shared across scenarios and does not
invalidate the comparison. Recorded here so it is not rediscovered as a defect.

## What runs: one protocol, every scenario, no world model trained

```
scenario -> deterministic anchor bank -> 3x3 joint-action grid
         -> J_own, J_cross, C_mixed, state dependence
         -> A -> O -> H_dense -> H_model -> S information ladder
         -> ADMISSION REPORT -> reject | admit
```

**Banks are reused, not recollected.** All five existing banks
(`buzz_wire_1196`, `transport_data_1190`, `wheel_1205`, `balance_repair_1236`,
`dropout_control_1193`) share `action_block = 5`, `sequence_steps = 25` and the
same seed family (3100-3103). The audit regenerates its own interventions from
restored snapshots, so bank provenance affects *which states* are sampled, not
the protocol. `balance_repair_1236` is used, never `balance_1233`, whose
heuristic branch was Transport's policy.

### Stage 0 — interaction structure

From each restored anchor, hold every action coordinate at a sampled reference
and sweep **both** agents over `{-1, 0, +1}` on one axis:

```math
J_{\rm own}=Y(+,0)-Y(-,0),\qquad
J_{\rm cross}=Y(0,+)-Y(0,-),
```
```math
C=Y(+,+)-Y(+,-)-Y(-,+)+Y(-,-)
```

`C` is the **mixed** second difference. It is exactly zero for any additively
separable response however large `J_cross` is, which is what separates "your
action affects me" from "our actions interact".

**Decomposed by responding body.** `J_ij = ∂Y_i/∂a_j` is defined on one body's
state. Measuring it on the whole concatenated `Y` would fold the intervened
agent's own large response into the cross term and make every task look
coupled. Agent 0 is the reference responder; shared bodies are reported
separately because the jointly controlled object belongs to neither agent.

**State dependence** is the dispersion of the per-anchor effect. A large but
constant effect is uninteresting — the actions-only head learns it.

### `Y` is task-specific by construction

Every body's full `(pos, vel, rot, ang_vel)`, scaled by its training standard
deviation, with columns constant on training data dropped. Wheel's line is
pinned and only rotates, so a position-only target reads exactly zero there
however much it turns — which is how Wheel was previously recorded as having no
interaction (K9). Buzz Wire's ball translates. One task's target must not be
imposed on the others.

### Stage 1 — information sufficiency

One head family, one budget, one selection procedure; only the input differs:

```math
A\rightarrow O\rightarrow H_{\rm dense}\rightarrow H_{\rm model}\rightarrow S,
\qquad
R_{\rm info}(X)=\frac{E_A-E_X}{E_A-E_S}
```

`S` is privileged state, a **diagnostic reference only** — it never enters a
model. `H_model` samples history at the world model's own **block stride**,
which is a different information representation from dense history: dense
windows mostly expose instantaneous velocity, block-strided windows expose
slower coupled motion. G0b measured only the dense form, and its longest window
(`k=12`, stride 1) is *shorter in wall-clock* than the model's own 3-frame
context, so `H_model` is a genuinely new measurement rather than a re-run.

## Registered taxonomy and thresholds

Fixed before measurement. `E` values are already divided by `‖ΔY_true‖`, so a
reference error **is** an inverse resolution.

| Condition | Verdict |
|---|---|
| diagnostic train error `< 0.5 ×` test error while test `> 1` | **undetermined — overfit**; refuse to classify, increase anchors and rerun |
| reference error `> 1/3`, or cross active on `< 50%` of anchors | **weak-interaction control** |
| cross resolvable, but `max R_info(O, H_dense, H_model) < 0.5` | **partially observable** (the Buzz Wire class) |
| cross resolvable and recoverable, but `C` at the floor | **observable additive** |
| cross resolvable, recoverable, `C` above the floor | **ADMIT** |

Only **ADMIT** scenarios proceed to H0/H1/H2 training. Two admitted scenarios
with different mechanics are preferred over choosing one: a single positive task
is a case study, two begin to support the general claim.

## Stages 3-4, registered but not built

**Stage 3.** On every admitted scenario, the identical factorial
`{H0,H1,H2} × {full, restricted coverage} × seeds`, with no per-task
architecture changes beyond unavoidable input/output dimensions. The restricted
regime must **preserve individual action marginals** while removing particular
joint combinations (e.g. suppress `(+,-)` and `(-,+)` while keeping plenty of
each sign per agent), or counterfactual generalization is confounded with
ordinary action extrapolation.

**Stage 4.** One common evaluation hierarchy, in order: ordinary prediction →
`E_CF` → fixed-plan ranking → selected regret → optimized planning. **No CEM
initially**: every model scores the identical frozen candidate set
`A(s) = {A^(1)…A^(K)}` and the simulator evaluates exactly those, so

```math
R_{\rm sel}=J(A_{\arg\min \hat J})-\min_k J(A_k)
```

isolates whether model quality changes decisions. CEM comes afterwards, and the
existing optimizer-tail work (A1.2 branch C) then attaches to this line rather
than competing with it.

The headline across tasks is whether ordinary prediction error explains selected
regret, or whether counterfactual fidelity does.

## Open discrepancy that must be resolved before the numbers are trusted

**The built-in positive control currently FAILS.** This note registered that
Buzz Wire must classify as `partially_observable` — "if the protocol does not
recover that, the protocol is wrong, not Buzz Wire." On the rebuilt
cross-specific ladder it classifies as **`observable_additive`**, because the
cross-agent effect now looks *recoverable*: `R_O = 0.674` on axis 0, against
**0.016** for the same task in G0b.

Two design differences could produce that, and they are not separated:

1. **How the head is fitted.** T-A2b, G0 and G0b all fit one head to the **full**
   `ΔY` and then *scored* it on a column subset. This benchmark fits a head to
   the cross columns **only**. Predicting the whole next-state response and
   reading off the cross part is a harder, worse-posed problem than predicting
   the cross part directly — the shared-body response is 7.43 against a 2.24
   cross term, so the pooled head spends its capacity elsewhere. **If this is
   the dominant cause, G0b's recovery fractions understate what the observation
   carries, and K29/K30 need revising.**
2. **Where the non-intervened action sits.** G0b sampled agent 0's action from
   the reference distribution; the 3x3 grid pins it at the midpoint. That
   changes the conditional being measured.

`Y` also differs — full informative body state here, MOTION-only there — though
on Buzz Wire the agent columns coincide (rotation is constant and dropped).

### G0d result (job 1515): cell pooling confirmed — and it changes the conclusion

Intervention design, anchors, seeds, scale, head family and `--fit-target` all
held fixed; only which `(agent, axis)` cells enter the fit changes.

| arm | `R_O` | `R_H` |
|---|---:|---:|
| pooled, all 4 cells — **G0/G0b's convention** | **0.293** | 0.316 |
| single cell `1:0` | **0.605** | 0.627 |
| single cell `0:0` | **0.638** | 0.697 |

The pooled arm reproduces job 1514 exactly, and the two mirrored single cells
agree with each other, so this is not direction-specific. **Cell pooling roughly
halves the measured recovery.**

#### The instrument is now fully explained

| convention | `R_O` |
|---|---:|
| pooled head, fitted to the full `ΔY` — **G0b as published** | **0.016** |
| pooled head, fitted to the cross block | 0.293 |
| per-cell head, fitted to the cross block | **0.605–0.638** |
| per-cell, sampled reference, full informative `Y` — the benchmark | 0.683 |

Nothing unexplained remains. The residual 0.605 → 0.683 is the `Y` definition
and reference count.

#### This overturns the headline conclusion, and that must be said plainly

K30 states that Buzz Wire's observation is the binding empirical constraint. It
rests on recovery fractions measured under a convention that is **doubly
depressive**: a head fitted to the whole next-state response, serving four
heterogeneous intervention cells at once.

Under a per-cell, cross-fitted head the observation recovers **0.605–0.638** of
the blind-to-reference gap — a clear majority, and above the 0.5 gate this note
registered. The honest revision is:

> Buzz Wire's observation carries a **majority** of the cross-agent effect that
> the privileged state carries. Exposing the ball adds the remainder. It is not
> true that the observation carries essentially nothing.

**Which convention is correct is a real question, not a formality.** They answer
different things:

* **Per-cell** asks *is the information present?* It is an upper bound on
  availability and is the right gate for an information audit, because pooling
  charges the input for a representational burden that is not about information.
* **Pooled** asks *can one function serve the whole intervention space?* That is
  closer to what a world model actually faces.

For G0's registered purpose — "is the mediating state recoverable from
legitimate observations?" — **per-cell is the correct convention**, and the
admission benchmark already uses it. G0/G0b used the other one without ever
choosing it.

#### Consequences, recorded before any re-measurement

* **K28, K29 and K30 are provisionally in doubt for every absolute fraction and
  every gate decision derived from one.** Their relative orderings were measured
  under one consistent instrument and survive.
* **T-A2b's latent conclusion is the one to re-measure first.** Its `R ≈ 0.02`
  for the latent and `R ≈ 0.86` for latent-plus-ball were both pooled full-`ΔY`
  fits. The *ordering* is almost certainly robust; whether the latent clears any
  gate is not.
* **The admission benchmark's convention is now justified rather than
  accidental**, and the note records why.

This is the third revision of this claim, and each one moved because of the
instrument rather than the task. That is itself the most transferable result
this project has produced.

### Sampled-reference ladder: the change landed, the control still fails

The information ladder now uses G0's sampled-reference design; the 3x3 midpoint
grid supplies `J_own`, `J_cross` and `C` only, and the artifact records which
design each number came from. **The registered smoke does not restore the
positive control**, so the long run stays unlaunched.

Buzz Wire, axis 0, cross ladder on the sampled design:

| input | `E` | `R_info` |
|---|---:|---:|
| `A` | 0.878 | — |
| `O` | 0.532 | **+0.683** |
| `H_dense` | 0.513 | +0.721 |
| `H_model` | 0.517 | +0.713 |
| `S` | 0.372 | 1.000 |

`R_O = 0.683` on the **sampled** design is essentially the midpoint grid's
`0.674`. Switching conditional moved almost nothing.

#### My earlier attribution was wrong, and a third factor explains it

I recorded "grid design carries 0.293 → 0.674". That is **withdrawn**. The two
runs differ in a factor I did not control:

| | G0 / job 1514 | admission benchmark |
|---|---|---|
| head fitting | cross-only | cross-only |
| intervention | sampled | sampled (now) |
| **cells per fit** | **all 4** (2 agents x 2 axes, one head) | **1** (per axis) |

G0's `build_rows` concatenates every `(reference, intervened, axis)` cell into a
single design matrix, so one head must serve four heterogeneous intervention
cells at once. The admission benchmark fits one head per axis. Axis 0 reads
0.683 and axis 1 reads 0.339; a single head compromising across both, plus the
two mirrored agent directions, is a plausible route to G0's pooled 0.293.

So the discrepancy decomposes as **head fitting target** (established, 1514:
0.016 → 0.293 at fixed pooling) and **cell pooling** (candidate, not yet
isolated), with **grid design contributing little**.

#### The mixed target is not measurable on Buzz Wire

`C`'s privileged reference barely beats the actions-only baseline — axis 0
`A = 2.394` against `S = 1.958`, axis 1 `A = 1.219` against `S = 1.452` — so the
span guard returns NaN on axis 1 and the overfit guard fires on axis 0. Both
guards behaved correctly; the honest reading is that **the second difference is
below what this instrument can resolve on this task at `h = 1`**, so the
`ADMIT` / `observable_additive` distinction cannot be drawn for Buzz Wire.

#### Consequence

The scenario returns `undetermined_diagnostic_overfit`, not the registered
`partially_observable`. **The instrument is still not calibrated against a task
whose answer we know**, and no scenario verdict should be trusted until it is.

The minimal remaining discriminator is to run the admission ladder with all
cells pooled into one fit, matching G0. If that reproduces ~0.293, cell pooling
is the whole remaining story and the benchmark must declare which convention it
uses and why. If it stays near 0.68, something else is still uncontrolled.

### G0c result (job 1514): both causes are real, and they split the gap

Identical G0 intervention design, anchors, seeds, scale and head family;
`--fit-target` is the only thing that changes. The `full` arm reproduces G0b
exactly (0.8951 / 0.8882 / 0.7634 / 0.4645), so the comparison is clean.

| fitting mode | `R_O` | `R_H` |
|---|---:|---:|
| `full` — fit all of `ΔY`, score a subset (G0b's behaviour) | **0.016** | 0.306 |
| `cross` — fit the cross block directly | **0.293** | 0.316 |
| `cross` + the 3x3 grid (admission benchmark) | **0.674** | 0.462 |

**Attribution.** Head fitting carries 0.016 → 0.293; the grid design carries
0.293 → 0.674. Neither alone explains the discrepancy, and both are real.

**K29 needs revising, and not in the direction it was written.** Its headline was
that history buys *scale* — recovery rising 0.016 → 0.306 from a single frame to
three. Under a head actually fitted to the cross block, the single raw
observation already reaches 0.293 and history adds **+0.023**. The apparent
history gain was largely the pooled head recovering from its own
misallocation, not information arriving with motion.

**K30 survives, for a narrower reason.** Under G0's design with the better-posed
head, `R_O = 0.293` and `R_H = 0.316` are still well below the 0.5 gate, so Buzz
Wire remains `partially_observable` and the observation remains the binding
constraint. The scoped wording adopted earlier already covers this.

**The 3x3 grid measures an easier conditional than a planner faces.** It pins
the non-intervened agent at the action midpoint — passive — so the cross
response is dominated by rigid-link geometry that observable positions largely
determine. G0 samples that agent's action, mixing its own actuation with the
coupling, and there the hidden ball state matters more. A planner evaluates
joint actions in which **both** agents act, so the sampled-reference conditional
is the decision-relevant one.

**Registered change to the benchmark, before it runs:** the cross-effect ladder
must be measured at a **sampled** reference action for the non-intervened agent,
not at the midpoint. The midpoint surface is still needed for `J_own`, `J_cross`
and `C`, which require a regular grid — but the *information* question must be
asked at the conditional a planner would query. Until that change lands, the
five-scenario verdicts are not trustworthy.

**Superseded resolution plan.** The original text below proposed exactly this
audit; it has now run.

**Registered resolution, before the five-scenario run is interpreted:** rerun the
G0 ladder with a cross-only fitted head on the G0 intervention design. That
isolates (1) from (2). Until then:

* **K29 and K30 are provisionally in doubt**, not withdrawn. Their *relative*
  orderings were measured under one consistent instrument and are unaffected;
  the *absolute* recovery fractions are what the discrepancy touches.
* **No scenario's admission verdict should be acted on** — the Buzz Wire
  classification is the control that says whether the instrument is calibrated,
  and it is currently disagreeing with a prior measurement of the same task.

This is recorded before the run rather than after, so the outcome cannot be
re-narrated.

## Results (job 1522)

Five scenarios x three seeds, `references = 2`, `horizon = 1` block, frozen
convention. COMPLETED in 29:43 on one full A100. Artifacts:
`outputs/stage0_admission_1522/`.

### No scenario is admitted

| scenario | axis | `J_own` | `J_cross` | **active** | `C_mixed` | verdict |
|---|---:|---:|---:|---:|---:|---|
| transport | 0 | 6.49–6.52 | **0.042–0.051** | **0.050–0.056** | 0.10–0.11 | weak-interaction control |
| wheel | 0 | 6.91–6.92 | **0.000–0.013** | **0.000–0.006** | 0.07 | weak-interaction control |
| balance | 1 | 3.80–3.83 | **0.183–0.214** | **0.305–0.316** | 0.38–0.41 | weak-interaction control |
| **buzz_wire** | 0 | 3.98–4.05 | **2.20–2.25** | **1.000** | 1.14–1.18 | undetermined (overfit) |
| dropout | 0 | 7.03 | **0.000** | **0.000** | 0.000 | weak-interaction control |

Seed agreement is tight throughout: the three seeds are independent draws of the
reference joint action, and no verdict changes across them.

### The two controls behave oppositely

**Dropout passes as the negative control.** `J_cross` is exactly 0.000 on both
axes, on all three seeds. Its agents have no cross-agent dynamics by
construction, and the benchmark reports that without qualification. The
instrument does not manufacture interaction where none exists.

**Buzz Wire does not cleanly pass as the positive control.** It is the only
scenario with a cross effect active on **100%** of anchors, and its observation
recovers **0.72** of the blind-to-reference gap — but it still fails to
classify, for two independent reasons:

* `cross_reference_relative_error = 0.372`, just above the registered
  `1/3` resolution threshold. Even the privileged state does not resolve the
  cross effect quite well enough to license an ordering.
* The mixed second difference is not measurable at all
  (`mixed_reference_relative_error = 1.958`), so the overfit guard fires on that
  ladder and classification is refused.

### The three candidate tasks are far weaker than Buzz Wire

This is the substantive finding, and it is not close:

```math
J_{\rm cross}:\quad
\text{buzz\_wire } 2.24
\;\gg\;
\text{balance } 0.18
\;>\;
\text{transport } 0.05
\;>\;
\text{wheel } 0.00
\;=\;
\text{dropout } 0.00
```

Activity is the binding failure: Transport moves agent 0 on **5%** of anchors,
Balance on **31%**, Wheel on **0%**. The registered gate needs 50%.

Transport's reading reproduces the existing record rather than contradicting it:
K9 records 31/239 active anchors there, and the M5 note records **0/239** after
one primitive step. Wheel's zero reproduces K9 as well.

### What this does and does not establish

**It does not say these tasks have no interaction.** It says that *at one action
block, under this intervention*, an action by one agent does not measurably move
the other. Transport's package has mass 50 against a `u_multiplier` of 0.6, so
five primitive steps of one agent's force may simply be too short to transmit.
The same caveat that retired K9's "no interaction" reading applies here, and is
why the verdict is "weak-interaction **control**", a role, not "no interaction",
a property.

**The obvious next question is horizon.** `horizon = 1` was inherited from the
Buzz Wire work, where the rigid joint transmits force immediately. A
force-superposition task on a heavy shared object plausibly needs several
blocks. Re-running the same benchmark at `horizon = 2, 3, 5` is cheap — the
banks store 25 primitive steps — and is the minimal test of whether these tasks
are weakly coupled or merely slow.

**Until that runs, no task is available for H0/H1/H2.** Buzz Wire remains the
only scenario with a measurable cross-agent effect, and it is the one whose
observation we already know is partially deficient.


