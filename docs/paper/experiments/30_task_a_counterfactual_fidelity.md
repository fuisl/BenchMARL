# T-A: counterfactual fidelity of the multi-agent latent transition

**Status: registered, not yet reported.** This note fixes the design, the
metrics and the decision rules **before** any result exists, following the A1.2
precedent (commits `1293046`, `8f6b9c6`). Updated: 2026-09-21.

Direction: [`../direction_2026-09-21.md`](../direction_2026-09-21.md).
Governance: [`../coding_rules.md`](../coding_rules.md),
[knowledge index](../../RESEARCH_KNOWLEDGE_INDEX.md) §10 update contract.

## The question

```math
\boxed{
\textbf{Does conditioning latent dynamics on other agents' hypothetical actions}
}
```
```math
\boxed{
\textbf{improve the fidelity of counterfactual interaction effects?}
}
```

Task A only. No planner, no closed-loop control, no objective selection. The
downstream question — whether better counterfactual prediction makes joint
planning better — is Task B and is deliberately not asked here.

## Arms

Three conditioning hypotheses at matched capacity on the Audit Gate A0
`lewm_reference` profile:

| Arm | Predictor | Cross block `∂ẑ^i/∂a^j`, `j≠i` |
|---|---|---|
| **H0** `independent` | `f(z_i,a_i)` | exactly 0, structurally |
| **H1** `joint` | `f(z_i,\mathbf Z,\mathbf A)` | learnable |
| **H2** `relational` | `f(z_i,a_i,\sum_{j\neq i}\phi(z_i,z_j,a_i,a_j))` | learnable |

H1 and H2 hold **identical information**. Any difference is inductive bias, so
H1 is the control that stops a relational result being read as an information
result. H0 is the structural floor.

Conditioner capacity is solved to a shared budget. Measured on this bank:
1,082,922 / 1,083,180 / 1,082,400 conditioner parameters (spread 0.07%) and
13,719,128 / 13,719,386 / 13,718,606 dynamics parameters (spread 0.006%).

**Profile.** Every arm uses `lewm_reference`, whose scheduler cadence, action
normalization, temporal context and SIGReg population were pinned to LeWM
`8edfeb33` and verified against vendored sources in
[`27_audit_gate_a0_reference_profile.md`](27_audit_gate_a0_reference_profile.md).
Historical `legacy_compact` results are **not** comparable to these and are not
being reproduced.

**Data.** Buzz Wire bank `outputs/buzz_wire_1196/data`. Both action regimes:
`correlated` restricts the joint-action support (held-out region recorded in the
manifest as `normalized a0_x*a1_x < 0`), `independent` covers it. That contrast
is M1 Row 2. Eight training seeds (4100–4107), root-level splits, 683 train /
108 validation / 117 test anchors over 128 root episodes.

**Task choice.** Buzz Wire, because its agents observe neither each other nor
the ball, so H0 is genuinely information-starved rather than handicapped by a
tunable knob; because its rigid joint is the one mechanism in this repository
with a replicated, above-floor cross-agent response (K1, two banks, eight seeds
each); and because it is the one task that supplies termination positives.

## T-A1 — the true interaction Jacobian and the measurement floor

**Runs before anything is trained. Simulator only, no learned model.**

The project's most expensive repeated error is reporting a model ordering on a
quantity below the instrument's resolution. Balance's true cross-agent response
was 9–70× below probe error and the ordering reported on it was noise (K8, now
retired); in experiment 14 swapping a linear probe for an MLP probe reversed the
winner. T-A1 measures the effect **first**, so T-A2 is confined to cells where an
answer is possible.

**Intervention.** From each restored held-out anchor, hold every action
coordinate at a sampled reference joint action `A`, then move one coordinate —
agent `i`'s axis `c` — to each end of its own range. The difference in the
simulator's next physical state is a finite-difference column of

```math
J_{\rm interaction}=
\begin{bmatrix}
\partial Y^A/\partial a^A & \partial Y^A/\partial a^B\\[3pt]
\partial Y^B/\partial a^A & \partial Y^B/\partial a^B
\end{bmatrix}
```

plus a shared row for the ball, which is nobody's private state. The step is the
full action range for every cell, so all blocks share one finite-difference step
and are directly comparable; the bank's historical `low + high - a` reflection on
agent 1's x is reported alongside as a consistency check only.

**Why the action is held constant for the whole horizon.** A Jacobian column is
evaluated at one action point. A block that reverted to the reference after five
primitive steps would measure a transient, not a column of `J`. Pinned by
`test_constant_plan_holds_the_intervened_coordinate_for_every_step`.

**Y and its scale.** Each body's `(pos_x, pos_y, vel_x, vel_y)` — the same
common physical target `physical_response.py` uses — divided by its
per-dimension standard deviation on **training** rows. One scale, shared by
T-A1 and every later model, so a magnitude here and a ratio there are on one
axis. Cross-model comparison in each model's own learned latent space is
forbidden (audit §5.3).

**Masking.** A step counts only while **both** branches are still live and every
step before it was live. Stored rows are zeroed after termination and their
difference would be an artifact of termination, not physics. Pinned by
`test_cumulative_valid_never_revives_a_terminated_anchor`.

**Uncertainty.** Bootstrap resampling **root episodes**, not anchors. Anchors
branch from shared source episodes and are not independent samples. Pinned by
`test_bootstrap_clusters_by_episode_not_by_anchor`.

**Preflight.** Bit-exact replay determinism (tolerance exactly 0.0) and a
task-config-versus-manifest identity check, so a drifted config or a
nondeterministic simulator cannot be read as an interaction.

**Grid.** 117 test anchors × 4 reference actions × (2 agents × 2 axes × 2
endpoints + reference + reflection), horizons 1–5 blocks of 5 primitive steps.

**Reported per cell:** mean scaled `|ΔY|` with a 95% episode-clustered interval,
median, p90, max, the fraction of anchors with effect above `1e-6`, and the live
anchor/episode count at that horizon. Plus the headline `cross/self` ratio and
the cross share of total response.

### T-A1 registered decision rule

Fixed before the run. `F_probe(m)` is arm `m`'s probe reconstruction error on
**true** encodings, measured in T-A2 on the same scaled `Y`.

| Condition | Verdict | Consequence |
|---|---|---|
| cell's mean true effect `≥ 3 × F_probe` for all arms | **usable** | T-A2 may report `E_CF` on it |
| `1 × F_probe ≤` effect `< 3 × F_probe` | **marginal** | report with the ratio printed beside it; no ordering claim |
| effect `< 1 × F_probe` | **unusable** | T-A2 must not report a model ordering on it |
| every cross cell unusable | **task rejected** | Buzz Wire cannot carry Task A; report and stop, do not switch metric |

The 3× resolution ratio is the convention experiments 14 and 16 already use. It
is set on that precedent, not on any observed T-A1 value.

**Cells are selected by this rule alone.** No cell may be added or dropped after
seeing a model result.

**Pipeline smoke, not evidence.** A 24-anchor / 1-reference / 2-block CPU run was
executed to confirm the code path, exactly as A0 step 4 did. Replay determinism
was exactly 0.0 on all three recorded quantities. It is a smoke check, it used a
truncated anchor set, and it must not be cited as a measurement or used to
justify any threshold above.

## T-A2 — counterfactual effect fidelity

**Stage 1 (training).** 3 kinds × 2 regimes × 8 seeds = 48 checkpoints under
`lewm_reference`. Produces checkpoints only; no counterfactual metric is computed
in this stage, so the T-A1 floor cannot leak into training.

**Stage 2 (evaluation).** On the usable cells only:

```math
\Delta Y = Y'(a^{j}_{\rm high})-Y'(a^{j}_{\rm low}),
\qquad
E_{\rm CF}=\frac{\lVert\Delta\hat Y-\Delta Y\rVert}{\lVert\Delta Y\rVert+\epsilon}
```

in the shared scaled physical coordinates. A model predicting no response scores
exactly `1`. `E_CF < 1` means the error is smaller than the effect being
resolved. H0 scores exactly `1` on every cross cell by construction and is the
floor, not a competitor.

**Probes.** Per-agent head (one agent's latent → that agent's motion) and a
global head for the shared ball, both as in `physical_response.py`. A
non-intervened agent's predicted state must depend only on its own latent, or the
probe could manufacture a response out of the intervened agent's latent. Both
probe families (ridge and MLP) are run, because experiment 14 found the family
can flip the winner; a result that survives only one family is reported as
probe-dependent.

**Reported per arm:** `E_CF` with episode-clustered intervals, the probe floor
on true encodings, the resolution ratio, and per-seed paired differences.

### T-A2 registered decision rule

Primary comparison is **H1 vs H0** and **H2 vs H1**, on `correlated`-trained
models evaluated on the held-out joint-action region, paired by seed.

| Condition | Verdict |
|---|---|
| H1 `E_CF` < H0 on ≥ 7/8 seeds, interval clear of zero | joint-action conditioning is **necessary** — the headline claim |
| additionally H2 < H1 on ≥ 7/8 seeds, interval clear of zero | relational structure adds **beyond information** |
| H2 ≈ H1 (interval contains zero) | report as **information, not architecture**; do not claim a relational advantage |
| H1 ≈ H0 | the latent model does not recover a cross-agent effect the simulator has; a **representation** failure, not a factorization one |
| all arms `E_CF ≥ 1` | no arm resolves the effect; report and go to §"if Task A fails" |

## T-A2b — counterfactual information localization (registered)

**Status: registered 2026-09-21, before Test A exists.** T-A2 scored the
composite `E → P → probe` and cannot say which stage lost the effect. T-A2b
localizes it. **This replaces T-A3 as the next experiment**; see "Why T-A3 is
not next" below.

Three tests, one target, one scale, one set of interventions — the same ones
T-A1 measured, regenerated from its seed and required to reproduce its recorded
cell means.

| Test | What it scores | Status |
|---|---|---|
| **A** current-latent sufficiency | `g(z_t, a^low, a^high) → ΔY`, fitted head, never rolls `P` forward | **new, the discriminator** |
| **B** true future latent | `probe(E(o_{t+1}))` differenced across branches | **already measured**: it is T-A2's `probe_floor`, **0.173–0.234** on cross — good |
| **C** predicted future latent | `probe(P(z_t, a))` differenced | **already measured**: T-A2 itself, **1.05–1.19** — bad |

Because B is good and C is bad, only two of the four diagnostic rows are live,
and **Test A alone decides between them**:

| A | B | C | Diagnosis |
|---|---|---|---|
| ~~bad~~ | ~~bad~~ | ~~bad~~ | ruled out — B is good |
| **bad** | good | bad | future observations encode the outcome, but `z_t` is **not counterfactually sufficient** |
| **good** | good | bad | `z_t` is sufficient; the **predictor** fails to use it |
| ~~good~~ | ~~good~~ | ~~good~~ | ruled out — C is bad |

### Test A's two controls are the experiment

A fitted head that scores well proves nothing on its own. Both controls share
the head's architecture, budget and selection procedure; only the input differs.

* `actions_only` — `g(a^low, a^high) → ΔY`. The **state-blind floor**. It can
  learn the average response to an intervention but nothing state-specific.
* `physical` — `g(s_t, a^low, a^high) → ΔY` on the recorded simulator state. The
  **information ceiling**, and model-independent, so it is fitted once.

Weight decay is selected on held-out **root episodes**, not rows: anchors from
one episode are correlated, and a row-wise split would let the head memorise an
episode and score on its siblings.

### T-A2b registered decision rule

Let `B` = `actions_only`, `P` = `physical`, `L` = `latent`, all cross-block
`E_CF`. Define the **recovery fraction**

```math
R=\frac{B-L}{B-P},
```

the share of the blind-to-ceiling gap the latent closes.

| Condition | Verdict | Next |
|---|---|---|
| `P` interval overlaps `B` | **diagnostic uninformative** | the effect is not predictable from the current state at this horizon; report and stop — do **not** read a null as an encoder result |
| `R ≥ 0.75` | `z_t` **is** counterfactually sufficient | the failure is the **predictor**; the encoder and JEPA objective are exonerated |
| `R ≤ 0.25` | `z_t` **is not** counterfactually sufficient | the failure is the **representation**; a direct result about the JEPA abstraction |
| `0.25 < R < 0.75` | **partial** | report the fraction; claim neither row |

Thresholds are set now, before Test A is computed. Test B and Test C are quoted
from T-A2 and are not re-derived to fit.

### Amendment, before any Test A result: the head must be shown adequate

A pipeline smoke on three checkpoints produced `physical` 0.864 against
`actions_only` 0.906 — a blind-to-ceiling gap of 0.04 with almost completely
overlapping intervals, which would have fired the "uninformative" branch above.
The likelier cause was the instrument: the first head was a single hidden layer
trained full-batch for 300 steps on unnormalized VMAS coordinates, which
underfits badly. Reporting that as "the effect is not predictable from the
current state" would have repeated experiment 14's error of charging probe
weakness to the thing being probed.

The head was therefore strengthened **before** any result was recorded — two
hidden layers, minibatched Adam for 400 epochs, inputs z-scored on train
statistics only — and two preconditions are added to the rule. **The decision
table above may not be read unless both pass:**

1. **Self-block control.** The same head, same inputs, must predict the *self*
   response well. T-A2 establishes that self-dynamics are well captured
   (`E_CF` ≈ 0.46–0.51), so a head that cannot predict the self block either is
   underfit and its cross number is uninformative about the input.
2. **Train error reported beside test.** A train cross-`E_CF` near the test
   value and near 1.0 means the head never fit, not that the input lacks the
   information.

If either precondition fails, the correct report is "the diagnostic did not
work", not a claim about `z_t`. The smoke numbers above are a pipeline check on
a superseded head and are **not** evidence.

### T-A2b-2 — the mechanism test, registered before T-A2b reports

If Test A returns `R ≈ 0` for every arm, the registered verdict is
"representation failure". That is a *localization*, not a mechanism, and the
obvious mechanism here is observability rather than the JEPA objective.

Buzz Wire's observation is exactly

```python
[agent.state.pos, agent.state.vel, agent.state.pos - goal.state.pos]
```

— six dimensions, verified against the installed `vmas==1.5.2` scenario source.
It contains **no ball, no linkage body and no partner**. The ball is rigidly
jointed to both agents and is the variable that mediates the cross-agent effect.
Two agent positions constrain the ball but do not determine it (two circles meet
in up to two points) and say nothing about its velocity. So a per-agent encoder
over these observations may be *structurally incapable* of counterfactual
sufficiency, independently of the encoder, the objective or the conditioner.

**Registered test.** Add one input condition to Test A:

```math
g(z_t \oplus s^{\rm ball}_t,\ a^{\rm low},\ a^{\rm high})\rightarrow \Delta Y
```

— the same head, same budget, same selection, on the latent **concatenated with
the recorded ball and linkage state**.

| Condition | Verdict |
|---|---|
| `latent ⊕ ball` reaches the `physical` ceiling | the deficit **is** the missing mediating state; the encoder and JEPA objective are exonerated, and the world token `z^G` in the planning direction is the indicated repair |
| `latent ⊕ ball` stays at the blind floor | the ball is not the missing ingredient; the encoder discards something else, and this becomes a genuine result about the JEPA abstraction |
| in between | report the recovery fraction; claim neither |

This distinction matters for what the paper says. "A latent world model was not
counterfactually sufficient" is a much weaker and less interesting claim than
either "because the observation omitted the mediating state, which a world token
fixes" or "even given the mediating state, the learned abstraction discarded it."

### Why T-A3 is not next

T-A3 was registered as conditional on an arm passing T-A2; none did. Beyond the
registration, ordering is the wrong question right now: asking whether responses
with `E_CF ≥ 1` rank interventions correctly risks exactly the failure mode the
audit catalogues — a downstream metric obscuring an unlocalized upstream defect.
T-A3 stays registered and stays gated.

## T-A4 — coverage × architecture interaction (registered, K22b)

**Status: design registered 2026-09-21, not implemented.** Promotes the
post-hoc K22b observation into a test with a controlled variable.

T-A2 found relational's advantage over joint 2.5–2.8× larger under restricted
coverage than under full coverage. That was **observed after looking at the
results**, across two regimes that were not designed as a coverage ladder, so it
is not a claim.

**Design.** Make coverage strength a controlled variable `C ∈ {0, 0.25, 0.5,
0.75, 1.0}`, where `C` is the fraction of the counterfactual joint-action region
(`normalized a0_x·a1_x < 0`, recorded in the bank manifest) excluded from
training. Train H1 and H2 at each level, seeds 4100–4107.

**The hypothesis is not `H2 < H1`.** It is

```math
\frac{\partial D(C)}{\partial C}<0,
\qquad D(C)=E_{\rm CF}^{H2}(C)-E_{\rm CF}^{H1}(C),
```

a **monotone trend**: as coverage is restricted, relational should increasingly
outperform unstructured joint conditioning.

**Required control.** Excluding a region shrinks the training set, and a smaller
set would degrade both arms and could manufacture a trend. Every level must be
resampled to equal transition count, and that count reported.

**Registered rule.** Spearman of `D(C)` against `C` across the five levels,
seeds clustered; a negative trend with an interval clear of zero supports K22b.
A significant `D` at a single level without a trend does **not**.

## T-A3 — counterfactual ordering

At a fixed held-out state, fix `a^A` and vary only `a^B` over `K` alternatives.
The simulator supplies the true effects; the model must order them.

**Metrics:** Spearman between true and predicted effect magnitude, pairwise
ordering accuracy, top-`k` recall. Reported per arm with episode-clustered
intervals and the rankable fraction — Spearman on a near-tied candidate set is
undefined or tie-dominated, which is a defect experiment 22 already had to
handle.

**Registered rule.** T-A3 is reported only for arms that pass T-A2. Ordering
quality on a model that cannot resolve the effect at all is not interpretable.
T-A3 is the last Task A step and is the natural handoff to Task B; it does not
by itself license a control run.

## What is withheld

* **No control.** No closed-loop run, no CEM evaluation, no objective selection.
* **No model selection on downstream performance.** Arms are compared only on the
  registered Task A metrics.
* **Test roots are spent here.** T-A1/T-A2/T-A3 use the bank's 117 test anchors.
  A later Task B control run needs freshly collected initial states, as the A1.2
  audit already required.
* **Branch C is deferred, not withdrawn.** A1.2 registered a sampler-matched G6a
  on `full32` and argued G6c should not wait. Both are Task B work on the
  structured surrogate; see [`../direction_2026-09-21.md`](../direction_2026-09-21.md) §9.

## If Task A fails

A negative result is a result, and it is a *better* result than another failed
control run. If no arm resolves the cross-agent effect that T-A1 proves the
simulator has, that is direct evidence for the representation half of the
question — the latent is not control-sufficient — and it converges with K17
(latent position lost at the first transition) and K18 (true-latent information
that the interface and rollout cannot use). Report it; do not change metric in
search of a positive.

## Artifacts and reproduction

| Step | Entry point | Launcher |
|---|---|---|
| T-A1 | `examples/world_model/interaction_jacobian.py` | `scripts/slurm/ta1_interaction_jacobian.sbatch` |
| T-A2 stage 1 | `examples/world_model/train.py --config-name world_model_reference` | `scripts/slurm/ta2_reference_baselines.sbatch` |
| T-A2 stage 2 | `examples/world_model/counterfactual_fidelity.py` | `scripts/slurm/ta2_counterfactual_fidelity.sbatch` |
| T-A3 | to be implemented | registered when T-A2 reports |

**T-A2 stage 2 implementation notes.** Interventions are regenerated from T-A1's
seed through T-A1's own helpers and must then reproduce its recorded per-cell
means, so the evaluation cannot drift away from the floor that licensed it.
`E_CF` is computed **separately per block** — self, cross, shared — because
self-dynamics dominate the response and every arm including H0 can represent
them, so a pooled number would be a self-dynamics claim wearing the name of an
interaction claim. `rolled_latent` dispatches on model profile: `lewm_reference`
refuses a one-frame rollout by design, so a restored anchor uses the
episode-start convention `model_input.PlanningContext` registers — the frame
repeated `history_size` times with zero past actions — rather than
reconstructing a history that does not exist.

Contracts: `test/test_world_model_interaction_jacobian.py`,
`test/test_world_model_reference_profile.py`,
`test/test_world_model_reference_context.py`. The first gates T-A1 inside its
allocation; the latter two gate T-A2 stage 1, so a reference claim that has
drifted from the vendored sources cannot produce checkpoints.

## Results

| Job | Step | Status |
|---|---|---|
| 1497 | T-A1 | **COMPLETE** |
| 1498 | T-A2 stage 1 | **COMPLETE** — 48/48 checkpoints |
| 1499 | T-A2 stage 2 | **failed at bash parse time**; produced nothing |
| 1500 | T-A2 stage 2 | **COMPLETE** — superseded by 1502 for labelling only |
| 1501 | T-A2 stage 2 | **COMPLETE** — superseded by 1502 for labelling only |
| 1502 | T-A2 stage 2 | **COMPLETE** — the reported run |

### T-A1 (job 1497): the cross blocks are large, and active everywhere

117 test anchors over **16 root episodes**, 4 reference actions, horizons 1–5
blocks. Replay determinism exactly `0.0` on next agent state, next package
state and reward. Artifacts: `outputs/ta1_interaction_jacobian_1497/`.

Scaled `|ΔY|` per unit action-range step, at one action block (5 primitive
steps), with 95% episode-clustered intervals:

| Block | Intervention → responder | mean | 95% CI | active |
|---|---|---:|---|---:|
| self | `a0` x → agent 0 | 3.969 | [3.640, 4.321] | 1.000 |
| **cross** | **`a0` x → agent 1** | **2.231** | [2.068, 2.362] | **1.000** |
| self | `a1` x → agent 1 | 4.094 | [3.801, 4.441] | 1.000 |
| **cross** | **`a1` x → agent 0** | **2.152** | [2.018, 2.273] | **1.000** |
| self | `a0` y → agent 0 | 7.005 | [6.887, 7.116] | 1.000 |
| cross | `a0` y → agent 1 | 0.815 | [0.699, 0.948] | 1.000 |
| self | `a1` y → agent 1 | 6.923 | [6.834, 6.998] | 1.000 |
| cross | `a1` y → agent 0 | 0.846 | [0.751, 0.961] | 1.000 |

**The off-diagonal is real and it is everywhere.** Every cross cell is active on
**100%** of anchors, and the two agents are near-symmetric (2.231 against 2.152
in x; 0.815 against 0.846 in y), which is what a shared rigid linkage should
produce and is a free correctness check on the measurement.

**The coupling is strongly axis-dependent.** In x, cross reaches 56% of self
(2.231 / 3.969). In y it reaches 12% (0.815 / 7.005). That is physically
coherent for a horizontal wire: lateral motion drags the partner through the
rigid links, vertical motion largely does not. **This is not something the task
description told us, and it is the kind of structure that a single
"is there interaction?" number would have hidden.**

**Cross-agent effects compound with horizon:**

| horizon (blocks) | mean self | mean cross | cross/self | cross share of total |
|---:|---:|---:|---:|---:|
| 1 | 5.498 | 1.511 | 0.275 | 0.216 |
| 2 | 5.095 | 1.817 | 0.357 | 0.263 |
| 3 | 3.936 | 1.886 | 0.479 | 0.324 |
| 4–5 | — | — | — | — |

Horizons 4 and 5 are **not measured**: under the maximal constant-action
intervention every anchor has terminated by then, and they are reported as NaN
with a zero count rather than as a silent zero.

The historical bank convention — reflecting agent 1's sampled x about the action
midpoint — gives 1.120 at h1, the same order as the endpoint measurement, so the
two conventions agree about the existence and scale of the effect.

#### Registered verdict

**K20 supported.** The cross blocks are far from zero, active on every anchor,
and above the probe floor by roughly 5x in the pilot measurement of that floor.
Buzz Wire can carry Task A, and **H0 is structurally misspecified here by a
large margin**: it sets to exactly zero a quantity that accounts for 21.6% of
the total measured response at one block and 32.4% at three.

T-A2 is authorized on the self and cross cells. The shared-body (ball and
linkage) cells are **not** authorized: the probe floor there is about 0.56
relative, a resolution ratio of 1.8x, which fails the registered 3x rule. That
exclusion was made by the rule, not by inspecting any model result.

#### Boundaries on this result

* **16 root episodes.** 117 anchors is not 117 independent samples; the
  intervals above rest on 16 clusters. This bounds precision, not direction.
* **One task, one bank.** Nothing here generalizes to Wheel, Dropout or
  Transport, where earlier probes found few or no active anchors (K9).
* **Endpoint interventions are maximal.** They make the blocks comparable and
  they are why deep horizons terminate. A smaller step would measure a more
  local derivative and survive longer.
* **This is the simulator, not a model.** T-A1 says the effect exists. It says
  nothing about whether any learned model recovers it.


#### Defect found and repaired

Job 1497 computed and wrote the complete Jacobian, then crashed formatting a
horizon-4 cell whose anchors had all terminated: the empty-group return path
omitted the anchor count the report reads. The repair touches only that path, so
every computed value in the job's JSON is unaffected and the artifact is kept
rather than regenerated. Pinned by
`test_bootstrap_reports_an_empty_cell_instead_of_crashing_the_report`.

### T-A2 (job 1502): the conditioned models do not resolve the effect — but not because they lack it

48 checkpoints, 3 kinds x 2 regimes x 8 seeds, both probe families, 13,424
scored anchor-cells per arm. Artifacts: `outputs/ta2_fidelity_1502/`.

Cross block, one action block, linear probe (MLP in parentheses where it
differs materially):

| Regime | Arm | `E_CF` | cosine | magnitude ratio | oracle per-anchor gain |
|---|---|---:|---:|---:|---:|
| correlated | **H0** independent | **1.0000** | +0.000 | 0.000 | 1.000 |
| correlated | **H1** joint | 1.1367 (1.1864) | +0.400 | 0.898 | 0.528 |
| correlated | **H2** relational | 1.0506 (1.0786) | +0.408 | 0.740 | 0.507 |
| independent | **H0** independent | **1.0000** | +0.000 | 0.000 | 1.000 |
| independent | **H1** joint | 1.0795 (1.1270) | +0.453 | 0.857 | 0.513 |
| independent | **H2** relational | 1.0482 (1.0834) | +0.449 | 0.762 | 0.506 |

Paired by seed, cross block:

| Comparison | linear | MLP |
|---|---|---|
| correlated, H1 vs H0 | +0.137, **0/8** seeds | +0.186, **0/8** |
| correlated, H2 vs H1 | −0.086, **8/8** | −0.108, **8/8** |
| independent, H1 vs H0 | +0.080, **0/8** | +0.127, **0/8** |
| independent, H2 vs H1 | −0.031, **6/8** | −0.044, **7/8** |

#### The measurement is valid

The cross-block probe floor is 0.173–0.234 relative, a resolution ratio of
**4.3–5.8x**, clearing the registered 3x rule in every cell. **Both probe
families agree on every ordering** — unlike experiment 14, where swapping
linear for MLP reversed the winner. So this is not the instrument.

The shared-body cells are **not** reported as a model ordering: their floor is
0.58–0.63, a ratio of about 1.6x, which fails the rule. Self blocks are
well captured by every arm (`E_CF` 0.46–0.51, cosine ~0.92) and do not
discriminate, which is the expected sanity result.

`independent` scores **exactly** 1.0000 with cosine 0.000 and magnitude 0.000 on
every cross cell, to machine precision, in both probe families. The structural
zero is confirmed empirically rather than assumed.

#### Registered verdict: the headline claim fails

**H1 does not beat H0. It loses on 0/8 seeds in every cell, under both probes**
— 32 of 32 seed-cells positive, with no exception. Every point estimate for a
conditioned arm is *above* 1, i.e. worse in squared error than predicting no
cross-agent response at all. Two registered rows fire — "H1 ≈ H0" and "all arms
`E_CF` ≥ 1" — so under the criterion fixed before the run, **no arm resolves the
cross-agent effect that T-A1 proved the simulator has.**

**Precision on what is and is not significant.** The absolute claim and the
comparative claim do not have the same strength, and they must not be quoted
interchangeably:

| Claim | Evidence | Strength |
|---|---|---|
| H1 never beats H0 | paired by seed, 0/8 in all four cells, both probes | **unambiguous** |
| a conditioned arm is *significantly* worse than predicting nothing | only `correlated / joint` has a 95% interval excluding 1.0 (linear [1.0198, 1.2498]; MLP [1.0562, 1.3139]) | **one cell of four** |

The other three cells straddle 1.0. With 16 root episodes the episode-clustered
absolute intervals are wide, while pairing by seed removes the between-seed and
between-episode variance that widens them — which is why the comparative result
is sharp and the absolute one is not. The registered falsification of K21 rests
on the paired test, which is the comparison the rule was written against, so the
verdict is unaffected. **"Conditioning is worse than predicting nothing" should
be stated as a point estimate, not as a significant effect.**

#### But the failure is gain, not absence of information

`E_CF` alone would say conditioning makes things worse. That reading is wrong,
and the decomposition registered alongside it is what shows why:

* cosine is **+0.39 to +0.45**, far from the 0.000 that an uninformed predictor
  produces. The conditioned arms genuinely detect the coupling.
* the magnitude ratio is **0.74–0.96** — roughly the right size.
* correcting the gain separately at each anchor would put `E_CF` at
  **0.51–0.55**, roughly halving the error against the zero baseline.

So the models carry substantial cross-agent information and emit a response of
approximately the correct magnitude that is **inconsistently directed across
states**. A response of the right size pointing the wrong way scores worse than
silence; that is why H0's structural zero wins a metric it cannot possibly
understand.

**The oracle bound is an oracle.** Its minimising gain differs at every anchor
and depends on the true response, so it is not deployable and it is *not*
evidence that one global rescaling would work. It bounds how much of the gap is
gain rather than direction. Whether any state-conditional calibration learnable
from data closes it is untested.

#### K22 gets support it was not expected to get

The registration recorded H2-beats-H1 as "open, and expected to be weak,"
because H1 and H2 hold identical information and `N=2` is fixed. Instead
**relational beats joint on the cross block in all four cells** — 8/8, 8/8, 6/8
and 7/8 seeds — with both probe families agreeing. Relational also predicts a
*smaller* cross response (0.74–0.83 against 0.86–0.96), which is what moves it
closer to 1 given similar cosine.

This is a genuine inductive-bias effect at fixed information and matched
capacity. It is bounded: both arms remain above 1, so the honest statement is
**relational is consistently less wrong, not that relational works.** It does
not license a relational architecture claim, which still needs
`N_train != N_test`.

#### The coverage contrast appears, where M4 said it did not

Relational's advantage over joint is **2.5–2.8x larger under restricted
joint-action coverage** than under full coverage:

| Probe | `correlated` (restricted) | `independent` (full) | ratio |
|---|---:|---:|---:|
| linear | −0.0862 (8/8 seeds) | −0.0312 (6/8) | 2.76x |
| MLP | −0.1078 (8/8) | −0.0436 (7/8) | 2.47x |

This is M1 Row 2's predicted contrast: relational structure should matter most
exactly where the data does not cover the joint actions being asked about. The
M4 note recorded that this contrast **failed to appear** — "the relational gain
is the same size in both regimes, so the coverage contrast the impact notes
predict does not appear" — and carried that as a caution against reading M4's
rollout advantage as interaction modelling.

On the repaired reference profile, measured on counterfactual response in common
physical coordinates rather than on rollout error, it appears. That is a
non-trivial change and it is the strongest available evidence that the
relational conditioner is doing interaction work rather than being better
conditioned.

Bounded: this is a difference of paired means across two regimes, not a
registered comparison — the T-A2 decision rule compared arms *within* a regime.
It is reported as an observation that a future registration should test, not as
a fired rule.

#### What this means for Task A

The result converges with K17 and K18 rather than contradicting them. The latent
is not empty of cross-agent structure — cosine ~0.4 is real signal — but what it
carries is not accurate enough, state by state, to serve as a counterfactual
model.

**Stated precisely:** the learned latent transition is **not counterfactually
faithful at the tested interface**, despite carrying measurable cross-agent
signal.

An earlier draft of this note called that "the representation half of the Task A
question answering in the negative." **That is withdrawn as premature.** T-A2
scores the composite

```math
E \;\rightarrow\; P \;\rightarrow\; \text{probe}
```

and cannot attribute the failure to any one of them. Assigning it to the encoder
is exactly the kind of unlocalized causal claim the audit exists to prevent —
and the decomposition in this very note, showing the models *do* carry
cross-agent structure, argues against it. The attribution question is what
[T-A2b](#t-a2b--counterfactual-information-localization-registered) is registered
to answer, and no representation claim should be made before it reports.

#### Boundaries

* One task, one bank, 16 root episodes, one horizon (one action block).
* `E_CF` pools the four intervention cells per block type; the axis asymmetry
  T-A1 found (56% of self in x, 12% in y) is not resolved per arm here.
* The oracle gain bound is an upper bound on what calibration could buy, not a
  method.
* No control, no planner, no Task B claim of any kind.
