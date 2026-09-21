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
| 1499 | T-A2 stage 2 | running |

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
