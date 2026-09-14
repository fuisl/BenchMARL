# Paper outline and evidence map — AAMAS 2027

Target: AAMAS 2027 Research Paper Track, Hanoi, 3–7 May 2027.
Abstract deadline **1 Oct 2026**; full paper **8 Oct 2026**.
Limits: 8 pages body + unlimited references, double-blind, LaTeX mandatory.
Template already built and compiling at `../../../aamas2027/`.

Produced by `/ars-outline` (academic-paper `outline-only` mode: Phase 0 intake →
Phase 1 literature → Phase 2 structure). No draft prose. Sources are
[direction.md](direction.md), [the proposal](multi_agent_latent_mpc_proposal.md),
[impact notes](multi_agent_world_model_impact_notes.md),
[litreview.md](litreview.md), [centralized-cem-mpc.md](centralized-cem-mpc.md),
[experiment_plan.md](experiment_plan.md), and the M2 evidence in
[experiments/](experiments/).

---

## 0. Working title and claim

**Title (primary candidate):** *Counterfactual Joint-Action Prediction for
Multi-Agent Latent MPC*

**One-sentence claim.** Multi-agent latent planning fails not because the world
model predicts logged trajectories poorly, but because it predicts poorly under
the counterfactual joint actions MPC actually queries; explicit relational
joint-action conditioning narrows that gap and the improvement propagates to plan
ranking and closed-loop control.

**RQ1 (primary).** Does interaction-aware joint-action prediction improve latent
plan ranking and closed-loop control in cooperative multi-agent MPC?

**RQ2 (secondary).** Does multi-step training matter more as agent interaction
increases?

---

## 1. Section-by-section outline

Word budget totals ≈ 6,500 words of body text, which is the realistic capacity of
8 pages in ACM `sigconf` two-column once figures and tables are placed. References
are unlimited and sit outside the budget.

### Abstract — 200 words

Four moves, in order: (a) multi-agent MPC needs a model that answers
*counterfactual* joint-action queries; (b) standard validation error does not
certify this, and the failure is specifically multi-agent because the quantity
that goes unseen is the *combination* of agents' actions; (c) we compare
independent, joint-concatenated, and relational latent predictors under matched
data, capacity, and planner, against a simulator oracle; (d) the headline finding
is the dissociation — comparable in-distribution error, different counterfactual
gap, and the difference propagates to ranking and control.

No numbers until §6 exists. Write this **last**.

### 1. Introduction — 850 words

| Move | Content |
|---|---|
| M1 | Latent world models plus online planning now work in single-agent control (PlaNet, Dreamer/DreamerV3, TD-MPC2, DINO-WM). Cooperative multi-agent control is the natural next target. |
| M2 | Existing multi-agent world models (MAMBA, MBVD, MAZero, CoDreamer, MARIE, DIMA) mostly *amortise* the model into a policy. MAZero is the exception that genuinely searches. Almost none are evaluated on the query MPC actually issues. |
| M3 | The gap: MPC deliberately proposes joint actions that never co-occurred in the behaviour data. Low logged error does not imply correct counterfactual ranking. This is the one thing that changes when you go from one agent to many — *actions affect other agents*. |
| M4 | Contributions (see §2 below). |
| M5 | Result preview + paper map. |

**Opening sentence must not be throat-clearing.** Start on the dissociation, not
on "Multi-agent reinforcement learning has attracted increasing attention."

### 2. Contributions — folded into §1, ~150 words of it

1. **Counterfactual generalisation as the evaluation target.** A protocol that
   separates in-distribution error $E_{\mathrm{ID}}$ from counterfactual error
   $E_{\mathrm{CF}}$ on *the same restored simulator states*, and reports the gap
   $G_{\mathrm{CF}} = E_{\mathrm{CF}} - E_{\mathrm{ID}}$.
2. **A minimal relational latent world model.** One architectural change — a
   pairwise interaction term — against two controls, with everything else held
   fixed.
3. **An empirical link from model structure to control**, traced through the
   four-stage chain, including where it breaks.
4. **An oracle-anchored measurement protocol** using exact VMAS snapshot/restore,
   so the learned models are scored against the true dynamics rather than against
   each other.

Claim 1 and claim 4 are the defensible novelty. Claim 2 is deliberately small and
should be *presented* as deliberately small.

### 3. Related Work — 550 words

Three short paragraphs, not a survey. The rule is that every cited work must be
positioned against the counterfactual question, not summarised.

- **Multi-agent world models.** MAMBA, MBVD, MABL, CoDreamer, MARIE, DIMA. These
  capture inter-agent dependence; the open point is that none report whether the
  model ranks *unseen* joint actions correctly.
- **Planning in learned latent models.** MuZero → MAZero; PlaNet/Dreamer/TD-MPC2
  as the single-agent planning lineage. MAZero is the closest prior work and must
  be engaged directly, not listed.
- **Evaluation of world models.** The tension between reconstruction/prediction
  metrics and decision-relevant metrics. This paragraph sets up §5.4.

**Explicitly disclaim:** we do not claim to introduce world models to MARL, nor
planning to MARL.

### 4. Problem Formulation — 700 words

- Cooperative Dec-POMDP with $N$ agents, continuous actions, shared reward.
- Per-agent latent $z^i_t$; joint latent $Z_t$; joint action
  $A_t = (a^1_t,\dots,a^N_t)$.
- The three predictor classes as *hypothesis classes over the same input*:
  independent $f(z^i,a^i)$; joint-concatenated $f(Z,A)$; relational
  $m_i = \sum_{j\neq i}\phi(z^i,z^j,a^i,a^j)$, $\hat z^{i}_{t+1} = f(z^i,a^i,m_i)$.
- **Definition of the counterfactual query.** Given state $S$ drawn from the data
  distribution and $\tilde A$ drawn from an intervention distribution that breaks
  the behaviour policy's action correlation, the planner needs
  $\hat F(S,\tilde A)$. Formalise $D_{\mathrm{intervention}}$ here — this is the
  paper's central definition and it must be tight.
- **Why joint-concatenated is the right control, not a strawman.** It has access
  to exactly the same information as the relational model. Any difference is
  attributable to inductive bias, not to information. Say this explicitly; a
  reviewer will otherwise assume an unfair comparison.

### 5. Method — 1,050 words

| Subsection | Words | Content |
|---|---:|---|
| 5.1 Latent representation | 200 | Encoder, per-agent latent, parameter sharing. Vector observations, no vision. |
| 5.2 Interaction-aware predictor | 300 | The pairwise term. Permutation equivariance. Why sum-pooling, not attention (capacity control). |
| 5.3 Training objective | 250 | Two-term loss $\mathcal{L}_{\text{pred}} + \lambda\mathcal{L}_{\text{SIGReg}}$. Anti-collapse rationale. One-step vs multi-step variants (this is the RQ2 knob). |
| 5.4 Centralised CEM-MPC | 300 | Factorised Gaussian over the joint action, elites, receding horizon, action blocking. Objective $J = -\sum_t\sum_i r_{i,t}$ through first termination. State that this replaces LeWM's latent-goal objective and requires reward prediction in the learned models. |

**Known dependency to resolve before drafting:** the learned models need a reward
head, because the planning objective is task reward, not latent goal distance.
This is implied by [experiment_plan.md](experiment_plan.md) §"What to do about
planning cost" but is not yet implemented anywhere. It belongs in 5.3, and it is a
real work item, not a writing detail.

### 6. Experimental Setup — 850 words

| Subsection | Words | Content |
|---|---:|---|
| 6.1 Environments | 200 | VMAS, vector observations. Transport as the primary counterfactual task (recombining pushes changes a shared object's motion); a weakly-coupled control task. Task defaults preserved. |
| 6.2 Datasets and coverage control | 250 | Full-coverage / correlated / cooperative-policy datasets. How joint-action combinations are held out while individual action coverage is retained. Episode-level splits. |
| 6.3 Oracle and snapshot/restore | 200 | Exact VMAS state restoration including episode clocks; bit-exact replay validation; oracle CEM-MPC as the dynamics reference. |
| 6.4 Metrics | 200 | $E_{\mathrm{ID}}$, $E_{\mathrm{CF}}$, $G_{\mathrm{CF}}$; Spearman $\rho_{\text{plan}}$; selected-plan regret; success/return; oracle gap. Matched CEM budgets across models. |

**Caveat that must appear in 6.4:** raw MSE across separately learned latent
spaces is not comparable. Headline comparisons use task-level metrics. This is
already flagged in M4 of the experiment plan and a reviewer will find it if we
don't.

### 7. Results — 1,650 words

Structured as the four-link chain, one subsection per link, each answering "does
the signal survive to here?"

| Subsection | Words | Figure/Table | The question |
|---|---:|---|---|
| 7.1 Logged prediction | 250 | T1 | Do the three models look the same under standard validation? *We want yes.* |
| 7.2 Counterfactual prediction | 450 | F1: $G_{\mathrm{CF}}$ vs. coverage | Does the gap separate them? *The paper's core figure.* |
| 7.3 Plan ranking | 450 | F2: $\rho_{\text{plan}}$, regret | Does the gap change which plan gets chosen? |
| 7.4 Closed-loop control | 350 | T2: success/return/oracle gap | Does it change control outcomes? |
| 7.5 Where the chain breaks | 150 | — | Honest accounting, including negative links. |

The **dissociation in 7.1 vs 7.2 is the result**. If 7.1 already separates the
models, the paper's framing weakens considerably and the story becomes "relational
models predict better", which is a much less interesting and much more crowded
claim. Design the training budget so 7.1 is genuinely matched.

### 8. Limitations — 400 words

Non-negotiable inclusions, each already identified in the planning docs:

- **Architecture cannot create missing information.** If the dataset never
  identifies an interaction, no predictor recovers it. State the boundary.
- Oracle MPC is a *dynamics* reference, not a globally optimal controller.
- Centralised planning; no decentralised execution, no communication learning.
- Vector observations only; no visual or foundation-model features.
- Limited task count and team sizes; team size follows task defaults.
- Seed count (target 5–10 for headline comparisons).

### 9. Conclusion — 250 words

Restate the dissociation, not the architecture.

### Mandatory back matter (outside the 8 pages where the venue allows)

Data availability, ethics declaration, CRediT author contributions, conflict of
interest, funding ("no funding to disclose"), AI-use statement.

---

## 2. Evidence map

Status legend: **HAVE** = evidence exists now · **IN FLIGHT** = submitted or
prepared · **NOT STARTED** = no implementation · **RISK** = on the critical path
for 8 Oct.

| # | Claim | § | Evidence needed | Status | Source |
|---|---|---|---|---|---|
| C1 | Exact counterfactual replay is possible in VMAS | 6.3 | Snapshot/restore reproduces trajectories; different actions give different outcomes | **HAVE** | `02_oracle_validation.md` — replay bit-exact on 6,000 candidates; selected-mean agrees to 1.91e-6 |
| C2 | Oracle CEM-MPC is a valid dynamics reference | 6.3, 7.4 | Oracle beats random with non-overlapping CIs; solves the task, not just shaping reward | **HAVE (Buzz Wire) · PARTIAL (Transport)** | Buzz Wire R=30: 11/20 goals. Transport: zero goals in 660 episode evaluations across 33 runs, reusing 20 development states. At 300 steps both H5/H10 pass the return gate on 3/3 seeds; success remains unvalidated. Keep reward and success distinct. |
| C3 | The planner's budget/horizon settings are justified | 5.4, 6.4 | H=1 vs H=5; K/R budget sensitivity | **PARTIAL** | Controlled Transport H5→H10 at C=1 raises mean return .515→.722 at 100 steps. Larger-budget effects vary. Comparisons against C=5 also change cadence. Contact/search explanations remain hypotheses; see audited M2 notes. Buzz Wire further ablations are paused. |
| C4 | Transport exhibits the cross-agent coupling the RQ needs | 6.1 | Intervention on one agent's push measurably changes shared-object motion from a restored state | **HAVE (pilot)** | Controlled contact fixture plus M3 job1190: package effects in 32/239 test anchors over 25 steps, 21/239 within five. Coverage remains source-dependent; see `03_datasets.md`. |
| C5 | Controlled-coverage datasets isolate action coverage | 6.2 | Manifests, coverage summaries, leakage checks | **HAVE (pilot)** | M3 job1190: 1,919 paired anchors, 96/16/16 root episode split, validated manifests/replay/reader. First transition/block has matched input states; later states depend on actions. |
| C6 | Three world models train reproducibly under matched budgets | 5.1–5.3, 7.1 | Matched capacity/latent/loss/data; latent variance healthy | **HAVE** | M4 job 1192: 3 baselines x 2 regimes x 3 seeds, dynamics parameters matched to 0.02% (3.815M), bit-exact checkpoint reload on all 18 runs, no collapse (rank 32.1, variance 0.898). One-step prediction does not separate the baselines; rollout does (relational -13%). Caveats: equal gain in both regimes, live non-interaction explanation pending the Dropout control, SIGReg near an unreachable floor |
| C7 | Models are comparable in-distribution but differ counterfactually | **7.1–7.2** | $E_{\mathrm{ID}}$ parity + $G_{\mathrm{CF}}$ separation | **MEASURED - NEGATIVE** | 05_counterfactual_prediction.md: in-distribution parity holds (one-step error does not separate the baselines), but the counterfactual half fails. On the non-intervened agents, relational predicts the cross-agent effect at 1.002x the no-response floor; the only significant gain is 0.2% of the effect size, and joint is worse than predicting nothing. Cause is upstream: 0/239 anchors show an effect at one primitive step, 33/239 at 25 |
| C8 | Counterfactual gap predicts plan-ranking quality | 7.3 | $\rho_{\text{plan}}$, selected-plan regret on shared candidate sets | **MEASURED - NEGATIVE** | M5 link 1 (05_plan_ranking.md): Spearman 0.01-0.13 for every baseline over 8 seeds; every paired interval spans zero bar one marginal case. M4's 12% rollout advantage does not reach ranking. Also blocked by the task: M2 candidate banks have ONE unique cost across 300 candidates at all 20 states, M3 anchors only 29% rankable. The counterfactual gap itself (C7) is still unmeasured, so the 'predicts' half is untested |
| C9 | Ranking quality predicts closed-loop control | 7.4 | Success/return + oracle gap at matched budgets | **NOT STARTED** · **RISK** | M5 |
| C10 | Benefit tracks measured cross-agent effect, not task identity | 7.5, 8 | Same comparison across ≥2 interaction mechanisms + weak-coupling control | **PARTIAL** | M4 job 1194: relational rollout advantage is 8/8 seeds on Transport (CI clear of zero) and absent on the Dropout weak-interaction control (3/8, CI spans zero). Bounded -- Dropout's correlated CI still contains the Transport-sized effect, and Dropout differs in more than interaction. Prediction only; no ranking or control yet |
| C11 | Multi-step training matters more as interaction increases (RQ2) | 7.5 | One-step vs multi-step × interaction strength | **NOT STARTED** | M6 |
| C12 | Results are not single-seed artefacts | 7.1–7.4 | 5–10 seeds, uncertainty reported | **PARTIAL** | Job 1194 ran 8 seeds x 3 baselines x 2 regimes on two tasks with paired bootstrap intervals throughout, and job 1193 is retained as a case where 3 seeds gave a mean and median of opposite sign. Covers M4-level prediction claims only; C7-C9 have nothing to be seeded yet |

### What this map says plainly

C1 has replay evidence on both tasks; C2–C3 retain the limitations above. **C7, C8, and C9 — the
entire empirical core — do not exist.** Sections 7.1 through 7.4 are currently
unwritable, and they are ~1,650 words, four of the paper's six figures/tables, and
the whole reason the paper is interesting. The work between here and there is M3
(datasets) → M4 (three trained models + reward head) → M5 (main results) → M6
(seeds).

**Updated risk (2026-09-14, jobs 1182→1183):** Transport oracle MPC improves
return under longer horizons/episodes but has no observed goals. Sparse rewarding
contact and limited search are candidate explanations; PPO architecture results
do not diagnose CEM. Do not relabel progress as task success. M3 proceeds with the
user's Transport choice to measure data support and physical intervention effects;
this can support M4 prediction experiments while the closed-loop success claim
remains open. Learned models cannot recover interactions absent from their data,
and beating a finite-budget oracle controller would require a separate explanation.

That is the schedule risk, stated without softening: **17 days to the abstract
deadline, 24 to the paper**, with the reward-head dependency in §5.3 not yet
implemented.

---

## 3. Citations to verify before drafting

IRON RULE: every citation verified by DOI or search before it enters the draft. Two
need attention early because the method is built on them:

- **LeWM** — the two-term objective, the CEM defaults (K=300, R=30, 30 elites,
  H=5, block 5), and the decoder-free framing all derive from it, but
  [02_oracle_validation.md](experiments/02_oracle_validation.md) cites a GitHub
  repo, not a paper. Find the citable artefact or reframe the method section to
  stand on its own.
- **SIGReg** — same issue; $\mathcal{L}_{\text{SIGReg}}$ appears in the loss
  without a resolved citation in any planning doc.

The [litreview.md](litreview.md) citation counts carry `<!-- cite: turnNsearchM -->`
markers, which are search-session artefacts, not references. They must not survive
into the `.bib`.

---

## 4. Figure and table inventory

Six artefacts for 8 pages, which is already at the density limit.

| ID | Type | Content | Section | Status |
|---|---|---|---|---|
| F1 | Line/scatter | $G_{\mathrm{CF}}$ vs joint-action coverage, three models | 7.2 | Blocked on M5 |
| F2 | Scatter + bar | $\rho_{\text{plan}}$ and selected-plan regret | 7.3 | Blocked on M5 |
| F3 | Schematic | The three predictor classes side by side | 5.2 | **Draftable now** |
| T1 | Table | One-step / k-step prediction error | 7.1 | Blocked on M5 |
| T2 | Table | Closed-loop success, return, oracle gap | 7.4 | Blocked on M5 |
| T3 | Table | Task interaction mechanisms and interventions | 6.1 | **Draftable now** |

F3 and T3 can be built immediately and are worth building now — they are the two
that clarify the argument rather than report numbers.

---

## 5. Draftable today vs blocked

**Draftable now (≈3,200 words, half the paper):** §3 Related Work, §4 Problem
Formulation, §5 Method, §6 Experimental Setup (as protocol), §8 Limitations, plus
F3 and T3.

**Blocked on M5:** §7 in full, the abstract's result sentences, §1's M5 preview,
and §9.

The sensible order is to draft the blocked-independent half now so that when M5
lands, only §7 and the numbers need writing.
