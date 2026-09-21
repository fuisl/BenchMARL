# Direction, 2026-09-21: separate the world model from the planner

**Status:** current statement of intent. Supersedes [`direction.md`](direction.md)
as the description of what the project is trying to establish and in what order.
It does not retire any measurement; see §9 for exactly what changes and what is
preserved.

**Authority:** this is a direction document. Under the
[knowledge index](../RESEARCH_KNOWLEDGE_INDEX.md) §2 reading order, intent
documents rank below experiment notes. Nothing here overrides a measured result.

## 1. What changed

The project has been running one chain end to end:

```math
\text{interaction modelling}\rightarrow\text{prediction}\rightarrow\text{ranking}\rightarrow\text{control}
```

Every link was trained, measured and diagnosed inside a single pipeline in which
the world model, the readout, the planner and the objective could all fail, and
several of them did. The result is a claim ledger with two established
prediction claims (K1, K2, K16), four falsified control claims (K3, K13, K17,
K19), and a long list of results that had to be retired because the instrument,
not the model, produced them.

The restructure is to stop treating that chain as one experiment:

```math
\boxed{\textbf{Task A: learn the world model}}
\qquad
\boxed{\textbf{Task B: plan with it}}
```

with **separate success criteria**, and Task A first. Task A's object is a
counterfactual transition model

```math
p(Z_{t+1}\mid Z_t,\mathbf A_t),\qquad
Z_t=(z^1_t,\ldots,z^N_t),\quad
\mathbf A_t=(a^1_t,\ldots,a^N_t),
```

and its question is only:

> Given the current joint latent state and a **hypothetical** joint action, does
> the model predict the correct next latent state?

A planner is not required to ask that, and until recently it was the only way we
asked it.

## 2. Why this is a correction and not a retreat

The motivating argument is worth stating precisely, because it removes a
justification the project has been leaning on.

**Counterfactual reasoning does not by itself require a relational model.** If a
centralized coordinator knows `Z_t` and *chooses* the whole joint action, a
sufficiently correct transition model already answers counterfactual questions.
Two independent per-agent models

```math
\hat z^{A\prime}=f_A(z^A,a^A),\qquad
\hat z^{B\prime}=f_B(z^B,a^B)
```

can be queried at any `(a^A,a^B)` pair, including combinations never observed
together. "What if A goes left while B goes right?" is answerable. Multiplicity
of agents is not, on its own, a reason to model interaction.

The ensemble fails only under a specific condition:

```math
z^{A\prime}\not\perp (z^B,a^B)\mid (z^A,a^A).
```

That is the real multi-agent content of the problem, and it is an **empirical
property of the task**, not something a narrative about cooperation establishes.
This is precisely the property T-A1 now measures directly (§5).

## 3. Three hypotheses, and what actually separates them

| | Form | Information | Cross block `∂ẑ^A/∂a^B` |
|---|---|---|---|
| **H0** independent | `f(z_i,a_i)` | own only | **exactly 0 by construction** |
| **H1** joint | `f(z_i,\mathbf Z,\mathbf A)` | all | learnable |
| **H2** relational | `f(z_i,a_i,\sum_{j\neq i}\phi(z_i,z_j,a_i,a_j))` | all | learnable |

H1 and H2 receive **the same information**. Whatever separates them is inductive
bias, not access. This reframes the project's own Gate 0b outcome — joint and
relational both beat independent, relational was not uniformly better than joint
— from a disappointment into the expected result:

```math
\boxed{\text{interaction \emph{information} matters more than relational \emph{architecture}.}}
```

The claim ledger already says this. K1 has relational winning response prediction
on Buzz Wire; K7 has joint winning plan ranking on Balance; "general relational
advantage" is recorded as **not supported**. The restructure makes that the
expected finding rather than an anomaly.

**One caveat on H0 that must not be dropped.** "Independent" means the
*predictor* is independent, not that its input is free of information about the
partner. Where observations contain relative positions, `z^A` already carries
partner information and H0 is not a clean null. Buzz Wire is the good case: its
agents observe neither each other nor the ball, so H0 is genuinely
information-starved there. Dropout is the bad case, and the protocol note
already warns against calling its whole observation transition independent.

## 4. Centralized and decentralized counterfactuals are different problems

**Centralized (what we run).** CEM samples `(a^A,a^B)` jointly. The hypothetical
partner action is *supplied by the planner*. There is no hidden intention
anywhere in the loop.

```math
F(Z_t,a^A,a^B)\rightarrow \hat Z_{t+1}.
```

**Decentralized (what we do not run).** Agent A chooses only `a^A` and must
marginalize:

```math
p(z^{A\prime}\mid z^A,a^A)=\sum_{a^B}p(z^{A\prime}\mid Z_t,a^A,a^B)\,p(a^B\mid Z_t).
```

Only here is partner-intent modelling literally necessary. COMBO is the clear
example of that setting: partially observable egocentric agents, an explicit
intent tracker predicting others' actions, feeding a compositional world model
and tree search.

**Consequence for our writing.** The relational conditioner must no longer be
motivated by "A must infer what B intends." That is false for our setup. The
correct motivation is:

> When evaluating a hypothetical joint action, predicting A's future may require
> conditioning on B's state and hypothetical action, because their dynamics
> interact.

Mixing in intent inference would add partial observability, partner modelling,
world modelling and planning at once. It stays out of scope.

## 5. Task A's success criteria: three levels, not an RMSE threshold

There is no universal "good" latent RMSE, because latent scale is arbitrary and
every model learns its own coordinates. The ladder is:

**Level 1 — predictive fidelity.** `‖ẑ_{t+1}-z_{t+1}‖²` on held-out data.
Necessary, weak, and already known not to separate the baselines one step ahead
(M4: one-step teacher forcing does not distinguish the three kinds).

**Level 2 — counterfactual effect fidelity.** For an intervention pair
`a^B, a^{B\prime}` at a fixed state:

```math
\Delta z^A = z^{A\prime}(a^{B\prime})-z^{A\prime}(a^B),
\qquad
E_{\rm CF}=\frac{\lVert\Delta\hat z-\Delta z\rVert}{\lVert\Delta z\rVert+\epsilon}.
```

`E_CF < 1` means the prediction error is smaller than the effect being resolved.
A model predicting no response scores exactly 1. This is the criterion Task A is
graded on.

This is already implemented. `physical_response.py` computes exactly this ratio
in common physical coordinates, with the scale fitted on training data and
shared across models, so H1 and H2 are not compared in two different learned
coordinate systems.

**Level 3 — counterfactual ordering.** Given `a^{B,1},\ldots,a^{B,K}` at one
state, does the model rank their effects correctly? Spearman, pairwise ordering
accuracy, top-`k` recall. This connects Task A to Task B without running a
planner.

## 6. The floor requirement (a necessary addition)

`E_CF < 1` is the right criterion but **it does not by itself fix the failure it
is meant to fix.** If the probe's reconstruction error on *true* next latents
already exceeds `‖Δz‖`, then `E_CF > 1` for every model, including a perfect
dynamics model. The number would then measure the instrument.

This is not hypothetical. It is how K8 was retired: Balance's true cross-agent
response sat 9–70× below probe error, and the model ordering reported on it was
noise. Experiment 14 also found that swapping a linear probe for an MLP probe
*reversed the winner*.

So every Task A result carries three numbers, not one:

1. the **true** effect size `‖Δz‖` in common physical coordinates;
2. the **probe floor** — reconstruction error on true encodings, per model;
3. `E_CF` itself, reported only on cells where (1) clears (2).

**T-A1 exists to establish (1) before any model is compared,** and it is run on
the simulator alone so it cannot be contaminated by a model or selected after
seeing one.

## 7. Why JEPA, properly stated

"JEPA is compute-efficient" is true and is not the reason to use it.

A pixel-generative world model must predict textures, shadows, lighting, sensor
noise and irrelevant background motion. Control cares about configuration,
contact, geometry, motion and affordances. A JEPA replaces

```math
(o_t,a_t)\rightarrow\hat o_{t+1}
\qquad\text{with}\qquad
(E(o_t),a_t)\rightarrow E(o_{t+1}),
```

so the claim is **abstraction**: `E` discards unpredictable nuisance while
keeping predictable, decision-relevant structure. The compute saving is a
consequence of that, not the motivation.

**And this is where the project has something real to say.** Nothing in
`L_pred + λ·SIGReg` guarantees the representation keeps the distinctions control
needs. Two states that look and predict alike under behaviour data can be mapped
together, `E(s_1)\approx E(s_2)`, yet diverge under a rare counterfactual action
`a^\dagger` the planner will happily propose. That is the tension between

```math
\boxed{\text{predictive abstraction}}
\quad\text{and}\quad
\boxed{\text{control-sufficient abstraction}}.
```

We are not speculating about this tension — **we have measured it.** K18 is
exactly this statement: useful task information in a latent does not guarantee
the planning interface and recursive rollout can use it. True-latent progress
ranks plans well; the reward interface does not; recursive rollout demotes the
known-good plan. K17 adds that the latent transition loses agent position at the
very first step, while a direct model on the same target succeeds (K16).

That upgrades the research question from

> Does relational LeWM predict physics better?

to

```math
\boxed{
\begin{array}{c}
\textbf{Can a latent predictive world model learn a}\\
\textbf{control-sufficient representation of multi-agent}\\
\textbf{counterfactual dynamics?}
\end{array}}
```

with two sub-questions: a **representation** question (does `Z` retain the
distinctions needed to predict intervention effects?) and a **factorization**
question (given `Z`, how should interactions be represented?).

## 8. What relational has to do to earn its place

With `N=2` fixed, a joint MLP can represent essentially anything the relational
model can, so an architecture claim needs more than a two-agent effect size.
Relational structure should win on one of:

* data efficiency;
* generalization to unseen joint actions;
* generalization to a different number of agents;
* permutation equivariance;
* transfer to a changed team composition.

Only the first two are reachable on the current Buzz Wire bank. Until
`N_train ≠ N_test` is tested, the scientifically safe claim is the weaker one:

```math
\boxed{\text{joint-action conditioning is necessary for interacting dynamics.}}
```

Task A is designed to support that claim honestly, and to say so if the
relational arm adds nothing on top of it.

## 9. What this changes, preserves, and defers

**Changed.** Task A is evaluated on counterfactual intervention fidelity and
ordering, not on average held-out rollout error and not on downstream control.
Task B is frozen out of Task A entirely: no model, seed, regime or objective may
be selected on closed-loop performance.

**Preserved.** Every claim in the ledger keeps its status. The gate protocol,
the episode-clustered uncertainty rule, the common-physical-coordinate rule, the
probe-floor rule and the update contract all still apply — §6 strengthens them
rather than relaxing them.

**Deferred, not retired.** [A1.2](experiments/29_a1_2_full_state_tail_localization.md)
registered **branch C**: a corrected, sampler-matched G6a on `full32`, and a G6c
calibration run that A1.2 argued should not wait. Both are Task B work on the
structured surrogate. They are deferred by this direction change and remain
registered; A1.2's branch registration is not withdrawn, and if Task A succeeds
they are the natural Task B entry point. The G6a implementation is already
repaired and unrun.

**Out of scope, unchanged.** Foundation-model initialization, visual inputs,
heterogeneous roles, uncertainty modelling, decentralized execution,
communication learning, MCTS, policy distillation.

## 10. Related work anchors

Our question is well connected to established model-based MARL problems, and the
positioning is "which factorization is sufficient," not "world models are new to
MARL."

* **COMBO** — compositional multi-agent world model with an intent tracker and
  tree search, for partially observable egocentric agents. The canonical example
  of the *decentralized* counterfactual we explicitly do not attempt.
  *(Venue recorded as ICLR 2025 from the working discussion; not yet verified —
  see the citation debt below.)*
* **DIMA** — recorded in [`litreview.md`](litreview.md) as NeurIPS 2025.
  Motivated by the same modelling difficulty from the other side: joint action
  spaces grow fast and interactions create structured dependencies.
* **MAMBA** — recorded in [`litreview.md`](litreview.md) as AAMAS 2022.
  Per-agent world models plus communication and imagined rollouts.

**Citation debt.** The literature documents still carry search-session markers
(`<!-- cite: turn... -->`) instead of a bibliography, and the knowledge index
lists converting them as required editorial cleanup. Venue claims in this
section inherit that status: MAMBA and DIMA are as recorded in `litreview.md`;
COMBO is newly introduced here and is **unverified**. Verify all three against
primary sources before any of them appears in a submission.

## 11. The project map

```math
\boxed{
\begin{array}{c}
\textbf{PART I — WORLD MODEL}\\[3pt]
(o_i)\xrightarrow{E}(z_i)\qquad (Z,\mathbf A)\xrightarrow{F_\theta}\hat Z'\\[3pt]
\text{H0 independent vs H1 joint vs H2 relational}\\
\downarrow\\
\text{true effect size and probe floor (T-A1)}\\
\text{counterfactual intervention fidelity (T-A2)}\\
\text{counterfactual ordering (T-A3)}\\[10pt]
\hline\\[-6pt]
\textbf{PART II — PLANNING}\\[3pt]
\mathbf A^{1:K}\xrightarrow{F_\theta}\hat Z^{1:K}\xrightarrow{J}\text{score}\rightarrow\mathbf A^*\\
\downarrow\\
\text{ranking / selected regret / control}
\end{array}}
```

Part I comes first. The registered design and decision rules are in
[`experiments/30_task_a_counterfactual_fidelity.md`](experiments/30_task_a_counterfactual_fidelity.md).
