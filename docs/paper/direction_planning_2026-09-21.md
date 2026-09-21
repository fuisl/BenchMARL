# Part II direction: the planning architecture and the causal chain to test

**Status:** current statement of intent for **Task B**. Extends, and does not
replace, [`direction_2026-09-21.md`](direction_2026-09-21.md), which governs
Task A and still puts Task A first. Under the
[knowledge index](../RESEARCH_KNOWLEDGE_INDEX.md) §2 reading order this is an
intent document: it never overrides a measured result.

Registered gates and decision rules:
[`experiments/31_planning_ladder.md`](experiments/31_planning_ladder.md).

## 1. What this document changes

The earlier direction split the project into Task A (learn the world model) and
Task B (plan with it), and deliberately left Task B undescribed. This describes
it: the architecture to build, the hypothesis chain to test, and the gate order.

The ambition is deliberately narrowed to something that can actually work while
staying clean enough to test:

```math
\boxed{
\text{joint latent world model}
+
\text{joint online planner}
+
\text{terminal value / task score}
+
\text{receding-horizon feedback}
}
```

with **relational structure inside the world model, not inside the planner.**
The planner stays centralized and unstructured; the factorization question lives
where Task A put it.

## 2. Architecture

**Structured tokens, not one flattened latent.**

```math
z^i_t=E(o^i_t),\qquad
z^G_t=E_G(o^{\rm shared}_t)\ \text{or}\ \operatorname{Pool}(z^1_t,\ldots,z^N_t),
\qquad
Z_t=\{z^G_t,z^1_t,\ldots,z^N_t\}.
```

A shared/world token matters here for a task-specific reason this project
already measured: Buzz Wire's ball is in **no agent's observation**, and
supplying it cut collisions from 0.94 to 0.17 (K5). `z^G` is where that state
belongs, rather than being smuggled into each agent's token.

**Interaction-aware transition.** For candidate joint action `a_t`:

```math
m_{i\leftarrow j}=\phi(z^i,z^j,a^i,a^j),
\qquad
c^i=\rho\!\left(z^i,a^i,\textstyle\sum_{j\neq i}m_{i\leftarrow j},z^G\right),
```
```math
\hat z^{i}_{t+1}=P(z^i,c^i),
\qquad
\hat z^{G}_{t+1}=P_G(z^G,\{z^i\},\mathbf a),
\qquad
F_\theta(Z_t,\mathbf a_t)\rightarrow \hat Z_{t+1}.
```

Independent per-agent models `f_i(z^i,a^i)` are **retained as a baseline and
rejected as the final architecture**, because T-A1 measured the cross terms they
set to exactly zero: `∂Y_A/∂a_B` and `∂Y_B/∂a_A` are active on 100% of held-out
anchors and account for 21.6% of total response at one block, rising to 32.4% at
three.

**The planner never infers intent.** It proposes both `a^A` and `a^B`, so the
decentralized partner-modelling problem stays out of scope, exactly as the Task A
direction argued.

## 3. Scoring: a learned task interface, not latent goal distance

Dynamics training stays LeWM-shaped, preserving the scientific link:

```math
\mathcal L_{\rm dyn}=\lVert \hat Z_{t+1}-Z_{t+1}\rVert^2+\lambda\,\mathrm{SIGReg}(Z).
```

But the planner is **not** forced to use `‖Ẑ_H − Z_g‖²`. This project already
falsified that route on its own terms: K11 records three failed
observation-space goal designs and one partial pass, and job 1229's goal oracle
collided in 90% of episodes and finished worse than random. Latent goal distance
is not a valid objective here.

Instead a learned reward head and terminal value:

```math
\hat r_t=R(\hat Z_t,\mathbf a_t,\hat Z_{t+1}),
\qquad
J(\mathbf A)=\sum_{h=0}^{H-1}\gamma^h\hat r_{t+h}+\gamma^H V(\hat Z_{t+H}).
```

**`R` and `V` are trained separately, on frozen or partially frozen latents.**
This is a scientific requirement, not a convenience: if reward gradients shape
the representation, the question "does the world model preserve counterfactual
structure?" becomes "did the objective turn the representation into a policy
representation?" An end-to-end variant is a later ablation, labelled as such.

```
 observations → shared encoder E → agent tokens + world token
                                        │
                              relational joint dynamics P
                                        │
                             imagined latent trajectory
                                   ┌────┴────┐
                                   ▼         ▼
                               reward R   terminal V
                                   └────┬────┘
                                        ▼
                                   plan score J
```

## 4. Control loop

Centralized over the joint trajectory `A = [a_t, …, a_{t+H−1}]`, executing the
first block only and re-observing:

```math
\mathbf A^*=\arg\max_{\mathbf A}J(\mathbf A),
\qquad \text{execute } \mathbf a^*_t,\ \text{then re-encode and replan.}
```

**`K=1` is not a preference, it is a repaired defect.** Executing five planned
blocks before re-observing gave about four feedback decisions per episode and
discarded the warm start; the cadence repair was worth 2.4x on Balance and all
of the failures on Buzz Wire (Gate 0 repairs, K4).

**CEM first, MPPI second.** CEM is the one component already exonerated: with
true dynamics it solves Buzz Wire 25/32 with zero collisions (K4). That makes
"the search was too weak" hard to invoke when a learned model fails. MPPI is then
a second optimizer on the **identical frozen** world model, which separates
optimizer aggressiveness from model error.

**Uncertainty, but only after the deterministic baseline.** An ensemble of
**complete joint models** `{F_{θ_m}}` — not one model per agent — scored
conservatively, `J_robust = mean_m J_m − β·std_m J_m`. The motivation is
measured, not hypothetical: A1.2 found the optimizer drives teacher-forced error
up 2.28x from CEM iteration 1 to 30, and full32's predicted-safest decile truly
collided 62% of the time while its collision head assigned those plans
probability 0.0001.

## 5. The hypothesis chain

The contribution is not the architecture; those ingredients exist in the
literature. It is the chain:

```math
\boxed{\text{interaction structure}\rightarrow\text{counterfactual fidelity}
\rightarrow\text{decision ranking}\rightarrow\text{joint control}}
```

| | Hypothesis | Current standing |
|---|---|---|
| **H1** | Interaction-conditioned dynamics represent true cross-agent interventions better than independent dynamics | **Not established.** T-A2: conditioned arms carry signal (cosine +0.39…+0.45) but do not resolve the effect (`E_CF` > 1) and lose to H0 on 0/8 seeds |
| **H2** | Under restricted joint-action coverage, `E_CF^rel < E_CF^joint` | **Observed post hoc (K22b), not tested.** Registered as T-A4 |
| **H3** | Across frozen models, lower `E_CF` ⇒ lower candidate-ranking error | **The missing arrow.** Never measured |
| **H4** | Better ranking / selected regret ⇒ better closed-loop return and success under identical MPC | **Falsified once already** (K3), on a pipeline whose upstream links were broken |

**H3 is the scientifically important one.** It is the claim that counterfactual
fidelity predicts planning quality *better than ordinary IID rollout error
does* — and this project already has the negative half of that comparison, since
one-step teacher-forced prediction does not separate the baselines at all (M4).

## 6. What counts as success

Not a rollout-MSE threshold; there is no universal value. The operational
definition is **selected regret**:

```math
R_{\rm sel}=J_{\text{oracle-best candidate}}-J_{\text{model-selected candidate}} .
```

A model does not need accurate trajectories. It needs ordering good enough to
pick a good plan. `R_sel → 0` means decision-correct even with imperfect state
prediction; closed-loop success is the final confirmation.

This also restores physical probes to their proper role: **diagnostic only.**
The hierarchy is latent prediction → counterfactual effect fidelity → candidate
ranking → selected regret → closed-loop control, and physical position readout
merely explains *why* an arrow fails.

## 7. The gate order, and the precondition on control

Registered in [`experiments/31_planning_ladder.md`](experiments/31_planning_ladder.md).

| Gate | Question | Status |
|---|---|---|
| **0** | Can this planner solve the task with a perfect model? | **Passed** — K4, 25/32 with zero collisions |
| **1** | Counterfactual world-model fidelity, including `E_CF(h)` | T-A1/T-A2 done at `h=1`; horizon sweep running (job 1505) |
| **2** | Fixed candidate ranking: ρ, top-k recall, selected regret | Not started; needs `R` and `V` |
| **3** | Optimizer-induced distribution shift across CEM/MPPI iterations | Not started |
| **4** | Closed-loop MPC on fresh roots | Not started |

**Hard precondition on Gates 2–4.** No significant control compute until at
least one model reaches `E_CF < 1` robustly on the interaction cells. Not
because 1 is a magic planning threshold, but because in this metric `E_CF = 1`
is literally the *predict-zero-interaction* baseline: it is not defensible to
ask a planner to exploit an interaction model that does not beat ignoring the
interaction. Attaching a planner now would most likely reproduce the
already-documented failure of an optimizer exploiting inaccurate counterfactual
responses.

T-A2b (job 1503) is what decides whether that precondition is reachable by
fixing the predictor or requires changing the representation.

## 8. The paper-strength result this is aiming at

Four models with **near-identical ordinary validation loss** but differing
counterfactual fidelity:

| Model | IID latent error | Counterfactual error | Rank ρ | Selected regret | Control |
|---|---|---|---|---|---|
| independent | similar | poor | poor | high | poor |
| joint | similar | medium | medium | medium | medium |
| relational | similar | better | better | lower | better |
| relational + CF coverage | similar | best | best | lowest | best |

That pattern would support:

```math
\boxed{
\begin{array}{c}
\textbf{Planning succeeds when the world model preserves counterfactual}\\
\textbf{joint-action structure, which ordinary prediction loss does not reveal.}
\end{array}}
```

**The first column is already half-established** — M4 found one-step
teacher-forced prediction does not separate the baselines — and the second
column is currently **not** cooperating: T-A2 has relational best among the
conditioned arms but still above the zero baseline. The table is a target, not a
finding, and the honest outcome may be that the chain breaks at H1.

## 9. Related work

Positioning is "which factorization is sufficient, and does counterfactual
fidelity predict planning quality," not "world models are new to MARL."

* **COMBO** — compositional multi-agent world model with online search and an
  intent tracker for partially observable egocentric agents. Cited here for
  compositional joint-action modelling plus online planning.
* **M3W** — latent world-model rollouts with predicted reward, a terminal
  critic, and a multi-agent MPPI planner. The closest published analogue of the
  architecture in §2–§4.
* **MAMBA**, **DIMA** — recorded in [`litreview.md`](litreview.md) as AAMAS 2022
  and NeurIPS 2025 respectively.

**Citation debt, unchanged and now larger.** The literature documents still
carry search-session markers instead of a bibliography. Venues stated in the
working discussion — COMBO at ICLR 2025, M3W at NeurIPS 2025 — are **recorded as
claimed and not yet verified against primary sources**, and M3W is newly
introduced here with no entry in `litreview.md` at all. Verify all of them, and
add M3W to the review, before any appears in a submission.
