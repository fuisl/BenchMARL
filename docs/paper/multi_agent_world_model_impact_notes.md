# Increasing the Impact of the Multi-Agent Latent MPC Paper

## Core Framing

### Main Concern

If the paper is framed only as:

> Learn a transition function $F_\theta(S_t, A_t)$ on VMAS and use it for MPC.

the contribution can look weak.

VMAS has a known and relatively cheap simulator. In such an environment, one could simply use the simulator itself for rollout:

```math
S_{t+1} = F_{\text{VMAS}}(S_t, A_t)
```

instead of learning:

```math
\hat S_{t+1} = F_\theta(S_t, A_t).
```

So the paper should **not** argue that a learned world model is practically necessary for VMAS.

Instead, VMAS should be used as an **oracle environment** where we know the true transition and can evaluate whether a learned world model truly understands multi-agent interactions.

### Recommended Framing

The stronger research question is:

> **When can a learned multi-agent world model replace an unavailable dynamics oracle for counterfactual planning?**

VMAS is useful because we can compare:

```math
S_{t+1}^{\text{oracle}}
=
F_{\text{VMAS}}(S_t,A_t)
```

against

```math
\hat S_{t+1}
=
F_\theta(S_t,A_t).
```

This gives exact ground truth for arbitrary interventions.

The simulator is therefore not the competitor.

> **The simulator is the microscope.**

It lets us test whether the learned model actually understands what happens when agents take joint actions that were not seen during training.

## Research Focus: Counterfactual Generalisation

The current problem is:

> Does relational modelling improve counterfactual joint-action prediction?

A stronger version is:

> **Can relational factorisation enable compositional generalisation to unseen joint actions in multi-agent world models?**

This makes the paper less about “learning VMAS physics” and more about the kind of generalisation required for multi-agent planning.

### Why This Is Specifically Multi-Agent

Suppose the training data contains only:

```math
(a_1^A,a_2^A)
```

and

```math
(a_1^B,a_2^B).
```

But it never contains:

```math
(a_1^A,a_2^B).
```

A centralized black-box model can learn mappings such as:

```math
(S,(A,A)) \rightarrow S'
```

and

```math
(S,(B,B)) \rightarrow S''.
```

But this does not necessarily mean that it understands:

- what agent 1 contributes,
- what agent 2 contributes,
- how the two effects combine.

A relational model instead tries to learn reusable interaction terms:

```math
\Delta_{j\rightarrow i}
=
\phi(z_i,z_j,a_i,a_j)
```

and then compose them:

```math
z_{t+1}^{i}
=
f
\left(
z_t^i,
a_t^i,
\sum_{j\neq i}
\Delta_{j\rightarrow i}
\right).
```

The scientific hypothesis becomes:

```math
\boxed{
\text{factorise interaction}
\rightarrow
\text{recombine learned effects}
\rightarrow
\text{generalise to unseen joint actions}
}
```

This is a stronger claim than simply saying:

> “A GNN predicts VMAS better.”

## Dataset Design: Control Joint-Action Coverage

If the dataset is random and densely covers the action space, the problem may become too easy.

So the data should deliberately create different levels of joint-action coverage.

### Full-Coverage Dataset

Agents execute diverse and approximately independent actions.

The model sees many combinations.

Expected result:

```math
\text{Central WM}
\approx
\text{Relational WM}.
```

This is mainly a sanity check.

### Correlated Dataset

Agents' actions are highly correlated.

For example:

```math
a^2 \approx a^1.
```

The training data covers only a thin portion of the joint-action space.

At evaluation, break the correlation:

```math
a^1 \perp a^2.
```

Then test whether the learned model can predict the new combinations.

This directly probes counterfactual generalisation.

### Cooperative-Policy Dataset

Collect trajectories from a competent cooperative policy.

The dataset mainly contains coordinated behaviour.

MPC then proposes perturbed or alternative joint actions.

This gives a realistic planning mismatch:

```math
\text{behaviour-policy distribution}
\neq
\text{planner query distribution}.
```

The learned model may fit the logged trajectories while failing when the planner asks “what if the agents coordinated differently?”

## Evaluation Framework

### Counterfactual Generalisation Gap

A useful metric is:

```math
G_{\mathrm{CF}}
=
E_{\mathrm{CF}}
-
E_{\mathrm{ID}},
```

where

```math
E_{\mathrm{ID}}
=
\mathbb E_{(S,A)\sim D}
\left[
\|\hat F(S,A)-F(S,A)\|
\right]
```

is in-distribution prediction error, and

```math
E_{\mathrm{CF}}
=
\mathbb E_{(S,\tilde A)\sim D_{\mathrm{intervention}}}
\left[
\|\hat F(S,\tilde A)-F(S,\tilde A)\|
\right]
```

is error under counterfactual joint actions.

A strong result would be:

```math
E_{\mathrm{ID}}^{\text{central}}
\approx
E_{\mathrm{ID}}^{\text{relational}},
```

but

```math
G_{\mathrm{CF}}^{\text{relational}}
<
G_{\mathrm{CF}}^{\text{central}}.
```

Interpretation:

> Conventional validation error may look similar, but the relational world model generalises better to unseen joint-action combinations.

This is much more interesting than only reporting one-step MSE.

### From Counterfactual Prediction to Planning

The next level is to test whether better counterfactual prediction actually helps planning.

The intended chain is:

```math
\boxed{\text{Logged prediction}}
\downarrow
\boxed{\text{Counterfactual prediction}}
\downarrow
\boxed{\text{Plan ranking}}
\downarrow
\boxed{\text{Closed-loop control}}
```

The central question is where this chain breaks.

### Plan-Ranking Fidelity

MPC does not need perfect reconstruction of every future state.

It mainly needs to rank candidate action sequences correctly.

For two candidate joint plans $A_1$ and $A_2$, if the true environment says:

```math
J(A_1) > J(A_2),
```

then a useful world model should satisfy:

```math
\hat J(A_1) > \hat J(A_2).
```

Measure this using rank correlation, for example:

```math
\rho_{\text{plan}}
=
\operatorname{Spearman}
\left(
\hat J(A^{(k)}),
J_{\text{oracle}}(A^{(k)})
\right).
```

This directly tests whether the world model gives the planner the correct ordering over candidate futures.

### Simulator-Based MPC as the Oracle Baseline

Do not hide the fact that VMAS can simulate exact rollouts.

Use it as the strongest reference.

Compare:

```math
\boxed{\text{Oracle MPC}}
```

using exact VMAS dynamics, against:

```math
\boxed{\text{Independent WM + MPC}}
```

```math
\boxed{\text{Central WM + MPC}}
```

```math
\boxed{\text{Relational WM + MPC}}.
```

Then define an oracle planning gap:

```math
\Delta_{\mathrm{oracle}}
=
J_{\mathrm{oracle}}
-
J_{\mathrm{learned}}.
```

The practical question becomes:

> **How much of oracle planning performance can be recovered from offline interaction data?**

And more specifically:

> **Does relational factorisation close the oracle gap better when joint-action coverage is limited?**

This makes the use of VMAS scientifically justified.

## Optional Extension: Agent-Count Generalisation

A useful optional extension is:

```math
N_{\mathrm{train}}=2
```

and test on:

```math
N_{\mathrm{test}}\in\{2,3,4\}.
```

A concatenated model learns a fixed input:

```math
F(z_1,z_2,a_1,a_2).
```

A relational model learns a shared interaction mechanism:

```math
\phi(z_i,z_j,a_i,a_j).
```

If that interaction law transfers from:

```math
2\rightarrow3\rightarrow4
```

agents, this suggests compositional dynamics generalisation.

This is a higher-value extension than adding pixels or a foundation encoder.

It should remain optional unless the core counterfactual result is already strong.

## Scope Control

Do not increase scope by immediately adding:

- V-JEPA,
- DINO,
- image observations,
- uncertainty models,
- CTDE,
- learned communication,
- planner distillation,
- MCTS,
- large-scale benchmarks.

These make the project larger, but not necessarily deeper.

The better way to increase impact is to make the question more precise:

> **What kind of generalisation does multi-agent planning require from a learned world model?**

## Scientific Positioning

The paper should not be:

> **We introduce Relational MA-LeWM.**

Instead:

> **We study counterfactual generalisation in learned multi-agent world models for planning.**

The method is deliberately simple.

The scientific focus is:

```math
\text{Does interaction factorisation help a learned model generalise from observed coordinated actions to unseen joint interventions?}
```

## Title Options

| Style | Title | Subtitle |
|---|---|---|
| Clear and academic | **Counterfactual Generalization in Multi-Agent World Models** | *From Transition Prediction to Reliable Joint Planning* |
| More direct | **When Prediction Is Not Enough for Multi-Agent Planning** | *Counterfactual Generalization of Learned World Models* |
| More method-linked | **Counterfactual Joint-Action Generalization for Multi-Agent Latent MPC** | — |
| More memorable | **Can Multi-Agent World Models Answer “What If?”** | *Counterfactual Joint-Action Generalization for Latent Planning* |

## Revised Paper Story

The full story can be reduced to:

1. Multi-agent world models can fit observed transitions.
2. MPC queries action combinations outside the behaviour-policy distribution.
3. Standard prediction error does not tell us whether those counterfactual queries are reliable.
4. VMAS provides an exact transition oracle, allowing direct intervention-based evaluation.
5. We test whether relational factorisation improves counterfactual prediction.
6. We measure whether this improvement transfers to better plan ranking and control.

In one line:

```math
\boxed{
\text{interaction structure}
\rightarrow
\text{counterfactual generalisation}
\rightarrow
\text{plan fidelity}
\rightarrow
\text{control}
}
```

## Proposed Contributions

### 1. Counterfactual Generalisation as the Core World-Model Problem

We formalise the distinction between:

```math
\text{in-distribution transition prediction}
```

and

```math
\text{counterfactual joint-action prediction}.
```

The paper studies whether a world model that performs well on logged trajectories can still fail under the joint interventions generated by a planner.

A possible metric is:

```math
G_{\mathrm{CF}}
=
E_{\mathrm{CF}}
-
E_{\mathrm{ID}}.
```

### 2. An Oracle-Based Evaluation Protocol for Multi-Agent Planning

Use VMAS as an exact dynamics oracle to evaluate:

- one-step prediction,
- counterfactual prediction,
- plan-ranking fidelity,
- closed-loop MPC performance.

This creates a controlled way to diagnose learned multi-agent world models rather than only comparing reward.

### 3. Relational Factorisation for Unseen Joint-Action Composition

Test whether explicitly factorising cross-agent effects:

```math
\Delta_{j\rightarrow i}
=
\phi(z_i,z_j,a_i,a_j)
```

improves generalisation to unseen joint-action combinations compared with:

- independent per-agent dynamics,
- centralized concatenated dynamics.

The contribution is not “using a GNN.”

The question is whether **interaction structure provides the right inductive bias for counterfactual planning**.

### 4. A Link Between World-Model Error and Planning Error

Measure the full chain:

```math
\text{prediction}
\rightarrow
\text{counterfactual prediction}
\rightarrow
\text{plan ranking}
\rightarrow
\text{control}.
```

The goal is to determine which model errors actually matter for multi-agent MPC.

A strong result would show that models with similar logged prediction error can have substantially different counterfactual and planning performance.

### 5. Optional: Compositional Transfer Across Team Size

If time allows, test whether a relational interaction model trained with two agents can transfer to three or four agents without relearning the interaction law.

This would strengthen the interpretation that the model learns reusable multi-agent interaction structure rather than simply memorising joint states.

## Final Paper Claim

> **Multi-agent world models should not be evaluated only by how well they predict logged trajectories. Reliable planning requires counterfactual generalisation to unseen joint actions. Relational factorisation can reduce this generalisation gap, improve plan ranking, and recover more of the performance of an oracle dynamics planner under limited joint-action coverage.**
