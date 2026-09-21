# Joint representation: world token, counterfactual training, and the gate before both

**Status: registered 2026-09-21, before any of it has run.** Decision rules are
fixed here ahead of results.

Direction: [`../direction_joint_representation_2026-09-21.md`](../direction_joint_representation_2026-09-21.md).
Prior results this builds on:
[`30_task_a_counterfactual_fidelity.md`](30_task_a_counterfactual_fidelity.md).

## Why these experiments exist

T-A2b (job 1503) established that the current latent is not counterfactually
sufficient and that the failure is **upstream of the predictor**. T-A2b-2 (job
1506) localized it: adding the omitted ball/linkage state to the same latent,
through the same head, recovers **86-88%** of the blind-to-ceiling gap on cross
`E_CF` and **96%** on cosine.

So the missing ingredient is an explicit representation of the jointly
controlled thing. Everything below follows from that.

## G0 - can history recover the mediating state? (the gate)

**Runs before any architecture is built.**

`z_G` must be learned from legitimate observations, not handed the simulator
state. The privileged state is what the *diagnostic* uses to establish a
ceiling; putting it in the model would answer an easier question. So the first
thing to know is whether the information is recoverable from what the agents can
actually see.

**Design.** Refit the T-A2b head with one added input condition: an
observation-**history** window over both agents,

```math
H_t=\bigl(o^{1}_{t-k:t},\ldots,o^{N}_{t-k:t},\ a^{1}_{t-k:t-1},\ldots,a^{N}_{t-k:t-1}\bigr),
```

against the same state-blind floor and physical ceiling, on the same
interventions, with the same head architecture, budget and episode-clustered
selection.

**Registered rule.** Recovery `R = (blind − history) / (blind − ceiling)` on
cross `E_CF`:

| Condition | Verdict | Consequence |
|---|---|---|
| `R ≥ 0.5` | the mediating state **is** inferable from legitimate observations | a learned `z_G` is feasible; proceed to H3/H4 |
| `R ≤ 0.2` | it is **not** | **no architecture over these observations can succeed.** Report that Buzz Wire's observation is inadequate for the counterfactual question - a task-design result, not a method result - and change the task or its observation before building anything |
| in between | partial | report the fraction; treat H3 as exploratory, not confirmatory |

**Ablation registered with it**, to say *which* omitted state matters rather
than only *that* some does: `latent ⊕ agent physical state` (rotation and
angular velocity, no ball) beside `latent ⊕ ball` and `latent ⊕ both`. If agent
physical state alone recovers the gap, the deficit is not the ball and §2 of the
direction needs revising.

### G0 partial result (jobs 1506, 1507): the encoder is exonerated, the ball is the variable

Shared conditions, identical head, interventions and scale throughout:

| Input | cross `E_CF` | `R` on `E_CF` | reading |
|---|---:|---:|---|
| `actions_only` (state-blind floor) | 0.8951 | 0.00 | — |
| **`observation_raw`** (raw 6-D, unencoded) | **0.8882** | **+0.02** | the observation itself carries **no** cross-agent information |
| `latent` | 0.879-0.895 | +0.02 to +0.04 | the encoder preserved what was there, which was nothing |
| **`latent ⊕ agent rot/ang-vel`** | **0.8793** | **+0.04** | other omitted physical state does **not** help |
| **`latent ⊕ ball/linkage`** | **0.517-0.523** | **+0.86 to +0.88** | **the ball is the missing variable, specifically** |
| `physical` (ceiling) | 0.4645 | 1.00 | — |

Three dissociations, and together they close the attribution:

1. **The JEPA abstraction is exonerated.** The raw unencoded observation scores
   0.8882, statistically on top of the 0.8951 blind floor. The encoder cannot be
   discarding cross-agent information, because the observation never had any.
   Had `observation_raw` beaten `latent`, this would have been a genuine result
   about lossy abstraction; it is not.
2. **It is not "any omitted state".** Agent rotation and angular velocity are
   also absent from the observation, and adding them moves nothing (0.8793
   against 0.8788). The registered ablation asked exactly this and the answer is
   clean.
3. **It is the mediating body.** Only the ball and linkage state moves the
   number, and it moves it most of the way to the ceiling.

`observation_raw` also predicts the **self** block well (0.2589, better than the
latent's 0.275 and far better than the blind floor's 0.399), so the head is
extracting what the observation does contain. This is a working instrument
reporting an absence, not an instrument failing.

#### What this settles, and what it does not

**Settled:** the counterfactual deficit is an observability property of the task,
not a property of the encoder, the objective, the conditioner or the predictor.
Every architectural arm we have trained was working from an input that cannot
answer the question.

**Not settled, and this is now the pivotal question:** all of the above is
**instantaneous**. The ball's position is constrained by two rigid joints to at
most two solutions given the agents' positions, and its velocity is constrained
by theirs — so **motion history may disambiguate what a single frame cannot.**
G0's registered history condition is therefore still the one that decides
whether a learned `z_G` is feasible at all, and it has not been run.

If history recovers the mediating state, `z_G` built from history is the
indicated architecture. If it does not, no architecture over these observations
can succeed and the honest conclusion is a task-design one.

### G0b result (job 1509): history helps, saturates around R ≈ 0.4, and never reaches the gate

Six window lengths, shared conditions only (they are model-independent), same
interventions, same scale, same head. Blind floor `E_CF` 0.8951 / cosine +0.459;
physical ceiling 0.4645 / +0.873.

| frames `k` | `E_CF` | 95% CI | `R` on `E_CF` | cosine | `R` on cosine |
|---:|---:|---|---:|---:|---:|
| 1 (single frame) | 0.8882 | [0.742, 1.007] | **0.016** | +0.741 | 0.681 |
| 2 | 0.8148 | [0.701, 0.913] | 0.187 | +0.764 | 0.736 |
| 3 | 0.7634 | [0.644, 0.865] | 0.306 | +0.750 | 0.702 |
| 5 | 0.7870 | [0.656, 0.900] | 0.251 | +0.748 | 0.698 |
| 8 | 0.7235 | [0.603, 0.826] | **0.399** | +0.775 | 0.762 |
| 12 | 0.7414 | [0.614, 0.848] | 0.357 | +0.777 | 0.766 |

#### Registered verdict: PARTIAL, and it does not improve over the tested windows

The best recovery is **`R = 0.399` at `k = 8`**, inside the registered
`0.2 < R < 0.5` band. Per the rule fixed before the run: **report the fraction
and treat H3 as exploratory, not confirmatory.**

The sweep adds something the single `k=3` measurement could not: recovery rises
with short history and then shows **no resolved further improvement over the
tested windows**. `R` climbs from 0.016 to roughly 0.3 by `k=3`, then 0.251,
0.399, 0.357 at `k = 5, 8, 12`. With 16 root episodes the intervals are wide and
those four values are **not resolvable from one another**, so this is an
*observed plateau over `k ≤ 12`*, not evidence that recoverability fundamentally
saturates — a longer window or a better-matched (e.g. recurrent) belief model
could do better. The largest point estimate is `R = 0.399`; no tested window
approaches the 0.5 gate, against **0.871** for handing over the ball state.

#### Direction is nearly free; scale is what history cannot buy

The two metrics separate sharply, and they say different things:

* **cosine is already at `R = 0.68` from a single frame** and rises only to 0.77.
  The *direction* of the cross-agent response is largely determined by the
  actions and the agents' own visible state.
* **`E_CF` recovery starts at 0.016** and saturates near 0.4. The
  state-dependent **magnitude** is the part history only partially recovers.

This is the same direction/scale dissociation T-A2b found in the latent, now
shown to be a property of **the observation stream itself** rather than of any
encoder. It also explains why every trained arm behaved identically: they were
all reading an input whose scale information is largely absent.

#### What this means for the architecture

Two rigid joints constrain the ball to at most two solutions given the agents'
positions, so motion *should* disambiguate it — and it partly does, which is why
`R` moves at all. But on this bank it recovers at most ~40% of the gap and stops
improving.

So a learned `z_G` over observation history would be built on at most ~40% of
the missing information recovered here. Combined with the constraint below —
that a `z_G` pooled from same-timestep latents is a re-parameterization and
cannot help at all — the supported reading is:

```math
\boxed{\begin{array}{c}\text{On Buzz Wire, the standard observation stream is the binding}\\
\text{\emph{empirical} constraint under the tested history windows.}\end{array}}
```

**Withdrawn as an overclaim (2026-09-21).** An earlier draft of this note wrote
"no architecture over the agents' observations can be counterfactually
sufficient." That is stronger than the experiment supports and is retracted.
`R_history < 0.5` over `k ≤ 12` with a flattened-history MLP does **not** establish

```math
I(H_\infty;\,s_{\rm ball})\approx 0 .
```

A recurrent or otherwise better-matched belief model, a longer window, or a
temporal statistic this diagnostic did not learn could extract more. What is
established is a statement about **this** function class and **these** windows:

> Within the tested 1-12 frame observation histories and diagnostic function
> class, standard Buzz Wire observations recover only a minority of the
> state-dependent cross-agent effect *magnitude*, whereas exposing the omitted
> ball/linkage state recovers most of it.

**The engineering decision is unchanged**, which is why the weaker claim is
sufficient: adding architectural structure cannot recover information absent
from the input, and we therefore do not treat further architectural complexity
over the original Buzz Wire observation as a promising confirmatory route. The
indicated intervention is at the **observation boundary** — `latent ⊕ ball`
recovers 86-88% immediately — not inside the network.

H3 and H4 remain registered. H3 is now explicitly **exploratory**: it may still
be worth knowing whether a world token extracts that ~40% better than a flat
history window does, but it cannot produce a counterfactually sufficient model on
this observation set, and it must not be presented as if it could.

### G0 final (job 1508): the localization is uniform across all 48 checkpoints

All three per-checkpoint conditions, 16 checkpoints per architecture:

| Input | cross `E_CF` | `R` on `E_CF` | cross cosine | `R` on cosine | self |
|---|---:|---:|---:|---:|---:|
| `actions_only` (blind floor) | 0.8951 | 0.000 | +0.459 | 0.000 | 0.399 |
| `observation_raw` | 0.8882 | 0.016 | +0.741 | 0.681 | 0.259 |
| `history` (3 frames, stride 1) | 0.7634 | 0.306 | +0.750 | 0.702 | 0.231 |
| latent — independent / joint / relational | 0.881 / 0.896 / 0.890 | 0.033 / −0.001 / 0.013 | +0.712 / +0.715 / +0.713 | 0.61 | 0.275 |
| **+ agent rot/ang-vel** | 0.877 / 0.882 / 0.880 | 0.043 / 0.031 / 0.034 | +0.725 / +0.727 / +0.726 | 0.64 | 0.272 |
| **+ ball/linkage** | **0.5245 / 0.5240 / 0.5204** | **0.861 / 0.862 / 0.870** | +0.858 / +0.856 / +0.858 | **0.96** | 0.140 |
| `physical` (ceiling) | 0.4645 | 1.000 | +0.873 | 1.000 | 0.119 |

**The heterogeneity check passes with no heterogeneity.** Across 48 checkpoints,
three architectures and two coverage regimes, the three conditions separate
identically: every latent arm within 0.015 of the blind floor, every
`+ agent-physical` arm within 0.02, every `+ ball` arm within 0.004 of the
others at `R ≈ 0.86`. No architecture, seed or regime deviates.

That is the strongest available statement of the localization: the deficit is a
property of **the observation**, invariant to everything we varied inside the
model.

### A constraint on `z_G` that follows from T-A2b, not from taste

**A world token computed from the same-timestep agent latents cannot help.**

```math
z^G_t=\operatorname{Pool}(z^1_t,\ldots,z^N_t)
```

is a deterministic function of `{z^i_t}`, so it carries no information those
latents do not already carry. T-A2b's head was given **all agents' latents
concatenated** — strictly more informative than any pooling of them — and
recovered `R ≈ 0` on cross `E_CF`. Adding a pooled token to that input is a
re-parameterization, and a re-parameterization cannot move an information-limited
measurement.

Therefore:

```math
\boxed{z^G \text{ must integrate HISTORY, or H3 is vacuous.}}
```

`z^G = G(\{z^i_t\}, H_t)` with `H_t` a window of past observations and executed
actions is the only form of the hypothesis that can be non-trivially true here.
This is why G0's history condition is not merely a gate on H3 but a precondition
for H3 being a meaningful experiment at all, and why a single-frame `z_G` arm is
**not** in the registered design.

## H3 - does an agent-neutral world token help?

**Blocked on G0 passing.**

```math
Z_t=\{z^1,\ldots,z^N,z^G\},\qquad z^G=G(\{z^i\},H_t)
```

with `z^G` learned, never given privileged state. Arms: relational with and
without `z^G`, matched capacity, same data, same seeds, same objective.

**Metric:** cross-block `E_CF` and cosine at `h=1,2,3`, the T-A2 protocol
unchanged so the numbers are comparable to jobs 1502 and 1505.

**Registered rule.** `E_CF` lower with `z^G` on at least 7/8 paired seeds with
an interval clear of zero, under **both** probe families. Anything less is
reported as unsupported, not as a trend.

## H4 - does counterfactual training help beyond nominal prediction?

**Separable from H3 by design**, so the direction's own prediction is testable.

```math
\mathcal L=\mathcal L_{\rm pred}+\lambda\,\mathrm{SIGReg}
+\mu\,\underbrace{\bigl\lVert[\hat Z'(A')-\hat Z'(A)]-[Z'(A')-Z'(A)]\bigr\rVert^2}_{\mathcal L_{\rm CF}}
```

Paired branches come from the existing intervention machinery applied to
**train-split** anchors, exactly as T-A2b already generates them, so no new data
collection and no test-split contact.

**The sharp claim.** H4 is supported only if `E_CF` improves **while ordinary
IID latent-prediction MSE barely changes**. Both must be reported together; an
`E_CF` gain bought by a large nominal-MSE gain is a different and much weaker
result.

**The direction's falsification condition, registered.** Because T-A2b-2 showed
the deficit is missing *input* rather than unextracted input, `L_CF` on the
current single-frame observation-only encoder is predicted **not** to close the
cross-agent gap. If it does, that reasoning is wrong and T-A2b-2's
interpretation must be re-examined. This is recorded now so the outcome cannot
be re-narrated later.

**Design:** 2x2 over `{z^G: on, off} x {L_CF: on, off}`, eight seeds per cell,
so H3 and H4 are not confounded and their interaction is visible.

**Registered `μ` handling.** `μ` is selected on the **validation** split by
counterfactual-effect error, never on test, and the selected value is reported.
A `μ` chosen on the reported metric would be circular.

## H5 - compositional shift

Last, because it needs new tasks or team sizes. Covers the coverage ladder
already registered as [T-A4](30_task_a_counterfactual_fidelity.md), plus agent
permutation, role reversal and `N_train ≠ N_test`. This is where a relational
architecture claim would finally be earned; with `N=2` fixed it cannot be.

## What is deliberately not done

* **No privileged state inside any model.** It belongs to the diagnostic only.
* **No intent inference.** The coordinator supplies the joint action.
* **No control compute** until a model reaches `E_CF < 1` on the interaction
  cells, per the [planning ladder](31_planning_ladder.md) precondition.
* **No architecture work at all** until G0 says the information is there.

## Results

*(none yet - registered ahead of its runs)*
