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
