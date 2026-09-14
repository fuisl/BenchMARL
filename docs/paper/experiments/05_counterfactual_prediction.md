# M5 link 1 — counterfactual prediction (claim C7)

2026-09-14. The paper's headline claim, measured for the first time, on two
tasks that differ in how much interaction their data contains.

**Result: negative on Transport, positive on Buzz Wire, and the difference
tracks measured coupling.** On Transport no baseline captures any of the
cross-agent effect (`relational` at 1.002x the no-response floor). On Buzz Wire,
whose rigid joint couples the agents structurally, `relational` captures **26%**
of the effect on 8/8 seeds in both regimes, and beats `joint` — which holds
identical information — on 7/8. That is claim C7 supported where the signal
exists, and claim C10's "benefit tracks measured cross-agent effect rather than
task identity" demonstrated by the contrast.

## What is measured

From one restored anchor state, the model predicts one action block twice: under
the logged joint action, and under a joint action where only agent 1's x
component is flipped. Both are scored against the simulator's true outcome, so

```
E_ID = error under the logged action
E_CF = error under the counterfactual action
G_CF = E_CF - E_ID
```

The measurement is taken on the **non-intervened agents**. Their own actions are
identical in both branches, so their next state can differ only through a
cross-agent effect. `independent` is bit-exactly unable to react to agent 1
(asserted in `test_world_model_models.py`), which gives the comparison a
structural floor.

One block, not one step: the M3 audit shows the simulator produces **no**
cross-agent effect after a single primitive step, so a one-step probe would test
coupling that does not yet exist.

Every model is scored on the same evaluation pairs regardless of its training
regime, so a correlated-trained model is queried in a region its data never
covered while an independent-trained model has seen it. That contrast is M1
Row 2. Models are the 48 Transport checkpoints from job 1194.

## G_CF is confounded; the intervention response is not

The first cut gave `G_CF` **negative for every baseline** on interaction-active
anchors (-0.049 to -0.064 relative), i.e. prediction was *better* under the
counterfactual action. That is not a model property: the flipped action leads to
a different, systematically easier target, so `E_CF - E_ID` mixes action novelty
with target difficulty because the two errors are measured against two different
targets.

The clean quantity compares how the prediction *moves* against how the simulator
*actually moved*:

```
response = || (pred_CF - pred_ID) - (truth_CF - truth_ID) ||^2
```

`independent` predicts a zero response by construction, so its score is exactly
the true effect size and forms a 1.000x floor. Below 1.0 means the model captured
part of the effect; above means it moved the wrong way.

The measurement validates on its own control: at interaction-**inactive**
anchors, where the truth does not move, `independent` returns
`G_CF = +0.00000` exactly.

## Result

31 of 239 test anchors are interaction-active, by both the audit label and the
observation actually moving. 8 seeds.

| Regime | Baseline | response | vs no-response floor |
|---|---|---:|---:|
| correlated | independent | 0.029963 | 1.000x |
| correlated | joint | 0.030565 | 1.020x |
| correlated | relational | 0.030025 | **1.002x** |
| independent | independent | 0.030444 | 1.000x |
| independent | joint | 0.030566 | 1.004x |
| independent | relational | 0.030379 | **0.998x** |

Paired against `independent`, negative = captured more of the effect:

| Regime | Baseline | mean | 95% CI | seeds better |
|---|---|---:|---:|---:|
| correlated | joint | +0.000601 | [+0.000282, +0.001040] | 0/8 |
| correlated | relational | +0.000062 | [-0.000043, +0.000190] | 2/8 |
| independent | joint | +0.000122 | [-0.000044, +0.000351] | 3/8 |
| independent | relational | -0.000065 | [-0.000125, -0.000011] | 7/8 |

**No baseline meaningfully predicts the cross-agent effect.** The only
statistically significant improvement — relational in the independent regime — is
**0.2% of the effect size**, which is not a difference any downstream stage could
use. `joint` is significantly *worse* than predicting no effect at all under
correlated actions.

The `G_CF` table is retained in the run output for completeness, but the
intervention response is the quantity that answers C7.

## What this settles

Four measurements now agree, and the interaction reading of M4's result does not
survive any of them:

| Measurement | Outcome |
|---|---|
| Dropout weak-interaction control | advantage absent (3/8), but Dropout differs in more than interaction |
| Stratified active/inactive | inconclusive; relative and absolute metrics disagree significantly |
| Plan ranking | no signal, Spearman 0.01-0.13 for every baseline |
| **Counterfactual response (C7)** | **no effect captured, 1.002x the no-response floor** |

M4's 12% multi-step rollout advantage is real and reproducible, but it is **not
interaction modelling**. It is a conditioning or optimisation effect, and the
paper cannot claim otherwise on this evidence.

## Why, and what follows

The cause is upstream of the models. Transport's data barely contains the
interaction: **0/239** test anchors show a cross-agent effect after one primitive
step, **22/239** after five, **33/239** after 25, and the coupling probe measured
a ratio of exactly 0.000 under every random behaviour policy. M3 anticipated this
case explicitly — "if the data never identifies an interaction, record that
limitation rather than expecting architecture alone to recover it."

This is a task property, not a model deficiency, and it is the same reward and
contact sparsity that produced 0 successes in 660 oracle evaluations and an
identically flat plan-cost landscape.

The controlled way to test it is a task where coupling is structural rather than
contingent on contact. Buzz Wire's rigid `Joint` ties both agents to the ball, so
every transition is coupled: a smoke collection shows an effect on **6/6 anchors
including at the first primitive step**, against Transport's 0/239. If relational
captures the effect there and not here, the benefit tracks *measured cross-agent
coupling* rather than task identity, which is claim C10 and the positive result
this line has not yet produced. Job 1196 runs that pipeline end to end.

Buzz Wire also terminates on wall contact, so its bank should supply the
termination positives that both Transport and Dropout lack.

## Buzz Wire — the same measurement where the interaction exists

Job 1196 ran M3 collection and M4 training end to end on Buzz Wire at the same
budgets, seeds and matched capacity as the Transport runs. Only the task differs.

The bank resolves all three of the scarcities that limited Transport:

| Property | Transport | Buzz Wire |
|---|---:|---:|
| anchors with a cross-agent effect | 33/239 (**0** at the first step) | **117/117** (all at the first step) |
| terminations in the bank | 0 | **252** (27.8% of snippets) |
| transitions with nonzero reward | 22.6% | **86%** |

113 of 117 test anchors are live through the block in both branches and all 113
are interaction-active. One correctness note: the collector zeroes actions once
an episode stops being live, and the two branches terminate at different steps,
so only anchors live in both carry a comparable intervention. Transport never
terminates inside a snippet, so this restriction changes nothing there and its
numbers are bit-identical; on Buzz Wire it is what makes the comparison well
posed.

| Regime | Baseline | response | vs no-response floor |
|---|---|---:|---:|
| correlated | independent | 0.260873 | 1.000x |
| correlated | joint | 0.211979 | 0.813x |
| correlated | relational | 0.191901 | **0.736x** |
| independent | independent | 0.253583 | 1.000x |
| independent | joint | 0.225405 | 0.889x |
| independent | relational | 0.188497 | **0.743x** |

Paired against `independent` (negative = captured more of the effect):

| Regime | Baseline | mean | 95% CI | seeds better |
|---|---|---:|---:|---:|
| correlated | joint | -0.048894 | [-0.064261, -0.031461] | 8/8 |
| correlated | relational | -0.068972 | [-0.085691, -0.048277] | 8/8 |
| independent | joint | -0.028178 | [-0.048864, -0.010234] | 8/8 |
| independent | relational | -0.065086 | [-0.090278, -0.036115] | 8/8 |

And the sharper comparison, `relational` against `joint`, which receive
**identical information** and differ only in inductive bias (permutation-
equivariant sum pooling against fixed-order concatenation):

| Regime | mean | 95% CI | seeds better |
|---|---:|---:|---:|
| correlated | -0.020078 | [-0.036831, -0.002760] | 7/8 |
| independent | -0.036908 | [-0.061684, -0.013023] | 7/8 |

So on Buzz Wire the ordering is `relational` < `joint` < `independent`, with both
gaps significant. The relational model is not merely using cross-agent
information; its structure uses that information better than a model with the
same access. That is exactly the argument §4 makes for why `joint` is a control
rather than a strawman, and it is the first evidence in this line that supports
it.

## What the contrast establishes

| | Transport | Buzz Wire |
|---|---:|---:|
| interaction density (anchors with an effect) | 14% | 100% |
| effect present after one primitive step | no | yes |
| relational captures | **0%** (1.002x) | **26%** (0.736x) |

The same three architectures, the same objective, the same budgets, capacity,
seeds and evaluation code. The only thing that changed is how much cross-agent
effect the data contains, and that is what decided whether relational structure
helped. Read together with the Dropout control, this is C10's claim measured on
three tasks spanning near-zero, sparse-contingent and structural coupling.

It also reframes the Transport result rather than retracting it. M4's 12%
rollout advantage there is real, reproducible and **not** interaction modelling;
the interaction claim needs a task whose data identifies the interaction, which
was M3's stated condition all along.

## Held-out confirmation (job 1201)

The numbers above come from a development bank whose test split was inspected
repeatedly while the evaluators were built. A fresh bank was therefore collected
with different state, split, action and branch seeds, trained on disjoint seeds,
and measured **once**, with nothing tuned against it.

| Regime | Baseline | development | **held-out** |
|---|---|---:|---:|
| correlated | joint | 0.813x | 0.675x |
| correlated | relational | 0.736x | **0.474x** |
| independent | joint | 0.889x | 0.503x |
| independent | relational | 0.743x | **0.495x** |

Paired against `independent`, all 8/8 seeds with intervals clear of zero
(relational, correlated: -0.179 [-0.195, -0.158]). The effect **confirms and
strengthens**: relational captures roughly half the cross-agent effect on data
nothing was fitted to, against a quarter on the development bank.

One seed base was rejected by the collector's replay check before a valid bank
was obtained, and that count is reported in the run log for the reason given
below.

## Limitation: replay is not universally bit-exact on Buzz Wire

Recorded because it bounds C1 on this task. The M3 collector verifies that
replaying the reference actions from the stored snapshots reproduces the stored
trajectory bit-exactly, and refuses to emit a bank that fails. The job-1196
development bank **passed** over 256 anchors, so every number above rests on
verified data.

A fresh collection at a different state seed **failed** that check (job 1200).
It is not a seed quirk: three separate seed sets pass at a 5-step smoke length
and fail at the full 25-step snippet length, which is the signature of numerical
divergence amplifying with rollout length. Buzz Wire integrates 15 physics
substeps under rigid joint constraints, and M3 had already noted that VMAS
carries width-dependent numerics.

The consequence is not that the measurement is wrong, but that a valid Buzz Wire
bank cannot be assumed -- it has to be checked, and some seeds will be rejected.
The held-out confirmation therefore tries several seed sets and keeps the first
bank passing the check, reporting how many were rejected. Rejecting on a
*validity* check cannot bias the C7 outcome, since nothing about the result is
consulted when choosing, but the count belongs in the record.

## Reproduce

```bash
python -m examples.world_model.counterfactual_evaluation \
  outputs/interaction_control_1194/transport --data outputs/transport_data_1190
python -m examples.world_model.counterfactual_evaluation \
  outputs/buzz_wire_1196/baselines --data outputs/buzz_wire_1196/data
```
