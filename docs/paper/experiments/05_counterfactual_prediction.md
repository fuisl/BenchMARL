# M5 link 1 — counterfactual prediction (claim C7)

2026-09-14. The paper's headline claim, measured for the first time.

**Result: negative. No baseline captures the cross-agent effect on Transport.
`relational` scores 1.002x the error of predicting no effect at all.**

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

## Reproduce

```bash
python -m examples.world_model.counterfactual_evaluation \
  outputs/interaction_control_1194/transport --data outputs/transport_data_1190
```
