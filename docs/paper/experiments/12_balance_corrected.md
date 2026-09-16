# Balance on a correctly collected bank: what changed, and what did not

2026-09-16. Job **1236** repeats job 1233's design exactly — same seeds, budgets,
sweep config and evaluators — on the collector fixed in `43432b1`. 1233's
heuristic branch was Transport's policy ([DEFECT.md](../../../outputs/balance_1233/DEFECT.md));
426 of its 1,620 anchors came from it and they were in the training split.

The two jobs are a like-for-like pair, so every number below is a direct
replacement rather than a new measurement.

**The bank is measurably healthier.** Balance's own policy drops the package far
less often, so episodes survive longer and more of the state space is reached:

| | 1233 (Transport's policy) | 1236 (Balance's policy) |
|---|---:|---:|
| Live heuristic steps | 7,706 | **12,678** |
| Anchors | 1,620 | **1,834** |
| Test anchors / interaction-active | 197 / 113 (57%) | 225 / 144 (64%) |
| Rankable plan states | 111/128 (87%) | **127/128 (99%)** |

---

## 1. Plan ranking: the joint advantage got stronger and spread to both regimes

Paired against `independent`, 8 seeds:

| regime | kind | 1233 Spearman | 1236 Spearman | 1233 regret | 1236 regret |
|---|---|---|---|---|---|
| correlated | joint | +0.0075 (6/8) | **+0.0331 (8/8)** | −0.197 (5/8) | **−1.275 (8/8)** |
| correlated | relational | −0.0017 (3/8) | −0.0140 (3/8) | +0.297 (4/8) | −0.196 (5/8) |
| independent | joint | +0.0863 (8/8) | **+0.0804 (8/8)** | −0.843 (8/8) | **−0.576 (8/8)** |
| independent | relational | +0.0135 (6/8) | +0.0279 (7/8) | −0.142 (4/8) | −0.174 (5/8) |

On the defective bank the correlated regime separated nothing. On the corrected
bank **`joint` wins both regimes, both metrics, 8/8 seeds**. The status report's
earlier hedge — "one regime on one bank" — no longer applies.

`relational` remains marginal, and is *negative* on correlated Spearman (3/8).

## 2. Intervention response: neither conditioned model beats not conditioning

Lower is better; 1.0 = no better than predicting zero response.

| regime | kind | 1233 | 1236 |
|---|---|---:|---:|
| correlated | joint | 1.696× | 1.233× |
| correlated | relational | 1.185× | **1.063×** |
| independent | joint | 2.105× | 1.457× |
| independent | relational | 1.267× | **1.113×** |

Correct data moves every cell toward 1.0 — roughly a third of the excess
disappears — and `relational` is closer to the floor than `joint` everywhere, as
on Buzz Wire. But **all four cells stay above 1.0**.

The paired test is unanimous and now replicated across two independently
collected banks: `joint` and `relational` are both *worse* than the
unconditioned model at predicting the intervention response, in **0/8 seeds
better**, every regime, both banks.

## 3. The dissociation this exposes

On the same 48 checkpoints, on the corrected bank:

| | `joint` vs `independent` |
|---|---|
| Plan ranking (Spearman and regret) | better, **8/8 seeds, both regimes** |
| Intervention-response prediction | worse, **0/8 seeds, both regimes** |

**Conditioning on other agents' actions makes decisions better and the
counterfactual mechanism worse.** That contradicts the project's own premise —
that better counterfactual prediction is what causes better ranking — and it is
the single most replicated finding in the Balance data.

It is consistent with the Decision-Metric Alignment preprint the review cites,
which reports good global ranking alongside near-zero elite correlations.

Two cautions before this is written as a result. The response metric lives in
each model's own latent space, while ranking uses a common simulator cost, so
these are not yet two measurements of one quantity — that is Gate 2. And the 128
plan-ranking states and 225 anchors come from a small number of root episodes.

## 4. Raw counterfactual error and stratified concentration

Unchanged in direction from 1233:

* raw `E_CF` on active anchors is **lowest for `joint`** (correlated 0.24793 vs
  relational 0.26405 vs independent 0.28832).
* relational's advantage over independent still **concentrates on
  interaction-active anchors**: −3.6% active vs −2.0% inactive (correlated, 8/8),
  −4.2% vs −1.6% (independent, 8/8).

So relational does something real and interaction-specific — it just is not what
decides plans on this task.

## 5. Horizon, corrected bank

| regime | predictor | h=1 | h=5 | h=10 | h=15 |
|---|---|---:|---:|---:|---:|
| correlated | independent | 0.2890 | 0.4017 | 0.4136 | 0.4214 |
| correlated | joint | **0.2261** | 0.3793 | 0.4032 | 0.4040 |
| correlated | relational | 0.2511 | **0.3782** | **0.3973** | **0.3970** |
| independent | independent | 0.4364 | 0.4600 | 0.4843 | 0.4703 |
| independent | joint | **0.2725** | **0.4430** | **0.4734** | **0.4516** |
| independent | relational | 0.3514 | 0.4502 | 0.4875 | 0.4798 |

Errors drop sharply against 1233 (h=15 correlated: 0.6523 → 0.4214). `joint` is
still best at h=1 in both regimes. This replaces §4 of
[`09_horizon_rollout.md`](09_horizon_rollout.md) for Balance.

## 6. What this settles

**Settled.** The Balance result is not an artefact of the source-policy defect.
Every direction survives correct data, and the plan-ranking result got stronger.
Balance genuinely favours the unrestricted `joint` predictor.

**Settled.** Two tasks now disagree, each replicated: Buzz Wire favours
`relational` (response ratios below 1.0, replicated on a holdout bank), Balance
favours `joint`. This is a real task-level difference, not seed noise.

**Not settled.** Why. The substitution hypothesis — relational compensates for
unobservable coupling, and stops paying once the coupling variable is observed —
fits, but Balance differs from Buzz Wire in physics, agent count, objective and
observability at once. Only a within-task observability intervention can separate
them, and [job 1223](../../../outputs/state_input_1223/) already ran one.

## Reproduce

```bash
BALANCE_DATA=... sbatch scripts/slurm/balance_repair.sbatch
diff <(cat outputs/balance_1233/plan_ranking.txt) \
     <(cat outputs/balance_repair_1236/plan_ranking.txt)
```
