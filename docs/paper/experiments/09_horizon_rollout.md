# Rollout error against horizon, and what the context limit actually is

2026-09-16, **rewritten the same day**. The first version of this note reported a
cliff at h=6 and an ordering reversal beyond it. Both were artefacts of rolling
the model through a positional slot that never received a prediction gradient.
The corrected tables below replace them; the withdrawn claims are listed in §6.

Produced by [`examples/world_model/horizon_rollout.py`](../../../examples/world_model/horizon_rollout.py)
via [`scripts/slurm/horizon_rescore.sbatch`](../../../scripts/slurm/horizon_rescore.sbatch),
job **1235** (Level 1, one `2g.10gb` slice). Checkpoints are the existing ones
from jobs 1196 (Buzz Wire), 1217 (Transport), 1228 (Buzz Wire at λ=0.009) and
1233 (Balance) — 8 seeds × 3 predictors × 2 regimes each. Nothing was trained.

**One block = 5 primitive steps** (`action_block=5`), so h=5 is a 25-step
rollout, the depth CEM actually uses.

---

## 1. The trained context is five positions, not six

`ARPredictor.pos_embedding` is `(1, 6, 192)`, sized to the six-frame training
snippet. But `dynamics_losses` predicts from `latent[:, :-1]`: the last frame is
only ever a *target*, never a predictor input. **Five positions receive a
prediction gradient; the sixth receives none.** A backward pass on a trained
checkpoint gives nonzero norms for positions 0–4 and exactly `0.0` for position 5
([review evidence](../../../outputs/review_20260916/evidence.json)).

`MultiAgentWorldModel.rollout` grows its history without bound and indexes that
embedding, so it raises at **block 7**, not block 6 as first reported. Block 6
does not raise — it silently reads the untrained slot, whose embedding has only
ever been decayed.

That slot is the entire "cliff". The same Transport checkpoint, same actions,
same roots:

| window | h=5 | h=15 |
|---|---:|---:|
| 6 positions (untrained slot in play) | 0.0565 | 1.2719 |
| 5 positions (trained only) | 0.0565 | **0.0739** |

Everything below uses a sliding window over the **trained** positions only. Past
h=5 the model is still extrapolating — it has never been asked for a sixth
consecutive prediction during training — but it is extrapolating with weights
that were fitted, which is a different thing from indexing noise.

## 2. Transport

All 128 episodes are live through h=15 (Transport has no early-failure mode), so
nothing here is confounded by survivorship.

| regime | predictor | h=1 | h=2 | h=3 | h=5 | h=8 | h=10 | h=15 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| correlated | independent | 0.0176 | 0.0283 | 0.0323 | 0.0355 | 0.0411 | 0.0435 | 0.0526 |
| correlated | joint | 0.0176 | 0.0249 | 0.0272 | 0.0304 | 0.0357 | 0.0384 | 0.0474 |
| correlated | relational | **0.0171** | **0.0227** | **0.0252** | **0.0284** | **0.0325** | **0.0348** | **0.0430** |
| independent | independent | 0.0166 | 0.0279 | 0.0325 | 0.0358 | 0.0424 | 0.0441 | 0.0478 |
| independent | joint | 0.0177 | 0.0245 | 0.0279 | 0.0310 | 0.0360 | 0.0389 | 0.0440 |
| independent | relational | **0.0164** | **0.0215** | **0.0243** | **0.0272** | **0.0323** | **0.0346** | **0.0377** |

## 3. Buzz Wire — read with the survivor counts

Buzz Wire terminates on collision, so the population shrinks and the later
columns mix horizon with survivorship: the episodes still alive at h=15 are the
easy half.

```
episodes live:  h=1:128  h=2:127  h=3:124  h=5:109  h=8:80  h=10:71  h=15:49  h=20:0
```

| regime | predictor | h=1 | h=2 | h=3 | h=5 | h=8 | h=10 | h=15 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| correlated | independent | 0.1841 | 0.2893 | 0.3244 | 0.3586 | 0.3624 | 0.3549 | 0.4026 |
| correlated | joint | 0.1669 | 0.2197 | 0.2433 | 0.2821 | 0.2794 | 0.2872 | 0.3146 |
| correlated | relational | **0.1480** | **0.2016** | **0.2261** | **0.2611** | **0.2589** | **0.2666** | **0.2935** |
| independent | independent | 0.2456 | 0.3504 | 0.3828 | 0.4196 | 0.4137 | 0.3817 | 0.4343 |
| independent | joint | 0.2449 | 0.3120 | 0.3274 | 0.3478 | 0.3520 | 0.3266 | 0.3728 |
| independent | relational | **0.1980** | **0.2671** | **0.2786** | **0.3063** | **0.3157** | **0.2886** | **0.3331** |

At λ=0.009 (job 1228), same task, same survivor counts, lower everywhere and the
same ordering:

| regime | predictor | h=1 | h=5 | h=10 | h=15 |
|---|---|---:|---:|---:|---:|
| correlated | independent | 0.1010 | 0.2268 | 0.2299 | 0.2704 |
| correlated | joint | 0.0991 | 0.1768 | 0.1844 | 0.2083 |
| correlated | relational | **0.0840** | **0.1632** | **0.1693** | **0.1925** |
| independent | independent | 0.1346 | 0.2526 | 0.2365 | 0.2795 |
| independent | joint | 0.1463 | 0.2096 | 0.2006 | 0.2318 |
| independent | relational | **0.1165** | **0.1848** | **0.1741** | **0.2054** |

## 4. Balance disagrees, and not because of the source-policy defect

Balance's heuristic branch was collected with Transport's policy
([DEFECT.md](../../../outputs/balance_1233/DEFECT.md)). That does **not** explain
this table: these rows are the `correlated` and `independent` training regimes,
and the ground truth is generated by `collect.sample_actions`. The heuristic
branch is not involved on either side.

```
episodes live:  h=1:128  h=5:128  h=8:128  h=10:127  h=15:109  h=20:0
```

| regime | predictor | h=1 | h=2 | h=3 | h=5 | h=8 | h=10 | h=15 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| correlated | independent | 0.3248 | 0.4471 | 0.4610 | 0.5437 | 0.5871 | 0.6044 | **0.6523** |
| correlated | joint | **0.2626** | **0.4054** | **0.4225** | **0.5078** | 0.5686 | 0.5995 | 0.6788 |
| correlated | relational | 0.2941 | 0.4197 | 0.4470 | 0.5355 | **0.5622** | **0.5847** | 0.6628 |
| independent | independent | 0.4625 | 0.4778 | 0.5616 | 0.5987 | 0.6452 | 0.6808 | **0.7097** |
| independent | joint | **0.2795** | **0.4209** | **0.5182** | **0.5835** | 0.6213 | 0.7086 | 0.7216 |
| independent | relational | 0.3823 | 0.4465 | 0.5198 | 0.5911 | **0.6171** | **0.6814** | 0.7021 |

`joint` wins the short horizons in both regimes, `relational` takes h=8–10, and
by h=15 nothing is separated — but by then every model is at 71–79% of latent
variance, so the h=15 column is not measuring much.

## 5. Calibration

Mean per-dimension latent variance, over the same population the tables score
(`outputs/horizon_rescore_1235/latent_variance.py`, 48 checkpoints each):

| task | mean latent variance | min | max |
|---|---:|---:|---:|
| Transport | 0.9258 | 0.9202 | 0.9328 |
| Buzz Wire | 0.8301 | 0.8229 | 0.8382 |
| Balance | 0.9186 | 0.9102 | 0.9247 |

An error at that level means the model is no better than predicting the latent
mean. Transport at h=15 sits at **4.6%** of it; Buzz Wire's best at h=15 is 35%;
Balance's best is 71%.

## 6. What this changes

**Withdrawn from the first version of this note.** All three were the untrained
slot, not the model:

* *"h=5 → h=8 is a cliff, not a decay"* — 0.0355 → 0.6628, an 18.7× jump.
  The corrected figure is 0.0355 → 0.0411, **1.16×**. There is no cliff.
* *"Outside the trained context the ordering reverses"* and *"the single-agent
  model is the best of the three at h≥8"*. It is not. On Transport and Buzz Wire
  `relational` is best at **every** horizon measured, in both regimes.
* *"By h=10 every model on both tasks is at or past latent variance"* and *"the
  useful prediction horizon ends between 25 and 50 primitive steps"*. Transport
  is at 4.6% of latent variance at 75 primitive steps.

**Withdrawn consequence.** *"Any control claim that needs lookahead beyond 25
primitive steps requires a model trained on longer snippets — a data-collection
change."* No longer supported. Longer snippets may still help; nothing here
demands them, and that is one fewer reason to recollect.

**Survives, and is now stronger.** The relational advantage over the single-agent
model is a compounding effect. On Transport it grows from +0.0005 at h=1 to
+0.0071 at h=5 (14×) and to +0.0096 at h=15, and it holds for the whole range
rather than being confined to the trained window. `one_step_error` alone would
miss it.

**New and unexplained.** Balance reverses the short-horizon ordering: `joint`
beats `relational` at h≤5 in both regimes. This is the same direction as
Balance's plan-ranking result and Stage 2's physical-input reversal, and it is
now confirmed on data the source-policy defect does not touch. It is the open
question in [`../status_2026-09-16.md`](../status_2026-09-16.md) §6.

## 7. Two measurement notes

* This script generates its own ground truth by rolling the simulator from the
  128 episode starts in `initial_states.pt`, because no bank contains rollouts
  longer than six frames. Actions come from the same `collect.sample_actions`
  regime sampler the training data used, so the model is asked about the action
  distribution it was fitted on.
* Numbers here are **not** directly comparable to `metrics.json`: that pools over
  validation snippets anchored mid-episode, while this starts every rollout at an
  episode start. The h≤5 ordering agrees; the absolute levels do not.
* These are **latent** errors in each model's own space. Comparing a column
  across two predictors compares two separately learned geometries. The ordering
  is a paired per-seed comparison on a common task, which is meaningful; the
  *magnitude* of a gap between two rows is not a physical quantity. A
  common-coordinate physical comparison is Gate 2 of
  [`../review_2026-09-16.md`](../review_2026-09-16.md).

## Reproduce

```bash
sbatch scripts/slurm/horizon_rescore.sbatch          # the four tables
PYTHONPATH=. .venv/bin/python \
  outputs/horizon_rescore_1235/latent_variance.py \
  outputs/stage1_refit_1217/transport outputs/transport_data_1190 15
```
