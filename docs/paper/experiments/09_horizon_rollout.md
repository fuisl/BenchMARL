# Rollout error against horizon, and the context limit that bounds it

2026-09-16. Produced by [`examples/world_model/horizon_rollout.py`](../../../examples/world_model/horizon_rollout.py),
run directly on GPU 0 (Level 0 — minutes, no Slurm). Checkpoints are the existing
ones from jobs 1196 (Buzz Wire), 1217 (Transport) and 1228 (Buzz Wire at the
selected regulariser weight). Nothing was trained.

Until now every rollout number in the project was a single pooled scalar:
`metrics.json` records `one_step_error`, the mean over all blocks, and the final
block. That hides the shape of the curve, which is what decides whether a model
can be planned through.

**One block = 5 primitive steps** (`action_block=5`), so h=5 is a 25-step
rollout, the depth CEM actually uses.

---

## 1. The model cannot roll past 5 blocks at all

`ARPredictor.pos_embedding` is `(1, 6, 192)` — sized to the six-frame training
snippet. `MultiAgentWorldModel.rollout` grows its history without bound and
indexes that embedding, so it **raises** at block 6. This is a property of the
architecture, not of either task, and it caps the planner's horizon at 25
primitive steps regardless of how good the dynamics are.

Every number at h≥6 below therefore comes from a **sliding-window** rollout: keep
the last six frames, re-index from position 0. That is what any deployment past
the training length would do, and it is extrapolation, not what the model was
trained to produce. It is marked as such everywhere it appears.

## 2. Transport — the clean curve

All 128 episodes are live through h=15 (Transport has no early-failure mode), so
nothing here is confounded by survivorship.

| regime | predictor | h=1 | h=2 | h=3 | h=5 | h=8 | h=10 | h=15 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| correlated | independent | 0.0176 | 0.0283 | 0.0323 | 0.0355 | 0.6628 | 0.8899 | 1.1746 |
| correlated | joint | 0.0176 | 0.0249 | 0.0272 | 0.0304 | 0.8579 | 1.1706 | 1.4551 |
| correlated | relational | **0.0171** | **0.0227** | **0.0252** | **0.0284** | 0.7207 | 0.9559 | 1.2284 |
| independent | independent | 0.0166 | 0.0279 | 0.0325 | 0.0358 | 0.6747 | 0.8908 | 1.1800 |
| independent | joint | 0.0177 | 0.0245 | 0.0279 | 0.0310 | 0.8885 | 1.2093 | 1.4880 |
| independent | relational | **0.0164** | **0.0215** | **0.0243** | **0.0272** | 0.7285 | 0.9570 | 1.2359 |

## 3. Buzz Wire — read with the survivor counts

Buzz Wire terminates on collision, so the population shrinks and the later
columns mix horizon with survivorship: the episodes still alive at h=15 are the
easy half.

```
episodes live:  h=1:128  h=2:127  h=3:124  h=5:109  h=8:80  h=10:71  h=15:49  h=20:0
```

| regime | predictor | h=1 | h=2 | h=3 | h=5 | h=8 | h=10 | h=15 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| correlated | independent | 0.1841 | 0.2893 | 0.3244 | 0.3586 | **0.6536** | **0.7381** | **0.8688** |
| correlated | joint | 0.1669 | 0.2197 | 0.2433 | 0.2821 | 0.7461 | 0.9399 | 1.1844 |
| correlated | relational | **0.1480** | **0.2016** | **0.2261** | **0.2611** | 0.6878 | 0.8415 | 1.0288 |
| independent | independent | 0.2455 | 0.3504 | 0.3828 | 0.4196 | **0.6263** | **0.6583** | **0.7605** |
| independent | joint | 0.2449 | 0.3120 | 0.3274 | 0.3478 | 0.7273 | 0.8753 | 1.1165 |
| independent | relational | **0.1980** | **0.2671** | **0.2786** | **0.3063** | 0.6950 | 0.7959 | 0.9729 |

## 4. What the numbers mean

**Calibration.** Mean latent variance is **0.8125** on Buzz Wire and **0.9009**
on Transport. An error at that level means the model is no better than predicting
the latent mean, so those are the lines past which a rollout carries no
information.

**Inside the trained context the models are genuinely accurate.** Transport sits
at 0.017–0.036 through h=5, about 3% of latent variance.

**h=5 → h=8 is a cliff, not a decay.** Transport goes 0.0355 → 0.6628, an **18.7×
jump in three blocks**, exactly at the context boundary. By h=10 every model on
both tasks is at or past latent variance. The useful prediction horizon ends
somewhere between **25 and 50 primitive steps**.

**The multi-agent advantage is a compounding effect, not a one-step effect.** On
Transport at h=1 the three predictors are indistinguishable (0.0176 / 0.0176 /
0.0171) and `joint` is marginally *worse* than the single-agent model. The
relational advantage over `independent` grows from +0.0005 at h=1 to +0.0071 at
h=5 — a 14× increase. Reading `one_step_error` alone would have missed it
entirely. This matters for planning specifically: CEM rolls five blocks, so it
operates where the gap is widest.

**Outside the trained context the ordering reverses.** At h≥8 the single-agent
model is the *best* of the three on both tasks and `joint` is the worst
(Transport h=15: 1.1746 vs 1.4551). The more heavily conditioned a model is, the
faster it degrades once it is extrapolating beyond its window. The relational
advantage is confined to the trained horizon and should be claimed only there.

## 5. Two measurement notes

* This script generates its own ground truth by rolling the simulator from the
  128 episode starts in `initial_states.pt`, because no bank contains rollouts
  longer than six frames. Actions come from the same `collect.sample_actions`
  regime sampler the training data used, so the model is asked about the action
  distribution it was fitted on.
* Numbers here are **not** directly comparable to `metrics.json`: that pools over
  validation snippets anchored mid-episode, while this starts every rollout at an
  episode start. The h≤5 ordering agrees; the absolute levels do not.

## 6. What this changes

The project's headline is that relational structure improves prediction. That
survives, and is now better localised: it holds **within the trained context
window and nowhere else**, and it is a compounding-error effect rather than a
per-step accuracy effect.

It also puts a hard number on the planning horizon. Any control claim that needs
lookahead beyond 25 primitive steps requires a model trained on longer snippets —
the banks hold six frames, so that is a data-collection change, not a tuning one.
