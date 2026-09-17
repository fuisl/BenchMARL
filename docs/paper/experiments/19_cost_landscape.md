# Safe but blind: the physical model learned the penalty and not the objective

2026-09-17. `examples/world_model/cost_landscape.py`, 4m21s on one MIG slice.
16 roots × 302 candidate plans, H=5 blocks (25 primitive steps), all 18 Stage 2
checkpoints of seed 4100.

[`17_observability_control.md`](17_observability_control.md) left three
explanations standing for a `physical` model that stops colliding and makes no
progress. This separates them.

---

## 1. What was held fixed

Job 1223 trained the three input conditions as a controlled intervention. Read
from the 144 `resolved_config.yaml` files, **everything below is identical**:

| | |
|---|---|
| data bank, regime, split, action block | `outputs/buzz_wire_1196/data`, both regimes, block 5 |
| **training snippets** | **683 train / condition** — the transform rewrites `observation` in place and touches nothing else |
| latent dim, hidden dim, depth, heads, dim_head, mlp_dim, dropout | 192, 512, 4, 8, 32, 512, 0.0 |
| **conditioner parameter budget** | **1,083,072 — solved for, not shared width** |
| optimiser | AdamW, lr 5e-5, wd 1e-3, clip 1.0, batch 128 |
| epochs | 100 dynamics + 50 readout |
| SIGReg weight | 0.09 |
| seeds | 4100–4107 |

**Two things are not identical, and neither can be.**

* **Encoder input width.** 6 → 18 → 24 dimensions, so the encoder's first layer
  grows: 103,104 → 109,248 → 112,320 parameters. That is **+9,216, or +0.24% of
  the model's 3.81M**. It is unavoidable — a 6-dimension input layer cannot be
  fed 24 numbers — and it is two orders of magnitude smaller than the effects
  reported below.
* **The reward head.** Each checkpoint has *its own* readout, fitted in stage 2
  on *its own* frozen dynamics. Same architecture, capacity, objective and epoch
  count; different weights. Sharing one head is impossible because each model
  learns its own latent space.

At evaluation (job 1273) all 18 checkpoints were scored **in one job**: same 32
roots, same references, same H, same cadence, same 300×30 CEM budget, and
`run()` re-seeds the planner generator identically per policy, so every policy
sees the same CEM noise.

## 2. The discriminating measurement

One candidate bank per root, scored twice — by the simulator, and by the learned
cost each planner actually optimises (`plan_ranking.model_costs`, i.e. reward
head × predicted survival, which is the whole objective).

The bank is **1 oracle plan + 1 zero plan + 300 random plans**. The oracle plan
is what the true-dynamics CEM chooses from that state at the same budget, and at
full budget it is genuinely good:

| plan | true return | final task distance |
|---|---:|---:|
| **oracle** | **+0.384** | **0.7087** |
| zero (do nothing) | 0.000 | 0.9008 |
| random (mean of 300) | −2.363 | 0.9013 |

Taking the learned cost's **argmin over all 302** removes search as a variable:
no optimiser can fail on an exhaustive comparison of the bank.

| input | zero %ile | oracle %ile | ρ(cost, true return) | **ρ(cost, task distance)** | chosen plan's distance | cost spread |
|---|---:|---:|---:|---:|---:|---:|
| `observation` | 0.84 – 0.96 | 0.16 – 0.43 | −0.02 … +0.07 | −0.03 … **+0.06** | 0.860 – 0.892 | 0.21 – 0.27 |
| `history` | 0.37 – 0.62 | 0.65 – 0.93 | −0.02 … +0.12 | −0.06 … +0.03 | 0.892 – 0.903 | 0.24 – 0.38 |
| **`physical`** | **0.37 – 0.50** | **0.44 – 0.78** | +0.05 … +0.11 | **−0.07 … −0.09** | **0.898 – 0.904** | **0.51 – 0.69** |

*%ile = fraction of the bank the learned cost prefers to that plan; 0.00 is the
cost's own favourite.*

## 3. Which explanation survives

**Not "safe but static".** If the model believed inaction were optimal, the zero
plan would sit at the cost's minimum. It sits **mid-pack, at the 37th–50th
percentile**, and the cost landscape is not flat — `physical` has the *largest*
spread of the three conditions (0.51–0.69 against `observation`'s 0.21–0.27). The
model has strong opinions; they are simply not about progress.

**Not a search failure.** The learned cost ranks the genuinely good oracle plan
at the **44th–78th percentile** — it is handed the answer and does not prefer it.
And the argmin over the entire 302-plan bank lands at task distance **0.898–0.904**,
which is the do-nothing distance (0.9008) and far from the oracle's 0.7087. There
is nothing for CEM to find. CEM is optimising this cost faithfully.

**The model cannot predict progress.** ρ between the learned cost and final task
distance is **negative for every `physical` cell** (−0.07 … −0.09) while ρ against
true *return* is mildly positive (+0.05 … +0.11). The cost tracks the part of the
reward that punishes collisions and is mildly **anti**-correlated with getting the
ball closer to the goal.

## 4. The dissociation this exposes

The two conditions fail in opposite, complementary ways:

| | knows about crashing | knows about progress |
|---|---|---|
| `observation` (6d) | **no** — ranks doing nothing at the 84th–96th percentile, i.e. actively dislikes safety, and collides in 94% of episodes | **partly** — ranks the oracle plan at the 16th–43rd percentile, the best of the three |
| `physical` (24d) | **yes** — collisions fall to 17% | **no** — anti-correlated with task distance |

Giving the encoder the ball taught the model **the penalty term and not the
objective term**. That is a precise account of "safe but inert", and it is not
what any of the three original hypotheses said.

It also explains the shape of job 1273: `physical` improved return from −9.91 to
−1.81 purely by removing collision penalties, and never moved task distance off
the do-nothing value.

## 5. What this licenses, and what it does not

**Licensed.** Control on Buzz Wire fails because the learned cost does not encode
task progress, not because the planner is weak and not because the model prefers
inaction. Any repair must change what the cost represents — reward shaping, a
progress-predicting head, or planning against a goal in a space the model can
represent — and *not* the search budget, the architecture, or the seed count.

**Not licensed.** Anything about architecture: the three predictors are within
noise of each other in every column here, as they were in jobs 1273 and 1282.
And this is one seed, 16 roots, one horizon — it identifies the failure mode, it
does not quantify it.

## Reproduce

```bash
python -m examples.world_model.cost_landscape \
    outputs/state_input_1223/runs --data outputs/buzz_wire_1196/data \
    --device cuda --roots 16 --random-plans 300 --num-samples 300 --num-iters 30
```
