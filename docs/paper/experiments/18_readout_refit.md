# Refitting the reward head is not the fix, and one long-standing number is retired

2026-09-17. Job **1282**, 30m18s, COMPLETED. 192 checkpoints, 96 anchor states ×
48 candidate plans, 8 seeds per cell. Plus two follow-up checks run directly.

[`17_observability_control.md`](17_observability_control.md) left one question
open. Giving the encoder the ball cut collisions five-fold but produced no task
progress, and it was unknown whether the `physical` model's rollout carries the
goal-directed dynamics at all or only enough to avoid the wire. The obvious
suspect was the reward head. It is not the suspect.

---

## 1. The `J_readout` column was measuring the wrong thing, and it is now fixed

`train.readout_losses` fits the reward head on **predicted** next-latents by
design, because planning consumes predicted latents (`train.py:120-125`). Scoring
that head on the simulator's own consecutive latents feeds it a distribution it
never saw. `--refit-readout` fits a fresh head of identical architecture,
capacity and objective on TRUE latent pairs — the comparison the column always
needed. The encoder stays frozen.

On the 48-model grid Gate 4 actually planned with:

| | Spearman |
|---|---|
| `J_readout` — trained head, off-distribution | −0.048 … −0.058 |
| **`J_refit` — head fitted on what it is scored on** | **+0.028 … +0.050** |

**Job 1203's ρ ≈ −0.25 is withdrawn.** It was never evidence that the reward head
cannot order plans; it was the distribution mismatch. That number has been cited
throughout this project — including as the central support for the observability
diagnosis in [`16_gate4_buzz_wire.md`](16_gate4_buzz_wire.md) §5.4 — and every
use of it should be re-read. The diagnosis it supported survives on other
evidence (jobs 1273 and 1276), but not on this.

## 2. A head that predicts reward twice as well ranks plans worse

The refitted head is not weak. On held-out validation snippets it is roughly
**twice as accurate** as the trained head, against a reward second moment of
8.57:

| input | split | refit head MSE | trained head MSE |
|---|---|---:|---:|
| `observation` | train | 1.99 | 6.25 |
| `observation` | validation | **3.46** | 6.87 |
| `physical` | train | 1.98 | 5.31 |
| `physical` | validation | **3.66** | 6.85 |

And its plan ranking is worse everywhere it matters:

| input | `J_model` (trained head, predicted latents) | `J_refit` (fitted head, true latents) |
|---|---:|---:|
| `history` | 0.043 – 0.099 | +0.012 … +0.021 |
| `observation` | 0.030 – 0.064 | +0.031 … +0.056 |
| **`physical`** | **0.146 – 0.212** | **−0.049 … −0.195** |

The best plan-ranking number in the whole table is `physical` under the
**trained** head on **predicted** latents — which is exactly the configuration
Gate 4 used. The planner was already using the best scorer available.

## 3. The obvious mechanism was tested and is wrong

The `physical` cells are worst under the `correlated` regime (−0.18 … −0.20)
and milder under `independent` (−0.05 … −0.06), which suggested the head was
fitted on the bank's regime-sampled actions and scored on uniform-random
candidate plans — and `correlated` excludes half the joint action space by
construction.

Rescoring with candidates drawn from the training regime instead of uniform:

| checkpoint | uniform candidates | regime-matched candidates |
|---|---:|---:|
| `physical` / correlated | −0.2170 | −0.2112 |
| `physical` / independent | −0.0919 | −0.0184 |
| `observation` / correlated | −0.0039 | +0.0369 |

**Matching the candidate distribution does not rescue it.** The hypothesis is
rejected for the case it was invented to explain.

One untested hypothesis remains, recorded as a hypothesis: plan cost differences
on Buzz Wire are dominated by *whether a candidate collides*, a rare, large
event, while a squared-error fit is dominated by the non-collision bulk. A head
that fits the bulk better can then be more confidently wrong on the tail that
decides the ranking — and the `physical` input, which fits the bulk best, would
be worst. Nothing here tests that.

## 4. What this settles

* **The readout is not the bottleneck.** Doubling reward-prediction accuracy made
  plan ranking worse, so no amount of readout capacity or refitting recovers
  control. The next model change should not be there.
* **Gate 4 was not handicapped.** It planned with the best-ranking configuration
  this table contains.
* **Reward accuracy and plan-ranking ability are different quantities on this
  task**, and the project has now measured a case where they move in opposite
  directions. That belongs beside the response-vs-control dissociation already
  recorded in [`14_gate2_gate4.md`](14_gate2_gate4.md) §5.

The line proposed at the end of
[`17_observability_control.md`](17_observability_control.md) §6 is therefore
closed. The measurement paper stands as written, with one more supporting result
and one withdrawn number.

## Reproduce

```bash
sbatch scripts/slurm/readout_repair.sbatch     # job 1282, with --refit-readout
```
