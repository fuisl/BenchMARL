# Giving the model the ball stops the crashes, and does not buy control

2026-09-17. Jobs **1273** (Gate 4 across input conditions, 3h33m) and **1276**
(readout/plan-ranking diagnostic, 2m47s). Both COMPLETED, exit 0:0.

[`16_gate4_buzz_wire.md`](16_gate4_buzz_wire.md) established that every learned
policy collided *more often than random actions* and died by step 38 of 100, and
identified the cause: Buzz Wire's agents observe `[pos, vel, pos − goal]`, so the
ball whose contact with the wire is the failure the reward punishes appears in no
agent's observation. A planner cannot avoid a constraint its model cannot
represent.

Job 1223 had already trained the same architectures under three input
conditions, so the repair needed no training — only the ability to *plan* with a
checkpoint whose encoder was fitted on 24 dimensions rather than 6, which
`model_input.py` now provides. One job scores all 18 checkpoints of seed 4100
against one set of references, so `observation` is an in-job control.

**The mechanism is confirmed. The conclusion is unchanged.**

---

## 1. Collisions fall by a factor of five, monotonically in observability

32 train roots, H=5 executing one block, native reward, 300×30 CEM — identical to
jobs 1259 and 1261–1263, so their reference rows are the same experiment.

| input | mean return | task distance | **collisions** | timeouts |
|---|---:|---:|---:|---:|
| `observation` (6d) — the condition that failed | −9.91 | 0.9615 | **0.94** | 0.06 |
| `history` (18d) | −5.20 | 0.9409 | **0.49** | 0.51 |
| `physical` (24d) — the ball supplied | −1.81 | 0.9920 | **0.17** | 0.83 |
| random | −6.25 | 0.9526 | 0.62 | 0.38 |
| do-nothing | 0.00 | 0.9528 | 0.00 | 1.00 |
| **oracle (true simulator)** | **+0.83** | **0.1259** | 0.00 | 0.22 |

Oracle success is 25/32. **Every learned cell is 0/32.**

The prediction was that a model which can see the ball would stop driving into
the wire, and it is right: **0.94 → 0.49 → 0.17**, ordered exactly by how much of
the true state the encoder receives, with mean return improving −9.91 → −5.20 →
−1.81 alongside it. The best `physical` cells collide in 1 episode of 32.

## 2. It stopped crashing by stopping

Read the last two columns together. `physical` times out in **83%** of episodes,
and its task distance is **0.9920 against do-nothing's 0.9528** — it ends
*further* from the goal than an agent that never acts.

| | return | task distance |
|---|---:|---:|
| do-nothing | 0.000 | 0.9528 |
| best learned cell (`physical` / relational / correlated) | **−0.323** | 0.9634 |

**0 of 18 cells beat doing nothing.** The entire return improvement is the
removal of collision penalties, not task progress. Giving the model the
constraint converted a planner that was *actively harmful* into one that is
*inert*.

That is a real repair of a real defect, and it is not control.

## 3. Plan ranking improves by the same factor, from a very low base

Job 1276, 96 anchor states × 48 candidate plans, 8 seeds per cell. Spearman of
each model's plan cost against the simulator's own.

| input | `J_model` (the cost CEM actually optimises) |
|---|---|
| `observation` | 0.030 – 0.064 |
| `history` | 0.043 – 0.097 |
| `physical` | **0.146 – 0.212** |

A 3–4× improvement, in the same order as §1, on the same checkpoints. This is the
cleanest evidence that the input condition — not the predictor architecture — is
what moves the planning pipeline on this task.

But ρ ≈ 0.18 is still a weak ordering. A 300-sample CEM selecting on a cost that
correlates with truth at 0.18 will pick mediocre plans, which is exactly what §2
shows it doing.

**Architecture does not order consistently here.** Within `physical`,
`independent` ranks best in one regime (0.2124) and `relational` worst (0.1463) —
the reverse of the physical-response result on the same bank. One more
dissociation between response accuracy and decision quality.

## 4. A defect in the diagnostic itself, recorded rather than reported as a result

`readout_diagnostic` reports `J_readout`: the learned readout applied to the
*simulator's own* latents, intended to isolate readout error from rollout error.
It is negative in all 18 cells (−0.001 to −0.061), and `recovered` — the gap from
`J_model` — is negative everywhere, meaning true latents rank *worse* than the
model's own predicted ones.

That is not a finding about the readout. `train.py:120-125` fits the readout on
**predicted** latents by design, because planning consumes predicted latents. The
simulator's true latents are therefore off-distribution for the head, and
`J_readout` measures that mismatch rather than readout quality. The comparison
needs a readout fitted on true latents before it can carry the claim its column
header makes.

Job 1203's ρ ≈ −0.25 has the same problem and should be re-read the same way.

## 5. What this settles

* The §1 mechanism is **confirmed**: the collision failure in
  [`16_gate4_buzz_wire.md`](16_gate4_buzz_wire.md) was an information defect, it
  is repairable without retraining, and the repair is large and monotone.
* The chain **prediction → control remains falsified**. 0/18 cells beat
  do-nothing, 0/32 successes, and task distance does not improve.
* **Observability, not architecture, is what moves this pipeline.** Across §1 and
  §3 the input condition produces 3–5× effects where the predictor produces
  none that order consistently.

One seed. Gate 4's own rule is to expand only a comparison that functions, and on
task progress this one still does not.

## 6. The next question, and it is narrower than before

The planner is not the limit: the same CEM with true dynamics solves 25/32. The
model now avoids the constraint but cannot find progress. So the open question is
whether the `physical` model's rollout represents the *goal-directed* part of the
dynamics at all, or only the part that keeps the ball away from the wire.

The measurement that would answer it is §3's, with the readout refitted on true
latents, per §4 — the same diagnostic, run correctly.

## Reproduce

```bash
GATE4_SEED=4100 sbatch scripts/slurm/buzz_wire_gate4_physical.sbatch  # 1273
sbatch scripts/slurm/readout_repair.sbatch                            # 1276
```
