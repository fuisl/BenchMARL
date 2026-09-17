# Gate 4 on Buzz Wire, and the prediction result that replicates

2026-09-16/17. Jobs **1261/1262/1263** (Gate 4, one seed each, 1h37–1h41) and
**1264** (two physical rescores, 26m). All four COMPLETED, exit 0:0.

The two halves point in opposite directions, and that is the finding.

* **Prediction replicates.** Relational captures the most cross-agent response
  on a second, independently collected bank, and the observability intervention
  orders exactly as the hypothesis predicts.
* **Control fails completely.** All 18 learned checkpoints score 0/32, and 17 of
  18 return *worse than random*. Not one beats doing nothing.

---

## 1. Gate 4 — the learned models do not control Buzz Wire

32 train roots, H=5 executing one block, native reward, 300×30 CEM. Identical to
job 1259 in roots, split, horizon, cadence and budget, so its reference rows are
the same experiment rather than a comparable one. Three seeds, six checkpoints
each. Source verified byte-identical across the three shards.

| policy | success | return | task_dist | coll |
|---|---:|---:|---:|---:|
| random | 0/32 | −6.250 | 0.9526 | 0.62 |
| do-nothing | 0/32 | 0.000 | 0.9528 | 0.00 |
| **oracle** | **25/32 (78%)** | **+0.827** | **0.1259** | **0.00** |

| regime | kind | return (per seed) | mean | task_dist | coll | success |
|---|---|---|---:|---:|---:|---:|
| correlated | independent | −9.38 −7.21 −11.24 | −9.27 | 0.9560 | 0.84 | 0/96 |
| correlated | joint | −10.97 −5.94 −10.94 | −9.29 | 0.9678 | 0.84 | 0/96 |
| correlated | relational | −7.50 −8.46 −10.94 | −8.97 | 0.9602 | 0.77 | 0/96 |
| independent | independent | −10.63 −11.55 −8.75 | −10.31 | 0.9514 | 0.94 | 0/96 |
| independent | joint | −10.63 −6.26 −9.69 | −8.86 | 0.9542 | 0.84 | 0/96 |
| independent | relational | −9.69 −10.32 −11.88 | −10.63 | 0.9575 | 0.94 | 0/96 |

**18 of 18 cells are worse than doing nothing. 17 of 18 are worse than random.**
Every learned task distance sits at 0.951–0.968 against do-nothing's 0.9528 —
the models move the system no closer to the goal than an agent that never acts,
while the true-dynamics planner on the same roots reaches 0.1259.

Paired within regime and seed, against the single-agent model:

| regime | kind | mean Δreturn | per-seed | better |
|---|---|---:|---|---:|
| correlated | joint | −0.01 | −1.60 +1.26 +0.30 | 2/3 |
| correlated | relational | +0.31 | +1.88 −1.25 +0.30 | 2/3 |
| independent | joint | +1.46 | +0.00 +5.30 −0.93 | 2/3 |
| independent | relational | −0.32 | +0.94 +1.23 −3.13 | 2/3 |

Every interval spans zero and every cell is 2/3. **Conditioning has no
detectable effect on control here**, which is what comparing six ways of failing
should look like. No architecture ranking may be drawn from this table.

### 1.1 The failure mode is driving into the wire

| policy | mean episode length (of 100) | collisions |
|---|---:|---:|
| do-nothing | 100.0 | 0.00 |
| oracle | 81.9 | 0.00 |
| random | 70.3 | 0.62 |
| learned (best, relational/correlated) | 61.8 | 0.66 |
| learned (worst, joint/independent) | 20.3 | 1.00 |

The learned policies crash in the first quarter of the episode, and they collide
*more often than random actions do*. That is not an inert planner; it is a
planner being actively steered into the wire by its model. The CEM is optimising
a predicted reward whose collision term the model does not represent, so the
highest-scoring plan is a fast, straight run at the goal through the wire.

This also explains the wall clock. Each shard took 1h37 rather than the
estimated 10–13h, because 400–480s per checkpoint buys only ~25 decisions before
the episode terminates. **The speed was the result announcing itself.**

### 1.2 What this does to the paper's central chain

Gate 4 was the one remaining experiment that had to succeed. It did not, and it
failed on the task chosen precisely because prediction and control could be
measured on it together. The chain "better cross-agent prediction → better joint
planning" is now **falsified end-to-end on the one task where both ends are
measurable**, rather than untested.

What survives is narrower and still real: the prediction half, below. What must
be dropped is any claim that the prediction advantage buys control.

## 2. The prediction result replicates on a second bank (job 1201 holdout)

48 checkpoints, 8 seeds, independently collected Buzz Wire bank. MLP probe.
Lower is better; 1.0 = no better than predicting no response.

| horizon | regime | kind | ratio | cos(ΔY) | paired vs independent | seeds |
|---|---|---|---:|---:|---:|---:|
| 3 | correlated | joint | 0.8809 | 0.3543 | −0.0723 [−0.1140, −0.0208] | 7/8 |
| 3 | correlated | **relational** | **0.8469** | **0.3985** | **−0.1063 [−0.1355, −0.0772]** | **8/8** |
| 3 | independent | joint | 0.9180 | 0.3433 | −0.0375 [−0.0659, −0.0087] | 6/8 |
| 3 | independent | **relational** | **0.8628** | **0.3773** | **−0.0928 [−0.1357, −0.0407]** | 7/8 |
| 5 | correlated | joint | 0.8865 | 0.3240 | −0.0881 [−0.1249, −0.0447] | 7/8 |
| 5 | correlated | **relational** | **0.8584** | **0.3736** | **−0.1162 [−0.1465, −0.0859]** | **8/8** |
| 5 | independent | joint | 0.8742 | 0.3710 | −0.0554 [−0.0957, −0.0159] | 5/8 |
| 5 | independent | **relational** | **0.8637** | **0.3680** | −0.0659 [−0.1012, −0.0273] | 6/8 |

Probe resolution 0.81–0.95× at every setting, so the quantity is measurable.
Job 1239 on the primary bank gave relational −0.0841 (8/8) and −0.0764 (7/8);
this bank gives −0.1162 (8/8) and −0.0659 (6/8). **Same sign, same magnitude,
same winner, different data.** Relational leads on direction everywhere
(cos 0.37–0.40 against the single-agent model's 0.20–0.25).

The linear probe again reverses it — relational 1.03–1.09, worst — and the
mechanism is again visible in the reconstruction column: linear decodes
relational's latent **75–80% worse** (|abs err| 0.205–0.215 vs 0.116–0.122),
and that decodability gap is charged to the model. Under the MLP the three sit
within 13% of each other. The reversal replicates too, which makes it a property
of linear readouts rather than an accident of one bank.

## 3. The observability intervention orders as predicted (job 1223, rescored)

144 checkpoints, 8 seeds, three input conditions trained on the same task, now
scored against the **same physical target** with Y, scale and anchors held
fixed. This is the controlled within-task experiment the review asked for; it
has never before been scorable, because its published comparison used
separately-learned latent MSEs. MLP probe, 5 blocks:

| input | probe resolution | independent | joint | relational | paired (corr. / indep.) | seeds |
|---|---|---:|---:|---:|---|---|
| physical (24d) | **0.52×** | 1.150 | **0.812** | 0.829 | −0.337 / −0.201 | **8/8 in all 4 cells** |
| observation (6d) | 0.79× | 0.964 | 0.909 | **0.879** | −0.085 / −0.066 | 8/8, 7/8 |
| history (18d) | **1.37× — unresolvable** | 0.987 | 0.958 | 0.958 | −0.029 / −0.012 | 5/8 |

The effect of conditioning grows monotonically with how much of the true state
the encoder can see: **−0.34 at full physical state, −0.085 at partial
observation, nothing at all from stacked history.** At h=3 the same ordering
holds with larger physical effects (−0.393/−0.191, 8/8).

Two cautions, both load-bearing:

* **The history condition is unmeasured, not null.** Its probe floor is 1.37–1.70
  (4.25–5.19 under a linear probe) — the readout's own error exceeds the
  predict-no-response baseline, which is incoherent and is the same pathology
  that invalidated every Balance number. Stacked 18d history produces a latent
  *harder* to decode than 6d observation. So this is two measurable points, not
  three, and the "history does not help" reading is not licensed.
* **`independent` is not pinned to 1.0 in the physical condition.** It scores
  1.150 and 1.030 under MLP but 0.80–0.84 under linear. The agent head reads one
  agent's own latent and is structurally incapable of a response; the **object
  head is global by construction** (`physical_response.py:30` — a shared object
  is nobody's private state), so the object dimensions of Y *can* respond under
  an independent predictor. `14_gate2_gate4.md` §3 says independent "cannot beat
  1.0 by construction"; that holds for the agent dimensions only and is corrected
  here.

## 4. What the project can and cannot claim as of tonight

**Can:**

* Relational conditioning captures more cross-agent response than a matched
  single-agent model, in common physical coordinates, on magnitude and
  direction, across **two independently collected banks** and 8 seeds each.
* The advantage scales with state observability, within one task, at matched
  capacity, against a common target.
* Cross-model physical comparisons are only valid with per-model reconstruction
  error reported beside them; a linear readout inverts the ordering on both banks.

**Cannot:**

* That better cross-agent prediction yields better joint planning. **Tested
  directly on Buzz Wire and falsified**: 0/32 everywhere, 17/18 cells worse than
  random, no architecture separation.
* Any architecture ranking from control, on either task. Balance's +19.33 for
  joint (job 1238) stands as a single-task result against a heuristic on a task
  whose response is unmeasurable; Buzz Wire shows no ranking at all.
* Anything from the history input condition.
* Any Balance counterfactual number, unchanged from `14_gate2_gate4.md`.

## 5. What this means for the deadline

The paper's planned contribution was the chain from prediction to control. That
chain is broken by our own experiment, on our own strongest task, with the
cleanest reference this project has ever had (78% oracle, zero failures).

The honest reframing is a **measurement paper**: cross-agent response *is*
measurable in common physical coordinates, conditioning *does* capture it, the
effect scales with observability — **and it does not transfer to control**,
demonstrated rather than asserted, with the failure mode identified (the model
does not represent the collision term the planner is optimising against).

Whether that clears AAMAS is a judgement call. What it is not is a result that
needs more seeds: 18 cells at 0/32 with returns below random is not a variance
problem, and the 8-seed expansion to Level 2 should **not** be scheduled to
rescue it.

The one experiment that would change the diagnosis rather than repeat it: train
or attach a **collision-aware reward/termination readout** and re-run Gate 4.
§1.1 identifies that term as the failure, and it is currently the only
mechanistic hypothesis on the table.

## Reproduce

```bash
GATE4_SEED=4100 sbatch scripts/slurm/buzz_wire_gate4.sbatch   # 1261; 4101→1262, 4102→1263
sbatch scripts/slurm/physical_rescore.sbatch                  # 1264
```
