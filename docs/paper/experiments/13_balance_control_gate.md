# Gate 1: Balance is controllable, the cadence was the bug, and the goal objective still is not

2026-09-16. Job **1237**, 5m30s. True simulator throughout — **no learned model
appears anywhere in this job**. 32 independent train roots from the corrected
bank (job 1236), full 100-step episodes, H=5 planning, native task reward,
300 samples × 30 CEM iterations per decision.

The two configurations differ **only** in how much of the plan is executed before
the next observation. Per-decision search budget is identical.

| policy | success | return | task_dist | goal_dist | coll | timeout |
|---|---|---:|---:|---:|---:|---:|
| **execute 5 blocks** (the cadence every job up to 1233 used) |
| random | 0/32 | −5.764 | 1.6139 | 2.8079 | 0.34 | 0.66 |
| zero (do-nothing) | 0/32 | −1.452 | 1.6052 | 3.0412 | 0.00 | 1.00 |
| heuristic (VMAS Balance) | 0/32 | +8.807 | 1.4526 | 2.8588 | 0.50 | 0.50 |
| oracle (CEM-MPC) | 0/32 | +22.425 | 1.3195 | 4.6012 | 0.47 | 0.53 |
| **execute 1 block** |
| random | 0/32 | −5.764 | 1.6139 | 2.8079 | 0.34 | 0.66 |
| zero | 0/32 | −1.452 | 1.6052 | 3.0412 | 0.00 | 1.00 |
| heuristic | 0/32 | +8.807 | 1.4526 | 2.8588 | 0.50 | 0.50 |
| oracle | 0/32 | **+54.310** | **1.0475** | 4.2068 | **0.00** | 1.00 |

The three non-planning references are bit-identical across the two blocks, which
is the control: they do not plan, so cadence cannot touch them.

---

## 1. Cadence was worth 2.4× return and all of the failures

Same objective, same search budget per decision, five times as many decisions:

| | execute 5 | execute 1 |
|---|---:|---:|
| oracle return | +22.425 | **+54.310** |
| oracle task distance | 1.3195 | **1.0475** |
| oracle collision rate | 0.47 | **0.00** |
| oracle wall clock | 51.4s | 251.9s |

Executing the whole plan made the controller **drop the package in 47% of
episodes**. With feedback every five primitive steps it drops it in **none**.

This was a one-line defect — `receding_horizon = args.horizon` — and it was in
every closed-loop result this project has produced. Any prior conclusion about
control was measured through it.

## 2. Gate 1 verdict: progress passes, success does not

The review's pass condition is *"simulator MPC produces repeatable native
progress/success beyond trivial baselines, and the proposed task/goal objective
agrees with native outcomes."*

**Progress — passes decisively.** Oracle return +54.310 against the shipped
heuristic's +8.807, do-nothing's −1.452 and random's −5.764. Task distance
1.0475 against do-nothing's 1.6052: a **35% reduction** in what doing nothing
leaves on the table, where the heuristic manages 10%.

**Success — fails.** 0/32. Every episode times out at 100 steps. The package
moves a long way toward the goal and never arrives. Note the shipped Balance
heuristic also completes 0/32 and drops the package half the time, so this is
not a planner-specific failure — no controller yet completes Balance in 100
steps.

**Objective agreement — fails, and in the familiar direction.** The oracle has
the **best** task distance (1.0475) and the **worst** goal distance (4.2068),
worse than do-nothing's 3.0412 and random's 2.8079. The observation-space goal
objective still anti-correlates with native progress, exactly as on Buzz Wire
(job 1229) and Transport. Balance observing its own task state did not fix it.

## 3. What this unblocks and what it does not

**Unblocked.** Balance is controllable with true dynamics. A learned model can
now be measured against a reference that demonstrably works, which was never
true on Buzz Wire or Transport. Gate 4 has a meaningful target: beat +54.310
return and 1.0475 distance, or explain the gap.

**Not unblocked.** Native success is not available as an outcome metric at
100 steps — no controller reaches it. Gate 4 must be scored on return and
physical progress, with success reported as the zero it is, or the task needs a
longer horizon before success becomes measurable.

**Still broken.** The goal-conditioned objective. Three tasks, three failures,
and Balance removes the explanation that worked for Buzz Wire (agents there
observe neither the ball nor the wire; Balance's agents observe the package, its
goal offset and its velocity). Whatever is wrong with endpoint-goal MPC is not
only partial observability. Any reward-free control claim needs this solved
first; the native reward objective does not depend on it.

## 4. A sizing note worth recording

The job was budgeted at 16 hours from a Level-0 probe that measured **132s per
decision** at 8 states. The real job ran at **12.9s per decision** for 32 states.
The probe was run on GPU 0 while another user held 35 GB of it; the job had a
dedicated MIG slice. Measuring on a contended device over-estimated by ~10×.
Size from the device the job will actually get.

## Reproduce

```bash
BALANCE_DATA=outputs/balance_repair_1236/data \
  sbatch --export=ALL,BALANCE_DATA=outputs/balance_repair_1236/data \
  scripts/slurm/balance_control.sbatch
```
