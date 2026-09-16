# Buzz Wire is controllable, and the cadence was hiding it

2026-09-16. Job **1259**, 1h18m. True simulator throughout — no learned model.
32 independent train roots from `outputs/buzz_wire_1196/data`, full episodes,
H=5, native reward, 300 samples × 30 CEM iterations per decision. The two
configurations differ only in how much of the plan executes before the next
observation.

Buzz Wire ships no `HeuristicPolicy`, so the references are random and
do-nothing.

| policy | success | return | task_dist | coll | timeout |
|---|---:|---:|---:|---:|---:|
| **execute 5 blocks** — the cadence every prior Buzz Wire run used |
| random | 0/32 | −6.250 | 0.9526 | 0.62 | 0.38 |
| do-nothing | 0/32 | 0.000 | 0.9528 | 0.00 | 1.00 |
| **oracle** | **15/32 (47%)** | −1.160 | 0.2377 | 0.19 | 0.34 |
| **execute 1 block** |
| random | 0/32 | −6.250 | 0.9526 | 0.62 | 0.38 |
| do-nothing | 0/32 | 0.000 | 0.9528 | 0.00 | 1.00 |
| **oracle** | **25/32 (78%)** | +0.827 | **0.1259** | **0.00** | 0.22 |

`goal_dist` is 0.0000 for the oracle in both rows because the goals were
generated *from* the oracle. It is circular here and is not a result.

---

## 1. The cadence was worth 31 points of success and every failure

| | execute 5 | execute 1 |
|---|---:|---:|
| native success | 47% | **78%** |
| collisions | 19% | **0%** |
| task distance | 0.2377 | **0.1259** |
| oracle wall clock | 501s (4 decisions) | 1648s (20 decisions) |

Same objective, same per-decision search budget. Feedback every five primitive
steps instead of every twenty-five takes the planner from solving roughly half
the episodes while crashing in a fifth of them, to solving three quarters and
crashing in none.

This is a larger effect than the same repair produced on Balance (job 1237:
+22.4 → +54.3 return, 47% → 0% failures). **Every Buzz Wire control result this
project has produced was measured through `receding_horizon = args.horizon`**
(see `outputs/closed_loop_sweep_1218/source/closed_loop.py:344`), so none of them
tested a working controller.

## 2. Buzz Wire can carry the paper alone

The two surviving tasks split cleanly, and only one of them can answer both
halves of the research question:

| | Buzz Wire | Balance |
|---|---|---|
| Cross-agent response measurable in common physical coordinates | **yes**, 0.79× probe error (job 1239) | **no**, 11–70× below the probe floor |
| Controller that solves the task | **yes**, 78% with zero failures | no, 0/32 for every policy ever run |
| Relational wins physical response | **yes**, 8/8 and 7/8 paired seeds | unmeasurable |
| Learned model beats the reference | untested | joint, +19.33 vs heuristic +8.81 (3 seeds) |

Prediction and control can now be measured on **one task, with the same models,
seeds and bank**. That removes the cross-task confound the review names as its
central objection, rather than arguing around it.

**Recommendation: drop Balance from the paper.** Its counterfactual numbers are
unmeasurable and its control result cannot be linked to any prediction result on
the same task. It remains a useful negative record of a task whose interaction
is real but too small to measure.

## 3. What is left to run

1. **Gate 4 on Buzz Wire** — the checkpoints that produced the 8/8 physical
   response win, planning against a 78% reference at execute-1. This is now the
   paper's single central experiment. **Sizing warning below.**
2. **Rescore job 1201's holdout bank** in physical coordinates. No training;
   replicates the prediction result on a second bank.
3. **Rescore job 1223 (Stage 2)** in physical coordinates. No training; the
   within-task observability intervention. Transform written and dry-run
   verified, not yet applied.

## 4. Gate 4 on Buzz Wire is a Level-2 job

Measured, not estimated. On Balance the oracle took 252s per episode-set at
execute-1 and learned models took 860–1080s — **3.4–4.3× the oracle**. Buzz
Wire's oracle takes **1648s** at the same cadence, so one learned checkpoint
should cost roughly **1.5–2 hours**.

| scope | checkpoints | estimated serial time |
|---|---:|---:|
| 3-seed pilot | 18 | 28–35 h |
| 8-seed full | 48 | 75–93 h |

Coding rule 13 puts anything past a short pilot on the H100 cluster rather than
`gpu-a240`. Options, in order of preference:

* hand the 8-seed grid to **Level 2** (`packed_mig`), which is what the rule
  asks for;
* shard by seed group across the 2 concurrent local jobs with a shared
  `--reference-cache` (the mechanism `closed_loop.sbatch` already uses), which
  halves wall clock but still leaves the pilot at ~16 h;
* reduce to 16 roots, halving cost at the price of wider intervals on a task
  where the reference already separates cleanly from do-nothing.

This should not be quietly scaled down to fit locally.

## Reproduce

```bash
sbatch scripts/slurm/buzz_wire_control.sbatch    # job 1259
```
