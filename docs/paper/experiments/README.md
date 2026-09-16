# Experiment index

Every job that produced a result, where its artifacts live, and which note
documents it. Written 2026-09-16 because the results had spread across
`outputs/`, wandb and three levels of `docs/` with no single place to look.

**Reading order for someone new:** [`../outline.md`](../outline.md) for what the
project claims and where it stands, then
[`../audit_2026-09-15.md`](../audit_2026-09-15.md) for what was wrong with the
measurements, then the notes below.

Every `outputs/<name>_<job>/` directory carries `commit.txt`, `git_status.txt`,
`working_tree.patch` and a `source/` copy of `examples/world_model/*.py` as they
were at submit time, so any table can be reproduced from its own directory
without trusting the current tree.

## Jobs

| Job | Output directory | What it produced | Note |
|---:|---|---|---|
| 1194 | `interaction_control_1194/` | Transport + Dropout interaction controls | `05_counterfactual_prediction.md` |
| 1196 | `buzz_wire_1196/` | Buzz Wire bank + the headline 48-model baseline grid | `03_datasets.md`, `04_model_baselines.md` |
| 1201 | `bw_remaining_1201/` | 144 width/regulariser models + 48-run holdout replication | **missing** — backfilled to marl-eval in `a486803`, reported only via 1224 |
| 1205 | `wheel_1205/` | Wheel bank and baselines | `06_wheel_sweep.md` |
| 1216 | `closed_loop_1216/` | First full closed-loop run | `05_closed_loop_control.md` |
| 1217 | `stage1_refit_1217/` | Readout refit on frozen dynamics; the terminal-mask repair | `08_stage1_rescoring.md` |
| 1218 | `closed_loop_sweep_1218/` | Closed loop, 4 seeds; 3 Transport shards lost to OOM | partly [`10_goal_objective.md`](10_goal_objective.md) §1; full note **missing** |
| 1221 | `goal_ceiling_1221/` | Goal oracle; Transport seeds 4100/4102/4103. `transport_seed4101.json` is **partial (7 of 15 policies)** — do not read | partly [`10_goal_objective.md`](10_goal_objective.md) §1; full note **missing** |
| 1222 | `heuristic_ref_1222/` | Full-budget references incl. the Transport heuristic | **missing** |
| 1223 | `state_input_1223/` | Stage 2: observation / history / physical × 3 predictors × 8 seeds, 144 runs | **missing** |
| 1224 | `c7_rescore_1224/` | C7 re-derived on every checkpoint with the repaired evaluator (F6) | **missing** |
| 1226 | `closed_loop_recover_1226/` | Transport seed 4101, recovered sequentially | **missing** |
| 1228 | `sigreg_selected_1228/` | Headline grid retrained at the selected regulariser weight, 48 runs + C7 | **missing** |
| 1229 | `goal_repair_1229/` | Reference-controller goals. **Cancelled**: goal oracle collided 90% and finished worse than random | [`10_goal_objective.md`](10_goal_objective.md) §2 |
| 1230 | `goal_gate_position_1230/` | Gate: velocity dropped from the goal distance. **Rejected** | [`10_goal_objective.md`](10_goal_objective.md) §3 |
| 1231 | `goal_gate_success_1231/` | Gate: goals from solved episodes only. **Partial pass** | [`10_goal_objective.md`](10_goal_objective.md) §4 |
| 1233 | `balance_1233/` | Balance bank, 48-run grid, open-loop evaluators | **missing** (running) |
| — | — | Rollout error against horizon, out to 20 blocks | [`09_horizon_rollout.md`](09_horizon_rollout.md) |

Failed and cancelled jobs are listed because the audit requires failed-run
evidence to be retained (coding rule 12). Jobs 1206–1215, 1219–1220, 1225 and
1227 are failures or cancellations superseded by a later job in the same row;
their directories remain on disk.

## Notes, by number

| Note | Covers |
|---|---|
| `00_setup.md` | Environment, Slurm, MIG layout |
| `01_protocol.md` | M1 question-to-evidence matrix, task selection rationale |
| `02_oracle_validation.md` | CEM-MPC oracle on the true simulator |
| `02_transport_comparisons.md` | Transport oracle horizon study |
| `03_datasets.md` | Bank construction, regimes, splits |
| `04_model_baselines.md` | The three predictors at matched capacity |
| `05_counterfactual_prediction.md` | C7 intervention response (**pre-F6 repair — see 1224**) |
| `05_plan_ranking.md` | Plan ranking, open loop |
| `05_closed_loop_control.md` | First closed-loop control |
| `06_wheel_sweep.md` | Wheel |
| `07_reporting.md` | `report.py`, marl-eval schema |
| `08_stage1_rescoring.md` | Stage 1 salvage, job 1217 |
| `09_horizon_rollout.md` | Rollout error vs horizon; the 6-frame context limit |
| `10_goal_objective.md` | Why the goal objective measured nothing, and the one design that partly works |

Two notes share the number `02` and three share `05`. They are not renumbered
here because other documents link to them by filename; new notes continue from
`09`.

## Known documentation debt

The outline calls this the project's largest correctness risk
([`../outline.md`](../outline.md) §6.2 E0). Nine jobs above have no note, and
they include every closed-loop control result, all of Stage 2, and the repaired
C7 numbers — that is, most of what the paper's results section would cite.

Results that exist **only** as JSON in `outputs/` and in wandb, cited by nothing:

* the closed-loop control tables (1218/1221/1222/1226) and their retraction
* Stage 2's information-vs-architecture reversal (1223)
* the repaired C7 numbers that replace the withdrawn F6 ratios (1224)
* the regulariser selection and its retrained grid (1228)
* ~~the three failed goal-objective designs and the partial pass~~ — now `10_goal_objective.md`

## Where results live outside this directory

* `outputs/<name>_<job>/` — per-job JSON, logs, checkpoints, GPU traces
* wandb project `cair-traffic/counterfactual-wm` — every run logged through
  BenchMARL's logger conventions. Job 1201's 192 backfilled runs are **local
  only**; they were deliberately not pushed pending a naming decision.
* `outputs/report/` — `report.py` output over the marl-eval files
