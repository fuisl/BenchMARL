# Experiment index

For the cross-document source-of-truth order, current claim ledger, retirement
ledger, and complete `docs/paper/` catalog, start with the
[research knowledge index](../../RESEARCH_KNOWLEDGE_INDEX.md).

Every job that produced a result, where its artifacts live, and which note
documents it. Written 2026-09-16 because the results had spread across
`outputs/`, wandb and three levels of `docs/` with no single place to look.

**Start here:** [`../status_2026-09-16.md`](../status_2026-09-16.md) — where the
project stands after jobs 1217-1233, with every claim's current confidence and
what must not be claimed. Then [`../outline.md`](../outline.md) for which paper
gets written, [`../audit_2026-09-15.md`](../audit_2026-09-15.md) for what was
wrong with the measurements, then the notes below.

**Follow-up review:** [`../review_2026-09-16.md`](../review_2026-09-16.md)
reproduced three defects and set the gate structure the work now follows. Its
Gate 0 is complete (commit `43432b1`, jobs 1235 and 1236); §§3, 5–8 of the status report
are rewritten or narrowed against it. Read it before any pre-1235 horizon number
or any description of job 1233's bank as competent Balance behaviour.

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
| 1233 | `balance_1233/` | Balance bank, 48-run grid, open-loop evaluators. 48/48, no OOM. **Heuristic branch is Transport's policy** — see `balance_1233/DEFECT.md` | **missing** |
| 1235 | `horizon_rescore_1235/` | Horizon curves re-derived at the trained context length, 4 checkpoint sets. Withdraws the h=6 cliff and the ordering reversal | [`09_horizon_rollout.md`](09_horizon_rollout.md) |
| 1236 | `balance_repair_1236/` | Balance recollected on the fixed collector; identical design to 1233. COMPLETED 37:50, 48/48 | [`12_balance_corrected.md`](12_balance_corrected.md) |
| 1238 | `balance_gate4_1238/` | Gate 4: 18 learned checkpoints through the same planner as 1237's oracle. COMPLETED 4:46 | [`14_gate2_gate4.md`](14_gate2_gate4.md) §1 |
| 1259 | `buzz_wire_control_1259/` | Gate 1 on Buzz Wire: 47% -> 78% success from the cadence repair, 0 failures. COMPLETED 1:18:37 | [`15_buzz_wire_control_gate.md`](15_buzz_wire_control_gate.md) |
| 1283–1290 | `agent_position_<job>/` | Superseded one-job-per-seed layout; cancelled, with 1283 partial after 2m41s | [`20_agent_position_validation.md`](20_agent_position_validation.md) |
| 1291 | `agent_position_concurrent_1291/` | Eight seeds × two regimes × three inputs × three shared per-agent models; 144/144 complete in 9m26s | [`20_agent_position_validation.md`](20_agent_position_validation.md) |
| 1292–1293 | `latent_position_1292/`, `latent_position_linear_1293/` | Initial true-latent-head measurement; complete but superseded by the planning-style head control in 1294–1295 | [`21_latent_position_validation.md`](21_latent_position_validation.md) |
| 1294–1295 | `latent_position_mlp_1294/`, `latent_position_linear_1295/` | 144/144 latent checkpoints decoded in physical coordinates with true- and predicted-latent heads; aggregate plots and six filmstrips per probe family | [`21_latent_position_validation.md`](21_latent_position_validation.md) |
| 1296 | `decision_information_1296/` | Gate 0d: reward interface fails on true latents; physical progress is decodable, rollout rejects the oracle, and CEM concentrates into worse true populations. COMPLETED 33:15 | [`22_decision_information_localization.md`](22_decision_information_localization.md) |
| 1331 | `structured_surrogate_1331/` | Gate 0e: coverage repairs OOD physics and raises plan rho 0.162->0.286, but structured control remains unsafe (1/48 successes, 69--88% collision). COMPLETED 35:34; **failed registered control gate** | [`23_planner_coverage_structured_surrogate.md`](23_planner_coverage_structured_surrogate.md) |
| 1239 | `physical_response_1239/` | Gate 2: intervention response in common physical coordinates, 3 horizons x 2 probe families. COMPLETED 20:45 | [`14_gate2_gate4.md`](14_gate2_gate4.md) §§2-4 |
| 1237 | `balance_control_1237/` | Gate 1: simulator MPC on Balance, execute-5 vs execute-1, 32 independent train roots. COMPLETED 5:30 | [`13_balance_control_gate.md`](13_balance_control_gate.md) |
| — | `review_20260916/` | The review's own CPU diagnostics: policy identity, positional gradients, both rollout windows | [`../review_2026-09-16.md`](../review_2026-09-16.md) |

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
| `09_horizon_rollout.md` | Rollout error vs horizon (**rewritten at the corrected context, job 1235**) |
| `10_goal_objective.md` | Why the goal objective measured nothing, and the one design that partly works |
| `11_gate0_repairs.md` | The review's three defects: what each cost, and the checks that now catch them |
| `12_balance_corrected.md` | Balance on a correctly collected bank; the ranking/response dissociation |
| `13_balance_control_gate.md` | Gate 1: Balance IS controllable; cadence was worth 2.4x and all the failures |
| `14_gate2_gate4.md` | First learned-control result; Balance's response is unmeasurable; probe family flips the ordering |
| `15_buzz_wire_control_gate.md` | Buzz Wire solves 78% at the fixed cadence; it can carry prediction AND control alone |
| `16_gate4_buzz_wire.md` | Gate 4 fails: 17/18 cells worse than random. Prediction replicates on a second bank; observability orders as predicted |
| `17_observability_control.md` | Seeing the ball cuts collisions 0.94->0.17 and plan ranking 4x, and still beats nothing 0/18 |
| `18_readout_refit.md` | A fairly fitted reward head predicts 2x better and ranks plans worse; job 1203's rho ~ -0.25 withdrawn |
| `19_cost_landscape.md` | The physical model learned the collision penalty and not the progress objective; search and inaction both ruled out |
| `20_agent_position_validation.md` | Direct shared per-agent next-position model; 144-fit registered sweep complete; shared context and physical state improve one-block motion |
| `21_latent_position_validation.md` | Same position target decoded from true, one-step predicted, and recursive latent states; latent pipeline fails against persistence and direct prediction |
| `22_decision_information_localization.md` | Frozen Gate 0d: localize plan-ranking loss across encoder, rollout, scalar reward head, structured task head, and successive CEM populations |
| `23_planner_coverage_structured_surrogate.md` | Gate 0e: collect competent-local and CEM hard-negative coverage, then test a task-sufficient physical surrogate with the true Buzz Wire cost |
| `24_planner_tail_failure.md` | Gate 5: freeze job 1331 and separate planner-induced one-step shift, recursive accumulation, and false-safe CEM tails |
| `25_feedback_cadence_ablation.md` | Does replanning cadence K compensate for the dynamics error Gate 5 localized, on the same frozen structured surrogate; K=1 reused from job 1331, K=2/5 computed |
| `26_g6a_planner_aware_aggregation.md` | Experiment 24/G6a: two-round structured DAgger with a matched generic-data control, re-diagnosed by an unchanged Gate 5 rerun |
| `27_audit_gate_a0_reference_profile.md` | Audit Gate A0 step 1: preserve the compact legacy model and add a separately named reference-compatible LeWM architecture/training profile |

Two notes share the number `02` and three share `05`. They are not renumbered
here because other documents link to them by filename; new notes continue from
`09`.

## Project-level summary

[`../project_report_2026-09-17.md`](../project_report_2026-09-17.md) — the whole
arc in one document, for readers who have not followed the job numbers.

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
