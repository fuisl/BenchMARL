# M2 — Validate the simulator oracle and planner

Status: **closed-loop implementation and CPU/CUDA/Slurm smoke validation complete; R=30 and R=10 pilots complete. Both pass the M2 return gate; R=30 reaches 11/20 goals and R=10 reaches 4/20. Horizon/full-budget ablations and held-out confirmation remain pending.** Buzz Wire is the selected M2 task. Snapshots now include VMAS episode clocks, the oracle batches B states across B·K scratch environments, and the evaluator compares MPC with random from identical saved states. Historical validation and training notes below are retained; the current workflow and evidence are at the end.

## What was inspected before writing anything

Per instruction, checked for an existing snapshot/restore mechanism before implementing one:

- `docs/source/concepts/*.rst`: the only "restore" concept in this repo is `experiment.restore_file` — **policy/optimizer checkpointing** (`examples/checkpointing/`), unrelated to simulator state. No existing concept doc covers simulator-state snapshotting.
- `examples/*`: none of the existing example directories (`running`, `checkpointing`, `evaluating`, `extending`, `configuring`, `callback`, `ensemble`, `sweep`, `plotting`) touch the raw VMAS `world`/`scenario` objects — they all operate at the `Experiment`/TensorDict level.
- `vmas/simulator/*.py` (installed `vmas==1.5.2`): no `get_state`/`set_state`/`state_dict` anywhere in the simulator core — confirmed by grep, not just absence of a docs mention.
- What *does* exist and is directly useful: VMAS's own `EntityState._reset(env_index)` (`vmas/simulator/core.py:286`) shows the intended pattern for per-index tensor updates (`TorchUtils.where_from_index`), and `World.entities` (`agents + landmarks`, `core.py:1221`) already includes joint connector landmarks automatically once `world.add_joint(...)` is called — confirmed empirically: Buzz Wire's `world.landmarks` lists `['goal', 'ball', 'joint agent_0 ball', 'joint agent_1 ball', 'wall 0', 'wall 1', 'floor 0', 'floor 1']`, so iterating `world.entities` already reaches every physically-relevant object, no scenario-specific plumbing needed for that part.

Conclusion: no existing utility to reuse; the smallest correct addition is a small, generic snapshot/restore pair, not a scenario-specific one (coding_rules rule 6/8).

## What was implemented

`examples/world_model/snapshot_restore.py` — `snapshot_state(env)` / `restore_state(env, snapshot)`.

Approach: rather than hand-listing each scenario's bookkeeping fields (`agent.shaping`, `landmark.eaten`, `package.global_shaping`, Buzz Wire's `self.pos_shaping`/`self.collided`, ...), the snapshot walks every entity in `world.entities` plus the scenario object itself and clones **any tensor attribute whose leading dimension equals the batch size**. This generically captures core physics (`pos`/`vel`/`rot`/`ang_vel` on `entity.state`) and each scenario's own extra state without per-task code, for any scenario whose extra state is plain tensor attributes — true for all six scenarios in the [task survey](../vmas_task_survey.md). Noted caveat in the code: a scenario driven through a stateful controller or action-delay queue (Give Way's `VelocityController`/`input_queue`) would need that handled separately; not needed for Buzz Wire.

## Validation performed

All on CPU (coding_rules rule 10: verify on CPU first), `task=vmas/buzz_wire`, 4 parallel envs, Level 0 (interactive, no Slurm).

1. **Reproduction check.** Reset, step 10 random actions to reach a non-trivial mid-episode state, snapshot. Roll forward 8 more steps with a random action sequence → trajectory A. Restore the snapshot, replay the *same* action sequence → trajectory B. **Result: `torch.equal` match on observations, rewards, and dones at every one of the 8 steps, across all 4 environments** — `python examples/world_model/snapshot_restore.py`:
   ```
   PASS: identical 8-step trajectory (obs/reward/done) replaying the same 4-env action sequence after restore.
   PASS: a different action sequence from the same restored state diverges.
   ```
2. **Sanity check (negative control).** From the same restored state, a *different* random action sequence produces *different* observations — rules out the trivial failure mode of the check passing because the environment ignores actions or the comparison is vacuous.
3. **Ablation (confirms the scenario-level capture is load-bearing, not redundant).** Repeated the same procedure but restored only the entity/physics state, dropping the scenario-level tensors (`self.pos_shaping`, `self.collided`) from the snapshot. Result: **observations still matched, but reward mismatched at step 0** — because Buzz Wire's reward is potential-based (`self.pos_rew = self.pos_shaping − new_shaping`) and depends on the *previous* shaping value, which lives on the scenario object, not on any entity. This confirms the generic "capture every batch-shaped tensor on the scenario too" step is doing real work, not just physics state would have sufficed.

## What this does not yet cover

- **Random-number-generator state.** Not captured. Verified this doesn't matter *for Buzz Wire*: its `reward`/`observation`/`done` contain no `.uniform_()` or other RNG calls (only `reset_world_at` does, which isn't invoked during a snapshot→replay window), and the reproduction check above is bit-exact without it. A task with per-step observation/action noise (e.g. Give Way's `obs_noise`, if set above its default 0) would need `torch.get_rng_state()`/`torch.random.set_rng_state()` added to the snapshot — flagged for whichever task ends up needing it, not implemented speculatively now (coding_rules rule 6).
- **Stateful controllers/action queues.** Give Way's `VelocityController` and `dt_delay` input queue hold internal state beyond plain tensor attributes on the scenario/entities; not exercised here since Buzz Wire uses neither.
- **Indexed branching (now implemented).** `broadcast_state(..., source_indices=...)` maps each destination slot to an explicit source slot. The original `env_index` single-source call remains available. Whole-batch restore and indexed broadcast both preserve episode counters.
- **"Without changing the live evaluation environment."** Interpreted as: the snapshot/restore/replay mechanism must be side-effect-free for whatever the environment is doing afterward. The validation above demonstrates exactly this reversibility (state → hypothetical rollout → restored to the identical original state), which is the property CEM-MPC will depend on to try many candidate plans from one live state without corrupting it.

## Behaviour-policy check

First attempted on Buzz Wire directly (Level 1, `3g.20gb` MIG, small env count). Two attempts hit real problems — a timeout from an under-sized env batch, then a config mistake (`render: true`, no OpenGL here) — both cancelled and their wandb runs deleted; see git history of this file if the blow-by-blow is ever needed. Neither is worth keeping since Buzz Wire has no published number to check the result against anyway (confirmed: not one of the original VMAS paper's 4 benchmarked tasks, and BenchMARL's public report isn't independently queryable — see the timing note below).

**Pivoted to a pipeline-calibration check on Give Way instead** (01_protocol.md, revised 2026-09-11): it *is* one of the original paper's 4 benchmarked tasks, with a sharp published result — parameter-shared MAPPO/IPPO fail it, only a centralized or non-shared policy succeeds. Reproducing that pattern validates the pipeline against a known answer before trusting it on an unpublished task like Buzz Wire.

**Historical calibration submission** (no jobs were running at the start of this implementation): `task=vmas/give_way`, `algorithm=mappo` (default `share_policy_params: true`, the config expected to fail), on the full A100 (`gres=gpu:a100:1`, not the MIG — see timing note below for why), matching BenchMARL's published `fine_tuned/vmas/conf/config.yaml` recipe: `on_policy_n_envs_per_worker=600`, `on_policy_collected_frames_per_batch=60000`, `max_n_frames=10000000`, `gamma=0.9`, `evaluation_episodes=200`, `render=false` (the one deliberate deviation: kept `loggers=[csv,wandb]` instead of the recipe's `[wandb]` only, per coding_rules rule 14/local-logging preference). Single algorithm for now; `share_policy_params=false` and IPPO follow once this one gives a clean read, and once CEM-MPC/the planner are in place before scaling further (per M1/M2's evidence-driven pacing).

**Where the data lives** (all three, independent of each other — nothing is wandb-only): local per-scalar CSV (`.../<run_name>/<run_name>/scalars/*.csv`, `step,value`, no header), a local marl-eval JSON summary (`create_json: True`), and wandb's own local buffer (`.../wandb/run-*/run-*.wandb` — exactly what `wandb sync` would upload later if a run used `WANDB_MODE=offline`) alongside the cloud copy at `wandb.ai/cair-traffic/counterfactual-wm`, tag `m1-give-way-published-comparison`.

### Why the earlier Buzz Wire attempt ran much slower than public BenchMARL/VMAS numbers

Before trusting any of the timings above, checked whether the slowdown was a bug or an explainable hardware/config gap. Could not confirm the exact GPU variant (SXM vs. PCIe) the public `matteobettini/benchmarl-public` wandb reports used — the report page is a JS-rendered app WebFetch can't execute, and this project's wandb API key doesn't have read access to that public project's underlying runs (a wandb permission quirk: a public *report* doesn't imply public *run* API access). What's fully measured instead, with four 2-iteration, single-process Level-0 timing probes (`task=vmas/buzz_wire` unless noted, MAPPO, `on_policy_collected_frames_per_batch=6000` throughout, evaluation off):

| Probe | Device | Envs/worker | Task | s/iteration |
|---|---|---|---|---:|
| A | `3g.20gb` MIG, **alone** | 60 | buzz_wire | 49.0 |
| B | full A100 (`GPU0`) | 60 | buzz_wire | 17.6 |
| C | full A100 (`GPU0`) | 600 (fine_tuned recipe's count) | buzz_wire | 8.3 |
| D | full A100 (`GPU0`) | 600 | balance (`substeps=1`) | 7.1 |

Findings, most to least significant:
- **MIG slice size dominates: A→B is a 2.8x speedup** from switching only the device, env count held fixed. This matches the hardware exactly — `nvidia-smi mig -lgip` reports the `3g.20gb` profile at 42 SMs versus 98 SMs for a MIG-mode-enabled full A100 (108 SMs completely un-partitioned), i.e. ≈2.3–2.6x compute by SM count alone.
- **Parallel env count matters almost as much: B→C is a further 2.1x speedup** (60→600 envs, same total 6000 frames/batch — just fewer, better-amortized sequential `env.step()` calls). Not a hardware limitation, a config choice: the cancelled Buzz Wire attempt used 60 envs/worker; the public `fine_tuned/vmas/conf/config.yaml` recipe (the config the public wandb report itself links as `Conf`) uses 600, now matched by the Give Way run above.
- **Two processes sharing one MIG (mappo+ippo packed together) turned out to cost little beyond the MIG-size hit**: the cancelled attempt's alternating ~35s/~60s per-iteration pattern averaged close to probe A's *solo*-MIG 49s — contention is real (the alternation itself is evidence of it) but smaller than expected, not multiplicative on top of the MIG penalty.
- **Buzz Wire's `substeps=15` (vs. the `substeps=1` default most other VMAS tasks use, confirmed by grepping every scenario) turned out to matter much less than assumed: C→D is only a 1.15x difference.** Physics substeps are a smaller share of total iteration time than fixed per-iteration overhead (Python-level env-step loop, kernel launches, buffer/logging bookkeeping) at this tiny model/entity-count scale — the original 155s/iteration collection time was almost entirely the 10-envs-per-worker config, not the task's physics cost.

Net: the observed slowness is fully explained by *known, measured, mostly-controllable* factors — a MIG slice at ~40% of a full A100's SMs (2.8x) times an under-sized env batch relative to the published recipe (2.1x) — not by anything wrong with the simulator, the snapshot/restore code, or an unexplained hardware discrepancy. This is why the Give Way calibration run above uses the full A100 with the published env count: getting a clean comparison mattered more here than staying on Level 1's usual MIG default.

## CEM planner components (M2, third bullet)

Grounded in the reference implementation rather than invented: the planner LeWorldModel actually uses is `stable_worldmodel.planning.solver.CEMSolver` (MIT), not code in the `le-wm` repo itself (that holds only the model and thin entry points). Reading it corrected several assumptions in [centralized-cem-mpc.md](../centralized-cem-mpc.md): LeWM runs **30** CEM iterations (not 3–5), K=300, 10% elites, `horizon=5` with `action_block=5` frameskip, and `receding_horizon == horizon` — it executes the *whole* plan, not just the first action. Three implementation details that are easy to get wrong: `var_scale` is a **standard deviation** despite the name; plain CEM **never clamps** (it plans in `StandardScaler`-normalised action space; only its iCEM variant clamps); and the first candidate each iteration is forced to the incumbent mean.

| Component | File | In → Out |
|---|---|---|
| State branching | `examples/world_model/snapshot_restore.py` | `broadcast_state(env, snapshot, env_index)` writes one snapshotted slot into all K slots of a scratch env, so K candidates branch from one evaluation state |
| Dynamics | `examples/world_model/oracle_dynamics.py` | `oracle_rollout(scratch_env, snapshot, candidates (K,H,N·d_a)) → {reward (K,H), live (K,H)}` |
| Objective | same | `NegativeTaskReward(rollout) → (K,)`; kept separate from the dynamics so "same planner, same cost, different world model" is structural rather than asserted |
| Optimiser | `examples/world_model/cem.py` | `cem_plan(cost_fn, …) → CEMResult(plan (B,H,D), candidates (B,S,H,D), costs (B,S), elite_idx, elite_cost_history)` |

Deliberate deviations from the reference, both recorded here: we plan in the environment's own action units and clamp to its bounds; and the joint action of N agents is flattened into the optimiser's `action_dim`, so the optimiser stays agent-agnostic and needs no multi-agent modification — all cross-agent structure lives in the dynamics it calls ("centralised CEM with factorised proposals": diagonal proposal, jointly coupled scoring and elite selection). Cost convention decided 2026-09-11: `J = −Σ_h r_h`, the true task reward for oracle *and* learned models, so the oracle gap compares planners rather than two different objectives; the learned model will therefore need a reward head, a deliberate departure from LeWM's two-term objective.

**Component checks** (`.venv/bin/python -m examples.world_model.snapshot_restore`, likewise `cem` and `oracle_dynamics`; 13 total):
- *Optimiser, no env or model involved*: recovers a closed-form quadratic optimum (1.8e-4); respects bounds; **beats random shooting at an equal rollout budget** (the test that catches a reversed `topk`, a mis-gathered elite set, or a collapsed std — all of which still return a plausible-looking plan); elite cost decreases; reproducible under a fixed seed; single-elite update stays finite; constant cost doesn't crash.
- *Oracle*: 64 candidates score distinctly; re-scoring is bit-identical; the objective is swappable without touching the dynamics; and **the oracle's cost equals what the live simulator actually produces** for the same plan from the same state — the property that makes it an oracle rather than an approximation.

## Closed-loop implementation (2026-09-11)

The current components are `snapshot_restore.py`, `oracle_dynamics.py`, `cem.py`, `mpc.py`, `metrics.py`, and the Hydra entry point `evaluate.py`, all under `examples/world_model/`. Configuration is `benchmarl/conf/oracle_mpc.yaml`; task and logger settings inherit BenchMARL defaults. The evaluator explicitly supports Buzz Wire success semantics. No training `Experiment`, algorithm registry, new backend, or dependency was added.

### Contracts and correctness

- Snapshots include `env._env.steps`. The original short replay checks missed this: with `max_steps=2`, restoring initial physics after one step produced a timeout on replay. The new regression test reproduces and prevents that bug.
- The oracle takes `(B,K,T,N*d_a)` primitive actions and returns reward/live `(B,K,T)`; the objective sums over T to produce `(B,K)`. Scratch slots are ordered state-major: `arange(B).repeat_interleave(K)`. This exploits VMAS vectorization so all evaluation episodes plan together, avoiding B serial planning loops.
- Scratch environments initialize once. Each evaluation restores state and clock, without random resets. Terminal-step rewards count; rewards after goal, collision, or timeout do not. The live environment remains untouched by candidate scoring.
- CEM still returns the final elite mean. Best-seen cost history is diagnostic only; the mean is not necessarily a previously scored candidate. Neither selected-mean cost nor closed-loop return has a monotonicity guarantee as sampling budget grows, even with exact dynamics.
- MPC horizon/receding horizon count blocks. `action_block=5` groups five independently optimized primitive actions, not repetitions of one action. H=5 means 25 VMAS steps and 100 optimization dimensions for two agents. Receding horizon 5 executes 25 steps; 1 executes 5. For replanning every primitive step use `action_block=1` and `receding_horizon=1`.
- Warm starts shift unused blocks and append zeros; standard deviation restarts at 1. With receding horizon equal to H, there is no unused tail.
- One episode per slot is evaluated without resetting/replacing completed slots. Rewards and outcomes latch at first termination. Success requires goal distance ≤0.01 without collision; simultaneous goal/collision is collision, and a collision-free goal on the time-limit step counts as success.
- `return` is the mean over agents, matching BenchMARL logging. `team_return` sums agents and matches the planner objective. Buzz Wire shares reward, so summed return is twice mean-agent return.

Batched/serial CPU oracle scores agree within 1e-5 (different VMAS batch widths produced float32 differences up to 2.2e-6). Same-width repeated scoring is bit-exact. Give Way's old self-check relied on resetting scratch controllers each call; it is now run on Buzz Wire, because controller/queue state is still outside snapshot support. This does not establish full-state support for every VMAS scenario.

### LeWM rationale

Defaults retain K=300, R=30, 30 elites and standard deviation 1 from [LeWM's solver config](https://github.com/lucas-maes/le-wm/blob/8edfeb336732b5f3ce7b8b210d0ba370a09e2cac/config/eval/solver/cem.yaml), and H=5, receding horizon 5, action block 5 from [its planning config](https://github.com/lucas-maes/le-wm/blob/8edfeb336732b5f3ce7b8b210d0ba370a09e2cac/config/eval/pusht.yaml). The [solver implementation](https://github.com/galilai-group/stable-worldmodel/blob/4821c8e6a3f0f83b7e6a80da3a757e026ea9026b/stable_worldmodel/planning/solver/cem.py) multiplies action dimension by block size. These are reference starting settings, not claimed optimal for Buzz Wire. Native action bounds and task-reward scoring remain deliberate adaptations. R=10 tests whether the larger reference budget buys useful control.

### Validation and measured compute

- Ten focused pytest cases cover timeout replay, batching versus serial evaluation, source/candidate alignment, live-state/RNG isolation, terminal masking (including real goals and collisions inside a block), block execution, mixed outcomes, warm starts, CEM reproducibility/validation, statistics and state-bank identity.
- The 13 component self-checks pass (oracle check now on Buzz Wire). CPU and CUDA end-to-end smoke configurations pass with CSV-only logging. Reloading the saved CPU state bank reproduces byte-identical episode CSV results and state-bank identity. Repository-pinned ufmt and flake8 checks pass; old formatting dependencies require a Python 3.10 tool environment, separate from the project's Python 3.11 environment.
- Slurm job 1177, `3g.20gb` A100 MIG: B=20, K=300, R=1, H=5, block=5, task limit 25. One CEM iteration took **2.632 s**; MPC including live execution took **5.050 s**; random took **2.409 s**. Artifacts: `outputs/m2_mig_timing/0/`. This is a timing/debug run with CSV only, not task-performance evidence.
- Before implementation, a full-A100 oracle probe took 2.450 s at 300 scratch slots and 2.463 s at 6,000 slots for 25 primitive steps. These are measured vector-width probes; end-to-end pilot timings are recorded separately.

### Running and reproducing

From the repository root (use `.venv/bin/python` in this checkout):

```bash
# Level 0: no wandb, two episodes, small deterministic CPU smoke.
.venv/bin/python -m examples.world_model.evaluate --config-name sweep/oracle_smoke
# Same smoke on CUDA.
.venv/bin/python -m examples.world_model.evaluate --config-name sweep/oracle_smoke experiment.sampling_device=cuda
# Level 1: full LeWM-budget pilot on the local MIG, CSV + wandb.
.venv/bin/python -m examples.world_model.evaluate --multirun hydra/launcher=packed_local hydra.launcher.max_workers_per_mig=1 hydra.sweep.dir=outputs/m2_buzz_wire_r30
# R=10, restoring the exact same bank.
.venv/bin/python -m examples.world_model.evaluate --multirun hydra/launcher=packed_local hydra.launcher.max_workers_per_mig=1 cem.num_iters=10 evaluation.states_file=outputs/m2_buzz_wire_r30/0/initial_states.pt hydra.sweep.dir=outputs/m2_buzz_wire_r10
```

Each run writes `initial_states.pt` plus a content digest, `resolved_config.yaml`, source copies/hashes and package/git provenance, `episodes.csv`, `summary.json`, `timing.json`, both action/reward/done trajectories, and a first-decision `candidate_bank.pt` with snapshot, candidate scores, final mean, elites and optimization histories. Candidate-bank actions are already unpacked into primitive steps. The bank can be rescored by another model without regenerating candidates. These small research artifacts live under ignored outputs, not git. The state-bank loader checks task settings and episode count; use the same device/software for exact replay. Separate seeds control initial states, CEM, random actions, and bootstrap resampling.

Return intervals are 95% percentile bootstrap intervals over episodes (10,000 resamples); paired differences use matched episode indices. Success uses a Wilson interval. These are pilot episode-level intervals for one planner seed, not across-seed evidence. M2's return gate requires MPC's lower CI to exceed random's upper CI. Goal rates are estimated first; numerical success threshold and held-out confirmation remain pending. Reward improvement without any collision-free goals explicitly leaves the objective/planning setup unvalidated.

### Broader sweeps: H100 handoff

Run these on the separate H100 cluster after verifying its Slurm partition/GRES/account settings. No H100 access is configured in this workspace. Transfer the state bank with the checkout; use a path accessible on workers. These are Level 2 ablations, not additional unscheduled local pilots.

```bash
# H=1 versus H=5; both execute one block (five primitive steps) per decision.
.venv/bin/python -m examples.world_model.evaluate --config-name sweep/oracle_horizon --multirun hydra/launcher=packed_mig hydra.launcher.max_workers_per_mig=1 evaluation.states_file=outputs/m2_buzz_wire_r30/0/initial_states.pt
# Controlled K/R grid, retaining 10% elites. Each invocation runs two budgets.
.venv/bin/python -m examples.world_model.evaluate --multirun hydra/launcher=packed_mig hydra.launcher.max_workers_per_mig=1 cem.num_samples=100 cem.num_elites=10 cem.num_iters=10,30 evaluation.states_file=outputs/m2_buzz_wire_r30/0/initial_states.pt
.venv/bin/python -m examples.world_model.evaluate --multirun hydra/launcher=packed_mig hydra.launcher.max_workers_per_mig=1 cem.num_samples=300 cem.num_elites=30 cem.num_iters=10,30 evaluation.states_file=outputs/m2_buzz_wire_r30/0/initial_states.pt
```

Investigate regressions using costs, collision/goal/timeout outcomes and saved plans. A finite-horizon optimization result is not a guarantee about full-episode success. Learned-model training and latent-goal scoring are intentionally deferred.

## Full-budget pilot result

Slurm job **1178**, A100 **3g.20gb MIG**, seed 0, 20 episodes, unmodified Buzz Wire task defaults (`max_steps=100`). H=5, action block 5, receding horizon 5, K=300, R=30, 30 elites. This is one pilot seed and a development state bank, not held-out confirmation.

| Policy | Mean-agent return (95% bootstrap CI) | Collision-free goals (95% Wilson CI) | Collisions | Timeouts | Evaluation time |
|---|---|---|---|---|---|
| Oracle MPC, R=30 | −1.306 [−3.318, 0.285] | 11/20 = 55% [34.2%, 74.2%] | 4/20 | 5/20 | 299.08 s |
| Random | −7.003 [−9.501, −4.505] | 0/20 = 0% [0%, 16.1%] | 13/20 | 7/20 | 9.63 s |

**M2 return gate passes:** MPC's lower CI (−3.318) exceeds random's upper CI (−4.505). The paired return improvement is **5.697 [3.620, 7.759]**. The success result shows that this objective/planner setup can solve the task in the pilot; it is not merely accumulating shaping reward without reaching goals. A numerical success gate is intentionally not chosen retrospectively. Horizon and budget comparisons and held-out confirmation remain pending.

There were four planning decisions at steps 0, 25, 50, and 75, taking 72.14, 72.12, 72.61, and 72.64 seconds respectively; active episode counts were 20, 18, 14, and 10. All 20 slots remain allocated and completed episodes are masked. The measured MPC total is **4.98 minutes** for the entire 20-episode batch.

Evidence: `outputs/m2_buzz_wire_r30/0/{summary.json,episodes.csv,timing.json,provenance.json,initial_states.pt,candidate_bank.pt}`. State-bank SHA256: `01532fbf923753c6aed5f4c28e352ee32f4368812f38d02b34c63e653a99920b`. [Wandb run](https://wandb.ai/cair-traffic/counterfactual-wm/runs/lu37jlsv). Source files and their hashes are retained with the run because the implementation was evaluated before committing.

Post-run replay validation: all 6,000 saved final-iteration candidates rescore **bit-exactly** on the simulator. Rescoring the selected final mean agrees with its first 25 actually executed steps to **1.91e-6** maximum absolute team-cost difference across the 20 episodes (float32 sum order). Evidence is `replay_validation.json` in the R=30 output directory. Terminal reward masking is included in this comparison.

## R=10 comparison and budget finding

Slurm job **1179** completed successfully with the same H, block size, K, elite count, planner seed, and **loaded R=30 state bank**; only R changes to 10. State-bank digests match, random episode CSV rows match exactly, and the first decision's first ten CEM elite-cost history entries match R=30 bit-exactly. This verifies that the two runs begin from the same optimization problem and sampling stream.

| Policy | Mean-agent return (95% bootstrap CI) | Collision-free goals (95% Wilson CI) | Collisions | Timeouts | Evaluation time |
|---|---|---|---|---|---|
| Oracle MPC, R=10 | 0.515 [0.439, 0.578] | 4/20 = 20% [8.1%, 41.6%] | 0/20 | 16/20 | 106.41 s |
| Identical random reference | −7.003 [−9.501, −4.505] | 0/20 | 13/20 | 7/20 | 9.52 s |

R=10 also passes the M2 return gate. **R=30 increases observed success (55% vs 20%) but reduces return**, with paired R=30 minus R=10 return **−1.821 [−3.841, −0.254]**. Its four collisions incur enough penalty to outweigh the additional goal completions in average reward. This is the concrete reason to keep both task-success and return metrics. It does not by itself establish a solver bug or prove that task reward is an invalid objective: finite-horizon optimization, final-mean selection, and full-episode outcomes differ. The matched-prefix histories and oracle replay checks rule out several basic wiring errors, while the horizon/budget ablations remain necessary to investigate the control tradeoff before fixing a learned-model evaluation protocol.

Evidence: `outputs/m2_buzz_wire_r10/0/`, aggregate paired comparison `outputs/m2_budget_comparison.json`, [Wandb run](https://wandb.ai/cair-traffic/counterfactual-wm/runs/ll0xvgci). Slurm jobs 1177, 1178 and 1179 all finished with exit code 0; no experiment was left running. Scheduler elapsed times (including startup/logging) were 13 s, 5 min 20 s, and 2 min 7 s respectively. R=30 remains the documented LeWM reference configuration; this pilot does not silently retune that default to maximize return.
