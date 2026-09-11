# M2 — Validate the simulator oracle and planner

Status: **in progress, stopped before CEM-MPC by request.** Snapshot/restore and deterministic-replay are validated (M2's first two bullets, on Buzz Wire, still the M1 pilot candidate); centralized CEM-MPC (M2's third bullet onward) is not yet implemented. A Give Way pipeline-calibration run (see below) is running first. See [experiment_plan.md](../experiment_plan.md) for the milestone definition and [01_protocol.md](01_protocol.md) for the current task rationale.

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
- **Per-index (partial-batch) snapshot/restore.** This validation snapshots/restores the whole batch at once. CEM-MPC and M3's counterfactual branching will need to snapshot one evaluation state and branch it across many candidate rollouts (broadcasting to a batch, or restoring only some indices via `TorchUtils.where_from_index`) — a natural next extension of `restore_state`, not built yet since it's part of the explicitly-deferred CEM-MPC work.
- **"Without changing the live evaluation environment."** Interpreted as: the snapshot/restore/replay mechanism must be side-effect-free for whatever the environment is doing afterward. The validation above demonstrates exactly this reversibility (state → hypothetical rollout → restored to the identical original state), which is the property CEM-MPC will depend on to try many candidate plans from one live state without corrupting it.

## Behaviour-policy check

First attempted on Buzz Wire directly (Level 1, `3g.20gb` MIG, small env count). Two attempts hit real problems — a timeout from an under-sized env batch, then a config mistake (`render: true`, no OpenGL here) — both cancelled and their wandb runs deleted; see git history of this file if the blow-by-blow is ever needed. Neither is worth keeping since Buzz Wire has no published number to check the result against anyway (confirmed: not one of the original VMAS paper's 4 benchmarked tasks, and BenchMARL's public report isn't independently queryable — see the timing note below).

**Pivoted to a pipeline-calibration check on Give Way instead** (01_protocol.md, revised 2026-09-11): it *is* one of the original paper's 4 benchmarked tasks, with a sharp published result — parameter-shared MAPPO/IPPO fail it, only a centralized or non-shared policy succeeds. Reproducing that pattern validates the pipeline against a known answer before trusting it on an unpublished task like Buzz Wire.

**Running now**: `task=vmas/give_way`, `algorithm=mappo` (default `share_policy_params: true`, the config expected to fail), on the full A100 (`gres=gpu:a100:1`, not the MIG — see timing note below for why), matching BenchMARL's published `fine_tuned/vmas/conf/config.yaml` recipe: `on_policy_n_envs_per_worker=600`, `on_policy_collected_frames_per_batch=60000`, `max_n_frames=10000000`, `gamma=0.9`, `evaluation_episodes=200`, `render=false` (the one deliberate deviation: kept `loggers=[csv,wandb]` instead of the recipe's `[wandb]` only, per coding_rules rule 14/local-logging preference). Single algorithm for now; `share_policy_params=false` and IPPO follow once this one gives a clean read, and once CEM-MPC/the planner are in place before scaling further (per M1/M2's evidence-driven pacing).

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

**Checks passing** (`python examples/world_model/{snapshot_restore,cem,oracle_dynamics}.py`, 13 total):
- *Optimiser, no env or model involved*: recovers a closed-form quadratic optimum (1.8e-4); respects bounds; **beats random shooting at an equal rollout budget** (the test that catches a reversed `topk`, a mis-gathered elite set, or a collapsed std — all of which still return a plausible-looking plan); elite cost decreases; reproducible under a fixed seed; single-elite update stays finite; constant cost doesn't crash.
- *Oracle*: 64 candidates score distinctly; re-scoring is bit-identical; the objective is swappable without touching the dynamics; and **the oracle's cost equals what the live simulator actually produces** for the same plan from the same state — the property that makes it an oracle rather than an approximation.

**Not yet built:** the receding-horizon MPC loop itself (execute → observe → replan, with warm start and `action_block`). So "centralised CEM-MPC" is currently optimiser + oracle dynamics without the closed loop. Also pending: `oracle_rollout` is implicitly single-state (B=1) while `cem_plan` supports B>1, and the plan-ranking metrics (Spearman ρ, elite agreement) are not implemented — when they are, validate them oracle-against-oracle first, where they must return exactly 1.0 by construction, before trusting them to judge a learned model.

## Next action

1. Read the Give Way MAPPO result: does it reproduce the paper's failure (flat/near-zero return, IPPO/MAPPO with parameter sharing can't coordinate the corridor)? If yes, the pipeline is calibrated; add `share_policy_params=false` (and IPPO) to confirm the paper's fix also reproduces.
2. Return to Buzz Wire (01_protocol.md's actual M1 recommendation) for the real behaviour-policy check, now trusting the pipeline.
3. Implement centralized CEM-MPC with simulator dynamics (M2's remaining bullets), using `snapshot_state`/`restore_state` above to branch candidate joint-action sequences from a fixed evaluation state — explicitly **not done in this pass** per instruction.
