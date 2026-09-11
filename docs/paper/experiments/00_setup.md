# M0 — Make the existing setup runnable

Status: **done**. See [experiment_plan.md](../experiment_plan.md) for the milestone definition and [coding_rules.md](../coding_rules.md) for conventions followed here.

## Hypothesis / goal

No scientific hypothesis at this milestone — the goal is purely operational: confirm that BenchMARL's existing VMAS integration and training loop run end to end in this checkout, on both CPU and an allocated CUDA device, before any world-model code is added.

## Commit and environment

- Commit: `d2518fb` ("docs: coding rules and experiment plans."), branch `feat/le-wm`.
- Python `3.11.15` via the checked-in `.venv` (`uv` `0.12.0`).
- Key package versions actually loaded at test time:

  | Package | Version |
  |---|---|
  | benchmarl (this checkout) | 1.5.2 |
  | torch | 2.7.1+cu126 |
  | torchrl | 0.11.0 |
  | tensordict | 0.11.0 |
  | numpy | 1.26.4 |
  | hydra-core | 1.3.6 |
  | vmas | 1.5.2 |
  | pytest (newly added, dev group) | 9.1.1 |

- GPUs visible: 2x `NVIDIA A100-PCIE-40GB` (`nvidia-smi -L`). GPU 0 is a full, un-partitioned A100. GPU 1 is split into three MIG instances: one `3g.20gb`, one `2g.10gb` (exclusive), and one `2g.10gb` dedicated to per-job MPS sharing (`/etc/slurm/gres.conf`). Node `gpu-a240` runs a real single-node Slurm cluster (`sinfo`/`scontrol show partition`): partition `gpu`, 32 CPUs, ~84 GB RAM, `gres` types `a100`, `a100_3g.20gb`, `a100_2g.10gb`, `a100_2g.10gb_mps`. See [docs/packed_slurm.md](../../packed_slurm.md) for the three-level compute policy and cluster submission.

## What was done

1. **Install the VMAS extra.**

   ```bash
   uv sync --frozen --extra vmas
   ```

   Result: `Checked 52 packages` — the extra was already present in `uv.lock` and installed in `.venv` (`vmas==1.5.2`), so this was a verification no-op rather than a fresh install. This contradicts the "VMAS is missing" note in [experiment_plan.md](../experiment_plan.md); the dependency was evidently installed between that scan and this session. Re-running the frozen sync is still the correct way to (re)produce the environment on a fresh machine.

2. **Reset/step smoke check** on three candidate tasks (navigation, transport, balance — the tasks named in the experiment plan) using `benchmarl.environments.VmasTask` directly, 4 vectorized envs, 10 random-action steps each, on CPU:

   ```bash
   .venv/bin/python <script using VmasTask.{NAVIGATION,TRANSPORT,BALANCE}.get_from_yaml().get_env_fun(...)>
   ```

   Result: all three tasks reset and stepped without error (`NAVIGATION`, `TRANSPORT`, `BALANCE` each completed 10 steps in well under a second). See "Next decision" below regarding interaction-strength selection for M1.

3. **One small MAPPO training + evaluation run, CSV logging, rendering disabled, on CPU:**

   ```bash
   .venv/bin/python benchmarl/run.py \
     algorithm=mappo task=vmas/balance \
     "experiment.loggers=[csv]" \
     experiment.render=false \
     experiment.evaluation=true experiment.evaluation_episodes=2 experiment.evaluation_interval=100 \
     experiment.max_n_iters=2 \
     experiment.on_policy_collected_frames_per_batch=100 \
     experiment.on_policy_n_envs_per_worker=2 \
     experiment.on_policy_n_minibatch_iters=1 \
     experiment.on_policy_minibatch_size=10 \
     experiment.checkpoint_interval=0 \
     experiment.train_device=cpu experiment.sampling_device=cpu \
     experiment.save_folder=outputs/00_setup/runs/cpu \
     seed=0
   ```

   Result: 2/2 iterations completed, `mean return = -11.86` at the final iteration, CSV/JSON logs written under `outputs/00_setup/runs/cpu/` (gitignored, not committed — see "Where new work belongs" in the experiment plan).

4. **Same run on an allocated CUDA device** (`CUDA_VISIBLE_DEVICES=0`, `experiment.train_device=cuda experiment.sampling_device=cuda`, `save_folder=outputs/00_setup/runs/cuda`): 2/2 iterations completed, `mean return = -6.61` at the final iteration. First iteration reward is `NaN` because no episode terminated within it yet (expected, matches the framework's own warning) — not a device error.

5. **Test tooling.** Added `pytest` as a dev dependency (`uv add --group dev pytest`, now `9.1.1`) since running BenchMARL's own tests is the natural way to validate the checkout, per the M0 note to "add test tooling when running the relevant tests."

   - `test/test_task.py` + `test/test_algorithm.py` (Hydra config/registry loading, no env rollout): **112 passed**.
   - `test/test_vmas.py` (actual VMAS rollouts through `Experiment`): **fails** — every test in this file uses the shared `experiment_config` fixture in `test/conftest.py`, which hardcodes `render=True`. Evaluation with rendering on tries to import `pyglet.gl`, which needs a working OpenGL context; this machine has neither `Xvfb`/`xvfb-run` nor the Python `OpenGL` bindings installed, and I don't have sudo to install them (`sudo -n true` fails). This is an environment gap, not a code bug — it doesn't block M0 because M0 explicitly asks for rendering *disabled* runs (steps 3–4 above), which pass.
   - `test/test_models.py` fails to collect — needs `torch_geometric`, a known-optional dependency (`gnn` extra) not required for the non-GNN baseline models planned for M1–M4.

6. **Slurm packed-launcher validation (Level 1, `gpu-a240`).** The checked-in `benchmarl/conf/hydra/launcher/packed_local.yaml` and `scripts/slurm/packed_local.sbatch` targeted a `local` partition and bare `gpu:1` gres that do not exist on this node (confirmed with `sinfo`, `scontrol show partition`, `scontrol show node`, `/etc/slurm/gres.conf` — real values are partition `gpu` and typed gres `a100`/`a100_3g.20gb`/`a100_2g.10gb`/`a100_2g.10gb_mps`). `sbatch scripts/slurm/packed_local.sbatch` failed immediately with `invalid partition specified: local`, confirming the config had never been exercised against a real allocation. Fixed both files to request `partition=gpu` and a MIG gres. First pass used the small exclusive `gpu:a100_2g.10gb:1` slice (8 CPUs, 16 GB); per follow-up direction that this box isn't resource-constrained for this project, moved to the larger `gpu:a100_3g.20gb:1` slice (42 SMs/20GB vs. 28 SMs/10GB) with 16 CPUs, 48 GB, `timeout_min: 120` — still routed through a MIG (isolated, already schedulable) rather than the full A100. Re-tested both submission paths end to end after each change:

   - Manual script: `sbatch scripts/slurm/packed_local.sbatch` → job completed (`sacct`: `COMPLETED`, exit `0:0`, ~17s both with the `2g.10gb` and the final `3g.20gb` config), ran 2 Joblib workers inside the allocation (`seed=0`, `seed=1`), each produced a training/eval iteration. Also fixed a cosmetic bug: `--output=slurm-%A_%a.out` expands `%a` to the sentinel `4294967294` for a non-array job; changed to `slurm-%j.out`.
   - Automated launcher: `uv run python benchmarl/run.py --config-name sweep/vmas_smoke --multirun hydra/launcher=packed_local` → Submitit submitted one real Slurm job (packed 2 Hydra jobs, 2 workers), completed successfully, CSV logs written under `multirun/YYYY-MM-DD/HH-MM-SS/{0,1}/`.

   `benchmarl/conf/hydra/launcher/packed_mig.yaml` and the new `scripts/slurm/packed_mig.sbatch` target the separate H100 cluster (Level 2) and were only checked for config-composition validity (`--cfg hydra`) — there is no access to that cluster from this session, so its `partition`/`gres`/`account`/`qos` values are unverified and should be re-checked with the same `sinfo`/`scontrol`/`gres.conf` commands before first use there.

## Result

**Done when** criteria met: a reproducible environment smoke check and one training/evaluation iteration pass, on CPU and on an allocated CUDA device. Both hold.

## Known gaps / uncertainty

- **Rendering/OpenGL is unavailable** in this environment (no `Xvfb`, no Python `OpenGL` bindings, no sudo). Any future code path that needs `experiment.render=true` (including BenchMARL's own `test_vmas.py` suite as currently written) will fail here. Not needed for M0–M7 as scoped (planning/evaluation is numeric, not visual), but worth flagging before relying on the existing test suite for CI.
- `torch-geometric` is still not installed; fine per the plan's note that "a small pairwise PyTorch predictor can avoid that dependency," so the relational world model (M4) should not require it.
- `packed_mig.yaml`/`packed_mig.sbatch` (H100 cluster, Level 2 per [coding_rules.md](../coding_rules.md) rule 13) are unverified — no access to that cluster from this session. Re-run the `sinfo`/`scontrol`/`gres.conf` check there before the first real submission, the same way `packed_local` was corrected for `gpu-a240` here.

## Artifacts

- Run outputs: `outputs/00_setup/runs/cpu/`, `outputs/00_setup/runs/cuda/` (gitignored `outputs/` tree, per the experiment plan's "keep large datasets/checkpoints out of `docs/paper`").
- `pyproject.toml` / `uv.lock`: added a `dev` dependency group with `pytest`.
- `benchmarl/conf/hydra/launcher/packed_local.yaml`, `scripts/slurm/packed_local.sbatch`: corrected to `gpu-a240`'s real partition/gres.
- `scripts/slurm/packed_mig.sbatch`: new manual H100-offload script mirroring `packed_mig.yaml` (unverified, see "Known gaps").
- `docs/packed_slurm.md`, `docs/paper/coding_rules.md` (rule 13): document the Level 0/1/2 compute policy.
- Test Slurm job artifacts (`multirun/2026-09-11/08-14-45/`, `slurm-1159.out`/`slurm-1160.out`) were deleted after verification; not needed as evidence beyond this note.

## Next decision

Proceed to **M1 — Freeze the experimental protocol**: pick the initial task among navigation/transport/balance (or another VMAS task) by interaction strength, fix observation/action/goal definitions, and specify data/training/planning budgets in `experiments/01_protocol.md`.
