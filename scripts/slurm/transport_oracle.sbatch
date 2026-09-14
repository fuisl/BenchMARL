#!/usr/bin/env bash
# One GPU allocation for Transport smoke, timing, and paired comparisons.
# User switched to the available 20 GB MIG with four workers (2026-09-14).
#SBATCH --job-name=transport-oracle
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:a100_3g.20gb:1
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the BenchMARL repository root}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
transport_workers="${TRANSPORT_WORKERS:-4}"
transport_output="${SLURM_SUBMIT_DIR}/outputs/transport_oracle_${SLURM_JOB_ID}"
mkdir -p "$transport_output"
exec > >(tee -a "$transport_output/batch.log") 2>&1
printf 'Transport comparison batch: job=%s output=%s\n' "$SLURM_JOB_ID" "$transport_output"
printf 'CUDA_VISIBLE_DEVICES=%s workers=%s\n' "${CUDA_VISIBLE_DEVICES:?}" "$transport_workers"
.venv/bin/python -u scripts/slurm/monitor_gpu_memory.py "$transport_output/gpu_memory.csv" &
gpu_monitor_pid=$!
# Only suppress an already-exited monitoring process during job cleanup.
trap 'kill "$gpu_monitor_pid" 2>/dev/null || true' EXIT

# Validate CUDA inside the actual allocation before any real experiment.
srun --ntasks=1 .venv/bin/python -m examples.world_model.evaluate \
    --config-name sweep/oracle_smoke task=vmas/transport task.max_steps=5 \
    experiment.sampling_device=cuda "hydra.run.dir=$transport_output/smoke"
srun --ntasks=1 .venv/bin/python -m examples.world_model.evaluate \
    task=vmas/transport cem.num_iters=1 task.max_steps=25 \
    'experiment.loggers=[csv]' "hydra.run.dir=$transport_output/timing"

# Save a single nonterminal development bank on the allocated GPU. Every
# planner seed and budget restores this file, regardless of initialization RNG.
srun --ntasks=1 .venv/bin/python - "$transport_output/initial_states.pt" <<'PY'
import sys
import torch
from benchmarl.environments import VmasTask
from examples.world_model.snapshot_restore import snapshot_state

task = VmasTask.TRANSPORT.get_from_yaml()
env = task.get_env_fun(20, True, 0, "cuda")()
try:
    env.set_seed(0)
    env.reset()
    torch.save({"task": task.config, "state_seed": 0, "snapshot": snapshot_state(env)}, sys.argv[1])
    print("Prepared 20 Transport states on", torch.cuda.get_device_name(0), flush=True)
finally:
    env.close()
PY

# Hydra expands 6 configurations x 3 planner seeds. All workers inherit the
# same allocated GPU; each worker also batches B*K simulator slots internally.
srun --ntasks=1 .venv/bin/python -m examples.world_model.evaluate \
    --config-name sweep/transport_oracle --multirun \
    hydra/launcher=joblib "hydra.launcher.n_jobs=$transport_workers" \
    'experiment.loggers=[csv,wandb]' \
    "evaluation.states_file=$transport_output/initial_states.pt" \
    "hydra.sweep.dir=$transport_output/comparisons"
srun --ntasks=1 .venv/bin/python -m examples.world_model.summarize_comparisons \
    "$transport_output/comparisons"
printf 'Completed Transport comparisons: %s/comparisons/comparison.csv\n' "$transport_output"
