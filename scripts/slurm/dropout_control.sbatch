#!/usr/bin/env bash
# M1 Row 4's weak-interaction control, in one job: collect a Dropout bank with
# the M3 protocol, then train the same three M4 baselines on it.
#
# Dropout's agents have no cross-agent dynamics (collide=False, no shared
# object, no other-agent observations) -- coupling exists only in the shared
# reward and termination. If `relational` keeps the ~13% multi-step rollout
# advantage it shows on Transport, that advantage is not interaction modelling
# and the M4 result has a non-interaction explanation.
#SBATCH --job-name=dropout-control
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:a100_3g.20gb:1
#SBATCH --time=01:00:00
#SBATCH --output=slurm-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the BenchMARL repository root}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
workers="${WORLD_MODEL_WORKERS:-3}"
output="${SLURM_SUBMIT_DIR}/outputs/dropout_control_${SLURM_JOB_ID}"
data_output="$output/data"
mkdir -p "$output"
exec > >(tee -a "$output/batch.log") 2>&1
printf 'Dropout control: job=%s output=%s workers=%s\n' \
    "$SLURM_JOB_ID" "$output" "$workers"
printf 'CUDA_VISIBLE_DEVICES=%s\n' "${CUDA_VISIBLE_DEVICES:?}"

.venv/bin/python -u scripts/slurm/monitor_gpu_memory.py "$output/gpu_memory.csv" &
gpu_monitor_pid=$!
# Only suppress an already-exited monitoring process during job cleanup.
trap 'kill "$gpu_monitor_pid" 2>/dev/null || true' EXIT

# Same M3 budgets and seeds as the Transport bank, so the control differs in
# task and nothing else. The shipped heuristic source is Transport-specific and
# stays off here.
srun --ntasks=1 .venv/bin/python -m examples.world_model.collect \
    task=vmas/dropout dataset.include_heuristic=false \
    "hydra.run.dir=$data_output"

srun --ntasks=1 .venv/bin/python - "$data_output" <<'PY'
import sys
from examples.world_model.dataset import OfflineSequences

for regime in ("independent", "correlated"):
    for split in ("train", "validation", "test"):
        dataset = OfflineSequences(sys.argv[1], regime, split, action_block=5)
        print(regime, split, len(dataset), "validated snippets", flush=True)
PY

# Same three baselines, same matched-capacity budget, same three seeds.
srun --ntasks=1 .venv/bin/python -m examples.world_model.train \
    --config-name sweep/world_model_baselines --multirun \
    hydra/launcher=joblib "hydra.launcher.n_jobs=$workers" \
    "data.root=$data_output" \
    "data.regime=correlated,independent" \
    wandb.group=m4-dropout-control \
    "hydra.sweep.dir=$output/baselines"

printf 'Completed Dropout control: %s/baselines\n' "$output"
