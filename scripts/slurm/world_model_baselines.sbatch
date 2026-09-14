#!/usr/bin/env bash
# M4: train the independent/joint/relational baselines on the fixed M3 bank.
# Both action regimes are trained: `correlated` is the restricted-coverage set
# the counterfactual comparison needs, `independent` is its full-coverage control.
#SBATCH --job-name=world-model-baselines
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
workers="${WORLD_MODEL_WORKERS:-3}"
output="${SLURM_SUBMIT_DIR}/outputs/world_model_${SLURM_JOB_ID}"
data_root="${WORLD_MODEL_DATA:-outputs/transport_data_1190}"
mkdir -p "$output"
exec > >(tee -a "$output/batch.log") 2>&1
printf 'M4 baselines: job=%s output=%s data=%s workers=%s\n' \
    "$SLURM_JOB_ID" "$output" "$data_root" "$workers"
printf 'CUDA_VISIBLE_DEVICES=%s\n' "${CUDA_VISIBLE_DEVICES:?}"
test -f "$data_root/manifest.json" || { echo "Missing dataset: $data_root" >&2; exit 1; }

.venv/bin/python -u scripts/slurm/monitor_gpu_memory.py "$output/gpu_memory.csv" &
gpu_monitor_pid=$!
# Only suppress an already-exited monitoring process during job cleanup.
trap 'kill "$gpu_monitor_pid" 2>/dev/null || true' EXIT

# Validate CUDA and the whole pipeline inside the allocation before the real runs.
srun --ntasks=1 .venv/bin/python -m examples.world_model.train \
    device=cuda 'loggers=[csv]' model.kind=relational \
    "data.root=$data_root" \
    train.dynamics_epochs=2 train.readout_epochs=1 train.sigreg_projections=64 \
    "hydra.run.dir=$output/smoke"

srun --ntasks=1 .venv/bin/python -m examples.world_model.train \
    --config-name sweep/world_model_baselines --multirun \
    hydra/launcher=joblib "hydra.launcher.n_jobs=$workers" \
    "data.root=$data_root" \
    "data.regime=correlated,independent" \
    "hydra.sweep.dir=$output/baselines"

printf 'Completed M4 baselines: %s/baselines\n' "$output"
