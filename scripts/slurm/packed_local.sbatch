#!/usr/bin/env bash
#SBATCH --job-name=benchmarl-packed
#SBATCH --partition=local
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00
#SBATCH --output=slurm-%A_%a.out

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:?Submit this script from the BenchMARL repository root}"

# Manual one-allocation alternative to the automatic Submitit launcher.
config_name="${1:-sweep/vmas_smoke}"
packed_workers="${PACKED_WORKERS:-${SLURM_CPUS_PER_TASK:-1}}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-$OMP_NUM_THREADS}"

echo "host=$(hostname) job=$SLURM_JOB_ID config=$config_name workers=$packed_workers"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi -L

srun --ntasks=1 \
    uv run --frozen --no-sync python benchmarl/run.py \
    --config-name "$config_name" \
    --multirun \
    hydra/launcher=joblib \
    hydra.launcher.n_jobs="$packed_workers"
