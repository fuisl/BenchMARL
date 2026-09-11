#!/usr/bin/env bash
# Tuned for the local single-node dev cluster on gpu-a240 (partition `gpu`,
# gres `gpu:a100_3g.20gb:1`; check `sinfo`/`scontrol show node` on a
# different box and adjust --partition/--gres accordingly). This box isn't
# resource-constrained for this project, so this uses a generous share of
# its 32 CPUs / ~84GB RAM while still going through a MIG slice rather than
# the full A100.
#SBATCH --job-name=benchmarl-packed
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --gres=gpu:a100_3g.20gb:1
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%j.out

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
