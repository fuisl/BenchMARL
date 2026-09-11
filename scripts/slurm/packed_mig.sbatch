#!/usr/bin/env bash
# Manual one-allocation alternative to `hydra/launcher=packed_mig`, for the
# separate H100 offload cluster (Level 2 in coding_rules.md — heavy/long
# sweeps, not the local gpu-a240 dev node).
#
# These SBATCH values mirror benchmarl/conf/hydra/launcher/packed_mig.yaml
# and have NOT been validated against a live H100 allocation from this
# session (no access to that cluster from here). Before relying on them,
# check `sinfo`, `scontrol show partition`, `scontrol show node`, and
# `/etc/slurm/gres.conf` on the H100 cluster itself and correct
# --partition/--gres/--account/--qos to match, the same way
# scripts/slurm/packed_local.sbatch was corrected for gpu-a240.
#SBATCH --job-name=benchmarl-packed
#SBATCH --partition=mig
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --account=normal
#SBATCH --qos=normal
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%j.out

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:?Submit this script from the BenchMARL repository root}"

# Default to the larger demo sweep (4 algorithms x 4 seeds); pass a different
# --config-name as $1 for the real M5/M6 sweep once it exists.
config_name="${1:-sweep/vmas_16}"
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
