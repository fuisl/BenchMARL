#!/usr/bin/env bash
# M1 Row 4's control at a seed count that can actually resolve it.
#
# The three-seed pilot (jobs 1192/1193) left the control inconclusive: one seed
# in three found a much worse optimum on Dropout for every baseline, which is
# enough to dominate a three-seed mean and standard deviation. This reruns both
# tasks over eight seeds so the pre-registered quantity -- the within-task
# relational-vs-independent rollout ratio -- has a usable estimate on each.
#
# Absolute errors are not comparable across tasks (Dropout observes 7 features,
# Transport 11, in separately learned latent spaces); only the within-task ratio
# is, which is why both tasks get the identical seed set and budget.
#SBATCH --job-name=interaction-control
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --gres=gpu:a100_3g.20gb:1
#SBATCH --time=03:00:00
#SBATCH --output=slurm-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the BenchMARL repository root}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
workers="${WORLD_MODEL_WORKERS:-5}"
seeds="${WORLD_MODEL_SEEDS:-4100,4101,4102,4103,4104,4105,4106,4107}"
output="${SLURM_SUBMIT_DIR}/outputs/interaction_control_${SLURM_JOB_ID}"
transport_data="${TRANSPORT_DATA:-outputs/transport_data_1190}"
dropout_data="${DROPOUT_DATA:-outputs/dropout_control_1193/data}"
mkdir -p "$output"
exec > >(tee -a "$output/batch.log") 2>&1
printf 'Interaction control: job=%s workers=%s seeds=%s\n' "$SLURM_JOB_ID" "$workers" "$seeds"
printf 'CUDA_VISIBLE_DEVICES=%s\n' "${CUDA_VISIBLE_DEVICES:?}"
for root in "$transport_data" "$dropout_data"; do
    test -f "$root/manifest.json" || { echo "Missing dataset: $root" >&2; exit 1; }
done

.venv/bin/python -u scripts/slurm/monitor_gpu_memory.py "$output/gpu_memory.csv" &
gpu_monitor_pid=$!
# Only suppress an already-exited monitoring process during job cleanup.
trap 'kill "$gpu_monitor_pid" 2>/dev/null || true' EXIT

run_task() {
    local name="$1" root="$2"
    srun --ntasks=1 .venv/bin/python -m examples.world_model.train \
        --config-name sweep/world_model_baselines --multirun \
        hydra/launcher=joblib "hydra.launcher.n_jobs=$workers" \
        "data.root=$root" \
        "data.regime=correlated,independent" \
        "seed=$seeds" \
        "wandb.group=m4-interaction-control-$name" \
        "hydra.sweep.dir=$output/$name"
}

run_task transport "$transport_data"
run_task dropout "$dropout_data"

printf 'Completed interaction control: %s/{transport,dropout}\n' "$output"
