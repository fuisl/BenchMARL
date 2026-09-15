#!/usr/bin/env bash
# Job C -- untouched held-out confirmation of the C7 result.
#
# The job-1196 bank is a development bank. Its test split was never trained on,
# but it has been inspected repeatedly while building the evaluators and choosing
# what to measure, and M3 is explicit that such a split "is not untouched
# headline confirmation". The C7 result is currently a development-set finding.
#
# This collects a completely fresh Buzz Wire bank with different state, split,
# action and branch seeds, retrains the three baselines on it, and runs the C7
# counterfactual response once. Nothing is tuned against it and it is measured a
# single time, so the number it produces is the one that can be reported as
# confirmation rather than discovery.
#
# The prediction to beat, from the development bank: relational reaches 0.736x
# and 0.743x the no-response floor on 8/8 seeds, and beats joint on 7/8.
#SBATCH --job-name=bw-holdout
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
seeds="${WORLD_MODEL_SEEDS:-5200,5201,5202,5203,5204,5205,5206,5207}"
output="${SLURM_SUBMIT_DIR}/outputs/bw_holdout_${SLURM_JOB_ID}"
data_output="$output/data"
mkdir -p "$output"
exec > >(tee -a "$output/batch.log") 2>&1
printf 'Buzz Wire holdout: job=%s seeds=%s\n' "$SLURM_JOB_ID" "$seeds"
printf 'CUDA_VISIBLE_DEVICES=%s\n' "${CUDA_VISIBLE_DEVICES:?}"

.venv/bin/python -u scripts/slurm/monitor_gpu_memory.py "$output/gpu_memory.csv" &
gpu_monitor_pid=$!
# Only suppress an already-exited monitoring process during job cleanup.
trap 'kill "$gpu_monitor_pid" 2>/dev/null || true' EXIT

# Fresh data seeds throughout: different initial states, different split
# assignment, different behaviour actions and different branch actions.
srun --ntasks=1 .venv/bin/python -m examples.world_model.collect \
    task=vmas/buzz_wire dataset.include_heuristic=false \
    dataset.state_seed=5100 dataset.split_seed=5101 \
    dataset.action_seed=5102 dataset.branch_seed=5103 \
    "hydra.run.dir=$data_output"

# Training seeds also disjoint from the development runs.
srun --ntasks=1 .venv/bin/python -m examples.world_model.train \
    --config-name sweep/world_model_baselines --multirun \
    hydra/launcher=joblib "hydra.launcher.n_jobs=$workers" \
    "data.root=$data_output" "data.regime=correlated,independent" \
    "seed=$seeds" \
    "hydra.sweep.dir=$output/baselines"

# Measured once. Do not iterate against this output.
srun --ntasks=1 .venv/bin/python -m examples.world_model.counterfactual_evaluation \
    "$output/baselines" --data "$data_output" | tee "$output/counterfactual.txt"

printf 'Completed Buzz Wire holdout: %s\n' "$output"
