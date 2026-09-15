#!/usr/bin/env bash
# Prototype sweep for the LeWM goal-conditioned plan cost.
#
# Task reward gives Spearman ~0.05 on both tasks. A GPU spot-check of the goal
# cost gave ~0.27 on Buzz Wire with the same models, which is the first ranking
# signal this project has seen and matches the diagnosis: the readout, not the
# dynamics, was the bottleneck.
#
# What this sweeps, and why each knob:
#   history_size 1 / 3 / 5 -- LeWM truncates the predictor context to 3 frames.
#     1 tests whether any context helps at all; 5 removes the truncation, since
#     our own rollout lets context grow and that is a deviation worth measuring.
#   both tasks -- Transport's readout works but its interaction is sparse, Buzz
#     Wire's is the reverse, so the goal cost should help them differently.
#
# Small allocation on purpose: no training, one batched simulator rollout plus
# forward passes over existing checkpoints.
#SBATCH --job-name=goal-proto
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:a100_2g.10gb:1
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the BenchMARL repository root}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
output="${SLURM_SUBMIT_DIR}/outputs/goal_proto_${SLURM_JOB_ID}"
mkdir -p "$output"
exec > >(tee -a "$output/batch.log") 2>&1
printf 'Goal prototype: job=%s\n' "$SLURM_JOB_ID"
printf 'CUDA_VISIBLE_DEVICES=%s\n' "${CUDA_VISIBLE_DEVICES:?}"

.venv/bin/python -u scripts/slurm/monitor_gpu_memory.py "$output/gpu_memory.csv" &
gpu_monitor_pid=$!
# Only suppress an already-exited monitoring process during job cleanup.
trap 'kill "$gpu_monitor_pid" 2>/dev/null || true' EXIT

# The fidelity tests are the claim that this matches LeWM, so run them in the
# allocation rather than trusting a local pass.
srun --ntasks=1 .venv/bin/python -m pytest test/test_world_model_goal.py -q \
    | tee "$output/fidelity_tests.txt"

run_case() {
    local name="$1" runs="$2" data="$3" history="$4"
    printf '\n=== %s (history_size=%s) ===\n' "$name" "$history"
    srun --ntasks=1 .venv/bin/python -m examples.world_model.goal_planning \
        "$runs" --data "$data" --device cuda \
        --states 96 --candidates 48 --history "$history" \
        | tee "$output/${name}_history${history}.txt"
}

for history in 1 3 5; do
    run_case buzz_wire outputs/buzz_wire_1196/baselines \
        outputs/buzz_wire_1196/data "$history"
    run_case transport outputs/interaction_control_1194/transport \
        outputs/transport_data_1190 "$history"
done

printf 'Completed goal prototype: %s\n' "$output"
