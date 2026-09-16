#!/usr/bin/env bash
# Gate 2 (review 2026-09-16, sections 3 and 6): the intervention response in
# common physical coordinates, across measurement horizons and both probe
# families, on every checkpoint set that has a counterfactual bank.
#
# Why this job exists. Every cross-model claim so far -- relational beats joint
# on Buzz Wire, joint beats relational on Balance -- compares errors measured in
# two different learned latent spaces. SIGReg does not make two encoders keep
# the same variables, so none of those comparisons is a statement about physics.
#
# A CPU pilot on Balance already found the thing that decides whether this can
# work at all: the cross-agent physical effect measures 0.00346 raw units after
# one block, while a linear probe's own reconstruction error is 0.24121 -- 70x
# larger. At five blocks the response grows 8x to 0.02836 and the probe error
# does not move, so it is still 9.4x too coarse. If that holds everywhere, then
# the C7 response numbers this project has reported for months are encoder noise
# rather than cross-agent coupling, and that is the finding.
#
# Two axes, because the pilot cannot tell them apart:
#   * HORIZON. The intervention flips agent 1's x action for the whole 25-step
#     snippet, but the default measurement reads it after 5 primitive steps.
#   * PROBE. A linear readout failing could mean the latent does not carry
#     physical state, or only that the readout is too weak. The MLP head is a
#     fixed-architecture capacity check, not a model to tune.
#
# Buzz Wire matters as much as Balance here: job 1223 is the project's only
# WITHIN-TASK observability intervention (observation 6d / history 18d /
# physical 24d, 144 checkpoints), and the review is right that comparing its
# separately-learned latent MSEs is not a 22% improvement in physical
# prediction. Rescoring it in common coordinates is what would turn the
# substitution hypothesis into a controlled result, and it needs no training.
#SBATCH --job-name=physical-response
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:a100_2g.10gb:1
#SBATCH --time=16:00:00
#SBATCH --output=slurm-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the BenchMARL repository root}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1
output="${SLURM_SUBMIT_DIR}/outputs/physical_response_${SLURM_JOB_ID}"
mkdir -p "$output"
exec > >(tee -a "$output/batch.log") 2>&1
printf 'Physical response: job=%s\n' "$SLURM_JOB_ID"
printf 'CUDA_VISIBLE_DEVICES=%s\n' "${CUDA_VISIBLE_DEVICES:?}"
git rev-parse HEAD > "$output/commit.txt"
git status --short > "$output/git_status.txt"
git diff --binary > "$output/working_tree.patch"
mkdir "$output/source"
cp examples/world_model/*.py "$output/source/"
cp scripts/slurm/physical_response.sbatch "$output/"

.venv/bin/python -u scripts/slurm/monitor_gpu_memory.py "$output/gpu_memory.csv" &
gpu_monitor_pid=$!
trap 'kill "$gpu_monitor_pid" 2>/dev/null || true' EXIT

score() {
    local name="$1" runs="$2" data="$3" probe="$4" horizon="$5" extra="${6:-}"
    printf '\n=== %s | probe=%s | horizon=%s blocks ===\n' "$name" "$probe" "$horizon"
    .venv/bin/python -u -m examples.world_model.physical_response \
        "$runs" --data "$data" --device cuda \
        --probe "$probe" --horizon-blocks "$horizon" $extra \
        2>&1 | tee "$output/${name}_${probe}_h${horizon}.txt"
}

bal_runs=outputs/balance_repair_1236/baselines
bal_data=outputs/balance_repair_1236/data
bw_runs=outputs/buzz_wire_1196/baselines
bw_data=outputs/buzz_wire_1196/data

# Balance first: it has the corrected bank and the pilot numbers to check
# against, so a disagreement here means the GPU path differs from the CPU one.
for horizon in 1 3 5; do
    score balance "$bal_runs" "$bal_data" linear "$horizon" --full-state
    score balance "$bal_runs" "$bal_data" mlp "$horizon" --full-state
done

# Buzz Wire's published headline, in coordinates that can carry a physical claim.
for horizon in 1 3 5; do
    score buzz_wire "$bw_runs" "$bw_data" linear "$horizon"
    score buzz_wire "$bw_runs" "$bw_data" mlp "$horizon"
done

# Stage 2 (job 1223) is deliberately NOT here. Its `history` and `physical`
# conditions rewrite the observation before training, so their encoders do not
# accept the bank's raw frames; scoring them needs that transform applied to
# both counterfactual branches. `physical_response` raises a named error on
# them rather than crashing on a shape. That transform is the next piece of
# work, and it is what turns the project's only within-task observability
# intervention into a physical result.

printf '\nCompleted physical response: %s\n' "$output"
