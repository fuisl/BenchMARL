#!/usr/bin/env bash
# The two physical-coordinate rescores that need no training.
#
# Job 1239 established that Buzz Wire's cross-agent response IS measurable in
# common physical coordinates (0.79x the probe's own error) and that relational
# wins it 8/8 and 7/8 paired. Two things are needed to make that a result rather
# than a single observation, and both run on checkpoints already on disk.
#
# 1. REPLICATION. Job 1201's holdout is a second, independently collected Buzz
#    Wire bank with its own 48-model grid. If relational wins there too, the
#    prediction claim replicates across banks rather than across seeds of one.
#
# 2. THE WITHIN-TASK OBSERVABILITY INTERVENTION. Job 1223 trained the same
#    architectures under three input conditions on the same task -- observation
#    (6d), history (18d), physical (24d). That is the controlled experiment the
#    review asks for, and the reason it has not counted is that its published
#    comparison used separately-learned latent MSEs. `physical_response` now
#    feeds each checkpoint the frames its own encoder was fitted on while
#    holding Y, the scale and the anchors fixed, so all three conditions are
#    scored against the SAME physical target.
#
# Both probe families are run everywhere. Job 1239 found that swapping linear
# for MLP flips relational from worst to best, because a linear readout decodes
# relational's latent 35% worse and that decodability gap is charged to the
# model as response error. Neither family is trusted without the per-model
# reconstruction error printed beside it.
#SBATCH --job-name=physical-rescore
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:a100_2g.10gb_mps:1
#SBATCH --time=08:00:00
#SBATCH --output=slurm-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the BenchMARL repository root}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1
output="${SLURM_SUBMIT_DIR}/outputs/physical_rescore_${SLURM_JOB_ID}"
mkdir -p "$output"
exec > >(tee -a "$output/batch.log") 2>&1
printf 'Physical rescore: job=%s\n' "$SLURM_JOB_ID"
printf 'CUDA_VISIBLE_DEVICES=%s\n' "${CUDA_VISIBLE_DEVICES:?}"
git rev-parse HEAD > "$output/commit.txt"
git status --short > "$output/git_status.txt"
git diff --binary > "$output/working_tree.patch"
mkdir "$output/source"
cp examples/world_model/*.py "$output/source/"
cp scripts/slurm/physical_rescore.sbatch "$output/"

.venv/bin/python -u scripts/slurm/monitor_gpu_memory.py "$output/gpu_memory.csv" &
gpu_monitor_pid=$!
trap 'kill "$gpu_monitor_pid" 2>/dev/null || true' EXIT

score() {
    local name="$1" runs="$2" data="$3" probe="$4" horizon="$5"
    printf '\n=== %s | probe=%s | horizon=%s ===\n' "$name" "$probe" "$horizon"
    .venv/bin/python -u -m examples.world_model.physical_response \
        "$runs" --data "$data" --device cuda \
        --probe "$probe" --horizon-blocks "$horizon" \
        2>&1 | tee "$output/${name}_${probe}_h${horizon}.txt"
}

# Replication on a second Buzz Wire bank.
for probe in linear mlp; do
    for horizon in 3 5; do
        score holdout outputs/bw_remaining_1201/holdout/baselines \
            outputs/bw_remaining_1201/holdout/data "$probe" "$horizon"
    done
done

# The within-task observability intervention, in coordinates that can carry a
# physical claim. Shares Buzz Wire's bank, so Y is identical across conditions.
for probe in linear mlp; do
    for horizon in 3 5; do
        score stage2 outputs/state_input_1223/runs \
            outputs/buzz_wire_1196/data "$probe" "$horizon"
    done
done

printf '\nCompleted physical rescore: %s\n' "$output"
