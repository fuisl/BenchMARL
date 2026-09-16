#!/usr/bin/env bash
# Gate 0 (review 2026-09-16, section 7): re-derive the horizon curve with the
# context length training actually supervises.
#
# The published curve in docs/paper/experiments/09_horizon_rollout.md rolled
# through six positional slots because `pos_embedding` allocates six. Training
# predicts from `latent[:, :-1]`, so only five ever receive a prediction
# gradient; a backward pass on a trained checkpoint gives exactly 0.0 at
# position 5 (outputs/review_20260916/evidence.json). Rolling into that slot at
# block 6 is what produced the reported "h=5 -> h=8 cliff": the same Transport
# checkpoint on the same actions and roots gives 0.0565 at a 5-wide window and
# 0.2731 at a 6-wide one.
#
# Nothing is trained and no weight changes. Same checkpoints, same banks, same
# ground-truth generator, one fewer context position. Every table this produces
# is directly comparable to the one it replaces.
#
# Single process on the non-MPS 10 GB slice: the rollouts are 128 episodes wide
# and the simulator truth dominates the runtime, not the model.
#SBATCH --job-name=horizon-rescore
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --mem=8G
#SBATCH --gres=gpu:a100_2g.10gb:1
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the BenchMARL repository root}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1
output="${SLURM_SUBMIT_DIR}/outputs/horizon_rescore_${SLURM_JOB_ID}"
mkdir -p "$output"
exec > >(tee -a "$output/batch.log") 2>&1
printf 'Horizon rescore at the trained context: job=%s\n' "$SLURM_JOB_ID"
printf 'CUDA_VISIBLE_DEVICES=%s\n' "${CUDA_VISIBLE_DEVICES:?}"
git rev-parse HEAD > "$output/commit.txt"
git status --short > "$output/git_status.txt"
git diff --binary > "$output/working_tree.patch"
mkdir "$output/source"
cp examples/world_model/*.py "$output/source/"
cp scripts/slurm/horizon_rescore.sbatch "$output/"

rescore() {
    local name="$1" runs="$2" data="$3"
    printf '\n=== %s ===\n' "$name"
    .venv/bin/python -u -m examples.world_model.horizon_rollout \
        "$runs" --data "$data" --blocks 20 --device cuda \
        2>&1 | tee "$output/${name}.txt"
}

# The three sets the original note reported, so the correction is a replacement
# rather than a new measurement.
rescore transport   outputs/stage1_refit_1217/transport   outputs/transport_data_1190
rescore buzz_wire   outputs/buzz_wire_1196/baselines      outputs/buzz_wire_1196/data
rescore sigreg_0009 outputs/sigreg_selected_1228/runs     outputs/buzz_wire_1196/data
# Not in the original note: Balance is the review's first control target, and a
# horizon curve there is needed before any plan is rolled through a model.
rescore balance     outputs/balance_1233/baselines        outputs/balance_1233/data

printf '\nCompleted horizon rescore: %s\n' "$output"
