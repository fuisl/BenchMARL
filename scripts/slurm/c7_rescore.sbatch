#!/usr/bin/env bash
# Stage 1 salvage (audit 2026-09-15, section 9): re-derive C7 on every existing
# checkpoint with the repaired evaluator.
#
# F6: the reported "percentage of interaction captured" divided one model's
# latent response error by a *different* model's no-response error. SIGReg does
# not make two encoders keep the same physical variables, so that ratio compared
# magnitudes across two learned spaces, and "26% of the physical effect
# captured" and its ~50% replication were too strong. One minus an error ratio
# is a reduction in squared response error in any case, not a fraction of
# displacement recovered.
#
# The evaluator has since been repaired -- each model is divided by its own
# no-response floor, and a shared physical target is scored separately -- but
# the eight-seed claims were never re-derived with it. Until they are, F6 stands
# and none of those numbers can be used. No training and no new data: the same
# checkpoints, the same anchors, a corrected denominator.
#
# Every checkpoint set is included, with its own bank. The two from job 1201 in
# particular have never been reported at all.
#
# Single process on the non-MPS 10 GB slice; the concurrent training sweep has
# the MPS one, where its workers actually overlap.
#SBATCH --job-name=c7-rescore
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --mem=6G
#SBATCH --gres=gpu:a100_2g.10gb:1
#SBATCH --time=06:00:00
#SBATCH --output=slurm-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the BenchMARL repository root}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1
output="${SLURM_SUBMIT_DIR}/outputs/c7_rescore_${SLURM_JOB_ID}"
mkdir -p "$output"
exec > >(tee -a "$output/batch.log") 2>&1
printf 'C7 rescore: job=%s\n' "$SLURM_JOB_ID"
printf 'CUDA_VISIBLE_DEVICES=%s\n' "${CUDA_VISIBLE_DEVICES:?}"
git rev-parse HEAD > "$output/commit.txt"
git status --short > "$output/git_status.txt"
git diff --binary > "$output/working_tree.patch"
mkdir "$output/source"
cp examples/world_model/*.py "$output/source/"
cp scripts/slurm/c7_rescore.sbatch "$output/"

rescore() {
    local name="$1" runs="$2" data="$3"
    printf '\n=== %s ===\n' "$name"
    .venv/bin/python -u -m examples.world_model.counterfactual_evaluation \
        "$runs" --data "$data" --device cuda \
        2>&1 | tee "$output/${name}.txt"
}

rescore transport   outputs/interaction_control_1194/transport outputs/transport_data_1190
rescore dropout     outputs/interaction_control_1194/dropout   outputs/dropout_control_1193/data
rescore buzz_wire   outputs/buzz_wire_1196/baselines           outputs/buzz_wire_1196/data
rescore wheel       outputs/wheel_1205/baselines               outputs/wheel_1205/data
rescore bw_holdout  outputs/bw_remaining_1201/holdout/baselines outputs/bw_remaining_1201/holdout/data

# The six width/regularization settings from job 1201, which the audit notes
# are available for rescoring with no validated control result attached.
for setting in outputs/bw_remaining_1201/sigreg/*/; do
    rescore "sigreg_$(basename "$setting")" "$setting" outputs/buzz_wire_1196/data
done

printf '\nCompleted C7 rescore: %s\n' "$output"
