#!/usr/bin/env bash
# G0d: is CELL POOLING the remaining factor?
#
# Job 1514 established head fitting at fixed pooling: 0.016 -> 0.293. Moving the
# admission ladder to the sampled-reference design then read 0.683 against the
# midpoint grid's 0.674, so the conditional was not the difference and my
# "grid design carries 0.293 -> 0.674" attribution was withdrawn.
#
# The factor left uncontrolled is cells per fit. G0's build_rows concatenates
# every (reference, intervened, axis) cell into one design matrix, so a single
# head serves four heterogeneous intervention cells. The admission benchmark
# fits one head per axis, reading 0.683 on axis 0 and 0.339 on axis 1.
#
# This holds the intervention design, anchors, seeds, scale, head family and
# fit-target all fixed, and changes ONLY which cells enter the fit. If the
# single-cell arm reproduces roughly 0.68, cell pooling is the whole remaining
# story and the benchmark need only declare its convention. If it stays near
# 0.293, something else is still uncontrolled and the instrument should be
# re-derived rather than patched again.
#SBATCH --job-name=g0d-pooling
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --gres=gpu:a100_3g.20gb:1
#SBATCH --time=03:00:00
#SBATCH --output=slurm-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the BenchMARL repository root}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="${SLURM_SUBMIT_DIR}${PYTHONPATH:+:$PYTHONPATH}"

runs="${G0D_RUNS:?Set G0D_RUNS}"
jacobian="${G0D_JACOBIAN:?Set G0D_JACOBIAN}"
data="${G0D_DATA:-outputs/buzz_wire_1196/data}"
output="${SLURM_SUBMIT_DIR}/outputs/g0d_cell_pooling_${SLURM_JOB_ID}"
mkdir -p "$output/source"
exec > >(tee -a "$output/batch.log") 2>&1

printf 'G0d cell-pooling audit: job=%s\n' "$SLURM_JOB_ID"
printf 'CUDA_VISIBLE_DEVICES=%s\n' "${CUDA_VISIBLE_DEVICES:?}"
git rev-parse HEAD > "$output/commit.txt"
git status --short > "$output/git_status.txt"
cp examples/world_model/counterfactual_localization.py "$output/source/"
cp scripts/slurm/g0d_cell_pooling.sbatch "$output/"

.venv/bin/python -m pytest -q test/test_world_model_interaction_jacobian.py \
    | tee "$output/preflight_tests.txt"

run_arm () {
    local label="$1"; shift
    printf '\n=== %s ===\n' "$label"
    srun --ntasks=1 .venv/bin/python -u -m examples.world_model.counterfactual_localization \
        "$runs" --data "$data" --jacobian "$jacobian" \
        --out "$output/result_$label" --fit-target cross --shared-only \
        --device cuda "$@" 2>&1 | tee "$output/run_$label.log"
    test -s "$output/result_$label/counterfactual_localization.json"
}

# Pooled: every cell in one head -- G0's convention, expected ~0.293.
run_arm pooled_all_cells
# Single cell: agent 1 moved on axis 0, read agent 0 -- the admission
# benchmark's convention, expected ~0.68 if pooling is the factor.
run_arm single_cell_1x0 --cells 1:0
# The mirrored cell, to show the effect is not specific to one direction.
run_arm single_cell_0x0 --cells 0:0

printf '\nCompleted G0d: %s\n' "$output"
