#!/usr/bin/env bash
# G0c: does the CROSS recovery fraction depend on how the head is fitted?
#
# The admission benchmark reads R_O = 0.674 for Buzz Wire's cross-agent effect.
# G0b read 0.016 for the same task. Two design differences could cause that and
# they are not separated:
#
#   (1) head fitting  -- T-A2b/G0/G0b fit ONE head to the full dY and scored a
#                        column subset. The admission benchmark fits the cross
#                        columns directly. On Buzz Wire the shared response is
#                        7.43 against a 2.24 cross term, so a pooled head
#                        minimising MSE spends its capacity elsewhere.
#   (2) grid design   -- G0b samples the non-intervened action; the 3x3 grid
#                        pins it at the midpoint.
#
# This isolates (1): identical G0 intervention design, identical anchors, seeds,
# scale and head family, changing ONLY --fit-target. If cross-only fitting lands
# near 0.67, G0b's instrument was the limitation and K29/K30 need revising. If
# it stays near 0.016, the grid design is responsible and the admission
# benchmark needs re-examining before any scenario verdict is trusted.
#
# Shared conditions only: the floor, raw observation, history and privileged
# reference are model-independent, so this is minutes rather than hours.
#SBATCH --job-name=g0c-fittarget
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

runs="${G0C_RUNS:?Set G0C_RUNS}"
jacobian="${G0C_JACOBIAN:?Set G0C_JACOBIAN}"
data="${G0C_DATA:-outputs/buzz_wire_1196/data}"
output="${SLURM_SUBMIT_DIR}/outputs/g0c_fit_target_${SLURM_JOB_ID}"
mkdir -p "$output/source"
exec > >(tee -a "$output/batch.log") 2>&1

printf 'G0c fit-target audit: job=%s\n' "$SLURM_JOB_ID"
printf 'CUDA_VISIBLE_DEVICES=%s\n' "${CUDA_VISIBLE_DEVICES:?}"
git rev-parse HEAD > "$output/commit.txt"
git status --short > "$output/git_status.txt"
cp examples/world_model/counterfactual_localization.py "$output/source/"
cp scripts/slurm/g0c_fit_target_audit.sbatch "$output/"

.venv/bin/python -m pytest -q test/test_world_model_interaction_jacobian.py \
    | tee "$output/preflight_tests.txt"

for mode in full cross; do
    printf '\n=== fit-target: %s ===\n' "$mode"
    srun --ntasks=1 .venv/bin/python -u -m examples.world_model.counterfactual_localization \
        "$runs" --data "$data" --jacobian "$jacobian" \
        --out "$output/result_$mode" --fit-target "$mode" --shared-only \
        --device cuda 2>&1 | tee "$output/run_$mode.log"
    test -s "$output/result_$mode/counterfactual_localization.json"
done
printf '\nCompleted G0c: %s\n' "$output"
