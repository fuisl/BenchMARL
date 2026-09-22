#!/usr/bin/env bash
# Do the two E_CF aggregations agree on Buzz Wire?
#
# Job 1527 returned E ~ 1e10 on every input because `E_CF` divided by
# ||dY_true|| PER ANCHOR and then averaged. That is stable only where every
# anchor's effect clears the floor. Buzz Wire satisfies it (100% active, median
# ~ mean); Balance at h=3 does not (56% active, mean 11x median).
#
# The registered fix pools first:  E_CF = sqrt( sum ||dYhat - dY||^2
#                                             / sum ||dY||^2 )
# A head predicting no response still scores exactly 1, so the property the
# frozen convention rests on is preserved (test:
# `test_pooled_ratio_scores_a_no_response_head_at_exactly_one`).
#
# This job re-runs job 1516 UNCHANGED except that both forms are now computed
# side by side on the same fits, same anchors, same seeds. It is the gate:
#
#   agree    -> the pooled form replaces the mean of ratios in the convention,
#               and the Balance re-run is interpretable.
#   disagree -> the published O = 0.605 / Z = 0.212 need re-deriving before
#               anything else proceeds, and this job's pooled column IS that
#               re-derivation.
#
# Nothing about the model, the bank or the intervention design changes. This is
# an instrument change only.
#
# SIZING, measured: job 1516 ran this exact configuration in 50:31 on one
# 3g.20gb slice with 32G. The added cost is one extra bootstrap per block.
#SBATCH --job-name=ecf-agree
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:a100_3g.20gb:1
#SBATCH --time=03:00:00
#SBATCH --output=slurm-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the BenchMARL repository root}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="${SLURM_SUBMIT_DIR}${PYTHONPATH:+:$PYTHONPATH}"

runs="${AG_RUNS:-outputs/ta2_reference_baselines_1498/runs}"
jacobian="${AG_JACOBIAN:-outputs/ta1_interaction_jacobian_1497/result/interaction_jacobian.json}"
data="${AG_DATA:-outputs/buzz_wire_1196/data}"
cells="${AG_CELLS:-1:0}"
output="${SLURM_SUBMIT_DIR}/outputs/ecf_agreement_${SLURM_JOB_ID}"
mkdir -p "$output/source"
exec > >(tee -a "$output/batch.log") 2>&1

printf 'E_CF aggregation agreement: job=%s cells=%s fit-target=cross\n' \
    "$SLURM_JOB_ID" "$cells"
printf 'runs=%s\njacobian=%s\n' "$runs" "$jacobian"
printf 'CUDA_VISIBLE_DEVICES=%s\n' "${CUDA_VISIBLE_DEVICES:?}"
test -d "$runs"
test -f "$jacobian"

git rev-parse HEAD > "$output/commit.txt"
git status --short > "$output/git_status.txt"
git diff --binary > "$output/working_tree.patch"
cp examples/world_model/{counterfactual_localization,counterfactual_fidelity,interaction_jacobian}.py \
    "$output/source/"
cp scripts/slurm/ecf_aggregation_agreement.sbatch "$output/"
cp outputs/ta2b_reconvention_1516/result/counterfactual_localization.json \
    "$output/job_1516_reference.json" 2>/dev/null || true

.venv/bin/python -u scripts/slurm/monitor_gpu_memory.py "$output/gpu_memory.csv" &
gpu_monitor_pid=$!
trap 'kill "$gpu_monitor_pid" 2>/dev/null || true' EXIT

.venv/bin/python -m pytest -q test/test_world_model_interaction_jacobian.py \
    | tee "$output/preflight_tests.txt"

# The intervention contract: these must reproduce T-A1's recorded cell means
# exactly, or the two aggregations would be compared on different physics.
srun --ntasks=1 .venv/bin/python -u -m examples.world_model.counterfactual_localization \
    "$runs" --data "$data" --jacobian "$jacobian" \
    --out "$output/result" --fit-target cross --cells $cells \
    --device cuda 2>&1 | tee "$output/run.log"

test -s "$output/result/counterfactual_localization.json"
printf '\nCompleted E_CF aggregation agreement: %s\n' "$output"
