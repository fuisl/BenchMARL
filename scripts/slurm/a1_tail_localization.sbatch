#!/usr/bin/env bash
# A1.2 -- full32 planner-tail localization.
#
# Question: does late-CEM teacher-forced error still grow after removing the
# state alias that A1.1 demonstrated in legacy14?
#
# One paired collection emits legacy14 and full32 banks whose every non-state
# column is bit identical, so the only difference between the arms is the state
# representation. Three arms are trained and measured under one protocol:
#   full32_h256     the successor state, at the standard width
#   legacy14_h256   the historical job-1331 width, for lineage comparability
#   legacy14_h264   capacity matched to full32_h256 (153,135 vs 153,633 params)
# Control is withheld (--skip-control): this pass is ranking and tail
# localization only, so nothing can be selected on closed-loop performance.
#SBATCH --job-name=a1-tail-localization
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --gres=gpu:a100_3g.20gb:1
#SBATCH --time=06:00:00
#SBATCH --output=slurm-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the BenchMARL repository root}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="${SLURM_SUBMIT_DIR}${PYTHONPATH:+:$PYTHONPATH}"

data="${A12_DATA:-outputs/buzz_wire_1196/data}"
runs="${A12_RUNS:-outputs/state_input_1223/runs}"
seeds="${A12_SEEDS:-9100,9101,9102}"
output="${SLURM_SUBMIT_DIR}/outputs/a1_tail_localization_${SLURM_JOB_ID}"
coverage="$output/coverage"
mkdir -p "$coverage" "$output/source"
exec > >(tee -a "$output/batch.log") 2>&1

printf 'A1.2: job=%s data=%s runs=%s seeds=%s\n' \
    "$SLURM_JOB_ID" "$data" "$runs" "$seeds"
printf 'CUDA_VISIBLE_DEVICES=%s\n' "${CUDA_VISIBLE_DEVICES:?}"
test -f "$data/manifest.json"

git rev-parse HEAD > "$output/commit.txt"
git status --short > "$output/git_status.txt"
git diff --binary > "$output/working_tree.patch"
cp examples/world_model/{structured_surrogate,planner_tail_failure,cem,mpc,decision_information,snapshot_restore,models}.py "$output/source/"
cp scripts/slurm/a1_tail_localization.sbatch "$output/"

.venv/bin/python -u scripts/slurm/monitor_gpu_memory.py "$output/gpu_memory.csv" &
gpu_monitor_pid=$!
trap 'kill "$gpu_monitor_pid" 2>/dev/null || true' EXIT

.venv/bin/python -m pytest -q \
    test/test_world_model_structured_surrogate.py \
    test/test_world_model_planner_tail_failure.py \
    test/test_world_model_decision_information.py \
    test/test_world_model.py::test_cem_batch_reproducibility_and_validation \
    | tee "$output/preflight_tests.txt"

printf '\n=== Stage 1: paired coverage collection (legacy14 + full32) ===\n'
.venv/bin/python -u -m examples.world_model.structured_surrogate collect \
    --data "$data" --runs "$runs" --output "$coverage" --device cuda \
    --state-profile legacy14 full32 \
    --model-seed 4100 --kind joint --seed 8400 \
    --horizon 5 --num-samples 300 --num-elites 30 \
    --root-batch 8 --replay-batch 256 \
    --hard-train 256 --hard-validation 64 --hard-test 64 --hard-plans 4 \
    --oracle-root-batch 8 --oracle-iters 30 --oracle-seed 8500 \
    --local-plans 8 --local-sigma-small 0.05 --local-sigma-large 0.15 \
    2>&1 | tee "$output/collection.log"

for profile in legacy14 full32; do
    test -s "$coverage/$profile/coverage.pt"
    test -s "$coverage/$profile/oracle_plans.pt"
    test -s "$coverage/$profile/collection_summary.json"
done

run_arm() {
    local arm="$1" profile="$2" hidden="$3"
    printf '\n=== Arm %s (%s, hidden=%s) ===\n' "$arm" "$profile" "$hidden"
    mkdir -p "$output/arm_$arm/result"
    .venv/bin/python -u -m examples.world_model.structured_surrogate run \
        --data "$data" --coverage "$coverage/$profile" \
        --output "$output/arm_$arm/result" --device cuda \
        --seeds "$seeds" --hidden "$hidden" \
        --epochs 200 --patience 25 --random-plans 300 --bank-seed 8600 \
        --horizon 5 --num-samples 300 --num-elites 30 --num-iters 30 \
        --control-seed 8700 --skip-control \
        2>&1 | tee "$output/train_$arm.log"
    test -s "$output/arm_$arm/result/result.json"

    .venv/bin/python -u -m examples.world_model.planner_tail_failure \
        --protocol a1_paired --source "$output/arm_$arm" \
        --coverage "$coverage/$profile" --seeds "$seeds" \
        --output "$output/tail_$arm" --device cuda \
        2>&1 | tee "$output/tail_$arm.log"
    test -s "$output/tail_$arm/result.json"
}

run_arm full32_h256   full32   256
run_arm legacy14_h256 legacy14 256
run_arm legacy14_h264 legacy14 264

printf '\n=== Stage 3: paired comparison ===\n'
.venv/bin/python -u -m examples.world_model.a1_tail_comparison \
    --run "$output" --output "$output/comparison" \
    2>&1 | tee "$output/comparison.log"
test -s "$output/comparison/comparison.json"

printf '\nPeak GPU memory:\n'
awk -F, 'NR>1 {gsub(/ MiB/,"",$4); if ($4+0>m) m=$4+0} END {print "  " m " MiB"}' \
    "$output/gpu_memory.csv"
printf 'Completed A1.2: %s\n' "$output"
