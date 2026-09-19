#!/usr/bin/env bash
# Experiment 24/G6a: planner-aware data aggregation, two DAgger rounds, three
# seeds, with a matched-size generic-augmentation control. Reruns Gate 5
# unchanged on every round/arm to test whether the registered OOD signature
# collapses. The 16 frozen Gate-5 test roots never enter collection or
# training; only job 1331's own oracle_plans.pt (unchanged) supplies them.
#SBATCH --job-name=g6a-aggregation
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

source_job="${G6A_SOURCE:-outputs/structured_surrogate_1331}"
gate5_baseline="${G6A_GATE5_BASELINE:-outputs/planner_tail_1335}"
data="${G6A_DATA:-outputs/buzz_wire_1196/data}"
output="${SLURM_SUBMIT_DIR}/outputs/g6a_aggregation_${SLURM_JOB_ID}"
mkdir -p "$output/source"
exec > >(tee -a "$output/batch.log") 2>&1

printf 'G6a: job=%s source=%s data=%s CUDA_VISIBLE_DEVICES=%s\n' \
    "$SLURM_JOB_ID" "$source_job" "$data" "${CUDA_VISIBLE_DEVICES:?}"
test -s "$source_job/result/result.json"
test -s "$source_job/coverage/coverage.pt"
test -s "$source_job/coverage/oracle_plans.pt"
test -s "$gate5_baseline/result/result.json"
git rev-parse HEAD > "$output/commit.txt"
git status --short > "$output/git_status.txt"
git diff --binary > "$output/working_tree.patch"
cp examples/world_model/{planner_aware_aggregation,g6a_report,planner_tail_failure,structured_surrogate,cem,mpc,decision_information,snapshot_restore}.py "$output/source/"
cp scripts/slurm/g6a_aggregation_gate.sbatch "$output/"

.venv/bin/python -u scripts/slurm/monitor_gpu_memory.py "$output/gpu_memory.csv" &
gpu_monitor_pid=$!
trap 'kill "$gpu_monitor_pid" 2>/dev/null || true' EXIT

.venv/bin/python -m pytest -q \
    test/test_world_model_planner_aware_aggregation.py \
    test/test_world_model_g6a_report.py \
    test/test_world_model_planner_tail_failure.py \
    test/test_world_model_structured_surrogate.py \
    test/test_world_model.py::test_cem_batch_reproducibility_and_validation \
    | tee "$output/preflight_tests.txt"

printf '\n=== Stage 1: planner-aware aggregation (2 rounds, 3 seeds, generic control) ===\n'
.venv/bin/python -u -m examples.world_model.planner_aware_aggregation \
    --source "$source_job" --data "$data" --output "$output" --device cuda \
    --rounds 2 --root-batch 8 --replay-batch 256 --hard-plans 4 \
    2>&1 | tee "$output/aggregation.log"
test -s "$output/aggregation_summary.json"

printf '\n=== Stage 2: rerun Gate 5 unchanged on every round/arm ===\n'
for source_dir in g6a_targeted_round1 g6a_targeted_round2 g6a_generic_round1 g6a_generic_round2; do
    printf 'Gate 5 on %s\n' "$source_dir"
    .venv/bin/python -u -m examples.world_model.planner_tail_failure \
        --source "$output/$source_dir" --output "$output/$source_dir/gate5_result" --device cuda \
        2>&1 | tee "$output/gate5_${source_dir}.log"
    test -s "$output/$source_dir/gate5_result/result.json"
done

printf '\n=== Stage 3: comparison report against the frozen job-1335 baseline ===\n'
.venv/bin/python -u -m examples.world_model.g6a_report \
    --baseline "$gate5_baseline/result" \
    --targeted-round1 "$output/g6a_targeted_round1/gate5_result" \
    --targeted-round2 "$output/g6a_targeted_round2/gate5_result" \
    --generic-round1 "$output/g6a_generic_round1/gate5_result" \
    --generic-round2 "$output/g6a_generic_round2/gate5_result" \
    --output "$output/g6a_report.json" \
    2>&1 | tee "$output/g6a_report.log"
test -s "$output/g6a_report.json"

printf '\nPeak GPU memory:\n'
awk -F, 'NR>1 {gsub(/ MiB/,"",$4); if ($4+0>m) m=$4+0} END {print "  " m " MiB"}' \
    "$output/gpu_memory.csv"
printf 'Completed G6a aggregation and re-diagnosis: %s\n' "$output"
