#!/usr/bin/env bash
# Gate 0e runs in a strict sequence: planner-induced data collection first,
# followed by the structured physical surrogate (Baseline B).
#SBATCH --job-name=structured-wm
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --gres=gpu:a100_3g.20gb:1
#SBATCH --time=08:00:00
#SBATCH --output=slurm-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the BenchMARL repository root}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="${SLURM_SUBMIT_DIR}${PYTHONPATH:+:$PYTHONPATH}"

data="${STRUCTURED_DATA:-outputs/buzz_wire_1196/data}"
runs="${STRUCTURED_RUNS:-outputs/state_input_1223/runs}"
output="${SLURM_SUBMIT_DIR}/outputs/structured_surrogate_${SLURM_JOB_ID}"
coverage="$output/coverage"
result="$output/result"
mkdir -p "$coverage" "$result" "$output/source"
exec > >(tee -a "$output/batch.log") 2>&1

printf 'Gate 0e: job=%s data=%s runs=%s CUDA_VISIBLE_DEVICES=%s\n' \
    "$SLURM_JOB_ID" "$data" "$runs" "${CUDA_VISIBLE_DEVICES:?}"
git rev-parse HEAD > "$output/commit.txt"
git status --short > "$output/git_status.txt"
git diff --binary > "$output/working_tree.patch"
cp examples/world_model/{structured_surrogate,cem,collect,decision_information,models,mpc}.py "$output/source/"
cp scripts/slurm/structured_surrogate_gate.sbatch "$output/"

.venv/bin/python -u scripts/slurm/monitor_gpu_memory.py "$output/gpu_memory.csv" &
gpu_monitor_pid=$!
trap 'kill "$gpu_monitor_pid" 2>/dev/null || true' EXIT

.venv/bin/python -m pytest -q \
    test/test_world_model_structured_surrogate.py \
    test/test_world_model_decision_information.py \
    test/test_world_model.py::test_cem_batch_reproducibility_and_validation \
    | tee "$output/preflight_tests.txt"

printf '\n=== Stage 1: planner-induced coverage collection ===\n'
.venv/bin/python -u -m examples.world_model.structured_surrogate collect \
    --data "$data" --runs "$runs" --output "$coverage" --device cuda \
    --model-seed 4100 --kind joint --seed 8400 \
    --horizon 5 --num-samples 300 --num-elites 30 \
    --root-batch 8 --replay-batch 256 \
    --hard-train 256 --hard-validation 64 --hard-test 64 --hard-plans 4 \
    --oracle-root-batch 8 --oracle-iters 30 --oracle-seed 8500 \
    --local-plans 8 --local-sigma-small 0.05 --local-sigma-large 0.15 \
    2>&1 | tee "$output/collection.log"

test -s "$coverage/coverage.pt"
test -s "$coverage/collection_summary.json"

printf '\n=== Stage 2: structured physical surrogate ===\n'
.venv/bin/python -u -m examples.world_model.structured_surrogate run \
    --data "$data" --coverage "$coverage" --output "$result" --device cuda \
    --seeds 9100,9101,9102 --hidden 256 --batch-size 512 \
    --epochs 200 --patience 25 --learning-rate 3e-4 --weight-decay 1e-4 \
    --random-plans 300 --bank-seed 8600 --topk 30 \
    --horizon 5 --num-samples 300 --num-elites 30 --num-iters 30 \
    --control-seed 8700 2>&1 | tee "$output/structured.log"

printf '\nPeak GPU memory:\n'
awk -F, 'NR>1 {gsub(/ MiB/,"",$4); if ($4+0>m) m=$4+0} END {print "  " m " MiB"}' \
    "$output/gpu_memory.csv"
printf 'Completed Gate 0e: %s\n' "$output"
