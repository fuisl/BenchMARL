#!/usr/bin/env bash
# Gate 5 is diagnostic only: no model fitting and no simulator truth in CEM.
# It replays the saved CEM populations in VMAS only after candidate selection.
#SBATCH --job-name=planner-tail
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

source_job="${GATE5_SOURCE:-outputs/structured_surrogate_1331}"
output="${SLURM_SUBMIT_DIR}/outputs/planner_tail_${SLURM_JOB_ID}"
mkdir -p "$output/source"
exec > >(tee -a "$output/batch.log") 2>&1

printf 'Gate 5: job=%s source=%s CUDA_VISIBLE_DEVICES=%s\n' \
    "$SLURM_JOB_ID" "$source_job" "${CUDA_VISIBLE_DEVICES:?}"
test -s "$source_job/result/result.json"
test -s "$source_job/coverage/oracle_plans.pt"
git rev-parse HEAD > "$output/commit.txt"
git status --short > "$output/git_status.txt"
git diff --binary > "$output/working_tree.patch"
cp examples/world_model/{planner_tail_failure,structured_surrogate,cem,decision_information,mpc,snapshot_restore}.py "$output/source/"
cp scripts/slurm/planner_tail_gate.sbatch "$output/"

.venv/bin/python -u scripts/slurm/monitor_gpu_memory.py "$output/gpu_memory.csv" &
gpu_monitor_pid=$!
trap 'kill "$gpu_monitor_pid" 2>/dev/null || true' EXIT

.venv/bin/python -m pytest -q \
    test/test_world_model_planner_tail_failure.py \
    test/test_world_model_structured_surrogate.py \
    test/test_world_model_decision_information.py \
    test/test_world_model.py::test_cem_batch_reproducibility_and_validation \
    | tee "$output/preflight_tests.txt"

.venv/bin/python -u -m examples.world_model.planner_tail_failure \
    --source "$source_job" --output "$output/result" --device cuda \
    2>&1 | tee "$output/gate5.log"

test -s "$output/result/result.json"
test -s "$output/result/information_contract.json"
test -s "$output/result/tail_physics.png"
test -s "$output/result/tail_decisions.png"

printf '\nPeak GPU memory:\n'
awk -F, 'NR>1 {gsub(/ MiB/,"",$4); if ($4+0>m) m=$4+0} END {print "  " m " MiB"}' \
    "$output/gpu_memory.csv"
printf 'Completed Gate 5: %s\n' "$output"
