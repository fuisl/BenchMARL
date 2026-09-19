#!/usr/bin/env bash
# Experiment 25 is diagnostic only: no retraining. It reuses job 1331's own
# K=1 closed-loop run and computes K=2/5 with the same frozen models.
#SBATCH --job-name=feedback-cadence
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --gres=gpu:a100_3g.20gb:1
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the BenchMARL repository root}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="${SLURM_SUBMIT_DIR}${PYTHONPATH:+:$PYTHONPATH}"

source_job="${EXP25_SOURCE:-outputs/structured_surrogate_1331}"
gate5_job="${EXP25_GATE5:-outputs/planner_tail_1335}"
output="${SLURM_SUBMIT_DIR}/outputs/feedback_cadence_${SLURM_JOB_ID}"
mkdir -p "$output/source"
exec > >(tee -a "$output/batch.log") 2>&1

printf 'Experiment 25: job=%s source=%s gate5=%s CUDA_VISIBLE_DEVICES=%s\n' \
    "$SLURM_JOB_ID" "$source_job" "$gate5_job" "${CUDA_VISIBLE_DEVICES:?}"
test -s "$source_job/result/result.json"
test -s "$source_job/coverage/oracle_plans.pt"
test -s "$gate5_job/result/result.json"
git rev-parse HEAD > "$output/commit.txt"
git status --short > "$output/git_status.txt"
git diff --binary > "$output/working_tree.patch"
cp examples/world_model/{feedback_cadence,planner_tail_failure,structured_surrogate,cem,mpc,snapshot_restore,decision_information}.py "$output/source/"
cp scripts/slurm/feedback_cadence_gate.sbatch "$output/"

.venv/bin/python -u scripts/slurm/monitor_gpu_memory.py "$output/gpu_memory.csv" &
gpu_monitor_pid=$!
trap 'kill "$gpu_monitor_pid" 2>/dev/null || true' EXIT

.venv/bin/python -m pytest -q \
    test/test_world_model_feedback_cadence.py \
    test/test_world_model_planner_tail_failure.py \
    test/test_world_model_structured_surrogate.py \
    test/test_world_model.py::test_cem_batch_reproducibility_and_validation \
    | tee "$output/preflight_tests.txt"

.venv/bin/python -u -m examples.world_model.feedback_cadence \
    --source "$source_job" --gate5 "$gate5_job" --output "$output/result" --device cuda \
    2>&1 | tee "$output/experiment25.log"

test -s "$output/result/result.json"
test -s "$output/result/feedback_cadence.png"

printf '\nPeak GPU memory:\n'
awk -F, 'NR>1 {gsub(/ MiB/,"",$4); if ($4+0>m) m=$4+0} END {print "  " m " MiB"}' \
    "$output/gpu_memory.csv"
printf 'Completed Experiment 25: %s\n' "$output"
