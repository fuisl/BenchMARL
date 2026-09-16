#!/usr/bin/env bash
# Gate 1 on Buzz Wire: is it controllable at the cadence that works?
#
# This is an audit, and it decides the shape of the paper.
#
# Buzz Wire is the only task whose cross-agent response is measurable in common
# physical coordinates -- job 1239 resolves it at 0.79x probe error, where
# Balance sits 11-70x below the noise floor. So Buzz Wire is where the
# prediction claim lives. But control has only ever been demonstrated on
# Balance, which means the project's "prediction improves, therefore control
# improves" chain currently spans two tasks, and that cross-task confound is the
# review's central objection.
#
# Every previous Buzz Wire control run went through closed_loop.py when it still
# set `receding_horizon = args.horizon` -- see
# outputs/closed_loop_sweep_1218/source/closed_loop.py:344. That executed all 25
# primitive actions before the next observation. On Balance, fixing exactly this
# was worth 2.4x return (+22.4 -> +54.3) and took the failure rate from 47% to
# zero. Buzz Wire has NEVER been run at the working cadence, so its recorded
# control failures do not establish that it cannot be controlled.
#
# If the oracle controls Buzz Wire here, prediction and control collapse onto one
# task with the same models and seeds, and the cross-task objection disappears.
# If it does not, prediction on Buzz Wire is the single complete result and
# control becomes a stated limitation rather than a claim. Either answer is
# worth ten minutes.
#
# Buzz Wire ships no HeuristicPolicy, so the references are random, do-nothing
# and the true-simulator planner. There is no hand-written policy to beat --
# which also means the goal source falls back to the reward oracle, costing one
# extra planned rollout per invocation.
#SBATCH --job-name=bw-control
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --gres=gpu:a100_3g.20gb:1
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the BenchMARL repository root}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1
data="${BW_DATA:-outputs/buzz_wire_1196/data}"
runs="${BW_RUNS:-outputs/buzz_wire_1196/baselines}"
roots="${BW_ROOTS:-32}"
output="${SLURM_SUBMIT_DIR}/outputs/buzz_wire_control_${SLURM_JOB_ID}"
mkdir -p "$output"
exec > >(tee -a "$output/batch.log") 2>&1
printf 'Buzz Wire control gate: job=%s data=%s roots=%s\n' "$SLURM_JOB_ID" "$data" "$roots"
printf 'CUDA_VISIBLE_DEVICES=%s\n' "${CUDA_VISIBLE_DEVICES:?}"
git rev-parse HEAD > "$output/commit.txt"
git status --short > "$output/git_status.txt"
git diff --binary > "$output/working_tree.patch"
mkdir "$output/source"
cp examples/world_model/*.py "$output/source/"
cp scripts/slurm/buzz_wire_control.sbatch "$output/"

.venv/bin/python -u scripts/slurm/monitor_gpu_memory.py "$output/gpu_memory.csv" &
gpu_monitor_pid=$!
trap 'kill "$gpu_monitor_pid" 2>/dev/null || true' EXIT

# --max-runs 0 scores references only; no learned model is involved. Roots are
# full-length episodes from the train split, leaving the 16 test roots frozen.
cadence() {
    local blocks="$1"
    printf '\n=== H=5, execute %s block(s) per decision ===\n' "$blocks"
    .venv/bin/python -u -m examples.world_model.closed_loop \
        "$runs" --data "$data" --device cuda \
        --states "$roots" --roots-split train \
        --horizon 5 --execute-blocks "$blocks" \
        --num-samples 300 --num-iters 30 \
        --objectives reward --max-runs 0 \
        --output "$output/execute${blocks}.json" \
        2>&1 | tee "$output/execute${blocks}.txt"
}

# The broken cadence first, so the comparison is against what was actually run.
cadence 5
cadence 1

printf '\nCompleted Buzz Wire control gate: %s\n' "$output"
