#!/usr/bin/env bash
# Gate 1 (review 2026-09-16, section 7): can Balance be controlled at all, with
# TRUE dynamics? No learned model appears anywhere in this job.
#
# Every control result so far is bounded by a task property rather than by the
# world model, and the project has never established that its planner works on a
# task where the objective is expressible. Until that is shown, a learned-model
# control failure cannot be attributed to the model.
#
# Two things are being measured, both of which the review identified:
#
# 1. CADENCE. closed_loop.py tied executed blocks to the planning horizon until
#    commit 43432b1, so at H=5 and action_block=5 all 25 primitive actions ran
#    before the next observation: four decisions in a 100-step episode, and a
#    warm start of all zeros because the whole plan was shifted away. Both
#    configurations below plan H=5; they differ only in how much they execute.
#    Per-decision search budget is identical, so episode compute is NOT --
#    execute-1 makes five times as many planning calls, which is reported.
#
# 2. INDEPENDENT ROOTS. Balance's 197 test anchors come from 16 root episodes
#    sampled at steps 0/20/40/60/80, so an interval over anchors understates
#    uncertainty and a step-80 anchor has 20 steps of episode left. --roots-split
#    takes one full-length episode per root. Development uses TRAIN roots and
#    leaves the 16 test roots frozen for the final comparison; nothing here is
#    learned, so a train root leaks nothing.
#
# Objective is the native task reward (--oracles reward). The observation-space
# goal is a separate question and a separate run: the review asks for the native
# objective first, and scoring both would double a multi-hour job.
#
# Sizing is measured, not guessed. One Balance CEM decision at 300 samples and
# 30 iterations took 132s for 8 states on a contended GPU 0; 32 roots is ~4x the
# environments, so execute-5 (4 decisions) is well under an hour and execute-1
# (20 decisions) is a few hours. The cheap configuration runs first and
# closed_loop persists after every policy, so a wall-clock kill loses at most
# the policy in flight.
#SBATCH --job-name=balance-control
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --gres=gpu:a100_3g.20gb:1
#SBATCH --time=16:00:00
#SBATCH --output=slurm-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the BenchMARL repository root}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1
data="${BALANCE_DATA:?Set BALANCE_DATA to the bank collected on the fixed collector}"
roots="${BALANCE_ROOTS:-32}"
# Only ever globbed for checkpoints, and --max-runs 0 keeps none of them; it is
# named honestly rather than pointed at the bank.
runs="${BALANCE_RUNS:-$(dirname "$data")/baselines}"
output="${SLURM_SUBMIT_DIR}/outputs/balance_control_${SLURM_JOB_ID}"
mkdir -p "$output"
exec > >(tee -a "$output/batch.log") 2>&1
printf 'Balance control gate: job=%s data=%s roots=%s\n' "$SLURM_JOB_ID" "$data" "$roots"
printf 'CUDA_VISIBLE_DEVICES=%s\n' "${CUDA_VISIBLE_DEVICES:?}"
git rev-parse HEAD > "$output/commit.txt"
git status --short > "$output/git_status.txt"
git diff --binary > "$output/working_tree.patch"
mkdir "$output/source"
cp examples/world_model/*.py "$output/source/"
cp scripts/slurm/balance_control.sbatch "$output/"

# Refuse a bank collected before the source-policy repair. Its heuristic column
# would be Transport's policy and the whole comparison would be meaningless.
.venv/bin/python - "$data" <<'PY'
import json
import sys
from pathlib import Path

manifest = json.loads((Path(sys.argv[1]) / "manifest.json").read_text())
policy = manifest.get("heuristic_policy")
assert policy == "vmas.scenarios.balance.HeuristicPolicy", (
    f"bank was collected with {policy!r}; see outputs/balance_1233/DEFECT.md"
)
print(f"bank heuristic policy: {policy}", flush=True)
PY

.venv/bin/python -u scripts/slurm/monitor_gpu_memory.py "$output/gpu_memory.csv" &
gpu_monitor_pid=$!
trap 'kill "$gpu_monitor_pid" 2>/dev/null || true' EXIT

# `--max-runs 0` scores the references only: random, do-nothing, the shipped
# Balance heuristic, and CEM-MPC on the true simulator. Do-nothing is the
# baseline the 2026-09-15 audit asked for and that beat random on goal distance
# in every previous table.
cadence() {
    local blocks="$1"
    printf '\n=== H=5, execute %s block(s) per decision ===\n' "$blocks"
    .venv/bin/python -u -m examples.world_model.closed_loop \
        "$runs" --data "$data" --device cuda \
        --states "$roots" --roots-split train \
        --horizon 5 --execute-blocks "$blocks" \
        --num-samples 300 --num-iters 30 \
        --oracles reward --max-runs 0 \
        --output "$output/execute${blocks}.json" \
        2>&1 | tee "$output/execute${blocks}.txt"
}

cadence 5
cadence 1

printf '\nCompleted Balance control gate: %s\n' "$output"
