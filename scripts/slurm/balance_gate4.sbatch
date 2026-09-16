#!/usr/bin/env bash
# Gate 4 (review 2026-09-16, section 7): does a LEARNED world model control
# Balance, measured against a reference that demonstrably works?
#
# This is the experiment the project has been trying to run since M1, and job
# 1237 is what finally made it answerable. Until today every control result was
# taken on a task where nothing controlled anything, so a learned-model failure
# could not be separated from a task that could not be done. Gate 1 fixed that:
# on 32 independent train roots, at H=5 executing one block, CEM-MPC on the true
# simulator returns +54.310 and closes 35% of the distance doing nothing leaves,
# against the shipped heuristic's +8.807 and 10%, and drops the package in none
# of 32 episodes where the old execute-everything cadence dropped it in 47%.
#
# So there is now a target. A learned model planning through the same CEM, the
# same cadence, the same roots and the same per-decision budget either
# approaches +54.310 or it does not, and the gap is attributable to the model.
#
# Three seeds per cell, as the review asks: "start with a three-seed pilot;
# expand only a functioning comparison to the established eight-seed
# convention". 3 predictors x 2 regimes x 3 seeds = 18 checkpoints.
#
# --objectives reward, because job 1237 measured the observation-space goal to
# be WORSE than doing nothing on Balance even with true dynamics (goal distance
# 4.2068 for the oracle against zero-action's 3.0412, while having the best task
# distance). Scoring a learned model against a broken objective would measure
# the objective, and it would double the job.
#
# Checkpoints come from job 1236's bank, collected after the source-policy
# repair. Roots are the same 32 train roots Gate 1 used, so the oracle row is
# directly comparable rather than approximately so.
#SBATCH --job-name=balance-gate4
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:a100_3g.20gb:1
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the BenchMARL repository root}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1
runs="${GATE4_RUNS:-outputs/balance_repair_1236/baselines}"
data="${GATE4_DATA:-outputs/balance_repair_1236/data}"
roots="${GATE4_ROOTS:-32}"
seeds="${GATE4_SEEDS:-4100,4101,4102}"
output="${SLURM_SUBMIT_DIR}/outputs/balance_gate4_${SLURM_JOB_ID}"
mkdir -p "$output"
exec > >(tee -a "$output/batch.log") 2>&1
printf 'Gate 4: job=%s runs=%s roots=%s seeds=%s\n' "$SLURM_JOB_ID" "$runs" "$roots" "$seeds"
printf 'CUDA_VISIBLE_DEVICES=%s\n' "${CUDA_VISIBLE_DEVICES:?}"
git rev-parse HEAD > "$output/commit.txt"
git status --short > "$output/git_status.txt"
git diff --binary > "$output/working_tree.patch"
mkdir "$output/source"
cp examples/world_model/*.py "$output/source/"
cp scripts/slurm/balance_gate4.sbatch "$output/"

# A bank from before the source-policy repair would put Transport's policy in
# the training split of every checkpoint scored here.
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

# References are rerun rather than reused: --execute-blocks and --objectives are
# both part of `reference_identity`, so a cache from job 1237 would be a
# different experiment. They cost ~4 minutes of the job.
.venv/bin/python -u -m examples.world_model.closed_loop \
    "$runs" --data "$data" --device cuda \
    --states "$roots" --roots-split train \
    --horizon 5 --execute-blocks 1 \
    --num-samples 300 --num-iters 30 \
    --objectives reward --seeds "$seeds" \
    --output "$output/gate4.json" \
    2>&1 | tee "$output/gate4.txt"

printf '\nCompleted Gate 4: %s\n' "$output"
