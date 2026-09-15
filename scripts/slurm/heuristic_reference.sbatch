#!/usr/bin/env bash
# References at full budget with VMAS's own heuristic added.
#
# Every comparison so far has had random as its only non-planning baseline, and
# random is uniform noise resampled every primitive step -- a floor, not a
# comparator. A reduced-budget probe put the Transport heuristic at return 3.179
# with 1/20 native successes, against the full-budget reward oracle's 1.278 and
# zero successes from every policy ever run on this task. If that survives here,
# "oracle" is not a task ceiling at all: it is CEM-MPC at horizon 5 with perfect
# dynamics, and a reactive hand-written policy beats it.
#
# That is worth knowing exactly, so this recomputes the full reference set at the
# real budget rather than trusting the probe.
#
# Buzz Wire ships no heuristic and is here for its unchanged references under the
# new cache identity; its column is absent rather than faked.
#
# References only -- no checkpoints are scored, so this is short and runs the
# oracles alone, which is the one workload that collapses when it shares a slice.
#SBATCH --job-name=heuristic-ref
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:a100_3g.20gb:1
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the BenchMARL repository root}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1
states="${CLOSED_LOOP_STATES:-20}"
output="${SLURM_SUBMIT_DIR}/outputs/heuristic_ref_${SLURM_JOB_ID}"
mkdir -p "$output"
exec > >(tee -a "$output/batch.log") 2>&1
printf 'Heuristic references: job=%s states=%s\n' "$SLURM_JOB_ID" "$states"
printf 'CUDA_VISIBLE_DEVICES=%s\n' "${CUDA_VISIBLE_DEVICES:?}"
git rev-parse HEAD > "$output/commit.txt"
git status --short > "$output/git_status.txt"
git diff --binary > "$output/working_tree.patch"
mkdir "$output/source"
cp examples/world_model/*.py "$output/source/"
cp scripts/slurm/heuristic_reference.sbatch "$output/"

.venv/bin/python -u scripts/slurm/monitor_gpu_memory.py "$output/gpu_memory.csv" &
gpu_monitor_pid=$!
trap 'kill "$gpu_monitor_pid" 2>/dev/null || true' EXIT

srun --ntasks=1 .venv/bin/python -m pytest \
    test/test_world_model_stage0.py test/test_world_model_closed_loop.py -q \
    | tee "$output/contract_tests.txt"

reference() {
    local name="$1" runs="$2" data="$3"
    printf '\n=== references: %s ===\n' "$name"
    .venv/bin/python -u -m examples.world_model.closed_loop \
        "$runs" --data "$data" --device cuda --states "$states" \
        --reference-cache "$output/${name}_refs.pt" --max-runs 0 \
        --output "$output/${name}_references.json" --wandb \
        2>&1 | tee "$output/${name}_references.txt"
}
reference transport outputs/stage1_refit_1217/transport outputs/transport_data_1190
reference buzz_wire outputs/stage1_refit_1217/buzz_wire outputs/buzz_wire_1196/data

printf '\nCompleted: %s\n' "$output"
