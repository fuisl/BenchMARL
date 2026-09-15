#!/usr/bin/env bash
# Stage 1: refit readouts on frozen dynamics, then re-run the diagnostic.
#
# Every existing readout was trained under a mask that required all five
# primitive transitions of a block to be valid. On the Buzz Wire development
# bank that discarded 79% of training terminations together with their -10
# collision penalties -- the dominant cost term.
#
# The diagnosis those checkpoints produced is therefore confounded. A readout
# correlation of -0.25 against the simulator's own plan costs was reported as
# evidence that Buzz Wire's reward is unrecoverable from agent observations. It
# is equally consistent with a readout that was never shown the penalties.
#
# Refitting separates them. Dynamics weights are untouched, so any change comes
# from the corrected mask alone. If the correlation stays negative, the
# observability explanation stands on its own; if it does not, the earlier
# attribution was wrong and needs retracting.
#
# Transport runs for contrast: its readout already correlated at +0.67 and it
# has no terminations at all, so its refit should change very little. A large
# change there would indicate something other than the terminal mask moved.
#
# One process per task, which is what the closed-loop measurements established
# for this slice -- a MIG without MPS time-slices processes, and more of them
# made the launch-heavy task dramatically slower rather than faster.
#SBATCH --job-name=stage1-refit
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100_3g.20gb:1
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the BenchMARL repository root}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1
output="${SLURM_SUBMIT_DIR}/outputs/stage1_refit_${SLURM_JOB_ID}"
mkdir -p "$output"
exec > >(tee -a "$output/batch.log") 2>&1
printf 'Stage 1 readout refit: job=%s\n' "$SLURM_JOB_ID"
printf 'CUDA_VISIBLE_DEVICES=%s\n' "${CUDA_VISIBLE_DEVICES:?}"
git rev-parse HEAD > "$output/commit.txt"
git status --short > "$output/git_status.txt"
git diff --binary > "$output/working_tree.patch"
mkdir "$output/source"
cp examples/world_model/*.py "$output/source/"
cp scripts/slurm/stage1_refit.sbatch "$output/"

.venv/bin/python -u scripts/slurm/monitor_gpu_memory.py "$output/gpu_memory.csv" &
gpu_monitor_pid=$!
trap 'kill "$gpu_monitor_pid" 2>/dev/null || true' EXIT

# The Stage 0 contracts are what make this refit meaningful, so check them in
# the allocation rather than trusting a local pass.
srun --ntasks=1 .venv/bin/python -m pytest test/test_world_model_stage0.py -q \
    | tee "$output/stage0_tests.txt"

# Smoke one checkpoint end to end before committing the allocation.
srun --ntasks=1 .venv/bin/python -m examples.world_model.refit_readout \
    outputs/buzz_wire_1196/baselines --data outputs/buzz_wire_1196/data \
    --output "$output/smoke" --device cuda --max-runs 1 \
    | tee "$output/smoke.txt"

refit_task() {
    local name="$1" runs="$2" data="$3"
    .venv/bin/python -u -m examples.world_model.refit_readout \
        "$runs" --data "$data" --output "$output/$name" --device cuda \
        > "$output/${name}_refit.txt" 2>&1
    # The diagnostic that produced the confounded claim, re-run on the refitted
    # readouts with the same candidates and the Stage 0 terminal alignment.
    .venv/bin/python -u -m examples.world_model.readout_diagnostic \
        "$output/$name" --data "$data" --device cuda \
        --states 96 --candidates 48 \
        > "$output/${name}_diagnostic.txt" 2>&1
}

pids=()
refit_task buzz_wire outputs/buzz_wire_1196/baselines outputs/buzz_wire_1196/data &
pids+=($!)
refit_task transport outputs/interaction_control_1194/transport \
    outputs/transport_data_1190 &
pids+=($!)

status=0
for pid in "${pids[@]}"; do
    wait "$pid" || status=1
done
[ "$status" -eq 0 ]

printf '\nCompleted Stage 1 refit: %s\n' "$output"
