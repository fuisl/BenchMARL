#!/usr/bin/env bash
# T-A2c (Gate 1 completion): counterfactual fidelity as a function of horizon.
#
# Planning consumes RECURSIVE predictions, so a planner does not care about
# E_CF at one block -- it cares about E_CF(h) over the horizon it rolls. T-A2
# measured h=1 only. This sweeps the horizons T-A1 could measure.
#
# T-A1 established what is measurable: cross/self rises 0.275 -> 0.357 -> 0.479
# over h=1,2,3, and h=4,5 are NOT measurable because every anchor has terminated
# under the maximal constant-action intervention. So this sweep stops at 3, and
# the live anchor count per horizon is reported rather than assumed.
#
# The interventions are regenerated from T-A1's seed and must reproduce its
# recorded cell means AT THAT HORIZON, so a horizon whose physics drifted from
# T-A1 aborts instead of quietly scoring a different counterfactual.
#
# Runs on the full A100 so it does not contend with the T-A2b MIG slice.
#SBATCH --job-name=ta2c-horizon
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=08:00:00
#SBATCH --output=slurm-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the BenchMARL repository root}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="${SLURM_SUBMIT_DIR}${PYTHONPATH:+:$PYTHONPATH}"

runs="${TA2C_RUNS:?Set TA2C_RUNS to the stage-1 runs directory}"
jacobian="${TA2C_JACOBIAN:?Set TA2C_JACOBIAN to the T-A1 interaction_jacobian.json}"
data="${TA2C_DATA:-outputs/buzz_wire_1196/data}"
horizons="${TA2C_HORIZONS:-1 2 3}"
seed="${TA2C_SEED:-7301}"
output="${SLURM_SUBMIT_DIR}/outputs/ta2c_horizon_${SLURM_JOB_ID}"
mkdir -p "$output/source"
exec > >(tee -a "$output/batch.log") 2>&1

printf 'T-A2c: job=%s runs=%s horizons=[%s]\n' "$SLURM_JOB_ID" "$runs" "$horizons"
printf 'CUDA_VISIBLE_DEVICES=%s\n' "${CUDA_VISIBLE_DEVICES:?}"
test -f "$jacobian"
test -d "$runs"

git rev-parse HEAD > "$output/commit.txt"
git status --short > "$output/git_status.txt"
git diff --binary > "$output/working_tree.patch"
cp examples/world_model/{counterfactual_fidelity,interaction_jacobian,physical_response}.py \
    "$output/source/"
cp scripts/slurm/ta2c_horizon_fidelity.sbatch "$output/"
cp "$jacobian" "$output/interaction_jacobian_input.json"

.venv/bin/python -u scripts/slurm/monitor_gpu_memory.py "$output/gpu_memory.csv" &
gpu_monitor_pid=$!
trap 'kill "$gpu_monitor_pid" 2>/dev/null || true' EXIT

for horizon in $horizons; do
    for probe in linear mlp; do
        printf '\n=== horizon %s blocks, probe %s ===\n' "$horizon" "$probe"
        .venv/bin/python -u -m examples.world_model.counterfactual_fidelity \
            "$runs" --data "$data" --jacobian "$jacobian" \
            --out "$output/result_h${horizon}_${probe}" \
            --horizon "$horizon" --seed "$seed" --probe "$probe" \
            --device cuda 2>&1 | tee "$output/run_h${horizon}_${probe}.log"
        test -s "$output/result_h${horizon}_${probe}/counterfactual_fidelity.json"
    done
done

printf '\nCompleted T-A2c: %s\n' "$output"
