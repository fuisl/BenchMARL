#!/usr/bin/env bash
# Stage 0/1: cross-scenario admission benchmark. NO world model is trained.
#
# Buzz Wire's cross-agent effect is large, real, active on 100% of anchors, and
# unlearnable because the observation omits the mediating body. Establishing
# that cost 48 checkpoints, a horizon sweep and a localization study. This
# benchmark exists so it is never spent again without checking first.
#
# One protocol on every scenario, producing one comparable ADMISSION REPORT:
# a 3x3 joint-action grid giving J_own, J_cross and the MIXED second difference
# C, then the A -> O -> H_dense -> H_model -> S information ladder. Only ADMIT
# scenarios proceed to H0/H1/H2.
#
# Banks are reused, not recollected: all five share action_block=5,
# sequence_steps=25 and the seed family 3100-3103, and the audit regenerates its
# own interventions from restored snapshots. balance_repair_1236 is used, never
# balance_1233, whose heuristic branch was Transport's policy.
#
# Buzz Wire is carried as the KNOWN partial-observability reference. If the
# protocol does not classify it as partially_observable, the protocol is wrong,
# not Buzz Wire -- that is the built-in positive control.
# Dropout is carried as the known weak-interaction control.
#
# SIZING. Per scenario per axis: (train + test anchors) x references x 9 grid
# cells of 5 primitive steps, then five head fits with decay selection. Measured
# on Buzz Wire at full anchor count, one scenario is well under an hour; five
# scenarios x up to 2 axes fits an overnight window with wide margin.
#SBATCH --job-name=stage0-admission
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:a100_3g.20gb:1
#SBATCH --time=14:00:00
#SBATCH --output=slurm-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the BenchMARL repository root}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="${SLURM_SUBMIT_DIR}${PYTHONPATH:+:$PYTHONPATH}"

references="${ADM_REFERENCES:-2}"
horizon="${ADM_HORIZON:-1}"
seed="${ADM_SEED:-7301}"
output="${SLURM_SUBMIT_DIR}/outputs/stage0_admission_${SLURM_JOB_ID}"
mkdir -p "$output/source"
exec > >(tee -a "$output/batch.log") 2>&1

printf 'Stage 0/1 admission: job=%s references=%s horizon=%s\n' \
    "$SLURM_JOB_ID" "$references" "$horizon"
printf 'CUDA_VISIBLE_DEVICES=%s\n' "${CUDA_VISIBLE_DEVICES:?}"

git rev-parse HEAD > "$output/commit.txt"
git status --short > "$output/git_status.txt"
git diff --binary > "$output/working_tree.patch"
cp examples/world_model/{admission_benchmark,counterfactual_localization,counterfactual_fidelity,interaction_jacobian,collect}.py \
    "$output/source/"
cp scripts/slurm/stage0_admission.sbatch "$output/"

.venv/bin/python -u scripts/slurm/monitor_gpu_memory.py "$output/gpu_memory.csv" &
gpu_monitor_pid=$!
trap 'kill "$gpu_monitor_pid" 2>/dev/null || true' EXIT

.venv/bin/python -m pytest -q test/test_world_model_interaction_jacobian.py \
    | tee "$output/preflight_tests.txt"

# Each scenario is written to the report as it completes, so a late failure
# cannot lose earlier work.
srun --ntasks=1 .venv/bin/python -u -m examples.world_model.admission_benchmark \
    --scenario transport=outputs/transport_data_1190 \
    --scenario wheel=outputs/wheel_1205/data \
    --scenario balance=outputs/balance_repair_1236/data \
    --scenario buzz_wire=outputs/buzz_wire_1196/data \
    --scenario dropout=outputs/dropout_control_1193/data \
    --out "$output/result" --references "$references" --horizon "$horizon" \
    --seed "$seed" --device cuda 2>&1 | tee "$output/run.log"

test -s "$output/result/admission_report.json"
printf '\nCompleted Stage 0/1 admission: %s\n' "$output"
