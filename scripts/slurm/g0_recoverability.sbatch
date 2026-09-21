#!/usr/bin/env bash
# G0: is the mediating state recoverable from LEGITIMATE observations at all?
#
# T-A2b-2 established that the counterfactual deficit is the omitted ball and
# linkage state: adding it to the same latent through the same head recovers
# 86-88% of the blind-to-ceiling gap. That makes an agent-neutral world token
# the indicated repair -- but a world token must be LEARNED from what the agents
# can see, not handed the simulator state. Privileged state is what the
# diagnostic uses to set a ceiling; putting it in the model answers an easier
# question.
#
# So before building any architecture, two cheap discriminators run beside the
# existing conditions:
#
#   observation_raw        the raw unencoded 6-D observation. If it BEATS the
#                          latent, the encoder discarded cross-agent information
#                          it was given, which is a JEPA-abstraction result. If
#                          they match, the encoder preserved what was there and
#                          the deficit is the observation itself.
#   latent_plus_agentphys  latent plus agent rotation and angular velocity, which
#                          the observation omits but which are NOT the mediating
#                          body. Separates "the ball is required" from "any
#                          omitted physical state would do".
#
# Registered rule and the task-design branch are in
# docs/paper/experiments/32_joint_representation.md.
#
# Runs on the full A100 so it does not contend with the MIG slice.
#SBATCH --job-name=g0-recover
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the BenchMARL repository root}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="${SLURM_SUBMIT_DIR}${PYTHONPATH:+:$PYTHONPATH}"

runs="${G0_RUNS:?Set TA2B_RUNS to the stage-1 runs directory}"
jacobian="${G0_JACOBIAN:?Set TA2B_JACOBIAN to the T-A1 interaction_jacobian.json}"
data="${G0_DATA:-outputs/buzz_wire_1196/data}"
horizon="${G0_HORIZON:-1}"
seed="${G0_SEED:-7301}"
# 3 matches the reference profile history_size; 1 disables the condition.
history_frames="${G0_HISTORY_FRAMES:-3}"
output="${SLURM_SUBMIT_DIR}/outputs/g0_recoverability_${SLURM_JOB_ID}"
mkdir -p "$output/source"
exec > >(tee -a "$output/batch.log") 2>&1

printf 'G0: job=%s runs=%s horizon=%s seed=%s\n' \
    "$SLURM_JOB_ID" "$runs" "$horizon" "$seed"
printf 'CUDA_VISIBLE_DEVICES=%s\n' "${CUDA_VISIBLE_DEVICES:?}"
test -f "$jacobian"
test -d "$runs"

git rev-parse HEAD > "$output/commit.txt"
git status --short > "$output/git_status.txt"
git diff --binary > "$output/working_tree.patch"
cp examples/world_model/{counterfactual_localization,counterfactual_fidelity,interaction_jacobian,physical_response}.py \
    "$output/source/"
cp scripts/slurm/g0_recoverability.sbatch "$output/"
cp "$jacobian" "$output/interaction_jacobian_input.json"

.venv/bin/python -u scripts/slurm/monitor_gpu_memory.py "$output/gpu_memory.csv" &
gpu_monitor_pid=$!
trap 'kill "$gpu_monitor_pid" 2>/dev/null || true' EXIT

.venv/bin/python -m pytest -q test/test_world_model_interaction_jacobian.py \
    | tee "$output/preflight_tests.txt"

srun --ntasks=1 .venv/bin/python -u -m examples.world_model.counterfactual_localization \
    "$runs" --data "$data" --jacobian "$jacobian" \
    --out "$output/result" --horizon "$horizon" --seed "$seed" \
    --history-frames "$history_frames" \
    --device cuda 2>&1 | tee "$output/run.log"

test -s "$output/result/counterfactual_localization.json"
printf '\nCompleted T-A2b: %s\n' "$output"
