#!/usr/bin/env bash
# Does O >> Z reproduce on Balance at h=3?
#
# On Buzz Wire, under the frozen convention (cross-fitted head, one intervention
# cell, sampled reference), the raw observation recovers 0.605 of the
# blind-to-reference gap while the learned latent recovers 0.212 -- the encoder
# keeps roughly a third of what its own input carries. That was a single task,
# and Buzz Wire is also the task whose observation omits the mediating body.
#
# Balance is the right second case. Job 1525 showed its cross-agent effect is
# TEMPORALLY MEDIATED: cross/self rises 0.035 -> 0.125 -> 0.164 over h = 1,2,3,
# and the strongest cell goes from 0.374 at 48% active to 0.807 at 74%. So the
# effect exists and is measurable at h=3, which is where this audit runs. Its
# mechanism is force superposition on a shared object, not Buzz Wire's rigid
# joint, and it has four agents rather than two.
#
# If O >> Z reproduces here, the result stops being a peculiarity of one task
# and starts looking like a limitation of predictive latent objectives for
# multi-agent counterfactual reasoning.
#
# NOTHING is changed in the model: no LeWM edit, no L_CF, no world token, no
# planning. Existing checkpoints, existing bank, frozen measurement convention.
#
# CAVEAT carried into the write-up: Balance checkpoints are `legacy_compact`,
# while the Buzz Wire numbers came from `lewm_reference`. The latent condition
# encodes a single frame either way, so the profile affects what was trained
# rather than how it is read -- but it is a cross-profile comparison and must be
# reported as one.
#SBATCH --job-name=balance-repr
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=40G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=16:00:00
#SBATCH --output=slurm-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the BenchMARL repository root}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="${SLURM_SUBMIT_DIR}${PYTHONPATH:+:$PYTHONPATH}"

data="${BAL_DATA:-outputs/balance_repair_1236/data}"
runs="${BAL_RUNS:-outputs/balance_repair_1236/baselines}"
horizon="${BAL_HORIZON:-3}"
references="${BAL_REFERENCES:-3}"
cells="${BAL_CELLS:-1:1 0:1}"
seed="${BAL_SEED:-7301}"
output="${SLURM_SUBMIT_DIR}/outputs/balance_representation_${SLURM_JOB_ID}"
mkdir -p "$output/source"
exec > >(tee -a "$output/batch.log") 2>&1

printf 'Balance representation audit: job=%s h=%s refs=%s cells=[%s]\n' \
    "$SLURM_JOB_ID" "$horizon" "$references" "$cells"
printf 'CUDA_VISIBLE_DEVICES=%s\n' "${CUDA_VISIBLE_DEVICES:?}"
git rev-parse HEAD > "$output/commit.txt"
git status --short > "$output/git_status.txt"
cp examples/world_model/{interaction_jacobian,counterfactual_localization,counterfactual_fidelity}.py \
    "$output/source/"
cp scripts/slurm/balance_representation_audit.sbatch "$output/"

.venv/bin/python -u scripts/slurm/monitor_gpu_memory.py "$output/gpu_memory.csv" &
gpu_monitor_pid=$!
trap 'kill "$gpu_monitor_pid" 2>/dev/null || true' EXIT

.venv/bin/python -m pytest -q test/test_world_model_interaction_jacobian.py \
    | tee "$output/preflight_tests.txt"

# Step 1: the true interaction Jacobian, which sets the measurement floor and
# is what the audit's regenerated interventions must reproduce.
srun --ntasks=1 .venv/bin/python -u -m examples.world_model.interaction_jacobian \
    --data "$data" --out "$output/ta1" --split test \
    --horizon "$horizon" --references "$references" --seed "$seed" \
    --device cuda 2>&1 | tee "$output/run_ta1.log"
test -s "$output/ta1/interaction_jacobian.json"

# Step 2: the representation audit, frozen convention, one cell per fit.
for cell in $cells; do
    tag="${cell/:/x}"
    printf '\n=== cell %s ===\n' "$cell"
    srun --ntasks=1 .venv/bin/python -u -m examples.world_model.counterfactual_localization \
        "$runs" --data "$data" --jacobian "$output/ta1/interaction_jacobian.json" \
        --out "$output/result_$tag" --fit-target cross --cells "$cell" \
        --horizon "$horizon" --references "$references" --seed "$seed" \
        --device cuda 2>&1 | tee "$output/run_$tag.log"
    test -s "$output/result_$tag/counterfactual_localization.json"
done

printf '\nCompleted Balance representation audit: %s\n' "$output"
