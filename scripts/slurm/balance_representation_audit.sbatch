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
# RERUN after job 1527. That job completed but produced NO MEASUREMENT: every
# input returned E ~ 1e10 with train matching test, so the overfit guard did not
# fire. The cause was the aggregation, not the fit -- `E_CF` divided by
# ||dY_true|| per anchor and averaged, and 44% of Balance anchors at h=3 carry
# essentially no response, so those rows dominated. `E_CF` is now pooled as a
# ratio of sums; both forms are reported side by side, and the companion job on
# Buzz Wire (`ecf_aggregation_agreement.sbatch`) is the gate on whether the
# pooled column may be read against the published Buzz Wire numbers.
#
# A second correction carried in from 1527: job 1525's 62-74% activity at h=3
# was measured on the 3x3 MIDPOINT GRID. The same cells under the SAMPLED
# REFERENCE design this audit uses are only ~56% active. Activity is a property
# of the intervention, not of the task alone, so the grid figures quoted above
# are not the ones that govern this run.
#
# NOTHING is changed in the model: no LeWM edit, no L_CF, no world token, no
# planning. Existing checkpoints, existing bank, frozen measurement convention.
#
# CAVEAT carried into the write-up: Balance checkpoints are `legacy_compact`,
# while the Buzz Wire numbers came from `lewm_reference`. The latent condition
# encodes a single frame either way, so the profile affects what was trained
# rather than how it is read -- but it is a cross-profile comparison and must be
# reported as one.
# SIZING, measured on job 1527, which ran this exact workload on a FULL A100:
# peak GPU memory 712 MiB of 40960, mean GPU utilisation 35%, peak RSS 2.42 GB
# of the 40 GB requested. The card was idle most of the time because the cost is
# the CPU-side simulator rollouts, not the head fits. Requesting a full A100 was
# over-provisioning that only served to queue the job behind another user; a
# 2g.10gb slice has 14x the GPU memory this needs, and 10 GB of RAM has 4x the
# headroom. Expect a longer wall clock than 2:54 on the smaller slice, which the
# 16 h limit absorbs.
#SBATCH --job-name=balance-repr
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=10G
#SBATCH --gres=gpu:a100_2g.10gb:1
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
