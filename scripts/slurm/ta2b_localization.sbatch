#!/usr/bin/env bash
# T-A2b: localize where the counterfactual effect is lost, E vs P vs interface.
#
# T-A2 scored the composite E -> P -> probe and found it fails on the cross
# block while carrying a clearly positive cosine. That number cannot say which
# stage lost the effect.
#
# Test B is already answered: T-A2's probe_floor reads the counterfactual
# difference off TRUE future encodings and sits at 0.173-0.234, so the encoder
# does represent the realized effect after it happens. Test C is T-A2 itself,
# 1.05-1.19. With B good and C bad, only two diagnostic rows survive and Test A
# -- can a head predict dY from the CURRENT latent -- decides between them.
#
# Test A is nothing without its controls, which run here as first-class arms:
#   actions_only  the state-blind floor (average response, no state)
#   physical      the information ceiling on the true simulator state
# A latent that cannot beat the blind floor is not counterfactually sufficient;
# one that reaches the ceiling exonerates the encoder and implicates P.
#
# HEAD ADEQUACY IS A PRECONDITION, NOT A DETAIL. A first pipeline smoke put the
# ceiling at 0.864 against a 0.906 floor -- a gap of 0.04 that would have fired
# the "uninformative" branch -- because the head was one layer, full-batch, 300
# steps, on unnormalized VMAS coordinates. The head is now two layers,
# minibatched, 400 epochs, z-scored on train statistics. The run reports the
# self block (which T-A2 shows is well captured, so a head failing it is
# underfit) and train error beside test. If either precondition fails the
# correct report is "the diagnostic did not work", not a claim about z_t.
#
# SIZING. Branch regeneration dominates the front: 683 train + 117 test anchors
# x 4 references x 9 rollouts. Then one head per condition with decay selection.
# The 4-reference count is a contract, not a knob: the test interventions must
# reproduce T-A1's recorded cell means or the run aborts.
#SBATCH --job-name=ta2b-localize
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:a100_3g.20gb:1
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the BenchMARL repository root}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="${SLURM_SUBMIT_DIR}${PYTHONPATH:+:$PYTHONPATH}"

runs="${TA2B_RUNS:?Set TA2B_RUNS to the stage-1 runs directory}"
jacobian="${TA2B_JACOBIAN:?Set TA2B_JACOBIAN to the T-A1 interaction_jacobian.json}"
data="${TA2B_DATA:-outputs/buzz_wire_1196/data}"
horizon="${TA2B_HORIZON:-1}"
seed="${TA2B_SEED:-7301}"
output="${SLURM_SUBMIT_DIR}/outputs/ta2b_localization_${SLURM_JOB_ID}"
mkdir -p "$output/source"
exec > >(tee -a "$output/batch.log") 2>&1

printf 'T-A2b: job=%s runs=%s horizon=%s seed=%s\n' \
    "$SLURM_JOB_ID" "$runs" "$horizon" "$seed"
printf 'CUDA_VISIBLE_DEVICES=%s\n' "${CUDA_VISIBLE_DEVICES:?}"
test -f "$jacobian"
test -d "$runs"

git rev-parse HEAD > "$output/commit.txt"
git status --short > "$output/git_status.txt"
git diff --binary > "$output/working_tree.patch"
cp examples/world_model/{counterfactual_localization,counterfactual_fidelity,interaction_jacobian,physical_response}.py \
    "$output/source/"
cp scripts/slurm/ta2b_localization.sbatch "$output/"
cp "$jacobian" "$output/interaction_jacobian_input.json"

.venv/bin/python -u scripts/slurm/monitor_gpu_memory.py "$output/gpu_memory.csv" &
gpu_monitor_pid=$!
trap 'kill "$gpu_monitor_pid" 2>/dev/null || true' EXIT

.venv/bin/python -m pytest -q test/test_world_model_interaction_jacobian.py \
    | tee "$output/preflight_tests.txt"

srun --ntasks=1 .venv/bin/python -u -m examples.world_model.counterfactual_localization \
    "$runs" --data "$data" --jacobian "$jacobian" \
    --out "$output/result" --horizon "$horizon" --seed "$seed" \
    --device cuda 2>&1 | tee "$output/run.log"

test -s "$output/result/counterfactual_localization.json"
printf '\nCompleted T-A2b: %s\n' "$output"
