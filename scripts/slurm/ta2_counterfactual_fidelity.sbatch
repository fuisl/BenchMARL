#!/usr/bin/env bash
# T-A2 stage 2: effect-normalized counterfactual fidelity across H0/H1/H2.
#
# Scores every stage-1 checkpoint against the interventions T-A1 measured, and
# reports E_CF separately for the self, cross and shared blocks of the Jacobian.
# A model predicting no response scores exactly 1, so `independent` is the
# structural floor on cross rather than a competitor -- a smoke run confirmed it
# scores 1.0000 with cosine 0.000 there, which is the design working.
#
# Requires T-A1's JSON. The interventions are regenerated from T-A1's seed and
# must reproduce its recorded per-cell means, so this cannot silently score the
# models against a different counterfactual than the one whose floor was
# registered.
#
# Both probe families run, because experiment 14 found the family can flip the
# winner. A result surviving only one family is reported as probe-dependent.
#
# The registered usability rule is applied to the reported relative probe floor:
# a cell is usable only at floor <= 1/3. The smoke run put the cross floor near
# 0.19 (ratio ~5x, usable) and the shared-body floor near 0.56 (ratio ~1.8x,
# NOT usable), so the rule is doing real work and is not a formality.
#SBATCH --job-name=ta2-fidelity
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:a100_3g.20gb:1
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the BenchMARL repository root}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="${SLURM_SUBMIT_DIR}${PYTHONPATH:+:$PYTHONPATH}"

runs="${TA2_RUNS:?Set TA2_RUNS to the stage-1 runs directory}"
jacobian="${TA2_JACOBIAN:?Set TA2_JACOBIAN to T-A1's interaction_jacobian.json}"
data="${TA2_DATA:-outputs/buzz_wire_1196/data}"
horizon="${TA2_HORIZON:-1}"
references="${TA2_REFERENCES:-4}"
seed="${TA2_SEED:-7301}"
output="${SLURM_SUBMIT_DIR}/outputs/ta2_fidelity_${SLURM_JOB_ID}"
mkdir -p "$output/source"
exec > >(tee -a "$output/batch.log") 2>&1

printf 'T-A2 stage 2: job=%s runs=%s jacobian=%s horizon=%s\n' \
    "$SLURM_JOB_ID" "$runs" "$jacobian" "$horizon"
printf 'CUDA_VISIBLE_DEVICES=%s\n' "${CUDA_VISIBLE_DEVICES:?}"
test -f "$jacobian"
test -d "$runs"

git rev-parse HEAD > "$output/commit.txt"
git status --short > "$output/git_status.txt"
git diff --binary > "$output/working_tree.patch"
cp examples/world_model/{counterfactual_fidelity,interaction_jacobian,physical_response,models,train}.py \
    "$output/source/"
cp scripts/slurm/ta2_counterfactual_fidelity.sbatch "$output/"
cp "$jacobian" "$output/interaction_jacobian_input.json"

.venv/bin/python -u scripts/slurm/monitor_gpu_memory.py "$output/gpu_memory.csv" &
gpu_monitor_pid=$!
trap 'kill "$gpu_monitor_pid" 2>/dev/null || true' EXIT

.venv/bin/python -m pytest -q test/test_world_model_interaction_jacobian.py \
    | tee "$output/preflight_tests.txt"

for probe in linear mlp; do
    printf '\n=== probe family: %s ===\n' "$probe"
    srun --ntasks=1 .venv/bin/python -u -m examples.world_model.counterfactual_fidelity \
        "$runs" --data "$data" --jacobian "$jacobian" \
        --out "$output/result_$probe" --horizon "$horizon" \
        --references "$references" --seed "$seed" --probe "$probe" \
        --device cuda 2>&1 | tee "$output/run_$probe.log"
    test -s "$output/result_$probe/counterfactual_fidelity.json"
done

printf '\nCompleted T-A2 stage 2: %s\n' "$output"
