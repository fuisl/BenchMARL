#!/usr/bin/env bash
# T-A2/T-A2c rerun with the F13 repair: authentic pre-anchor context.
#
# rolled_latent used to hand lewm_reference the episode-start convention --
# current frame repeated, zero past actions -- at anchors that are 73%
# mid-episode, while training always fed real windows. Jobs 1502 and 1505 are
# therefore contaminated; this supersedes both. Nothing is retrained: only the
# evaluation context changes, so any difference is attributable to F13 alone.
#
# The context is rebuilt at the BLOCK stride, because dataset.py builds the
# reference sequence as [observation[0]] + next_observation[block-1::block]. A
# 3-frame context spans 2*block primitive steps, not 2. Alignment against each
# anchor's own observation is asserted on every call, and source_step > 0 with
# no real context now raises instead of silently fabricating.
#
# Preregistered in note 30: delta_10 = E_CF[H1] - E_CF[H0] and delta_21 =
# E_CF[H2] - E_CF[H1], paired by seed. If their signs and seed counts survive,
# K21 and K22 are strengthened. If a sign flips, the malformed context
# interacted with architecture and both are re-opened.
#
# Horizons 1..3 because K24 is more exposed than the one-step result: a bad
# initial context propagates through recursive rollout, so the observed decay
# may mix true compounding with initial-context OOD.
#SBATCH --job-name=ta2f-fixed
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --gres=gpu:a100_3g.20gb:1
#SBATCH --time=08:00:00
#SBATCH --output=slurm-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the BenchMARL repository root}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="${SLURM_SUBMIT_DIR}${PYTHONPATH:+:$PYTHONPATH}"

runs="${TA2F_RUNS:?Set TA2F_RUNS to the stage-1 runs directory}"
jacobian="${TA2F_JACOBIAN:?Set TA2F_JACOBIAN to the T-A1 interaction_jacobian.json}"
data="${TA2F_DATA:-outputs/buzz_wire_1196/data}"
horizons="${TA2F_HORIZONS:-1 2 3}"
seed="${TA2F_SEED:-7301}"
output="${SLURM_SUBMIT_DIR}/outputs/ta2_f13_corrected_${SLURM_JOB_ID}"
mkdir -p "$output/source"
exec > >(tee -a "$output/batch.log") 2>&1

printf 'T-A2 F13-corrected: job=%s runs=%s horizons=[%s]\n' "$SLURM_JOB_ID" "$runs" "$horizons"
printf 'CUDA_VISIBLE_DEVICES=%s\n' "${CUDA_VISIBLE_DEVICES:?}"
test -f "$jacobian"
test -d "$runs"

git rev-parse HEAD > "$output/commit.txt"
git status --short > "$output/git_status.txt"
git diff --binary > "$output/working_tree.patch"
cp examples/world_model/{counterfactual_fidelity,interaction_jacobian,physical_response}.py \
    "$output/source/"
cp scripts/slurm/ta2_f13_corrected.sbatch "$output/"
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

printf '\nCompleted T-A2 F13-corrected: %s\n' "$output"
