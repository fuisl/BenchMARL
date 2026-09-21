#!/usr/bin/env bash
# G0b: does MORE history recover the mediating state?
#
# G0 put the ladder at: single frame R=0.016, 3-frame history R=0.306, ball
# state handed over directly R=0.871. So motion history does carry real
# information about the ball that no single frame does -- which is what the
# rigid-joint geometry predicts, since two joints constrain the ball to at most
# two solutions given agent positions and history disambiguates them -- but
# three frames recovers under a third of the gap.
#
# The registered G0 rule puts R=0.306 in the PARTIAL band, which makes H3
# exploratory rather than confirmatory. This sweep asks whether that is a
# property of the task or of the window length, and it is the last cheap
# question before any architecture is built.
#
# Only the shared conditions are needed: the floor, the raw observation, the
# history window and the physical ceiling are all model-independent. --shared-only
# skips the 48-checkpoint loop that dominates runtime, so this is ~25 min rather
# than 2.5 hours.
#
# Reading it: R rising with k and crossing 0.5 means a learned z_G over a longer
# window is viable. R flattening well below 0.5 means the information is not in
# the observation stream at any window length, and the honest conclusion is that
# Buzz Wire's observation is inadequate for the counterfactual question -- a
# task-design result, which is registered in note 32 as an acceptable outcome.
#SBATCH --job-name=g0b-history
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --gres=gpu:a100_3g.20gb:1
#SBATCH --time=06:00:00
#SBATCH --output=slurm-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the BenchMARL repository root}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="${SLURM_SUBMIT_DIR}${PYTHONPATH:+:$PYTHONPATH}"

runs="${G0B_RUNS:?Set G0B_RUNS to the stage-1 runs directory}"
jacobian="${G0B_JACOBIAN:?Set G0B_JACOBIAN to the T-A1 interaction_jacobian.json}"
data="${G0B_DATA:-outputs/buzz_wire_1196/data}"
frames="${G0B_FRAMES:-1 2 3 5 8 12}"
seed="${G0B_SEED:-7301}"
output="${SLURM_SUBMIT_DIR}/outputs/g0b_history_sweep_${SLURM_JOB_ID}"
mkdir -p "$output/source"
exec > >(tee -a "$output/batch.log") 2>&1

printf 'G0b: job=%s frames=[%s]\n' "$SLURM_JOB_ID" "$frames"
printf 'CUDA_VISIBLE_DEVICES=%s\n' "${CUDA_VISIBLE_DEVICES:?}"
test -f "$jacobian"

git rev-parse HEAD > "$output/commit.txt"
git status --short > "$output/git_status.txt"
git diff --binary > "$output/working_tree.patch"
cp examples/world_model/{counterfactual_localization,counterfactual_fidelity,interaction_jacobian}.py \
    "$output/source/"
cp scripts/slurm/g0_history_sweep.sbatch "$output/"

.venv/bin/python -u scripts/slurm/monitor_gpu_memory.py "$output/gpu_memory.csv" &
gpu_monitor_pid=$!
trap 'kill "$gpu_monitor_pid" 2>/dev/null || true' EXIT

.venv/bin/python -m pytest -q test/test_world_model_interaction_jacobian.py \
    | tee "$output/preflight_tests.txt"

for k in $frames; do
    printf '\n=== history frames: %s ===\n' "$k"
    srun --ntasks=1 .venv/bin/python -u -m examples.world_model.counterfactual_localization \
        "$runs" --data "$data" --jacobian "$jacobian" \
        --out "$output/result_k${k}" --seed "$seed" \
        --history-frames "$k" --shared-only \
        --device cuda 2>&1 | tee "$output/run_k${k}.log"
    test -s "$output/result_k${k}/counterfactual_localization.json"
done

printf '\nCompleted G0b: %s\n' "$output"
