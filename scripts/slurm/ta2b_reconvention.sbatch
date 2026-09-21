#!/usr/bin/env bash
# T-A2b re-measured under the per-cell, cross-fitted convention.
#
# Job 1515 showed the measurement is strongly convention-dependent and that
# neither convention was ever chosen. G0/G0b, and T-A2b before them, fitted ONE
# head to the whole next-state response while serving four heterogeneous
# (agent, axis) intervention cells. On the observation arm that convention reads
# 0.016 where a per-cell cross-fitted head reads 0.605-0.638.
#
# T-A2b's conclusions -- latent R ~ 0.02, latent+ball R ~ 0.86, "the predictor is
# exonerated and the observation is the problem" -- were all measured under the
# depressive convention. That conclusion drove the entire pivot to the admission
# benchmark, so it has to be re-measured before anything is built on it.
#
# Nothing is retrained. Same 48 checkpoints, same bank, same anchors, seeds,
# scale, interventions and head family. Only --fit-target and --cells change.
#
# Expected: the ORDERING (latent << latent+ball) should survive, since it was
# measured under one consistent instrument. Whether the latent clears any gate
# is the open question.
#SBATCH --job-name=ta2b-reconv
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:a100_3g.20gb:1
#SBATCH --time=08:00:00
#SBATCH --output=slurm-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the BenchMARL repository root}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="${SLURM_SUBMIT_DIR}${PYTHONPATH:+:$PYTHONPATH}"

runs="${RC_RUNS:?Set RC_RUNS}"
jacobian="${RC_JACOBIAN:?Set RC_JACOBIAN}"
data="${RC_DATA:-outputs/buzz_wire_1196/data}"
cells="${RC_CELLS:-1:0}"
output="${SLURM_SUBMIT_DIR}/outputs/ta2b_reconvention_${SLURM_JOB_ID}"
mkdir -p "$output/source"
exec > >(tee -a "$output/batch.log") 2>&1

printf 'T-A2b re-convention: job=%s cells=%s fit-target=cross\n' "$SLURM_JOB_ID" "$cells"
printf 'CUDA_VISIBLE_DEVICES=%s\n' "${CUDA_VISIBLE_DEVICES:?}"
git rev-parse HEAD > "$output/commit.txt"
git status --short > "$output/git_status.txt"
git diff --binary > "$output/working_tree.patch"
cp examples/world_model/counterfactual_localization.py "$output/source/"
cp scripts/slurm/ta2b_reconvention.sbatch "$output/"

.venv/bin/python -u scripts/slurm/monitor_gpu_memory.py "$output/gpu_memory.csv" &
gpu_monitor_pid=$!
trap 'kill "$gpu_monitor_pid" 2>/dev/null || true' EXIT

.venv/bin/python -m pytest -q test/test_world_model_interaction_jacobian.py \
    | tee "$output/preflight_tests.txt"

srun --ntasks=1 .venv/bin/python -u -m examples.world_model.counterfactual_localization \
    "$runs" --data "$data" --jacobian "$jacobian" \
    --out "$output/result" --fit-target cross --cells $cells \
    --device cuda 2>&1 | tee "$output/run.log"

test -s "$output/result/counterfactual_localization.json"
printf '\nCompleted T-A2b re-convention: %s\n' "$output"
