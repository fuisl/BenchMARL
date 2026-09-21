#!/usr/bin/env bash
# T-A1: measure the TRUE interaction Jacobian on the simulator. No learned model.
#
# Task A asks whether a learned latent transition preserves counterfactual
# joint-action effects. Before comparing models we measure the effect, because
# the project's most expensive repeated mistake has been reporting a model
# ordering on a quantity below the resolution of the instrument: Balance's true
# cross-agent response turned out to be 9-70x below probe error (experiment 14),
# and the ordering reported there was noise.
#
# From each held-out anchor we hold every action coordinate at a sampled
# reference joint action and move ONE coordinate -- agent i's axis c -- to each
# end of its range. The difference in the simulator's next physical state is a
# finite-difference column of the interaction Jacobian. Diagonal blocks are
# self-dynamics; off-diagonal blocks are the entire multi-agent content, and an
# independent predictor sets them to exactly zero by construction.
#
# This run decides which cells the T-A2 model comparison is allowed to use.
# It trains nothing and selects nothing, so it cannot leak into a later result.
#
# SIZING, measured. A 24-anchor / 1-reference / 2-block CPU smoke took ~2 min.
# The full grid is 117 test anchors x 4 references x (4 cells x 2 endpoints +
# 1 reference + 1 reflection) = ~4,700 rollouts of 25 primitive steps on one
# MIG slice. An hour is generous.
#SBATCH --job-name=ta1-jacobian
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:a100_3g.20gb:1
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the BenchMARL repository root}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="${SLURM_SUBMIT_DIR}${PYTHONPATH:+:$PYTHONPATH}"

data="${TA1_DATA:-outputs/buzz_wire_1196/data}"
split="${TA1_SPLIT:-test}"
horizon="${TA1_HORIZON:-5}"
references="${TA1_REFERENCES:-4}"
seed="${TA1_SEED:-7301}"
output="${SLURM_SUBMIT_DIR}/outputs/ta1_interaction_jacobian_${SLURM_JOB_ID}"
mkdir -p "$output/source"
exec > >(tee -a "$output/batch.log") 2>&1

printf 'T-A1: job=%s data=%s split=%s horizon=%s references=%s seed=%s\n' \
    "$SLURM_JOB_ID" "$data" "$split" "$horizon" "$references" "$seed"
printf 'CUDA_VISIBLE_DEVICES=%s\n' "${CUDA_VISIBLE_DEVICES:?}"
test -f "$data/manifest.json"

git rev-parse HEAD > "$output/commit.txt"
git status --short > "$output/git_status.txt"
git diff --binary > "$output/working_tree.patch"
cp examples/world_model/{interaction_jacobian,collect,snapshot_restore,physical_response,plan_ranking}.py \
    "$output/source/"
cp scripts/slurm/ta1_interaction_jacobian.sbatch "$output/"

.venv/bin/python -u scripts/slurm/monitor_gpu_memory.py "$output/gpu_memory.csv" &
gpu_monitor_pid=$!
trap 'kill "$gpu_monitor_pid" 2>/dev/null || true' EXIT

# The registered contracts gate the run: a measurement whose episode masking or
# episode-clustered uncertainty has drifted must not produce data.
.venv/bin/python -m pytest -q test/test_world_model_interaction_jacobian.py \
    | tee "$output/preflight_tests.txt"

srun --ntasks=1 .venv/bin/python -u -m examples.world_model.interaction_jacobian \
    --data "$data" --out "$output/result" --split "$split" \
    --horizon "$horizon" --references "$references" --seed "$seed" \
    --device cuda 2>&1 | tee "$output/run.log"

test -s "$output/result/interaction_jacobian.json"
printf '\nCompleted T-A1: %s\n' "$output"
