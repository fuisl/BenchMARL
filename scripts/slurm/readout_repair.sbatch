#!/usr/bin/env bash
# Can the model imagine the collision, once it can see the thing that collides?
#
# This is the cheap half of the Gate 4 repair, and it decides whether the
# expensive half is worth running.
#
# Jobs 1261-1263 planned with models whose encoder never sees the ball. Buzz
# Wire's agents observe [pos, vel, pos - goal]; the ball jointed between them --
# whose contact with the wire IS the failure the reward punishes -- appears in no
# agent's observation. Job 1203 measured the consequence directly: the reward
# readout ranks plans at Spearman ~ -0.25 even when handed the simulator's OWN
# latents, so the ordering is wrong before any rollout error enters. Gate 4 paid
# it in full -- every learned policy collided more often than random actions
# (0.77-0.94 against 0.62) and died by step 38 of 100, because nothing in the
# predicted return falls when the plan drives into the wire.
#
# That is an information defect, not a dynamics defect, and no number of seeds
# reaches it. A pixel-based world model never meets it: the obstacle is in the
# frame, so predicting the frame means predicting the obstacle. Hand-picking a
# vector observation picked the constraint out, and the self-supervised
# objective then had no way to put it back.
#
# Job 1223 already trained the three input conditions on this bank, so the test
# needs no training. `readout_diagnostic` separates the two learned stages:
#
#   J_model     predicted latents + readout   (dynamics error AND readout error)
#   J_readout   the simulator's own latents + readout   (readout error alone)
#
# The prediction: `observation` reproduces job 1203's anti-correlated J_readout,
# and `physical` -- the same architectures, capacity, seeds and bank, with the
# ball appended -- turns it positive. If it does not, the readout is not the
# blocker and Gate 4 should not be re-run; if it does, the control experiment is
# justified and its expected size is known.
#
# `history` is included because it is the honest middle case: the hidden state is
# inferable from motion rather than supplied. Its physical-response score was
# unmeasurable (probe floor 1.37-1.70, job 1264), so a null here is expected and
# is not evidence that history cannot work.
#SBATCH --job-name=readout-repair
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:a100_3g.20gb:1
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the BenchMARL repository root}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1
output="${SLURM_SUBMIT_DIR}/outputs/readout_repair_${SLURM_JOB_ID}"
mkdir -p "$output"
exec > >(tee -a "$output/batch.log") 2>&1
printf 'Readout repair: job=%s\n' "$SLURM_JOB_ID"
printf 'CUDA_VISIBLE_DEVICES=%s\n' "${CUDA_VISIBLE_DEVICES:?}"
git rev-parse HEAD > "$output/commit.txt"
git status --short > "$output/git_status.txt"
git diff --binary > "$output/working_tree.patch"
mkdir "$output/source"
cp examples/world_model/*.py "$output/source/"
cp scripts/slurm/readout_repair.sbatch "$output/"

.venv/bin/python -u scripts/slurm/monitor_gpu_memory.py "$output/gpu_memory.csv" &
gpu_monitor_pid=$!
trap 'kill "$gpu_monitor_pid" 2>/dev/null || true' EXIT

# All 144 Stage 2 checkpoints in one pass: the three input conditions share the
# anchors, the candidate plans and the simulator's own costs, so they are
# compared against one ground truth rather than three.
.venv/bin/python -u -m examples.world_model.readout_diagnostic \
    outputs/state_input_1223/runs --data outputs/buzz_wire_1196/data \
    --states 96 --candidates 48 --device cuda \
    2>&1 | tee "$output/stage2_readout.txt"

# The 48-model grid Gate 4 actually planned with, as the reference point. Its
# `observation` numbers are what job 1203 reported and what must be reproduced
# before any improvement above them means anything.
.venv/bin/python -u -m examples.world_model.readout_diagnostic \
    outputs/buzz_wire_1196/baselines --data outputs/buzz_wire_1196/data \
    --states 96 --candidates 48 --device cuda \
    2>&1 | tee "$output/baselines_readout.txt"

printf '\nCompleted readout repair: %s\n' "$output"
