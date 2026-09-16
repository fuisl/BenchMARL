#!/usr/bin/env bash
# Gate 4 on Buzz Wire: does the model that predicts the intervention best also
# plan best, on the SAME task?
#
# This is the paper's central experiment and the only remaining one that must
# succeed. Everything else is measured.
#
# Buzz Wire is the one task that can answer both halves of the question. Its
# cross-agent response is measurable in common physical coordinates -- job 1239
# resolves it at 0.79x the probe's own error, where Balance sits 11-70x below
# the floor -- and relational wins that comparison 8/8 and 7/8 paired on
# magnitude and direction. Job 1259 then showed it is controllable: the true
# simulator solves 25/32 (78%) with ZERO failures at H=5 executing one block,
# against 0/32 for random and for doing nothing.
#
# So the reference to beat is 78% success and task distance 0.1259, measured on
# these exact 32 roots with this exact budget. A learned model either approaches
# it or it does not, and the gap is the model's -- which has never been true on
# any task in this project before today.
#
# SHARDED BY SEED, one seed per job, six checkpoints each (3 predictors x 2
# regimes). That is deliberate: if only one shard finishes, the result is a
# COMPLETE six-cell comparison at one seed rather than two thirds of a grid with
# whole architectures missing. Submit one job per seed; two run concurrently
# under the local QoS.
#
# Each shard recomputes its own references (~55 min: one planned rollout to
# generate goals, one for the oracle). That is 3x redundant across shards and is
# chosen over a shared --reference-cache because concurrent shards would race on
# writing it, and an hour of oracle is cheaper than a corrupted reference.
#
# SIZING, measured rather than estimated. Job 1259's Buzz Wire oracle took 1648s
# for 32 roots at execute-1. On Balance, learned checkpoints ran 3.4-4.3x the
# oracle's cost, so expect roughly 1.5-2h per checkpoint here and 10-13h per
# shard. The 8-seed grid this pilot would expand to is 75-93h serial, which
# coding rule 13 puts on the H100 cluster -- `packed_mig` targets a partition
# that does not exist on this machine, so that expansion needs a hand-off rather
# than a bigger local timeout.
#SBATCH --job-name=bw-gate4
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:a100_3g.20gb:1
#SBATCH --time=23:00:00
#SBATCH --output=slurm-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the BenchMARL repository root}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1
seed="${GATE4_SEED:?Set GATE4_SEED to one training seed, e.g. 4100}"
runs="${GATE4_RUNS:-outputs/buzz_wire_1196/baselines}"
data="${GATE4_DATA:-outputs/buzz_wire_1196/data}"
roots="${GATE4_ROOTS:-32}"
output="${SLURM_SUBMIT_DIR}/outputs/bw_gate4_${seed}_${SLURM_JOB_ID}"
mkdir -p "$output"
exec > >(tee -a "$output/batch.log") 2>&1
printf 'Buzz Wire Gate 4: job=%s seed=%s roots=%s\n' "$SLURM_JOB_ID" "$seed" "$roots"
printf 'CUDA_VISIBLE_DEVICES=%s\n' "${CUDA_VISIBLE_DEVICES:?}"
git rev-parse HEAD > "$output/commit.txt"
git status --short > "$output/git_status.txt"
git diff --binary > "$output/working_tree.patch"
mkdir "$output/source"
cp examples/world_model/*.py "$output/source/"
cp scripts/slurm/buzz_wire_gate4.sbatch "$output/"

.venv/bin/python -u scripts/slurm/monitor_gpu_memory.py "$output/gpu_memory.csv" &
gpu_monitor_pid=$!
trap 'kill "$gpu_monitor_pid" 2>/dev/null || true' EXIT

# Identical to job 1259 in roots, split, horizon, cadence and search budget, so
# that job's oracle row is the same experiment rather than a comparable one.
# closed_loop persists after every policy, so a wall-clock kill loses at most
# the checkpoint in flight.
.venv/bin/python -u -m examples.world_model.closed_loop \
    "$runs" --data "$data" --device cuda \
    --states "$roots" --roots-split train \
    --horizon 5 --execute-blocks 1 \
    --num-samples 300 --num-iters 30 \
    --objectives reward --seeds "$seed" \
    --output "$output/gate4_seed${seed}.json" \
    2>&1 | tee "$output/gate4_seed${seed}.txt"

printf '\nCompleted Buzz Wire Gate 4 seed %s: %s\n' "$seed" "$output"
