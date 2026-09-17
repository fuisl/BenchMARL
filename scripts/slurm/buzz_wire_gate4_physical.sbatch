#!/usr/bin/env bash
# Gate 4, re-run with the constraint inside the model.
#
# Jobs 1261-1263 falsified the prediction-to-control chain on Buzz Wire: all 18
# learned cells scored 0/32, seventeen returned worse than random, and every
# learned policy collided MORE often than random actions (0.77-0.94 against
# 0.62). The diagnosis is not variance and not the dynamics. The agents observe
# [pos, vel, pos - goal]; the ball jointed between them, whose contact with the
# wire is the failure the reward punishes, is in no agent's observation. Job 1203
# had already measured the consequence -- the reward readout ranks plans at
# Spearman ~ -0.25 given the simulator's OWN latents -- and CEM then maximises a
# predicted return that never falls when the plan drives into the wire.
#
# A world model cannot avoid a constraint it cannot represent. Pixel-based world
# models never meet this because the obstacle is in the frame; hand-picking a
# vector observation picked the constraint out, and no amount of self-supervision
# puts back information the input does not carry.
#
# THE EXPERIMENT. Job 1223 trained all three input conditions on this bank, at
# matched capacity, on the same seeds 4100-4102 the failed Gate 4 used. Running
# one seed here scores all 18 of its checkpoints against one set of references,
# so the comparison is WITHIN a single job:
#
#   observation (6d)   the condition that failed -- the in-job control
#   history     (18d)  hidden state inferable from motion, not supplied
#   physical    (24d)  the ball appended outright -- the constraint made visible
#
# If `physical` controls and `observation` does not, the failure was information
# and the chain is repairable. If `physical` fails too, the failure is the latent
# dynamics or the planner, the chain is not repairable by this route, and the
# measurement paper is the honest one. Both answers are worth the GPU.
#
# Identical to jobs 1259 and 1261-1263 in roots, split, horizon, cadence and
# search budget, so their oracle row -- 25/32 (78%) success, 0 collisions, task
# distance 0.1259 -- is the same experiment rather than a comparable one.
#
# --execute-blocks 1 is required, not merely chosen: `history` stacks consecutive
# block boundaries, so a planner observing every five blocks would feed its
# encoder frames five blocks apart and call them adjacent. `ObservationBuilder`
# refuses that rather than silently producing it.
#
# --objectives reward is the point. The goal objective needs no readout at all,
# so it cannot test whether the model can imagine a collision; it is also
# independently broken here (job 1229: the goal oracle collided in 90% of
# episodes and finished worse on the task than random).
#
# SIZING, measured. Jobs 1261-1263 took 1h37-1h41 for 6 checkpoints plus 1640s of
# references. 18 checkpoints is roughly 2.5h of scoring plus references, so ~3.5h
# per seed, and the early-termination speedup shrinks as policies stop crashing.
# The 8-hour limit leaves room for a `physical` arm that survives full episodes.
#SBATCH --job-name=bw-gate4-phys
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:a100_3g.20gb:1
#SBATCH --time=08:00:00
#SBATCH --output=slurm-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the BenchMARL repository root}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1
seed="${GATE4_SEED:?Set GATE4_SEED to one Stage 2 seed, 4100-4107}"
runs="${GATE4_RUNS:-outputs/state_input_1223/runs}"
data="${GATE4_DATA:-outputs/buzz_wire_1196/data}"
roots="${GATE4_ROOTS:-32}"
output="${SLURM_SUBMIT_DIR}/outputs/bw_gate4_phys_${seed}_${SLURM_JOB_ID}"
mkdir -p "$output"
exec > >(tee -a "$output/batch.log") 2>&1
printf 'Buzz Wire Gate 4 (input conditions): job=%s seed=%s roots=%s\n' \
    "$SLURM_JOB_ID" "$seed" "$roots"
printf 'CUDA_VISIBLE_DEVICES=%s\n' "${CUDA_VISIBLE_DEVICES:?}"
git rev-parse HEAD > "$output/commit.txt"
git status --short > "$output/git_status.txt"
git diff --binary > "$output/working_tree.patch"
mkdir "$output/source"
cp examples/world_model/*.py "$output/source/"
cp scripts/slurm/buzz_wire_gate4_physical.sbatch "$output/"

.venv/bin/python -u scripts/slurm/monitor_gpu_memory.py "$output/gpu_memory.csv" &
gpu_monitor_pid=$!
trap 'kill "$gpu_monitor_pid" 2>/dev/null || true' EXIT

# closed_loop persists after every policy, so a wall-clock kill loses at most the
# checkpoint in flight.
.venv/bin/python -u -m examples.world_model.closed_loop \
    "$runs" --data "$data" --device cuda \
    --states "$roots" --roots-split train \
    --horizon 5 --execute-blocks 1 \
    --num-samples 300 --num-iters 30 \
    --objectives reward --seeds "$seed" \
    --output "$output/gate4_seed${seed}.json" \
    2>&1 | tee "$output/gate4_seed${seed}.txt"

printf '\nCompleted Buzz Wire Gate 4 (input conditions) seed %s: %s\n' "$seed" "$output"
