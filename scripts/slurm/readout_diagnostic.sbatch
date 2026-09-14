#!/usr/bin/env bash
# Is plan ranking broken by the dynamics or by the readout?
#
# Ranking gives Spearman ~0.05 for every baseline on Buzz Wire, where the cost
# landscape is fully non-degenerate (117/117 rankable) and the models capture 26%
# of the cross-agent effect. A model plan cost passes through two learned stages,
# so this scores the same candidates a third way -- the learned readout applied
# to the simulator's own latents -- to separate them:
#
#   J_true     simulator states,  simulator rewards   (ground truth)
#   J_readout  simulator states,  learned readout     (readout error only)
#   J_model    predicted latents, learned readout     (both errors)
#
# A large recovery from J_model to J_readout means the rollout is what breaks
# ranking; near zero means the readout cannot order plans and better dynamics
# would not help.
#
# Small by design: a 10 GB MIG slice, since the work is one batched simulator
# rollout plus forward passes over 48 checkpoints. Runs alongside job 1201 on the
# 20 GB slice rather than competing with it.
#SBATCH --job-name=readout-diag
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:a100_2g.10gb:1
#SBATCH --time=01:00:00
#SBATCH --output=slurm-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the BenchMARL repository root}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
output="${SLURM_SUBMIT_DIR}/outputs/readout_diag_${SLURM_JOB_ID}"
mkdir -p "$output"
exec > >(tee -a "$output/batch.log") 2>&1
printf 'Readout diagnostic: job=%s\n' "$SLURM_JOB_ID"
printf 'CUDA_VISIBLE_DEVICES=%s\n' "${CUDA_VISIBLE_DEVICES:?}"

.venv/bin/python -u scripts/slurm/monitor_gpu_memory.py "$output/gpu_memory.csv" &
gpu_monitor_pid=$!
# Only suppress an already-exited monitoring process during job cleanup.
trap 'kill "$gpu_monitor_pid" 2>/dev/null || true' EXIT

# Buzz Wire first: it is the task where ranking should have worked.
srun --ntasks=1 .venv/bin/python -m examples.world_model.readout_diagnostic \
    outputs/buzz_wire_1196/baselines --data outputs/buzz_wire_1196/data \
    --device cuda --states 96 --candidates 48 | tee "$output/buzz_wire.txt"

# Transport for contrast: there the readout works well in isolation (R^2 ~ 0.74)
# but almost no candidate carries reward, so a flat J_readout there would say
# something different -- that the cost itself carries no ordering.
srun --ntasks=1 .venv/bin/python -m examples.world_model.readout_diagnostic \
    outputs/interaction_control_1194/transport --data outputs/transport_data_1190 \
    --device cuda --states 96 --candidates 48 | tee "$output/transport.txt"

printf 'Completed readout diagnostic: %s\n' "$output"
