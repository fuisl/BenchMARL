#!/usr/bin/env bash
# Escalated Transport oracle budget/horizon/episode-length push (2026-09-14),
# following up on job 1182's 0/18 success result (all-timeout, package-contact
# bottleneck at horizon=5/max_steps=100). Two sweeps:
#   A) horizon_budget — fixed 100-step episode, push CEM horizon and/or
#      search budget (reuses job 1182's exact development state bank).
#   B) longepisode — 300-step episode (3x), replan1/horizon10 CEM settings
#      (each run regenerates its own matching-task state bank from the same
#      state_seed=0, since a max_steps=300 bank cannot equal a 100-step one).
#SBATCH --job-name=transport-oracle-horizon
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:a100_3g.20gb:1
#SBATCH --time=03:00:00
#SBATCH --output=slurm-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the BenchMARL repository root}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
transport_workers="${TRANSPORT_WORKERS:-4}"
transport_output="${SLURM_SUBMIT_DIR}/outputs/transport_oracle_${SLURM_JOB_ID}"
state_bank="${SLURM_SUBMIT_DIR}/outputs/transport_oracle_1182/initial_states.pt"
mkdir -p "$transport_output"
exec > >(tee -a "$transport_output/batch.log") 2>&1
printf 'Transport oracle escalation batch: job=%s output=%s\n' "$SLURM_JOB_ID" "$transport_output"
printf 'CUDA_VISIBLE_DEVICES=%s workers=%s state_bank=%s\n' "${CUDA_VISIBLE_DEVICES:?}" "$transport_workers" "$state_bank"
test -f "$state_bank" || { echo "Missing state bank: $state_bank" >&2; exit 1; }
.venv/bin/python -u scripts/slurm/monitor_gpu_memory.py "$transport_output/gpu_memory.csv" &
gpu_monitor_pid=$!
# Only suppress an already-exited monitoring process during job cleanup.
trap 'kill "$gpu_monitor_pid" 2>/dev/null || true' EXIT

# Validate CUDA inside the actual allocation before any real experiment.
srun --ntasks=1 .venv/bin/python -m examples.world_model.evaluate \
    --config-name sweep/oracle_smoke task=vmas/transport task.max_steps=5 \
    experiment.sampling_device=cuda "hydra.run.dir=$transport_output/smoke"

# Sweep A: fixed 100-step episode, push horizon and/or search budget. Reuses
# job 1182's exact 20 development states (same task.max_steps=100, so the
# state-bank task-config equality check in evaluate.py passes).
srun --ntasks=1 .venv/bin/python -m examples.world_model.evaluate \
    --config-name sweep/transport_oracle_horizon --multirun \
    hydra/launcher=joblib "hydra.launcher.n_jobs=$transport_workers" \
    'experiment.loggers=[csv,wandb]' \
    "evaluation.states_file=$state_bank" \
    "hydra.sweep.dir=$transport_output/horizon_budget"
srun --ntasks=1 .venv/bin/python -m examples.world_model.summarize_comparisons \
    "$transport_output/horizon_budget"

# Sweep B: 300-step episode (3x). No evaluation.states_file override -- each
# run regenerates its own bank from state_seed=0 under its own task.max_steps,
# which the summarizer verifies still matches bit-for-bit across all 6 runs.
srun --ntasks=1 .venv/bin/python -m examples.world_model.evaluate \
    --config-name sweep/transport_oracle_longepisode --multirun \
    hydra/launcher=joblib "hydra.launcher.n_jobs=$transport_workers" \
    'experiment.loggers=[csv,wandb]' \
    "hydra.sweep.dir=$transport_output/longepisode"
srun --ntasks=1 .venv/bin/python -m examples.world_model.summarize_comparisons \
    "$transport_output/longepisode"

printf 'Completed Transport oracle escalation: %s/{horizon_budget,longepisode}/comparison.csv\n' "$transport_output"
