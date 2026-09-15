#!/usr/bin/env bash
# The interaction-density experiment: run M3 + M4 end to end on Buzz Wire.
#
# C7 failed on Transport -- no baseline captures any of the cross-agent effect
# (relational scores 1.002x the no-response floor). The diagnosis is upstream of
# the models: Transport's data barely contains the interaction, with 0/239 test
# anchors showing an effect after one primitive step and 33/239 after 25.
#
# Buzz Wire's rigid joint couples the agents structurally. A smoke collection
# shows an effect on 6/6 anchors including at the first step, so the signal C7
# needs is present by construction rather than by luck of contact. If relational
# captures the effect here and not on Transport, the benefit tracks measured
# cross-agent coupling rather than task identity -- which is claim C10, and the
# positive result the Transport line could not produce.
#
# Buzz Wire also terminates on wall contact, so this bank should finally contain
# the termination positives that both Transport and Dropout lack.
#SBATCH --job-name=buzz-wire-pipeline
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --gres=gpu:a100_3g.20gb:1
#SBATCH --time=03:00:00
#SBATCH --output=slurm-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the BenchMARL repository root}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
workers="${WORLD_MODEL_WORKERS:-5}"
seeds="${WORLD_MODEL_SEEDS:-4100,4101,4102,4103,4104,4105,4106,4107}"
output="${SLURM_SUBMIT_DIR}/outputs/buzz_wire_${SLURM_JOB_ID}"
data_output="$output/data"
mkdir -p "$output"
exec > >(tee -a "$output/batch.log") 2>&1
printf 'Buzz Wire pipeline: job=%s workers=%s seeds=%s\n' "$SLURM_JOB_ID" "$workers" "$seeds"
printf 'CUDA_VISIBLE_DEVICES=%s\n' "${CUDA_VISIBLE_DEVICES:?}"

.venv/bin/python -u scripts/slurm/monitor_gpu_memory.py "$output/gpu_memory.csv" &
gpu_monitor_pid=$!
# Only suppress an already-exited monitoring process during job cleanup.
trap 'kill "$gpu_monitor_pid" 2>/dev/null || true' EXIT

# Same M3 budgets and seeds as the Transport bank; only the task differs. The
# shipped heuristic source is Transport-specific and stays off.
srun --ntasks=1 .venv/bin/python -m examples.world_model.collect \
    task=vmas/buzz_wire dataset.include_heuristic=false \
    "hydra.run.dir=$data_output"

srun --ntasks=1 .venv/bin/python - "$data_output" <<'PY'
import sys
from examples.world_model.dataset import OfflineSequences

for regime in ("independent", "correlated"):
    for split in ("train", "validation", "test"):
        dataset = OfflineSequences(sys.argv[1], regime, split, action_block=5)
        print(regime, split, len(dataset), "validated snippets", flush=True)
PY

# Same three baselines, matched capacity budget, same eight seeds as job 1194.
srun --ntasks=1 .venv/bin/python -m examples.world_model.train \
    --config-name sweep/world_model_baselines --multirun \
    hydra/launcher=joblib "hydra.launcher.n_jobs=$workers" \
    "data.root=$data_output" \
    "data.regime=correlated,independent" \
    "seed=$seeds" \
    "hydra.sweep.dir=$output/baselines"

printf 'Completed Buzz Wire pipeline: %s/baselines\n' "$output"
