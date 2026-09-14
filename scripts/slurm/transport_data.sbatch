#!/usr/bin/env bash
# M3 fixed development dataset, following the local MIG pilot convention.
#SBATCH --job-name=transport-data
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:a100_3g.20gb:1
#SBATCH --time=00:20:00
#SBATCH --output=slurm-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the BenchMARL repository root}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
data_output="${SLURM_SUBMIT_DIR}/outputs/transport_data_${SLURM_JOB_ID}"
mkdir -p "$data_output"
exec > >(tee -a "$data_output/batch.log") 2>&1
srun --ntasks=1 .venv/bin/python scripts/slurm/monitor_gpu_memory.py \
    "$data_output/gpu_memory.csv" --samples=1
.venv/bin/python -u scripts/slurm/monitor_gpu_memory.py "$data_output/gpu_memory.csv" &
data_monitor_pid=$!
trap 'kill "$data_monitor_pid" 2>/dev/null || true' EXIT

# A genuine collection run retains the default CSV and online wandb loggers.
# CLI overrides can name the M2 bank excluded from all new initial states.
srun --ntasks=1 .venv/bin/python -m examples.world_model.collect \
    "$@" "hydra.run.dir=$data_output"
srun --ntasks=1 .venv/bin/python - "$data_output" <<'PY'
import sys
from examples.world_model.dataset import OfflineSequences

for regime in ("independent", "correlated"):
    for split in ("train", "validation", "test"):
        dataset = OfflineSequences(sys.argv[1], regime, split, action_block=5)
        print(regime, split, len(dataset), "validated snippets", flush=True)
PY
