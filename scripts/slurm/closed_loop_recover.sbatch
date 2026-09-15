#!/usr/bin/env bash
# Re-score the Transport shards lost to CUDA OOM, one at a time.
#
# Sizing this by the mean per-shard memory was wrong twice. Job 1218 ran eight
# shards and lost three; job 1221 ran four and lost one. The measured
# steady-state for a *Transport* shard is 4.8-6.4 GB, not the ~3 GB average
# across both tasks, so four of them need ~19 GB of a 19.62 GB slice and the
# first one to hit a transient allocation dies. Buzz Wire shards are the cheap
# ones that dragged the average down.
#
# So this runs shards sequentially. A shard is minutes of work and the slice is
# free by the time this starts; there is nothing to gain from packing it and a
# demonstrated way to lose.
#
# References come from the parent job's cache, so the goals and the random,
# oracle and goal_oracle episodes are the same ones the surviving shards were
# scored against, and no scratch simulator is built.
#SBATCH --job-name=cl-recover
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --gres=gpu:a100_3g.20gb:1
#SBATCH --time=03:00:00
#SBATCH --output=slurm-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the BenchMARL repository root}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1
states="${CLOSED_LOOP_STATES:-20}"
# Whatever the parent job failed to finish. Override to recover a different set.
seeds="${CLOSED_LOOP_SEEDS:-4101}"
parent="${CLOSED_LOOP_PARENT:-outputs/goal_ceiling_1221}"
output="${SLURM_SUBMIT_DIR}/outputs/closed_loop_recover_${SLURM_JOB_ID}"
mkdir -p "$output"
exec > >(tee -a "$output/batch.log") 2>&1
printf 'Closed-loop recovery: job=%s seeds=%s parent=%s\n' \
    "$SLURM_JOB_ID" "$seeds" "$parent"
printf 'CUDA_VISIBLE_DEVICES=%s\n' "${CUDA_VISIBLE_DEVICES:?}"
git rev-parse HEAD > "$output/commit.txt"
mkdir "$output/source"
cp examples/world_model/*.py "$output/source/"
cp scripts/slurm/closed_loop_recover.sbatch "$output/"

cache="${SLURM_SUBMIT_DIR}/${parent}/transport_refs.pt"
[ -f "$cache" ] || { echo "No reference cache at $cache"; exit 1; }

.venv/bin/python -u scripts/slurm/monitor_gpu_memory.py "$output/gpu_memory.csv" &
gpu_monitor_pid=$!
trap 'kill "$gpu_monitor_pid" 2>/dev/null || true' EXIT

for seed in $seeds; do
    printf '\n=== transport seed %s ===\n' "$seed"
    .venv/bin/python -u -m examples.world_model.closed_loop \
        outputs/stage1_refit_1217/transport --data outputs/transport_data_1190 \
        --device cuda --states "$states" \
        --seeds "$seed" --skip-references --reference-cache "$cache" \
        --output "$output/transport_seed${seed}.json" --wandb \
        2>&1 | tee "$output/transport_seed${seed}.txt"
done

printf '\nPeak GPU memory:\n'
awk -F, 'NR>1 {gsub(/ MiB/,"",$4); if ($4+0>m) m=$4+0} END {print "  " m " MiB of 20096"}' \
    "$output/gpu_memory.csv"
printf 'Completed recovery: %s\n' "$output"
