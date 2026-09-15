#!/usr/bin/env bash
# The goal objective's missing ceiling, and the Transport shards job 1218 lost.
#
# Two things job 1218 established.
#
# First, the reward oracle is not a ceiling for the goal objective. It ended
# 0.839 from the generated goal on Buzz Wire where random ended 0.256: it
# maximizes task reward, and the goal is drawn from an arbitrary trajectory, so
# optimizing one moves away from the other. Every goal result so far therefore
# has a random floor and nothing above it. `goal_oracle` is the same CEM, the
# same budget and the same goals with the true simulator as dynamics and goal
# distance as the objective -- the ceiling the comparison needs. Both tasks get
# it, including Buzz Wire, whose learned policies are already complete and valid
# from 1218 and are not re-run here.
#
# Second, eight shards do not fit this slice. Three Transport shards died of
# CUDA OOM, two of them failing on 48 MiB and 94 MiB requests with under 45 MiB
# free -- a full slice, not fragmentation, so the fix is fewer processes rather
# than allocator flags. Per-shard use ran 2.5-6.4 GB, so four fit 20 GB with
# real headroom where eight did not. Only Transport is re-run: one shard per
# training seed, all three predictors and both regimes inside each.
#SBATCH --job-name=goal-ceiling
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
# Four shards, a monitor and single-threaded BLAS: 28 CPUs/72G was sized
# for eight shards and does not fit beside another user's job on this
# 32-CPU/84G node, which left the submission pending on Resources.
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --gres=gpu:a100_3g.20gb:1
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the BenchMARL repository root}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1
states="${CLOSED_LOOP_STATES:-20}"
seeds="${CLOSED_LOOP_SEEDS:-4100 4101 4102 4103}"
output="${SLURM_SUBMIT_DIR}/outputs/goal_ceiling_${SLURM_JOB_ID}"
mkdir -p "$output"
exec > >(tee -a "$output/batch.log") 2>&1
printf 'Goal ceiling + Transport recovery: job=%s states=%s seeds=%s\n' \
    "$SLURM_JOB_ID" "$states" "$seeds"
printf 'CUDA_VISIBLE_DEVICES=%s\n' "${CUDA_VISIBLE_DEVICES:?}"
git rev-parse HEAD > "$output/commit.txt"
git status --short > "$output/git_status.txt"
git diff --binary > "$output/working_tree.patch"
mkdir "$output/source"
cp examples/world_model/*.py "$output/source/"
cp scripts/slurm/closed_loop_goal_ceiling.sbatch "$output/"

.venv/bin/python -u scripts/slurm/monitor_gpu_memory.py "$output/gpu_memory.csv" &
gpu_monitor_pid=$!
trap 'kill "$gpu_monitor_pid" 2>/dev/null || true' EXIT

# The goal oracle must score the same distance EpisodeStats reports, or it
# bounds nothing. That equality is a test, so it runs in the allocation.
srun --ntasks=1 .venv/bin/python -m pytest \
    test/test_world_model_stage0.py test/test_world_model_closed_loop.py -q \
    | tee "$output/contract_tests.txt"

bw=outputs/stage1_refit_1217/buzz_wire
bw_data=outputs/buzz_wire_1196/data
tr=outputs/stage1_refit_1217/transport
tr_data=outputs/transport_data_1190

# ---- phase 1: references, alone, cached -------------------------------------
# random, oracle and goal_oracle. Both oracles roll states x num_samples
# simulator environments through every CEM iteration, which is the one workload
# that collapses when it shares the slice -- 4827 s shared against 299 s alone.
reference() {
    local name="$1" runs="$2" data="$3"
    printf '\n=== references: %s ===\n' "$name"
    .venv/bin/python -u -m examples.world_model.closed_loop \
        "$runs" --data "$data" --device cuda --states "$states" \
        --reference-cache "$output/${name}_refs.pt" --max-runs 0 \
        --output "$output/${name}_references.json" --wandb \
        2>&1 | tee "$output/${name}_references.txt"
}
reference buzz_wire "$bw" "$bw_data"
reference transport "$tr" "$tr_data"

# ---- phase 2: Transport learned policies, four shards -----------------------
# A shard takes both references and goals from the cache, so it never builds the
# states x num_samples scratch simulator -- the bulk of a process's memory.
shard() {
    local seed="$1"
    .venv/bin/python -u -m examples.world_model.closed_loop \
        "$tr" --data "$tr_data" --device cuda --states "$states" \
        --seeds "$seed" --skip-references \
        --reference-cache "$output/transport_refs.pt" \
        --output "$output/transport_seed${seed}.json" --wandb \
        > "$output/transport_seed${seed}.txt" 2>&1
}

printf '\n=== transport learned policies: %s shards ===\n' "$(echo $seeds | wc -w)"
pids=()
for seed in $seeds; do
    shard "$seed" & pids+=($!)
done

status=0
for pid in "${pids[@]}"; do
    wait "$pid" || status=1
done

printf '\nPeak GPU memory:\n'
awk -F, 'NR>1 {gsub(/ MiB/,"",$4); if ($4+0>m) m=$4+0} END {print "  " m " MiB of 20096"}' \
    "$output/gpu_memory.csv"
printf 'Completed: %s\n' "$output"
[ "$status" -eq 0 ]
