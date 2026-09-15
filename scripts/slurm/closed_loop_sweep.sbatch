#!/usr/bin/env bash
# Closed-loop control, re-run on the Stage 0/1 code with refitted readouts.
#
# Job 1216's learned results are not usable. It ran before the measurement
# repairs: its goal planner aimed at endpoints computed by the defective
# block-boundary helper, and its reward planner used both the collapsing
# survival prefix and readouts trained under the mask that discarded 79% of
# Buzz Wire's terminations. Only its random and oracle references were clean.
#
# Concurrency is shaped by what job 1216 measured rather than by guesswork.
# A MIG without MPS time-slices processes, so packing them hurts anything that
# issues many small kernels. The oracle is the extreme case -- it rolls
# states x num_samples simulator environments through every CEM iteration, and
# measured 4827 s while sharing against 299 s alone. The learned planners are
# the opposite: they roll latents on the GPU and step the simulator only to
# execute, so they pack well.
#
# So the job is two phases. References run first, one process per task, alone,
# and are cached. The learned sweep then runs many shards that reuse that cache
# -- and because a shard needs neither the oracle nor goal generation, it never
# builds the 6000-environment scratch simulator, which is the bulk of a
# process's memory. That is what makes a wide sweep fit one 20 GB slice.
#SBATCH --job-name=closed-loop-sweep
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=28
#SBATCH --mem=72G
#SBATCH --gres=gpu:a100_3g.20gb:1
#SBATCH --time=06:00:00
#SBATCH --output=slurm-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the BenchMARL repository root}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1
states="${CLOSED_LOOP_STATES:-20}"
# One shard per training seed: the widest split the grid allows, and each shard
# then covers all three predictors in both regimes.
seeds="${CLOSED_LOOP_SEEDS:-4100 4101 4102 4103}"
output="${SLURM_SUBMIT_DIR}/outputs/closed_loop_sweep_${SLURM_JOB_ID}"
mkdir -p "$output"
exec > >(tee -a "$output/batch.log") 2>&1
printf 'Closed-loop sweep: job=%s states=%s seeds=%s\n' "$SLURM_JOB_ID" "$states" "$seeds"
printf 'CUDA_VISIBLE_DEVICES=%s\n' "${CUDA_VISIBLE_DEVICES:?}"
git rev-parse HEAD > "$output/commit.txt"
git status --short > "$output/git_status.txt"
git diff --binary > "$output/working_tree.patch"
mkdir "$output/source"
cp examples/world_model/*.py "$output/source/"
cp scripts/slurm/closed_loop_sweep.sbatch "$output/"

.venv/bin/python -u scripts/slurm/monitor_gpu_memory.py "$output/gpu_memory.csv" &
gpu_monitor_pid=$!
trap 'kill "$gpu_monitor_pid" 2>/dev/null || true' EXIT

# The Stage 0 contracts are what make these numbers mean anything.
srun --ntasks=1 .venv/bin/python -m pytest \
    test/test_world_model_stage0.py test/test_world_model_closed_loop.py -q \
    | tee "$output/contract_tests.txt"

# Readouts refitted on frozen dynamics in job 1217; dynamics are job 1196/1194's.
bw=outputs/stage1_refit_1217/buzz_wire
bw_data=outputs/buzz_wire_1196/data
tr=outputs/stage1_refit_1217/transport
tr_data=outputs/transport_data_1190

# ---- phase 1: references, alone, cached -------------------------------------
# Not concurrent, and deliberately so: this is the one workload that collapses
# when it shares the slice.
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

# ---- phase 2: learned policies, packed --------------------------------------
shard() {
    local name="$1" runs="$2" data="$3" seed="$4"
    .venv/bin/python -u -m examples.world_model.closed_loop \
        "$runs" --data "$data" --device cuda --states "$states" \
        --seeds "$seed" --skip-references \
        --reference-cache "$output/${name}_refs.pt" \
        --output "$output/${name}_seed${seed}.json" --wandb \
        > "$output/${name}_seed${seed}.txt" 2>&1
}

printf '\n=== learned policies: %s shards per task ===\n' "$(echo $seeds | wc -w)"
pids=()
for seed in $seeds; do
    shard buzz_wire "$bw" "$bw_data" "$seed" & pids+=($!)
    shard transport "$tr" "$tr_data" "$seed" & pids+=($!)
done

status=0
for pid in "${pids[@]}"; do
    wait "$pid" || status=1
done
[ "$status" -eq 0 ]

printf '\nPeak GPU memory:\n'
awk -F, 'NR>1 {gsub(/ MiB/,"",$4); if ($4+0>m) m=$4+0} END {print "  " m " MiB of 20096"}' \
    "$output/gpu_memory.csv"
printf 'Completed closed-loop sweep: %s\n' "$output"
