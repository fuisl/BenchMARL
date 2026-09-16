#!/usr/bin/env bash
# Re-measure closed-loop control against a goal that means something.
#
# What this fixes. Jobs 1218/1221/1226 scored every learned planner on distance
# to a goal taken from the endpoint of a uniform random 25-step plan -- LeWM's
# own convention, and reachable by construction, which is why it was chosen. It
# is also task-orthogonal, and the goal oracle proves it: it reaches that goal in
# 20/20 episodes on both tasks and ends at task distance 0.8993 on Transport
# where random ends at 0.8988, and 1.0937 on Buzz Wire where random ends at
# 1.0852. Reaching the goal perfectly buys nothing. Every "fraction of the
# random -> goal-oracle gap closed" number in the project is therefore a
# fraction of nothing and is withdrawn.
#
# Why selecting a better random plan does not repair it. Over 300 candidates and
# 25 steps, the best endpoint by task distance improves on candidate 0 by 0.0012
# on Transport and 0.045 on Buzz Wire (measured, this commit). Random action
# sequences make almost no task progress, so no goal drawn from one can carry
# task content regardless of how it is selected.
#
# So the goal now comes from the strongest controller each task has:
# `--goal-source reference`. Transport gets the scenario's hand-written policy,
# the only thing that has ever scored a native success there. Buzz Wire has no
# heuristic and gets the reward oracle, which moves the ball from 1.095 to 0.557
# and succeeds 5/20. The goal stays an endpoint a policy actually reached from
# the same state under the same step budget, so reachability is untouched.
#
# Also new, and the audit's own contract item 5: a `zero` reference. A goal a
# motionless agent already satisfies is not a goal, and the project has never
# run the baseline that would have caught this.
#
# References first and alone -- the oracle rolls states x num_samples simulator
# environments per CEM iteration and must not share the slice. The checkpoint
# shards then reuse its cache and run sequentially: job 1218 lost three shards to
# OOM at eight, job 1221 lost one at four, and a Transport shard holds 4.8-6.4 GB
# of the 19.62 GB slice.
#SBATCH --job-name=goal-repair
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:a100_3g.20gb:1
#SBATCH --time=08:00:00
#SBATCH --output=slurm-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the BenchMARL repository root}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1
states="${CLOSED_LOOP_STATES:-20}"
seeds="${CLOSED_LOOP_SEEDS:-4100 4101 4102 4103}"
output="${SLURM_SUBMIT_DIR}/outputs/goal_repair_${SLURM_JOB_ID}"
mkdir -p "$output"
exec > >(tee -a "$output/batch.log") 2>&1
printf 'Goal-source repair: job=%s states=%s seeds=%s\n' \
    "$SLURM_JOB_ID" "$states" "$seeds"
printf 'CUDA_VISIBLE_DEVICES=%s\n' "${CUDA_VISIBLE_DEVICES:?}"
git rev-parse HEAD > "$output/commit.txt"
git status --short > "$output/git_status.txt"
git diff --binary > "$output/working_tree.patch"
mkdir "$output/source"
cp examples/world_model/*.py "$output/source/"
cp scripts/slurm/goal_source_repair.sbatch "$output/"

.venv/bin/python -u scripts/slurm/monitor_gpu_memory.py "$output/gpu_memory.csv" &
gpu_monitor_pid=$!
trap 'kill "$gpu_monitor_pid" 2>/dev/null || true' EXIT

# The goal oracle must score the same distance EpisodeStats reports, and the
# terminal observation a goal is drawn from must be the terminal frame rather
# than whatever a finished slot drifted to. Both are tests, so they run here.
srun --ntasks=1 .venv/bin/python -m pytest \
    test/test_world_model_closed_loop.py test/test_world_model.py -q \
    | tee "$output/contract_tests.txt"

# Buzz Wire first: it is the task that is actually controllable from these
# anchors -- the reward oracle moves the ball 1.095 -> 0.557 and succeeds 5/20 --
# so it is the one that can carry a control claim either way.
for task in buzz_wire transport; do
    if [ "$task" = buzz_wire ]; then
        runs=outputs/buzz_wire_1196/baselines
        data=outputs/buzz_wire_1196/data
    else
        runs=outputs/stage1_refit_1217/transport
        data=outputs/transport_data_1190
    fi
    cache="$output/${task}_refs.pt"

    printf '\n=== %s references (goal source: reference policy) ===\n' "$task"
    .venv/bin/python -u -m examples.world_model.closed_loop \
        "$runs" --data "$data" --device cuda --states "$states" \
        --goal-source reference --max-runs 0 --reference-cache "$cache" \
        --output "$output/${task}_references.json" --wandb \
        2>&1 | tee "$output/${task}_references.txt"

    for seed in $seeds; do
        printf '\n=== %s seed %s ===\n' "$task" "$seed"
        .venv/bin/python -u -m examples.world_model.closed_loop \
            "$runs" --data "$data" --device cuda --states "$states" \
            --goal-source reference --seeds "$seed" \
            --skip-references --reference-cache "$cache" \
            --output "$output/${task}_seed${seed}.json" --wandb \
            2>&1 | tee "$output/${task}_seed${seed}.txt"
    done
done

printf '\nPeak GPU memory:\n'
awk -F, 'NR>1 {gsub(/ MiB/,"",$4); if ($4+0>m) m=$4+0} END {print "  " m " MiB of 20096"}' \
    "$output/gpu_memory.csv"
printf 'Completed goal-source repair: %s\n' "$output"
