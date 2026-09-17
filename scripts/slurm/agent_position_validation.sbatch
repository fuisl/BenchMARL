#!/usr/bin/env bash
# Bottom-up world validation: directly predict each agent's next (x, y) with
# one transition model shared across agents. The models are small, so one 20 GB
# MIG allocation runs six independent fits concurrently instead of queueing one
# allocation per seed.
#SBATCH --job-name=agent-position
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
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

data="${POSITION_DATA:-outputs/buzz_wire_1196/data}"
max_concurrent="${POSITION_MAX_CONCURRENT:-6}"
output="${SLURM_SUBMIT_DIR}/outputs/agent_position_concurrent_${SLURM_JOB_ID}"
mkdir -p "$output"
exec > >(tee -a "$output/batch.log") 2>&1

git rev-parse HEAD > "$output/commit.txt"
git status --short > "$output/git_status.txt"
git diff --binary > "$output/working_tree.patch"
mkdir "$output/source"
cp examples/world_model/{agent_position,dataset,models}.py "$output/source/"
cp scripts/slurm/agent_position_validation.sbatch "$output/"

printf 'data=%s concurrent=%s CUDA_VISIBLE_DEVICES=%s\n' \
    "$data" "$max_concurrent" "${CUDA_VISIBLE_DEVICES:?}"

.venv/bin/python -u scripts/slurm/monitor_gpu_memory.py \
    "$output/gpu_memory.csv" &
gpu_monitor_pid=$!
trap 'kill "$gpu_monitor_pid" 2>/dev/null || true' EXIT

run_one() {
    local seed="$1" state_input="$2" regime="$3" kind="$4"
    local name="seed${seed}_${state_input}_${regime}_${kind}"
    local run_output="$output/result/$name"
    printf 'START %s\n' "$name"
    .venv/bin/python -u -m examples.world_model.agent_position \
        --data "$data" \
        --output "$run_output" \
        --regimes "$regime" \
        --state-inputs "$state_input" \
        --kinds "$kind" \
        --seed "$seed" \
        --device cuda \
        > "$output/${name}.log" 2>&1
    printf 'DONE  %s\n' "$name"
}

pids=()
names=()
wait_batch() {
    local failed=0 index
    for index in "${!pids[@]}"; do
        if ! wait "${pids[$index]}"; then
            printf 'FAILED %s (see %s/%s.log)\n' \
                "${names[$index]}" "$output" "${names[$index]}"
            failed=1
        fi
    done
    pids=()
    names=()
    if ((failed)); then
        return 1
    fi
}

for seed in 4100 4101 4102 4103 4104 4105 4106 4107; do
    for state_input in observation history physical; do
        for regime in correlated independent; do
            for kind in independent joint relational; do
                name="seed${seed}_${state_input}_${regime}_${kind}"
                run_one "$seed" "$state_input" "$regime" "$kind" &
                pids+=("$!")
                names+=("$name")
                if ((${#pids[@]} >= max_concurrent)); then
                    wait_batch
                fi
            done
        done
    done
done
if ((${#pids[@]})); then
    wait_batch
fi

printf '\nCompleted agent-position validation: %s\n' "$output"
