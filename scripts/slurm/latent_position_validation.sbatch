#!/usr/bin/env bash
# Decode the 144 Stage-2 latent checkpoints to individual-agent positions.
# Six input/regime groups run concurrently inside one 20 GB MIG allocation.
# Each group fits probes on train roots, selects on validation roots, evaluates
# test roots, and renders one shared-root filmstrip. A final merge renders the
# aggregate one-step comparison and recursive horizon curves.
#SBATCH --job-name=latent-position
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

runs="${LATENT_POSITION_RUNS:-outputs/state_input_1223/runs}"
data="${LATENT_POSITION_DATA:-outputs/buzz_wire_1196/data}"
direct="${LATENT_POSITION_DIRECT:-outputs/agent_position_concurrent_1291/result}"
probe="${LATENT_POSITION_PROBE:-mlp}"
output="${SLURM_SUBMIT_DIR}/outputs/latent_position_${probe}_${SLURM_JOB_ID}"
mkdir -p "$output"
exec > >(tee -a "$output/batch.log") 2>&1

git rev-parse HEAD > "$output/commit.txt"
git status --short > "$output/git_status.txt"
git diff --binary > "$output/working_tree.patch"
mkdir "$output/source"
cp examples/world_model/{latent_position,dataset,horizon_rollout,physical_response}.py "$output/source/"
cp scripts/slurm/latent_position_validation.sbatch "$output/"

printf 'runs=%s data=%s direct=%s probe=%s CUDA_VISIBLE_DEVICES=%s\n' \
    "$runs" "$data" "$direct" "$probe" "${CUDA_VISIBLE_DEVICES:?}"
.venv/bin/python -u scripts/slurm/monitor_gpu_memory.py "$output/gpu_memory.csv" &
gpu_monitor_pid=$!
trap 'kill "$gpu_monitor_pid" 2>/dev/null || true' EXIT

pids=()
names=()
for state_input in observation history physical; do
    for regime in correlated independent; do
        name="${state_input}_${regime}"
        group="$output/$name"
        printf 'START %s\n' "$name"
        .venv/bin/python -u -m examples.world_model.latent_position evaluate \
            "$runs" --data "$data" --direct-results "$direct" \
            --state-input "$state_input" --regime "$regime" \
            --probe "$probe" --device cuda --output "$group" \
            > "$output/$name.log" 2>&1 &
        pids+=("$!")
        names+=("$name")
    done
done

failed=0
for index in "${!pids[@]}"; do
    if wait "${pids[$index]}"; then
        printf 'DONE  %s\n' "${names[$index]}"
    else
        printf 'FAILED %s (see %s/%s.log)\n' \
            "${names[$index]}" "$output" "${names[$index]}"
        failed=1
    fi
done
if ((failed)); then
    exit 1
fi

.venv/bin/python -u -m examples.world_model.latent_position merge \
    "$output"/*/summary.json --output "$output"
printf '\nCompleted latent-position validation: %s\n' "$output"
