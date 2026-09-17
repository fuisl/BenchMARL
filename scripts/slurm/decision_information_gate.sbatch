#!/usr/bin/env bash
# Gate 0d: localize whether planning loses information in the encoder, latent
# transition, scalar reward readout, or the model-selected CEM distribution.
# The six input/regime groups run concurrently on one 20 GB MIG slice. Each
# group evaluates the three frozen seed-4100 conditioner checkpoints in series.
# No encoder, dynamics model, or existing reward head is retrained.
#SBATCH --job-name=decision-info
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --gres=gpu:a100_3g.20gb:1
#SBATCH --time=16:00:00
#SBATCH --output=slurm-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the BenchMARL repository root}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="${SLURM_SUBMIT_DIR}${PYTHONPATH:+:$PYTHONPATH}"

runs="${GATE0D_RUNS:-outputs/state_input_1223/runs}"
data="${GATE0D_DATA:-outputs/buzz_wire_1196/data}"
output="${SLURM_SUBMIT_DIR}/outputs/decision_information_${SLURM_JOB_ID}"
mkdir -p "$output/bank" "$output/source"
exec > >(tee -a "$output/batch.log") 2>&1

printf 'Gate 0d: job=%s runs=%s data=%s CUDA_VISIBLE_DEVICES=%s\n' \
    "$SLURM_JOB_ID" "$runs" "$data" "${CUDA_VISIBLE_DEVICES:?}"
printf 'Frozen panel: seed 4100, 3 inputs x 2 regimes x 3 conditioners\n'
git rev-parse HEAD > "$output/commit.txt"
git status --short > "$output/git_status.txt"
git diff --binary > "$output/working_tree.patch"
cp examples/world_model/{decision_information,cem,dataset,model_input,plan_ranking}.py "$output/source/"
cp scripts/slurm/decision_information_gate.sbatch "$output/"

.venv/bin/python -u scripts/slurm/monitor_gpu_memory.py "$output/gpu_memory.csv" &
gpu_monitor_pid=$!
trap 'kill "$gpu_monitor_pid" 2>/dev/null || true' EXIT

.venv/bin/python -m pytest -q \
    test/test_world_model_decision_information.py \
    test/test_world_model.py::test_cem_batch_reproducibility_and_validation \
    test/test_world_model_dataset.py::test_action_blocks_preserve_agent_and_primitive_order \
    | tee "$output/preflight_tests.txt"

# Build simulator truth once. All 18 models are compared on this exact bank.
.venv/bin/python -u -m examples.world_model.decision_information build-bank \
    --data "$data" --output "$output/bank" --device cuda \
    --roots 16 --random-plans 300 --horizon 5 \
    --num-samples 300 --num-elites 30 --num-iters 30 --seed 7100

# Six independent evaluation processes intentionally share the 20 GB slice.
# Each process handles its three conditioner checkpoints serially, keeping the
# simultaneously resident model count bounded while fully using the allocation.
pids=()
names=()
for state_input in observation history physical; do
    for regime in correlated independent; do
        name="${state_input}_${regime}"
        group="$output/$name"
        printf 'START %s\n' "$name"
        .venv/bin/python -u -m examples.world_model.decision_information evaluate \
            "$runs" --data "$data" --bank "$output/bank/candidate_bank.pt" \
            --output "$group" --state-input "$state_input" --regime "$regime" \
            --seed 4100 --topk 30 --stage-roots 8 \
            --num-samples 300 --num-elites 30 --num-iters 30 \
            --head-epochs 300 --head-patience 30 --cem-seed 7300 --device cuda \
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

.venv/bin/python -u -m examples.world_model.decision_information merge \
    "$output"/*/summary.json --output "$output" \
    | tee "$output/aggregate_table.txt"

printf '\nPeak GPU memory:\n'
awk -F, 'NR>1 {gsub(/ MiB/,"",$4); if ($4+0>m) m=$4+0} END {print "  " m " MiB"}' \
    "$output/gpu_memory.csv"
printf 'Completed Gate 0d with rendered outputs: %s\n' "$output"
