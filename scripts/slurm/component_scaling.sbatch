#!/usr/bin/env bash
# Component scaling: which part of the world model limits counterfactual
# fidelity, and is it capacity or data?
#
# One-at-a-time scaling around the lewm_reference base (joint, correlated,
# lambda=0.09), four seeds per cell. The audits localize the lost cross-agent
# information to the encoder (0.7% of parameters) and exonerate the predictor
# (76%), so the registered expectations are:
#
#   S1  encoder width/depth and latent dim move R_latent and cross E_CF;
#       predictor depth does not.
#   S2  at a fixed bank, capacity alone does not improve test E_CF; the data
#       fraction does (Wang et al. 2025 found depth scaling barely helps offline).
#
# Data cells hold the NUMBER OF UPDATES fixed (500 = 5 batches x 100 epochs at
# the full bank), so less data is not also less optimisation. Root episodes are
# subsampled, nested, by `data.train_fraction`.
#
# `dim` is also the predictor's token width, so the dim cells scale both; they
# are reported as "latent width (with predictor width)", not as encoder cells.
#
# Run directories are <cell>_seed<N>, so the evaluators' --label-from-dir keeps
# the cells apart.
#SBATCH --job-name=comp-scale
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=14
#SBATCH --mem=48G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=06:00:00
#SBATCH --output=slurm-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the BenchMARL repository root}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="${SLURM_SUBMIT_DIR}${PYTHONPATH:+:$PYTHONPATH}"

data="${SCALE_DATA:-outputs/buzz_wire_1196/data}"
kind="${SCALE_KIND:-joint}"
regime="${SCALE_REGIME:-correlated}"
seeds="${SCALE_SEEDS:-4100 4101 4102 4103}"
max_concurrent="${SCALE_MAX_CONCURRENT:-12}"
output="${SLURM_SUBMIT_DIR}/outputs/component_scaling_${SLURM_JOB_ID}"
mkdir -p "$output/source" "$output/runs"
exec > >(tee -a "$output/batch.log") 2>&1

# name|hydra overrides
cells=(
    "base|"
    "encw128|model.encoder_hidden_dim=128"
    "encw2048|model.encoder_hidden_dim=2048"
    "encd2|model.encoder_depth=2"
    "encd4|model.encoder_depth=4"
    "encbig|model.encoder_depth=3 model.encoder_hidden_dim=1024"
    "predd1|model.depth=1"
    "predd2|model.depth=2"
    "predd12|model.depth=12"
    "condq|model.conditioner_budget=270768"
    "cond4|model.conditioner_budget=4332288"
    "dim96|model.dim=96"
    "dim384|model.dim=384"
    "data025|data.train_fraction=0.25 train.dynamics_epochs=500"
    "data05|data.train_fraction=0.5 train.dynamics_epochs=250"
)

printf 'Component scaling: job=%s data=%s kind=%s regime=%s seeds=[%s]\n' \
    "$SLURM_JOB_ID" "$data" "$kind" "$regime" "$seeds"
printf '%s\n' "${cells[@]}" > "$output/cells.txt"
printf 'CUDA_VISIBLE_DEVICES=%s\n' "${CUDA_VISIBLE_DEVICES:?}"
test -f "$data/manifest.json"

git rev-parse HEAD > "$output/commit.txt"
git status --short > "$output/git_status.txt"
git diff --binary > "$output/working_tree.patch"
cp examples/world_model/{train,models,model_input,dataset}.py "$output/source/"
cp benchmarl/conf/world_model_reference.yaml benchmarl/conf/world_model.yaml "$output/"
cp scripts/slurm/component_scaling.sbatch "$output/"

.venv/bin/python -u scripts/slurm/monitor_gpu_memory.py "$output/gpu_memory.csv" &
gpu_monitor_pid=$!
trap 'kill "$gpu_monitor_pid" 2>/dev/null || true' EXIT

.venv/bin/python -m pytest -q \
    test/test_world_model_reference_profile.py \
    test/test_world_model_reference_context.py \
    | tee "$output/preflight_tests.txt"

pids=()
wait_batch() {
    local status=0
    for pid in "${pids[@]}"; do
        wait "$pid" || status=1
    done
    pids=()
    return "$status"
}

for cell in "${cells[@]}"; do
    name="${cell%%|*}"
    overrides="${cell#*|}"
    for seed in $seeds; do
        run="${name}_seed${seed}"
        printf 'START %s [%s]\n' "$run" "$overrides"
        (
            .venv/bin/python -u -m examples.world_model.train \
                --config-name world_model_reference \
                device=cuda "seed=$seed" "model.kind=$kind" \
                "data.root=$data" "data.regime=$regime" \
                "hydra.run.dir=$output/runs/$run" \
                $overrides > "$output/runs_$run.log" 2>&1
            printf 'DONE %s\n' "$run"
        ) &
        pids+=("$!")
        if ((${#pids[@]} >= max_concurrent)); then
            wait_batch
        fi
    done
done
if ((${#pids[@]})); then
    wait_batch
fi

completed=$(find "$output/runs" -name model.pt | wc -l)
printf '\nCheckpoints written: %s\n' "$completed"
test "$completed" -eq $(( ${#cells[@]} * $(wc -w <<<"$seeds") ))
printf 'Completed component scaling: %s\n' "$output"
