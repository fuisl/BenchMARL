#!/usr/bin/env bash
# Single-agent gate: can the latent world model imagine ONE agent's own motion?
#
# If an independent (single-agent) latent model cannot roll out and respond to
# its own action correctly, no multi-agent result built on the same pipeline is
# interpretable. VMAS navigation with n_agents=1 and collisions off is a point
# mass whose dynamics are LINEAR in (observation, action): a ridge fit on the
# observation is exact (validation MSE ~1e-15 on the smoke bank). Any failure
# here is therefore the latent pipeline's, not the data's.
#
# Bank: 512 root episodes, 200 steps, anchors every 25 steps, 50-step (10-block)
# snippets. The self-intervention flips the agent's own x action from step 15
# (block 3), so both branches share three real context frames.
#
# Cells (independent kind, lewm_reference, 3 seeds each):
#   base       lambda 0.09, first window, 100 epochs       (the T-A2 recipe)
#   win        + random training window (LeWM samples windows; ours used one)
#   lam0009    lambda 0.009
#   lam0009win lambda 0.009 + random window
#   ep300      300 epochs                                  (optimization budget)
#   data025    1/4 of train roots, 400 epochs              (fixed updates)
#
# Pass rule is registered in examples/world_model/self_rollout.py (G1, G2).
#SBATCH --job-name=single-gate
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=08:00:00
#SBATCH --output=slurm-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the BenchMARL repository root}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="${SLURM_SUBMIT_DIR}${PYTHONPATH:+:$PYTHONPATH}"

seeds="${GATE_SEEDS:-4100 4101 4102}"
max_concurrent="${GATE_MAX_CONCURRENT:-9}"
output="${SLURM_SUBMIT_DIR}/outputs/single_agent_gate_${SLURM_JOB_ID}"
data="${GATE_DATA:-$output/data}"
mkdir -p "$output/source" "$output/runs"
exec > >(tee -a "$output/batch.log") 2>&1

printf 'Single-agent gate: job=%s data=%s seeds=[%s]\n' "$SLURM_JOB_ID" "$data" "$seeds"
printf 'CUDA_VISIBLE_DEVICES=%s\n' "${CUDA_VISIBLE_DEVICES:?}"
git rev-parse HEAD > "$output/commit.txt"
git status --short > "$output/git_status.txt"
git diff --binary > "$output/working_tree.patch"
cp examples/world_model/{collect,train,models,dataset,self_rollout}.py "$output/source/"
cp scripts/slurm/single_agent_gate.sbatch "$output/"

.venv/bin/python -u scripts/slurm/monitor_gpu_memory.py "$output/gpu_memory.csv" &
gpu_monitor_pid=$!
trap 'kill "$gpu_monitor_pid" 2>/dev/null || true' EXIT

.venv/bin/python -m pytest -q \
    test/test_world_model_reference_profile.py \
    test/test_world_model_reference_context.py \
    test/test_world_model_collection.py \
    | tee "$output/preflight_tests.txt"

if [[ ! -f "$data/manifest.json" ]]; then
    printf '\n=== collect ===\n'
    .venv/bin/python -u -m examples.world_model.collect \
        task=vmas/navigation task.n_agents=1 task.collisions=False task.max_steps=200 \
        dataset.episodes=512 dataset.sequence_steps=50 dataset.anchor_stride=25 \
        +dataset.intervention_start_step=15 dataset.branch_batch_size=1024 \
        experiment.sampling_device=cuda "experiment.loggers=[]" \
        "hydra.run.dir=$data" > "$output/collect.log" 2>&1
fi
test -f "$data/manifest.json"

# name|hydra overrides
cells=(
    "base|"
    "win|+train.reference_window=random"
    "lam0009|train.sigreg_weight=0.009"
    "lam0009win|train.sigreg_weight=0.009 +train.reference_window=random"
    "ep300|train.dynamics_epochs=300"
    "data025|data.train_fraction=0.25 train.dynamics_epochs=400"
)
printf '%s\n' "${cells[@]}" > "$output/cells.txt"

pids=()
wait_batch() {
    local status=0
    for pid in "${pids[@]}"; do wait "$pid" || status=1; done
    pids=()
    return "$status"
}
run_one() {
    local cell="$1" seed="$2" overrides="$3"
    local name="${cell}_seed${seed}"
    printf 'START %s\n' "$name"
    .venv/bin/python -u -m examples.world_model.train \
        --config-name world_model_reference \
        device=cuda "seed=$seed" model.kind=independent \
        "data.root=$data" data.regime=independent \
        "loggers=[csv]" "hydra.run.dir=$output/runs/$name" \
        $overrides > "$output/runs_$name.log" 2>&1
    printf 'DONE %s\n' "$name"
}
for entry in "${cells[@]}"; do
    cell="${entry%%|*}"
    overrides="${entry#*|}"
    for seed in $seeds; do
        run_one "$cell" "$seed" "$overrides" &
        pids+=("$!")
        if ((${#pids[@]} >= max_concurrent)); then wait_batch; fi
    done
done
if ((${#pids[@]})); then wait_batch; fi
printf 'Checkpoints: %s\n' "$(find "$output/runs" -name model.pt | wc -l)"

printf '\n=== evaluate ===\n'
for probe in linear mlp; do
    .venv/bin/python -u -m examples.world_model.self_rollout "$output/runs" \
        --data "$data" --output "$output/eval_$probe" --probe "$probe" \
        --device cuda > "$output/eval_$probe.log" 2>&1 &
    pids+=("$!")
done
wait_batch
test -s "$output/eval_linear/summary.json"
test -s "$output/eval_mlp/summary.json"
printf 'Completed single-agent gate: %s\n' "$output"
