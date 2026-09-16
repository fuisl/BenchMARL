#!/usr/bin/env bash
# Balance: bank, baseline grid and open-loop evaluation, matching the design used
# for Transport (1194), Buzz Wire (1196) and Wheel (1205) so the four are
# comparable.
#
# Why a fourth task. Every control result so far is bounded by a property of the
# task rather than of the world model. Buzz Wire's agents never observe the ball
# or the wire, so an observation-space goal can neither express the task nor see
# the failure mode: jobs 1229/1230/1231 measured a planner with TRUE dynamics,
# chasing a real task goal, colliding in 85% of episodes against random's 65%.
# Transport is uncontrollable from its anchors -- its own heuristic closes 0.032
# of the starting 0.9006 across a full episode, and doing nothing loses 0.0012.
# Neither task can separate "the model is wrong" from "the task cannot be done
# here", which is what the paper needs to claim anything about control.
#
# Balance has neither defect, and both were checked before scheduling rather than
# after:
#   * the package position, its offset to the goal, its velocity and the line's
#     state are all in the agent observation, so the task IS expressible as an
#     observation-space goal;
#   * the shipped heuristic scores +43.44 against random's -22.09 and
#     zero-action's -5.75 over 64 episodes -- the widest competent-vs-random
#     margin in the suite, so goals can be drawn from a controller that works.
# Four agents jointly balancing one package keeps the physical coupling that
# makes the relational question worth asking.
#
# Unlike Wheel, heuristic trajectories ARE collected. That is the whole point:
# today's failures all trace back to having no competent behaviour in the bank.
#SBATCH --job-name=balance-pipeline
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --gres=gpu:a100_3g.20gb:1
#SBATCH --time=06:00:00
#SBATCH --output=slurm-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the BenchMARL repository root}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1
workers="${WORLD_MODEL_WORKERS:-5}"
output="${SLURM_SUBMIT_DIR}/outputs/balance_${SLURM_JOB_ID}"
data="$output/data"
mkdir -p "$output"
exec > >(tee -a "$output/batch.log") 2>&1
printf 'Balance pipeline: job=%s workers=%s output=%s\n' "$SLURM_JOB_ID" "$workers" "$output"
printf 'CUDA_VISIBLE_DEVICES=%s\n' "${CUDA_VISIBLE_DEVICES:?}"
git rev-parse HEAD > "$output/commit.txt"
git status --short > "$output/git_status.txt"
git diff --binary > "$output/working_tree.patch"
cp scripts/slurm/balance_pipeline.sbatch benchmarl/conf/sweep/balance_baselines.yaml "$output/"
mkdir "$output/source"
cp examples/world_model/*.py "$output/source/"

.venv/bin/python -u scripts/slurm/monitor_gpu_memory.py "$output/gpu_memory.csv" &
gpu_monitor_pid=$!
# Only suppress an already-exited monitoring process during job cleanup.
trap 'kill "$gpu_monitor_pid" 2>/dev/null || true' EXIT

# Validate the real allocation without creating a wandb debug run.
srun --ntasks=1 .venv/bin/python - <<'PY'
import torch

assert torch.cuda.is_available()
assert torch.cuda.device_count() == 1
device = torch.cuda.get_device_properties(0)
assert 18 * 2**30 < device.total_memory < 21 * 2**30, device
print("Allocated device:", device, flush=True)
PY

# Same root episodes, split/action seeds and budgets as the prior three tasks,
# and the native Balance config, so the only difference from Wheel is the task.
srun --ntasks=1 .venv/bin/python -m examples.world_model.collect \
    task=vmas/balance dataset.include_heuristic=true \
    "hydra.run.dir=$data"

srun --ntasks=1 .venv/bin/python - "$data" <<'PY'
import sys
from examples.world_model.dataset import OfflineSequences

for regime in ("independent", "correlated"):
    for split in ("train", "validation", "test"):
        dataset = OfflineSequences(sys.argv[1], regime, split, action_block=5)
        print(regime, split, len(dataset), "validated snippets", flush=True)
PY

# Verify concurrent GPU training and checkpoint reload before the real sweep.
srun --ntasks=1 .venv/bin/python -m examples.world_model.train \
    --config-name sweep/balance_baselines --multirun \
    hydra/launcher=joblib "hydra.launcher.n_jobs=$workers" \
    "data.root=$data" seed=4100 'loggers=[csv]' \
    train.dynamics_epochs=2 train.readout_epochs=1 \
    "hydra.sweep.dir=$output/smoke"

# Three baselines x two regimes x eight paired training seeds; defaults retain
# 100 dynamics + 50 readout epochs, matched capacity, and CSV + online wandb.
srun --ntasks=1 .venv/bin/python -m examples.world_model.train \
    --config-name sweep/balance_baselines --multirun \
    hydra/launcher=joblib "hydra.launcher.n_jobs=$workers" \
    "data.root=$data" "hydra.sweep.dir=$output/baselines"

# Evaluators intentionally run on CUDA. Full-state labels include Balance torque.
mkdir -p "$output/evaluation_source"
cp examples/world_model/{counterfactual_evaluation,plan_ranking,stratified_evaluation}.py \
    "$output/evaluation_source/"
git diff --binary > "$output/evaluation_working_tree.patch"
srun --ntasks=1 .venv/bin/python -m examples.world_model.counterfactual_evaluation \
    "$output/baselines" --data "$data" --device cuda --full-state \
    | tee "$output/counterfactual.txt"
srun --ntasks=1 .venv/bin/python -m examples.world_model.plan_ranking \
    "$output/baselines" --data "$data" --device cuda --states 128 --candidates 64 \
    --seed 5100 --cache "$output/plan_ranking_truth.pt" \
    | tee "$output/plan_ranking.txt"
srun --ntasks=1 .venv/bin/python -m examples.world_model.stratified_evaluation \
    "$output/baselines" --data "$data" --device cuda --full-state \
    --output "$output/stratified_scores.json" | tee "$output/stratified.txt"

printf 'Completed Balance pipeline: %s\n' "$output"
