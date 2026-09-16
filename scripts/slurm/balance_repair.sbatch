#!/usr/bin/env bash
# Balance, recollected: the same pipeline as job 1233 on the fixed collector.
#
# Job 1233's heuristic branch is Transport's policy driving Balance's four
# agents -- `collect.py` named `vmas.scenarios.transport` instead of resolving
# the scenario being collected. Recomputed over all 30,824 stored agent-actions,
# Transport's policy reproduces them to 1.46e-9 and Balance's own differs by
# 1.0795. See outputs/balance_1233/DEFECT.md and review_2026-09-16.md section 2.
#
# 426 of that bank's 1,620 anchors come from the wrong policy, and they are in
# the training split, so every model trained on it saw them. This reruns the
# identical design -- same seeds, same budgets, same sweep config, same
# evaluators -- changing only the collector, so job 1233's tables and this job's
# are a like-for-like pair and the question "does the Balance disagreement
# survive correct data?" has a direct answer.
#
# Deliberately NOT changed here: the training loader still reads the random
# correlated/independent branches, so heuristic trajectories contribute anchor
# states rather than action sequences. Mixing competent segments with controlled
# deviations from them is Gate 3 of the review, and it is a data-design change
# that must not be confounded with this repair.
#
# The +43.44 heuristic figure quoted in balance_pipeline.sbatch came from a probe
# whose output was never saved and is not reproducible; the reproducible
# comparison is outputs/review_20260916/evidence.json, which replays Balance's
# own policy from 64 of job 1233's roots for +9.5601 per agent against the
# -15.6391 those roots stored.
#SBATCH --job-name=balance-repair
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
output="${SLURM_SUBMIT_DIR}/outputs/balance_repair_${SLURM_JOB_ID}"
data="$output/data"
mkdir -p "$output"
exec > >(tee -a "$output/batch.log") 2>&1
printf 'Balance repair: job=%s workers=%s output=%s\n' "$SLURM_JOB_ID" "$workers" "$output"
printf 'CUDA_VISIBLE_DEVICES=%s\n' "${CUDA_VISIBLE_DEVICES:?}"
git rev-parse HEAD > "$output/commit.txt"
git status --short > "$output/git_status.txt"
git diff --binary > "$output/working_tree.patch"
cp scripts/slurm/balance_repair.sbatch benchmarl/conf/sweep/balance_baselines.yaml "$output/"
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

srun --ntasks=1 .venv/bin/python -m examples.world_model.collect \
    task=vmas/balance dataset.include_heuristic=true \
    "hydra.run.dir=$data"

# The defect this job exists to repair, checked against the bank it just wrote
# rather than against the code that wrote it. Fails the job before any training.
srun --ntasks=1 .venv/bin/python - "$data" <<'PY'
import json
import sys
from pathlib import Path

import torch
from vmas.scenarios.balance import HeuristicPolicy

data = Path(sys.argv[1])
manifest = json.loads((data / "manifest.json").read_text())
assert manifest["heuristic_policy"] == "vmas.scenarios.balance.HeuristicPolicy", manifest

stored = torch.load(data / "trajectories_heuristic.pt", weights_only=True)
observation, action = stored["observation"], stored["action"]
episodes, steps, agents, features = observation.shape
recomputed = HeuristicPolicy(continuous_action=True).compute_action(
    observation.reshape(-1, features), 1.0
).reshape(episodes, steps, agents, -1)
# Padded steps after an episode ends carry zero actions the policy never saw.
live = stored["valid"]
difference = (action - recomputed).abs()[live]
print(f"{live.sum()} live steps, mean |stored - balance policy| = "
      f"{float(difference.mean()):.3e}, max {float(difference.max()):.3e}", flush=True)
assert float(difference.max()) < 1e-4, "stored heuristic actions are not Balance's policy"
PY

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

# Three baselines x two regimes x eight paired training seeds, identical to 1233.
srun --ntasks=1 .venv/bin/python -m examples.world_model.train \
    --config-name sweep/balance_baselines --multirun \
    hydra/launcher=joblib "hydra.launcher.n_jobs=$workers" \
    "data.root=$data" "hydra.sweep.dir=$output/baselines"

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

# The horizon curve on the corrected bank, so section 4 of
# docs/paper/experiments/09_horizon_rollout.md has a like-for-like replacement.
srun --ntasks=1 .venv/bin/python -m examples.world_model.horizon_rollout \
    "$output/baselines" --data "$data" --blocks 20 --device cuda \
    | tee "$output/horizon.txt"

printf 'Completed Balance repair: %s\n' "$output"
