#!/usr/bin/env bash
# E4: re-run the headline Buzz Wire grid at the regulariser weight job 1224
# selected, then re-derive C7 on the result with the repaired evaluator.
#
# Job 1224's width/regularisation ladder is a controlled comparison the project
# already paid for and has never reported. At dim 192 it shows LeWM's default
# sigreg weight of 0.09 costs the relational model 12 points of captured
# intervention effect (0.731x -> 0.612x at 0.009) while leaving `joint`
# unchanged, and lowers latent rollout error for all three models. Every headline
# number in the project is currently measured at the worse setting.
#
# Independent of the goal-source repair running beside it: this is training and
# open-loop counterfactual measurement, and touches no closed-loop objective.
#
# On the MPS slice deliberately. A MIG without MPS time-slices processes; MPS
# co-schedules, so concurrent training workers actually overlap. Three workers on
# 10 GB, the sizing job 1223 ran 144 runs under without incident.
#SBATCH --job-name=sigreg-sel
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=10G
#SBATCH --gres=gpu:a100_2g.10gb_mps:1
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the BenchMARL repository root}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1
workers="${WORLD_MODEL_WORKERS:-3}"
seeds="${WORLD_MODEL_SEEDS:-4100,4101,4102,4103,4104,4105,4106,4107}"
data="${WORLD_MODEL_DATA:-outputs/buzz_wire_1196/data}"
output="${SLURM_SUBMIT_DIR}/outputs/sigreg_selected_${SLURM_JOB_ID}"
mkdir -p "$output"
exec > >(tee -a "$output/batch.log") 2>&1
printf 'SIGReg selected weight: job=%s workers=%s seeds=%s data=%s\n' \
    "$SLURM_JOB_ID" "$workers" "$seeds" "$data"
printf 'CUDA_VISIBLE_DEVICES=%s\n' "${CUDA_VISIBLE_DEVICES:?}"
git rev-parse HEAD > "$output/commit.txt"
git status --short > "$output/git_status.txt"
git diff --binary > "$output/working_tree.patch"
mkdir "$output/source"
cp examples/world_model/*.py "$output/source/"
cp scripts/slurm/sigreg_selected.sbatch "$output/"
cp benchmarl/conf/sweep/world_model_sigreg.yaml "$output/"

.venv/bin/python -u scripts/slurm/monitor_gpu_memory.py "$output/gpu_memory.csv" &
gpu_monitor_pid=$!
trap 'kill "$gpu_monitor_pid" 2>/dev/null || true' EXIT

srun --ntasks=1 .venv/bin/python -m examples.world_model.train \
    --config-name sweep/world_model_sigreg --multirun \
    hydra/launcher=joblib "hydra.launcher.n_jobs=$workers" \
    "data.root=$data" "seed=$seeds" "hydra.sweep.dir=$output/runs"

# The point of the re-run is the intervention response, so measure it here with
# the repaired evaluator rather than leaving another undocumented sweep on disk.
printf '\n=== C7 on the re-run grid ===\n'
srun --ntasks=1 .venv/bin/python -u -m examples.world_model.counterfactual_evaluation \
    "$output/runs" --data "$data" --device cuda \
    2>&1 | tee "$output/c7.txt"

printf '\nPeak GPU memory:\n'
awk -F, 'NR>1 {gsub(/ MiB/,"",$4); if ($4+0>m) m=$4+0} END {print "  " m " MiB"}' \
    "$output/gpu_memory.csv"
printf 'Completed sigreg selected: %s/runs\n' "$output"
