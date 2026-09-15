#!/usr/bin/env bash
# All remaining Buzz Wire work in one allocation on the full A100.
#
# Replaces the cancelled 1197/1198/1199 chain. Two things were wrong with that
# submission, and only one of them was about resources:
#
#   * the evaluators default to --device cpu and the sbatch never overrode it,
#     so the chain job held a 20 GB MIG slice completely idle and ran
#     single-threaded. They are passed --device cuda here. CPU and CUDA outputs
#     were verified identical before this was submitted, so the change is a
#     speed fix and not a silent change of results;
#   * a dependency chain of three jobs serialises work that fits in one
#     allocation, and leaves the later stages stranded if an earlier one is
#     cancelled -- which is exactly what happened.
#
# Stage 1 reuses the simulator truth already computed by the cancelled 1197
# (outputs/bw_cache/plan_ranking_truth.pt), so that work is not repeated.
#SBATCH --job-name=bw-full
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=72G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=06:00:00
#SBATCH --output=slurm-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the BenchMARL repository root}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
workers="${WORLD_MODEL_WORKERS:-10}"
dev_runs="outputs/buzz_wire_1196/baselines"
dev_data="outputs/buzz_wire_1196/data"
output="${SLURM_SUBMIT_DIR}/outputs/bw_full_${SLURM_JOB_ID}"
mkdir -p "$output"
exec > >(tee -a "$output/batch.log") 2>&1
printf 'Buzz Wire full: job=%s workers=%s\n' "$SLURM_JOB_ID" "$workers"
printf 'CUDA_VISIBLE_DEVICES=%s\n' "${CUDA_VISIBLE_DEVICES:?}"

.venv/bin/python -u scripts/slurm/monitor_gpu_memory.py "$output/gpu_memory.csv" &
gpu_monitor_pid=$!
# Only suppress an already-exited monitoring process during job cleanup.
trap 'kill "$gpu_monitor_pid" 2>/dev/null || true' EXIT

########## Stage 1: carry the chain forward on the development bank ##########
cp -n outputs/bw_cache/plan_ranking_truth.pt "$output/plan_ranking_truth.pt" || true

srun --ntasks=1 .venv/bin/python -m examples.world_model.counterfactual_evaluation \
    "$dev_runs" --data "$dev_data" --device cuda | tee "$output/counterfactual.txt"

srun --ntasks=1 .venv/bin/python -m examples.world_model.plan_ranking \
    "$dev_runs" --data "$dev_data" --device cuda --states 128 --candidates 64 \
    --cache "$output/plan_ranking_truth.pt" | tee "$output/plan_ranking.txt"

srun --ntasks=1 .venv/bin/python -m examples.world_model.stratified_evaluation \
    "$dev_runs" --data "$dev_data" --device cuda | tee "$output/stratified.txt"

########## Stage 2: untouched held-out confirmation ##########
# Fresh state, split, action and branch seeds; training seeds disjoint from the
# development runs. Measured exactly once -- nothing is tuned against it.
holdout="$output/holdout"
srun --ntasks=1 .venv/bin/python -m examples.world_model.collect \
    task=vmas/buzz_wire dataset.include_heuristic=false \
    dataset.state_seed=5100 dataset.split_seed=5101 \
    dataset.action_seed=5102 dataset.branch_seed=5103 \
    "hydra.run.dir=$holdout/data"

srun --ntasks=1 .venv/bin/python -m examples.world_model.train \
    --config-name sweep/world_model_baselines --multirun \
    hydra/launcher=joblib "hydra.launcher.n_jobs=$workers" \
    "data.root=$holdout/data" "data.regime=correlated,independent" \
    "seed=5200,5201,5202,5203,5204,5205,5206,5207" \
    "hydra.sweep.dir=$holdout/baselines"

srun --ntasks=1 .venv/bin/python -m examples.world_model.counterfactual_evaluation \
    "$holdout/baselines" --data "$holdout/data" --device cuda \
    | tee "$output/holdout_counterfactual.txt"

########## Stage 3: does fixing the SIGReg regime amplify the signal? ##########
# The anti-collapse term sits near a floor set by rank deficiency rather than
# width, running roughly 330:1 against prediction. Latent width and lambda are
# applied identically to all three baselines, and each point is scored by the C7
# response rather than prediction error -- the question is whether a better
# conditioned representation captures MORE of the cross-agent effect. Four seeds,
# since this is a search; the winning point needs eight before it is reported.
for dim in 16 48 192; do
    for weight in 0.09 0.009; do
        point="dim${dim}_lambda${weight}"
        printf '\n=== sigreg %s ===\n' "$point"
        srun --ntasks=1 .venv/bin/python -m examples.world_model.train \
            --config-name sweep/world_model_baselines --multirun \
            hydra/launcher=joblib "hydra.launcher.n_jobs=$workers" \
            "data.root=$dev_data" "data.regime=correlated,independent" \
            "seed=4100,4101,4102,4103" \
            "model.dim=$dim" "train.sigreg_weight=$weight" \
            "hydra.sweep.dir=$output/sigreg/$point"
        srun --ntasks=1 .venv/bin/python -m examples.world_model.counterfactual_evaluation \
            "$output/sigreg/$point" --data "$dev_data" --device cuda \
            | tee "$output/sigreg_${point}.txt"
    done
done

printf 'Completed Buzz Wire full: %s\n' "$output"
