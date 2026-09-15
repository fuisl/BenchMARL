#!/usr/bin/env bash
# Job B -- does fixing the SIGReg regime amplify the interaction signal?
#
# The anti-collapse term currently sits near a floor it cannot reach. SIGReg
# tests random projections against N(0, I), so it penalises rank deficiency as
# well as non-Gaussian shape; with effective rank ~32 in a 192-dim latent most
# directions read as a point mass, and at convergence the weighted term runs
# roughly 330:1 against prediction. Measured floors for rank-9 data are 74-80 for
# K >= 16 against ~1.0 for a full-rank isotropic population.
#
# Two knobs, both already in the config and both applied identically to all three
# baselines so the comparison stays controlled:
#   * model.dim -- a latent the encoder can actually fill, since the floor is set
#     by rank deficiency rather than width;
#   * train.sigreg_weight -- LeJEPA's lambda was calibrated where the statistic
#     approaches its O(1) null, not where it plateaus near 79.
#
# Each point is trained and then scored with the C7 counterfactual response, so
# the question answered is "does a better-conditioned representation capture MORE
# of the cross-agent effect", not merely "does prediction error fall".
#
# Four seeds per point: this is a hyperparameter search, not a headline claim.
# The winning point should be re-run at eight seeds before anything is reported.
#SBATCH --job-name=sigreg-regime
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --gres=gpu:a100_3g.20gb:1
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the BenchMARL repository root}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
workers="${WORLD_MODEL_WORKERS:-5}"
seeds="${WORLD_MODEL_SEEDS:-4100,4101,4102,4103}"
data="${BW_DATA:-outputs/buzz_wire_1196/data}"
output="${SLURM_SUBMIT_DIR}/outputs/sigreg_regime_${SLURM_JOB_ID}"
mkdir -p "$output"
exec > >(tee -a "$output/batch.log") 2>&1
printf 'SIGReg regime sweep: job=%s data=%s seeds=%s\n' "$SLURM_JOB_ID" "$data" "$seeds"
printf 'CUDA_VISIBLE_DEVICES=%s\n' "${CUDA_VISIBLE_DEVICES:?}"

.venv/bin/python -u scripts/slurm/monitor_gpu_memory.py "$output/gpu_memory.csv" &
gpu_monitor_pid=$!
# Only suppress an already-exited monitoring process during job cleanup.
trap 'kill "$gpu_monitor_pid" 2>/dev/null || true' EXIT

# Separate multirun per point, so each has its own directory that the evaluator
# can key by (regime, kind, seed) without collisions.
for dim in 16 48 192; do
    for weight in 0.09 0.009; do
        point="dim${dim}_lambda${weight}"
        printf '\n=== %s ===\n' "$point"
        srun --ntasks=1 .venv/bin/python -m examples.world_model.train \
            --config-name sweep/world_model_baselines --multirun \
            hydra/launcher=joblib "hydra.launcher.n_jobs=$workers" \
            "data.root=$data" "data.regime=correlated,independent" \
            "seed=$seeds" "model.dim=$dim" "train.sigreg_weight=$weight" \
            "hydra.sweep.dir=$output/$point"
        srun --ntasks=1 .venv/bin/python -m examples.world_model.counterfactual_evaluation \
            "$output/$point" --data "$data" | tee "$output/${point}_counterfactual.txt"
    done
done

printf 'Completed SIGReg regime sweep: %s\n' "$output"
