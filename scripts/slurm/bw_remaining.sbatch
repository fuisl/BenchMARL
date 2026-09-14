#!/usr/bin/env bash
# Stages 2 and 3 of the Buzz Wire work, on a MIG slice rather than the full A100.
#
# Stage 1 finished in job 1200 and took 49 seconds of GPU time once the
# evaluators were actually put on CUDA, so a whole A100 is not warranted for
# this and the shared node should keep it.
#
# Job 1200 died collecting the held-out bank: the collector's own integrity
# check found that replaying the reference actions from the stored snapshots did
# not reproduce the stored trajectory bit-exactly. That check is doing its job --
# it refuses to emit a bank whose counterfactual branch cannot be trusted -- and
# it is not a seed quirk. Buzz Wire runs 15 physics substeps with rigid joints,
# and the failure appears at the full 25-step snippet length while a 5-step smoke
# passes on the same seeds, which is the signature of divergence amplifying with
# rollout length. The 1196 development bank did pass this check bit-exactly over
# 256 anchors, so the C7 result already recorded stands on verified data.
#
# The held-out confirmation therefore tries several seed sets and keeps the first
# bank that passes the integrity check. Rejecting banks that fail a *validity*
# check cannot bias the C7 outcome, because nothing about the result is consulted
# while choosing -- but the number of rejected seeds is printed so the limitation
# is recorded rather than hidden.
#SBATCH --job-name=bw-remaining
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --gres=gpu:a100_3g.20gb:1
#SBATCH --time=03:00:00
#SBATCH --output=slurm-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the BenchMARL repository root}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
workers="${WORLD_MODEL_WORKERS:-5}"
dev_data="outputs/buzz_wire_1196/data"
output="${SLURM_SUBMIT_DIR}/outputs/bw_remaining_${SLURM_JOB_ID}"
mkdir -p "$output"
exec > >(tee -a "$output/batch.log") 2>&1
printf 'Buzz Wire remaining: job=%s workers=%s\n' "$SLURM_JOB_ID" "$workers"
printf 'CUDA_VISIBLE_DEVICES=%s\n' "${CUDA_VISIBLE_DEVICES:?}"

.venv/bin/python -u scripts/slurm/monitor_gpu_memory.py "$output/gpu_memory.csv" &
gpu_monitor_pid=$!
# Only suppress an already-exited monitoring process during job cleanup.
trap 'kill "$gpu_monitor_pid" 2>/dev/null || true' EXIT

########## Stage 2: untouched held-out confirmation ##########
holdout="$output/holdout"
rejected=0
collected=""
for base in 5100 6100 7100 8100 9100; do
    printf '\n=== held-out collection attempt, seed base %s ===\n' "$base"
    if srun --ntasks=1 .venv/bin/python -m examples.world_model.collect \
        task=vmas/buzz_wire dataset.include_heuristic=false \
        "dataset.state_seed=$base" "dataset.split_seed=$((base + 1))" \
        "dataset.action_seed=$((base + 2))" "dataset.branch_seed=$((base + 3))" \
        "hydra.run.dir=$holdout/data"; then
        collected="$base"
        break
    fi
    rejected=$((rejected + 1))
    rm -rf "$holdout/data"
done
printf '\nheld-out bank: seed base %s accepted, %s rejected by the replay check\n' \
    "${collected:-none}" "$rejected"
test -n "$collected" || { echo "No seed produced a bit-exact bank" >&2; exit 1; }

srun --ntasks=1 .venv/bin/python -m examples.world_model.train \
    --config-name sweep/world_model_baselines --multirun \
    hydra/launcher=joblib "hydra.launcher.n_jobs=$workers" \
    "data.root=$holdout/data" "data.regime=correlated,independent" \
    "seed=5200,5201,5202,5203,5204,5205,5206,5207" \
    wandb.group=m4-buzz-wire-holdout \
    "hydra.sweep.dir=$holdout/baselines"

# Measured once. Nothing is tuned against this output.
srun --ntasks=1 .venv/bin/python -m examples.world_model.counterfactual_evaluation \
    "$holdout/baselines" --data "$holdout/data" --device cuda \
    | tee "$output/holdout_counterfactual.txt"

########## Stage 3: does fixing the SIGReg regime amplify the signal? ##########
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
            "wandb.group=m4-sigreg-$point" \
            "hydra.sweep.dir=$output/sigreg/$point"
        srun --ntasks=1 .venv/bin/python -m examples.world_model.counterfactual_evaluation \
            "$output/sigreg/$point" --data "$dev_data" --device cuda \
            | tee "$output/sigreg_${point}.txt"
    done
done

printf 'Completed Buzz Wire remaining: %s\n' "$output"
