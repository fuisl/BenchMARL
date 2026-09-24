#!/usr/bin/env bash
# T-A2 stage 1: train the three conditioning hypotheses on the repaired profile.
#
#   H0 independent   f(z_i, a_i)              -- cross blocks are exactly zero
#   H1 joint         f(z_i, Z, A)             -- same information as H2
#   H2 relational    f(z_i, a_i, sum_j phi())  -- differs from H1 only in bias
#
# H1 and H2 see the SAME information. Whatever separates them is inductive bias,
# not access, so this sweep is not a test of "does interaction matter" -- T-A1
# already measures that on the simulator -- but of which factorization recovers
# a measurable effect from a fixed dataset.
#
# Every run uses `lewm_reference`, the Audit Gate A0 profile whose scheduler,
# action normalization and temporal context are pinned to LeWM 8edfeb33 and
# verified against vendored sources. The historical `legacy_compact` results are
# NOT comparable to these and are not being reproduced here.
#
# Conditioner capacity is solved to a shared budget, so the three kinds match to
# within 0.07% of conditioner parameters and 0.006% of dynamics parameters
# (measured on this bank: 1,082,922 / 1,083,180 / 1,082,400). A difference
# between them cannot be attributed to one model simply being larger.
#
# Both action regimes run because M1 Row 2 is the coverage contrast: `correlated`
# restricts the joint-action support the planner will later leave, `independent`
# covers it. The held-out region is recorded in the bank manifest as
# `normalized a0_x*a1_x < 0`.
#
# This stage produces checkpoints only. No counterfactual metric is computed
# here and no model is selected, so the T-A1 floor cannot leak into training.
#
# SIZING, measured. The A0 single-seed sanity run (job 1469) trained 100 epochs
# in 68 s on one 3g.20gb slice. 3 kinds x 8 seeds x 2 regimes = 48 runs; at six
# concurrent fits that is well under an hour.
#SBATCH --job-name=ta2-baselines
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
export PYTHONPATH="${SLURM_SUBMIT_DIR}${PYTHONPATH:+:$PYTHONPATH}"

data="${TA2_DATA:-outputs/buzz_wire_1196/data}"
seeds="${TA2_SEEDS:-4100 4101 4102 4103 4104 4105 4106 4107}"
kinds="${TA2_KINDS:-independent joint relational}"
regimes="${TA2_REGIMES:-correlated independent}"
max_concurrent="${TA2_MAX_CONCURRENT:-6}"
# Extra Hydra overrides shared by every run, e.g. "train.sigreg_weight=0.009",
# and a tag naming the variant in the output directory. Empty reproduces 1498.
overrides="${TA2_OVERRIDES:-}"
tag="${TA2_TAG:+${TA2_TAG}_}"
output="${SLURM_SUBMIT_DIR}/outputs/ta2_reference_baselines_${tag}${SLURM_JOB_ID}"
mkdir -p "$output/source"
exec > >(tee -a "$output/batch.log") 2>&1

printf 'T-A2 stage 1: job=%s data=%s concurrent=%s\n' \
    "$SLURM_JOB_ID" "$data" "$max_concurrent"
printf 'kinds=[%s] regimes=[%s] seeds=[%s]\n' "$kinds" "$regimes" "$seeds"
printf 'overrides=[%s]\n' "$overrides"
printf 'CUDA_VISIBLE_DEVICES=%s\n' "${CUDA_VISIBLE_DEVICES:?}"
test -f "$data/manifest.json"

git rev-parse HEAD > "$output/commit.txt"
git status --short > "$output/git_status.txt"
git diff --binary > "$output/working_tree.patch"
cp examples/world_model/{train,models,model_input,dataset}.py "$output/source/"
cp test/_lewm_pinned_8edfeb33.py test/_stable_pretraining_pinned_v017.py "$output/source/"
cp benchmarl/conf/world_model_reference.yaml "$output/"
cp scripts/slurm/ta2_reference_baselines.sbatch "$output/"

.venv/bin/python -u scripts/slurm/monitor_gpu_memory.py "$output/gpu_memory.csv" &
gpu_monitor_pid=$!
trap 'kill "$gpu_monitor_pid" 2>/dev/null || true' EXIT

# A reference claim that no longer matches the vendored LeWM and
# stable-pretraining sources must not produce checkpoints.
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

run_one() {
    local kind="$1" regime="$2" seed="$3"
    local name="${kind}_${regime}_seed${seed}"
    printf 'START %s\n' "$name"
    .venv/bin/python -u -m examples.world_model.train \
        --config-name world_model_reference \
        device=cuda "seed=$seed" \
        "model.kind=$kind" \
        "data.root=$data" "data.regime=$regime" \
        "hydra.run.dir=$output/runs/$name" \
        $overrides \
        > "$output/runs_$name.log" 2>&1
    printf 'DONE %s\n' "$name"
}

mkdir -p "$output/runs"
for kind in $kinds; do
    for regime in $regimes; do
        for seed in $seeds; do
            run_one "$kind" "$regime" "$seed" &
            pids+=("$!")
            if ((${#pids[@]} >= max_concurrent)); then
                wait_batch
            fi
        done
    done
done
if ((${#pids[@]})); then
    wait_batch
fi

completed=$(find "$output/runs" -name model.pt | wc -l)
printf '\nCheckpoints written: %s\n' "$completed"
test "$completed" -gt 0
printf 'Completed T-A2 stage 1: %s\n' "$output"
