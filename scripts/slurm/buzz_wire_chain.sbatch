#!/usr/bin/env bash
# Job A -- carry the evidence chain forward on Buzz Wire.
#
# C7 is positive there (relational captures 26% of the cross-agent effect, 8/8
# seeds, and beats joint on 7/8). The next link, plan ranking, was measured on
# Transport and came back flat -- but Transport's cost landscape is degenerate,
# with only 7/24 states having any cost variation at all. On Buzz Wire a probe
# gives 24/24 rankable states with median per-state cost std 7.39, so the
# measurement is well posed for the first time.
#
# Runs all three evaluators over the job-1196 checkpoints: counterfactual
# response (C7, re-reported here for the record), plan ranking (C8) and the
# stratified diagnostic. No training.
#SBATCH --job-name=bw-chain
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:a100_3g.20gb:1
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the BenchMARL repository root}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
runs="${BW_RUNS:-outputs/buzz_wire_1196/baselines}"
data="${BW_DATA:-outputs/buzz_wire_1196/data}"
output="${SLURM_SUBMIT_DIR}/outputs/bw_chain_${SLURM_JOB_ID}"
mkdir -p "$output"
exec > >(tee -a "$output/batch.log") 2>&1
printf 'Buzz Wire chain: job=%s runs=%s data=%s\n' "$SLURM_JOB_ID" "$runs" "$data"

srun --ntasks=1 .venv/bin/python -m examples.world_model.counterfactual_evaluation \
    "$runs" --data "$data" | tee "$output/counterfactual.txt"

# 128 states x 64 candidates; simulator truth is cached inside the output dir so
# the ranking can be re-analysed without re-simulating.
srun --ntasks=1 .venv/bin/python -m examples.world_model.plan_ranking \
    "$runs" --data "$data" --states 128 --candidates 64 \
    --cache "$output/plan_ranking_truth.pt" | tee "$output/plan_ranking.txt"

srun --ntasks=1 .venv/bin/python -m examples.world_model.stratified_evaluation \
    "$runs" --data "$data" | tee "$output/stratified.txt"

printf 'Completed Buzz Wire chain: %s\n' "$output"
