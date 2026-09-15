#!/usr/bin/env bash
# Stage 2 (audit 2026-09-15, section 9): isolate information from architecture.
#
# The question this answers. Buzz Wire agents observe their own position,
# velocity and offset to the goal -- six numbers. The task is defined on a ball
# jointed to both agents, whose displacement drives the reward, and no agent
# observes it. Every learned result on this task is therefore ambiguous between
# "the model is not good enough" and "the input does not contain the answer".
# The audit calls this an information concern and explicitly not a proof of
# irreducible failure: joint observations and history may infer the hidden
# state, and its geometric check only ruled out the ball sitting exactly at the
# agents' midpoint (mean discrepancy 0.0408 over 908 anchors).
#
# Three inputs separate the cases. `observation` is what the agents see;
# `history` adds the last three really-observed frames, so hidden state can be
# inferred from motion; `physical` appends the recorded entity states, the upper
# bound where the hidden variable is simply given. If `physical` fails too, the
# problem is the objective, the actions or the optimization, and no amount of
# input will fix it. If it wins, observation sufficiency is the thing to pursue.
#
# The whole baseline grid is retrained rather than reusing job 1196's
# `observation` arm. Those readouts were fitted under the mask that discarded
# 79% of Buzz Wire's terminations, so they are not a like-for-like control for
# models trained on the repaired contract.
#
# Model kind stays in the sweep because whether relational structure still helps
# once the information is sufficient is the open question behind it.
#
# On the MPS slice deliberately. A MIG without MPS time-slices processes, which
# is what made eight closed-loop shards no faster than two; MPS co-schedules, so
# concurrent training workers actually overlap here.
#SBATCH --job-name=state-input
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=10G
#SBATCH --gres=gpu:a100_2g.10gb_mps:1
#SBATCH --time=06:00:00
#SBATCH --output=slurm-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the BenchMARL repository root}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1
# Three concurrent workers on a 10 GB slice. Job 1218 filled a 20 GB slice to
# 95% and lost three shards to OOM; this leaves real headroom instead.
workers="${WORLD_MODEL_WORKERS:-3}"
seeds="${WORLD_MODEL_SEEDS:-4100,4101,4102,4103,4104,4105,4106,4107}"
data="${WORLD_MODEL_DATA:-outputs/buzz_wire_1196/data}"
output="${SLURM_SUBMIT_DIR}/outputs/state_input_${SLURM_JOB_ID}"
mkdir -p "$output"
exec > >(tee -a "$output/batch.log") 2>&1
printf 'State-input isolation: job=%s workers=%s seeds=%s data=%s\n' \
    "$SLURM_JOB_ID" "$workers" "$seeds" "$data"
printf 'CUDA_VISIBLE_DEVICES=%s\n' "${CUDA_VISIBLE_DEVICES:?}"
git rev-parse HEAD > "$output/commit.txt"
git status --short > "$output/git_status.txt"
git diff --binary > "$output/working_tree.patch"
mkdir "$output/source"
cp examples/world_model/*.py "$output/source/"
cp scripts/slurm/state_input_isolation.sbatch "$output/"
cp benchmarl/conf/sweep/world_model_state_input.yaml "$output/"

.venv/bin/python -u scripts/slurm/monitor_gpu_memory.py "$output/gpu_memory.csv" &
gpu_monitor_pid=$!
trap 'kill "$gpu_monitor_pid" 2>/dev/null || true' EXIT

# The input transform must not break the dynamics target: the model predicts
# latent[:, 1:] from latent[:, :-1], so an augmentation that misaligned
# observation and next_observation would train against the wrong frame while
# every older test still passed. That equality is a test, so check it here.
srun --ntasks=1 .venv/bin/python -m pytest test/test_world_model_dataset.py -q \
    | tee "$output/dataset_tests.txt"

# What each condition actually feeds the encoder, recorded beside the results.
srun --ntasks=1 .venv/bin/python - "$data" <<'PY' | tee "$output/input_widths.txt"
import sys
from examples.world_model.dataset import OfflineSequences, STATE_INPUTS

for state_input in STATE_INPUTS:
    data = OfflineSequences(
        sys.argv[1], "correlated", "train", action_block=5, state_input=state_input
    )
    print(f"{state_input:12s} obs_dim={data[0]['observation'].shape[-1]:3d} "
          f"snippets={len(data)}", flush=True)
PY

srun --ntasks=1 .venv/bin/python -m examples.world_model.train \
    --config-name sweep/world_model_state_input --multirun \
    hydra/launcher=joblib "hydra.launcher.n_jobs=$workers" \
    "data.root=$data" \
    "data.regime=correlated,independent" \
    "seed=$seeds" \
    "hydra.sweep.dir=$output/runs"

printf '\nPeak GPU memory:\n'
awk -F, 'NR>1 {gsub(/ MiB/,"",$4); if ($4+0>m) m=$4+0} END {print "  " m " MiB"}' \
    "$output/gpu_memory.csv"
printf 'Completed state-input isolation: %s/runs\n' "$output"
