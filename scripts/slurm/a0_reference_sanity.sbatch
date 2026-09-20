#!/usr/bin/env bash
# Audit Gate A0 step 4: one-seed sanity run of the repaired lewm_reference
# profile. This is not a comparison and not a sweep. It asks only whether the
# reference implementation trains sensibly in practice now that the action
# interface, scheduler cadence and temporal context are reference-matched.
# The legacy_compact vs lewm_reference comparison is a separately registered
# step and is deliberately NOT run here.
#SBATCH --job-name=a0-reference-sanity
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
export PYTHONUNBUFFERED=1
export PYTHONPATH="${SLURM_SUBMIT_DIR}${PYTHONPATH:+:$PYTHONPATH}"

data_root="${A0_DATA:-outputs/buzz_wire_1196/data}"
regime="${A0_REGIME:-correlated}"
seed="${A0_SEED:-4100}"
output="${SLURM_SUBMIT_DIR}/outputs/a0_reference_sanity_${SLURM_JOB_ID}"
mkdir -p "$output/source"
exec > >(tee -a "$output/batch.log") 2>&1

printf 'A0 sanity: job=%s data=%s regime=%s seed=%s\n' \
    "$SLURM_JOB_ID" "$data_root" "$regime" "$seed"
printf 'CUDA_VISIBLE_DEVICES=%s\n' "${CUDA_VISIBLE_DEVICES:?}"
test -f "$data_root/manifest.json"

git rev-parse HEAD > "$output/commit.txt"
git status --short > "$output/git_status.txt"
git diff --binary > "$output/working_tree.patch"
cp examples/world_model/{train,models,model_input,dataset}.py "$output/source/"
cp test/_lewm_pinned_8edfeb33.py test/_stable_pretraining_pinned_v017.py "$output/source/"
cp benchmarl/conf/world_model_reference.yaml "$output/"
cp scripts/slurm/a0_reference_sanity.sbatch "$output/"

.venv/bin/python -u scripts/slurm/monitor_gpu_memory.py "$output/gpu_memory.csv" &
gpu_monitor_pid=$!
trap 'kill "$gpu_monitor_pid" 2>/dev/null || true' EXIT

# The pinned-conformance tests gate the run: a reference claim that no longer
# matches the vendored LeWM/stable-pretraining sources must not produce data.
.venv/bin/python -m pytest -q \
    test/test_world_model_reference_profile.py \
    test/test_world_model_reference_context.py \
    | tee "$output/preflight_tests.txt"

srun --ntasks=1 .venv/bin/python -u -m examples.world_model.train \
    --config-name world_model_reference \
    device=cuda "seed=$seed" \
    "data.root=$data_root" "data.regime=$regime" \
    "hydra.run.dir=$output/run" \
    2>&1 | tee "$output/train.log"

test -s "$output/run/model.pt"
test -s "$output/run/metrics.json"

printf '\nPeak GPU memory:\n'
awk -F, 'NR>1 {gsub(/ MiB/,"",$4); if ($4+0>m) m=$4+0} END {print "  " m " MiB"}' \
    "$output/gpu_memory.csv"
printf 'Completed A0 sanity: %s\n' "$output"
