#!/usr/bin/env bash
# Is the observation-vs-latent gap (K36) the encoder, or the diagnostic head?
#
# Job 1538 (pooled): observation R = 0.537, latent R = 0.188. K36 reads that as
# the JEPA encoder discarding two thirds of what the observation carries. But
# the head sees a 6-D input in one arm and a 192-D input in the other, at ~700
# training anchors, and a deterministic 6 -> 192 encoder can only lose
# information by being non-injective. Three controls separate the readings:
#
#   observation_projected   raw observation through a fixed random linear map
#                           to width 192: same information, latent's width
#   latent_plus_observation latent with the raw observation appended
#   latent_untrained        same architecture and encode path, random weights
#
#   encoder discards it  -> projected ~ raw obs;  untrained >> trained latent
#   head's dimensionality -> projected ~ latent;  untrained ~ trained latent
#
# Job 1538's configuration otherwise unchanged: cross-fitted head, cell 1:0,
# pooled E_CF, sampled reference. ENC_RUNS may point at a directory of symlinks
# to split the checkpoints across slices, or list several such directories
# (space separated) to run them as parallel processes inside this one job: the
# work is single-threaded CPU (simulator + small head fits) and uses <1 GB of
# GPU memory, so one allocation can carry many processes.
#SBATCH --job-name=enc-confound
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=20G
#SBATCH --gres=gpu:a100_2g.10gb:1
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the BenchMARL repository root}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="${SLURM_SUBMIT_DIR}${PYTHONPATH:+:$PYTHONPATH}"

runs="${ENC_RUNS:-outputs/ta2_reference_baselines_1498/runs}"
jacobian="${ENC_JACOBIAN:-outputs/ta1_interaction_jacobian_1497/result/interaction_jacobian.json}"
data="${ENC_DATA:-outputs/buzz_wire_1196/data}"
cells="${ENC_CELLS:-1:0}"
# Extra evaluator arguments (e.g. "--label-from-dir", "--horizon 3
# --references 3"), and an optional runtime split of ONE runs directory into
# ENC_CHUNKS parallel processes, so this job can be queued before the
# checkpoints it evaluates exist.
extra="${ENC_EXTRA:-}"
chunks="${ENC_CHUNKS:-}"
tag="${ENC_TAG:+${ENC_TAG}_}"
output="${SLURM_SUBMIT_DIR}/outputs/encoder_confound_${tag}${SLURM_JOB_ID}"
mkdir -p "$output/source"
exec > >(tee -a "$output/batch.log") 2>&1

printf 'Encoder confound audit: job=%s cells=%s\nruns=%s\n' "$SLURM_JOB_ID" "$cells" "$runs"
printf 'CUDA_VISIBLE_DEVICES=%s\n' "${CUDA_VISIBLE_DEVICES:?}"
test -f "$jacobian"
if [[ -n "$chunks" ]]; then
    test -d "$runs"
    i=0
    for run in "$runs"/*/; do
        test -f "$run/model.pt" || continue
        mkdir -p "$output/chunks/c$((i % chunks))"
        ln -s "$(readlink -f "$run")" "$output/chunks/c$((i % chunks))/"
        i=$((i + 1))
    done
    runs=$(ls -d "$output"/chunks/c* | tr '\n' ' ')
fi
for dir in $runs; do test -d "$dir"; ls -l "$dir" >> "$output/runs_listing.txt"; done

git rev-parse HEAD > "$output/commit.txt"
git status --short > "$output/git_status.txt"
git diff --binary > "$output/working_tree.patch"
cp examples/world_model/{counterfactual_localization,counterfactual_fidelity,interaction_jacobian}.py \
    "$output/source/"
cp scripts/slurm/encoder_confound_audit.sbatch "$output/"

.venv/bin/python -u scripts/slurm/monitor_gpu_memory.py "$output/gpu_memory.csv" &
gpu_monitor_pid=$!
trap 'kill "$gpu_monitor_pid" 2>/dev/null || true' EXIT

.venv/bin/python -m pytest -q test/test_world_model_interaction_jacobian.py \
    | tee "$output/preflight_tests.txt"

pids=()
for dir in $runs; do
    name=$(basename "$dir")
    .venv/bin/python -u -m examples.world_model.counterfactual_localization \
        "$dir" --data "$data" --jacobian "$jacobian" \
        --out "$output/result_$name" --fit-target cross --cells $cells \
        --encoder-controls $extra --device cuda > "$output/run_$name.log" 2>&1 &
    pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
test "$status" -eq 0
for dir in $runs; do
    test -s "$output/result_$(basename "$dir")/counterfactual_localization.json"
done
printf '\nCompleted encoder confound audit: %s\n' "$output"
