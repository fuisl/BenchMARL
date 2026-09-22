#!/usr/bin/env bash
# Stage 0 ONLY: does the cross-agent effect emerge at a planning-relevant horizon?
#
# TRUE SIMULATOR ONLY. No probe, no latent, no JEPA, no information ladder, no
# R_info, no reconstruction, no CEM. Just J_own(h), J_cross(h) and C(h) measured
# on the real environment at h = 1, 2, 3, 5 action blocks.
#
# Job 1522 asked whether perturbing one agent moves the other within ONE block,
# and found Transport 5% active, Balance 31%, Wheel 0%. That is a question about
# LOCAL mechanical coupling. Transport's chain runs
#   a_2 -> package acceleration -> package motion -> configuration -> Y_1,
# and at five primitive steps the last arrow may barely have happened. So the
# open question is whether 1522 found weak interaction or merely DELAYED
# interaction, and one horizon cannot tell them apart.
#
# Buzz Wire is the positive reference (2.24, 100% active at h=1) and Dropout the
# negative one (exactly 0.000). If Buzz Wire stays strong, Dropout stays zero,
# and Transport/Balance rise with horizon, the instrument is working and the
# original admission horizon asked the wrong dynamical question.
#
# Each contrast now carries its OWN validity mask, so an unrelated corner
# rollout can no longer censor a valid cross-effect row. That mattered less for
# screening; it matters now that this gate decides whether a task is usable.
#
# The question is only which existing VMAS task provides the clean OBSERVABLE
# interaction setting Buzz Wire does not: a real cross-agent effect, recoverable
# from legitimate observation, with measurable interaction structure. Only tasks
# that pass go on to H0/H1/H2.
#
# Buzz Wire is carried as the reference case, now characterized by job 1516 as
#   R_observation = 0.605   R_latent = 0.212   R_latent+ball = 0.899
# i.e. the observation carries a majority of the effect and the learned latent
# preserves much less of it. Dropout is carried as the weak-interaction control.
#
# The measurement convention is FROZEN (see note 33): cross-block fitting target,
# one (agent, axis) cell per head, sampled-reference ladder intervention, full
# per-column-scaled body state as Y, 3x3 midpoint grid for J_own/J_cross/C only.
#
# Three seeds run CONCURRENTLY on one full A100. Each seed is an independent
# draw of the reference joint actions, so agreement across seeds is replication
# of the admission verdict rather than of the arithmetic. Each writes its own
# report directory; a seed that fails loses only itself.
#
# Transport is ordered first so the most valuable candidate completes even if
# the window is short, and each scenario's report is written as it finishes.
#SBATCH --job-name=stage0-admission
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=15
#SBATCH --mem=48G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the BenchMARL repository root}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="${SLURM_SUBMIT_DIR}${PYTHONPATH:+:$PYTHONPATH}"

references="${ADM_REFERENCES:-2}"
horizons="${ADM_HORIZONS:-1 2 3 5}"
seeds="${ADM_SEEDS:-7301 7302 7303}"
output="${SLURM_SUBMIT_DIR}/outputs/stage0_admission_${SLURM_JOB_ID}"
mkdir -p "$output/source"
exec > >(tee -a "$output/batch.log") 2>&1

printf 'Stage 0/1 admission: job=%s seeds=[%s] references=%s horizons=[%s]\n' \
    "$SLURM_JOB_ID" "$seeds" "$references" "$horizons"
printf 'CUDA_VISIBLE_DEVICES=%s\n' "${CUDA_VISIBLE_DEVICES:?}"

git rev-parse HEAD > "$output/commit.txt"
git status --short > "$output/git_status.txt"
git diff --binary > "$output/working_tree.patch"
cp examples/world_model/{admission_benchmark,counterfactual_localization,counterfactual_fidelity,interaction_jacobian,collect}.py \
    "$output/source/"
cp scripts/slurm/stage0_admission.sbatch "$output/"

.venv/bin/python -u scripts/slurm/monitor_gpu_memory.py "$output/gpu_memory.csv" &
gpu_monitor_pid=$!
trap 'kill "$gpu_monitor_pid" 2>/dev/null || true' EXIT

.venv/bin/python -m pytest -q test/test_world_model_interaction_jacobian.py \
    | tee "$output/preflight_tests.txt"

pids=()
for seed in $seeds; do
    .venv/bin/python -u -m examples.world_model.admission_benchmark \
        --scenario transport=outputs/transport_data_1190 \
        --scenario wheel=outputs/wheel_1205/data \
        --scenario balance=outputs/balance_repair_1236/data \
        --scenario buzz_wire=outputs/buzz_wire_1196/data \
        --scenario dropout=outputs/dropout_control_1193/data \
        --out "$output/result_seed${seed}" --references "$references" \
        --horizons $horizons --seed "$seed" --device cuda \
        > "$output/run_seed${seed}.log" 2>&1 &
    pids+=("$!")
    printf 'launched seed %s (pid %s)\n' "$seed" "$!"
done

status=0
for pid in "${pids[@]}"; do
    wait "$pid" || status=1
done

printf '\nReports written:\n'
find "$output" -name admission_report.json -printf '  %p\n'
printf '\nCompleted Stage 0/1 admission (exit status %s): %s\n' "$status" "$output"
exit "$status"
