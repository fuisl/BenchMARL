#!/usr/bin/env bash
# The cheap A-vs-S admission gate: 5 scenarios x 4 horizons, overnight.
#
# WHAT THIS IS NOT. It is not a representation audit. No latent conditions, no
# H0/H1/H2 checkpoints, no `observation_raw`, no `history`. Those are the
# expensive ladder, and job 1540 spent 3 h fitting them on a cell where the
# instrument could not resolve anything.
#
# WHAT IT MEASURES, per (task, horizon):
#
#   J_cross, active fraction      the effect EXISTS      (descriptive)
#   E_A  (actions_only)           the state-blind floor
#   E_S  (physical)               the privileged reference
#   E_A - E_S                     the effect is RESOLVABLE  (admission)
#
# The admission criterion is the LAST one, and it is new. Job 1525 measured a
# real Balance interaction at h=3 -- J_cross rising 0.049 -> 1.110 over h=1..5
# -- and job 1540 still could not measure anything there, because the privileged
# physical state did not beat the blind head under the sampled-reference design.
# Activity and measurability are distinct. The old 50%-activity threshold is
# demoted to a descriptor and is no longer an admission rule on its own.
#
# Separation, not just sign: the reference's 95% upper bound must sit below the
# blind floor's lower bound, both resampled over ROOT EPISODES. A positive gap
# inside bootstrap noise is not a pass.
#
# CELL SELECTION, which is the fix for a real error. Jobs 1527 and 1540 audited
# Balance cells 1:1 and 0:1 -- axis 1, J_cross ~0.52 at 56% active -- while that
# task's strongest cross cells are axis 0, J_cross 1.554 at 75% active. The cell
# list was inherited rather than derived. Here every cell is chosen from that
# task and horizon's own T-A1 ranking: the gate runs where the effect is
# LARGEST, so a reference that cannot beat blind there will not beat it
# anywhere, and a rejection means something.
#
# Buzz Wire is the positive control (job 1538: the ladder resolves cleanly at
# h=1). Dropout is the negative control (J_cross exactly 0.000 at every horizon
# in job 1525). A run that does not admit Buzz Wire, or that admits Dropout, has
# a broken instrument and its other rows mean nothing.
#
# Pooled `E_CF` throughout, per job 1538.
#
# SIZING, measured. Job 1540 ran the same intervention regeneration plus four
# shared conditions on Balance h=3: the shared block finished inside 7 minutes,
# peak 712 MiB GPU and 2.42 GB RSS. This fits TWO conditions instead of four
# and skips the per-checkpoint loop entirely. Five task streams in parallel,
# horizons sequential within a stream: ~1 h expected, 8 h limit.
#SBATCH --job-name=admission-gate
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=32G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=08:00:00
#SBATCH --output=slurm-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the BenchMARL repository root}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="${SLURM_SUBMIT_DIR}${PYTHONPATH:+:$PYTHONPATH}"

horizons="${GATE_HORIZONS:-1 2 3 5}"
references="${GATE_REFERENCES:-3}"
seed="${GATE_SEED:-7301}"
output="${SLURM_SUBMIT_DIR}/outputs/admission_gate_${SLURM_JOB_ID}"
mkdir -p "$output/source"
exec > >(tee -a "$output/batch.log") 2>&1

declare -A BANKS=(
    [transport]=outputs/transport_data_1190
    [wheel]=outputs/wheel_1205/data
    [balance]=outputs/balance_repair_1236/data
    [buzz_wire]=outputs/buzz_wire_1196/data
    [dropout]=outputs/dropout_control_1193/data
)

printf 'A-vs-S admission gate: job=%s horizons=[%s] refs=%s\n' \
    "$SLURM_JOB_ID" "$horizons" "$references"
printf 'CUDA_VISIBLE_DEVICES=%s\n' "${CUDA_VISIBLE_DEVICES:?}"
for task in "${!BANKS[@]}"; do
    test -f "${BANKS[$task]}/manifest.json" \
        || { printf 'MISSING BANK: %s -> %s\n' "$task" "${BANKS[$task]}"; exit 1; }
done

git rev-parse HEAD > "$output/commit.txt"
git status --short > "$output/git_status.txt"
git diff --binary > "$output/working_tree.patch"
cp examples/world_model/{interaction_jacobian,counterfactual_localization,counterfactual_fidelity,admission_gate}.py \
    "$output/source/"
cp scripts/slurm/admission_gate_sweep.sbatch "$output/"

.venv/bin/python -u scripts/slurm/monitor_gpu_memory.py "$output/gpu_memory.csv" &
gpu_monitor_pid=$!
trap 'kill "$gpu_monitor_pid" 2>/dev/null || true' EXIT

.venv/bin/python -m pytest -q test/test_world_model_interaction_jacobian.py \
    | tee "$output/preflight_tests.txt"

run_task() {
    local task="$1" data="${BANKS[$1]}"
    for horizon in $horizons; do
        local cell_dir="$output/$task/h$horizon"
        mkdir -p "$cell_dir"
        printf '[%s h=%s] T-A1\n' "$task" "$horizon"
        if ! .venv/bin/python -u -m examples.world_model.interaction_jacobian \
            --data "$data" --out "$cell_dir/ta1" --split test \
            --horizon "$horizon" --references "$references" --seed "$seed" \
            --device cuda > "$cell_dir/ta1.log" 2>&1; then
            printf '[%s h=%s] T-A1 FAILED (see %s)\n' "$task" "$horizon" "$cell_dir/ta1.log"
            continue
        fi

        # The gate runs on the strongest CROSS cell this task and horizon has,
        # read from the T-A1 it just produced. Never a hardcoded cell list.
        local cell
        cell=$(.venv/bin/python -m examples.world_model.admission_gate \
            --pick-cell "$cell_dir/ta1/interaction_jacobian.json") || {
            printf '[%s h=%s] no cross cell\n' "$task" "$horizon"; continue; }
        printf '[%s h=%s] gate on strongest cross cell %s\n' "$task" "$horizon" "$cell"

        if ! .venv/bin/python -u -m examples.world_model.counterfactual_localization \
            "$data" --data "$data" \
            --jacobian "$cell_dir/ta1/interaction_jacobian.json" \
            --out "$cell_dir/gate" --fit-target cross --cells "$cell" \
            --conditions actions_only physical --shared-only \
            --horizon "$horizon" --references "$references" --seed "$seed" \
            --device cuda > "$cell_dir/gate.log" 2>&1; then
            printf '[%s h=%s] GATE FAILED (see %s)\n' "$task" "$horizon" "$cell_dir/gate.log"
            continue
        fi
        printf '[%s h=%s] done\n' "$task" "$horizon"
    done
}

pids=()
for task in "${!BANKS[@]}"; do
    run_task "$task" &
    pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do
    wait "$pid" || status=1
done
printf '\nAll streams finished (status %s)\n' "$status"

.venv/bin/python -m examples.world_model.admission_gate \
    --summarize "$output" --out "$output/admission_gate.json" \
    | tee "$output/report.txt"

test -s "$output/admission_gate.json"
printf '\nCompleted A-vs-S admission gate: %s\n' "$output"
