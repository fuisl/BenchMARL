#!/usr/bin/env bash
# Latent rollout error against horizon, per seed, for every trained set.
#
# Run directly on a MIG slice rather than through Slurm: the whole sweep is
# about two minutes of GPU and needs ~1GB, so queueing it behind a multi-hour
# job costs more than it saves. Pass the slice as CUDA_VISIBLE_DEVICES.
#
# Writes one JSON per set with PER-SEED curves and each checkpoint's own latent
# variance. The printed tables carry means only, and a mean cannot express the
# paired per-seed comparison that is the only valid way to rank two models whose
# latent spaces were learned separately.
set -euo pipefail
cd "$(dirname "$0")/.."
out="${1:-outputs/horizon_curves}"
mkdir -p "$out"
git rev-parse HEAD > "$out/commit.txt"
git status --short > "$out/git_status.txt"

run() {
    printf '\n=== %s ===\n' "$1"
    .venv/bin/python -u -m examples.world_model.horizon_rollout \
        "$2" --data "$3" --blocks 20 --device cuda \
        --output "$out/$1.json" 2>&1 | tee "$out/$1.txt"
}

run transport   outputs/stage1_refit_1217/transport   outputs/transport_data_1190
run buzz_wire   outputs/buzz_wire_1196/baselines      outputs/buzz_wire_1196/data
run sigreg_0009 outputs/sigreg_selected_1228/runs     outputs/buzz_wire_1196/data
# The REPAIRED Balance bank (job 1236). Job 1235 scored `balance_1233`, whose
# heuristic branch was collected with Transport's policy. That defect does not
# touch these rows -- they are the correlated/independent training regimes and
# the truth is generated here -- but there is no reason to keep citing the
# superseded bank.
run balance     outputs/balance_repair_1236/baselines outputs/balance_repair_1236/data
