#!/usr/bin/env bash
# Imagined rollouts beside the simulator, for every trained set.
#
# Run directly on a MIG slice (pass it as CUDA_VISIBLE_DEVICES). About 25s per
# task.
#
# The SIMULATOR runs on CPU and the model and probes on the GPU, which is not a
# fallback but the measured right answer. These figures step only the 128
# recorded episode starts, and VMAS at that batch is all per-step overhead: 58
# ms/step on CPU against 1462 ms/step on a MIG slice shared with a running CEM
# job. Putting the whole thing on the GPU took over ten minutes a task; putting
# the whole thing on CPU moved the bottleneck to the float64 MLP probe. Split,
# it is 25 seconds.
#
# `--objects` and `--hub` are per task because the recorded bodies differ:
# Buzz Wire stores [ball, joint, joint] and only the ball is worth drawing;
# Transport stores [package 0]; Balance stores [package, line] and BOTH matter,
# with the agents carrying the line while the package rides on it -- so the
# spokes there go to body 1, not body 0.
set -euo pipefail
cd "$(dirname "$0")/.."
out="${1:-outputs/horizon_curves}"
mkdir -p "$out"

film() {
    local name="$1" runs="$2" data="$3" blocks="$4"; shift 4
    printf '\n=== %s ===\n' "$name"
    .venv/bin/python -u -m examples.world_model.plot_imagination \
        "$runs" --data "$data" --device cuda --sim-device cpu --blocks "$blocks" \
        --output "$out/filmstrip_$name.png" "$@"
}

film buzz_wire   outputs/buzz_wire_1196/baselines      outputs/buzz_wire_1196/data      8 \
    --objects 1 --hub 0 --object-names ball
film transport   outputs/stage1_refit_1217/transport   outputs/transport_data_1190      10 \
    --objects 1 --hub 0 --object-names package
film balance     outputs/balance_repair_1236/baselines outputs/balance_repair_1236/data 10 \
    --objects 2 --hub 1 --object-names package,line
film sigreg_0009 outputs/sigreg_selected_1228/runs     outputs/buzz_wire_1196/data      8 \
    --objects 1 --hub 0 --object-names ball
