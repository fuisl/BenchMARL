#!/usr/bin/env bash
# M5's last link: closed-loop control with a learned world model.
#
# Everything this project has measured is open-loop. C7, plan ranking and the
# goal-cost prototype all score plans that are never executed, so there is no
# evidence yet about control -- the end of the chain the proposal names,
# counterfactual accuracy -> plan ranking -> closed-loop control. The oracle
# reached 11/20 on Buzz Wire and 0/360 on Transport; no learned model has ever
# been put in the loop at all.
#
# Same CEM budget, horizon and states as the oracle pilot; the only thing that
# changes is the dynamics the planner rolls. Two learned costs, reported apart:
#
#   reward  the objective the experiment plan committed to, J = -sum_t sum_i r
#   goal    LeWM's terminal latent-goal distance, which needs no readout
#
# The run has a prediction to falsify. Job 1203 showed Buzz Wire's readout is
# anti-correlated (rho ~ -0.25) even given the simulator's own latents, because
# the reward depends on the ball and no agent observes it -- so `reward` should
# fail there while `goal` need not. Transport is the mirror image: its readout
# ranks plans well (rho ~ 0.67) but its oracle never reaches a goal, so it tests
# return rather than success.
#
# Evaluation states are drawn from each bank's test split, so no model is asked
# to control a state it trained on.
#
# Sizing, from a measured Level-0 probe rather than an estimate: 33 s per
# checkpoint for `reward` and 21 s for `goal`, so 48 checkpoints scored twice is
# ~43 min, plus ~5 min of oracle and ~1 min of setup. One task per submission
# ~43 min. Those were measured on GPU 0, the full un-partitioned A100, and
# proved to be a large underestimate: on this slice, with both tasks sharing it,
# job 1212 measured 243 s per Transport checkpoint rather than 54 s. The grid is
# cut to four seeds and the wall-clock raised accordingly, which puts each task
# near 1.6 hours. Results are now written after every policy, so a wall-clock
# kill costs the remaining checkpoints rather than all of them.
#
# The H100 `mig` partition is not reachable from this cluster (`sinfo` shows
# only `local-a100`), so Level 2 is not available.
#SBATCH --job-name=closed-loop
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100_3g.20gb:1
#SBATCH --time=06:00:00
#SBATCH --output=slurm-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the BenchMARL repository root}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1
states="${CLOSED_LOOP_STATES:-20}"
# Four of the eight training seeds. Job 1212 measured ~4 minutes per checkpoint
# on this slice with both tasks sharing it -- 243 s for Transport, reward plus
# goal -- so the full 48-checkpoint grid is 3.2 hours per task and does not fit.
# 24 checkpoints keeps all three predictors and both regimes, which are the
# comparisons, and spends the budget on those rather than on seeds 4104-4107.
# One process per task, which measurement says is the right number here.
#
# Four processes were tried, on the reasoning that two were using 6.1 GB of a
# 20 GB slice. VRAM was never the limit -- four peaked at 6.9 GB -- but Buzz
# Wire's random policy went from 26 s to 200 s, and raising the allocation from
# 12 cores to 28 changed that by 0.1 s, so it was not CPU contention either.
#
# It is the slice itself. A MIG without MPS time-slices processes rather than
# running their kernels concurrently, and Buzz Wire takes 15 physics substeps
# per step for its rigid joints against Transport's 1, so it issues roughly
# fifteen times the kernels and pays that switching cost fifteen times over.
# Transport, measured in the same jobs, never degraded: 1.8-3.4 s throughout.
#
# The node's `a100_2g.10gb_mps` gres exists for genuine sharing, but two
# Transport processes alone need more than its 10 GB.
seeds="${CLOSED_LOOP_SEEDS:-4100,4101,4102,4103}"
output="${SLURM_SUBMIT_DIR}/outputs/closed_loop_${SLURM_JOB_ID}"
mkdir -p "$output"
exec > >(tee -a "$output/batch.log") 2>&1
printf 'Closed-loop MPC: job=%s states=%s seeds=%s\n' \
    "$SLURM_JOB_ID" "$states" "$seeds"
printf 'CUDA_VISIBLE_DEVICES=%s\n' "${CUDA_VISIBLE_DEVICES:?}"
git rev-parse HEAD > "$output/commit.txt"
git status --short > "$output/git_status.txt"
git diff --binary > "$output/working_tree.patch"
mkdir "$output/source"
cp examples/world_model/*.py "$output/source/"
cp scripts/slurm/closed_loop.sbatch "$output/"

.venv/bin/python -u scripts/slurm/monitor_gpu_memory.py "$output/gpu_memory.csv" &
gpu_monitor_pid=$!
# Only suppress an already-exited monitoring process during job cleanup.
trap 'kill "$gpu_monitor_pid" 2>/dev/null || true' EXIT

# The injected-dynamics tests are the claim that the planner really is driven by
# the learned model, so run them in the allocation rather than trusting a local
# pass. A cost that were silently ignored would produce a plausible control
# result that means nothing.
srun --ntasks=1 .venv/bin/python -m pytest test/test_world_model_closed_loop.py -q \
    | tee "$output/injection_tests.txt"

# One tiny deterministic run before the real evaluation. Jobs 1206 and 1207 died
# on a missing import inside the driver's main loop that the unit tests could not
# see, because they exercise the helpers directly rather than through main().
# This costs under a minute and exercises every policy path end to end.
smoke() {
    local name="$1" runs="$2" data="$3"
    srun --ntasks=1 .venv/bin/python -m examples.world_model.closed_loop \
        "$runs" --data "$data" --device cuda --states 2 \
        --num-samples 8 --num-iters 2 --max-runs 1 \
        --output "$output/smoke_${name}.json" | tee "$output/smoke_${name}.txt"
}

# No srun for these: they run as background processes inside the allocation this
# script already holds, and wrapping each in its own srun step would have them
# contend for the single task slot and serialise exactly what must be concurrent.
#
# One process per (task, seed group). Each writes its own results file, which
# examples.world_model.report aggregates -- the same shape as every other sweep
# here, where many per-run outputs are combined by a reader rather than merged
# by the producer.
#
# The first group of each task also computes the random and oracle references,
# which are per-state rather than per-checkpoint. Giving them their own process
# would score some seed twice; giving them to every group would repeat the
# oracle's expensive planning.
run_group() {
    local name="$1" runs="$2" data="$3" group="$4" references="$5"
    local tag="${group//,/_}"
    local extra=(--skip-oracle)
    [ "$references" = "with-references" ] && extra=()
    .venv/bin/python -u -m examples.world_model.closed_loop \
        "$runs" --data "$data" --device cuda --states "$states" \
        --seeds "$group" "${extra[@]}" \
        --output "$output/${name}_${tag}.json" \
        > "$output/${name}_${tag}.txt" 2>&1
}

bw=outputs/buzz_wire_1196/baselines
bw_data=outputs/buzz_wire_1196/data
tr=outputs/interaction_control_1194/transport
tr_data=outputs/transport_data_1190

# Smoke each task's pipeline, sequentially, before committing the allocation to
# the long runs. Separate output names: concurrent tasks sharing one smoke file
# would race.
smoke buzz_wire "$bw" "$bw_data"
smoke transport "$tr" "$tr_data"

pids=()
run_group buzz_wire "$bw" "$bw_data" "$seeds" with-references & pids+=($!)
run_group transport "$tr" "$tr_data" "$seeds" with-references & pids+=($!)

# Fail the job if any process fails, rather than reporting a partial experiment.
status=0
for pid in "${pids[@]}"; do
    wait "$pid" || status=1
done
[ "$status" -eq 0 ]

printf 'Completed closed-loop MPC: %s\n' "$output"
