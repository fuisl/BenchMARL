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
# ~43 min, plus ~5 min of oracle and ~1 min of setup. The two tasks run
# concurrently on the one slice, so the job is about one task's wall-clock --
# roughly job 1194's hour. The H100 `mig` partition is not reachable from this
# cluster (`sinfo` shows only `local-a100`), so Level 2 cannot be used here.
#
# Note the timings above were measured on GPU 0, the full un-partitioned A100.
# This slice has roughly 40% of its SMs, so treat them as a lower bound.
#SBATCH --job-name=closed-loop
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --gres=gpu:a100_3g.20gb:1
#SBATCH --time=03:00:00
#SBATCH --output=slurm-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the BenchMARL repository root}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1
states="${CLOSED_LOOP_STATES:-20}"
output="${SLURM_SUBMIT_DIR}/outputs/closed_loop_${SLURM_JOB_ID}"
mkdir -p "$output"
exec > >(tee -a "$output/batch.log") 2>&1
printf 'Closed-loop MPC: job=%s states=%s\n' "$SLURM_JOB_ID" "$states"
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

# No srun here. The two tasks run as background processes inside the allocation
# this script already holds; wrapping each in its own srun step would have them
# contend for the single task slot and serialise exactly what we want
# concurrent.
run_task() {
    local name="$1" runs="$2" data="$3"
    .venv/bin/python -u -m examples.world_model.closed_loop \
        "$runs" --data "$data" --device cuda --states "$states" \
        --output "$output/${name}.json" > "$output/${name}.txt" 2>&1
}

# Both tasks run concurrently in the one allocation, the way the training
# sweeps already pack several workers onto a slice. A single evaluator process
# peaks near 3.3 GB of this 20 GB slice, so two fit with room to spare, and two
# ~50 minute runs finish in one allocation rather than queueing behind each
# other. Nothing in the driver changes: each is an ordinary invocation writing
# its own results.
#
# Buzz Wire is the task with a success criterion AND a demonstrated oracle
# ceiling (11/20), and the one whose readout job 1203 condemned. Transport is
# the mirror image: a working readout over a degenerate objective.
# Smoke each task's own pipeline first, sequentially, before committing the
# allocation to the long runs. Separate output names: concurrent tasks sharing
# one smoke file would race.
smoke buzz_wire outputs/buzz_wire_1196/baselines outputs/buzz_wire_1196/data
smoke transport outputs/interaction_control_1194/transport outputs/transport_data_1190

run_task buzz_wire outputs/buzz_wire_1196/baselines outputs/buzz_wire_1196/data &
buzz_wire_pid=$!
run_task transport outputs/interaction_control_1194/transport \
    outputs/transport_data_1190 &
transport_pid=$!

# Fail the job if either task fails, rather than reporting half an experiment.
wait "$buzz_wire_pid"
wait "$transport_pid"

printf 'Completed closed-loop MPC: %s\n' "$output"
