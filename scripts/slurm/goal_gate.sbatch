#!/usr/bin/env bash
# Gate: can a planner with TRUE dynamics reach a goal without wrecking the task?
#
# References only, no checkpoints. If the goal oracle cannot do this, no learned
# model will, and there is nothing to measure -- so this runs before any sweep
# rather than after one. Two designs failed this gate already:
#
#   arbitrary goal (jobs 1218-1226): goal oracle reaches it 20/20 and ends at the
#     task distance random reaches. Task-orthogonal.
#   reference-controller goal (job 1229): goal oracle reaches it 12/20, collides
#     in 90% of episodes and ends WORSE than random on the task. Actively harmful.
#
# Two candidate repairs, one per variant, run on separate slices to get both
# answers in one pass:
#
#   VARIANT=position  drop velocity from the goal distance. Job 1229's failure is
#     a velocity-matching pathology: the goal says arrive HERE AT THIS SPEED, and
#     on Buzz Wire that means driving through the wire. The agents never observe
#     the ball or the wire, so the objective cannot see the crash it causes.
#     Keeps where-to-be, drops how-fast-to-arrive.
#
#   VARIANT=success   take the goal only from episodes the reference controller
#     SOLVED. Job 1229's goals included seven crash states out of twenty, because
#     the oracle approaches monotonically until it collides and its best frame is
#     its crash frame in 20/20 episodes. A solved episode cannot have that
#     problem: the ball is on target and nothing collided. Costs states -- the
#     Buzz Wire reward oracle succeeds 5/20 -- so a 4x pool is run and the
#     unsolved states are dropped, which changes the evaluation set into "states
#     a competent controller can solve" and is reported as such.
#
# Buzz Wire only. Transport is uncontrollable from these anchors -- VMAS's own
# heuristic closes 0.032 of the starting 0.9006 in a full episode, and doing
# nothing loses 0.0012 -- so it cannot gate anything.
#
# Pass: goal oracle reaches the goal AND ends well below random's 1.0852 task
# distance AND does not collide more than random's 0.70. Anything else is a
# third failed design and gets reported as the negative result.
#SBATCH --job-name=goal-gate
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the BenchMARL repository root}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1
variant="${GOAL_VARIANT:?Set GOAL_VARIANT=position or success}"
states="${CLOSED_LOOP_STATES:-20}"
output="${SLURM_SUBMIT_DIR}/outputs/goal_gate_${variant}_${SLURM_JOB_ID}"
mkdir -p "$output"
exec > >(tee -a "$output/batch.log") 2>&1
printf 'Goal gate: job=%s variant=%s states=%s\n' "$SLURM_JOB_ID" "$variant" "$states"
printf 'CUDA_VISIBLE_DEVICES=%s\n' "${CUDA_VISIBLE_DEVICES:?}"
git rev-parse HEAD > "$output/commit.txt"
git status --short > "$output/git_status.txt"
git diff --binary > "$output/working_tree.patch"
mkdir "$output/source"
cp examples/world_model/*.py "$output/source/"
cp scripts/slurm/goal_gate.sbatch "$output/"

.venv/bin/python -u scripts/slurm/monitor_gpu_memory.py "$output/gpu_memory.csv" &
gpu_monitor_pid=$!
trap 'kill "$gpu_monitor_pid" 2>/dev/null || true' EXIT

# The goal oracle must optimize exactly the distance EpisodeStats reports, which
# the dimension weighting could silently break. That equality is a test.
srun --ntasks=1 .venv/bin/python -m pytest \
    test/test_world_model_closed_loop.py test/test_world_model.py -q \
    | tee "$output/contract_tests.txt"

if [ "$variant" = position ]; then
    extra=(--goal-source reference --goal-metric position)
else
    extra=(--goal-source success --state-pool "$((4 * states))" --goal-metric full)
fi

.venv/bin/python -u -m examples.world_model.closed_loop \
    outputs/buzz_wire_1196/baselines --data outputs/buzz_wire_1196/data \
    --device cuda --states "$states" --max-runs 0 "${extra[@]}" \
    --reference-cache "$output/buzz_wire_refs.pt" \
    --output "$output/buzz_wire_references.json" --wandb \
    2>&1 | tee "$output/buzz_wire_references.txt"

printf '\nPeak GPU memory:\n'
awk -F, 'NR>1 {gsub(/ MiB/,"",$4); if ($4+0>m) m=$4+0} END {print "  " m " MiB"}' \
    "$output/gpu_memory.csv"
printf 'Completed goal gate (%s): %s\n' "$variant" "$output"
