#!/usr/bin/env bash
# Single-agent control: plan VMAS navigation to its goal with the learned world
# model, LeWM protocol (CEM 300/30/30, horizon 5 blocks, terminal latent-goal
# cost), replanning every block. One process per job-1566 cell (CTRL_CELLS),
# 3 seeds each, CTRL_EPISODES fresh episodes (seed 9100, disjoint from the training bank's state seed).
# References: random, PD controller, oracle CEM on the same objective.
#SBATCH --job-name=single-ctrl
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=06:00:00
#SBATCH --output=slurm-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the BenchMARL repository root}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
export PYTHONPATH="${SLURM_SUBMIT_DIR}${PYTHONPATH:+:$PYTHONPATH}"

runs="${CTRL_RUNS:-outputs/single_agent_gate_1566/runs}"
cells="${CTRL_CELLS:-base ep300}"
episodes="${CTRL_EPISODES:-64}"
# One small reward-readout diagnostic on a single checkpoint, alongside.
reward_run="${CTRL_REWARD_RUN:-ep300_seed4100}"
output="${SLURM_SUBMIT_DIR}/outputs/single_agent_control_${SLURM_JOB_ID}"
mkdir -p "$output/source"
exec > >(tee -a "$output/batch.log") 2>&1
printf 'Single-agent control: job=%s runs=%s cells=[%s]\n' "$SLURM_JOB_ID" "$runs" "$cells"
git rev-parse HEAD > "$output/commit.txt"
git status --short > "$output/git_status.txt"
git diff --binary > "$output/working_tree.patch"
cp examples/world_model/{single_agent_control,mpc,cem,closed_loop,goal_planning,model_input}.py "$output/source/"
cp scripts/slurm/single_agent_control.sbatch "$output/"
.venv/bin/python -u scripts/slurm/monitor_gpu_memory.py "$output/gpu_memory.csv" &
monitor=$!
trap 'kill "$monitor" 2>/dev/null || true' EXIT

pids=()
first=1
for cell in $cells; do
    extra="--skip-oracle"
    if ((first)); then extra=""; first=0; fi
    .venv/bin/python -u -m examples.world_model.single_agent_control \
        "$runs/${cell}_seed4100" "$runs/${cell}_seed4101" "$runs/${cell}_seed4102" \
        --output "$output/$cell" --objectives goal --episodes "$episodes" $extra \
        --device cuda > "$output/run_$cell.log" 2>&1 &
    pids+=("$!")
done
if [[ -n "$reward_run" ]]; then
    .venv/bin/python -u -m examples.world_model.single_agent_control \
        "$runs/$reward_run" --output "$output/reward_$reward_run" --objectives reward \
        --episodes 16 --skip-oracle --device cuda > "$output/run_reward.log" 2>&1 &
    pids+=("$!")
fi
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
grep -h "success" "$output"/run_*.log | sort -u
exit "$status"
