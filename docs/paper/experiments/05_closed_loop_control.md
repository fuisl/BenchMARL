# Closed-loop control with a learned world model

2026-09-15. M5's third link. [Counterfactual prediction](05_counterfactual_prediction.md)
and [plan ranking](05_plan_ranking.md) cover the first two; this is the first time
a learned world model in this project drives a planner.

## Hypothesis

Everything measured so far is open-loop. C7, plan ranking and the goal-cost
prototype all score plans the planner never executes, so the chain the
[proposal](../multi_agent_latent_mpc_proposal.md) names --
`counterfactual accuracy -> plan ranking -> closed-loop control` -- is missing
its end. `mpc.py` scored candidates with `oracle_plan_costs` and nothing else,
so no learned model had ever been put in the loop.

Two predictions, both falsifiable by this run:

1. **The reward-readout planner fails on Buzz Wire.** Job 1203 measured the
   learned readout as *anti-correlated* there (rho ~ -0.25) even when given the
   simulator's own latents, because the reward depends on the ball and no agent
   observes the ball. A planner that maximises a mis-signed cost should do worse
   than random, not merely worse than the oracle.
2. **The goal planner is not bound by that.** LeWM's terminal latent-goal
   distance needs neither a reward head nor a termination head, and job 1204
   gave it roughly five times the ranking signal (Spearman ~0.31 against ~0.05)
   on the same checkpoints.

If both hold, the readout -- not the dynamics, and not the relational structure
-- is what stands between this project's prediction results and control.

## Fixed design

The oracle pilot is the reference, so its horizon, budget, cadence and objective
are held and exactly one thing changes: the dynamics the planner rolls.

| Choice | Setting |
|---|---|
| Policies | `random`, `oracle`, `reward\|kind\|regime\|seed`, `goal\|kind\|regime\|seed` |
| Tasks | Buzz Wire (job 1196 bank and checkpoints); Transport (banks 1190 / job 1194) |
| Checkpoints | 3 baselines × 2 regimes × 8 seeds = 48 per task, each scored twice |
| CEM | K=300, R=30, 30 elites (10% of K), H=5 blocks, `init_std` 1.0 |
| MPC | receding horizon 5 blocks, action block 5, warm start |
| Objective (`reward`) | `J = -sum_t sum_i r_i,t`, survival-weighted, the plan's committed cost |
| Objective (`goal`) | LeWM terminal latent-goal distance, `history_size` 3 |
| Evaluation states | 20 drawn from each bank's **test split**, seed 7100 |
| Goal source | terminal observation of one random plan from the same state |
| Uncertainty | bootstrap CIs on return; Wilson intervals on success |

Evaluation states come from the test split, so no model is asked to control a
state it trained on. Every policy sees the same states and the same planner
seed.

### Why the two costs are reported apart

[experiment_plan.md](../experiment_plan.md) fixes task reward as the planning
objective and defers latent-goal scoring, requiring that "any future comparison
using it must be labeled separately". `goal` is therefore a labelled
alternative, not a substitute.

They are also not measuring the same thing. `reward` and `oracle` are scored by
the scenario's own `done()`, so their success columns are comparable. `goal`
is scored against an *achieved* observation drawn from a random rollout, which
is LeWM's protocol -- reachable by construction, so the ranking question is well
posed -- and therefore measures goal-reaching, not task success. Reading `goal`'s
success column as task competence would be wrong.

## Implementation

`evaluate_policy` gains one parameter, `plan_costs(snapshot, observation,
candidates) -> (B,K)`, defaulting to the simulator oracle. The oracle needs the
snapshot and ignores the observation; a learned planner is the reverse, planning
from what the agents can actually see. Nothing else in the loop changes, and an
explicit `oracle_costs(scratch)` is pinned by test to produce episode rows
identical to the previous hard-wired path.

The two learned costs reuse existing, already-tested scoring code rather than
restating it: `plan_ranking.model_costs` for the readout objective, including
its survival weighting, and `goal_planning.goal_plan_costs` for LeWM's.

One defect found and fixed before the run: both helpers return costs on CPU,
because their own callers compare on CPU, while CEM selects elites with `topk`
and gathers from candidates on the planner's device. Left alone that hands a
CUDA planner CPU indices -- the same class of mismatch that killed job 1202, and
invisible on a CPU-only run. A device-parameterised regression test covers it and
runs on CUDA inside the allocation.

## Validation

- `test/test_world_model_closed_loop.py`: the explicit oracle path reproduces the
  default path's episode rows exactly; an injected cost is actually called and
  receives planner shapes (primitive steps, not blocks) and changes the executed
  actions; `achieved_goals` returns a reachable observation; learned costs return
  on the planner's device on both CPU and CUDA.
- 75 tests pass across the world-model suite after the `mpc.py` change.
- Level 0: a 2-state, 8-sample, 2-iteration CPU run completes all four policy
  types end to end.

```bash
sbatch scripts/slurm/closed_loop.sbatch
squeue -u "$USER"
```

## Results

Pending. The run also records planning wall-clock per decision for every policy,
which is the first direct measurement of what the learned model buys in speed:
the oracle spends 72.4 s per decision at R=30 because it rolls K=300 × R=30
simulator trajectories, and a latent rollout replaces those with batched forward
passes.
