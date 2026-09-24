# 36 — Single-agent control: can the learned world model plan navigation to its goal?

**Status: completed and interpreted, 2026-09-24.** Jobs 1570 and 1571 (1569
cancelled for being too slow; it produced nothing). Artifacts:
`outputs/single_agent_control_1570/`, `outputs/single_agent_control_1571/`.
Planner: `examples/world_model/single_agent_control.py`. Launcher:
`scripts/slurm/single_agent_control.sbatch`. Checkpoints: the 18 gate models of
job 1566 ([35](35_single_agent_gate.md)); nothing was retrained.

## Why

Note 35 showed that the latent pipeline can *imagine* one agent's motion. This
note asks the closed-loop question LeWM itself answers: can a planner drive the
real environment to a goal using only the learned model? If one agent on linear
dynamics cannot be controlled, no multi-agent planning claim is possible.

## Setup

* **Task:** VMAS `navigation`, `n_agents=1`, collisions off, 100-step budget
  (the task default). Start and goal are uniform in [−1, 1]². Success is the
  scenario's own `done()`: the agent within 0.1 of its goal. Start-to-goal
  distance `d0` has a median of 1.05 and a maximum of 1.98.
* **Episodes:** seed 9100. Job 1570 uses 64 episodes; job 1571 uses the first 32
  of the same set, so episode indices match.
* **Planner (LeWM's shipped protocol):**
  * CEM with LeWM's solver config: 300 samples, 30 elites, 30 iterations, init
    std 1. Horizon 5 blocks × 5 steps.
  * Cost: squared L2 between the **final** imagined latent and the encoded goal
    observation (`JEPA.criterion`). The goal observation is the agent at its goal
    and at rest, `[g_x, g_y, 0, 0, 0, 0]`, the vector-state analogue of a goal
    image.
  * Context: three real frames. The episode starts at rest, so LeWM's
    start-of-episode convention (repeat the first frame, zero past actions) is
    physically exact here.
  * One deliberate deviation: replan after every block (`receding_horizon=1`),
    not LeWM's five. The reference history requires it, and this project measured
    the five-block cadence as the dominant control failure.
* **References on the same episodes:**
  * random actions;
  * a saturated PD controller toward the goal (gain 4, damping 4);
  * oracle CEM, which plans the **same** latent-free objective (distance to the
    goal observation) with the true simulator, so the gap to it belongs to the
    world model;
  * the learned model planned with its **reward head** instead of goal distance
    (one checkpoint, 16 episodes).

## Results

| policy | gate (35) | episodes | success | median steps |
|---|---|---:|---:|---:|
| random actions | — | 64 / 32 | 1/64, 0/32 | — |
| PD controller | — | 64 / 32 | 64/64, 32/32 | 42 / 42 |
| oracle CEM (true simulator) | — | 64 / 32 | 64/64, 32/32 | 33 / 36 |
| **learned, base recipe** | G1 pass | 3 × 64 | **190/192 (99%)** | 43–45 |
| **learned, random window** | G1 pass | 3 × 32 | **96/96 (100%)** | 41–45 |
| **learned, λ = 0.009** | G1 pass | 3 × 32 | **96/96 (100%)** | 43–44 |
| **learned, λ = 0.009 + window** | G1 pass | 3 × 32 | **95/96 (99%)** | 43–44 |
| **learned, 300 epochs** | G1 pass, lowest error | 3 × 64 | **176/192 (92%)** | 38–42 |
| **learned, ¼ data** | **G1 fail** | 3 × 32 | **91/96 (95%)** | 41–46 |
| learned 300 epochs, reward-head cost | — | 16 | 2/16 | 94 |

Per-seed results: base 64, 63, 63; 300 epochs 61, 57, 58 (of 64); ¼ data 31, 31,
29; λ + window 32, 31, 32 (of 32). Planning cost about 750–1,400 s per 32
batched episodes per checkpoint on one A100, against 350 s for the oracle.

### Where the learned planner fails, by start-to-goal distance

| recipe | `d0` < 1 | 1 ≤ `d0` < 1.5 | `d0` ≥ 1.5 |
|---|---:|---:|---:|
| base (64 ep × 3) | 0/87 | 0/54 | 2/51 |
| 300 epochs (64 ep × 3) | 0/87 | 8/54 | 8/51 |
| ¼ data (32 ep × 3) | 0/36 | 3/36 | 2/24 |
| window, λ, λ + window (32 ep × 9) | 0/108 | 1/108 | 0/72 |

* **No learned planner failed a goal closer than 1.0**, in 318 attempts.
* **Most failures are stalls.** In 18 of the 24 failed episodes, the closest
  approach was within 0.3 of the starting distance, so the agent barely made
  progress on a far goal. Two failures are near-misses: the closest approach was
  0.11 against a 0.10 threshold. Four others made partial progress.
* **The same episodes fail across seeds and recipes.** Episodes 3, 25, 27, 34 and 36 fail in two
  or more of the three 300-epoch seeds; episode 52 (`d0` 1.75) is both of the
  base recipe's failures.

## Interpretation

1. **The learned latent model supports goal-reaching planning on this task.**
   With LeWM's protocol and nothing tuned, every recipe reaches 92–100%.
   Successful episodes take about as many steps as the PD controller, and 5–10
   more than CEM on the true simulator.
2. **Latent goal distance is the usable objective; the reward head is not.**
   The same checkpoint scores 2/16 with its reward head, which is consistent
   with K6 and K18. This supports LeWM's design choice of a reward-free
   latent-distance cost.
3. **Prediction accuracy does not predict control.** The recipe with half the
   rollout error (300 epochs) has the most failures. The recipe that fails the
   prediction gate (¼ data) still reaches 95%. What control needs appears to be
   a latent distance that still orders states far from the goal, and more
   training can sharpen prediction while flattening that far-field ordering.
   **This is a hypothesis; it was not measured here.**
4. **Failures are far-goal stalls, not near-goal confusion.** The
   planning horizon covers 25 steps, while PD needs about 42 on the median
   episode, so far goals rely on the latent distance decreasing well before the
   goal is in reach. The goal state itself (at the goal, at rest) is rare in
   the random-action training data (0.1% of frames within 0.2 of the goal at
   speed < 0.05). But near goals never fail, so that rarity is **not** the
   failure mechanism.
5. **Scope.** One agent, linear dynamics, no obstacles, fresh episodes on one
   seed. This establishes that the planning interface works end to end on this
   stack. It says nothing about interaction between agents, and K11's negative
   result on Buzz Wire (multi-agent, coupled) stands.

## Correction to the in-session analysis

The first reading of job 1570 reported that failures were **near** goals
(`d0` 0.28–1.34) where the agent "drives away", and it attributed them to the
out-of-distribution goal state. That reading used wrong start and goal
coordinates. Reconstructing the episodes from the evaluation seed and checking
them against the random policy fixes this: under random actions the closest
approach can never be below `d0`, and that holds for all 64 episodes only with
the corrected coordinates. The failures are far-goal stalls, as tabled above.

## What would test the open hypotheses

* **Cost landscape:** latent goal distance against true distance along straight
  paths, for failing and succeeding episodes, per recipe. It predicts a flatter
  far-field slope for the 300-epoch checkpoints.
* **LeWM's own goal protocol:** the goal is the observation 25 steps ahead on a
  real trajectory, with a 50-step budget. This keeps goals within the planning
  horizon and in-distribution, and is directly comparable to the paper.
* **A longer horizon or subgoals** on the failing far-goal episodes.
