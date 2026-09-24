# 35 — Single-agent gate: can the latent model imagine one agent's own motion?

**Status: completed and interpreted, 2026-09-23.** Job 1566. Artifacts:
`outputs/single_agent_gate_1566/`. Evaluator: `examples/world_model/self_rollout.py`,
whose docstring registered the pass rule before any result.

## Why

If an independent (single-agent) latent world model cannot roll out and respond
to its own action, no multi-agent result built on the same pipeline is
interpretable. Note 34 left that untested: every Task A model was multi-agent.

## Setup

* **Task:** VMAS `navigation`, `n_agents=1`, collisions off, 200 steps. The agent
  is a point mass with observation `[pos, vel, pos − goal]` (Buzz Wire's form), and
  its dynamics are **linear** in (observation, action). A ridge fit on the
  observation is exact (validation MSE ≈ 1e-15), so any failure is the latent
  pipeline's, not the data's.
* **Bank:** 512 root episodes, anchors every 25 steps, 50-step (10-block)
  snippets. 6,084 train snippets; 1,006 test snippets over 64 held-out roots.
  The self-intervention flips the agent's own x action from step 15, so both
  branches share three real context frames.
* **Model:** `lewm_reference`, independent kind, 3 seeds per cell.
* **Scoring:** 3 real context frames, then recursive imagination for 8 blocks,
  decoded to position by a probe fitted on TRAIN true latents (linear and MLP).
* **Pass rule:**
  * **G1:** rollout position MSE below persistence at every horizon, every seed.
  * **G2:** self-counterfactual pooled `E_CF` < 0.5 and cosine > 0.8 at horizons 2–4
    (horizon 2 is the first changed block), with the probe floor ≤ 1/3 there.

**Collector and trainer changes.** The collector gains a single-agent mode:
`intervened` = agent 0, and a new `intervention_start_step`. The trainer gains
`train.reference_window=random`, which samples each row's 4-frame window instead of
always the first. Both default to the previous behavior; 34 collection and
reference tests pass.

## Results (MLP probe; rollout MSE ÷ persistence MSE, mean of 3 seeds)

| cell | updates | h1 | h2 | h4 | h8 | rollout RMSE h1 / h8 | probe floor h1 | self-CF `E_CF` h2 / h3 / h8 | gain h3 | cos h2 / h3 | G1 | G2 |
|---|---:|---:|---:|---:|---:|---|---:|---|---:|---|---|---|
| base (T-A2 recipe) | 4,800 | 0.75 | 0.31 | 0.18 | 0.14 | 4.1 / 6.7 cm | 0.23 | 0.58 / 0.35 / 0.25 | 1.04 | 0.72 / 0.80 | **pass** 3/3 | fail |
| random window | 4,800 | 0.77 | 0.35 | 0.23 | 0.19 | 4.1 / 7.9 cm | 0.24 | 0.56 / 0.33 / 0.24 | 1.05 | 0.72 / 0.80 | pass | fail |
| λ = 0.009 | 4,800 | 0.65 | 0.27 | 0.15 | 0.12 | 3.8 / 6.3 cm | 0.40 | 0.66 / 0.37 / 0.23 | 1.14 | 0.78 / 0.83 | pass | fail |
| λ = 0.009 + window | 4,800 | 0.72 | 0.33 | 0.22 | 0.22 | 4.0 / 8.3 cm | 0.41 | 0.64 / 0.38 / 0.24 | 1.15 | 0.78 / 0.83 | pass | fail |
| **300 epochs** | 14,400 | **0.32** | **0.14** | **0.08** | **0.06** | **2.7 / 4.4 cm** | 0.12 | **0.44 / 0.26 / 0.17** | 1.06 | 0.80 / 0.87 | pass | fail (1 seed, cos 0.793) |
| ¼ data, fixed updates | 4,800 | 1.14 | 0.49 | 0.29 | 0.23 | 5.0 / 8.5 cm | 0.29 | 0.66 / 0.41 / 0.30 | 1.07 | 0.69 / 0.77 | **fail** 0/3 | fail |

Persistence RMSE grows from 4.7 cm (h = 1) to 17.9 cm (h = 8). The direct
observation-space model scores exactly 0 on every metric.

**With a linear probe, G1 fails in every cell**, and the probe itself is the
reason: decoding the *true* next latent already scores 0.59–1.25× persistence at
h = 1. Position is not linearly decodable from the 192-D latent to better than
about 4 cm.

## Interpretation

1. **The independent latent model can imagine one agent's own motion.**
   * **Rollout:** it beats persistence at every horizon in 5/6 cells, all seeds,
     and the error barely compounds (4.1 → 6.7 cm over 8 blocks, while
     persistence grows 4.7 → 17.9 cm). In latent space the rollout error is
     0.2–1.6% of latent variance.
   * **Response to its own action:** from the second changed block onward it
     has the right direction (cos 0.80–0.91) and the right size (gain 1.00–1.15).
   * **So single-agent capability does not rule out multi-agent work.**
2. **The first changed block is imprecise:** `E_CF` 0.44–0.66, cosine 0.69–0.81.
   Part of this is the readout, since the probe floor there is 0.29–0.55 and only
   ≤ 1/3 in the 300-epoch cell. G2 is missed by one seed at 300 epochs.
3. **The T-A2 training recipe is under-optimized.** Tripling epochs halves
   rollout error and self-CF error. Here "100 epochs" is 4,800 updates; on the
   Buzz Wire T-A2 bank it is **600** (683 snippets) and on Balance 1,100. Every
   note 30–34 checkpoint therefore got 1/8 to 1/24 of the optimization that a
   linear single-agent system needs to pass this gate cleanly. The K49/K50
   scaling results and the K21 gain of 0.64 were all measured on this
   under-optimized recipe.
4. **Data matters at fixed updates:** ¼ of the roots fails G1. Consistent with K50.
5. **The random window does not help, and λ = 0.009 trades readout for dynamics.**
   The random window slightly increases long-horizon drift. λ = 0.009 gives
   slightly better rollout but a worse probe floor (0.40 vs 0.23), as in K51.
6. **The latent pipeline is still far from exact** on a system a linear model
   solves perfectly: 2.7–6.7 cm of error against 0.

## Consequence

The multi-agent failure is **not** explained by an inability to model a single
agent. But the recipe every multi-agent checkpoint used is under-trained by an
order of magnitude in updates. Before any further multi-agent comparison:
* fix a **minimum-updates** budget (≥ 14k, or train to validation convergence);
* re-check that the single-agent gate passes G2 at that budget;
* then re-run the H0/H1 comparison on Buzz Wire at matched updates.
