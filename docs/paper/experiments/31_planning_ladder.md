# Planning ladder: Gates 0-4 from counterfactual fidelity to joint control

**Status: registered 2026-09-21, before any gate past 1 has run.** Decision
rules are fixed here ahead of results, following the A1.2 precedent.

Direction: [`../direction_planning_2026-09-21.md`](../direction_planning_2026-09-21.md).
Task A ladder: [`30_task_a_counterfactual_fidelity.md`](30_task_a_counterfactual_fidelity.md).

## The chain under test

```math
\boxed{\text{interaction structure}\rightarrow\text{counterfactual fidelity}
\rightarrow\text{decision ranking}\rightarrow\text{joint control}}
```

Each gate tests one arrow. **A gate may not be run until the previous one
passes**, because this project's central failure was reading a downstream metric
while an upstream link was broken.

## Hard precondition on Gates 2-4

> **No significant control compute until at least one model reaches
> `E_CF < 1` robustly on the interaction cells.**

`E_CF = 1` is not an arbitrary bar. In this metric it is exactly the
*predict-zero-interaction* baseline, so a model above it is one a planner would
do better to ignore than to consult. T-A2 currently has every conditioned arm
above it.

This precondition is registered so it cannot be quietly relaxed if Gates 2-4
look tempting. **If no model ever satisfies it, the paper is the Task A
measurement paper and Gates 2-4 are not run at all.**

## Gate 0 - can the planner solve the task with a perfect model?

**Passed, already.** K4: true-simulator CEM solves Buzz Wire 25/32 with zero
collisions, at `--execute-blocks 1`, on 32 independent roots
([`15_buzz_wire_control_gate.md`](15_buzz_wire_control_gate.md)).

Its value is as a control, not a result: any later learned-model failure at the
same horizon, action block, sample count and objective cannot be blamed on
search incapacity. **Gate 4 must reuse these settings exactly**, or that
protection is lost.

## Gate 1 - counterfactual world-model fidelity

Task A. T-A1 (job 1497) and T-A2 (job 1502) cover `h=1`.

**Extension registered here:** planning consumes *recursive* predictions, so a
single-block number does not describe what a planner rolls. Gate 1 requires

```math
E_{\rm CF}(h),\qquad h=1,2,3
```

with cosine and magnitude ratio at each horizon. `h=4,5` are excluded because
T-A1 showed every anchor has terminated there under the maximal constant-action
intervention; they are reported as unmeasurable, never as zero.

Running as **job 1505** on the 48 existing checkpoints, both probe families.

**Registered reading.** `E_CF(h)` rising steeply with `h` implicates recursive
compounding; flat-but-high implicates the one-step response itself. The two
imply different repairs, so the shape is the result, not the `h=1` value alone.

### Gate 1 horizon result (job 1505)

48 checkpoints, both probe families, `correlated` regime shown; the linear and
MLP probes agree on every ordering.

| `h` | arm | `E_CF` | cosine | magnitude | live anchors |
|---:|---|---:|---:|---:|---:|
| 1 | independent | 1.0000 | +0.000 | 0.000 | 13,424 |
| 1 | joint | 1.137 | **+0.400** | 0.898 | |
| 1 | relational | 1.051 | **+0.408** | 0.740 | |
| 2 | independent | 1.0000 | +0.000 | 0.000 | 4,736 |
| 2 | joint | 1.389 | **+0.230** | 1.072 | |
| 2 | relational | 1.213 | **+0.238** | 0.832 | |
| 3 | independent | 1.0000 | +0.000 | 0.000 | 968 |
| 3 | joint | 1.280 | **+0.015** | 0.724 | |
| 3 | relational | 1.267 | **−0.138** | 0.611 | |

**The headline is the cosine, not `E_CF`.** The directional cross-agent content
that exists at one block decays monotonically and is **gone by three**: joint
reaches +0.015, statistically indistinguishable from the uninformed 0.000, and
relational goes **negative** at −0.138, i.e. anti-correlated with the true
response. Full-coverage (`independent`-regime) models decay more slowly
(+0.453 → +0.352 → +0.240 for joint) but in the same direction.

`E_CF` itself rises 1.14 → 1.39 from `h=1` to `h=2`, so recursive compounding is
implicated by the registered reading. Its apparent partial recovery at `h=3` must
**not** be read as improvement — see the survivorship caveat below.

**The relational advantage is horizon-limited.** Paired by seed on cross:

| `h` | correlated, relational vs joint | independent regime |
|---:|---|---|
| 1 | −0.086 / −0.108, **8/8** seeds | −0.031 / −0.044, 6-7/8 |
| 2 | −0.175 / −0.196, **8/8** | −0.029 / −0.088, 6-7/8 |
| 3 | −0.013 / −0.017, **3-4/8** | +0.026 / +0.025, 3-4/8 |

It peaks at `h=2` and **vanishes at `h=3`**. So K22b, and the relational result
generally, is a claim about short-horizon counterfactual response, not about
rollouts of the depth a planner uses.

**H1 never beats H0 at any horizon:** 0/8 seeds in all twelve cells. That
falsification is now horizon-robust, not an `h=1` artifact.

#### Survivorship caveat on `h=3`

Live anchors fall 13,424 → 4,736 → 968. The `h=3` rows describe only episodes
that survived fifteen primitive steps of a **maximal constant-action**
intervention, which is a selected and atypical subpopulation. The probe floor
stays healthy there (resolution 4.4-6.6, clearing the registered 3x rule), so
the numbers are measurements rather than noise — but they are measurements on a
different, smaller population than `h=1`, and the apparent `E_CF` recovery at
`h=3` is most likely that selection rather than better prediction. Cross-horizon
comparisons of `E_CF` **levels** are therefore not licensed; the cosine trend,
which moves monotonically and in the same direction in every arm and both
probes, is the defensible reading.

#### Consequence for Gates 2-4

A planner rolling `H = 5` blocks would consume predictions whose cross-agent
directional content is zero or anti-correlated beyond roughly two blocks. This
independently reinforces the control precondition above: the issue is not only
that `E_CF > 1` at one block, but that whatever interaction signal exists does
not survive to the depth a planner needs.

## Gate 2 - fixed candidate ranking

**The most important experiment before control**, and the first that touches
`R` and `V`.

From a fixed real state, `K = 300` joint trajectories. The simulator gives true
returns `J_1..J_K`; the model gives `Ĵ_1..Ĵ_K`.

| Metric | Why |
|---|---|
| Spearman `ρ(Ĵ, J)` | ordering quality — report the **rankable fraction** beside it, since a near-tied candidate set makes ρ undefined or tie-dominated |
| top-k recall | whether the good plans survive into the elite set CEM would keep |
| **selected regret** `R_sel = J* − J_argmax Ĵ` | **the headline.** The operational definition of a useful planning model |

`R` and `V` are trained on **frozen** latents. If reward gradients reshape the
representation, "does the world model preserve counterfactual structure?"
silently becomes "did the objective turn it into a policy representation?" An
end-to-end variant is a later ablation and must be labelled as one.

**Known hazard, carried forward.** A better-fitted reward head has already been
shown to rank plans *worse* (K6, K12): halving held-out reward MSE lowered
ranking quality. So Gate 2 is scored on ranking and regret, never on reward MSE.

### Gate 2's registered rule, and H3

H3 is the missing causal arrow and the scientifically important claim:

```math
E_{\rm CF}\downarrow \;\Rightarrow\; R_{\rm sel}\downarrow .
```

Measured **across frozen models** — the three kinds, eight seeds, both coverage
regimes — as the rank correlation between each model's `E_CF` and its
`R_sel`, clustered by training seed.

| Condition | Verdict |
|---|---|
| `ρ(E_CF, R_sel) > 0`, interval clear of zero | **H3 supported**: counterfactual fidelity predicts planning quality |
| interval contains zero | H3 unsupported; ranking is governed by something else |
| additionally `ρ(E_CF, R_sel) > ρ(IID rollout error, R_sel)` | the paper's sharper claim: **counterfactual fidelity predicts planning quality better than ordinary prediction loss does** |

The comparison arm is already half in hand: M4 found one-step teacher-forced
prediction does **not** separate the baselines, so an IID-error predictor of
`R_sel` is expected to be weak. That contrast is the point.

## Gate 3 - optimizer-induced distribution shift

Run CEM and MPPI on the **identical frozen** model. Save candidate populations
at iterations 1, 5, 10, 20, 30 and measure at each: model error / `E_CF`,
ranking ρ, best true return, selected regret, and ensemble spread where
available.

This phenomenon is already documented on the structured surrogate — A1.2 found
teacher-forced error growing 2.28x from iteration 1 to 30 after the Markov
repair, with the predicted-safest decile truly colliding 62% of the time
(registered branch C). Gate 3 tests whether it reproduces in the **latent
multi-agent** model, which is the setting the paper is actually about.

**Registered rule.** Growth ratio at iteration 30 over iteration 1, threshold
1.5, matching A1.2 so the two are comparable. CEM-versus-MPPI separates
optimizer aggressiveness from model error: both failing implicates the model or
the scoring interface; MPPI surviving implicates the optimizer.

## Gate 4 - closed-loop MPC

**Fresh, untouched roots.** The Buzz Wire bank has only 16 test-split initial
states and Task A has now spent them, so Gate 4 requires newly collected initial
states. This was already owed from the A1.2 audit.

Arms: zero/random, true-dynamics MPC, H0, H1, H2, optionally H2 + ensemble;
CEM and later MPPI. Identical planner budget, `K=1` cadence, Gate 0's settings.

Metrics: success rate, return, task progress, collision/safety, and selected
regret on logged decisions.

**Note on K3.** H4 has been falsified once already — every learned cell scored
0/32 while the true-dynamics planner reached 25/32. That was on a pipeline whose
upstream links were later shown broken, so it is not dispositive; it is the
reason the gates now run in order.

## What is deliberately not done yet

* **No planner attached to the current models.** See the precondition.
* **No ensemble** until the deterministic baseline is understood.
* **No MPPI** until CEM has run on the same frozen model.
* **No end-to-end `R`/`V` training** until the frozen-latent version reports.
* **T-A3 ordering** stays gated on an arm passing T-A2.

## Artifacts

| Gate | Entry point | Launcher | Status |
|---|---|---|---|
| 0 | `examples/world_model/closed_loop.py` | `scripts/slurm/buzz_wire_control.sbatch` | passed (K4) |
| 1, `h=1` | `examples/world_model/counterfactual_fidelity.py` | `scripts/slurm/ta2_counterfactual_fidelity.sbatch` | job 1502 |
| 1, `E_CF(h)` | same | `scripts/slurm/ta2c_horizon_fidelity.sbatch` | **job 1505 running** |
| 1, localization | `examples/world_model/counterfactual_localization.py` | `scripts/slurm/ta2b_localization.sbatch` | **job 1503 running** |
| 2 | to be implemented | — | blocked on the precondition |
| 3 | partially exists (`planner_tail_failure.py`, frozen to the structured surrogate) | — | blocked on Gate 2 |
| 4 | `closed_loop.py` + a fresh root collection | — | blocked on Gate 3 |

## Results

*(none yet past Gate 1; this note is registered ahead of its runs)*
