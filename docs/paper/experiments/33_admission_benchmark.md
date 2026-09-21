# Cross-scenario admission benchmark

**Status: registered 2026-09-21, before any scenario has been measured.**
Thresholds and the taxonomy are fixed here ahead of results.

Direction: [`../direction_joint_representation_2026-09-21.md`](../direction_joint_representation_2026-09-21.md).
Prior results this replaces as the main line:
[`30_task_a_counterfactual_fidelity.md`](30_task_a_counterfactual_fidelity.md),
[`32_joint_representation.md`](32_joint_representation.md).

## Why

Buzz Wire's cross-agent effect is large, real, active on 100% of anchors — and
unlearnable, because the observation omits the mediating body (K28, K30). We
spent 48 checkpoints, a horizon sweep and a localization study establishing
that. **The purpose of this benchmark is to never spend that again without
checking first.**

Buzz Wire is therefore retained **scientifically** as the partial-observability
reference and retired **experimentally** as the primary task. Its value in the
paper is the contrast: a framework that explains *why* a world model cannot
learn a counterfactual effect, rather than reporting which network won.

## What runs: one protocol, every scenario, no world model trained

```
scenario -> deterministic anchor bank -> 3x3 joint-action grid
         -> J_own, J_cross, C_mixed, state dependence
         -> A -> O -> H_dense -> H_model -> S information ladder
         -> ADMISSION REPORT -> reject | admit
```

**Banks are reused, not recollected.** All five existing banks
(`buzz_wire_1196`, `transport_data_1190`, `wheel_1205`, `balance_repair_1236`,
`dropout_control_1193`) share `action_block = 5`, `sequence_steps = 25` and the
same seed family (3100-3103). The audit regenerates its own interventions from
restored snapshots, so bank provenance affects *which states* are sampled, not
the protocol. `balance_repair_1236` is used, never `balance_1233`, whose
heuristic branch was Transport's policy.

### Stage 0 — interaction structure

From each restored anchor, hold every action coordinate at a sampled reference
and sweep **both** agents over `{-1, 0, +1}` on one axis:

```math
J_{\rm own}=Y(+,0)-Y(-,0),\qquad
J_{\rm cross}=Y(0,+)-Y(0,-),
```
```math
C=Y(+,+)-Y(+,-)-Y(-,+)+Y(-,-)
```

`C` is the **mixed** second difference. It is exactly zero for any additively
separable response however large `J_cross` is, which is what separates "your
action affects me" from "our actions interact".

**Decomposed by responding body.** `J_ij = ∂Y_i/∂a_j` is defined on one body's
state. Measuring it on the whole concatenated `Y` would fold the intervened
agent's own large response into the cross term and make every task look
coupled. Agent 0 is the reference responder; shared bodies are reported
separately because the jointly controlled object belongs to neither agent.

**State dependence** is the dispersion of the per-anchor effect. A large but
constant effect is uninteresting — the actions-only head learns it.

### `Y` is task-specific by construction

Every body's full `(pos, vel, rot, ang_vel)`, scaled by its training standard
deviation, with columns constant on training data dropped. Wheel's line is
pinned and only rotates, so a position-only target reads exactly zero there
however much it turns — which is how Wheel was previously recorded as having no
interaction (K9). Buzz Wire's ball translates. One task's target must not be
imposed on the others.

### Stage 1 — information sufficiency

One head family, one budget, one selection procedure; only the input differs:

```math
A\rightarrow O\rightarrow H_{\rm dense}\rightarrow H_{\rm model}\rightarrow S,
\qquad
R_{\rm info}(X)=\frac{E_A-E_X}{E_A-E_S}
```

`S` is privileged state, a **diagnostic reference only** — it never enters a
model. `H_model` samples history at the world model's own **block stride**,
which is a different information representation from dense history: dense
windows mostly expose instantaneous velocity, block-strided windows expose
slower coupled motion. G0b measured only the dense form, and its longest window
(`k=12`, stride 1) is *shorter in wall-clock* than the model's own 3-frame
context, so `H_model` is a genuinely new measurement rather than a re-run.

## Registered taxonomy and thresholds

Fixed before measurement. `E` values are already divided by `‖ΔY_true‖`, so a
reference error **is** an inverse resolution.

| Condition | Verdict |
|---|---|
| diagnostic train error `< 0.5 ×` test error while test `> 1` | **undetermined — overfit**; refuse to classify, increase anchors and rerun |
| reference error `> 1/3`, or cross active on `< 50%` of anchors | **weak-interaction control** |
| cross resolvable, but `max R_info(O, H_dense, H_model) < 0.5` | **partially observable** (the Buzz Wire class) |
| cross resolvable and recoverable, but `C` at the floor | **observable additive** |
| cross resolvable, recoverable, `C` above the floor | **ADMIT** |

Only **ADMIT** scenarios proceed to H0/H1/H2 training. Two admitted scenarios
with different mechanics are preferred over choosing one: a single positive task
is a case study, two begin to support the general claim.

## Stages 3-4, registered but not built

**Stage 3.** On every admitted scenario, the identical factorial
`{H0,H1,H2} × {full, restricted coverage} × seeds`, with no per-task
architecture changes beyond unavoidable input/output dimensions. The restricted
regime must **preserve individual action marginals** while removing particular
joint combinations (e.g. suppress `(+,-)` and `(-,+)` while keeping plenty of
each sign per agent), or counterfactual generalization is confounded with
ordinary action extrapolation.

**Stage 4.** One common evaluation hierarchy, in order: ordinary prediction →
`E_CF` → fixed-plan ranking → selected regret → optimized planning. **No CEM
initially**: every model scores the identical frozen candidate set
`A(s) = {A^(1)…A^(K)}` and the simulator evaluates exactly those, so

```math
R_{\rm sel}=J(A_{\arg\min \hat J})-\min_k J(A_k)
```

isolates whether model quality changes decisions. CEM comes afterwards, and the
existing optimizer-tail work (A1.2 branch C) then attaches to this line rather
than competing with it.

The headline across tasks is whether ordinary prediction error explains selected
regret, or whether counterfactual fidelity does.

## Open discrepancy that must be resolved before the numbers are trusted

**The built-in positive control currently FAILS.** This note registered that
Buzz Wire must classify as `partially_observable` — "if the protocol does not
recover that, the protocol is wrong, not Buzz Wire." On the rebuilt
cross-specific ladder it classifies as **`observable_additive`**, because the
cross-agent effect now looks *recoverable*: `R_O = 0.674` on axis 0, against
**0.016** for the same task in G0b.

Two design differences could produce that, and they are not separated:

1. **How the head is fitted.** T-A2b, G0 and G0b all fit one head to the **full**
   `ΔY` and then *scored* it on a column subset. This benchmark fits a head to
   the cross columns **only**. Predicting the whole next-state response and
   reading off the cross part is a harder, worse-posed problem than predicting
   the cross part directly — the shared-body response is 7.43 against a 2.24
   cross term, so the pooled head spends its capacity elsewhere. **If this is
   the dominant cause, G0b's recovery fractions understate what the observation
   carries, and K29/K30 need revising.**
2. **Where the non-intervened action sits.** G0b sampled agent 0's action from
   the reference distribution; the 3x3 grid pins it at the midpoint. That
   changes the conditional being measured.

`Y` also differs — full informative body state here, MOTION-only there — though
on Buzz Wire the agent columns coincide (rotation is constant and dropped).

**Registered resolution, before the five-scenario run is interpreted:** rerun the
G0 ladder with a cross-only fitted head on the G0 intervention design. That
isolates (1) from (2). Until then:

* **K29 and K30 are provisionally in doubt**, not withdrawn. Their *relative*
  orderings were measured under one consistent instrument and are unaffected;
  the *absolute* recovery fractions are what the discrepancy touches.
* **No scenario's admission verdict should be acted on** — the Buzz Wire
  classification is the control that says whether the instrument is calibrated,
  and it is currently disagreeing with a prior measurement of the same task.

This is recorded before the run rather than after, so the outcome cannot be
re-narrated.

## Results

*(none yet — registered ahead of its runs)*
