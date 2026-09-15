# Stage 1 — rescoring the existing checkpoints

2026-09-15, job 1217. Follows the Stage 0 measurement repairs in `c1e27eb` and
the [audit](../audit_2026-09-15.md). No dynamics were trained: every model here
is one already on disk from jobs 1194, 1196, 1201 and 1205.

## 1. The readout anti-correlation was the terminal-mask bug

Job 1203 reported Buzz Wire's learned readout at Spearman ~-0.25 against the
simulator's own plan costs, *given the simulator's own latents*. That was read as
evidence that the reward is unrecoverable from agent observations, because it
depends on a ball no agent observes — a task property rather than a model
deficiency. Those readouts were trained under a mask that discarded 79% of
training terminations and every -10 collision penalty with them.

Refitting the readouts on frozen dynamics, changing nothing else:

| regime / model | J_readout before | after |
|---|---:|---:|
| correlated / independent | -0.238 | **-0.011** |
| correlated / joint | -0.249 | **-0.007** |
| correlated / relational | -0.267 | **-0.009** |
| independent / independent | -0.197 | -0.043 |
| independent / joint | -0.213 | -0.045 |
| independent / relational | -0.247 | -0.055 |

Transport is the control: it has **no terminations at all**, so the mask had
nothing to discard there, and its readout moved only marginally — +0.66/+0.68 to
+0.705/+0.726. The reward fit tells the same story. Buzz Wire's relative reward
error went 1.044 to 0.955, with **0 of 48 readouts beating a constant predictor
before and 48 of 48 after**; Transport's went 0.256 to 0.162, already below 1.0
in both cases.

**The partial-observability attribution is retracted.** The negative sign was an
artifact. The corrected readout is approximately zero, not negative: it is
uninformative rather than misleading. Observability remains a live hypothesis
consistent with a null correlation, but the evidence previously cited for it —
the negative sign — does not exist. Testing it properly is Stage 2.

Reward-based plan ranking on Buzz Wire is **not** rescued by the fix: `J_model`
stays at 0.015-0.061. A plan cost built on an uninformative readout is
uninformative.

## 2. C7 survives its own confound correction

The published ratio divided one model's intervention response by `independent`'s
inertia — two different learned latent spaces. Each model is now scored against
its own no-response floor, which is a genuine floor for `independent` because its
response equals its own inertia by construction.

| bank | regime | independent | joint | relational |
|---|---|---:|---:|---:|
| held-out (1201) | correlated | 1.000x | 0.673x | **0.470x** |
| held-out (1201) | independent | 1.000x | 0.499x | **0.488x** |
| development (1196) | correlated | 1.000x | 0.813x | **0.734x** |
| development (1196) | independent | 1.000x | 0.889x | **0.741x** |

A **shared physical probe** is reported alongside, because a latent ratio cannot
say whether a model knows *which* anchors carry a large effect. It ranks each
model's predicted response magnitude against the simulator's true
observation-space response — one common target for every model:

| bank | regime | independent | joint | relational |
|---|---|---|---:|---:|
| held-out | correlated | constant 0 | 0.327 | **0.479** |
| held-out | independent | constant 0 | 0.427 | **0.470** |
| development | correlated | constant 0 | 0.173 | **0.301** |

`independent` predicts exactly zero response at every anchor, so its rank
correlation is undefined rather than poor, and is reported as such.

## 3. Goal A/B/C localises the remaining failure

A is the simulator's physical goal distance over corrected endpoints; B adds the
learned representation over **true** endpoints; C adds the learned rollout.
A-to-B is representation geometry, B-to-C is what prediction costs.

| task | B vs A | C vs B | B regret | C regret | random regret | B top | C top |
|---|---:|---:|---:|---:|---:|---:|---:|
| Buzz Wire | **0.534** | 0.380 → **0.444** | 0.0127 | 0.026-0.030 | 0.038 | 0.41 | 0.12 |
| Transport | 0.372 | **0.82** | 0.0565 | 0.062 | 0.100 | 0.15 | 0.12 |

Read across, the bottleneck is task-specific and is **not scoring** — this path
never touches the reward readout:

* **Buzz Wire: the rollout.** The representation orders physical goals at 0.534,
  and the rollout collapses top-plan agreement from 41% to 12%.
* **Transport: the representation.** The rollout preserves ordering almost
  perfectly (0.82) and the representation is the limit at 0.372.

`B vs A` is identical across all three architectures on both tasks, to three
decimal places. Representation geometry is architecture-independent here.
Relational helps only on Buzz Wire and only at the rollout stage — exactly where
its prediction advantage was measured — improving `C vs B` from 0.380 to 0.444
and giving the lowest selected-plan regret.

## What this changes

| Claim | Status |
|---|---|
| Relational captures cross-agent effects | **Holds**, and now has a physically interpretable form |
| Buzz Wire's readout is anti-correlated because the ball is unobserved | **Retracted** — the sign was the terminal-mask bug |
| The chain breaks at the readout | **Wrong for Buzz Wire**: it breaks at the rollout; the readout is merely uninformative |
| The chain breaks at the rollout on Transport | **Holds**, and is now measured directly as B vs C |

## Next

Stage 2 as the audit orders it: isolate information on Buzz Wire — current input
against actual joint history against explicit ball/physical state, one fixed
architecture, one seed first. That is the test the retracted claim was standing
in for, and the A/B/C result makes it sharper: the question is why a
representation that orders goals at 0.534 cannot be rolled forward without
losing most of that ordering.
