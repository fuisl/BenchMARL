# M5 link 1 — plan ranking

2026-09-14. First of M5's four links: do better predictions rank candidate plans
better? This is a partial M5 record. Closed-loop control and the counterfactual
gap `G_CF` are not in it.

**Headline: the prediction advantage does not propagate to ranking, and on
Transport most states are not rankable at all.**

## Protocol

Same planner, same objective, different dynamics. For each evaluation state one
shared candidate set is scored twice: by the simulator (exact cost, via the M2
`oracle_plan_costs` path) and by each learned model (latent rollout plus the
reward readout). The model objective mirrors the oracle's exactly,
`J = -sum_t sum_i r_i,t`, so the only thing differing between the two scores is
the dynamics.

States are M3 test anchors — held out from training by episode-level split.
Candidates are drawn uniformly in native action bounds, which for a
correlated-regime model is deliberately outside its action coverage; that is the
planner's real query, not a nuisance. Models are the 48 Transport checkpoints
from job 1194 (3 baselines x 2 regimes x 8 seeds).

## Most states cannot be ranked

Spearman is undefined when every candidate ties, and on Transport that is the
common case:

- The **M2 candidate banks are entirely unrankable**: across all 20 development
  states, all 300 CEM candidates have exactly **one unique cost** (0.0, standard
  deviation 0). This also settles a question 02_transport_comparisons.md left
  explicitly open — whether candidates tie at unproductive states. At the initial
  solve they do, exactly.
- On M3 test anchors it is better but still a minority: **7/24 sampled states
  (29%)** have true costs that vary at all, over a range [-3.71, +0.32].

Only states with varying true cost are scored, and that fraction is reported
with the result. This is a property of the measurement, not a filter applied to
flatter the numbers: rank correlation on a constant vector does not exist.

## Result

24 states, 32 candidates each, 7 rankable, 8 seeds.

| Regime | Baseline | Spearman | Selected-plan regret |
|---|---|---:|---:|
| correlated | independent | 0.068 | 0.753 |
| correlated | joint | 0.134 | 0.634 |
| correlated | relational | 0.097 | 0.666 |
| independent | independent | 0.012 | 0.761 |
| independent | joint | 0.054 | 0.721 |
| independent | relational | 0.033 | 0.744 |

Paired against `independent`, per seed:

| Regime | Baseline | Metric | mean | 95% CI | seeds better |
|---|---|---|---:|---:|---:|
| correlated | joint | Spearman | +0.066 | [-0.059, +0.180] | 5/8 |
| correlated | joint | regret | -0.119 | [-0.247, -0.003] | 5/8 |
| correlated | relational | Spearman | +0.029 | [-0.046, +0.100] | 6/8 |
| correlated | relational | regret | -0.087 | [-0.165, +0.001] | 6/8 |
| independent | joint | Spearman | +0.042 | [-0.103, +0.172] | 5/8 |
| independent | joint | regret | -0.040 | [-0.169, +0.085] | 4/8 |
| independent | relational | Spearman | +0.021 | [-0.073, +0.103] | 5/8 |
| independent | relational | regret | -0.017 | [-0.104, +0.075] | 5/8 |

**No model ranks plans usefully.** Spearman is 0.01–0.13 everywhere, close to no
rank correlation. Every paired interval spans zero except `joint` regret under
correlated actions, which only marginally excludes it, and the per-seed win
counts (4–6 of 8) are consistent with coin flips. Relational is not even the best
ranker; `joint` leads on both metrics, not significantly.

So the chain breaks at link 2. M4's multi-step prediction advantage — 12%,
8/8 seeds, intervals clear of zero — **does not survive into the
decision-relevant metric**. That is §7.5's "where the chain breaks" material and
it should be reported as such rather than buried.

## What limits this result

1. **Seven rankable states** and 32 candidates per state. The rankable fraction
   is the binding scarcity, not seed count; more seeds will not fix it.
2. **The readout is half-unvalidated.** Predicted costs come from the reward head
   (which works on Transport, R^2 ~ 0.74) and the termination head (which has
   never seen a positive example on either task). A short-horizon plan cost is
   dominated by the reward term, but this is an assumption, not a check.
3. **Ranking may be near-degenerate even where defined.** 34% of candidates carry
   nonzero cost; the rest tie at zero, so much of each ranking is decided among
   tied plans.

## Buzz Wire — where the measurement is well posed (job 1200, stage 1)

Transport's ranking result could be dismissed as an artefact of its degenerate
cost landscape. Buzz Wire removes that excuse: **117/117 states are rankable**,
with median per-state cost standard deviation 7.39 against Transport's exactly
0.000000, and the same models there capture 26% of the cross-agent effect
(see [counterfactual prediction](05_counterfactual_prediction.md)).

117 states, 64 candidates, 8 seeds.

| Regime | Baseline | Spearman | Selected-plan regret |
|---|---|---:|---:|
| correlated | independent | 0.0574 | 4.557 |
| correlated | joint | 0.0473 | 5.391 |
| correlated | relational | 0.0594 | 4.151 |
| independent | independent | 0.0452 | 5.069 |
| independent | joint | 0.0430 | 4.856 |
| independent | relational | 0.0519 | 4.855 |

Paired against `independent`, every interval contains zero:

| Regime | Baseline | Metric | mean | 95% CI | seeds better |
|---|---|---|---:|---:|---:|
| correlated | relational | Spearman | +0.0020 | [-0.0246, +0.0279] | 5/8 |
| correlated | relational | regret | -0.4055 | [-1.3248, +0.5989] | 5/8 |
| independent | relational | Spearman | +0.0067 | [-0.0228, +0.0342] | 5/8 |
| independent | relational | regret | -0.2144 | [-0.8339, +0.4029] | 4/8 |

**Ranking fails here too**, and this time it cannot be blamed on the task.
Spearman is 0.043-0.059 for every baseline on a fully non-degenerate landscape,
with models that measurably capture part of the interaction. So the break is
between counterfactual prediction and plan ranking, not upstream of it.

That is a more informative negative than Transport's. Capturing 26% of a
cross-agent effect in latent space evidently does not order 25-step plan costs
correctly, and two plausible reasons are separable by experiment: the remaining
74% of the effect may dominate the cost, or the error may enter through the
reward readout rather than the dynamics. Scoring candidates with the *simulator's*
rewards applied to *predicted* latents would separate them, and costs nothing
beyond a variant of the existing evaluator.

## Why ranking fails — two different causes (job 1203)

A model plan cost passes through two learned stages, so the same flat Spearman
could come from either. Scoring the candidates a third way separates them: the
learned readout applied to the **simulator's own** latents, obtained by rolling
the simulator and keeping its observations at each block boundary.

    J_true     simulator states,  simulator rewards   (ground truth)
    J_readout  simulator states,  learned readout     (readout error only)
    J_model    predicted latents, learned readout     (both errors)

96 states, 48 candidates, 8 seeds, Spearman against J_true:

| Task | Baseline | J_model | J_readout | recovered |
|---|---|---:|---:|---:|
| Transport | independent | 0.051 | **+0.672** | +0.554 |
| Transport | relational | 0.135 | **+0.663** | +0.528 |
| Buzz Wire | independent | 0.051 | **-0.238** | -0.289 |
| Buzz Wire | relational | 0.051 | **-0.267** | -0.317 |

**The two tasks fail for opposite reasons.**

On **Transport** the readout ranks plans well from true latents (rho ~ 0.67), so
the readout is adequate and the **autoregressive rollout** is what destroys the
ordering. Part of that is a train/serve mismatch we introduced: the readout is
fitted on teacher-forced one-step predictions but applied to latents compounded
over five rollout steps.

On **Buzz Wire** the readout is *anti-correlated* (rho ~ -0.25) even given
perfect latents, so no improvement in dynamics could rescue ranking there. The
cause was predicted in advance by [01_protocol.md](01_protocol.md): Buzz Wire's
reward depends on the **ball**, and no agent observes the ball. "The
reward-relevant quantity (the ball's position, and its collisions) is not part of
either agent's observation." A readout over agent observations therefore cannot
represent the quantity it is asked to predict. This is partial observability, not
a model deficiency, and it means Buzz Wire cannot support reward-based planning
until the ball state is added to the observation or the latent.

So Buzz Wire is the right task for the **dynamics** claim (C7) and the wrong one
for the **planning** claim, while Transport is the reverse. That is a task-design
conclusion, and it is sharper than "ranking does not work".

### A defect in the model plan cost, found by this audit

`model_costs` originally discarded the termination head and summed predicted
reward over every block, while the true cost stops accumulating at termination.
Transport contains no terminations so it was harmless there, but 27.8% of Buzz
Wire snippets terminate, so the model over-counted exactly the plans that end
early. The cost now weights each block by the predicted probability of still
being alive at its start, which is the probabilistic form of the oracle's mask.

Corrected numbers are materially unchanged -- Buzz Wire relational 0.0588 against
0.0594, Transport unchanged to three decimals -- so the defect did not produce
the negative result. It is recorded because it was real, not because it mattered.

## What this implies for the task choice

Three separate measurements now point at the same underlying property of
Transport rather than at the models: 0 successes in 660 oracle evaluations,
identically zero reward variation across candidate plans at most states, and a
termination signal with no positive examples. Reward sparsity is the common
cause, and it limits ranking, termination supervision and closed-loop success
simultaneously.

This is now a task-selection question, not a model question, and it belongs in
the plan before more M5 effort goes into Transport. Buzz Wire has a sharp binary
outcome, genuine terminations (wall contact), and an oracle that reaches 11/20
goals — so all three blocked measurements would be available there. The M1 Row 2
argument for a 2-agent task with a small joint-action space also still stands.
Nothing here retracts Transport as the M5 task; it records that the evidence for
switching has accumulated and the decision has not been made.

## Reproduce

```bash
python -m examples.world_model.plan_ranking \
  outputs/interaction_control_1194/transport \
  --data outputs/transport_data_1190 --states 24 --candidates 32
```

Simulator truth is cached to a file on first run, so re-analysis costs only the
model forward passes.
