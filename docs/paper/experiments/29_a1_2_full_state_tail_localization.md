# A1.2: full32 planner-tail localization

Status: complete. Design and decision rule were registered before any result
existed (commits `1293046`, `8f6b9c6`); job 1470 ran from that code. Registered
outcome: **branch C** -- the optimizer tail survives the Markov repair. Control
deliberately withheld. Updated: 2026-09-20.

> **Branch C is deferred, not withdrawn (2026-09-21).** The registered next
> steps -- sampler-matched G6a on `full32`, and the G6c calibration run this
> note argued should not wait -- are Task B work on the structured surrogate.
> [`../direction_2026-09-21.md`](../direction_2026-09-21.md) puts Task A first,
> so both are paused with their registration intact. The G6a implementation is
> already repaired and unrun. Nothing measured in job 1470 changes.

## The question

A1.1 falsified `legacy14`'s Markov assumption: holding all 14 coordinates and
the action fixed while changing only an omitted linkage-body velocity changes
the next state, the collision outcome in 4.76% of paired cases, and team reward
by as much as 20.001. `full32` removes that alias.

Job 1335 (Gate 5) found that teacher-forced error rises from CEM iteration 1 to
30, and read that as the planner driving the surrogate off its training
distribution. That reading was made inside a model we now know is non-Markov,
so the effect could have been state aliasing rather than optimizer-induced
coverage shift. A1.2 asks the one question that separates them:

> Does late-CEM teacher-forced error still grow after removing the demonstrated
> state alias?

## Why the comparison is paired

Nothing upstream of `blockify()` depends on the state profile. The coverage CEM
scores candidates with the latent world model and observations; the oracle
scores them in the simulator. So one collection pass emits both representations
of the *same* transitions, and `collect --state-profile legacy14 full32` refuses
to write unless every non-state column -- action, collision, clearance, team
reward, progress, split, family, source, root and stage -- is bit identical row
for row. This is strictly stronger than two seed-matched collections, and it
means any difference between the arms is attributable to the state
representation alone.

## Arms

| Arm | Profile | Hidden | Parameters | Role |
|---|---|---:|---:|---|
| `full32_h256` | full32 | 256 | 153,633 | the successor state under test |
| `legacy14_h264` | legacy14 | 264 | 153,135 | capacity-matched control (-0.32%) |
| `legacy14_h256` | legacy14 | 256 | 144,399 | job-1331 width, for lineage |

The audit asked for approximate parameter matching rather than a shared hidden
width, because full32's wider input and output add capacity on their own. At a
shared 256 the successor would carry 6.39% more parameters, so `legacy14_h264`
is the primary control and the historical width is retained alongside it so the
job-1331 lineage stays visible.

Three model seeds per arm (9100--9102). CEM populations are recorded at
iterations 1, 5, 10, 20 and 30 and each candidate is scored both teacher-forced
and recursively.

## Protocol separation

`planner_tail_failure` now takes `--protocol`. The frozen job-1331 contract
`gate5_legacy14` still refuses any successor state, exactly as the previous
audit required, so old Gate 5 cannot silently accept full32 and the historical
lineage stays interpretable. `a1_paired` is the registered place where full32 is
measured, and it accepts either profile so that both arms go through one
identical procedure rather than two differently-shaped diagnostics.

The legacy14 coordinate indices are gone. Coordinate groups are derived from the
state profile, so ball, agent and linkage error are reported apart instead of
being scored on whatever happens to sit at index 8:10. `link_body_*` is `None`
throughout a legacy14 arm, which is precisely the omission A1.1 demonstrated.

## What is withheld

Control is withheld by `--skip-control`. This pass is ranking and tail
localization only. No model, seed or objective can be selected on closed-loop
performance, and the held-out roots are not spent before the pre-registered
control run.

## Root set

The 16 held-out roots are the same ones Gate 5 used. This is deliberate and is a
departure from the audit's suggestion of a newly frozen set. A1.2's comparison is
internally paired -- both arms see identical roots, identical data and one
identical CEM procedure -- so root reuse cannot bias the comparison between them;
it can only affect absolute-level claims, which A1.2 does not make. Since the
question is whether *the same* phenomenon persists, measuring it on the same
roots is the sharper test. The Buzz Wire bank has only 16 test-split initial
states, so a genuinely fresh bank requires collecting new initial states. That
is reserved for the control run, where the audit asked for it and where it
matters.

## Registered decision rule

Fixed in `a1_tail_comparison.decide` and tested before the run. Growth is the
ratio of teacher-forced ball-position RMSE at CEM iteration 30 to iteration 1,
at horizon 5. The threshold is 1.5.

| Condition | Branch | Next experiment |
|---|---|---|
| full32 growth >= 1.5 | C | corrected, sampler-matched G6a **on full32** |
| legacy14 grows, full32 <= half its growth, nothing else fires | A | alias was upstream of Gate 5; do **not** run G6a |
| teacher-forced flat but recursive growth >= 1.5 | B | multi-step compounding; G6b |
| state accurate but true collision in predicted best 10% >= 0.25 and exceeds predicted by >= 0.15 | D | event calibration; G6c |
| growth not measurable in some arm | undetermined | report, do not branch |

An arm whose growth cannot be measured reports `undetermined` rather than being
counted as evidence for any branch.

## Results (job 1470)

COMPLETED in 56:53 on one `3g.20gb` MIG slice. The paired collection produced
49,125 transitions in each bank, and the writer verified that all twelve
non-state columns are bit identical row for row. Three model seeds per arm.
Every arm selected the same objective (`probability`), so the decision metrics
below are not confounded by objective choice. Artifacts:
`outputs/a1_tail_localization_1470/`.

### Registered answer

$$E_{\rm TF}^{30}/E_{\rm TF}^{1} = 2.28 \ \ (\text{full32}) \quad\text{vs}\quad 3.11 \ \ (\text{legacy14, capacity matched})$$

Teacher-forced error still grows strongly with CEM iteration after the known
state alias is removed, far above the registered threshold of 1.5. The run
lands in **branch C** against both the capacity-matched and the historical-width
control. G6a fired on 3 of 3 seeds in every arm.

**The optimizer-induced distribution-shift claim is substantially stronger than
it was under job 1335.** It now survives in a model whose state is not known to
be aliased.

### Paired comparison, horizon 5

| Metric | legacy14 h256 | legacy14 h264 | full32 h256 |
|---|---:|---:|---:|
| TF ball RMSE, iteration 1 | 0.0090 | 0.0090 | 0.0079 |
| TF ball RMSE, iteration 30 | 0.0344 | 0.0278 | 0.0180 |
| **TF growth, iter30/iter1** | **3.81** | **3.11** | **2.28** |
| recursive ball RMSE, iteration 30 | 0.1479 | 0.1386 | 0.0911 |
| recursive growth, iter30/iter1 | 7.13 | 6.54 | 4.73 |
| recursive - TF gap at 30 | 0.1206 | 0.1145 | 0.0768 |
| clearance RMSE at 30 | 0.0212 | 0.0170 | 0.0061 |
| link-body RMSE at 30 | -- | -- | 0.1473 |
| plan Spearman at 30 | 0.2226 | 0.2610 | 0.1868 |
| **true collision in predicted best 10% at 30** | 0.3854 | 0.2625 | **0.6229** |
| predicted collision in that same decile | 0.0001 | 0.0001 | 0.0001 |
| collision Brier at 30 | 0.3869 | 0.2593 | 0.6165 |
| selected regret at 30 | 4.60 | 1.68 | 6.70 |

`full_dynamic_coordinate_rmse` is reported per arm but is **not comparable
across profiles**: full32 predicts 30 dynamic coordinates including the linkage
bodies, which are by far the hardest (0.1473), while legacy14 predicts 12. A
larger number there reflects a harder target, not a worse model.

### The alias was real but is not the main driver

full32 is better on every physical quantity. One-step ball error at iteration 30
falls 35%, the recursive-minus-teacher-forced accumulation gap falls 33%, and
clearance error falls 64%. The growth ratio itself falls from 3.11 to 2.28. So
the linkage alias A1.1 demonstrated *was* contributing to the Gate-5 signature.

But it was a contributor, not the cause. A 2.28x late-CEM blow-up remains in a
state with no known alias, on all three seeds. Removing the alias moved the
number roughly a quarter of the way toward 1.0 and nowhere near the registered
repair criterion.

### The surprise: better physics, worse decisions

The arm with the best dynamics has the worst decision quality.

| CEM iteration | 1 | 5 | 10 | 20 | 30 |
|---|---:|---:|---:|---:|---:|
| legacy14 h264, true collision in predicted best 10% | 0.115 | 0.054 | 0.154 | 0.256 | 0.263 |
| full32 h256, same | 0.104 | 0.085 | 0.269 | 0.562 | 0.623 |

Both arms start in the same place and diverge as CEM optimizes. By iteration 30
nearly two thirds of full32's predicted-safest decile truly collides, while its
collision head assigns those same plans a probability of 0.0001. Plan-ranking
Spearman is lower than either legacy arm (0.187 vs 0.261) and selected regret is
four times the capacity-matched control.

The G6c false-safe rule fired on 3 of 3 full32 seeds, against 2 of 3 in both
legacy arms.

A coherent reading is that a more accurate dynamics model lets CEM optimize
harder and push further into the region where the *collision head* is
miscalibrated. The failure moves from the dynamics into the event model rather
than disappearing. That is still optimizer exploitation of model error, which is
consistent with branch C; it just relocates which head is being exploited. This
reading is a hypothesis generated by A1.2, not something A1.2 tested.

### What this does not show

No control was run, so nothing here says whether full32 closes the loop better
or worse. The 16 roots are the longitudinal Gate-5 roots, so these are
comparative numbers, not absolute performance estimates. The collision-head
reading above is a hypothesis.

## Next step

Branch C is the registered outcome: **the corrected, sampler-matched G6a, run on
full32 rather than on legacy14.** The G6a implementation was already repaired
(matched sampler, 50/50 base/augmentation, namespaced roots) and its rerun was
deliberately held until this diagnostic reported.

The false-safe result argues that G6c should not wait for G6a to finish. The
calibration failure is larger in full32 than in the model it replaced, it grows
monotonically with CEM iteration, and it fired on every seed.
