# A1.2: full32 planner-tail localization

Status: design and decision rule registered before any result existed
(commit `1293046`); job 1470 submitted from that commit. Control deliberately
withheld. Updated: 2026-09-20.

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

## Results

Pending job 1470.
