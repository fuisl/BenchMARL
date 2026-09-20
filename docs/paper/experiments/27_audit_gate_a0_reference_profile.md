# Audit Gate A0: parallel LeWM reference profile

Status: steps 1--3 implemented and audited; the single-seed sanity run
(job 1469) passed. The scheduler, action-interface and external-conformance
items from the 2026-09-20 review are closed against primary sources. No sweep,
no comparison, and no controller evaluation authorized. Updated: 2026-09-20.

## Purpose

The historical `MultiAgentWorldModel` is retained as `legacy_compact` so every
existing checkpoint and result remains reproducible.  A separate
`lewm_reference` profile establishes the architecture and training recipe that
the pinned LeWM revision uses, except for the adaptations registered below.
This is a compatibility repair, not a new result and not a reinterpretation of
the old latent experiments.

## Step-1 contract

| Parameter | Value | Label |
|---|---:|---|
| latent width | 192 | R |
| temporal history | 3 | R |
| prediction length | 1 | R |
| predictor depth | 6 | R |
| attention heads | 16 | R |
| head dimension | 64 | R |
| predictor FFN | 2048 | R |
| dropout | 0.1 | R |
| projector | 192-2048-BN-GELU-192 | R |
| prediction projector | 192-2048-BN-GELU-192 | R |
| AdamW learning rate | 5e-5 | R |
| weight decay | 1e-3 | R |
| epochs / batch / gradient clip | 100 / 128 / 1 | R |
| CUDA precision | bf16 | R |
| scheduler | 1% linear warmup, then cosine to zero, stepped per optimizer update | R |
| action normalization | z-score fitted on valid training actions | R |
| vector MLP encoder | task vector replaces image ViT | MA |
| per-agent tokens and cross-agent conditioner | multi-agent extension | MA |
| reward/termination readout | frozen-dynamics diagnostic departure | D |
| action block and task observation | Buzz Wire-specific | T |

`R` is reference matched, `MA` is a necessary multi-agent/modality adaptation,
`D` is a diagnostic departure, and `T` is task-specific.

Training consumes the first exact four-frame window of each existing offline
snippet.  Frames 0--2 and actions 0--2 are predictor context; frames 1--3 are
the shifted targets.  The training loader drops its incomplete final batch,
while validation does not.

## Step-2 temporal and SIGReg contract

The reference controller now plans from an actual rolling context rather than
reconstructing history from one current frame:

| decision | observation history | executed action history |
|---:|---|---|
| 0 | `[o0, o0, o0]` | `[0, 0]` |
| 1 | `[o0, o0, o1]` | `[0, a0]` |
| 2 | `[o0, o1, o2]` | `[a0, a1]` |

Each frame is encoded independently. Candidate action `a_t` is appended to the
two executed blocks, the predictor consumes the three aligned positions, and
only the latest predicted position is shifted into the next context. Reference
MPC therefore requires `K=1`: every executed block must be followed by a real
observation before replanning. Passing a single observation to either reward or
goal planning now fails loudly instead of silently reverting to the historical
one-frame rollout.

SIGReg is applied after the reference projector while preserving time. The
registered multi-agent default is `(T,B*N,D)`, treating agents as population
members at each temporal position. The diagnostic alternative applies
`(T,B,D)` independently per agent and averages the losses. With one agent, the
default is exactly the pinned `(T,B,D)` call. Incomplete sequences are excluded
from SIGReg to keep a rectangular population, but their valid transitions are
still retained by the masked prediction loss. Legacy checkpoints retain their
historical `(1,B*T*N,D)` reduction and cannot opt into the reference modes.

Conformance tests pin the episode-start history, temporal/action window shifts,
single-agent SIGReg call, both multi-agent SIGReg choices, planning input
rejection, and the MPC executed-action hook. CPU tests are mandatory and the
planning path is also smoke-tested on CUDA when available.

## Step-3 reference-interface closure

Three fidelity items were raised against steps 1--2. All three were resolved
against the pinned sources rather than against our own restatement of them:
`stable_pretraining` v0.1.7 (tag commit `bce7c8b3`, 2026-05-17, the newest
release when LeWM `8edfeb33` was written) and LeWM `8edfeb33` itself. The two
files the conformance tests depend on are vendored under `test/` with their
upstream MIT licenses.

**Scheduler cadence.** LeWM `train.py` requests
`{"scheduler": {"type": "LinearWarmupCosineAnnealingLR"}, "interval": "epoch"}`.
That interval is never acted on: `stable_pretraining/module.py` sets
`automatic_optimization = False` and its `training_step` calls
`schedulers[idx].step()` immediately after each `opt.step()`, which is the only
scheduler advance in the package. Per-update stepping is therefore the pinned
behaviour and our existing cadence was already correct; the prior review's
inference from the `epoch` label does not hold. The defaults are also now
sourced rather than assumed: `DEFAULT_SCHEDULER_FACTORIES` supplies
`warmup_steps = max(1, int(0.01 * estimated_stepping_batches))`,
`max_steps = estimated_stepping_batches`, `warmup_start_lr = 0.0` and
`eta_min = 0.0`. Because `max_steps` counts optimizer updates, stepping per
epoch would leave the cosine almost undecayed at the end of training.
`test_reference_scheduler_matches_release_at_pinned_lewm_cutoff` runs our
`LambdaLR` factor and the vendored v0.1.7 class over all 250 updates of a
representative extent and requires bit-identical learning rates.

**Action normalization.** LeWM applies `get_column_normalizer` to every
non-pixel loaded column, including `action`, so the action encoder consumes
z-scored actions. `lewm_reference` now fits those statistics, stores them in
the checkpoint as `action_mean`/`action_std`, and applies them inside
`predict`. Because `predict` is the single entry point for both the training
step and every planner path, training and planning cannot diverge. CEM, stored
trajectories and coverage banks stay in native VMAS units; the transform lives
at the model boundary only. Loading a `lewm_reference` checkpoint without
fitted statistics, or with statistics disagreeing with the `state_dict`, fails.

Two conventions follow the reference rather than the blocked tensor. The
statistics are fitted on valid **training** rows only -- the reference script
fits on the full dataset, and this deviation is deliberate, to keep validation
information out of the experiment. Second, LeWM normalizes the raw `action`
column and only afterwards concatenates `frameskip` steps
(`action_encoder.input_dim = frameskip * dataset.get_dim("action")`), so one
statistic per primitive coordinate is shared by every position in the block.
Our blocked `(L,N,block*A)` tensor is block-major, so the fit reduces over
block positions and tiles the result across the blocked width. Fitting the
blocked width directly would have produced `block * A` independent statistics,
which is not the reference convention; a test pins the shared-statistic
behaviour using two block positions with different column means. The corrected
sample denominator of `torch.std` and the reference drop of non-finite rows are
retained.

**External conformance.** The earlier rollout test compared our rollout helper
against a manual loop over our own `predict`, so it pinned our windowing but
proved nothing about the reference. `test_single_agent_output_matches_vendored_pinned_implementation`
now loads identical weights into the vendored LeWM `Embedder`, `ARPredictor`
and projector `MLP` -- upstream structure and `einops` spellings retained -- and
requires bit-identical `N=1` rollouts. Because AdaLN-zero makes the predictor
ignore its conditioning at initialization, the test first perturbs the
modulation weights, so the comparison actually exercises the native-to-
normalized action boundary. Removing the action transform makes it fail.

## Step-4 single-seed sanity run (job 1469)

One seed, one regime, no sweep and no comparison. The question is only whether
the repaired reference implementation trains sensibly in practice. The
`legacy_compact` versus `lewm_reference` comparison is a separately registered
step and was deliberately not run here.

Buzz Wire bank `outputs/buzz_wire_1196/data`, `correlated` regime, seed 4100,
683 training sequences at batch 128 with `drop_last`, so 5 updates per epoch
and 500 dynamics updates in total (5 warmup). Completed in 68 s on one
`3g.20gb` MIG slice. The pinned-conformance tests ran as a preflight gate
inside the allocation and passed, so no data was produced by an implementation
that had drifted from the vendored sources. Artifacts:
`outputs/a0_reference_sanity_1469/`.

| Check | Result | Verdict |
|---|---|---|
| dynamics loss, epoch 0 -> 99 | 5.919 -> 0.593 (min 0.569 at 94) | converged |
| prediction term | 0.263 -> 0.074 | converged |
| SIGReg term | 62.85 -> 5.78 | converged |
| gradient norm | 11.3 -> 1.48, no spikes | stable |
| non-finite losses | none in 150 epochs | pass |
| latent variance | 0.885 | no collapse |
| effective rank | 42.4 of 192 | highest recorded in this repository |
| AdaLN gate scale | 0.00284 | near the top of 738 historical runs |
| checkpoint reload difference | exactly 0.0 | pass |
| termination head validated | true | pass |

The AdaLN-zero weight norm is only a proxy, so action conditioning was also
measured directly on the trained checkpoint. Re-sampling the action block while
holding the latent context fixed moves the prediction by 0.68 latent standard
deviations, and replacing it with the null action moves it by 0.51. Perturbing
agent 0's action alone moves agent 0 by 0.606 and agent 1 by 0.219, so the
relational conditioner carries real cross-agent coupling. The predictor is
action-conditioned and is not the identity.

The reward readout did not learn: relative error 1.008, against a target
variance of 7.86. This is **not** an A0 regression. Across 240 historical Buzz
Wire runs the same quantity has median 1.003 and range 0.992--1.021, and the
failure of the reward interface on true latents is the registered finding of
[`22_decision_information_localization.md`](22_decision_information_localization.md).
The readout is labelled `D`, a diagnostic departure, in the step-1 contract; it
is not a reference-matched component and this run does not change its status.

Verdict: the repaired reference implementation trains sensibly. A0 is closed
for implementation purposes. This run is a sanity check, not evidence about
latent world-model quality, and must not be cited as a comparison.

No controller evaluation, sweep, or between-profile comparison was launched.

Pinned fixtures: `test/_lewm_pinned_8edfeb33.py`,
`test/_stable_pretraining_pinned_v017.py`.
Implementation: `examples/world_model/models.py`,
`examples/world_model/model_input.py`, `examples/world_model/mpc.py`,
`examples/world_model/plan_ranking.py`, `examples/world_model/goal_planning.py`,
and `examples/world_model/train.py`. Configuration:
`benchmarl/conf/world_model_reference.yaml`.
