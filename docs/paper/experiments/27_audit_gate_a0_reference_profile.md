# Audit Gate A0: parallel LeWM reference profile

Status: steps 1--2 implemented; code review pending; no scientific run
authorized. Updated: 2026-09-20.

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
| scheduler | 1% linear warmup, then cosine to zero | R |
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

No training, controller evaluation, or scientific experiment was launched for
this implementation step.

Implementation: `examples/world_model/models.py`,
`examples/world_model/model_input.py`, `examples/world_model/mpc.py`,
`examples/world_model/plan_ranking.py`, `examples/world_model/goal_planning.py`,
and `examples/world_model/train.py`. Configuration:
`benchmarl/conf/world_model_reference.yaml`.
