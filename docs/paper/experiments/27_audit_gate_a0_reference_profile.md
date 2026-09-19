# Audit Gate A0, step 1: parallel LeWM reference profile

Status: implementation only; no scientific run authorized. Date: 2026-09-19.

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

## Deliberately deferred to step 2

This profile bounds autoregressive prediction to three positions, but planning
still starts from one current frame.  Step 2 must add the real rolling
observation/action buffer before any controller is called LeWM-faithful.
Likewise, SIGReg still uses the legacy population reduction until step 2
registers and tests `(T,B,D)` and `(T,B*N,D)` semantics.  No A0 experiment may
run before those two contracts are complete.

Implementation: `examples/world_model/models.py` and
`examples/world_model/train.py`. Configuration:
`benchmarl/conf/world_model_reference.yaml`.
