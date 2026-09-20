# Audit Gate A1: full structured-state successor

Status: implementation and contract tests only; no collection, training, or
control run authorized. Date: 2026-09-20.

## Why this profile exists

The job-1331 structured surrogate used a 14-D state containing two agents'
positions and velocities, the ball position and velocity, and the goal. A
paired-snapshot counterexample proved that state is not Markov: holding all 14
coordinates and the action fixed while changing only an omitted linkage-body
velocity changes the next 14-D state.

The historical `legacy14` profile remains loadable so jobs 1331--1337 are
reproducible. It must be described only as a privileged diagnostic. The new
`full32` profile removes this known alias by retaining every recorded dynamic
body coordinate:

| Component | Bodies | Coordinates per body | Width |
|---|---:|---|---:|
| agents | 2 | position, velocity, rotation, angular velocity | 12 |
| ball and linkage bodies | 3 | position, velocity, rotation, angular velocity | 18 |
| goal | 1 | position | 2 |
| total | | | 32 |

The learned transition predicts the first 30 dynamic coordinates and carries
the two goal coordinates unchanged. Planning progress uses the registered ball
slice and the final goal slice, so no legacy index is reused accidentally.

## Compatibility boundary

State width determines the model profile. Historical 14-D checkpoints infer
`legacy14` even though they predate explicit profile metadata. New checkpoints
record `state_profile`. A disagreement between metadata and tensor shapes fails
loading. The frozen job-1331 Gate-5 implementation explicitly rejects `full32`:
the successor requires a separately registered diagnostic instead of silently
changing the state under an old protocol.

Coverage manifests now record profile, state width, dynamic width, and field
order. `structured_surrogate.py collect --state-profile full32` is the intended
future collection entry point. The default remains `legacy14` to prevent an old
command from silently producing an incompatible bank.

## Validation boundary

The implementation tests exact field order, preservation of both linkage
bodies through blockification, profile-derived transition-head dimensions,
finite MPC costs for both profiles, legacy checkpoint compatibility, and
rejection of a full-state checkpoint by frozen Gate 5. The paired counterexample
also verifies that `full32` distinguishes the two states that `legacy14`
aliases.

This removes one demonstrated source of state aliasing; it does not by itself
prove universal Markov sufficiency. Before a scientific control run, validate
one-step replay on paired full simulator snapshots and register a new held-out
diagnostic without selecting its state fields from control performance.

