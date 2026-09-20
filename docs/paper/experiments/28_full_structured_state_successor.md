# Audit Gate A1: full structured-state successor

Status: A1.1 paired-snapshot sufficiency evaluation complete; no coverage
collection, model training, or control run performed. Updated: 2026-09-20.

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

## A1.1 paired-snapshot evaluation

The audit ran from clean commit `21a5e3a` with PyTorch 2.7.1 and VMAS 1.5.2.
Six deterministic batches used seeds 9700--9702 after five random warm-up steps
and 9710--9712 after twenty steps. Of 384 requested roots, 382 remained live and
entered the paired intervention test. All result files are under
`outputs/state_sufficiency_a1_20260920/` and contain the source hash, clean git
status, versions, configuration, and per-intervention measurements.

For each root, the evaluator duplicates the complete simulator snapshot,
changes one omitted field in the second copy, applies the identical continuous
joint action, and compares the next `full32` state, team reward, collision, and
termination.

| Intervention | Reachable status | Maximum next-state difference | Outcome |
|---|---|---:|---|
| prior agent force, both agents | naturally nonzero (max 0.9997) | 0 | overwritten by the new action |
| prior agent torque, both agents | invariant zero in all 382 states | 166.667 when forced to 0.75 | off-manifold counterexample only |
| position/collision reward caches | reset every reward call | 0 | no effect |
| collision flag cache | reset every reward call | 0 | no effect |
| position-shaping cache | exactly ball-goal distance; residual 0 | 0 | corrupting it changes team reward by 1.0 |
| local step-counter change | naturally external | 0 | no local physical or reward effect |
| counter set to timeout boundary | naturally external | 0 | termination differs for 100% of pairs |
| linkage velocity positive control | represented by `full32` | 1.152 | input and successor both distinguish it |

The linkage positive control also changed collision/termination for 4.76% of
pairs and changed team reward by as much as 20.001, confirming that the omitted
linkage state responsible for the `legacy14` alias is now exposed.

### Scoped verdict

`full32` passes the tested **reachable one-step physical-transition** contract
under Buzz Wire's standard action interface. No tested omitted field that can
vary on reachable trajectories changed the next physical state. The nonzero
torque intervention does change physics, but agent torque remained exactly zero
on every reachable state and is not controlled by the task action interface.

It does **not** constitute the full episodic Markov state by itself. Timeout
termination depends on the external environment step counter. Controllers and
evaluations must therefore carry remaining episode time separately, or state
explicitly that the learned model covers physical/collision dynamics rather
than timeout prediction. The shaping cache is not an independent state variable
on reachable trajectories because it is exactly recoverable from the ball and
goal, but arbitrary corrupted snapshots can change raw simulator reward.

## Validation boundary

The implementation tests exact field order, preservation of both linkage
bodies through blockification, profile-derived transition-head dimensions,
finite MPC costs for both profiles, legacy checkpoint compatibility, and
rejection of a full-state checkpoint by frozen Gate 5. The paired counterexample
also verifies that `full32` distinguishes the two states that `legacy14`
aliases.

This removes the demonstrated linkage-state alias and passes the registered
omitted-field intervention suite. It does not prove universal sufficiency over
all impossible or corrupted simulator snapshots. Before control, register a
new full32-specific held-out diagnostic, including the external episode-clock
contract, without selecting fields or thresholds from control performance.
