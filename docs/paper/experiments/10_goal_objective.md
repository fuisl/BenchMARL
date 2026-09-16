# The goal objective: three designs, three failures, one partial pass

2026-09-16. Jobs 1227, 1229, 1230, 1231, and the re-reading of 1218/1221/1222/1226
that started it. No model was trained for any of this: every planner here uses the
**true simulator** as its dynamics. That is the point — if a planner with perfect
dynamics cannot reach a goal without wrecking the task, no learned model will, and
there is nothing for a world-model result to mean.

Buzz Wire throughout, except where Transport is named. Buzz Wire is the task that
can be controlled from its anchors; Transport cannot (§1.2).

---

## 1. What was wrong with the original goal

### 1.1 The goal was a random-walk endpoint

`closed_loop.achieved_goals` took the goal from the terminal observation of
**candidate 0 of a uniform random action bank**, `goal_offset=5` blocks ahead —
25 primitive steps. LeWM's own convention, and reachable by construction, which is
why it was chosen.

Jobs 1218/1221/1222/1226 scored every learned planner against it. Re-reading those
runs on the task's own distance rather than the goal distance:

| Task | goal oracle: goal reached | goal oracle: task distance | random: task distance |
|---|---:|---:|---:|
| Buzz Wire | **20/20** | 1.0937 | 1.0852 |
| Transport | **20/20** | 0.8993 | 0.8988 |

A planner that reaches its goal in every episode ends at, or just behind, where
random behaviour ends. **The objective is orthogonal to the task.** Every
"learned planners close 30% (Buzz Wire) / 41% (Transport) of the random →
goal-oracle gap" number is therefore a fraction of nothing, and is withdrawn.

### 1.2 Selecting a better random plan does not repair it

The obvious fix — keep rolling all 300 candidates, but pick the endpoint that gets
*furthest on the task* — was implemented, measured and discarded:

| Task | best of 300 | candidate 0 | mean of 300 |
|---|---:|---:|---:|
| Buzz Wire | 1.0486 | 1.0908 | 1.0936 |
| Transport | **0.8979** | 0.8991 | 0.8990 |

Random action sequences make **almost no task progress**, so no goal drawn from
one can carry task content however it is selected. On Transport the best of 300
improves by 0.0012.

This also produced the Transport result that removed it from the control claim.
Rolling the strongest policies from the evaluation anchors:

```
vmas/transport   start 0.9006    zero-action -> 0.8994 (t=100)
                                 VMAS heuristic -> 0.8916 (t=25), 0.8687 (t=100)
vmas/buzz_wire   start 1.0950    zero-action -> 1.0948 (t=100)
```

The best controller that exists for Transport closes **3.5% of the starting
distance in a full episode**. Transport's universal 0/20 native success is a
property of the task and the anchors, not evidence about any world model.

## 2. Design 2 — the goal from a competent controller (job 1229)

Take the goal from where the strongest controller each task has actually ends up:
the scenario heuristic where one exists, the reward oracle otherwise. On Buzz Wire
that oracle moves the ball 1.0950 → 0.5567 and succeeds 5/20 — genuinely
competent.

| policy | goal reached | goal dist | task dist | collision |
|---|---:|---:|---:|---:|
| random | 0/20 | 0.8023 | 1.0852 | 0.70 |
| do nothing | 0/20 | 0.7826 | 1.0948 | 0.05 |
| reward oracle *(goal source)* | 20/20 | 0.0000 | **0.5567** | 0.35 |
| **goal oracle** | 12/20 | 0.1560 | **0.9536** | **0.90** |

**Worse than the first design.** Pursuing the goal collides in 90% of episodes
against random's 70%, and finishes further from the task than doing nothing. The
job was cancelled 22 minutes in rather than spend eight hours measuring learned
models against it.

Two causes were separable, so each got its own gate.

### 2.1 Crash-state goals

The oracle collides in 7 of 20 episodes, so 7 goals were the states it crashed in.
An intermediate fix — take the observation at each episode's *best* task frame
rather than its terminal frame — turned out to be **inert**: `best == final` for
**20/20** episodes, because the oracle approaches monotonically until it collides.
Its best frame *is* its crash frame.

### 2.2 Velocity matching

A full-observation goal says *arrive here carrying this velocity*. On Buzz Wire
matching the velocity means driving hard through a narrow corridor.

## 3. Gate A — drop velocity from the goal distance (job 1230). Rejected.

`--goal-metric position`, velocity components zeroed in both `EpisodeStats` and
`GoalDistance` so the oracle still bounds what is reported.

| policy | goal dist | task dist | collision |
|---|---:|---:|---:|
| random | 0.8192 | 1.0852 | 0.70 |
| do nothing | 0.7952 | 1.0948 | 0.05 |
| reward oracle | 0.0000 | 0.5194 | 0.20 |
| **goal oracle** | 0.2152 | **0.9841** | **0.85** |

Essentially unchanged from 0.9536 / 0.90. **The collisions are not a
velocity-matching artefact.** Hypothesis rejected.

## 4. Gate B — goals only from solved episodes (job 1231). Partial pass.

Run the reference controller over a 4× pool and keep only states it *solved*; the
goal is the observation there. The ball is on target and nothing has collided, so
a crash state cannot be selected.

The oracle solved **20 of 80** pooled states. Goals land at task distance
**0.0018** — actual task-goal states.

| policy | native success | goal dist | task dist | collision |
|---|---:|---:|---:|---:|
| random | 0/20 | 0.7960 | 0.5029 | 0.65 |
| do nothing | 0/20 | 0.7572 | 0.5069 | 0.10 |
| reward oracle | **17/20** | 0.1943 | **0.0772** | 0.15 |
| **goal oracle** | **2/20** | 0.1499 | **0.3658** | **0.85** |

**What passed.** The goal oracle finally beats random and do-nothing on the task
(0.3658 against 0.5029 / 0.5069) and scores **2/20 native successes — the first
the goal objective has ever produced**.

**What failed.** It still collides 85% against random's 65%, against a stated
pass condition of "no more than random".

**Selection effect, to be reported wherever this table is.** These are the states
a competent controller can solve, so random sits at 0.5029 rather than 1.0852.
The evaluation set is no longer a random sample of test states.

## 5. The mechanism

Buzz Wire agents observe `[own pos, own vel, own pos − own goal]`. They never
observe the ball, and never observe the wire.

A goal in that space can say **where the agents should be**. It cannot say **do
not let the ball touch the wire**, because the objective contains no term that
changes when the ball approaches a wall. Reward planning sees the collision
penalty; goal planning is structurally blind to it. On identical states the reward
oracle scores 17/20 and the goal oracle 2/20.

This is a property of the *objective in a partially observed task*, not a tuning
failure and not a defect of any learned model. It generalises: any
observation-space goal on a task whose failure mode is unobserved has this hazard,
and the diagnostic is cheap — score the goal oracle on the task's own metric and
on its collision rate, not only on goal distance.

## 6. What this changes

* Link 3 of the project's chain is **not measured**, rather than measured and
  failed. See [`../outline.md`](../outline.md) §2.5.
* The control tables need three columns that were missing: the task's own
  distance, the collision rate, and a **do-nothing** baseline. Do-nothing beats
  random on goal distance in every table above — the audit's contract item 5,
  never run before today, and it would have caught design 1 immediately.
* Buzz Wire cannot carry a goal-conditioned control claim at all, and Transport
  cannot carry a control claim of any kind. Balance was added to the suite for
  exactly this reason — see [`README.md`](README.md), job 1233.

## 7. Reproduce

```
# design 1, for comparison
python -m examples.world_model.closed_loop <runs> --data <bank> \
    --goal-source arbitrary --max-runs 0 --output refs.json
# gate A
... --goal-source reference --goal-metric position
# gate B
... --goal-source success --state-pool 80
```
