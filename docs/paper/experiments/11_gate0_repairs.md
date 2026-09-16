# Gate 0: three defects, what each one cost, and how they are now caught

2026-09-16. Repairs for the defects reproduced in
[`../review_2026-09-16.md`](../review_2026-09-16.md) §2. Code in commit
`43432b1`; corrected horizon curve as job **1235**; corrected Balance bank as job
**1236**. Contract checks in
[`test/test_world_model_contracts.py`](../../../test/test_world_model_contracts.py).

Each defect produced a number that read as a result. The existing 88-test suite
passed throughout, because every test named Transport or Buzz Wire explicitly and
all three defects lived in what happens to a task that is neither.

---

## 1. The Balance heuristic bank was Transport's policy

`collect.py` imported `vmas.scenarios.transport` at module level and built
`HeuristicPolicy` from it for **every** scenario it collected. `scenario_module`
— the task's own module — was already resolved forty lines above, but only to
hash for provenance.

Verification over all 30,824 valid agent-actions in job 1233's bank:

| Policy recomputed from the saved observations | Mean absolute difference | Matching within 1e-5 |
|---|---:|---:|
| Transport heuristic | 1.46e-9 | 100% |
| Balance heuristic | 1.0795 | 0% |

426 of that bank's 1,620 anchors come from the wrong policy, and they are in the
training split, so every model in job 1233 saw them.

**Repair.** Dispatch through `scenario_module`; record the policy in
`manifest.json` as `heuristic_policy`; label the old bank in
[`outputs/balance_1233/DEFECT.md`](../../../outputs/balance_1233/DEFECT.md).

**Verified on the new bank (job 1236).** The job recomputes every live stored
action through Balance's own policy and refuses to train if they disagree:

```
12678 live steps, mean |stored - balance policy| = 0.000e+00, max 0.000e+00
```

The same check against job 1233's bank reports `1.079e+00`. A side effect worth
recording: the correct policy leaves **12,678** live steps against the old bank's
**7,706** — it drops the package far less often, which is what a competent policy
on this task looks like.

**What this does not invalidate.** Job 1233's physics are real and its models were
compared on identical data, so its tables remain internally valid for that bank.
What is withdrawn is the description of the bank as containing competent Balance
behaviour, and with it the coverage argument that motivated collecting it.

## 2. The horizon cliff was an untrained positional slot

`ARPredictor.pos_embedding` allocates six positions, but `dynamics_losses`
predicts from `latent[:, :-1]` — the sixth frame is only ever a *target*. Five
positions receive a prediction gradient; the sixth receives none. A backward pass
on a trained checkpoint gives nonzero norms for positions 0–4 and exactly `0.0`
for position 5.

`horizon_rollout.py` sized its sliding window from the allocation, so block 6
read that slot. The published curve's "18.7× cliff" is that read. Two further
corrections fell out: `rollout` raises at block **7**, not 6; and the goal rollout
path already used a three-position context, so the "universal 25-step
architectural cap" was never a cap on both planning paths.

**Repair.** `trained_frames(model)` returns `pos_embedding.shape[1] - 1`, derived
from where the supervision actually lands. The corrected tables are in
[`09_horizon_rollout.md`](09_horizon_rollout.md); three claims are withdrawn
there.

**Verified.** `test_rollout_context_covers_only_supervised_positions` runs a real
backward pass and asserts the reached positions equal `range(frames - 1)`, so a
future predictor with different supervision fails rather than silently
extrapolating.

## 3. Balance had no closed-loop outcome contract

Three call sites chose `transport_outcome if task == transport else
buzz_wire_outcome`. Balance therefore got Buzz Wire's reading, which dereferences
`scenario.ball` — an attribute Balance does not have. Job 1233 was unaffected
because it ran open-loop evaluators only, but it would have blocked the first
control run.

Balance also needs the distinction the ternary could not express: its native
`done()` is `on_the_ground OR package-overlaps-goal`, so counting `done` as
success scores every fall as a completion.

**Repair.** One `TASK_OUTCOMES` table and a `task_outcome(task_name)` that raises
for an unrecorded task. `balance_outcome` returns package-goal overlap for
success, `on_the_ground` for failure, package-goal distance for progress, with
**failure taking precedence** when both land on the same frame — matching Buzz
Wire. `EpisodeStats.outcome_fn` and `evaluate_policy`'s `outcome_fn` lost their
defaults, because a default terminal label is the same defect waiting for the
next task.

**Verified.** `test_balance_outcome_separates_completion_from_falling` asserts
`(reached | grounded) == env.done()`, which is the condition under which
`EpisodeStats.update` does not raise "Unclassified task termination". A Level-0
Balance closed loop (4 states, 32 samples, 3 iterations, GPU 0) now completes:

```
policy                   success    return  task_dist  goal_dist   coll  timeout
random                 0/4     0%    -2.780     1.6410     1.0717   0.25     0.75
zero                   0/4     0%    -3.052     1.6438     1.1214   0.25     0.75
heuristic              0/4     0%    -0.622     1.5945     0.1221   0.50     0.50
oracle                 0/4     0%     0.398     1.6093     3.3767   0.25     0.75
goal_oracle            0/4     0%    -8.006     1.6433     0.5376   0.75     0.25
```

**This is a smoke, not a result.** Four states at 1/10 the sample budget and 1/10
the iterations. It says the path runs and every termination is classified. It
does raise the question Gate 1 exists to answer — the simulator oracle moves
`task_dist` from ~1.64 to 1.6093 and completes nothing — but at this budget that
is not evidence about Balance's controllability.

## 4. Also repaired: the planner never got feedback

`closed_loop.py` set `receding_horizon = args.horizon`, so the executed and
planned horizons were the same object. At H=5 and `action_block=5` that is **all
25 primitive actions executed before the next observation**: at most four
decisions in a 100-step episode, fewer from a late anchor, and a warm start that
is entirely zeros because the whole plan was shifted away. For a task about
contact and balance this is close to open loop.

`--execute-blocks` (default **1**) separates them. It is recorded in
`reference_identity`, so a reference cache built at another cadence cannot be
reused silently, and in the wandb config. Changing cadence multiplies planning
calls per episode, so episode compute must be reported alongside per-decision
budget.

## 5. Also labelled: two comparisons that cross encoders

Per review §3, without changing any number:

* the C7 paired-response table subtracts errors measured in two different learned
  latent spaces. It now prints that it is a within-pair sign test, not a physical
  margin.
* the "shared physical probe" correlates response **magnitude** with observation
  **magnitude**. It now prints that it decodes no physical vector and checks no
  direction, so a prediction pointing the wrong way can rank perfectly.

A common-coordinate physical response is Gate 2, not a labelling change.

## 5b. One hypothesis closed off cheaply

Review §4 also raises a search risk: `cem_plan` returns `elites.mean(dim=1)`
computed **after** the final `cost_fn` call, so the plan that actually executes
is never scored at that update, and averaging two good but different plans can
produce one that is neither.

Measured on Balance with the true simulator and the native reward cost, 8 train
roots at the real budget (300 samples, 30 iterations), rescoring the returned
plan with the planner's own cost:

| | value |
|---|---:|
| states where the executed mean is worse than the best sample | **8 / 8** |
| mean gap | +0.4348 (**1.2%** of mean \|best cost\|) |
| max gap | +0.8262 |

The direction the review predicted is real and unanimous. The magnitude is 1.2%,
and the executed mean sits **between** the previous incumbent and the best sample
in every state — each update improves the plan that runs. Elite averaging is not
what is stopping Balance control, and this does not justify changing the planner.
Worth rerunning on an objective with genuinely multi-modal elites before
concluding it never matters.
[Probe and output](../../../outputs/search_diagnostics/results.txt)

## 6. What Gate 0 does not fix

* `include_heuristic=true` adds heuristic **anchor states**; the training loader
  still reads the random correlated/independent branches. Correct dispatch alone
  does not teach sustained lifting. Mixing competent segments with controlled
  deviations is Gate 3.
* Nothing here measures control. Gate 1 does that, on the true simulator.
* The Balance metric disagreement is untouched — see
  [`../status_2026-09-16.md`](../status_2026-09-16.md) §6.

## Reproduce

```bash
.venv/bin/python -m pytest test/test_world_model_contracts.py -q
PYTHONPATH=. .venv/bin/python outputs/review_20260916/diagnose.py
sbatch scripts/slurm/horizon_rescore.sbatch     # job 1235, corrected curves
sbatch scripts/slurm/balance_repair.sbatch      # job 1236, corrected bank
```
