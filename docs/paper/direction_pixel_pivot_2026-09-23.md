# Direction: move Task A to pixel environments shared with LeWM

**Status: intent, 2026-09-23. Nothing here has run.** This note never overrides a
measured result. Evidence for the decision is in
[experiment 34](experiments/34_convention_audit_scaling_balance.md) and claim K52
of the [knowledge index](../RESEARCH_KNOWLEDGE_INDEX.md).

## 1. Why leave VMAS

* **One gradable interaction.** Buzz Wire's x-coupling is the only cross effect
  any instrument resolved. Its y-coupling is weak, the task is survival-limited
  beyond h = 2, and it has 16 test episodes.
* **The other tasks do not grade.** Balance's admitted cell fails the T-A2 probe
  floor (K47). Transport and Wheel have no resolvable interaction (K45).
* **Observation matching dominated the work.** VMAS observations omit the
  mediating body (Buzz Wire's ball, K5/K28/K30). Every representation claim was
  confounded with observability, and the vector encoder (6 → 192) is not the
  architecture LeWM validates.
* **The comparison the project wants is with LeWM itself.** Using its
  environments, data format, encoder, planner and baselines removes the
  environment-matching question and leaves only the architecture as the variable.

## 2. What LeWM actually provides (checked against local checkout `8edfeb3`)

Source: `/tmp/lewm-audit.5KYL6Q/le-wm` (same commit as our vendored conformance
fixtures, note 27). **`/tmp` is not durable** — re-clone before relying on it.

| Item | What the checkout shows |
|---|---|
| Stack | `stable-worldmodel` (environments, planning, evaluation) + `stable-pretraining` (training). LeWM's repo holds only `jepa.py`, `module.py`, `train.py`, `eval.py` and Hydra configs |
| Environments | `swm/PushT-v1`, TwoRoom, `swm/OGBCube-v0` (OGBench cube), DMC Reacher |
| Data | Offline HDF5/lance on HuggingFace (`quentinll/lewm` collection): `pusht_expert_train`, `tworoom`, `ogbench/cube_single_expert`, `reacher`. Keys: `pixels`, `action`, plus `proprio`/`state`/`observation`, frameskip 5 |
| Planning | CEM (Adam solver also provided), horizon 5, receding horizon 5, action block 5, 50 episodes, goal 25 steps ahead, image 224 |
| State restore | Eval configs call `_set_state` / `set_state(qpos, qvel)` from dataset states. **This is the hook the counterfactual instrument needs**: branch from a restored state under two actions |
| Checkpoints | LeWM on HF for pusht, cube, tworoom, reacher |
| Baselines | PLDM, LeJEPA, IVL, IQL, GCBC, DINO-WM, DINO-WM-noprop, on Google Drive. Training scripts live in `stable-worldmodel/scripts/train`, not in the LeWM repo |

**Every LeWM environment is single-agent.** That is the main gap between "the most
comparable setting" and the multi-agent research question (§4).

## 3. Baselines to inspect, and why each matters here

The descriptions below are from memory of the papers, **not yet checked against
`stable-worldmodel`'s implementations**. Read the code before citing any of them.

| Baseline | Why it matters for this project |
|---|---|
| **PLDM** | Reward-free JEPA-style latent dynamics with an explicit anti-collapse regularizer, planned with MPC. The closest existing comparison for "JEPA + planner" and for the claim that planning with a latent model generalizes from offline data |
| **DINO-WM** (and `noprop`) | Frozen pretrained encoder plus a learned predictor. Separates "the encoder learned the interaction" from "the predictor did", which is exactly the encoder-vs-predictor question 34 could not settle on VMAS |
| **LeJEPA** | Same regularizer family as LeWM's SIGReg without the world-model training. A control for the λ result (K51: retention and fidelity move oppositely) |
| **IQL, GCBC, IVL** | Model-free / goal-conditioned policies. Task B reference points, not Task A comparisons |

## 4. The multi-agent question in pixels: candidate settings

| Option | Interaction | Comparability with LeWM | Cost | Assessment |
|---|---|---|---|---|
| **A. Two-pusher PushT** | Two agents push one T-block. The block is the mediating body **and it is visible in pixels** | Same renderer, physics (pymunk), data format, planner, image size | Add a second agent to `PushT-v1`, a two-agent data collector and a joint-action action space | **Recommended.** It reproduces Buzz Wire's mediated coupling without its observability confound |
| B. Two-agent TwoRoom | Weak: only through collisions or a shared doorway | High | Low | Likely to repeat Transport and Wheel's inactivity (K45). Useful only as a negative control |
| C. Render VMAS to pixels | Known ground truth (T-A1 exists) | Low: not LeWM's environments | Medium | Keeps survival limits and axis asymmetry; defeats the purpose |
| D. Single-agent LeWM tasks only | None across agents | Exact | Lowest | Required as a reproduction gate (P0), not as the research setting |

## 5. What transfers from the VMAS phase

The *instrument* transfers. The numbers do not.

* Grade the counterfactual effect **per interaction axis**. Report gain
  ‖ΔŶ‖/‖ΔY‖ and cosine beside `E_CF`, never `E_CF` alone (K46, K48).
* Admit a cell only if **the grading probe itself** clears floor ≤ 1/3 on it (K47).
* Require minimum root episodes per cell **and per horizon** before reading an interval.
* In every representation audit, carry the untrained-encoder and
  latent ⊕ input controls (K36 revision). With pixels, "the input" is 224×224×3,
  so the head-capacity control matters more, not less.
* Arms H0/H1/H2 at matched capacity, and the Task A / Task B split.
* When a convention changes, re-score every dependent claim in the same commit.

## 6. First gates (proposed, to be registered in a new note before running)

| Gate | Question | Pass condition |
|---|---|---|
| **P0** | Does our stack reproduce LeWM on single-agent PushT? | Published checkpoint and our retrain both reach LeWM's reported planning success within its interval. This is the architecture-conformance anchor that replaces note 27's vector-profile tests |
| **P1** | Does two-pusher PushT have a gradable cross effect? | T-A1 on restored states: cross effect active, a pixel→state probe floor ≤ 1/3 per axis, enough anchors per horizon, A-vs-S separation — all on the **same** instrument |
| **P2** | Does joint-action conditioning resolve it from pixels? | H1 vs H0 per axis, paired by seed, with gain and cosine, both probe families |
| **P3** | Is the result specific to LeWM's objective? | The same P2 measurement on PLDM and DINO-WM predictors trained on the two-pusher data |

## 7. Open decisions and verification debt

1. **Two-pusher PushT or another multi-agent pixel task?** Decide before P1.
   The deciding factor is whether the cross effect is gradable per axis.
2. **Install `stable-worldmodel` in a separate environment.** The BenchMARL venv
   already has pymunk 7.3, gymnasium 1.3 and pygame, but version pins are unchecked.
3. **Data storage.** Put HuggingFace datasets and `$STABLEWM_HOME` on `/data/fuisloy`,
   not `/home`. `/home` was full on 2026-09-23.
4. **Read the baselines' code** (§3) before any of their descriptions is cited.
5. **Archive the VMAS artifacts.** `outputs/` now lives at
   `/data/fuisloy/benchmarl_outputs` (symlinked); the evidence for notes 30–34 is there.
