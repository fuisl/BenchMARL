# Direction v3: predict how joint outcomes change, not where the world goes

**Status:** current statement of intent for the **method**. Supersedes neither
measurement document; it says what to build and why, and it is grounded in a
result measured hours before it was written (T-A2b-2, job 1506).

Companions: [Task A direction](direction_2026-09-21.md),
[planning direction](direction_planning_2026-09-21.md).
Registered experiments: [`experiments/32_joint_representation.md`](experiments/32_joint_representation.md).

## 1. Are we already following this? An honest audit

Partly. The measurement half is exactly this direction; the **method** half is
not implemented at all.

| Element | Status |
|---|---|
| Centralized setting: joint action supplied, predict joint consequence, no intent inference | **Followed.** Registered in the Task A direction §4 |
| Counterfactual fidelity as the central object, not a diagnostic | **Followed.** T-A1, T-A2, T-A2b are exactly this |
| **Level I** kinematics `ẑ_i'` | **Measured** — the self block |
| **Level II** interaction `∂z_i'/∂a_j` | **Measured** — the cross block, and the whole of T-A1/T-A2 |
| **Level III** joint task outcome `ẑ_G'` | **Absent.** Not represented, not predicted, not measured |
| Agent-neutral world token `z_G` | **Absent.** `grep` over `models.py` finds no such token. It appears in the planning direction as intent only |
| History / belief-state encoding `H_t → Z_t` | **Partial.** The reference predictor consumes 3 temporal positions, but `state_input: observation` means the **encoder sees one frame**. There is no learned belief state |
| Counterfactual consistency loss `L_CF` | **Absent.** The objective is exactly `prediction + λ·SIGReg`, verified in `train.py:dynamics_losses` |
| H3 / H4 / H5 as registered experiments | **Absent** until this document |

So: we have been measuring the right thing and building the ordinary thing.

## 2. The result that makes `z_G` non-optional

T-A2b (job 1503) found the current latent is not counterfactually sufficient —
recovery `R ≈ 0` on cross `E_CF` for all three conditioner kinds — and that the
failure is **upstream of the predictor**. T-A2b-2 (job 1506) then localized it
exactly:

| Input to the same head | cross `E_CF` | `R` on `E_CF` | `R` on cosine |
|---|---:|---:|---:|
| state-blind floor | 0.8951 | — | — |
| latent alone | 0.881–0.895 | **+0.02** | +0.61 |
| **latent ⊕ ball/linkage state** | **0.517–0.523** | **+0.86 to +0.88** | **+0.96** |
| physical ceiling | 0.4645 | 1.00 | 1.00 |

Adding the omitted mediating state to the *same* latent, with the *same* head,
recovers **86–88%** of the gap. Buzz Wire's observation is literally
`[pos, vel, pos − goal]` — no ball, no linkage, no partner — while the ball is
rigidly jointed to both agents and mediates the entire cross-agent effect.

**Therefore the encoder and the JEPA objective are exonerated, and the missing
ingredient is an explicit representation of the jointly-controlled thing.** That
is precisely what an agent-neutral `z_G` is.

This converts `z_G` from a plausible architectural idea into the indicated
repair for a measured deficit.

## 3. What humans suggest, stated as engineering

The useful claim is not "humans predict other people." It is:

```math
\boxed{\text{predict the consequences needed to coordinate toward a shared outcome.}}
```

Two people carrying a table do not reconstruct pixels. They maintain a compact
joint state — where the table is, how it is oriented, whether it is rotating
correctly — predict the consequence of a candidate joint action, act, observe
the error, and update. That is model-predictive control with a learned,
lossy, task-relevant state, which is the JEPA bet.

Cooperation splits the consequence into three levels, and coordination needs all
three to be coherent:

| Level | Quantity | Our status |
|---|---|---|
| **I** kinematics | `ẑ_i'` — what moves where | measured, and well captured (self `E_CF` ≈ 0.28 from latent) |
| **II** interaction | `∂z_i'/∂a_j` — how your action changes my outcome | measured, and **not** resolved (`E_CF` > 1) |
| **III** joint outcome | `ẑ_G'` — what our combined action achieves | **not represented** |

Our models currently must encode Level III implicitly inside two agent-centric
latents. T-A2b-2 says that is exactly where they fail.

**Intent stays out of scope.** In centralized MPC the coordinator proposes
`(a_A, a_B)`, so the model needs "if we both do this, what follows?" — a motor
forward model, not an intention model. Intent inference becomes necessary only
in the decentralized setting, which remains a separate project.

## 4. Architecture

```math
Z_t=\{z^1_t,\ldots,z^N_t,\;z^G_t\},
\qquad
z^i_t=E(H^i_t),
\qquad
z^G_t=G(\{z^i_t\},H_t)
```

with `H_t` a **history** window of observations and executed actions, not a
single frame — because interaction is inferred from motion, and because the
instantaneous observation provably omits the mediating state (§2).

```math
m_{ij}=\phi(z^i,z^j,a^i,a^j),\qquad m_G=\textstyle\sum_{i\neq j}m_{ij},
```
```math
\hat z^{i\prime}=P_i(z^i,a^i,m_i,z^G),
\qquad
\hat z^{G\prime}=P_G(z^G,m_G,\mathbf A).
```

**`z_G` must be learned from legitimate observations**, not handed the simulator
state. Privileged state is what the *diagnostic* used to establish the ceiling;
using it in the model would answer a different, easier question. Whether
predictive learning can recover the mediating state from history is the open
question, and it is registered as the gate on this whole architecture (§6, G0).

## 5. The method contribution: train on differences

Ordinary LeWM trains `Ẑ' ≈ Z'`. But coordination depends on *differences between
consequences*. With paired counterfactual branches `A` and `A'`:

```math
\boxed{
\mathcal L_{\rm CF}=
\bigl\lVert\,[\hat Z'(A')-\hat Z'(A)]-[Z'(A')-Z'(A)]\,\bigr\rVert^2 }
```

This stays entirely in latent space: no decoder, no reward, no collision target,
no physics supervision. It tells the representation *don't merely predict each
future independently; preserve how the future changes when a partner's action
changes.*

```math
\text{ordinary JEPA: predict where the world goes}
\quad\longrightarrow\quad
\text{MA-JEPA: predict how the world changes when the joint action changes}
```

That is a sharper contribution than "we added a graph network to LeWM."

**A prediction this direction must own.** `L_CF` backpropagates into `E`, so it
is a representation objective, not merely a predictor objective. But **no loss
can recover information the observation does not contain.** T-A2b-2 showed the
deficit is missing *input*, not a failure to extract present input. So:

> `L_CF` applied to the current single-frame, observation-only encoder should
> **not** close the cross-agent gap. It should close it only when paired with an
> input that can carry the mediating state — history, or `z_G`, or both.

If `L_CF` alone closes the gap, this reasoning is wrong and T-A2b-2 needs
re-examining. That is a real falsification condition, and it is why H3 and H4
are registered as separable factors rather than shipped together.

## 6. Hypotheses

| | Hypothesis | Standing |
|---|---|---|
| **H1** | For physically coupled tasks, `P(z_i'\|z_i,a_i)` cannot reproduce cross-agent consequences | **Established** (T-A1): cross terms active on 100% of anchors, 21.6% of total response at one block |
| **H2** | Joint context makes the model interaction-*sensitive* but ordinary next-latent prediction still fails to preserve state-specific counterfactual effects | **Established** (T-A2/T-A2b): cosine +0.40 but `E_CF` > 1; direction recovered, scale not |
| **H3** | An explicit agent-neutral `z_G` improves counterfactual fidelity over per-agent latents alone, especially where the consequence concerns a shared object | **Registered, not run.** T-A2b-2 makes it the leading candidate |
| **H4** | `L_CF` improves `E_CF` even when ordinary IID latent MSE barely changes — i.e. nominal ≠ counterfactual fidelity | **Registered, not run.** The sharp methodological claim |
| **H5** | Relational structure gains most under compositional shift: missing joint-action combinations, role swaps, changed `N` | **Partial.** K22b observed the coverage half; permutation and `N_train ≠ N_test` untested |

**H4 is the one that would make ordinary prediction loss look inadequate as a
model-selection criterion**, which M4 already half-showed: one-step teacher-forced
prediction does not separate the three baselines at all.

## 7. Ordering, and the gate that comes first

**G0 — can history recover the mediating state?** Before building `z_G`, test
whether the information is present in legitimate observations at all. Fit the
T-A2b head on an observation-*history* window instead of a single frame. If
history approaches the physical ceiling, a learned `z_G` is feasible. If it stays
at the blind floor, **no architecture over these observations can work**, and the
honest conclusion is that Buzz Wire's observation is inadequate for the
counterfactual question — a task-design result, not a method result.

This gate is cheap, it reuses the T-A2b machinery, and it prevents building an
architecture that cannot succeed.

Then: **H3** (`z_G`) and **H4** (`L_CF`) as separable factors, ideally a 2×2, so
the prediction in §5 is testable rather than confounded. **H5** last, since it
needs new tasks or team sizes.

Control work stays behind the precondition already registered in the planning
ladder: no significant control compute until a model reaches `E_CF < 1` on the
interaction cells.

## 8. What the paper becomes

Not "does a JEPA predict multi-agent futures?" but:

```math
\boxed{
\begin{array}{c}
\textbf{Can latent predictive learning acquire the compact joint representation}\\
\textbf{needed to predict how cooperative outcomes change under alternative}\\
\textbf{joint actions?}
\end{array}}
```

That gives JEPA a reason to exist beyond cheapness, gives the relational model a
defined role, makes `z_G` an empirical necessity rather than a design taste, and
puts the counterfactual experiments at the centre instead of the margin.

## 9. Citation debt

The human joint-action framing above is used as **engineering motivation, not
evidence**. The "what / when / where" decomposition of joint-action prediction
and the 2026 agent-neutral predictive-processing account are recorded as
described in the working discussion and are **unverified against primary
sources**; neither appears in [`litreview.md`](litreview.md). The existing
`<!-- cite: turn... -->` markers there remain unconverted. None of this may enter
a submission before the sources are checked and added.
