# M4 — Three latent world-model baselines

2026-09-14. **M4 pilot complete** against its stated gate: all three predictors
train and reload reproducibly on the fixed M3 bank, at matched capacity, with
latent-health and prediction checks recorded. No counterfactual evaluation,
plan ranking or closed-loop control is in this milestone; those are M5.

## What was built

`examples/world_model/models.py` ports the predictor stack, the SIGReg
anti-collapse term and the action embedder from official LeWM revision
`8edfeb336732b5f3ce7b8b210d0ba370a09e2cac` (MIT, `lucas-maes/le-wm`), keeping
its causal autoregressive prediction, AdaLN-zero action conditioning, undetached
prediction target and `loss = MSE(pred, target) + weight * SIGReg(emb)`.

Recorded deviations: a shared per-agent MLP encoder replaces the ViT (VMAS is
vector-observation); `einops` is not a project dependency so reshapes are
native; padded blocks are masked out of prediction, readout and SIGReg because
our snippets end at episode boundaries; and a reward/termination readout is
added because we plan with `J = -sum_t sum_i r_i,t` rather than the reference's
terminal latent-goal cost.

The three baselines differ **only** in how agent `j` reaches agent `i`:

| Baseline | Conditioning of agent i |
|---|---|
| independent | `W[a_i, z_i]` — no cross-agent path |
| joint | `W[a_i, z_i, (z_j, a_j) in fixed agent order]` |
| relational | `W[a_i, z_i, sum_{j!=i} phi(z_i, z_j, a_i, a_j)]` |

`joint` and `relational` receive identical information and differ only in
inductive bias (fixed concatenation vs. permutation-equivariant sum pooling),
which is what makes `joint` a control rather than a strawman.

Encoder, action embedder, predictor and readout are architecturally identical
across baselines; only the conditioner changes, and its hidden width is solved
per baseline to equalise parameter count.

## Validation

**22 focused tests** (`test/test_world_model_models.py`) assert the information
paths rather than the loss curves, because a wiring mistake would leak
cross-agent information while every curve still looked plausible: `independent`
is bit-exactly invariant to another agent's action and observation; `joint` and
`relational` are not; `relational` is permutation equivariant and `joint`
provably is not; all three are causal in time; rollout agrees with teacher
forcing on the first step; SIGReg scores collapsed latents worse than isotropic
ones; masked reductions ignore padding and reject an empty mask.

**AdaLN-zero makes an untrained predictor the exact identity.** The reference
initialises `adaLN_modulation`'s last layer to zero, so conditioning is inert in
both value and gradient at initialisation and all three baselines start as the
same function. Tests of the information path therefore have to probe a model
whose gates have moved. `conditioning_gate_scale` is reported with the
validation metrics for the same reason: if those gates stayed at zero, the three
baselines would be literally identical and any comparison between them void.
They move (≈0.003 after 100 epochs).

Formatting and lint follow the repository's pinned `black==22.3.0` and
`flake8==4.0.1` with bugbear/comprehensions.

## Runs

Job **1191** (single seed, unmatched capacity) is retained as a negative
methodological result: `independent`/`relational`/`joint` held 3.03M/3.62M/3.82M
dynamics parameters, so its apparent relational advantage over `independent` was
confounded by 19% more capacity and is not reported as evidence.

Job **1192** is the reported pilot: 3 baselines x 2 action regimes x 3 training
seeds (4100/4101/4102) = 18 runs, 15m08s, exit 0. Conditioner hidden widths
1870/512/624 give dynamics parameter counts **3,815,192 / 3,815,342 /
3,814,670** — matched to within 0.02%. Dataset `outputs/transport_data_1190`,
100 dynamics epochs then 50 readout epochs, LeWM's optimiser settings
(AdamW, lr 5e-5, weight decay 1e-3, gradient clip 1.0, SIGReg weight 0.09).

### Prediction

Mean ± sd over three seeds. Errors are MSE in each model's own latent space and
are **not comparable between baselines in absolute terms** (M4's standing
caveat); the comparison of interest is the within-metric ordering.

| Metric | Regime | independent | joint | relational |
|---|---|---:|---:|---:|
| one-step | correlated | 0.0357 ± 0.0010 | 0.0382 ± 0.0013 | 0.0362 ± 0.0014 |
| one-step | independent | 0.0412 ± 0.0021 | 0.0451 ± 0.0008 | 0.0414 ± 0.0013 |
| rollout | correlated | 0.0629 ± 0.0043 | 0.0600 ± 0.0011 | **0.0547 ± 0.0010** |
| rollout | independent | 0.0688 ± 0.0053 | 0.0669 ± 0.0021 | **0.0593 ± 0.0013** |
| final-step | correlated | 0.0627 ± 0.0032 | 0.0587 ± 0.0018 | **0.0533 ± 0.0029** |
| final-step | independent | 0.0702 ± 0.0043 | 0.0673 ± 0.0045 | **0.0581 ± 0.0016** |

**One-step teacher-forced prediction does not separate the baselines**;
`independent` and `relational` are within one standard deviation and `joint` is
marginally worst. **Multi-step rollout does separate them**, with `relational`
lowest in both regimes: 13.0% below `independent` under correlated actions and
13.8% below under independent actions.

Three observations that limit what this supports:

1. **The separation is over prediction horizon, not joint-action coverage.**
   It is a different dissociation from the one the paper hypothesises. The
   counterfactual query (§7.2, `G_CF`) is untested here: every number above is
   on logged actions.
2. **The relational gain is the same size in both regimes** (13.0% vs 13.8%).
   The impact notes expect relational ≈ central ≈ independent under full
   coverage, with relational helping mainly under restricted coverage. That
   contrast does not appear. Either coverage is not the operative variable at
   this horizon, or both regimes are effectively restricted — consistent with
   M3's finding that only 32/239 test anchors show any cross-agent physical
   effect over 25 steps.
3. **A non-interaction explanation was live.** Sum pooling may simply be a
   better-conditioned map than a 1870-wide MLP on `[a_i, z_i]`, improving
   rollout stability without modelling interaction at all. The Dropout control
   below tests exactly this.

Three seeds is a pilot count; M6 targets 5–10 for headline comparisons, and the
control below supersedes these three-seed numbers with eight.

## M1 Row 4 — weak-interaction control (job 1194)

Dropout has no cross-agent dynamics: agents do not collide, share no object and
never observe each other, so coupling exists only in the shared reward and
termination. If the relational rollout advantage is interaction modelling it
should not appear there; if it is a conditioning effect it should transfer.

The **quantity and its reading were fixed before the data existed**: the
within-task paired relational-vs-independent rollout change. Absolute errors are
not comparable across tasks (7 observation features against 11, in separately
learned latent spaces), so only the within-task ratio is. Seeds are shared
across baselines and fix both initialisation and batch order, so differences are
paired per seed rather than compared as group means.

A first three-seed attempt (job 1193) was **inconclusive and is not reported as
a result**: one seed of three found a much worse optimum on Dropout for every
baseline, enough to set both the mean and the spread, and the mean and median
pointed in opposite directions. Job 1194 reran both tasks over eight seeds at
identical budgets — 96 runs, all completed.

| Task | Regime | mean | median | 95% CI (bootstrap) | seeds better |
|---|---|---:|---:|---:|---:|
| Transport | correlated | -12.0% | -10.6% | **[-14.8, -9.7]** | **8/8** |
| Transport | independent | -12.8% | -12.4% | **[-15.9, -10.1]** | **8/8** |
| Dropout | correlated | +9.4% | +4.2% | [-15.2, +35.1] | 3/8 |
| Dropout | independent | +27.9% | +7.8% | [-7.8, +68.1] | 3/8 |

On Transport the advantage holds for **every seed in both regimes** with
intervals clear of zero (sign test p ~ 0.008 per regime). On Dropout the point
estimates are positive — relational slightly worse — and both intervals contain
zero. The effect does not transfer to a task without cross-agent dynamics,
which is the well-behaved outcome Row 4 asks for.

Two limits on how far that carries:

1. **The Dropout correlated interval [-15.2, +35.1] still contains -13%**, so
   that regime cannot reject an effect of the Transport size; the independent
   regime's [-7.8, +68.1] does exclude it. The defensible claim is the 8/8
   versus 3/8 contrast, not a demonstrated null on Dropout.
2. **Dropout differs from Transport in more than interaction** — 7 observation
   features rather than 11, sparse reward, different dynamics, and markedly less
   stable optimisation (per-seed rollout spread 5.5x against 1.2x on Transport,
   affecting all three baselines about equally). The absence could owe to any of
   these. A same-task interaction ablation would be a tighter control; VMAS does
   not offer one for Transport directly.

Two facts carry over unchanged and both bear on M5. `terminated_rate` is 0.0000
on Dropout as well, because random actions do not reach the goal inside a
25-step snippet, so the termination head is unvalidated on either task. And
Dropout's **reward readout fails** — relative error ~2.5, worse than predicting
the mean, against 0.25 on Transport — so Dropout cannot support a planning claim
without separate diagnosis, even though it serves as a prediction control.

Reproduce with `python -m examples.world_model.compare_baselines
outputs/interaction_control_1194`. The summariser is version controlled rather
than left in a run directory, because M5 needs the same comparison.

## Stratified diagnostic — does the advantage live where the interaction is?

The Dropout control compares two tasks, so a null there has causes besides the
absence of interaction. The tighter within-task version scores the same models on
the same data, split by whether an anchor actually shows a physical cross-agent
effect under the M3 intervention (31/239 test anchors at five primitive steps).

**The diagnostic is inconclusive, and the two natural metrics disagree
significantly in opposite directions.** Relational beats independent in both
strata on 8/8 seeds, but:

| Metric | active | inactive | active - inactive |
|---|---:|---:|---:|
| relative | -6.4% | -14.0% | **+7.6 pts [+4.9, +10.0]** |
| absolute | -0.01296 | -0.00711 | **-0.00584 [-0.00882, -0.00271]** |

Interaction-active anchors carry **4.11x** the error while the reduction ratio is
**1.82x**. That sits between two equally natural nulls: "uniformly r% better
everywhere" predicts a ratio equal to the error ratio (4.11), and "constant
absolute reduction" predicts 1.0. Neither is privileged by theory, so the
observation is bracketed rather than resolved, and this proxy cannot decide
whether the advantage is interaction modelling or better conditioning.

The pre-registered reading was that concentration on active anchors would be the
interaction signature. The absolute metric says concentrated, the relative says
the opposite; reporting only one would manufacture a result. What settles the
question is the counterfactual measurement itself -- `E_CF` against `E_ID` on the
same restored states -- which isolates joint-action dependence instead of
inferring it from state difficulty. That is M5 Row 1 and remains to be run.

Reproduce: `python -m examples.world_model.stratified_evaluation
outputs/interaction_control_1194/transport --data outputs/transport_data_1190`.

### Latent health

| Baseline | effective rank | latent variance | SIGReg | reward relative error | checkpoint reload |
|---|---:|---:|---:|---:|---:|
| independent | 32.1 / 192 | 0.898 | 133.90 | 0.266 | 0.0 |
| joint | 32.1 / 192 | 0.898 | 133.92 | 0.255 | 0.0 |
| relational | 32.1 / 192 | 0.898 | 133.93 | 0.253 | 0.0 |

No collapse: latent variance ≈ 0.9 and rank is stable. **Checkpoint reload is
bit-exact (0.0 maximum prediction difference) for all 18 runs**, which is M4's
reproducibility gate.

The latent geometry is identical across baselines to three significant figures.
That is itself a finding: the encoder's output distribution is set by SIGReg,
not by which conditioner sits downstream of it.

## The SIGReg regime is wrong for this setting

Measured, not assumed. SIGReg tests random 1-D projections against N(0, I), so
it penalises anisotropic covariance as well as non-Gaussian shape. Our latents
reach effective rank ≈32 in a 192-dimensional space, so ~160 directions carry
almost no variance and are read as a point mass. The achievable floor for
rank-9 data, by direct measurement:

| latent width K | 8 | 16 | 32 | 64 | 192 |
|---|---:|---:|---:|---:|---:|
| SIGReg floor | 36.6 | 74.0 | 73.6 | 76.6 | 79.5 |

A genuinely full-rank isotropic population scores ≈1.0 at every width. The
observed 133.9 therefore sits near an **unreachable** floor, and the floor is
set by rank deficiency rather than by width — narrowing K to 16 or 32 would not
remove it.

Consequence for the objective: at convergence `0.09 x 133.9 ≈ 12.1` against a
prediction loss of ≈0.036, roughly **330:1** in favour of a term that cannot
reach zero.

Two further facts bound the interpretation. The per-agent observation is 11
features with linear effective rank 9.2 after standardisation, so a 192-wide
isotropic target is unattainable by construction. And one of those 11 features,
`on_goal`, is **exactly constant at 0.0 across all 44,524 frames of both
regimes** — the same root cause as the zero-termination finding below.

This does not invalidate the baseline comparison, which applies the identical
objective to all three. It does threaten the paper's §7.1–7.2 dissociation: that
argument needs the three models to agree in-distribution *because each is
well-fitted*, not because a dominant unreachable term left all three underfitted.
Candidate responses — latent width at an attainable rank, applying SIGReg in a
projector space as LeWM's own config does and our port does not, and re-tuning
`lambda` with the `N`-scaled Epps-Pulley statistic normalised — are recorded for
test but **not yet run**, so no claim rests on them.

Relevant theory: LeJEPA (arXiv:2511.08544) proves the isotropic Gaussian
optimal for *unknown* downstream tasks (probing bias/variance over arbitrary
targets), justifies 1-D sketching by a hyperspherical Cramér-Wold argument,
recommends `lambda`≈0.05 over [0.01, 0.1] and 512–1024 projections, and gives
**no principled rule for embedding dimension**, explicitly not treating the case
where it exceeds the data's intrinsic dimension. Our downstream task is known
and singular — predict reward and rank joint-action plans — which weakens that
prior's justification here and makes `lambda` and `K` ours to tune rather than
inherit. This also resolves the outline's open SIGReg citation.

## Reward and termination readout

The readout is the interface `J = -sum_t sum_i r_i,t` needs and the reference
does not have. It maps a latent transition to block-summed per-agent reward and
a termination logit, and is fitted in a **second stage on frozen dynamics**, so
the reward signal never shapes the representation being compared. "Shared" means
identical architecture, capacity and optimisation across baselines, not shared
weights: each model learns its own latent space, so one set of weights cannot
read all three.

**Reward: works.** Relative error 0.253–0.266, i.e. roughly a quarter of the
variance a constant-mean predictor would incur (R² ≈ 0.74), consistently across
baselines. Raw MSE is reported alongside the target variance (0.00334) because
an MSE of 0.0008 is otherwise uninterpretable.

**Termination: cannot be validated on this dataset.** `terminated_rate` is
exactly 0.00000 in every split of both regimes — zero positive examples
anywhere. An always-false head is perfectly accurate and entirely useless, so
runs now carry an explicit `termination_head_validated: false` flag rather than
letting that pass as a fitted component. This is consistent with M2 (no goals in
660 Transport evaluations) and with the dead `on_goal` observation feature.

**This is an open blocker for learned MPC, not a resolved item.** Within a short
planning horizon on Transport, termination may never fire and the objective
reduces to the reward sum; that is a reasonable operating assumption but it is
an assumption, and it must be stated in §5.4 rather than implied. Validating the
termination component needs data containing terminations — Buzz Wire terminates
on wall contact and would supply them, which is an argument for the paired-task
design M1 Row 4 already wants.

## Commands

```bash
# Disposable CPU check (CSV only).
.venv/bin/python -m examples.world_model.train device=cpu 'loggers=[csv]' \
  model.kind=relational train.dynamics_epochs=2 train.readout_epochs=1 \
  hydra.run.dir=outputs/m4_cpu_check

# Reported pilot (job 1192), CSV + wandb.
sbatch scripts/slurm/world_model_baselines.sbatch

OMP_NUM_THREADS=1 .venv/bin/python -m pytest test/test_world_model_models.py -q
```

Artifacts per run: `metrics.json`, `parameters.json`, `history.csv`,
`model.pt`, `resolved_config.yaml`, `provenance.json` (commit, source hashes,
LeWM revision), and `source/`. Dataset root `outputs/transport_data_1190`;
run root `outputs/world_model_1192/`.

## Next

M5 needs the counterfactual evaluation these runs do not provide: `E_ID` vs
`E_CF` on the same restored states, then plan ranking and closed-loop control.
Before any interaction claim, run the Dropout control described above, since the
current rollout advantage has a live non-interaction explanation. The SIGReg
regime and the unvalidated termination head should both be settled before
learned MPC, because both affect whether a plan score means anything.
