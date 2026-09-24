# 34 — Pooled re-scoring, SIGReg sweep, component scaling, and the Balance replication

**Status: completed and interpreted, 2026-09-23.** Jobs 1543–1565. This note
closes the VMAS phase of Task A. It **re-opens K21, K22, K22b and K24**, revises
K36, and records the mistakes that made those claims look settled. The
consolidated mistake list is in §8; the pivot away from VMAS is in
[`../direction_pixel_pivot_2026-09-23.md`](../direction_pixel_pivot_2026-09-23.md).

Instruments: T-A2 as in [30](30_task_a_counterfactual_fidelity.md), with authentic
block-strided context. Pooled `E_CF = sqrt(Σ‖ΔŶ−ΔY‖² / Σ‖ΔY‖²)` is the frozen
convention of [33](33_admission_benchmark.md). "Mean of ratios" (MoR) is the old
per-anchor average that K21–K24 were recorded under. The encoder audit is the
T-A2b cross-fitted head on cell `1:0`, reported as `R = (E_A − E) / (E_A − E_S)`,
where `E_A` is the actions-only head and `E_S` the physical-state head.

## 1. Jobs

| Job | What | Artifacts (`outputs/`) |
|---|---|---|
| 1543 / 1544 | Retrain the 48-checkpoint T-A2 bank at SIGReg λ = 0.009 / 0.9 (reference 1498 is λ = 0.09) | `ta2_reference_baselines_sigreg0009_1543/`, `…sigreg09_1544/` |
| 1545 | T-A2 on the canonical 1498 bank, pooled + MoR, h = 1–3, both probes | `ta2_f13_corrected_1545/` |
| 1548 / 1549 | T-A2 on 1543 / 1544, h = 1, both probes | `ta2_f13_corrected_1548/`, `…1549/` |
| 1546 / 1547 | Encoder audit, λ = 0.09, correlated / independent | `encoder_confound_1498_{correlated_1546,independent_1547}/` |
| 1550, 1553 | Encoder audit, λ = 0.009 correlated; then λ = 0.009 independent + λ = 0.9 both (packed) | `encoder_confound_sigreg0009_correlated_1550/`, `encoder_confound_packed_1553/` |
| 1551, 1552 | Cancelled; folded into 1553 | — |
| 1554, 1555 | **Failed in 23 s: `/home` 100% full.** `outputs/` moved to `/data/fuisloy/benchmarl_outputs` (checksum-verified, symlinked) | — |
| 1560 | Component scaling: 15 cells × 4 seeds, joint/correlated, Buzz Wire | `component_scaling_1560/` |
| 1562 / 1563 | T-A2 (h = 1, 2) / encoder audit on the 60 scaling checkpoints | `ta2_f13_corrected_1562/`, `encoder_confound_scaling_1563/` |
| 1561 | Balance, 4 agents, 3 kinds × 2 regimes × 8 seeds | `ta2_reference_baselines_balance_1561/` |
| 1564 / 1565 | T-A2 h = 1–3 / encoder audit h = 3 cell `1:0` on Balance | `ta2_f13_corrected_1564/`, `encoder_confound_balance_h3_1565/` |

Integrity: all training runs completed and reload exactly. Each scaling override
changed parameter count and effective rank as intended. The scaling base cell
reproduces λ = 0.09 (`R` 0.195 vs 0.19, pooled `E_CF` 0.742 vs 0.756 at 4 vs 8
seeds). All shared audit conditions reproduce job 1538 (`R_O` = 0.537).

## 2. Under pooled scoring, H1 beats H0 on Buzz Wire's strong coupling

Job 1545, λ = 0.09, correlated regime, linear probe. H0 scores exactly 1 on every cross cell.

| h | joint vs H0, pooled | axis-0 cells `0:0`, `1:0` | axis-1 cells `0:1`, `1:1` | joint vs H0, MoR (same fits) |
|---:|---|---|---|---|
| 1 | **8/8**, Δ −0.245 | E 0.66–0.68, cos +0.80, **8/8** | E 1.04–1.14, cos −0.07…+0.09 | 0/8 |
| 2 | **8/8**, Δ −0.221 | E 0.66–0.68, cos +0.78…+0.82, 8/8 | E 1.16–1.23 | 0/8 |
| 3 | 0/8, Δ +0.097 | 8/8 (Δ −0.37…−0.44) | loses | 0/8 |

The independent regime and the MLP probe agree: 8/8 pooled at h = 1–2.
Per-axis gain for joint at h = 1 is ‖ΔŶ‖/‖ΔY‖ = **0.64** on axis 0 and 0.46 on axis 1.

* **The verdict depends on the aggregation.** MoR averages per-anchor ratios, so
  the many small-effect axis-1 anchors dominate it (joint axis-1: MoR 1.45,
  pooled 1.09). Pooling weights by effect energy, which is dominated by axis 0.
  The **per-axis** reading holds under both rules: H1 resolves the strong
  x-coupling (56% of self, T-A1) with the right direction and about ⅔ of the
  size, and does not resolve the weak y-coupling (12% of self).
* **h = 3 is support-limited.** 968 anchors per arm survive, and cell `1:0`'s
  bootstrap interval collapses to a point, [0.65, 0.65]. That is the same failure
  as the admission gate's false admission (note 33). The pooled h = 3 loss comes
  from axis 1.
* **Relational vs joint (K22):** pooled 1/8 (linear) and 6/8 (MLP) at h = 1, and
  4/8 and 7/8 at h = 2. That is probe-dependent and null. On axis 0 relational is
  slightly worse (0–1/8). On axis 1 it wins 6–8/8, **only by predicting a smaller
  response** (gain 0.37 vs 0.46 at cosine ≈ 0).
* **Horizon decay (K24):** axis-0 cosine is +0.80 → +0.78…+0.82 from h = 1 to 2,
  so it does **not** decay. The recorded decay averaged in axis 1 (cos → −0.18)
  and the survivor-limited h = 3.

## 3. SIGReg λ: retention and fidelity move in opposite directions

| λ | trained latent `R` | untrained encoder `R` | latent ⊕ observation `R` | latent ⊕ ball `R` | pooled joint `E_CF`, h = 1 | joint vs H0, pooled |
|---:|---:|---:|---:|---:|---:|---|
| 0.009 | **0.39–0.44** | 0.33–0.34 | 0.51–0.56 | 0.97–0.99 | 0.92–0.98 | 6–8/8, Δ −0.03…−0.09 |
| 0.09 (reference) | 0.14–0.24 | 0.33–0.34 | 0.35–0.45 | 0.92–0.94 | 0.74–0.77 | 8/8, Δ −0.23…−0.27 |
| 0.9 | 0.07–0.17 | 0.33–0.34 | 0.34–0.36 | 0.88–0.89 | 0.74–0.79 | 8/8, Δ −0.21…−0.26 |

Retention is monotone in λ. The weakest regularizer keeps the most cross-agent
information and gives the **worst** fidelity. Its T-A2 probe floor also rises
(0.24–0.48 vs 0.18–0.23), so part of that loss may be the probe reading a less
regular latent.

## 4. Part of the encoder loss is the diagnostic head, not the encoder (K36 revised)

At λ = 0.09 the raw observation recovers `R` = 0.537 and the trained latent
0.14–0.24. Three controls in the same audit split that gap:

| control | `R` | reading |
|---|---:|---|
| observation, random full-rank linear map to 192-D | 0.556 | width alone costs nothing |
| **untrained encoder**, same architecture | 0.33–0.36 (base); 0.23–0.27 for deeper or wider encoders | a random nonlinear 6→192 map costs 0.18–0.31 through the head alone |
| **latent ⊕ observation** | 0.35–0.45 | **below the observation alone** — adding a deterministic function of the input makes the head worse |

The latent head also overfits heavily: train `E` 0.18 against test 0.49, with
683 train anchors. So about **half** the observation-to-latent gap
(0.54 → ≈0.34) is the head's sample efficiency on nonlinear 192-D features. The
other half (≈0.34 → 0.19) is JEPA training at λ ≥ 0.09. At λ = 0.009, training
*adds* retention instead (0.34 → 0.42). "The encoder loses two thirds" (K36)
overstates the encoder's share.

## 5. Component scaling (registered S1/S2 in `scripts/slurm/component_scaling.sbatch`)

Seed-paired Δ pooled cross `E_CF` against base (4 seeds; h = 1, 2 × linear, MLP;
negative is better):

| cell | Δ `E_CF` | seeds better | latent `R` (base 0.195) |
|---|---|---|---:|
| data ½ (fixed 500 updates) | +0.030 … +0.045 | **0/16** | 0.139 |
| data ¼ | +0.052 … +0.066 | **0/16** | 0.205 |
| encoder width 2048 | −0.002 … −0.033 | 3–4/4 | 0.225 |
| latent width 384 / 96 | −0.009…+0.008 / 0.000…+0.026 | mixed | 0.161 / 0.275 |
| encoder depth 2 / 4 / large (3×1024) | −0.03…+0.014 / **+0.05…+0.10** / +0.00…+0.04 | mixed | 0.190 / 0.210 / 0.203 |
| predictor depth 1 / 2 / 12 | +0.016…+0.036 / +0.004…+0.012 / +0.004…+0.010 | ≤2/4 | 0.185 / 0.241 / 0.243 |
| conditioner ¼× / 4× | +0.007…+0.017 / ±0.002 | ≤2/4 | 0.197 / 0.202 |

* **S2 supported.** No capacity knob gains more than 0.033. Removing data costs
  fidelity monotonically. Latent `R` does not move with data, so the data effect
  is in the predictor. Low-data cells reach training loss 0.022 vs 0.075
  (memorization). This is evidence from *removing* data; that more data helps is
  not yet shown.
* **S1 fails for the encoder, holds for the predictor.** Encoder width and depth
  leave `R` at 0.18–0.23. Encoder depth 4 is *worse* on axis 0 (+0.13…+0.15): it
  raises the probe floor (0.35 vs 0.18) and halves the response (gain 0.51). Its
  axis-1 "gain" is the same shrinkage as in §2. Predictor depth beyond 2 does
  nothing.
* **Retention does not predict fidelity.** Spearman ρ(`R`, axis-0 `E_CF`) across
  15 cells is −0.22 (h = 1) and −0.15 (h = 2), not significant. Across λ, the sign
  is the wrong way (§3).

## 6. Balance (4 agents, admitted cell `1:0`): no gradable result

**T-A2 (1564).** Pooled H1 vs H0 is **0/8** at every h, probe and regime (joint
Δ +0.39 at h = 1, +0.07 at h = 3). Relational beats joint **8/8 everywhere**.
Relational is below H0 on cells `1:0` and `2:0` at h ≥ 2 on 4–8/8 seeds, but by
0.2–0.8% of the effect, with a seed range that crosses zero.

That ordering is **shrinkage, not fidelity**. At h = 3 on axis 0, relational's
gain is 0.135 and joint's 0.297, both at cosine ≈ 0.03–0.12. With direction that
weak, `E ≈ sqrt(1 + m² − 2m·cos)` falls toward 1 as `m → 0`. This reproduces
0.993 and 1.033.

**No Balance cell passes the registered probe-floor rule** (floor ≤ 1/3). Cell
`1:0` has a floor of 0.52 / 0.39 / 0.41 at h = 1 / 2 / 3; the other cells range
0.38–1.79. The admission gate (note 33) admitted `1:0` using a *different*
instrument: separation between an actions-only head and a physical-state head.
**Admission did not imply that the T-A2 probe can resolve the cell.**

**Encoder audit (1565), h = 3, cell `1:0`.** `E_A` 0.746, `E_S` 0.628,
observation 0.613, history 0.606. The observation matches or beats the physical
reference (`R_O` 1.13). The trained latent recovers `R` 0.91–0.94, the untrained
encoder 1.02–1.03, and latent ⊕ ball 0.90–0.95. The full gap is 0.1175 against
±0.03 intervals, so none of these can be told apart. Encoder losses below about
0.25 `R` are undetectable here.

**Reading.** On Balance `z_t` supports a head at the physical reference, yet the
trained dynamics emit almost no cross response. This compares two instruments,
so treat it as indicative, not established.

## 7. Synthesis

* **The trained predictor, not the encoder's capacity, is where fidelity is lost.**
  * On Buzz Wire's strong coupling, H1 carries the cross effect at about ⅔ gain,
    well above the probe floor (0.66 vs 0.18).
  * On Balance the dynamics carry almost none.
  * No architectural or capacity knob moves fidelity. Removing data does.
    λ moves retention, in the direction opposite to fidelity.
* **The relational inductive bias has no measured advantage** once shrinkage is
  separated from direction. That holds on both tasks.
* **The VMAS test bed cannot carry the question further:**
  * Buzz Wire: one gradable axis, survival-limited beyond h = 2, 16 test episodes.
  * Balance: every cell is below the probe resolution.
  * Transport, Wheel, Dropout: no resolvable interaction (note 33).
  * The observation-matching problem (the ball is unobserved, K5/K28/K30) is
    confounded with every representation claim.

## 8. Mistakes, concisely

1. **The aggregation decided the verdict and was never varied.** K21, K22 and K24
   were recorded under MoR. Pooling was adopted in note 33 for Balance, but
   Buzz Wire's canonical results were never re-scored, so its conclusions stood
   on the superseded convention until job 1545.
2. **Pooling across cells hid axis structure.** Buzz Wire's cross effect is two
   different effects (x strong, y weak). Every "pooled over four cells" number
   averaged a success with a failure.
3. **E_CF rewards shrinkage.** With low cosine, a smaller response scores closer
   to 1. "Relational beats joint" was never checked against the gain before it
   was cited (K22, K22b, and the Balance 8/8).
4. **Survivor-limited horizons were read as decay.** h = 3 on Buzz Wire rests on
   a collapsed interval. Recorded as K24.
5. **The admission criterion and the grading instrument differed.** A cell
   admitted by the head-separation gate failed T-A2's probe-floor rule.
6. **Diagnostic-head capacity was charged to the encoder.** K36 ("loses two
   thirds") was recorded before the untrained and latent ⊕ observation controls
   existed.
7. **Inherited cell lists and conventions.** Jobs 1527 and 1540 audited the wrong
   Balance cells, and the earlier convention chain (1514 → 1515 → 1516) flipped
   K35 and K36 in turn. Each fix was local. None triggered a re-score of dependent
   claims.
8. **Operations:** a full `/home` killed two submissions. One evaluation was run
   single-process with 1/8 cores used. Jobs 1543–1553 went
   undocumented until this note.

## 9. What carries forward

* Report fidelity **per interaction axis**, with the gain ‖ΔŶ‖/‖ΔY‖ and cosine
  beside `E_CF`. Never report `E_CF` alone.
* Admit a cell only if the grading probe itself clears its floor on that cell.
* Carry the untrained-encoder and latent ⊕ input controls in every
  representation audit.
* Require a minimum number of root episodes per cell *and per horizon* before
  any interval is read.
* When a convention changes, re-score **every** claim that depends on it in the
  same commit.
