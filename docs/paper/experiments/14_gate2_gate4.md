# Gates 2 and 4: the first learned-control result, and the first valid physical comparison

2026-09-16. Jobs **1238** (Gate 4, 4h46m) and **1239** (Gate 2, 20m45s).

---

## 1. Gate 4 — a learned world model does control Balance

32 independent train roots, H=5 executing one block, native reward, 300×30 CEM —
identical to the true-simulator reference in job 1237, so the oracle row is the
same experiment rather than a comparable one. Three seeds per cell.

| policy | return | per-seed | dist | coll | % of oracle | % of gap closed |
|---|---:|---|---:|---:|---:|---:|
| random | −5.76 | | 1.6139 | 0.34 | — | — |
| do-nothing | −1.45 | | 1.6052 | 0.00 | — | 0% |
| **shipped heuristic** | **+8.81** | | 1.4526 | 0.50 | 16% | 27% |
| **oracle (true simulator)** | **+54.31** | | 1.0475 | 0.00 | 100% | 100% |
| correlated / independent | +10.95 | +7.3 +16.5 +9.0 | 1.4770 | 0.04 | 20% | 23% |
| correlated / **joint** | **+17.76** | +25.1 +13.3 +14.9 | 1.3933 | 0.20 | 33% | 38% |
| correlated / relational | +5.41 | +6.8 −4.0 +13.4 | 1.5074 | 0.29 | 10% | 18% |
| independent / independent | +7.22 | +8.2 +11.3 +2.2 | 1.5111 | 0.07 | 13% | 17% |
| independent / **joint** | **+19.33** | +22.3 +33.2 +2.5 | 1.3682 | 0.29 | 36% | 42% |
| independent / relational | +10.68 | −4.3 +27.1 +9.2 | 1.4651 | 0.19 | 20% | 25% |

Paired within regime and seed, against the single-agent model:

| regime | kind | mean Δreturn | per-seed | better |
|---|---|---:|---|---:|
| correlated | joint | **+6.81** | +17.8 −3.3 +5.9 | 2/3 |
| correlated | relational | −5.54 | −0.5 −20.5 +4.4 | 1/3 |
| independent | joint | **+12.10** | +14.1 +21.9 +0.2 | **3/3** |
| independent | relational | +3.45 | −12.5 +15.9 +7.0 | 2/3 |

**What passes.** A learned world model plans better than the shipped heuristic
(+19.33 vs +8.81) and better than the single-agent model it is matched against.
`joint` is the best conditioned model in both regimes, which agrees with its 8/8
plan-ranking win on the same bank. This is the project's first learned-control
result of any kind.

**What does not.** Every cell is 0/32 native successes. The best learned model
reaches **36% of the oracle's return** and closes 42% of the distance the oracle
closes. Seed variance is larger than the effect being claimed — `joint` on
independent data spans +2.5 to +33.2 across three seeds, and `relational` spans
−12.5 to +15.9 paired. Three seeds cannot support a ranking between `joint` and
`relational`; they can support "conditioning helps" and not much more.

**Collision rates move the wrong way.** The oracle drops the package in 0% of
episodes and the single-agent model in 4–7%, but `joint` drops it in 20–29%.
Higher return with more failures means the learned models are trading safety for
progress in a way the true-dynamics planner does not.

## 2. Gate 2 — most of this project's counterfactual numbers were unmeasurable

The evaluator's first check is whether the probe can resolve the quantity at all:
the true cross-agent effect against the probe's own absolute reconstruction error.

| task | horizon | true response | probe error | ratio |
|---|---|---:|---:|---:|
| Balance | 1 block | 0.00346 | 0.24121 | **69.7×** |
| Balance | 5 blocks | 0.02836 | 0.26633 | **9.4×** |
| Balance (MLP) | 5 blocks | 0.02836 | 0.31329 | **11.1×** |
| Buzz Wire | 1 block | 0.08466 | 0.08789 | 1.04× |
| Buzz Wire | 3 blocks | 0.10442 | 0.09432 | **0.90×** |
| **Buzz Wire (MLP)** | 5 blocks | 0.12611 | 0.09930 | **0.79×** |

**Balance's cross-agent effect is one to two orders of magnitude below what any
readout can resolve, and an MLP probe does not help** — so this is not readout
weakness. Its probe floor (1.77) even exceeds the models' own ratios (1.41–1.97),
which is incoherent and is what a pure-noise measurement looks like. Every
Balance counterfactual number this project has reported, in either latent or
physical space, should be treated as unmeasured.

Buzz Wire's response is 36× larger and resolves from 3 blocks on.

## 3. Buzz Wire in common physical coordinates — the headline survives

MLP probe, 5 blocks, 8 seeds, 64 informative anchors. Lower is better; 1.0 = no
better than predicting no response.

| regime | kind | ratio | probe floor | cos(ΔY) |
|---|---|---:|---:|---:|
| correlated | independent | 0.9713 | 0.5409 | 0.1881 |
| correlated | joint | 0.9139 | 0.5409 | 0.2703 |
| correlated | **relational** | **0.8873** | 0.5410 | **0.3225** |
| independent | independent | 0.9788 | 0.5394 | 0.1406 |
| independent | joint | 0.9634 | 0.5395 | 0.1855 |
| independent | **relational** | **0.9024** | 0.5395 | **0.2786** |

Paired against `independent`: relational **−0.0841 (8/8)** and **−0.0764 (7/8)**;
joint −0.0575 (7/8) and −0.0155 (5/8, CI crosses zero).

`independent` scores 0.97–0.98 against a structural 1.0 it cannot beat by
construction, which is the metric validating itself.

**Relational is best on both magnitude and direction.** This is the first time
the project's headline has been measured in coordinates that do not depend on
which model produced them, and it holds.

## 4. The probe family flips the ordering, and only one family is fair

The same comparison with a **linear** probe:

| regime | kind | ratio | abs err |
|---|---|---:|---:|
| correlated | independent | 0.9627 | 0.14083 |
| correlated | joint | **0.9305** | 0.17236 |
| correlated | relational | 1.0111 | **0.18945** |

Relational goes from best (0.8873) to worst (1.0111) purely by changing the
readout. The reason is in the last column: a linear probe decodes relational's
latent **35% worse** than the single-agent model's (0.189 vs 0.141), and that
decodability gap is charged to the model as response error. The MLP decodes all
three to within 3% of each other (0.132–0.138), so only it compares dynamics
rather than decodability.

**Any cross-model physical claim must report the per-model reconstruction error
alongside it.** A probe that fits one architecture's geometry better than
another's will produce whatever ordering its own bias implies, and the linear
result here is an instance of exactly that.

## 5. Where this leaves the two tasks

They are no longer contradictory, because only one of them is measured:

* **Buzz Wire** — relational captures the most cross-agent response, in physical
  coordinates, on magnitude and direction, 8/8 seeds.
* **Balance** — response is unmeasurable. What *is* measured there is decisions:
  `joint` wins plan ranking 8/8 in both regimes and gives the best learned
  control, +19.33 against the heuristic's +8.81.

So the live claim is no longer "relational beats joint" or its reverse. It is
that **response accuracy and control performance are different quantities that
do not have to agree**, and this project now has one task measuring each.

## 6. What must not be claimed from this

* That `joint` beats `relational` at control. Three seeds, and the per-seed
  spread exceeds the gap.
* Any Balance counterfactual or intervention-response number, at any horizon,
  in any space.
* That relational's physical-response win transfers to control. It has not been
  tested — Buzz Wire has no working controller (job 1237's gate was Balance).
* Any cross-model physical comparison from a probe whose per-model
  reconstruction errors are not reported and comparable.

## Reproduce

```bash
sbatch scripts/slurm/balance_gate4.sbatch        # job 1238
sbatch scripts/slurm/physical_response.sbatch    # job 1239
```
