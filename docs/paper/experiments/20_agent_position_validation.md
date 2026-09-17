# Gate 0b — can a shared per-agent model predict physical position?

**Status:** **complete**. Job **1291** finished all 144 registered fits on
2026-09-17 in 9m26s with exit code 0. Six fits ran concurrently inside one
20 GB MIG allocation. The earlier one-job-per-seed layout, jobs 1283–1290, was
cancelled at the user's request after job 1283 ran for 2m41s; those partial
artifacts are infrastructure evidence only and are excluded below.

The latent rollout tables answer whether a predicted latent stays close to the
same encoder's latent target. They do **not** answer whether an agent is in the
right physical location. The imagination figures add a learned probe, so their
visible position error combines two terms:

1. physical position not being recoverable from the true latent; and
2. the predicted latent differing from the true latent.

This gate removes both ambiguities. It asks the smaller question first:

> Can one transition function, shared across agent identities, predict each
> agent's next simulator position from the current input and action block?

## Model and three information paths

[`SharedAgentPositionModel`](../../../examples/world_model/models.py) predicts
the physical displacement `(dx, dy)` over one model action block. In the current
banks that is five primitive simulator steps. It is not yet a long-horizon
rollout model.

All agents use the same observation encoder, action encoder, and output head.
The three variants match the main experiment's information contracts:

| Variant | Information available for agent `i`'s prediction |
|---|---|
| `independent` | agent `i`'s observation and action only |
| `joint` | every agent's observation/action in fixed order |
| `relational` | agent `i` plus a sum of learned pairwise messages |

The variant-specific conditioner widths are solved against one parameter
budget, while the encoders and output head are identical. Model and conditioner
parameter counts are saved with every run.

The evaluator can repeat those three variants under the established
`observation`, `history`, and `physical` input conditions. This separates two
questions that the latent rollout number mixes: whether the information is
present and whether a particular latent representation preserves it.

## Protocol

- Fit observation, action, and displacement normalization on valid **training**
  transitions only.
- Select the checkpoint by validation standardized MSE with early stopping.
- Report the final metrics once on the episode-disjoint **test** split.
- Report coordinate and Euclidean RMSE in simulator units, per-axis and
  per-agent results, test-root count, and the error for every root.
- Normalize the main score by a persistence baseline that predicts no movement.
  A relative MSE below `1.0` means the learned transition predicts motion better
  than copying the current position.
- Keep correlated and independent action regimes separate.
- Treat one training seed as a pipeline pilot, not an architecture conclusion.

The dataset now exposes `agent_state` at exactly the same block boundaries as
the observation sequence. The physical state is a target only; it becomes an
input solely when `state_input=physical` is explicitly selected.

## Level-0 smoke result

A five-epoch CPU run on the existing Buzz Wire bank, correlated regime,
observation input, seed 4100 exercised real loading, training, validation,
checkpointing, and test evaluation:

| Variant | Coordinate RMSE | MSE / no-motion MSE | Improvement over no motion |
|---|---:|---:|---:|
| independent | 0.00783 | 0.194 | 80.6% |
| joint | 0.00775 | 0.190 | 81.0% |
| relational | 0.00751 | 0.178 | 82.2% |

Both agents individually beat persistence by about 80–83%, and all predictions
were within 5 cm. This says the new measurement is learnable and the pipeline is
wired correctly. It is **not a paper result**: it used one seed, five epochs,
one regime, and the test split during implementation verification. The
registered pilot must write a fresh result directory, and any selected claim
must later be confirmed on untouched roots or a new bank.

Most importantly, this smoke result does not rehabilitate the latent model. It
shows that a small directly supervised model can predict one-block agent motion.
The comparison with latent imagination becomes informative only after both are
reported in the same physical units on the same test roots.

## Full registered sweep

```bash
sbatch scripts/slurm/agent_position_validation.sbatch

# Select another existing bank without editing the script.
POSITION_DATA=outputs/balance_repair_1236/data \
  sbatch scripts/slurm/agent_position_validation.sbatch
```

Job 1291 ran all eight training seeds across both action regimes, all three
information paths, and all three input conditions: 144 fits total, six at a
time. Results and checkpoints are written under
`outputs/agent_position_concurrent_1291/` with commit, dirty-tree patch, source
snapshot, configuration, per-root metrics, per-fit logs, GPU memory, and
training histories. Slurm reports `COMPLETED`, exit `0:0`, 16 CPUs, 48 GB RAM,
and peak process RSS 5.15 GB. All 144 result directories contain a checkpoint
and `summary.json`; the batch log records 144 `DONE` and no `FAILED` events.
Peak sampled GPU allocation was 1,836 MiB of the 20,096 MiB slice. The slice was
fully reserved as requested, although these small MLPs are compute/CPU-feed
limited rather than memory limited.

## Results

The table reports means over training seeds 4100–4107 on the fixed 16-root test
split. `MSE / persist.` is coordinate MSE divided by the no-motion baseline;
lower is better and `1.0` is persistence. The persistence coordinate RMSE is
0.01777 in the correlated bank and 0.01766 in the independent bank.

| Input | Action regime | Conditioner | Coordinate RMSE | MSE / persist. | Error removed | Within 1 cm |
|---|---|---|---:|---:|---:|---:|
| observation | correlated | independent | 0.00640 | 0.1295 | 87.0% | 73.2% |
| observation | correlated | joint | **0.00571** | **0.1031** | **89.7%** | **79.3%** |
| observation | correlated | relational | 0.00588 | 0.1094 | 89.1% | 77.4% |
| observation | independent | independent | 0.00764 | 0.1869 | 81.3% | 60.8% |
| observation | independent | joint | **0.00612** | **0.1201** | **88.0%** | **77.8%** |
| observation | independent | relational | 0.00641 | 0.1317 | 86.8% | 74.4% |
| history | correlated | independent | 0.00630 | 0.1255 | 87.4% | 74.8% |
| history | correlated | joint | **0.00562** | **0.0998** | **90.0%** | **80.8%** |
| history | correlated | relational | 0.00565 | 0.1012 | 89.9% | 80.3% |
| history | independent | independent | 0.00728 | 0.1697 | 83.0% | 64.8% |
| history | independent | joint | **0.00610** | **0.1191** | **88.1%** | **77.0%** |
| history | independent | relational | 0.00621 | 0.1236 | 87.6% | 76.4% |
| physical | correlated | independent | 0.00374 | 0.0443 | 95.6% | 96.0% |
| physical | correlated | joint | 0.00247 | 0.0193 | 98.1% | 99.1% |
| physical | correlated | relational | **0.00246** | **0.0192** | **98.1%** | **99.3%** |
| physical | independent | independent | 0.00487 | 0.0761 | 92.4% | 88.0% |
| physical | independent | joint | 0.00322 | 0.0334 | 96.7% | 97.4% |
| physical | independent | relational | **0.00302** | **0.0292** | **97.1%** | **98.0%** |

The train-set mean-displacement baseline has relative MSE 1.006 in the
correlated regime and 1.001 in the independent regime. The improvement is
therefore learned conditional motion, not merely a nonzero average drift.

### Paired architecture result

The same eight seeds are paired within every input/regime cell. Percentile
bootstrap intervals below resample those eight paired seed differences. A
negative difference favors the conditioned model.

| Input | Regime | joint − independent | relational − independent |
|---|---|---:|---:|
| observation | correlated | −0.0264 [−0.0288, −0.0235] | −0.0201 [−0.0227, −0.0174] |
| observation | independent | −0.0669 [−0.0684, −0.0652] | −0.0552 [−0.0574, −0.0528] |
| history | correlated | −0.0257 [−0.0268, −0.0246] | −0.0243 [−0.0274, −0.0215] |
| history | independent | −0.0506 [−0.0526, −0.0488] | −0.0461 [−0.0490, −0.0431] |
| physical | correlated | −0.0250 [−0.0259, −0.0240] | −0.0251 [−0.0260, −0.0242] |
| physical | independent | −0.0428 [−0.0454, −0.0396] | −0.0469 [−0.0484, −0.0447] |

Both conditioned models beat the independent model on **8/8 seeds in all 12
comparisons**. This establishes that other-agent/shared context improves this
one-block target on this bank. The gain is also distributed across test roots:
after averaging seeds within each root, conditioned models improve 16/16 roots
in eleven comparisons and 15/16 for relational observation input under
correlated actions. It does not establish a universal relational
advantage. Joint beats relational reliably for observation input; relational
beats joint reliably for physical input under independent actions; the other
cells are smaller or mixed. In particular, joint − relational is −0.00138
with a 95% interval [−0.00372, +0.00094] for correlated history and +0.00010
[−0.00089, +0.00119] for correlated physical input.

### Input result

Physical input reduces relative MSE versus observation in every architecture,
regime, and seed. The paired mean reduction ranges from 0.0838 to 0.1108, and
all six 95% intervals exclude zero. Physical input contains the shared object
state as well as the agents' states, so this is evidence that omitted shared
state matters even for predicting an individual agent's motion.

History gives only a modest gain. Five of six mean comparisons favor history,
but the independent-action joint model changes by only −0.0009 with interval
[−0.0043, +0.0020]. History cannot be described as a uniformly effective
substitute for the physical state.

The result is not driven by one identity. Averaged per-agent relative MSEs are,
for example, 0.0831/0.0694 for the physical-independent model under independent
actions and 0.0292/0.0292 for the relational counterpart. Every headline
improvement appears for both agents.

## Decision and scientific boundary

**Gate 0b passes for one-step direct motion.** A transition function shared
across agents predicts each agent's position five simulator steps ahead, and it
does so far better than persistence in every registered cell. Cross-agent
conditioning and access to shared physical state both provide repeatable gains.

This result does **not** validate the latent world model used for planning:

- this model is trained directly on `(dx, dy)`; the planning world model emits a
  latent vector and never consumes this position head;
- this is teacher-forced one-block prediction, not an autoregressive rollout;
- all eight fits reuse one data bank and one fixed set of 16 test roots, so the
  seed intervals measure optimization variation, not new-bank uncertainty;
- position accuracy alone does not measure the ball/wire state, reward, task
  progress, plan ranking, or closed-loop success.

The rendering mismatch is therefore **not evidence that Buzz Wire motion is
intrinsically unlearnable**. [Experiment 21](21_latent_position_validation.md)
performs the localized comparison: the direct model passes while every latent
pipeline is worse than persistence at one block, even with a position head
adapted to predicted latents. The largest loss is representation/one-step latent
transition; recursive rollout adds further error but is not the first failure.
