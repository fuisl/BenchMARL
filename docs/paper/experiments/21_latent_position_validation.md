# Gate 0c — does the latent world model preserve physical position?

**Status:** complete. MLP-probe job **1294** evaluated all 144 Stage-2 latent
checkpoints in 5m02s; linear-probe control job **1295** completed in 30s. Both
used one 20 GB MIG allocation with six input/regime groups concurrent. Both
jobs finished `COMPLETED`, exit `0:0`, with no missing cells.

Experiment 20 established that the underlying one-block motion problem is easy
for a directly supervised transition. This experiment applies the same physical
target to the existing latent world models and separates representation,
transition, readout-distribution, and recursive-rollout error.

## Question and measurement

For each of the 144 checkpoints from job 1223 (8 seeds × 3 state inputs × 2
action regimes × 3 conditioners), evaluate agent `(x,y)` on the same
episode-disjoint test split as experiment 20.

Two position heads are necessary because the planner and the alignment test ask
different questions:

1. **True-latent head:** fit `encoder(observation) -> position` on train roots.
   Applying it to true test latents measures probe/representation error;
   applying the same head to predicted latents measures whether the transition
   stays in the encoder's coordinate system.
2. **Predicted-latent head:** fit teacher-forced
   `transition(latent, action) -> next position` on train roots. This matches
   the planning readout's contract and tests whether physical position remains
   recoverable even if predicted latents are systematically shifted from true
   latents.

Regularization is selected on validation roots. Test roots never fit either
head. MLP and linear families are run separately because experiment 14 showed
that probe family can otherwise choose an apparent architecture winner.

The measurement reports:

- true-latent probe error;
- teacher-forced one-block error through each head;
- recursive rollout error at horizons 1–5 blocks (5–25 simulator steps);
- the direct `(dx,dy)` model from job 1291 on the identical target/split;
- persistence, which predicts that the agent does not move.

## Main MLP result

These are means over both regimes, all three conditioners, and eight training
seeds. RMSE is per coordinate in simulator metres. `MSE / persistence` belongs
to the predicted-latent head; values above 1 mean it is worse than copying the
current position.

| Input | True-latent probe | True-head one block | Predicted-head one block | Direct `(dx,dy)` | MSE / persistence | Predicted-head h=5 |
|---|---:|---:|---:|---:|---:|---:|
| observation | 0.01594 | 0.11729 | 0.03153 | **0.00636** | 3.17 | 0.05517 |
| history | 0.02421 | 0.11248 | 0.03396 | **0.00619** | 3.68 | 0.04897 |
| physical | 0.02652 | 0.07055 | 0.03113 | **0.00330** | 3.09 | 0.05231 |

The true-head result proves that predicted latents are strongly shifted from
the encoder's latent coordinate system. Refitting on predicted latents recovers
much of that shift: roughly 7–12 cm falls to about 3 cm. It does not recover an
accurate physical transition.

Across the **144 individual MLP runs**:

- every predicted-latent head is worse than persistence: relative MSE ranges
  **2.58–4.02**;
- every predicted-latent head is worse than its matched direct model: the RMSE
  ratio ranges **3.84–13.13**, mean **6.85**;
- only 5.9–6.6% of predictions lie within 1 cm, versus 60.8–99.3% for the
  direct models;
- recursive h=5 error is **1.24–2.29×** h=1 error.

The failure is therefore present before long-horizon accumulation. Rollout
drift makes it worse, but it is not the primary cause.

## Probe-family control

| Head family | Observation one block | History one block | Physical one block | Mean latent/direct ratio | All 144 worse than persistence? |
|---|---:|---:|---:|---:|---|
| MLP on predicted latents | 0.03153 | 0.03396 | **0.03113** | 6.85× | yes |
| Linear on predicted latents | **0.03254** | 0.03956 | 0.05035 | 9.27× | yes |

Linear and MLP heads disagree about how easily each representation decodes, as
expected from experiment 14. They agree on the scientific conclusion. The best
individual linear ratio is still 3.74× the direct model, and its one-block MSE
is also worse than persistence in all 144 cells.

## Architecture and input interpretation

Conditioner choice is not the bottleneck. Within each MLP input/regime cell,
the spread between independent, joint, and relational one-block RMSE is only
0.00030–0.00285 m. No architecture wins consistently.

Physical input makes the direct model dramatically better (mean 0.00330 m) but
does not make the predicted-latent readout better than observation input
(0.03113 versus 0.03153 m). Position is present explicitly in every agent's
base observation, and physical input adds the shared bodies; the latent
training pipeline does not retain and transition that information at the
precision the physical predictor demonstrates is available.

This refines the diagnosis:

1. **Representation/readout floor:** true positions are not cleanly decoded
   from true latents, especially for history and physical inputs.
2. **Latent alignment:** predicted latents leave the true-latent decoder's
   coordinate system immediately.
3. **Usable predicted state:** a planning-style head adapts to that shift but
   remains worse than no motion for one-block position prediction.
4. **Rollout:** recursive use adds error, but the decisive loss happens at the
   first transition.

## Rendered evidence

Job 1294 writes aggregate figures and a filmstrip for every input/regime pair:

- `outputs/latent_position_mlp_1294/one_step_comparison.png` — true-latent
  floor, true-head transition, predicted-head transition, and direct model;
- `outputs/latent_position_mlp_1294/rollout_curves.png` — physical RMSE through
  five recursive blocks;
- `outputs/latent_position_mlp_1294/<input>_<regime>/filmstrip.png` — simulator
  positions, true-latent reconstruction, and predicted-latent-head imagination
  on the same moving test snippet.

The filmstrips make the numerical diagnosis visible: the decoded imagined
agents move in the broadly correct direction but sit several centimetres from
their simulator positions at the first block and continue to drift.

Linear-control versions of every figure are under
`outputs/latent_position_linear_1295/`.

## Decision

**The latent world model fails physical validation.** Buzz Wire motion is
learnable, but the current latent objective does not preserve sufficiently
accurate individual-agent position for planning. This rules out the explanation
that the qualitative rendering mismatch is merely an unfortunate episode or a
single decoder family.

The next model change should keep the planner latent-only while adding an
explicit physical consistency constraint during training—for example, an
auxiliary agent/object-state decoder on true and predicted latents—and require
that its frozen predicted-latent readout beat persistence before any new control
sweep. Merely changing independent/joint/relational conditioning is not
supported by this result.
