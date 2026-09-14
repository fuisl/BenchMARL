# M3 — Controlled Transport datasets for M4

2026-09-14. **M3 pilot complete:** fixed data, manifests, coverage summaries,
episode/snapshot leakage checks, replay validation and an M4 sequence reader.
Buzz Wire remains paused. No learned world model has been trained in this milestone.

## What the M2 evidence changes

Read the [proposal](../multi_agent_latent_mpc_proposal.md),
[impact notes](../multi_agent_world_model_impact_notes.md),
[literature review](../litreview.md), [plan](../experiment_plan.md), and both
[Buzz Wire](02_oracle_validation.md) and [Transport](02_transport_comparisons.md)
experiment records. The Transport records were checked against all 33 run
summaries and episode CSVs; their denominators and confounded comparisons are
corrected in the M2 note.

- Buzz Wire R30 reached 11/20 goals; R10 reached 4/20 with higher mean return.
  Reward, success and search budget therefore must remain separate measurements.
- Transport had **zero goals in 660 episode evaluations**, across 33 runs on the
  same 20 development states. These are not 660 independent starting states.
  At fixed C=1 and 100-step episodes, H5→H10 raises mean return .515→.722.
  At 300 steps H5/H10 both pass the return gate on all three seeds, still with
  zero goals. M2's task-success objective remains unvalidated for Transport.
- Reward sparsity, contact coverage and finite search are plausible explanations,
  not a demonstrated causal diagnosis. M3 now measures actual physical effects.
  Keep the existing reward and scenario success definitions; do not redefine
  goal progress as success.
- A learned controller might beat finite-budget oracle CEM through search or
  approximation effects. That would not demonstrate dynamics more accurate than
  VMAS. The supported research direction remains interaction structure →
  counterfactual prediction → candidate ranking → control, comparing the three
  learned baselines under matched conditions. Benchmark-wide superiority is a
  hypothesis outside the evidence established here.

## Protocol and data contract

Use BenchMARL's default Transport task: four agents, one package, 100 primitive
steps. Model observations remain the default 11 features per agent. The collector
uses `VmasTask.get_env_fun`, the existing snapshots including episode clocks,
Hydra, PyTorch/TensorDict and TorchRL's CSV/wandb loggers. Online `Experiment`
collectors/replay buffers are not repurposed into an offline trainer.

Root episodes are assigned to train/validation/test **before** trajectories,
anchors or counterfactuals are constructed. Seeds are state **3100**, split
**3101**, source actions **3102**, branch actions **3103**. All source policies
restore identical initial states. Every child keeps its root episode's split.
The 20 M2 development snapshots are explicitly excluded from the new initial bank.

Both action regimes have uniform individual action marginals in native bounds:

- **Independent:** every agent/action coordinate samples independently.
- **Correlated:** independent absolute normalized magnitudes, with signs shared
  across agents for each coordinate. Matched sampling seeds preserve magnitudes.
- **Held-out combination:** normalized `a0_x * a1_x < 0`. It is absent from
  correlated training and has probability 1/2 under independent sampling.
  Test probes flip only agent1's x action at each primitive step; all other
  agents' actions stay fixed. This tests a specified support change, not every
  possible unseen action combination. The independent regime intentionally
  supports this region as the coverage control.

Source trajectories supply snapshots every 20 steps while alive. Both action
regimes branch for up to 25 steps from **every same anchor**. Scratch environments
use a constant width of 256, padding the last batch; this avoids simulator-width
rounding differences between the full bank and smaller counterfactual subsets.
First transitions have identical input states and counts. A first five-action
block also starts from an identical state, but subsequent states depend on those
actions. Arbitrarily shifting windows along branches or training on ordinary
source trajectories would lose the exact input-state control; report that as a
separate multi-step/state-distribution experiment.

| Artifact | Contents |
|---|---|
| `initial_states.pt` | Root IDs, split assignments and full initial snapshots |
| `trajectories_{independent,correlated,heuristic}.pt` | Source transitions `(128,100,...)`, padded after first termination |
| `anchors.pt` | Snapshot bank, source episode, source policy and source clock |
| `samples_{independent,correlated}.pt` | One paired snippet per anchor; observations/actions/next observations/rewards, masks and physical diagnostic states |
| `counterfactual_test.pt` | Test-only intervention snippets and anchor IDs; reference is the matching correlated sample |
| `manifest.json` | Schema, native bounds, seeds, task, state identities and artifact SHA256s |
| `coverage.json`, `leakage_checks.json` | Support, marginals, physical effects and replay/leakage evidence |
| `resolved_config.yaml`, `provenance.json`, `source/` | Exact configuration, code/dependency versions and source hashes/copies |
| `timing.json`, `gpu_memory.csv`, `batch.log` | Runtime, allocation memory samples and retained stdout/errors |

Transition fields are observation/next observation `(S,T,4,11)`, action
`(S,T,4,2)`, per-agent reward `(S,T,4,1)`, and boolean valid/done/terminated/truncated
`(S,T)`. Diagnostic agent/package physics stores absolute position, velocity,
rotation and angular velocity; these and snapshots are **not model inputs**.
Include terminal-step reward; zero subsequent records. VMAS's TorchRL adapter
copies timeout into `terminated`, so this collector derives actual task termination
from `scenario.done()` and gives it precedence over timeout. No automatic resets
or transitions across episode boundaries are stored.

## Pilot sequence and results

**Job 1187**, first two source policies only: 128 roots, 1,280 anchors,
30,720 valid branch transitions per regime. Data collection took **8.83 s**,
Slurm elapsed **22 s**, sampled MIG peak **326 MiB / 20,096 MiB**, no OOM.
Only **1/160** test anchors showed a package intervention effect, and 1/160 an
effect on another agent, over 25 primitive steps. This is insufficient support
for expecting architecture alone to discover broad interaction behavior.
Artifacts: `outputs/transport_data_1187/`;
[wandb](https://wandb.ai/cair-traffic/counterfactual-wm/runs/0fu0so1w).

**Job 1190**, the M4 pilot dataset: add the installed VMAS Transport
`HeuristicPolicy` as a third **source of states**. This is a deliberate response
to measured coverage weakness and M3's cooperative-policy trajectory item.
It uses default observations and the shipped controller unchanged. Its geometry
assumes one default-sized package, which the collector checks explicitly.
The training branch policies remain independent/correlated; do not mix heuristic
source actions into the restricted-coverage training set.

| Source policy | Valid transitions | Mean-agent episode return | Goal episodes | Moving-package transition fraction |
|---|---:|---:|---:|---:|
| Independent | 12,800 | .00220 | 0/128 | 1.13% |
| Correlated | 12,800 | .000163 | 0/128 | .23% |
| VMAS heuristic | 12,761 | 5.135 | 3/128 | 71.04% |

The heuristic goal estimate is **2.34% [0.80%, 6.66%]**, 95% Wilson. This is a
useful interaction-state provider, not a high-success expert or a matched-state
comparison with the M2 controller. A trained competent cooperative policy remains
a future dataset extension, using BenchMARL's collection/checkpoint interfaces.

The resulting bank has **1,919 anchors** (1,663 unique snapshots; shared reset
states intentionally repeat across source policies). Splits contain **96/16/16
root episodes** and **1,440/240/239 anchors**. Both regimes contain exactly
**34,560 / 5,760 / 5,740** valid train/validation/test primitive transitions,
or **46,060 each**. The first-step matched-state budget is 1,440 training examples
per regime; the larger transition count is not 34,560 independent anchor states.

Independent train actions enter the excluded joint region **50.05%** of the time;
correlated train actions enter it **0%**. Maximum absolute marginal means are
.00525 and .00298 respectively; coordinate standard deviations are .575–.579,
consistent with uniform `[-1,1]` marginals. Full 10-bin marginal histograms are
retained, including validation/test summaries.

Measure intervention effects using absolute position/velocity differences above
`1e-6`, masking once either branch terminates. Changing relative observations
alone does not count as physical coupling.

| Probe horizon | Package effect anchors | Other-agent effect anchors |
|---|---:|---:|
| 1 primitive step | 0/239 | 0/239 |
| 5 primitive steps | 21/239, spanning 9/16 roots | 22/239, spanning 12/16 roots |
| 25 primitive steps | 32/239, spanning 15/16 roots | 33/239, spanning 16/16 roots |

Most effects come from heuristic-source states: 31/79 package and 32/79 other-agent
effect anchors over 25 steps. Repeated anchors are correlated, so these counts
are coverage diagnostics, not independent-binomial confidence intervals.
Evidence: `dataset_audit.json`, with reproducible `audit_dataset.py` beside it.

The absent one-primitive-step effect is consistent with installed VMAS Transport's
single physics substep: action and contact forces are computed before integrating
positions. Effects through changed contact geometry appear later. Use the
**five-primitive-action block** for the first M4 counterfactual comparison and
report the one-step negative control; do not conclude a model misses action
coupling from a probe where the true simulator has none.

Job 1190 collected in **12.29 s**; total Slurm elapsed **26 s**, exit **0:0**.
Sampled MIG peak was **356 MiB / 20,096 MiB**, no observed OOM. Process PyTorch
peaks were 9.45 MiB allocated / 22 MiB reserved; those exclude CUDA/runtime memory.
All `.pt` artifacts total **86.7 MiB**. Dataset root: **`outputs/transport_data_1190/`**;
[wandb run](https://wandb.ai/cair-traffic/counterfactual-wm/runs/e9u63pn7).
Initial-state content SHA256:
`3d33537c1a525dfd213032354e9a4c1e662bdd23cd69c555473c1775aebbb9b6`.
Anchor content SHA256:
`974a05c08942603d787e0f3e7861d47c97b18cead07911944c6fa9fbbd5654a5`.

Failed attempts are retained: **1188** stopped at Hydra argument parsing; moving
configuration flags before positional overrides fixed the launcher. **1189**
collected partial data but failed before the manifest because VMAS dynamically
loads `Scenario`, preventing `inspect.getfile(type(scenario))`. Hashing the explicit
installed module fixes provenance. Its wandb run is marked `collection_complete=0`;
neither failed directory is an M4 dataset. The final path was then tested end to end.

Historical M2 memory CSVs used the parent GPU rather than the MIG slice; their
peaks are invalid. The corrected monitor resolves `cuDeviceGetUuid_v2` inside the
allocation and reads the exact MIG instance. A scheduled probe and both successful
M3 jobs verify the 20,096 MiB device. PyTorch's device-properties UUID alone returns
the parent UUID on this driver and is insufficient.

## Validation, commands and M4 handoff

**39 focused tests** cover existing oracle behavior, true simulator replay,
source ordering/isolation, task vs timeout endings, terminal rewards, held-out
support, unchanged marginals, episode/snapshot/excluded-bank leakage, controller
feedback, physical-effect negative controls, file corruption, block ordering,
partial masks, TensorDict batches and complete Hydra collection/reload. All
**13 existing self-checks** pass. Pinned ufmt/flake8, YAML parsing and shell syntax
checks pass. CPU and GPU smoke tests explicitly use CSV only.

The final bank passes: root-split inheritance, zero duplicate snapshots across
splits, zero overlap with 20 excluded M2 states, bit-exact replay of 256 anchor
snippets, and bit-exact replay of all 239 counterfactual reference snippets.
Checksums, finite tensors, bounds and temporal/termination alignment are verified
when loading either regime. Six regime/split readers were validated in the job.

Commands from the repository root:

```bash
# Disposable CPU check (CSV only).
.venv/bin/python -m examples.world_model.collect \
  --config-name sweep/offline_data_smoke dataset.include_heuristic=true \
  hydra.run.dir=outputs/m3_cpu_check

# Initial independent/correlated pilot (job 1187).
sbatch scripts/slurm/transport_data.sbatch \
  dataset.excluded_states_file=outputs/transport_oracle_1182/initial_states.pt

# Canonical coverage-enriched M4 pilot (job 1190), CSV + wandb.
sbatch scripts/slurm/transport_data.sbatch \
  --config-name sweep/offline_data_coverage \
  dataset.excluded_states_file=outputs/transport_oracle_1182/initial_states.pt

# Validation; no wandb runs created by tests.
OMP_NUM_THREADS=1 .venv/bin/python -m pytest \
  test/test_world_model.py test/test_world_model_collection.py \
  test/test_world_model_dataset.py test/test_gpu_memory.py -q
```

Launches used parent commit `ef65548` plus the exact uncommitted source snapshots
and hashes in each run's `source/` and `provenance.json`. New data/config files are
local research extensions; large artifacts remain ignored by git. Broader data
or training sweeps still belong on the H100 cluster under the compute rules.

```python
import torch
from torch.utils.data import DataLoader
from examples.world_model.dataset import OfflineSequences

data = OfflineSequences("outputs/transport_data_1190", "correlated", "train", action_block=5)
batch = next(iter(DataLoader(data, batch_size=32, collate_fn=torch.stack)))
# observation: (B,6,4,11); action: (B,5,4,10)
# First block is the controlled comparison: obs[:,0] -> obs[:,1], action[:,0].
# primitive_action: (B,5,5,4,2), ordered block, primitive time, agent, coordinate.
```

`valid` marks complete blocks, `observation_valid` marks usable encoded frames,
and `primitive_valid` retains partial terminal blocks for reward/termination
supervision. Mask padding in prediction **and anti-collapse** computations.
Fit shared observation normalization using valid training inputs only; use native
action bounds. Source physics/snapshots must remain excluded from model inputs.

For M4 use official LeWM revision **`8edfeb336732b5f3ce7b8b210d0ba370a09e2cac`**:
[training loop](https://github.com/lucas-maes/le-wm/blob/8edfeb336732b5f3ce7b8b210d0ba370a09e2cac/train.py),
[predictor/SIGReg](https://github.com/lucas-maes/le-wm/blob/8edfeb336732b5f3ce7b8b210d0ba370a09e2cac/module.py),
[training settings](https://github.com/lucas-maes/le-wm/blob/8edfeb336732b5f3ce7b8b210d0ba370a09e2cac/config/train/lewm.yaml),
[sequence settings](https://github.com/lucas-maes/le-wm/blob/8edfeb336732b5f3ce7b8b210d0ba370a09e2cac/config/train/data/pusht.yaml),
and [rollout interface](https://github.com/lucas-maes/le-wm/blob/8edfeb336732b5f3ce7b8b210d0ba370a09e2cac/jepa.py).
The reference uses prediction MSE plus SIGReg, no detached prediction target,
latent size 192, history 3 and SIGReg weight .09. Its five-step action blocks
concatenate distinct actions. Preserve our episode splits, training-only
statistics and masks instead of copying upstream random window splitting or
boundary NaN replacement.

Next implement independent, joint-concatenated and relational predictors with
matched encoders, loss and optimization budgets; verify variance, gradients,
multi-step predictions and checkpoint reloads on this same fixed bank. First
test five-step transitions from matched anchors; history/multi-step changes are
explicit subsequent comparisons. Reuse suitable BenchMARL MLP/GNN conventions.
Before M5, validate the common reward/termination readout required by `J=-sum r`;
LeWM's latent-goal cost and two-term training alone do not implement that objective.

This is one **development data seed**, inspected while choosing the source mixture.
The test split is excluded from training but is not untouched headline confirmation.
Keep source-stratified diagnostics, retain the random-only negative finding, and
use fresh held-out states/seeds for later scientific confirmation. Neither M3's
coverage improvement nor the heuristic's three goals establishes a learned-model
or full-benchmark performance claim.
