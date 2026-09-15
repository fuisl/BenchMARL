# Experiment plan

Build on [direction.md](direction.md), the [proposal](multi_agent_latent_mpc_proposal.md), and the [impact notes](multi_agent_world_model_impact_notes.md). Use the broader [literature review](litreview.md) as background. Follow [coding_rules.md](coding_rules.md) for implementation.

**Main question:** Does explicit interaction modelling improve prediction of unseen joint-action combinations, plan ranking, and closed-loop multi-agent MPC?

**First target:** one suitable VMAS task with its default configuration, three learned models, and simulator-based oracle MPC working end to end. Choose the eventual task suite by the evidence needed for the research question. Team size follows the task default; it is not a required sweep.

**Design principle:** organize experiments around **interaction structure → counterfactual generalisation → plan ranking → control**. Keep baseline defaults and revise this plan as pilot evidence reveals what is needed. Numerical settings are recorded for reproducibility, not fixed permanently per task.

## Current repository and environment

Snapshot: 2026-09-11, commit `41774c5` (`slurm update`). The working tree was clean before these documents were added.

| Area | What exists | How we build on it |
|---|---|---|
| Environment integration | `benchmarl/environments/vmas/` and task YAMLs under `benchmarl/conf/task/vmas/` | Reuse task creation, continuous actions, vectorization, and TensorDict observations. |
| Candidate tasks | Existing VMAS scenarios and the [task survey](vmas_task_survey.md) | Select complementary interaction mechanisms through M1. Task names alone do not establish an ordering of interaction strength. |
| Model components | MLP, GNN, DeepSets, CNN, GRU, LSTM; model/config interfaces | Reuse suitable components and conventions for encoders and predictors. Existing policy/critic models are not yet world models. |
| Baseline training | `Experiment`, MAPPO/IPPO and other algorithms, collectors, replay buffers, callbacks | Use for cooperative behaviour-policy training and collection. |
| Experiment support | Hydra YAMLs, seeds, CSV logging, evaluation, checkpointing, plotting examples | Reuse configuration and reporting conventions. Existing online RL training is not an offline world-model trainer. |
| Launching | Packed Slurm launcher, local/MIG configs, small VMAS sweeps, `docs/packed_slurm.md` | Validate a single process before using existing sweeps. Measure resource use before increasing concurrency. |
| Extension guidance | `examples/extending/`, `docs/source/concepts/extending.rst`, `test/` | Follow existing task/model/algorithm boundaries and test style. |

Local checks:

The following records the initial scan, not a continuously updated environment inventory.

- `.venv` uses Python **3.11.15**; BenchMARL **1.5.2** imports from this checkout.
- Installed: PyTorch **2.7.1**, TorchRL **0.11.0**, TensorDict **0.11.0**, NumPy **1.26.4**, Hydra **1.3.6**, Hydra Submitit launcher **1.2.0**.
- CUDA is available; two NVIDIA A100 GPUs with 40 GB each are visible. This is device visibility, not a scheduling allocation.
- `uv`, `sbatch`, and `srun` are available. Slurm submission and cluster allocations were not tested; the checked-in MIG configuration targets different hardware.
- **VMAS is missing.** Attempting to construct the navigation environment failed with `ImportError: vmas python package was not found`. No environment rollout or training run was validated.
- `torch-geometric`, `wandb`, and `pytest` are missing. PyTorch Geometric is needed only if using the existing GNN implementation; a small pairwise PyTorch predictor can avoid that dependency. CSV logging needs no W&B setup.
- `pyproject.toml`, `.python-version`, and `uv.lock` already define the dependency setup. No dependencies were changed during this scan.

**Revision check (2026-09-11):** VMAS **1.5.2**, PyTorch Geometric **2.8.0.post1**, W&B **0.30.0**, and pytest **9.1.1** are now installed. This supersedes the missing-package entries above. This revision checked package metadata only; it did not rerun environment or training validation.

**Current M2 revision (2026-09-11):** simulator snapshots (including episode clocks), batched oracle costs, CEM, and a configurable closed-loop Buzz Wire evaluator are implemented under `examples/world_model/`. The first 20-episode Buzz Wire pilot passes M2's return gate and reaches 11 goals at R=30; R=10 reaches 4 goals with higher mean return and fewer collisions. See [oracle validation](experiments/02_oracle_validation.md) for tests, the budget tradeoff, and deferred H100 ablations. Learned-model training, anti-collapse losses, and learned ranking/oracle-gap comparisons remain unimplemented; offline datasets are now available through M3 below.

**Current task selection (2026-09-14):** Buzz Wire experiments are paused at the user's request. Transport replay and outcome semantics are validated. Jobs 1182 and 1183 completed 18 and 15 real runs, respectively, with **0 successes in 660 MPC episode evaluations on repeated versions of the same 20 development states**. The 300-step `replan1` and `horizon10` settings clear the return-vs-random gate on all three planner seeds (mean returns 2.791 and 3.661), but task-success objective validity remains unvalidated. At fixed one-block execution cadence and 100 steps, increasing H=5 to H=10 raises mean return from 0.515 to 0.722; comparison with `lewm`'s 0.259 also changes cadence. Nonzero reward occurs on four development states across job 1182 and ten across the 300-step `horizon10` runs. Poor useful-contact coverage is a working hypothesis, not an established cause. M3 will preserve the task reward and success definitions, measure interaction coverage explicitly, and keep these development states out of its datasets. See [Transport comparisons](experiments/02_transport_comparisons.md) for audited counts, results, confounds, and the invalid historical MIG memory measurements.

**Wheel follow-up (2026-09-14):** the user requested a matched Wheel sweep on the
20 GB MIG: three baselines × two regimes × eight training seeds, five concurrent
workers, followed by counterfactual and plan-ranking evaluation. The
[Wheel audit and protocol](experiments/06_wheel_sweep.md) records native reward
semantics, rotation-aware interaction labels, validation and launch artifacts.
This is the current task request; the Transport/Buzz Wire notes above are history.

The VMAS task adapter currently returns `None` for `state_spec`. Agent observations must not be assumed to contain the full simulator state. Exact counterfactual replay needs explicit handling of relevant simulator and scenario state.

## Milestones

### M0 — Make the existing setup runnable

- Install the existing VMAS extra: `uv sync --frozen --extra vmas`.
- Reset and step candidate tasks; run one small MAPPO experiment with CSV logging and rendering disabled.
- Confirm CPU operation, then CUDA operation on an allocated device. Add test tooling when running the relevant tests.

**Done when:** a reproducible environment smoke check and one training/evaluation iteration pass. Record commands and versions in `experiments/00_setup.md`.

### M1 — Design the evidence needed to answer the research question

Use BenchMARL's existing task and baseline algorithm defaults. Keep task parameters in task YAMLs and training settings in experiment profiles. The new world models and CEM planner need one shared starting configuration; there are no existing defaults for those components in this repo. Override a setting only for an explicit experimental comparison or a demonstrated implementation need.

The immediate deliverable is this question-to-evidence matrix, developed in `experiments/01_protocol.md`:

| Part of the main question | Experiment | Evidence |
|---|---|---|
| Does logged prediction hide counterfactual failure? | Evaluate each model on behaviour-like and recombined joint actions from the same held-out simulator states. | Logged error, counterfactual error, and their gap. |
| Does interaction structure help beyond access to joint information? | Compare joint-concatenated and relational predictors on identical datasets, alongside the independent reference. Vary joint-action coverage. | Counterfactual gap and plan-ranking differences, especially under restricted coverage. |
| Do better counterfactual predictions improve decisions? | Score shared candidate plans using each model and the simulator, then run closed-loop MPC. | Rank correlation, selected-plan regret, success/return, and oracle gap. |
| Is the benefit associated with cross-agent dynamics? | Repeat the same comparisons across complementary interaction mechanisms; measure the effect of changing another agent's action from a restored state. | Whether benefits track measured cross-agent effects; a weak physical-interaction control. |

These are parts of one research question. One-step versus multi-step training remains supporting analysis.

**Task literature review:** do a focused selection review, starting with the [VMAS paper](https://arxiv.org/abs/2207.03530), the existing task survey, and related world-model evaluations. For each candidate, record its interaction mechanism, a concrete joint-action intervention, the resulting observable effect, and how it helps discriminate between the models. Prefer existing supported tasks; add a new domain only if it supplies missing evidence.

Initial candidates, with proposed roles rather than a final ranking:

| Candidate | Proposed role in this paper |
|---|---|
| Dropout | Control with uncoupled agent motion. Shared goal flags, reward, and termination still introduce dependencies, so do not label its entire observation transition independent. |
| Give Way or Passage | Test interactions through collision and shared space. Select one if its intervention outcomes provide evidence beyond object manipulation. |
| Transport | Test how recombining agents' pushes changes a shared object's motion. A strong candidate for the main counterfactual experiment. |
| Wheel | Test coordinated angular-velocity control as a possible complement to positional goals; check whether it fits the simple latent planning objective. |
| Buzz Wire | Literature-motivated alternative with direct mechanical linkage between agents. |

The physical task descriptions are supported by the [official VMAS scenario catalogue](https://github.com/proroklab/VectorizedMultiAgentSimulator#list-of-environments). CoDreamer evaluates Flocking, Discovery, and Buzz Wire; its description of Buzz Wire gives a concrete case where one agent can pull another into a boundary. Its VMAS experiments use discrete actions, so use them as task-selection evidence rather than an identical continuous-MPC protocol. [CoDreamer, evaluation environments](https://arxiv.org/html/2406.13600v1#A4.SS7)

The proposed roles above are our experimental interpretation, to be checked in pilots. Do not assume that a heavier object guarantees stronger useful coupling or that cooperation alone demonstrates coupled dynamics. Keep task defaults for the baseline suite; any interaction-strength modification is a separate, motivated ablation.

**What to do about planning cost:** the agreed M2 pilot uses `J = −Σ_t Σ_i r_i,t`, undiscounted task reward through first termination, held fixed for oracle and future learned-model comparisons. This deliberately replaces LeWM's latent goal objective and will require reward prediction in learned models. Check actual collision-free goals alongside return before accepting the objective. A finite-horizon reward improvement without goals leaves the objective/planning setup unvalidated; investigate horizon and optimizer selection as well as reward alignment. Latent goal scoring is deferred, and any future comparison using it must be labeled separately.

**What to do about settings and observations:** inherit defaults, record actual model inputs, and use shared data/evaluation states and matched budgets within each model comparison. Joint and relational models receive the same joint information. The independent model uses its default per-agent observations and own action; no task-observation redesign is required. Choose pilot budgets from measured runtime and vary budgets or horizons only when they answer a question. Keep held-out evaluation separate from design decisions.

**Done when:** every proposed comparison has a question, intervention, control, metric, and candidate task rationale. No exhaustive task grid or permanently frozen per-task settings are required. The next action is a small intervention pilot on the most informative candidate.

### M2 — Validate the simulator oracle and planner

- Implement minimal snapshot/restore for the selected task, including relevant scenario variables and random state.
- Check that restoring the same state and replaying the same actions reproduces the trajectory, without changing the live evaluation environment.
- Implement centralized CEM-MPC with simulator dynamics first; verify that its objective produces useful control.
- Validate the agreed task-reward objective using goal/collision/timeout outcomes and return. Keep scoring fixed in later oracle-versus-learned comparisons; a latent goal objective would be a separate comparison.
- Use fixed evaluation states and candidate plans for comparable ranking measurements.

**Done when:** deterministic replay checks pass and oracle MPC return exceeds random with non-overlapping 95% episode-bootstrap CIs on fixed evaluation states. Also report collision-free goal success with Wilson intervals: the pilot estimates this rate before choosing a numerical gate for held-out confirmation. Check H=1 versus H=5 at fixed execution cadence and measure search-budget sensitivity; episode return is not guaranteed monotone in CEM budget. Record evidence in `experiments/02_oracle_validation.md`.

### M3 — Build controlled offline datasets

- Collect diverse independent actions, correlated actions, and cooperative-policy trajectories; start with the first two.
- Hold out joint-action combinations while retaining coverage of individual actions. For continuous actions, define held-out regions or correlation changes explicitly.
- Retain useful variation in cross-agent effects. If the data never identifies an interaction, record that limitation rather than expecting architecture alone to recover it.
- Match data budgets and control state-distribution differences where possible, so action coverage is the intended comparison.
- Split by episode; store observations, actions, next observations, termination flags, episode IDs, and simulator snapshots needed for evaluation.

**Done when:** fixed datasets have manifests, coverage summaries, and leakage checks in `experiments/03_datasets.md`.

**M3 protocol revision (2026-09-14):** the first two action regimes are implemented
for Transport under `examples/world_model/collect.py`. Their per-agent action
marginals are uniform; the correlated regime restricts signs across agents, and
test interventions break that correlation. Independent/correlated source
trajectories supply a shared anchor bank. Both regimes branch from each exact
anchor, and every child inherits its root episode's split. Only the initial
transition/block is an exact state-distribution-controlled comparison; later
branch states depend on the sampled actions. Physical interaction diagnostics
must distinguish absolute state changes from relative-observation changes.
The M4 reader preserves action blocks, primitive rewards, terminal masks, and
episode identity. See [M3 datasets](experiments/03_datasets.md) for artifacts,
measured coverage, validation, and the remaining cooperative-policy extension.

**M3 pilot result:** the random-only bank (job 1187) had physical package effects
in only 1/160 test anchors. Adding VMAS's shipped Transport heuristic as a third
source of shared states produced the completed **job 1190** bank: 128 root
episodes, 1,919 paired anchors, 46,060 primitive transitions per regime, and
32/239 test anchors with package intervention effects over 25 steps (21/239
within five). The source heuristic reached 3/128 goals; it supplies useful contact
states but is not a high-success expert. All manifests, episode/snapshot leakage
checks, replay checks and M4 reader checks pass. This satisfies the initial M3
dataset gate; trained cooperative-policy data and independent confirmation remain
later extensions. Proceed to M4 using `outputs/transport_data_1190/`.

### M4 — Establish three world-model baselines

- Implement independent, joint-concatenated, and relational latent predictors.
- Keep encoder design, latent size, loss, data, and optimization budgets consistent; match parameter counts where practical and report them.
- Use the same prediction plus anti-collapse objective. Check latent variance and one-step/multi-step prediction before connecting MPC.
- Treat raw MSE across separately learned latent spaces cautiously; use common task-level metrics for headline comparisons.

Use the official LeWM reference revision recorded in the M3 note for encoder,
predictor and SIGReg semantics. Preserve our episode splits and explicit masks;
do not copy upstream window-level random splitting or fit normalization on
validation/test samples. Establish the controlled first-block baseline before a
multi-step ablation. The task-reward/termination readout needed for M5 is a separate
explicit interface to validate; two-term latent training alone does not supply it.

**Done when:** all three models train and reload reproducibly on the same pilot dataset. Record results in `experiments/04_model_baselines.md`.

**M4 pilot result (2026-09-14, job 1192):** gate met. Three baselines x two action
regimes x three seeds, conditioner widths solved so dynamics parameter counts match
to within 0.02% (3.815M each); checkpoint reload is bit-exact for all 18 runs; no
latent collapse. One-step teacher-forced prediction does **not** separate the
baselines; multi-step rollout does, with relational 13.0%/13.8% below independent
under correlated/independent actions. Two cautions carried forward: the relational
gain is the same size in both regimes, so the coverage contrast the impact notes
predict does not appear; and sum pooling being better conditioned is a live
non-interaction explanation, so M1 Row 4's Dropout control was run (job 1194,
eight seeds, both tasks): the relational advantage holds on Transport for 8/8
seeds in both regimes with bootstrap intervals clear of zero, and does **not**
transfer to Dropout (3/8 seeds, intervals containing zero). That is Row 4's
well-behaved outcome, bounded by two caveats recorded in the M4 note -- Dropout's
correlated interval still contains the Transport-sized effect, and Dropout
differs from Transport in more than interaction. The reward readout works (R^2 ~ 0.74 on frozen dynamics); the
termination head has **zero positive examples in this dataset** and is explicitly
unvalidated, which remains an open blocker for `J = -sum r` under learned MPC. The
SIGReg objective sits near an unreachable floor set by latent rank deficiency
(~330:1 against the prediction term); candidate fixes are recorded but unrun. See
[M4 baselines](experiments/04_model_baselines.md).

**M5 link 1 result (2026-09-14, jobs 1194 and 1196):** C7 is measured. On
Transport it is negative -- no baseline captures any of the cross-agent effect
(relational at 1.002x a no-response floor), and four independent measurements
agree that M4's rollout advantage there is not interaction modelling. The cause
is upstream: 0/239 anchors show an effect after one primitive step. Running the
identical pipeline on **Buzz Wire**, whose rigid joint couples the agents
structurally, reverses it: relational captures **26%** of the effect on 8/8 seeds
in both regimes and beats joint-concatenated on 7/8, with joint holding identical
information. The benefit appears exactly where measured coupling does, which is
C10. Buzz Wire's bank also supplies the termination positives (252) that both
other tasks lack. See
[counterfactual prediction](experiments/05_counterfactual_prediction.md).

### M5 — Run the central experiment

- Measure logged versus counterfactual prediction error and their gap.
- Measure plan-ranking correlation and the true cost of each model's selected plan on shared candidate sets.
- Evaluate closed-loop success/return and the gap to simulator-based MPC using matched CEM horizons and search budgets.
- Expand only to the task mechanisms and data regimes needed by the M1 evidence matrix. Keep task defaults for the baseline suite.

**Done when:** `experiments/05_main_results.md` connects prediction, ranking, and control for all models, including negative findings. Oracle MPC is a dynamics reference, not a guaranteed globally optimal controller.

### M6 — Explain the results and establish repeatability

- Use M5 to establish the role of action coverage and interaction mechanisms. Add a controlled interaction-strength change only if needed to support that explanation, then test one-step versus multi-step training. Change one factor at a time.
- Use a capacity-matched check if model size could explain the result; vary planning horizon if rollout error appears limiting.
- Start with one seed for debugging, three for pilots, and target 5–10 independent seeds for headline comparisons. Report uncertainty across seeds and separate training/data seeds from evaluation episodes.
- Record training cost, planning latency, and peak memory before scaling the Slurm sweep.

**Done when:** `experiments/06_ablations.md` identifies supported explanations and reports variability, not only the best run.

### M7 — Assemble the paper evidence

- Consolidate figures for coverage versus counterfactual error, plan ranking, and closed-loop oracle gap.
- Map each claim to a result and reproducible command; document failures and limitations.
- Keep visual inputs, foundation models, decentralized execution, and team-size generalization as later extensions.

**Done when:** `experiments/07_paper_evidence.md` links every headline claim to its configuration, data, runs, and figure.

## Where new work belongs

- Keep this folder for scientific decisions and results. Create the milestone documents under `docs/paper/experiments/` as work begins; avoid empty scaffolding.
- Put small runnable research entry points under `examples/`, following the existing extension examples. Promote reusable code into the appropriate `benchmarl/` component when its interface is clear.
- Reuse BenchMARL's task, model, configuration, logging, and launcher conventions. A short offline training loop is appropriate where the online `Experiment` lifecycle does not fit; avoid building a second experiment framework.
- Keep large datasets and checkpoints outside `docs/paper` and version control; use configured artifact paths or existing ignored `outputs/` and `multirun/` locations.

Every experiment note records: **hypothesis → exact command/config and commit → dataset/splits/seeds → artifact links → result and uncertainty → next decision**.

Complete each milestone's validation before expanding its scope. A failed hypothesis is a result; an unvalidated evaluation pipeline is unfinished work.

Revisit M1 after each pilot: keep, revise, or remove experiments according to whether they answer the research question. Record the reason and protocol version; retain negative results and distinguish exploratory changes from subsequent held-out evaluation. This plan supersedes earlier instructions to freeze per-task settings, including that wording in the background task survey.
