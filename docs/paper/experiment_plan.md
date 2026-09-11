# Experiment plan

Build on [direction.md](direction.md), the [proposal](multi_agent_latent_mpc_proposal.md), and the [impact notes](multi_agent_world_model_impact_notes.md). Use the broader [literature review](litreview.md) as background. Follow [coding_rules.md](coding_rules.md) for implementation.

**Main question:** Does explicit interaction modelling improve prediction of unseen joint-action combinations, plan ranking, and closed-loop multi-agent MPC?

**First target:** one VMAS task, two agents, three learned models, and simulator-based oracle MPC working end to end. Then expand to 2–3 tasks and 2–4 agents. Select tasks that support these team sizes.

## Current repository and environment

Snapshot: 2026-09-11, commit `41774c5` (`slurm update`). The working tree was clean before these documents were added.

| Area | What exists | How we build on it |
|---|---|---|
| Environment integration | `benchmarl/environments/vmas/` and task YAMLs under `benchmarl/conf/task/vmas/` | Reuse task creation, continuous actions, vectorization, and TensorDict observations. |
| Candidate tasks | Navigation, transport, balance, and other VMAS scenarios | Pilot navigation with interaction settings varied, then a cooperative task such as transport. Verify interaction strength and observability before choosing the final suite. |
| Model components | MLP, GNN, DeepSets, CNN, GRU, LSTM; model/config interfaces | Reuse suitable components and conventions for encoders and predictors. Existing policy/critic models are not yet world models. |
| Baseline training | `Experiment`, MAPPO/IPPO and other algorithms, collectors, replay buffers, callbacks | Use for cooperative behaviour-policy training and collection. |
| Experiment support | Hydra YAMLs, seeds, CSV logging, evaluation, checkpointing, plotting examples | Reuse configuration and reporting conventions. Existing online RL training is not an offline world-model trainer. |
| Launching | Packed Slurm launcher, local/MIG configs, small VMAS sweeps, `docs/packed_slurm.md` | Validate a single process before using existing sweeps. Measure resource use before increasing concurrency. |
| Extension guidance | `examples/extending/`, `docs/source/concepts/extending.rst`, `test/` | Follow existing task/model/algorithm boundaries and test style. |

Local checks:

- `.venv` uses Python **3.11.15**; BenchMARL **1.5.2** imports from this checkout.
- Installed: PyTorch **2.7.1**, TorchRL **0.11.0**, TensorDict **0.11.0**, NumPy **1.26.4**, Hydra **1.3.6**, Hydra Submitit launcher **1.2.0**.
- CUDA is available; two NVIDIA A100 GPUs with 40 GB each are visible. This is device visibility, not a scheduling allocation.
- `uv`, `sbatch`, and `srun` are available. Slurm submission and cluster allocations were not tested; the checked-in MIG configuration targets different hardware.
- **VMAS is missing.** Attempting to construct the navigation environment failed with `ImportError: vmas python package was not found`. No environment rollout or training run was validated.
- `torch-geometric`, `wandb`, and `pytest` are missing. PyTorch Geometric is needed only if using the existing GNN implementation; a small pairwise PyTorch predictor can avoid that dependency. CSV logging needs no W&B setup.
- `pyproject.toml`, `.python-version`, and `uv.lock` already define the dependency setup. No dependencies were changed during this scan.

**Not implemented in the scanned project code:** offline world-model datasets/training, the three latent dynamics variants, the shared anti-collapse loss, CEM-MPC, simulator snapshot/restore for counterfactual evaluation, and plan-ranking/oracle-gap metrics.

The VMAS task adapter currently returns `None` for `state_spec`. Agent observations must not be assumed to contain the full simulator state. Exact counterfactual replay needs explicit handling of relevant simulator and scenario state.

## Milestones

### M0 — Make the existing setup runnable

- Install the existing VMAS extra: `uv sync --frozen --extra vmas`.
- Reset and step candidate tasks; run one small MAPPO experiment with CSV logging and rendering disabled.
- Confirm CPU operation, then CUDA operation on an allocated device. Add test tooling when running the relevant tests.

**Done when:** a reproducible environment smoke check and one training/evaluation iteration pass. Record commands and versions in `experiments/00_setup.md`.

### M1 — Freeze the experimental protocol

- Choose the initial task, observations, action bounds, goal representation, and task success criterion.
- Define how predicted latents produce planning costs; a known simulator reward cannot automatically be evaluated on a latent vector. Validate the goal-distance objective before expanding tasks.
- Fix data sizes, episode splits, training budgets, candidate plans, planning horizons, seed policy, and primary metrics.
- Check which other-agent information is already present in each observation; describe the independent baseline's actual information access.

**Done when:** `experiments/01_protocol.md` specifies one complete comparison without unresolved input or objective choices.

### M2 — Validate the simulator oracle and planner

- Implement minimal snapshot/restore for the selected task, including relevant scenario variables and random state.
- Check that restoring the same state and replaying the same actions reproduces the trajectory, without changing the live evaluation environment.
- Implement centralized CEM-MPC with simulator dynamics first; verify that its objective produces useful control.
- Use fixed evaluation states and candidate plans for comparable ranking measurements.

**Done when:** deterministic replay checks pass and oracle MPC improves over a random-action reference. Record evidence in `experiments/02_oracle_validation.md`.

### M3 — Build controlled offline datasets

- Collect diverse independent actions, correlated actions, and cooperative-policy trajectories; start with the first two.
- Hold out joint-action combinations while retaining coverage of individual actions. For continuous actions, define held-out regions or correlation changes explicitly.
- Match data budgets and control state-distribution differences where possible, so action coverage is the intended comparison.
- Split by episode; store observations, actions, next observations, termination flags, episode IDs, and simulator snapshots needed for evaluation.

**Done when:** fixed datasets have manifests, coverage summaries, and leakage checks in `experiments/03_datasets.md`.

### M4 — Establish three world-model baselines

- Implement independent, joint-concatenated, and relational latent predictors.
- Keep encoder design, latent size, loss, data, and optimization budgets consistent; match parameter counts where practical and report them.
- Use the same prediction plus anti-collapse objective. Check latent variance and one-step/multi-step prediction before connecting MPC.
- Treat raw MSE across separately learned latent spaces cautiously; use common task-level metrics for headline comparisons.

**Done when:** all three models train and reload reproducibly on the same pilot dataset. Record results in `experiments/04_model_baselines.md`.

### M5 — Run the central experiment

- Measure logged versus counterfactual prediction error and their gap.
- Measure plan-ranking correlation and the true cost of each model's selected plan on shared candidate sets.
- Evaluate closed-loop success/return and the gap to simulator-based MPC using matched CEM horizons and search budgets.
- Expand the working pilot to weak/strong interaction tasks and the three data regimes.

**Done when:** `experiments/05_main_results.md` connects prediction, ranking, and control for all models, including negative findings. Oracle MPC is a dynamics reference, not a guaranteed globally optimal controller.

### M6 — Explain the results and establish repeatability

- Vary action coverage and interaction strength, then one-step versus multi-step training. Change one factor at a time.
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
