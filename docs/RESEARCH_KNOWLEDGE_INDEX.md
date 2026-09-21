# Research knowledge index and source-of-truth map

**Last consolidated:** 2026-09-21
**Scope:** all 44 Markdown documents under `docs/paper/`
**Purpose:** make the research record navigable, distinguish current evidence
from historical plans, and prevent a new experiment from silently rewriting an
old conclusion.

This is the entry point for the paper research. It does not replace the detailed
notes or raw artifacts. It records where each kind of knowledge lives, which
source currently has authority, what has been withdrawn, and what remains open.

## Contents

1. [Current answer in one page](#1-current-answer-in-one-page)
2. [Authority and reading order](#2-authority-and-reading-order)
3. [Knowledge map](#3-knowledge-map)
4. [Current claim ledger](#4-current-claim-ledger)
5. [What went wrong](#5-what-went-wrong)
6. [Retired and superseded results](#6-retired-and-superseded-results)
7. [Experiment chronology](#7-experiment-chronology)
8. [Complete document catalog](#8-complete-document-catalog)
9. [Documentation gaps](#9-documentation-gaps)
10. [Update contract](#10-update-contract)

## 1. Current answer in one page

### Research question

**Restructured 2026-09-21** by [direction](paper/direction_2026-09-21.md). The
project is split into two problems with separate success criteria, Task A first:

* **Task A — learn the world model.** Can a latent predictive world model learn a
  control-sufficient representation of multi-agent counterfactual dynamics? Graded
  on effect-normalized counterfactual fidelity, not on downstream control.
* **Task B — plan with it.** Given that model, how should the joint action be
  optimized? Frozen out of Task A entirely.

The historical single-chain question — can an interaction-aware latent world
model generalize to unseen joint actions, rank candidate plans, and improve
closed-loop cooperative control? — is what the evidence below was gathered
against, and its end-to-end form is what failed.

### Current evidence

| Link in the proposed chain | Current conclusion | Best evidence |
|---|---|---|
| Counterfactual prediction | **Supported on Buzz Wire.** In common physical coordinates, relational conditioning beats the matched independent model on two independently collected banks and eight seeds per bank. | [Gate 2 and Gate 4](paper/experiments/14_gate2_gate4.md), [Buzz Wire replication](paper/experiments/16_gate4_buzz_wire.md) |
| Effect of observability | **Supported.** Supplying more task state increases measurable cross-agent response, improves plan ranking, and reduces collisions. | [Buzz Wire replication](paper/experiments/16_gate4_buzz_wire.md), [observability control](paper/experiments/17_observability_control.md) |
| Plan ranking | **Weak and task-dependent.** Buzz Wire ranking remains poor; Balance favors the unrestricted joint model, but Balance's counterfactual response lies below the measurement floor. | [plan ranking](paper/experiments/05_plan_ranking.md), [corrected Balance](paper/experiments/12_balance_corrected.md), [observability control](paper/experiments/17_observability_control.md) |
| Closed-loop learned control | **The central prediction-to-control claim fails on Buzz Wire.** The true-simulator planner succeeds on 25/32 episodes; all learned cells are 0/32. Adding the missing ball state reduces collisions fivefold but produces no task progress, and 0/18 cells beat doing nothing. | [Buzz Wire control gate](paper/experiments/15_buzz_wire_control_gate.md), [learned Gate 4](paper/experiments/16_gate4_buzz_wire.md), [observability control](paper/experiments/17_observability_control.md) |
| Reward-head explanation | **Rejected as the main fix.** A refitted head halves held-out reward MSE but ranks plans worse. Reward prediction accuracy is not plan-ranking quality. | [readout refit](paper/experiments/18_readout_refit.md) |
| Direct agent motion | **Learnable for one block.** A shared transition predicts each agent's position five simulator steps ahead far better than persistence; shared conditioning and physical state improve every registered comparison. This is a direct supervised diagnostic, not latent-world-model validation. | [agent-position validation](paper/experiments/20_agent_position_validation.md) |
| Latent physical validity | **Fails at the first transition.** A head fitted on predicted latents recovers part of their coordinate shift, but all 144 checkpoints remain worse than persistence and 3.84–13.13× worse than matched direct models. Recursive rollout adds error but is not the first failure. | [latent-position validation](paper/experiments/21_latent_position_validation.md) |
| Decision-information localization | **Two failures are separable.** The reward interface is poor even on true latents; true-latent progress is useful, but recursive rollout rejects the known-good plan and CEM removes useful candidates as it optimizes. | [decision localization](paper/experiments/22_decision_information_localization.md) |
| Planner-induced coverage | **Causally useful but insufficient.** It cuts oracle/local physical error about fivefold, improves plan ranking, and creates task progress; control remains unsafe. | [structured-surrogate gate](paper/experiments/23_planner_coverage_structured_surrogate.md) |
| Structured physical control | **Fails the registered gate.** With latent coordinates and scalar reward removed, the full models still collide in 69--88% of episodes and succeed 1/48 times. | [structured-surrogate gate](paper/experiments/23_planner_coverage_structured_surrogate.md) |
| General relational advantage | **Not supported.** Buzz Wire favors relational response prediction; corrected Balance favors joint plan ranking and learned return. | [corrected Balance](paper/experiments/12_balance_corrected.md), [Gate 2 and Gate 4](paper/experiments/14_gate2_gate4.md) |

### Defensible paper story

The original positive chain—better interaction modelling → better prediction →
better ranking → better control—does not hold end to end. The defensible result
is a measurement and failure-decomposition paper:

1. cross-agent response can be measured in a common physical space;
2. relational conditioning improves that response on Buzz Wire and replicates;
3. observability changes prediction, ranking, and safety much more than model
   architecture does;
4. observability repairs safety but not progress, while coverage repairs
   progress but not safety or successful control;
5. planner-induced coverage can recover physical generalization and progress
   without recovering safe control;
6. reward MSE, one-step physics, average plan ranking, selected-tail quality,
   and control are distinct quantities and must not substitute for one another.

The latest concise narrative is the
[project report consolidated through 2026-09-18](paper/project_report_2026-09-17.md). Detailed
numbers should be cited from the experiment note that produced them, not copied
from this index.

## 2. Authority and reading order

When two documents disagree, use this order:

1. **Experiment note for the correcting job** plus its saved artifacts.
1b. **Current direction:** [direction, 2026-09-21](paper/direction_2026-09-21.md)
   for what the project is trying to establish and in what order. It is an intent
   document and never overrides a measured result.
2. **Latest project synthesis:**
   [project report, 2026-09-17](paper/project_report_2026-09-17.md).
3. **Rolling claim ledger:**
   [status report](paper/status_2026-09-16.md). Despite its filename, it contains
   additions from 2026-09-17; check the dated blocks inside it.
4. **Audit/review documents** for defect provenance and the reason a protocol
   changed.
5. **Experiment plan and outline** for intended work, not proof that work ran.
6. **Proposals, direction notes, literature reviews, and method explainers** for
   motivation and background only.

Within an experiment note, prefer an explicitly marked corrected/replacement
table over an earlier table. Raw artifacts under `outputs/<job>/` are the final
provenance layer, but most of `outputs/` is gitignored and must be archived with
the paper if the work leaves this machine.

Recommended reading paths:

- **Ten-minute overview:** this index → [project report](paper/project_report_2026-09-17.md).
- **Understand the failure:** project report → [audit](paper/audit_2026-09-15.md) → [review](paper/review_2026-09-16.md) → experiments 11–18.
- **Reproduce a result:** [experiment index](paper/experiments/README.md) → the numbered note → its `outputs/<job>/` directory.
- **Write the paper:** project report → claim ledger below → correcting experiment notes → [literature review](paper/litreview.md).
- **Change the method:** current failure diagnosis first; consult [EBM notes](paper/EBM.md) or other proposals only after the relevant gate identifies that model family as the remaining variable.

## 3. Knowledge map

| Knowledge area | Primary source | Supporting sources | Authority note |
|---|---|---|---|
| Current project state and final narrative | [project report](paper/project_report_2026-09-17.md) | [status](paper/status_2026-09-16.md), [outline](paper/outline.md) | Project report is the current synthesis; status and outline contain older layers. |
| Current research question and ordering | [direction, 2026-09-21](paper/direction_2026-09-21.md) | [registered Task A ladder](paper/experiments/30_task_a_counterfactual_fidelity.md) | Current intent: Task A before Task B, graded on counterfactual fidelity with an explicit measurement floor. |
| Original research question and scope | [direction](paper/direction.md) | [proposal](paper/multi_agent_latent_mpc_proposal.md), [impact notes](paper/multi_agent_world_model_impact_notes.md) | Historical intent. Their expected positive chain is now partly falsified. |
| Literature and positioning | [literature review](paper/litreview.md) | [EBM review](paper/EBM.md), method references inside the review | Background, not experimental evidence. Several documents contain search-tool citation placeholders that need conversion before publication. |
| CEM-MPC concepts and mathematics | [centralized CEM-MPC](paper/centralized-cem-mpc.md) | [oracle validation](paper/experiments/02_oracle_validation.md) | The explainer is conceptual. The implementation note supersedes its assumed LeWM hyperparameters and execution cadence. |
| Experimental governance | [coding rules](paper/coding_rules.md) | [experiment plan](paper/experiment_plan.md), [reporting](paper/experiments/07_reporting.md) | Rules are active; parts of the plan are historical or explicitly superseded. |
| Task semantics and selection | [VMAS task survey](paper/vmas_task_survey.md) | [protocol](paper/experiments/01_protocol.md), [review](paper/review_2026-09-16.md) | Survey is exploratory. Installed VMAS code and validated task-specific notes govern actual runs. |
| Defect and measurement audit | [audit](paper/audit_2026-09-15.md) | [review](paper/review_2026-09-16.md), [Gate 0 repairs](paper/experiments/11_gate0_repairs.md) | Audit findings F1–F12 explain why early conclusions changed. Correcting jobs decide present validity. |
| Data construction and coverage | [planner-induced coverage](paper/experiments/23_planner_coverage_structured_surrogate.md) | [datasets](paper/experiments/03_datasets.md), [corrected Balance](paper/experiments/12_balance_corrected.md) | Coverage now has a within-model behavior/full control: it improves OOD physics, ranking, and distance, but not safety or success. |
| Model architecture and training | [model baselines](paper/experiments/04_model_baselines.md) | [proposal](paper/multi_agent_latent_mpc_proposal.md), [readout refit](paper/experiments/18_readout_refit.md) | Matched independent/joint/relational design is current; old SIGReg “not run” text is stale. |
| Counterfactual response | [Gate 2 physical comparison](paper/experiments/14_gate2_gate4.md) | [initial C7](paper/experiments/05_counterfactual_prediction.md), [replication](paper/experiments/16_gate4_buzz_wire.md) | Use common-coordinate results. Pre-F6 latent-space ratios are secondary diagnostics, not physical effects. |
| Rollout horizon | [corrected horizon note](paper/experiments/09_horizon_rollout.md) | [Gate 0 repairs](paper/experiments/11_gate0_repairs.md) | Corrected job 1235 replaces the reported 18.7× cliff. |
| Goal and reward objectives | [goal objective](paper/experiments/10_goal_objective.md) | [Stage 1 rescoring](paper/experiments/08_stage1_rescoring.md), [readout refit](paper/experiments/18_readout_refit.md) | Random-endpoint goal gaps and the old readout anticorrelation are retired. |
| Plan ranking | [decision localization](paper/experiments/22_decision_information_localization.md) | [plan ranking](paper/experiments/05_plan_ranking.md), [structured-surrogate gate](paper/experiments/23_planner_coverage_structured_surrogate.md) | Interpret with rankable-state counts, probe floor, candidate distribution, oracle percentile, and the selected CEM tail—not Spearman alone. |
| Closed-loop control | [Buzz Wire Gate 4](paper/experiments/16_gate4_buzz_wire.md) | [observability control](paper/experiments/17_observability_control.md), [structured-surrogate gate](paper/experiments/23_planner_coverage_structured_surrogate.md) | All pre-cadence-repair control conclusions are historical. Buzz Wire falsifies both latent and current structured learned control. |
| Direct physical transition diagnostic | [agent-position validation](paper/experiments/20_agent_position_validation.md) | [datasets](paper/experiments/03_datasets.md), [horizon](paper/experiments/09_horizon_rollout.md) | Establishes one-block physical motion learnability only; it does not validate the latent rollout or planning objective. |
| Latent-to-physical position validation | [latent-position validation](paper/experiments/21_latent_position_validation.md) | [direct position](paper/experiments/20_agent_position_validation.md), [physical response](paper/experiments/14_gate2_gate4.md) | Separates true-latent probe floor, latent alignment, predicted-latent readout, and recursive drift on the same physical target. |
| Decision-interface and optimizer failure | [decision localization](paper/experiments/22_decision_information_localization.md) | [cost landscape](paper/experiments/19_cost_landscape.md), [readout refit](paper/experiments/18_readout_refit.md) | Current decomposition of true-latent information, reward/progress scoring, recursive imagination, and CEM population shift. |
| Structured physical surrogate | [planner-induced coverage](paper/experiments/23_planner_coverage_structured_surrogate.md) | [direct position](paper/experiments/20_agent_position_validation.md), [decision localization](paper/experiments/22_decision_information_localization.md) | Registered Baseline B result: one-step generalization and progress improve, safe control fails. |
| Run/artifact lookup | [experiment index](paper/experiments/README.md) | Individual numbered notes | Current through jobs 1296 and 1331, but still incomplete for jobs 1261–1282; see documentation gaps. |

## 4. Current claim ledger

Statuses mean: **established** = directly measured with a valid current metric;
**bounded** = useful evidence with an explicit limitation; **falsified** = the
planned claim failed its direct test; **open** = not yet tested well enough;
**retired** = do not cite as evidence.

| ID | Claim | Status | Evidence and boundary |
|---|---|---|---|
| K1 | Relational conditioning captures more cross-agent response than an independent predictor on Buzz Wire. | **Established** | Common physical coordinates, two banks, eight seeds each; [14](paper/experiments/14_gate2_gate4.md), [16](paper/experiments/16_gate4_buzz_wire.md). This is task-specific. |
| K2 | The effect of conditioning grows when the model observes more of the true state. | **Established for two measurable input conditions** | Physical input effect about −0.34 versus observation about −0.085; history is below probe resolution and is not a third valid point. [16](paper/experiments/16_gate4_buzz_wire.md) |
| K3 | Better cross-agent response yields better learned control. | **Falsified on Buzz Wire** | Both the original and observability-repaired learned controllers are 0/32; [16](paper/experiments/16_gate4_buzz_wire.md), [17](paper/experiments/17_observability_control.md). Do not generalize this negative beyond the tested pipeline. |
| K4 | The CEM implementation and budget can control Buzz Wire with true dynamics. | **Established** | 25/32 successes, zero collisions; [15](paper/experiments/15_buzz_wire_control_gate.md). This exonerates the basic planner/budget, not every scoring choice under learned dynamics. |
| K5 | Missing ball state caused the catastrophic Buzz Wire collision behavior. | **Established** | Supplying physical state reduces collisions from 0.94 to 0.17; [17](paper/experiments/17_observability_control.md). It fixes safety behavior, not progress. |
| K6 | Refitting or enlarging the reward head is the remaining cheap fix. | **Rejected** | The refit roughly halves reward MSE and worsens plan ranking; [18](paper/experiments/18_readout_refit.md). |
| K7 | Joint conditioning improves Balance decisions. | **Bounded** | Corrected bank: joint improves plan ranking in both regimes, 8/8 seeds; three-seed learned control also favors joint but cannot rank joint versus relational reliably. [12](paper/experiments/12_balance_corrected.md), [14](paper/experiments/14_gate2_gate4.md) |
| K8 | Balance measures counterfactual physical response. | **Retired / unmeasurable** | The true effect is 9–70× below probe error. [14](paper/experiments/14_gate2_gate4.md) |
| K9 | Transport, Wheel, and Dropout establish a cross-task coupling law. | **Not supported** | Transport has only 31/239 active anchors; Wheel and Dropout have zero under the tested intervention. Absence under one probe is not task-wide absence. |
| K10 | The model has a hard 25-step rollout limit and an 18.7× error cliff. | **Retired** | Caused by reading an unsupervised positional slot; [09](paper/experiments/09_horizon_rollout.md), [11](paper/experiments/11_gate0_repairs.md). |
| K11 | Observation-space endpoint goals provide a valid reward-free control objective here. | **Rejected in the tested designs** | Three failures and one partial pass; native task outcomes and safety do not agree with goal distance. [10](paper/experiments/10_goal_objective.md) |
| K12 | Job 1203's readout Spearman near −0.25 proves the reward head points the wrong way. | **Retired** | It mixed a trained-on-predicted head with true latents. Correct refit gives roughly +0.028 to +0.050, but still no useful ranking. [18](paper/experiments/18_readout_refit.md) |
| K13 | Data coverage is sufficient to repair learned control. | **Falsified for the tested one-shot mixture** | Within the structured model, full coverage sharply improves oracle/local physics, rank correlation, and final distance, but leaves 69--88% collisions and 1/48 successes. The hard negatives came from the old planner, not an iterative structured-planner loop. [23](paper/experiments/23_planner_coverage_structured_surrogate.md) |
| K14 | Average collision prediction quality is enough to protect CEM. | **Falsified for the tested surrogate** | Full models reach collision AUROC about 0.98 on held-out mixtures but CEM still selects unsafe motion. This supports a competitive-tail/calibration problem; it does not by itself prove which loss or weighting will fix it. [23](paper/experiments/23_planner_coverage_structured_surrogate.md) |
| K15 | The physical-input learned cost fails because it does not represent task progress, rather than because CEM cannot optimize it. | **Bounded** | The oracle plan ranks poorly even when placed directly in a 302-plan bank, while the learned argmin makes do-nothing-level progress; one seed and 16 roots. [19](paper/experiments/19_cost_landscape.md) |
| K16 | One-block individual-agent motion is learnable by a transition shared across identities, and shared context helps. | **Established on the fixed Buzz Wire bank** | Every registered model beats persistence; joint and relational each beat independent on 8/8 seeds in every input/regime cell. Physical input is best in every cell. This is direct `(dx, dy)` supervision, one bank, one block, and does not establish latent-rollout validity. [20](paper/experiments/20_agent_position_validation.md) |
| K17 | The current latent transition preserves sufficiently accurate agent position for planning. | **Falsified on the fixed Buzz Wire bank** | With a head fitted on predicted latents, all 144 checkpoints are worse than persistence and 3.84–13.13× worse than matched direct models. Both MLP and linear heads agree; the error is already present at h=1. [21](paper/experiments/21_latent_position_validation.md) |
| K18 | Useful task information in a latent guarantees that the planning interface and recursive rollout can use it. | **Falsified on Buzz Wire** | True-latent progress ranks plans well, the reward interface does not, and recursive rollout demotes the known-good plan; later CEM populations lose true quality. [22](paper/experiments/22_decision_information_localization.md) |
| K19 | Removing latent coordinates and scalar reward is sufficient for a cheap physical world model to control Buzz Wire. | **Falsified for Baseline B** | Full-coverage structured models make replicated distance progress but do not achieve positive return or safe success. [23](paper/experiments/23_planner_coverage_structured_surrogate.md) |
| K20 | Buzz Wire's true cross-agent effect is large enough, and on enough held-out states, to grade a model on it. | **Established (job 1497), bounded to 16 root episodes** | Every off-diagonal cell is active on **100%** of 117 held-out anchors, the two agents are near-symmetric, and cross response is 21.6% of the total at one block rising to 32.4% at three. The coupling is strongly axis-dependent: 56% of self in x, 12% in y. H0 is therefore structurally misspecified here by a large margin. Agent cells are authorized for T-A2; shared-body cells are **not** (probe floor 1.8x, failing the registered 3x rule). [30](paper/experiments/30_task_a_counterfactual_fidelity.md) |
| K21 | Joint-action conditioning is necessary to predict interacting latent dynamics (H1 beats H0). | **Falsified as stated (job 1502)** | H1 loses to H0 on **0/8 seeds** in every cell under both probe families, and both conditioned arms sit *above* `E_CF = 1` — worse in squared error than predicting no cross-agent response. The measurement is above floor (resolution 4.3–5.8x) and both probes agree, so this is not the instrument. [30](paper/experiments/30_task_a_counterfactual_fidelity.md) |
| K21b | The conditioned models carry no cross-agent information. | **Rejected — the failure is gain, not absence** | Cosine is +0.39 to +0.45 against exactly 0.000 for H0, magnitude ratio 0.74–0.96, and correcting the gain per anchor would put `E_CF` at 0.51–0.55. The models emit a roughly correct-sized cross response that is inconsistently directed across states, which scores worse than silence. The per-anchor bound is an **oracle** and is not evidence that one global rescaling would work. [30](paper/experiments/30_task_a_counterfactual_fidelity.md) |
| K22 | Relational structure adds something beyond access to joint information (H2 beats H1). | **Supported on the cross block (job 1502), bounded** | Registered as "expected to be weak" and it was not: relational beats joint on 8/8, 8/8, 6/8 and 7/8 seeds across the four cells, both probe families agreeing, at identical information and matched capacity. Bounded hard — **both arms remain above 1**, so the claim is that relational is consistently *less wrong*, not that it works. An architecture claim still needs `N_train != N_test`. [30](paper/experiments/30_task_a_counterfactual_fidelity.md) |
| K23 | An `E_CF < 1` criterion removes the measurement-floor problem on its own. | **Rejected a priori** | If probe reconstruction error on true encodings exceeds the true effect, `E_CF > 1` for every model including a perfect one, and the number measures the instrument. Every Task A result must carry the true effect size, the probe floor, and their ratio. This is how K8 was retired. [30](paper/experiments/30_task_a_counterfactual_fidelity.md) |

## 5. What went wrong

The failure was cumulative. Several layers each produced plausible-looking
numbers, and the project initially treated those numbers as if they measured the
same scientific chain.

### 5.1 Research-design failures

| Problem | Consequence | Present control |
|---|---|---|
| The paper committed early to a positive chain. | Negative or mismatched intermediate metrics were interpreted as model failures instead of measurement failures. | The gate protocol validates task, objective, measurement floor, and oracle control before architecture claims. |
| Tasks were selected by narrative coupling, not measured decision relevance. | Transport was nearly uncontrollable from selected anchors; Wheel and Dropout had no active anchors under the chosen probe; Balance effects were below probe resolution. | Report active-anchor counts, task progress, and measurement floor before training sweeps. |
| Coverage was assumed from action marginals and branching. | Logged data often did not identify the interaction the model was asked to learn. | Job 1331 adds competent-local and CEM-query data on split-safe roots. It establishes a coverage effect, while leaving iterative on-policy structured-planner aggregation open. |
| Architecture, observability, objective, and task changed across comparisons. | “Structure substitutes for information” looked causal before a within-task input intervention existed. | Hold task/data/planner fixed and vary input condition; treat cross-task comparisons as descriptive. |

### 5.2 Implementation and protocol defects

| Defect | What it corrupted | Status/source |
|---|---|---|
| Terminal blocks were masked out of reward/termination training. | Most Buzz Wire collision penalties disappeared from the readout data. | Repaired and rescored; [audit F1](paper/audit_2026-09-15.md), [Stage 1](paper/experiments/08_stage1_rescoring.md). |
| Balance “heuristic” data used Transport's policy. | The bank was mislabeled as competent Balance behavior. | Repaired in job 1236 with an identity contract; [Gate 0](paper/experiments/11_gate0_repairs.md), [corrected bank](paper/experiments/12_balance_corrected.md). |
| Horizon evaluation used a positional slot with no prediction gradient. | An artificial 18.7× rollout cliff and false architectural cap. | Repaired in job 1235; [horizon note](paper/experiments/09_horizon_rollout.md). |
| Closed loop executed all five planned blocks before observing again. | Only about four feedback decisions per episode; warm start discarded the full plan. | `--execute-blocks=1`; cadence gains were large on Balance and Buzz Wire. [Gate 0](paper/experiments/11_gate0_repairs.md), [Balance](paper/experiments/13_balance_control_gate.md), [Buzz Wire](paper/experiments/15_buzz_wire_control_gate.md). |
| Balance lacked its own outcome adapter. | Falls could be mistaken for success and the control path could crash. | Explicit task outcome table and contract test; [Gate 0](paper/experiments/11_gate0_repairs.md). |
| Goal endpoints and final distances could use post-terminal observations. | “Reachable goal” and distance claims were not necessarily about the true terminal state. | Original goal results retired; corrected goal gates use task outcomes and safety metrics. [audit F3–F5](paper/audit_2026-09-15.md), [goal note](paper/experiments/10_goal_objective.md). |
| The readout diagnostic changed both mask and latent distribution. | A negative Spearman was attributed first to observability, then to masking, before the remaining distribution mismatch was isolated. | Job 1282 refit on true latent pairs; old −0.25 number retired. [18](paper/experiments/18_readout_refit.md). |
| Cached truth/config identity was incompletely checked. | Reusing a cache or reconstructing task defaults could silently mismatch a bank. | Risk recorded in audit F11; retain explicit per-job cache paths and validate hashes/configs. |

### 5.3 Measurement failures

| Problem | Why it matters | Required interpretation |
|---|---|---|
| Cross-model latent errors used different learned coordinate systems. | A smaller number need not mean a smaller physical error. | Use common physical targets; treat pre-F6 ratios only as within-model diagnostics. |
| Probe error sometimes exceeded the effect being measured. | A reported model ordering can be pure readout noise. | Publish probe reconstruction error and an explicit resolution ratio beside every physical-response result. |
| Linear and MLP probes reversed the winner. | Probe bias was charged to the dynamics model. | Use a probe family with comparable reconstruction across models and report that comparison. |
| Many candidate sets were constant or nearly tied. | Spearman was undefined or dominated by ties. | Report rankable fraction and decision margins, not only mean correlation. |
| Anchors were treated as independent samples although they shared root episodes. | Uncertainty was overstated. | Cluster by root episode and separate training-seed variation from bank variation. |
| Reward MSE, response error, plan ranking, and return were treated as one chain. | Improvements in one did not predict improvements in the next. | Maintain separate metrics and require a direct downstream test for each causal arrow. |

### 5.4 Documentation failures

- There was no declared precedence among proposals, plans, audits, status
  reports, and corrected experiment notes.
- Dated reports were edited after their filename date, so date alone does not
  establish freshness.
- Old statements often remained inline with a later “revision” paragraph,
  making search results return both the false and corrected claim.
- Nine historical jobs lack dedicated notes, while several jobs 1261–1282 have
  notes but are absent from the job table in the experiment index.
- The same experiment numbers (`02` and `05`) were reused, weakening chronological
  navigation.
- Important evidence lives only in ignored `outputs/` directories or local W&B
  history; links do not constitute an archive.

## 6. Retired and superseded results

Do not cite the left-hand statement without its correction.

| Retired statement or artifact | Replacement |
|---|---|
| “18.7× error cliff after the trained horizon”; single-agent becomes best at long horizons. | Corrected sliding-window results in [09](paper/experiments/09_horizon_rollout.md); the cliff was an unused positional slot. |
| Job 1233 contains competent Balance heuristic behavior. | Job 1236 uses the correct Balance policy; [11](paper/experiments/11_gate0_repairs.md), [12](paper/experiments/12_balance_corrected.md). |
| Goal planners close 30%/41% of the random-to-oracle gap. | The goal was a random endpoint orthogonal to the native task; all such gap percentages are withdrawn. [10](paper/experiments/10_goal_objective.md) |
| Job 1203 readout Spearman ≈ −0.25 proves an anti-correlated reward head. | Correctly matched refit gives a small positive correlation, still not useful; [18](paper/experiments/18_readout_refit.md). |
| The readout failure proves missing ball observations. | Observability is supported directly by jobs 1273/1276, not by the retired readout number. |
| Pre-F6 C7 ratios are “fraction of physical interaction captured.” | Use physical-response ratios and direction in [14](paper/experiments/14_gate2_gate4.md) and [16](paper/experiments/16_gate4_buzz_wire.md). |
| Balance counterfactual response establishes a model ordering. | Balance response is below probe resolution and is unmeasurable. |
| Wheel/Dropout have no interaction. | The selected agent/axis/horizon had zero active anchors; no task-wide absence was established. |
| Pre-1237 closed-loop results validate or invalidate a controller. | They used a five-block execution cadence; use corrected Gate 1/Gate 4 results. |
| `experiment_plan.md` statements that VMAS/W&B/pytest are missing. | [M0 setup](paper/experiments/00_setup.md) records installed versions and successful CPU/CUDA checks. |
| `04_model_baselines.md` text saying the SIGReg alternatives were not run. | Audit §2.2 records the completed sweep; λ=0.009 is promising for Buzz Wire, not a universal optimum. |

## 7. Experiment chronology

This is a knowledge chronology, not a complete scheduler history. Failed and
partial jobs remain evidence but must not be pooled with completed comparisons.

| Phase | Jobs/notes | Knowledge gained | Current standing |
|---|---|---|---|
| M0–M1: setup and question design | [00](paper/experiments/00_setup.md), [01](paper/experiments/01_protocol.md) | Environment, compute/logging rules, question-to-evidence matrix, candidate tasks. | Operational foundation; task recommendations were later revised. |
| M2: oracle/planner validation | [02 oracle](paper/experiments/02_oracle_validation.md), [02 Transport](paper/experiments/02_transport_comparisons.md) | Simulator snapshot/replay, CEM contract, budget/cadence comparisons, Transport difficulty. | Early cadence assumptions superseded by Gate 0/1. |
| M3: controlled banks | [03](paper/experiments/03_datasets.md) | Root-level splits, correlated/independent action regimes, replay and coverage records. | Valid data machinery; relevant interaction coverage remained weak on several tasks. |
| M4: model baselines | [04](paper/experiments/04_model_baselines.md) | Matched independent/joint/relational models, latent health, readouts, regularization issue. | Architecture comparison retained; some original interpretations superseded. |
| Early M5: prediction/ranking/control | [05 C7](paper/experiments/05_counterfactual_prediction.md), [05 ranking](paper/experiments/05_plan_ranking.md), [05 control](paper/experiments/05_closed_loop_control.md), [06](paper/experiments/06_wheel_sweep.md) | Prediction gains, weak ranking, no early learned success, Wheel negative results. | Read through audit corrections; early control used broken cadence. |
| Reporting repair | [07](paper/experiments/07_reporting.md) | BenchMARL naming, W&B and marl-eval schema, migration limits. | Active reporting convention; W&B counts include migrated duplicates. |
| Audit and Stage 1 | [audit](paper/audit_2026-09-15.md), [08](paper/experiments/08_stage1_rescoring.md) | F1–F12, readout-mask repair, own-floor C7, A/B/C decomposition. | Essential provenance; some Stage 1 interpretations were refined again by job 1282. |
| Horizon and goal repair | [09](paper/experiments/09_horizon_rollout.md), [10](paper/experiments/10_goal_objective.md) | Correct horizon behavior; endpoint goals do not align with native tasks/safety. | Current for those questions. |
| Gate 0 and corrected Balance | [11](paper/experiments/11_gate0_repairs.md), [12](paper/experiments/12_balance_corrected.md) | Policy dispatch, outcome contract, cadence split, corrected bank; joint ranking survives. | Current. |
| Gate 1–2 and learned Balance control | [13](paper/experiments/13_balance_control_gate.md), [14](paper/experiments/14_gate2_gate4.md) | True-dynamics control works with feedback; learned Balance progress; physical measurement floor; valid Buzz Wire response. | Current, with explicit limits on three-seed control and Balance response. |
| Buzz Wire end-to-end test | [15](paper/experiments/15_buzz_wire_control_gate.md), [16](paper/experiments/16_gate4_buzz_wire.md) | Oracle reaches 78%; relational response replicates; every learned policy fails control. | Central end-to-end result. |
| Mechanism repair | [17](paper/experiments/17_observability_control.md) | Ball state sharply reduces collisions and improves ranking but yields no progress. | Current; one seed by gate design because the comparison did not function. |
| Final cheap alternative | [18](paper/experiments/18_readout_refit.md) | Correct readout diagnostic, retired −0.25, better reward MSE does not improve ranking. | Current; rare-event hypothesis remains open. |
| Cost-landscape diagnosis | [19](paper/experiments/19_cost_landscape.md) | The physical-input model represents collision penalty but does not rank task progress; exhaustive bank rules out CEM search as the cause. | Current one-seed mechanism result; no architecture claim. |
| Bottom-up physical validation | [20](paper/experiments/20_agent_position_validation.md) | Direct next-position model shared across agents; all models beat persistence, shared conditioning helps, and physical input is strongest. | Complete 144-fit result; bounded to one-step direct supervision on one fixed bank. |
| Latent physical validation | [21](paper/experiments/21_latent_position_validation.md) | Direct-vs-latent comparison, true/predicted latent heads, recursive physical curves, and filmstrips. | Complete 144-checkpoint MLP sweep plus linear control; current evidence that the latent pipeline loses position at the first transition. |
| Decision-information localization | [22](paper/experiments/22_decision_information_localization.md) | Separates true-latent information, reward/progress interfaces, recursive rollout, and successive CEM populations. | Complete three-seed Gate 0d; current evidence for two distinct interface/rollout failures and optimizer exploitation. |
| Planner coverage and Baseline B | [23](paper/experiments/23_planner_coverage_structured_surrogate.md) | Split-safe competent/CEM-query collection and behavior-vs-full structured physical control. | Complete three-seed registered gate; coverage improves intermediate metrics and motion, while safe control fails. |
| Reference-profile repair | [27](paper/experiments/27_audit_gate_a0_reference_profile.md), [28](paper/experiments/28_full_structured_state_successor.md) | `lewm_reference` pinned to LeWM `8edfeb33` against vendored sources; `full32` removes the demonstrated `legacy14` linkage alias. | Implementation and one-seed sanity complete; no comparison authorized from them. |
| Planner-tail localization | [29](paper/experiments/29_a1_2_full_state_tail_localization.md) | Late-CEM teacher-forced growth survives the Markov repair (2.28x in full32); better dynamics came with worse decisions and a higher false-safe rate. | Complete. Registered branch C **deferred** as Task B work by the 2026-09-21 direction; registration intact. |
| **Task A — counterfactual fidelity** | [30](paper/experiments/30_task_a_counterfactual_fidelity.md) | True interaction Jacobian and measurement floor (T-A1), effect-normalized counterfactual fidelity across H0/H1/H2 (T-A2), counterfactual ordering (T-A3). | **Complete through T-A2.** T-A1 (1497): the off-diagonal is real, active on 100% of anchors, axis-dependent. T-A2 (1498 + 1502): no arm resolves it — H1 loses to H0 on 0/8 seeds — but cosine ~0.4 shows the failure is gain, not absent information, and relational consistently beats joint. T-A3 remains gated on an arm passing T-A2; none did. |

## 8. Complete document catalog

Every Markdown file under `docs/paper/` appears below.

### 8.1 Synthesis, governance, and research history

| Document | Contains | Use now |
|---|---|---|
| [project_report_2026-09-17.md](paper/project_report_2026-09-17.md) | Current plain-language project arc, model, gates, results, failures, observability repair, final readout result. | **Primary synthesis.** |
| [status_2026-09-16.md](paper/status_2026-09-16.md) | Rolling claim-by-claim status, task health, corrected horizon, goal failures, later 2026-09-17 addenda. | Detailed ledger; filename date understates its content. |
| [outline.md](paper/outline.md) | AAMAS paper options, story, figures/tables, experiment priorities, claims not to make, glossary. | Historical writing plan; validate every result against later notes. |
| [audit_2026-09-15.md](paper/audit_2026-09-15.md) | Run inventory, defects F1–F12, measurement contracts, revised staged plan, original claim ledger. | Defect provenance and research-process analysis. |
| [review_2026-09-16.md](paper/review_2026-09-16.md) | Independent reproduction of defects, task recommendations, Gates 0–5, statistical guidance, literature implications. | Protocol rationale; several requested gates have since run. |
| [experiment_plan.md](paper/experiment_plan.md) | M0–M7 milestones, repository/environment notes, evidence matrix, file placement. | Planning history and broad workflow; contains inline supersessions. |
| [coding_rules.md](paper/coding_rules.md) | Implementation, validation, compute, logging, reproducibility, and configuration rules. | Active working conventions. |
| [experiments/README.md](paper/experiments/README.md) | Job-to-artifact map, numbered-note index, known documentation debt. | Operational lookup; incomplete for the newest jobs. |

### 8.2 Problem formulation and method background

| Document | Contains | Use now |
|---|---|---|
| [direction_2026-09-21.md](paper/direction_2026-09-21.md) | The Task A / Task B split, the H0/H1/H2 hypotheses and what actually separates them, centralized vs decentralized counterfactuals, the three-level Task A success ladder, the measurement-floor requirement, why JEPA (abstraction, not compute), and what relational must do to earn its place. | **Current statement of intent.** |
| [direction.md](paper/direction.md) | Concise original research question, architecture, minimal experiment, scope. | Historical framing; its expected positive chain is not the result. |
| [multi_agent_latent_mpc_proposal.md](paper/multi_agent_latent_mpc_proposal.md) | Full proposal: gap, hypotheses, architecture, objectives, CEM, evaluation, expected contributions, slide summary. | Design background; implementation revision is dated and later results supersede expectations. |
| [multi_agent_world_model_impact_notes.md](paper/multi_agent_world_model_impact_notes.md) | Counterfactual-generalization framing, dataset regimes, oracle evaluation, plan ranking, compositionality, paper positioning. | Conceptual motivation; positive final claim is no longer current. |
| [centralized-cem-mpc.md](paper/centralized-cem-mpc.md) | Tutorial and mathematical formulation of centralized CEM-MPC, factorization, costs, evaluation. | Method explainer; actual LeWM settings are corrected in experiment 02. |
| [EBM.md](paper/EBM.md) | Energy-based transition models, factorization, sampling/training recipes, VMAS program, risks, positioning. | Future-method option only; explicitly defer until current failure warrants it. |
| [litreview.md](paper/litreview.md) | MARL, latent world models, planning, foundation models, benchmarks, limitations, gaps, reading list. | Broad literature base; citations need publication cleanup and freshness checks. |
| [vmas_task_survey.md](paper/vmas_task_survey.md) | VMAS task mechanics, observations/actions/rewards, interaction mechanisms, configuration reuse. | Task background, not a final task ranking. |

### 8.3 Experiment records

| Document | Contains | Current status |
|---|---|---|
| [00_setup.md](paper/experiments/00_setup.md) | Environment, dependencies, CPU/CUDA smoke runs, Slurm and W&B setup. | Complete operational record. |
| [01_protocol.md](paper/experiments/01_protocol.md) | Question-to-evidence matrix and candidate-task rationale. | Complete historical protocol; later task evidence narrows it. |
| [02_oracle_validation.md](paper/experiments/02_oracle_validation.md) | Simulator oracle, snapshot/replay, LeWM CEM behavior, budget and timing. | Core implementation record; cadence later repaired. |
| [02_transport_comparisons.md](paper/experiments/02_transport_comparisons.md) | Transport comparison jobs, coverage, horizon/budget escalation, semantics. | Historical task evidence; Transport later removed from the control claim. |
| [03_datasets.md](paper/experiments/03_datasets.md) | Controlled data regimes, schema, roots/splits, validation, pilot results. | Current data-contract background. |
| [04_model_baselines.md](paper/experiments/04_model_baselines.md) | Three matched models, validation, prediction/latent results, readout, SIGReg. | Architecture record; some “next” text is stale. |
| [05_closed_loop_control.md](paper/experiments/05_closed_loop_control.md) | First closed-loop hypothesis, design, implementation, validation, early results. | Historical; audit and cadence repair supersede interpretation. |
| [05_counterfactual_prediction.md](paper/experiments/05_counterfactual_prediction.md) | Initial C7 design, Transport failure, Buzz Wire signal and held-out confirmation. | Preserve history; cite later physical-coordinate results for headline claims. |
| [05_plan_ranking.md](paper/experiments/05_plan_ranking.md) | Rankable-state audit, Transport/Buzz Wire ranking, readout decomposition and defect discovery. | Valid diagnostic history; later input/readout experiments refine it. |
| [06_wheel_sweep.md](paper/experiments/06_wheel_sweep.md) | Wheel bank/model sweep, angular-state audit, inactive anchors, null ranking, mechanism caveats. | Negative task record; do not infer task-wide absence of interaction. |
| [07_reporting.md](paper/experiments/07_reporting.md) | W&B/marl-eval conventions, migration, aggregate reporting and dependency limits. | Active reporting reference. |
| [08_stage1_rescoring.md](paper/experiments/08_stage1_rescoring.md) | Terminal-mask repair, own-floor C7, goal A/B/C decomposition. | Correcting record; old readout diagnosis is further superseded by experiment 18. |
| [09_horizon_rollout.md](paper/experiments/09_horizon_rollout.md) | Trained context length, corrected multi-step curves, calibration, retired cliff. | Current horizon evidence. |
| [10_goal_objective.md](paper/experiments/10_goal_objective.md) | Random, competent-controller, position-only, and solved-episode goal designs. | Current reason goal-based control claims were dropped. |
| [11_gate0_repairs.md](paper/experiments/11_gate0_repairs.md) | Wrong-policy bank, positional-slot defect, outcome adapter, cadence, metric labels, CEM mean check. | Current repair ledger. |
| [12_balance_corrected.md](paper/experiments/12_balance_corrected.md) | Corrected Balance bank, stronger joint ranking, response/ranking dissociation, horizon. | Current Balance data result; response later declared below probe resolution. |
| [13_balance_control_gate.md](paper/experiments/13_balance_control_gate.md) | True-dynamics Balance control at two feedback cadences and objective validity. | Current oracle/control-cadence evidence. |
| [14_gate2_gate4.md](paper/experiments/14_gate2_gate4.md) | Learned Balance control, measurement floor, common-coordinate Buzz Wire response, probe-family bias. | Current with listed boundaries. |
| [15_buzz_wire_control_gate.md](paper/experiments/15_buzz_wire_control_gate.md) | Buzz Wire oracle controllability and cadence repair. | Current planner reference. |
| [16_gate4_buzz_wire.md](paper/experiments/16_gate4_buzz_wire.md) | Learned Buzz Wire control failure, collision mechanism, second-bank response replication, observability response. | Central current result. |
| [17_observability_control.md](paper/experiments/17_observability_control.md) | Observation/history/physical inputs in control and ranking; collision repair without progress. | Central current mechanism result. |
| [18_readout_refit.md](paper/experiments/18_readout_refit.md) | Corrected readout diagnostic, reward MSE versus ranking dissociation, rejected candidate-shift explanation. | Current readout-specific result. |
| [19_cost_landscape.md](paper/experiments/19_cost_landscape.md) | Exhaustive learned-cost comparison of oracle, zero, and random plans; separates static preference, progress blindness, and search failure. | Current one-seed failure diagnosis. |
| [20_agent_position_validation.md](paper/experiments/20_agent_position_validation.md) | Direct shared per-agent position model, physical metrics, paired seed analysis, and latent-model boundary. | Complete 144-fit registered sweep. |
| [21_latent_position_validation.md](paper/experiments/21_latent_position_validation.md) | True-latent probe floor, predicted-latent planning head, direct-model comparison, recursive physical rollout, and rendered filmstrips. | Complete 144-checkpoint MLP result plus linear control. |
| [22_decision_information_localization.md](paper/experiments/22_decision_information_localization.md) | True-latent versus rollout reward/progress ranking and CEM-stage population truth. | Current decision-level failure localization. |
| [23_planner_coverage_structured_surrogate.md](paper/experiments/23_planner_coverage_structured_surrogate.md) | Planner-induced coverage collection, structured physical surrogate, held-out ranking, and closed-loop control. | Registered experiment; failed control gate with positive coverage effect. |
| [29_a1_2_full_state_tail_localization.md](paper/experiments/29_a1_2_full_state_tail_localization.md) | Paired legacy14/full32 tail localization; the optimizer tail survives the Markov repair. | Complete; branch C deferred as Task B work, registration intact. |
| [30_task_a_counterfactual_fidelity.md](paper/experiments/30_task_a_counterfactual_fidelity.md) | Registered Task A ladder: true interaction Jacobian and floor (T-A1), effect-normalized counterfactual fidelity (T-A2), counterfactual ordering (T-A3), with decision rules fixed before any result. | **Latest registered experiment; running.** |

## 9. Documentation gaps

### Missing or incomplete experiment records

The experiment index identifies nine older jobs without complete dedicated
notes: **1201, 1218, 1221, 1222, 1223, 1224, 1226, 1228, and 1233**. Their most
important conclusions are partially reconstructed in later notes, but the
original configuration/result/provenance record remains incomplete.

The inverse gap also exists: jobs **1261–1264, 1273, 1276, and 1282** are
documented in experiments 16–18 but are not present in the `experiments/README`
job table. The table also says it maps every job, so its promise and contents
currently disagree.

### Unresolved scientific work

- **Task A is reported through T-A2 and its next step is open.** T-A3 was
  registered as conditional on an arm passing T-A2, and none did, so running it
  as written would measure ordering on models that cannot resolve the effect.
  The result instead poses a sharper question the current design does not
  answer: **is the inconsistent direction of the learned cross response
  correctable by any state-conditional calibration learnable from data, or is
  the latent itself the limit?** The oracle per-anchor gain bound (0.51–0.55)
  is what makes that question worth asking and does not answer it.
  [experiment 30](paper/experiments/30_task_a_counterfactual_fidelity.md)
- **A1.2 branch C is deferred, not withdrawn.** The sampler-matched G6a on
  `full32` and the G6c calibration run are Task B work on the structured
  surrogate; the G6a implementation is already repaired and unrun. See
  [direction](paper/direction_2026-09-21.md) §9.
- **Iterative on-policy coverage remains open.** Job 1331's hard negatives came
  from the frozen latent/reward planner, not from the newly fitted structured
  planner whose errors CEM ultimately exploited.
- Final evaluation on **fresh unseen roots** is still owed if any learned
  controller becomes functional; current Gate 4 uses train-split roots while
  preserving test roots.
- Collision-tail calibration on the structured planner's own low-risk selected
  candidates is still unmeasured; aggregate AUROC around 0.98 did not protect
  closed-loop CEM.
- Buzz Wire history-input response is below the probe's resolution and should
  not be called null.
- Balance learned-control architecture ordering is based on three training seeds
  with variance larger than the model gap.
- Local `outputs/` evidence and the 1201 W&B backfill need an archival plan.
- Direct per-agent prediction passes, while the same target decoded from the
  latent pipeline fails at the first block and degrades recursively; see
  [experiments 20](paper/experiments/20_agent_position_validation.md) and
  [21](paper/experiments/21_latent_position_validation.md).
- The structured surrogate improves average five-block ranking but worsens the
  true return of its selected tail; horizon-localized ranking and calibration
  are the next unresolved measurement. See
  [experiment 23](paper/experiments/23_planner_coverage_structured_surrogate.md).

### Editorial cleanup still needed

- Mark historical proposals with a visible banner linking here.
- Stop appending new dates to `status_2026-09-16.md`; create a new dated report
  or keep a genuinely undated `current_status.md`.
- Convert search-session citation markers such as `turn...`/`cite...` into a
  real bibliography before using the literature and CEM documents in a paper.
- Give future experiment notes unique, monotonic identifiers; keep the duplicate
  `02`/`05` filenames only for backward-compatible links.
- Archive a manifest containing job ID, commit, dirty patch hash, data hash,
  checkpoint hashes, resolved task/planner configs, roots, seeds, completion
  state, note path, and retirement/supersession status.

## 10. Update contract

Use this procedure for every new experiment:

1. Create one experiment note with hypothesis, exact intervention and control,
   task/data/checkpoint identity, seeds/roots, metrics, measurement-floor checks,
   result, limitations, artifact paths, and reproduction command.
2. Add the job to `paper/experiments/README.md`; never make W&B the only index.
3. Update the claim ledger in this file by changing a status, not by deleting the
   previous fact. If a claim is retired, add it to the retirement table and name
   the replacing job.
4. Update the project synthesis only after the experiment note exists. The
   synthesis summarizes; it is not the sole evidence record.
5. Keep four labels distinct: **planned**, **running**, **completed**, and
   **interpreted**. A completed job is not automatically a valid measurement.
6. Never overwrite an artifact. A correction gets a new job/output directory and
   an explicit `supersedes` link.
7. Report the full causal chain separately: prediction/response, plan ranking,
   selected-plan regret, native progress/safety, and success. Do not infer a
   downstream link from an upstream metric.
8. Before a sweep, require: competent true-dynamics control, a task-aligned
   objective, non-degenerate candidates, relevant interaction coverage, a metric
   above its measurement floor, and contract tests for task identity, terminal
   semantics, context positions, and feedback cadence.

This index should stay short enough to scan and strict enough to arbitrate a
conflict. Detailed prose and tables belong in the linked source documents.
