# Coding rules for the paper experiments

These rules apply to code added or changed for the [experiment plan](experiment_plan.md).

1. **Build on BenchMARL.** Treat the existing Meta repository structure as the foundation. Preserve its clear separation of tasks, models, algorithms, configuration, experiments, and launchers. Follow the nearest existing implementation and extension example.

2. **Keep code simple and concise.** A programmer should be able to read each line and immediately understand its purpose. Prefer direct operations, descriptive names, and short functions with one clear responsibility. Readability matters more than minimizing line count.

3. **Let incorrect code fail.** Keep the original exception and traceback. Do not add broad `try/except`, silent fallbacks, automatic retries, default results, or catch-and-continue loops to make a broken experiment finish.

4. **Check essential assumptions explicitly.** A short shape check, configuration error, or finite-loss check is useful when it catches a real mistake near its source. Avoid layers of defensive checks for hypothetical inputs. Handle an exception only when recovery is necessary and well defined.

5. **Make the scientific logic visible.** Keep prediction, loss calculation, optimization, rollout, and plan selection easy to follow. Document tensor dimensions at important boundaries. Use comments for mathematical intent or non-obvious assumptions, rather than narrating every operation.

6. **Avoid speculative abstractions.** Do not introduce new frameworks, registries, base-class hierarchies, wrapper chains, or generic backends for one experiment. Extract a helper when it removes meaningful repetition or clarifies the computation.

7. **Reuse the existing tools.** Prefer PyTorch, TensorDict/TorchRL, existing task/model interfaces, Hydra/dataclass configuration, and current logging and sweep machinery. Add a dependency only when a concrete requirement justifies it.

8. **Keep extensions local.** Add the smallest component needed for the current milestone. Use the existing interfaces where they fit; use a small explicit research script where they do not. Avoid copying or restructuring the core experiment framework to support an offline prototype.

9. **Keep configuration intentional.** Put experiment choices in the established configuration pattern. Do not scatter constants across scripts or expose dozens of options before they are needed. Record the resolved configuration with each run.

10. **Make debugging cheap.** Provide one tiny, deterministic run before a sweep. Use CSV logging and disable rendering for initial checks. Verify on CPU first where practical, then on the intended accelerator. Never silently switch devices or skip failed runs.

11. **Test the behaviour that matters.** Prioritize simulator replay, action/state alignment, episode boundaries, candidate-plan scoring, gradients, and checkpoint reloads. Follow `test/` conventions and run checks relevant to changed code. Avoid tests that merely repeat implementation details.

12. **Keep changes focused and reproducible.** Follow the repository's formatting and contribution conventions. Avoid unrelated refactors, generated scaffolding, duplicated utilities, or large committed artifacts. Record code version, data version, configuration, and seeds; retain failed-run evidence.

13. **Match compute to three levels, and never leave heavy work running unscheduled.** See [docs/packed_slurm.md](../packed_slurm.md) for the concrete commands.
    - **Level 0 — interactive, direct GPU.** Smoke tests, a few episodes of reset/step, one tiny training/eval iteration, or anything else that develops or confirms the pipeline itself: run it directly against a GPU, no Slurm. These are seconds-to-minutes, one-off, and disposable.
    - **Level 1 — Slurm validation and small experiments on the local dev node (`gpu-a240`).** Once Level 0 confirms the pipeline direction, run the actual submission path (`hydra/launcher=packed_local`, or `scripts/slurm/packed_local.sbatch`) on this node's `3g.20gb` MIG slice. This box is not resource-constrained for this project, so there is no need to under-use it: `packed_local` requests a generous share of the node's 32 CPUs/~84GB RAM already. Still go through a MIG rather than the full A100 — MIG scheduling is already set up and keeps allocations isolated — and scale `total_cpus`/`mem_gb`/`timeout_min` up further if a specific pilot run needs it.
    - **Level 2 — heavy or long compute, offloaded to the H100 cluster.** Full sweeps, many seeds, ablations, or anything expected to run past a short pilot must be scheduled to Slurm and run on the separate H100 cluster (`hydra/launcher=packed_mig`, or `scripts/slurm/packed_mig.sbatch`), not on `gpu-a240`. Never leave a Level-2-sized job running unattended on the local dev node. If a task turns out to need Level 2, say so and hand it off rather than scaling it down to fit locally or letting it run for hours interactively.
    - Partition names, `gres` strings, and account/QoS values are cluster-specific. Before trusting a launcher config on a machine it wasn't already validated on, check `sinfo`, `scontrol show partition`, `scontrol show node`, and `/etc/slurm/gres.conf` on that machine and adjust.

Before keeping new code, ask: **Is its purpose obvious, does it fit the existing project, and can we remove anything without losing correctness or clarity?**
