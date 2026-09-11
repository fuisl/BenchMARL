# Packed Hydra sweeps on Slurm

The packed launcher keeps the normal BenchMARL/Hydra multirun syntax. Hydra
expands the Cartesian sweep first; the launcher then submits one or two Slurm
jobs and runs several experiment processes inside each allocated GPU/MIG.

## Three levels of compute (see coding_rules.md rule 13)

- **Level 0 — interactive, direct GPU.** Smoke tests, reset/step checks, one
  tiny training/eval iteration: run `benchmarl/run.py` directly, no Slurm.
- **Level 1 — Slurm validation and small experiments on the local dev node
  (`gpu-a240`).** `hydra/launcher=packed_local`, or
  `scripts/slurm/packed_local.sbatch` directly. Targets the `3g.20gb` MIG
  with a generous share of the node's 32 CPUs/~84GB RAM — this box isn't
  resource-constrained for this project, so there's no need to under-use it;
  still go through a MIG rather than the full A100 since that scheduling is
  already set up. Tested end to end (both the automated launcher and the
  manual sbatch script) against this node's real Slurm setup — see
  `docs/paper/experiments/00_setup.md`.
- **Level 2 — heavy/long compute, offloaded to the H100 cluster.**
  `hydra/launcher=packed_mig`, or `scripts/slurm/packed_mig.sbatch`. Never
  run Level-2-sized work on `gpu-a240`.

Partition names, `gres` strings, and account/QoS are cluster-specific and not
portable. Before trusting any launcher config on a machine it wasn't already
validated on, run:

```bash
sinfo
scontrol show partition
scontrol show node
cat /etc/slurm/gres.conf   # exact GRES Type strings, e.g. a100_2g.10gb
```

and adjust `partition`/`gres`/`account`/`qos` in the relevant
`benchmarl/conf/hydra/launcher/*.yaml` (and the matching `.sbatch` script's
`#SBATCH` lines) to match. `packed_mig.yaml` and `packed_mig.sbatch` target
the separate H100 cluster and have not been validated from this session —
there is no access to that cluster here.

## 1. Install

```bash
uv sync --frozen --extra vmas
```

Run this on the login node from the repository root. The repository and its
`.venv` must be visible from the worker nodes.

## 2. Test locally

```bash
uv run python benchmarl/run.py \
  --config-name sweep/vmas_smoke \
  --multirun \
  hydra/launcher=packed_local
```

This submits a real, non-interactive Slurm batch job to the local `gpu`
partition, requesting the `3g.20gb` MIG. Use `squeue --me` to monitor it.
Results and Submitit logs are under `multirun/YYYY-MM-DD/HH-MM-SS/`.

### Manual alternative: plain `sbatch`

If you would rather not wait on the Python process that drives Submitit (it
blocks until the Slurm job finishes), submit the same kind of run directly:

```bash
sbatch scripts/slurm/packed_local.sbatch            # defaults to sweep/vmas_smoke
sbatch scripts/slurm/packed_local.sbatch sweep/vmas_16
```

This runs one Slurm allocation with `hydra/launcher=joblib` fanning the sweep
out across `--cpus-per-task` local worker processes inside it — the same
one-allocation packing the automated launcher does, without Submitit holding
the shell open. `scripts/slurm/packed_mig.sbatch` is the equivalent script
for the H100 cluster (Level 2); its `#SBATCH` values are unverified from this
session, see the note above.

## 3. Submit a lab sweep

Use the same multirun shown in the BenchMARL README, adding only the launcher:

```bash
uv run python benchmarl/run.py --multirun \
  hydra/launcher=packed_mig \
  algorithm=mappo,ippo,masac \
  task=vmas/balance \
  seed=0,1,2,3,4,5
```

That command contains 3 algorithms x 1 task x 6 seeds = 18 experiments. With
the checked-in `packed_mig` policy it produces:

- two Slurm array elements, each requesting exactly one 3g.40gb MIG;
- nine experiment configurations in each element, assigned round-robin;
- eight concurrent Python workers per element because the two jobs split the
  16-CPU user limit; and
- one queued configuration in each element, started immediately when any
  worker finishes (there is no wave barrier).

The Slurm resource settings are in
`benchmarl/conf/hydra/launcher/packed_mig.yaml`. The generated batch script uses
`sbatch --array=0-1%2` and launches each allocation with `srun` through
Submitit. If only one MIG is currently free, the other element waits in the
queue; it does not duplicate the sweep.

## Policy and tuning

`policy: balanced` is the default:

- 1-16 experiments: one MIG;
- more than 16 experiments: two MIGs, split as evenly as possible; and
- more configurations than workers: workers dynamically take the next config
  as they finish.

For 18 experiments, `balanced` is the throughput/headroom-first choice. It is
faster only when sharing one MIG is a meaningful GPU bottleneck. Because the
cluster still caps the two jobs at 16 CPUs in total, both layouts have at most
16 active experiment processes.

For light experiments, `compact` is usually the allocation-cost-first starting
point: keep all 18 on one MIG and let two wait for a worker:

```bash
uv run python benchmarl/run.py --multirun \
  hydra/launcher=packed_mig \
  hydra.launcher.policy=compact \
  algorithm=mappo,ippo,masac \
  task=vmas/balance \
  seed=0,1,2,3,4,5
```

Compact mode still caps concurrency at 16; it runs the remaining two configs
as workers become free. Running 18 simultaneously would exceed the stated
16-CPU quota. Benchmark a representative subset with both policies: use
`balanced` only if its reduced run time justifies occupying the second MIG.

The number `16` is a scheduling limit, not proof that a real training config
fits efficiently. GPU compute can saturate before VRAM does. Start lower when
moving beyond the smoke settings, for example:

```bash
hydra.launcher.max_workers_per_mig=4
```

Then increase only after measuring peak VRAM, GPU utilization, throughput, and
host RAM. `one_mig_threshold` controls when a second MIG is requested;
`max_workers_per_mig` independently controls process concurrency.

Submitit waits in the invoking shell so Hydra can collect all results, but the
Slurm jobs are batch jobs and continue after an SSH disconnect once submitted.
Use `squeue --me` and the Submitit logs to reconnect to their status; use
`tmux` if retaining the launcher's final exit status matters.
